"""Evaluation output writer."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import quant_research_platform.reports.evaluation as evaluation_report
from quant_research_platform.artifacts.outputs.common_validation import (
    derive_output_bundle_root,
    validate_output_frame,
)
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    EVALUATION_ARTIFACTS,
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
    validate_evaluation_artifact_structure,
    validate_evaluation_diagnostics_consistency,
)
from quant_research_platform.artifacts.outputs.manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
)
from quant_research_platform.artifacts.outputs.parquet import (
    _artifact_frames_reservation,
    _artifact_table_write_state,
    _artifact_writer_fixed_reservation,
    _parquet_row_group_size,
    write_parquet,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    evidence_text_stream_unsafe_reason,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    current_runtime_resource_policy,
    require_free_disk_headroom,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest
from quant_research_platform.storage.parquet import (
    _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
)

_EVALUATION_PENDING_TABLE_LIMIT = 64
_EVALUATION_PENDING_TABLE_OVERHEAD_BYTES = 4 * 1024


def write_evaluation_outputs_from_chunks(
    chunks: Iterable[
        tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]
        | OwnedValue[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]]
    ],
    *,
    summary_factory: Callable[
        [],
        tuple[
            EvaluationDiagnostics,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ],
    ],
    spec: EvaluationSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    n_groups: int,
) -> Path:
    """Persist managed evaluation chunks without retaining aggregate frames."""

    output_bundle = derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="evaluation_outputs",
    )

    def _write(temp_root: Path) -> None:
        facts = _write_evaluation_artifact_chunks(
            chunks,
            temp_root=temp_root,
            spec=spec,
            n_groups=n_groups,
        )
        (
            evaluation_diagnostics,
            ic_summary,
            group_spread_summary,
            coverage_summary,
        ) = summary_factory()
        validate_output_frame(
            ic_summary,
            required=EVALUATION_IC_SUMMARY_COLUMNS,
            error_name="evaluation_ic_summary",
        )
        validate_output_frame(
            group_spread_summary,
            required=EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
            error_name="evaluation_group_spread_summary",
        )
        validate_evaluation_artifact_structure(
            metrics=pd.DataFrame(columns=EVALUATION_METRICS_COLUMNS),
            grouped_label_values=pd.DataFrame(
                columns=EVALUATION_GROUPED_LABEL_VALUE_COLUMNS
            ),
            ic_summary=ic_summary,
            group_spread_summary=group_spread_summary,
            spec=spec,
            n_groups=n_groups,
        )
        facts["ic_summary_rows"] = len(ic_summary)
        facts["group_spread_summary_rows"] = len(group_spread_summary)
        validate_evaluation_diagnostics_consistency(
            evaluation_diagnostics=evaluation_diagnostics,
            spec=spec,
            n_groups=n_groups,
            **facts,
        )
        write_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        rendered_summary = evaluation_report.render_evaluation_summary(
            spec=spec,
            evaluation_diagnostics=evaluation_diagnostics,
            metrics=pd.DataFrame(columns=EVALUATION_METRICS_COLUMNS),
            ic_summary=ic_summary,
            group_spread_summary=group_spread_summary,
            coverage_summary=coverage_summary,
        )
        unsafe_summary = evidence_text_stream_unsafe_reason(
            rendered_summary[offset : offset + 2048].encode("utf-8")
            for offset in range(0, len(rendered_summary), 2048)
        )
        if unsafe_summary is not None:
            raise ValueError(f"evaluation_summary_share_unsafe: {unsafe_summary}")
        (temp_root / "summary.md").write_text(
            rendered_summary,
            encoding="utf-8",
        )
        write_manifest(
            temp_root / "manifest.json",
            output_manifest_payload(
                output_spec=output_spec,
                metadata=evaluation_output_metadata(
                    spec=spec,
                    evaluation_diagnostics=evaluation_diagnostics,
                ),
                files=EVALUATION_ARTIFACTS,
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle


def _write_evaluation_artifact_chunks(
    chunks: Iterable[
        tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]
        | OwnedValue[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]]
    ],
    *,
    temp_root: Path,
    spec: EvaluationSpec,
    n_groups: int,
) -> dict[str, object]:
    batch_row_limit = 8_192
    writers: dict[str, pq.ParquetWriter] = {}
    writer_states: dict[str, ExitStack] = {}
    pending_states = {
        "metrics": ExitStack(),
        "grouped_label_values": ExitStack(),
    }
    pending_tables: dict[str, list[pa.Table]] = {
        "metrics": [],
        "grouped_label_values": [],
    }
    pending_rows = {"metrics": 0, "grouped_label_values": 0}
    writer_memory = (
        {
            "metrics": (_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES, 0),
            "grouped_label_values": (_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES, 0),
        }
        if current_runtime_resource_policy() is not None
        else {}
    )
    decoded_metadata: dict[str, dict[bytes, bytes]] = {}
    paths = {
        "metrics": temp_root / "metrics.parquet",
        "grouped_label_values": temp_root / "grouped_label_values.parquet",
    }
    required = {
        "metrics": EVALUATION_METRICS_COLUMNS,
        "grouped_label_values": EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    }
    row_counts = {name: 0 for name in paths}
    fixed_state: ExitStack | None = None
    rows_input = 0
    rows_after_filter_mode = 0
    rows_after_signal_label_dropna = 0
    metric_rows = 0
    grouped_label_value_rows = 0
    timestamp_count = 0
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    skipped_timestamp_count = 0
    filter_columns_used: tuple[str, ...] = ()
    filter_definition_value = ""

    def flush(name: str) -> None:
        tables = pending_tables[name]
        if not tables:
            return
        try:
            table = pa.concat_tables(tables)
            persisted, table_state, batch_rows, next_memory = (
                _artifact_table_write_state(
                    table,
                    previous_metadata=decoded_metadata.get(name),
                    retained_memory=writer_memory.get(name, (0, 0)),
                )
            )
            writer_memory[name] = next_memory
            active_states = writer_states.get(name)
            if active_states is None:
                active_states = ExitStack()
                writer_states[name] = active_states
            active_states.enter_context(table_state)
            decoded_metadata[name] = dict(persisted.schema.metadata or {})
            writer = writers.get(name)
            if writer is None:
                try:
                    writer = pq.ParquetWriter(
                        paths[name],
                        persisted.schema,
                        compression="zstd",
                        write_batch_size=batch_rows,
                        max_rows_per_page=batch_rows,
                        use_dictionary=False,
                        write_statistics=False,
                        write_page_index=False,
                        write_page_checksum=False,
                    )
                except BaseException as exc:
                    state_to_close = writer_states.pop(name)
                    try:
                        state_to_close.close()
                    except BaseException as close_error:
                        exc.add_note(
                            "evaluation writer state release also failed: "
                            f"{type(close_error).__name__}"
                        )
                    writer_memory.pop(name)
                    raise
                writers[name] = writer
            writer.write_table(
                persisted,
                row_group_size=_parquet_row_group_size(persisted.num_rows),
            )
            row_counts[name] += pending_rows[name]
        finally:
            primary = sys.exception()
            try:
                pending_states[name].close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "evaluation pending frame release also failed: "
                    f"{type(close_error).__name__}"
                )
            finally:
                pending_states[name] = ExitStack()
                pending_tables[name] = []
                pending_rows[name] = 0

    def write_frame(name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        policy = current_runtime_resource_policy()
        if policy is not None:
            pending_states[name].enter_context(
                policy.lease(
                    "artifacts.parquet.pending",
                    max(1, table.nbytes) + _EVALUATION_PENDING_TABLE_OVERHEAD_BYTES,
                )
            )
        pending_tables[name].append(table)
        pending_rows[name] += len(frame)
        if pending_rows[name] >= batch_row_limit:
            flush(name)
        elif len(pending_tables[name]) >= _EVALUATION_PENDING_TABLE_LIMIT:
            flush(name)

    primary: BaseException | None = None
    try:
        require_free_disk_headroom(
            temp_root,
            reserve_bytes=2 * _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
        )
        fixed_state = _artifact_writer_fixed_reservation(
            writer_count=2,
            index_bytes=0,
        )
        with closing_iterator(iter(chunks)) as active_chunks:
            for item in active_chunks:
                if isinstance(item, OwnedValue):
                    value = item.value
                    owned_lease = item.lease
                else:
                    value = item
                    owned_lease = None
                metrics, grouped, metadata = value
                validate_output_frame(
                    metrics,
                    required=required["metrics"],
                    error_name="evaluation_metrics",
                )
                validate_output_frame(
                    grouped,
                    required=required["grouped_label_values"],
                    error_name="evaluation_grouped_label_values",
                )
                validate_evaluation_artifact_structure(
                    metrics=metrics,
                    grouped_label_values=grouped,
                    ic_summary=pd.DataFrame(columns=EVALUATION_IC_SUMMARY_COLUMNS),
                    group_spread_summary=pd.DataFrame(
                        columns=EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS
                    ),
                    spec=spec,
                    n_groups=n_groups,
                )
                timestamps = strict_timestamp_series(
                    metrics["timestamp"],
                    error="evaluation_diagnostics_inconsistent",
                )
                n_obs = strict_non_negative_count_series(metrics["n_obs"])
                rows_input += int(metadata["rows_input"])
                rows_after_filter_mode += int(metadata["rows_after_filter_mode"])
                rows_after_signal_label_dropna += int(n_obs.sum())
                metric_rows += len(metrics)
                grouped_label_value_rows += len(grouped)
                if spec.evaluation_axis == "single_asset_time_series":
                    chunk_timestamp_count = int(metadata["timestamp_count"])
                    timestamp_count += chunk_timestamp_count
                    skipped_timestamp_count += int(metadata["skipped_timestamp_count"])
                    chunk_min = metadata["min_timestamp"]
                    chunk_max = metadata["max_timestamp"]
                    if (
                        chunk_timestamp_count <= 0
                        or not isinstance(chunk_min, str)
                        or not isinstance(chunk_max, str)
                    ):
                        raise ValueError("evaluation_diagnostics_inconsistent")
                else:
                    timestamp_count += int(timestamps.nunique())
                    skipped_timestamp_count += int(
                        timestamps.loc[n_obs.lt(2)].nunique()
                    )
                    chunk_min = (
                        time_value_isoformat(
                            pd.Timestamp(timestamps.min()).to_pydatetime(),
                            preserve_datetime=True,
                        )
                        if not timestamps.empty
                        else None
                    )
                    chunk_max = (
                        time_value_isoformat(
                            pd.Timestamp(timestamps.max()).to_pydatetime(),
                            preserve_datetime=True,
                        )
                        if not timestamps.empty
                        else None
                    )
                if chunk_min is not None and chunk_max is not None:
                    min_timestamp = (
                        chunk_min
                        if min_timestamp is None
                        else min(min_timestamp, chunk_min)
                    )
                    max_timestamp = (
                        chunk_max
                        if max_timestamp is None
                        else max(max_timestamp, chunk_max)
                    )
                chunk_filter_columns = tuple(
                    str(column) for column in metadata["filter_columns_used"]
                )
                chunk_filter_definition = str(metadata["filter_definition"])
                if not filter_definition_value:
                    filter_columns_used = chunk_filter_columns
                    filter_definition_value = chunk_filter_definition
                elif (
                    filter_columns_used != chunk_filter_columns
                    or filter_definition_value != chunk_filter_definition
                ):
                    raise ValueError("evaluation_diagnostics_inconsistent")
                with _artifact_frames_reservation(
                    ((metrics, ()), (grouped, ())),
                    owned_lease=owned_lease,
                ):
                    write_frame("metrics", metrics)
                    write_frame("grouped_label_values", grouped)
        flush("metrics")
        flush("grouped_label_values")
        for name in tuple(writers):
            writer = writers.pop(name)
            state_to_close = writer_states.pop(name)
            try:
                writer.add_key_value_metadata(decoded_metadata[name])
            except BaseException as exc:
                try:
                    writer.close()
                except BaseException as close_error:
                    exc.add_note(
                        "evaluation parquet writer close also failed: "
                        f"{type(close_error).__name__}"
                    )
                try:
                    state_to_close.close()
                except BaseException as close_error:
                    exc.add_note(
                        "evaluation writer state release also failed: "
                        f"{type(close_error).__name__}"
                    )
                raise
            try:
                writer.close()
            except BaseException as exc:
                try:
                    state_to_close.close()
                except BaseException as close_error:
                    exc.add_note(
                        "evaluation writer state release also failed: "
                        f"{type(close_error).__name__}"
                    )
                raise
            state_to_close.close()
        for name, columns in required.items():
            if row_counts[name] == 0:
                write_parquet(
                    pd.DataFrame(columns=list(columns)),
                    paths[name],
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
    except BaseException as exc:
        primary = exc
        for pending_state in pending_states.values():
            try:
                pending_state.close()
            except BaseException as close_error:
                primary.add_note(
                    "evaluation pending frame release also failed: "
                    f"{type(close_error).__name__}"
                )
        for name in tuple(writers):
            writer = writers.pop(name)
            try:
                writer.close()
            except BaseException as close_error:
                primary.add_note(
                    "evaluation parquet writer close also failed: "
                    f"{type(close_error).__name__}"
                )
        for name, state_to_close in tuple(writer_states.items()):
            writer_states.pop(name)
            try:
                state_to_close.close()
            except BaseException as close_error:
                primary.add_note(
                    "evaluation writer state release also failed: "
                    f"{type(close_error).__name__}"
                )
        raise
    finally:
        if fixed_state is not None:
            try:
                fixed_state.close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "evaluation writer fixed reservation release also failed: "
                    f"{type(close_error).__name__}"
                )
    return {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter_mode,
        "rows_after_signal_label_dropna": rows_after_signal_label_dropna,
        "metric_rows": metric_rows,
        "grouped_label_value_rows": grouped_label_value_rows,
        "ic_summary_rows": 0,
        "group_spread_summary_rows": 0,
        "timestamp_count": timestamp_count,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "skipped_timestamp_count": skipped_timestamp_count,
        "filter_columns_used": filter_columns_used,
        "filter_definition_value": filter_definition_value,
    }
