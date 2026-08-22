"""Evaluation output writer."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.artifacts._evaluation_output_validation import (
    EVALUATION_ARTIFACTS,
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
    validate_evaluation_artifact_structure,
    validate_evaluation_diagnostics_consistency,
)
from quant_research_platform.artifacts._output_common_validation import (
    derive_output_bundle_root,
    validate_output_frame,
)
from quant_research_platform.artifacts._output_manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
)
from quant_research_platform.artifacts._parquet_writers import (
    _artifact_frame_reservation,
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
from quant_research_platform.policy.share_safety import (
    evidence_text_stream_unsafe_reason,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    require_free_disk_headroom,
)
from quant_research_platform.storage._dataset_physical import (
    _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest

_EVALUATION_PENDING_TABLE_LIMIT = 64
_EVALUATION_PENDING_TABLE_OVERHEAD_BYTES = 4 * 1024


def write_evaluation_outputs(
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    *,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local evaluation outputs into an output bundle."""

    output_bundle = derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="evaluation_outputs",
    )

    validate_output_frame(
        metrics,
        required=EVALUATION_METRICS_COLUMNS,
        error_name="evaluation_metrics",
    )
    validate_output_frame(
        grouped_label_values,
        required=EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
        error_name="evaluation_grouped_label_values",
    )
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
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        spec=spec,
        n_groups=evaluation_diagnostics.n_groups,
    )
    validate_evaluation_diagnostics_consistency(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        evaluation_diagnostics=evaluation_diagnostics,
        spec=spec,
    )

    def _write(temp_root: Path) -> None:
        write_parquet(
            metrics,
            temp_root / "metrics.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_parquet(
            grouped_label_values,
            temp_root / "grouped_label_values.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
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
        rendered_summary = _render_evaluation_summary(
            spec=spec,
            evaluation_diagnostics=evaluation_diagnostics,
            metrics=metrics,
            ic_summary=ic_summary,
            group_spread_summary=group_spread_summary,
        )
        unsafe_summary = evidence_text_stream_unsafe_reason(
            rendered_summary[offset : offset + 2048].encode("utf-8")
            for offset in range(0, len(rendered_summary), 2048)
        )
        if unsafe_summary is not None:
            raise ValueError(f"evaluation_summary_share_unsafe: {unsafe_summary}")
        (temp_root / "summary.md").write_text(rendered_summary, encoding="utf-8")
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


def _write_evaluation_outputs_from_chunks(
    chunks: Iterable[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]],
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
        _write_evaluation_artifact_chunks(
            chunks,
            temp_root=temp_root,
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
        write_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
            pre_reserved=True,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
            pre_reserved=True,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        rendered_summary = _render_evaluation_summary(
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
    chunks: Iterable[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]],
    *,
    temp_root: Path,
) -> None:
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
        with _artifact_frame_reservation(
            frame,
            payload_already_reserved=True,
        ):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            policy = current_runtime_resource_policy()
            if policy is not None:
                pending_states[name].enter_context(
                    policy.lease(
                        "artifacts.parquet.pending",
                        max(1, table.nbytes)
                        + _EVALUATION_PENDING_TABLE_OVERHEAD_BYTES,
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
        with closing_dataset_iterator(iter(chunks)) as active_chunks:
            for metrics, grouped, _ in active_chunks:
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
            except BaseException:
                pass
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


def _render_evaluation_summary(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
    metrics: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    coverage_summary: pd.DataFrame | None = None,
) -> str:
    signals = ", ".join(signal.signal_id for signal in spec.signals)
    labels = ", ".join(spec.label_columns)
    lines = [
        "# Evaluation Summary",
        "",
        f"- run_id: `{spec.run_id}`",
        f"- market_id: `{spec.market_id}`",
        f"- research_dataset_id: `{spec.research_dataset_id}`",
        f"- signals: `{signals}`",
        f"- labels: `{labels}`",
        f"- rows_input: `{evaluation_diagnostics.rows_input}`",
        f"- rows_after_filter: `{evaluation_diagnostics.rows_after_filter_mode}`",
        f"- timestamp_count: `{evaluation_diagnostics.timestamp_count}`",
        f"- min_timestamp: `{evaluation_diagnostics.min_timestamp}`",
        f"- max_timestamp: `{evaluation_diagnostics.max_timestamp}`",
        "",
        "## IC Summary",
        "",
        _markdown_table(
            ic_summary,
            ("signal", "label", "mean_ic", "mean_rankic"),
        ),
        "",
        "## Signal/Label Coverage Summary",
        "",
        _markdown_table(
            (
                _signal_label_coverage_summary(metrics)
                if coverage_summary is None
                else coverage_summary
            ),
            ("signal", "label", "mean_n_obs", "min_n_obs", "non_empty_period_count"),
        ),
        "",
        "## Group Spread Summary",
        "",
        _markdown_table(
            group_spread_summary,
            ("signal", "label", "top_bottom_label_value_spread_mean"),
        ),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.loc[:, list(columns)].iterrows():
        rows.append(
            "| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |"
        )
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", r"\|")


def _signal_label_coverage_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["signal", "label", "mean_n_obs", "min_n_obs", "non_empty_period_count"]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    source = metrics.copy()
    if "n_obs" not in source:
        return (
            source.loc[:, ["signal", "label"]]
            .drop_duplicates()
            .assign(
                mean_n_obs=pd.NA,
                min_n_obs=pd.NA,
                non_empty_period_count=pd.NA,
            )
        )
    source["n_obs"] = strict_non_negative_count_series(source["n_obs"])
    grouped = source.groupby(["signal", "label"], as_index=False, sort=True)["n_obs"]
    summary = grouped.agg(
        mean_n_obs="mean",
        min_n_obs="min",
        non_empty_period_count=lambda values: int(values.gt(0).sum()),
    )
    return summary.loc[:, columns]
