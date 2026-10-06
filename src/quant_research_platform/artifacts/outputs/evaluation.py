"""Evaluation output writer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa

import quant_research_platform.reports.evaluation as evaluation_report
from quant_research_platform.artifacts.outputs.common_validation import (
    derive_output_bundle_root,
    validate_output_frame,
)
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    EVALUATION_ARTIFACTS,
    validate_evaluation_artifact_structure,
    validate_evaluation_diagnostics_consistency,
)
from quant_research_platform.artifacts.outputs.manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
)
from quant_research_platform.artifacts.outputs.parquet import write_parquet
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
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
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    filesystem_write_text,
    open_filesystem_file,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest
from quant_research_platform.storage.parquet import parquet_table_writer

_EVALUATION_PENDING_TABLE_LIMIT = 64


def write_evaluation_outputs_from_chunks(
    chunks: Iterable[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]],
    *,
    summary_factory: Callable[
        [], tuple[EvaluationDiagnostics, pd.DataFrame, pd.DataFrame, pd.DataFrame]
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
            chunks, temp_root=temp_root, spec=spec, n_groups=n_groups
        )
        evaluation_diagnostics, ic_summary, group_spread_summary, coverage_summary = (
            summary_factory()
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
        filesystem_write_text(
            temp_root / "summary.md", rendered_summary, encoding="utf-8"
        )
        write_manifest(
            temp_root / "manifest.json",
            output_manifest_payload(
                output_spec=output_spec,
                metadata=evaluation_output_metadata(
                    spec=spec, evaluation_diagnostics=evaluation_diagnostics
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
    spec: EvaluationSpec,
    n_groups: int,
) -> dict[str, object]:
    paths = {
        "metrics": temp_root / "metrics.parquet",
        "grouped_label_values": temp_root / "grouped_label_values.parquet",
    }
    required = {
        "metrics": EVALUATION_METRICS_COLUMNS,
        "grouped_label_values": EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    }
    row_counts = dict.fromkeys(paths, 0)
    empty_frames = {}
    pending_tables = {name: [] for name in paths}
    pending_rows = dict.fromkeys(paths, 0)
    writers = {}
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
    with closing_iterator(ExitStack()) as resources:

        def flush(name):
            tables = pending_tables[name]
            if not tables:
                return
            if name not in writers:
                handle = resources.enter_context(
                    closing_iterator(open_filesystem_file(paths[name], "xb"))
                )
                writers[name] = resources.enter_context(
                    parquet_table_writer(handle, parent=temp_root, compression="zstd")
                )
            writers[name](pa.concat_tables(tables))
            row_counts[name] += pending_rows[name]
            tables.clear()
            pending_rows[name] = 0

        def write_frame(name, frame):
            if frame.empty:
                empty_frames[name] = frame.copy()
                return
            pending_tables[name].append(
                pa.Table.from_pandas(frame, preserve_index=False)
            )
            pending_rows[name] += len(frame)
            if (
                pending_rows[name] >= 8192
                or len(pending_tables[name]) >= _EVALUATION_PENDING_TABLE_LIMIT
            ):
                flush(name)

        with closing_iterator(iter(chunks)) as active_chunks:
            for item in active_chunks:
                value = item
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
                    metrics["timestamp"], error="evaluation_diagnostics_inconsistent"
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
                    if chunk_timestamp_count == 0:
                        if (
                            not metrics.empty
                            or not grouped.empty
                            or chunk_min is not None
                            or (chunk_max is not None)
                        ):
                            raise ValueError("evaluation_diagnostics_inconsistent")
                    elif (
                        chunk_timestamp_count < 0
                        or not isinstance(chunk_min, str)
                        or (not isinstance(chunk_max, str))
                    ):
                        raise ValueError("evaluation_diagnostics_inconsistent")
                else:
                    timestamp_count += int(timestamps.nunique())
                    skipped_timestamp_count += int(
                        timestamps.loc[n_obs.lt(2)].nunique()
                    )
                    chunk_min = (
                        time_value_isoformat(
                            pd.Timestamp(timestamps.min()), preserve_datetime=True
                        )
                        if not timestamps.empty
                        else None
                    )
                    chunk_max = (
                        time_value_isoformat(
                            pd.Timestamp(timestamps.max()), preserve_datetime=True
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
                write_frame("metrics", metrics)
                write_frame("grouped_label_values", grouped)
        for name in paths:
            flush(name)
    for name, columns in required.items():
        if row_counts[name] == 0:
            write_parquet(
                empty_frames.get(name, pd.DataFrame(columns=list(columns))),
                paths[name],
                engine="pyarrow",
                compression="zstd",
                index=False,
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
