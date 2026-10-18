"""Evaluation output writer."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.evaluation_daily import (
    DAILY_KEYS,
    DAILY_SCHEMAS,
    METRICS_SCHEMA,
    daily_frame,
    daily_key_table,
    validate_daily_counts,
    validate_daily_frame,
    validate_response_counts,
)
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    EVALUATION_ARTIFACTS,
    validate_evaluation_artifact_structure,
    validate_evaluation_diagnostics_consistency,
    validate_metric_group_counts,
    validate_output_frame,
)
from quant_research_platform.artifacts.outputs.manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
)
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import write_manifest
from quant_research_platform.storage.parquet import parquet_table_writer

_EVALUATION_PENDING_TABLE_LIMIT = 64


class _EvaluationDateKeys:
    """Validate chronological daily keys while retaining only the current date."""

    def __init__(self, *, primary_key):
        self.primary_key = primary_key
        self.timestamp = None
        self.index = None

    def add_table(self, table):
        timestamps = strict_timestamp_series(
            pd.Series(table["timestamp"].to_numpy()),
            error="invalid_evaluation_timestamp_values",
        )
        if not timestamps.is_monotonic_increasing or (
            len(timestamps)
            and self.timestamp is not None
            and timestamps.iloc[0] < self.timestamp
        ):
            raise ValueError("evaluation_timestamp_order_invalid")
        start = 0
        for timestamp in timestamps.drop_duplicates():
            stop = int(timestamps.searchsorted(timestamp, side="right"))
            if timestamp != self.timestamp:
                self.timestamp = timestamp
                self.index = ExactPrimaryKeyIndex(primary_key=self.primary_key)
            if self.index.add_table(table.slice(start, stop - start)):
                return True
            start = stop
        return False


def write_evaluation_outputs_from_chunks(
    chunks: Iterable[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]],
    *,
    summary_factory: Callable[
        [], tuple[EvaluationDiagnostics, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ],
    spec: EvaluationSpec,
    output_path: Path,
    metadata: Mapping | None = None,
) -> Path:
    """Persist managed evaluation chunks without retaining aggregate frames."""
    output_bundle = Path(output_path)
    if metadata is not None:
        from quant_research_platform.artifacts.outputs.evaluation_validation import (
            validate_evaluation_science,
        )
        from quant_research_platform.contracts.json_codec import (
            qrp_json_mapping_snapshot,
        )

        metadata = qrp_json_mapping_snapshot(metadata)
        validate_evaluation_science(metadata)

    def _write(temp_root: Path) -> None:
        facts = _write_evaluation_artifact_chunks(
            chunks,
            temp_root=temp_root,
            spec=spec,
            science=metadata,
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
            timestamp_count=evaluation_diagnostics.timestamp_count,
        )
        facts["ic_summary_rows"] = len(ic_summary)
        facts["group_spread_summary_rows"] = len(group_spread_summary)
        validate_evaluation_diagnostics_consistency(
            evaluation_diagnostics=evaluation_diagnostics,
            spec=spec,
            **facts,
        )
        write_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
        )
        write_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
        )
        write_manifest(
            temp_root / "manifest.json",
            output_manifest_payload(
                definition_id="evaluation_outputs",
                metadata=evaluation_output_metadata(
                    spec=spec,
                    evaluation_diagnostics=evaluation_diagnostics,
                    science=metadata,
                ),
                files=[
                    *EVALUATION_ARTIFACTS,
                    *(
                        f"{name}.parquet"
                        for name in (metadata or {}).get("optional_member_rows", {})
                    ),
                ],
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle


def _write_evaluation_artifact_chunks(
    chunks: Iterable[tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]],
    *,
    temp_root: Path,
    spec: EvaluationSpec,
    science: dict | None = None,
) -> dict[str, object]:
    paths = {
        "metrics": temp_root / "metrics.parquet",
        "grouped_label_values": temp_root / "grouped_label_values.parquet",
    }
    required = {
        "metrics": EVALUATION_METRICS_COLUMNS,
        "grouped_label_values": EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    }
    key_indexes = {
        "metrics": _EvaluationDateKeys(primary_key=("timestamp", "signal", "label")),
        "grouped_label_values": _EvaluationDateKeys(
            primary_key=("timestamp", "signal", "label", "group")
        ),
    }
    optional = (science or {}).get("optional_member_rows", {})
    for name in optional:
        paths[name] = temp_root / f"{name}.parquet"
        required[name] = tuple(DAILY_SCHEMAS[name].names)
        key_indexes[name] = _EvaluationDateKeys(primary_key=DAILY_KEYS[name])
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
    filter_columns_used = required_filter_columns(spec.filter_mode)
    filter_definition_value = filter_definition(spec.filter_mode)
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
                    parquet_table_writer(handle, compression="zstd")
                )
            writers[name](pa.concat_tables(tables))
            row_counts[name] += pending_rows[name]
            tables.clear()
            pending_rows[name] = 0

        def write_frame(name, frame):
            if frame.empty:
                empty_frames[name] = frame.copy()
                return
            table = pa.Table.from_pandas(
                frame,
                schema=METRICS_SCHEMA if name == "metrics" else DAILY_SCHEMAS.get(name),
                preserve_index=False,
            )
            pending_tables[name].append(table)
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
                for name, frame in (
                    ("metrics", metrics),
                    ("grouped_label_values", grouped),
                ):
                    if key_indexes[name].add_table(
                        pa.Table.from_pandas(frame, preserve_index=False)
                    ):
                        raise ValueError(f"duplicate_evaluation_{name}_keys")
                validate_evaluation_artifact_structure(
                    metrics=metrics,
                    grouped_label_values=grouped,
                    ic_summary=pd.DataFrame(columns=EVALUATION_IC_SUMMARY_COLUMNS),
                    group_spread_summary=pd.DataFrame(
                        columns=EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS
                    ),
                    spec=spec,
                    science=science,
                )
                validate_metric_group_counts(metrics, grouped, spec=spec)
                for name in optional:
                    frame = metadata[name]
                    validate_daily_frame(
                        frame,
                        name=name,
                        spec=spec,
                        science=science,
                        check_domains=False,
                    )
                    if key_indexes[name].add_table(daily_key_table(frame, name)):
                        raise ValueError(f"duplicate_evaluation_{name}_keys")
                    if name == "daily_strata":
                        for kind, block in frame.groupby("stratification", sort=False):
                            science["stratum_domains"][kind] = sorted(
                                set(science["stratum_domains"][kind])
                                | {
                                    None if pd.isna(value) else value
                                    for value in block.stratum
                                },
                                key=lambda value: (value is not None, value or ""),
                            )
                    write_frame(name, frame)
                validate_response_counts(
                    metadata.get("response_metrics"),
                    metadata.get("response_groups"),
                    science=science,
                )
                if "daily_coverage" in optional:
                    validate_daily_counts(
                        metrics,
                        metadata["daily_coverage"],
                        metadata["daily_strata"],
                        spec=spec,
                        science=science,
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
                if (
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
                empty_frames.get(
                    name,
                    daily_frame([], name)
                    if name in DAILY_SCHEMAS
                    else pd.DataFrame(columns=list(columns)),
                ),
                paths[name],
                schema=METRICS_SCHEMA if name == "metrics" else DAILY_SCHEMAS.get(name),
            )
    for name in optional:
        science["optional_member_rows"][name] = row_counts[name]
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


def _evaluation_spec(metadata):
    from quant_research_platform.artifacts.outputs.evaluation_validation import (
        require_signal_metadata_list,
    )

    return EvaluationSpec(
        evaluation_id=metadata["evaluation_id"],
        market_id=metadata["market_id"],
        research_dataset_id=metadata["research_dataset_id"],
        signals=require_signal_metadata_list(
            metadata, "signals", "output_manifest_metadata_invalid"
        ),
        label_columns=tuple(metadata["label_columns"]),
        label_availability_columns=metadata["label_availability_columns"],
        evaluation_axis=metadata["evaluation_axis"],
        filter_mode=metadata["filter_mode"],
        n_groups=metadata["evaluation_diagnostics"]["n_groups"],
    )


_EVALUATION_COLUMNS = {
    "metrics": EVALUATION_METRICS_COLUMNS,
    "grouped_label_values": EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    "ic_summary": EVALUATION_IC_SUMMARY_COLUMNS,
    "group_spread_summary": EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    **{name: tuple(schema.names) for name, schema in DAILY_SCHEMAS.items()},
}


def _iter_evaluation_member_batches(root, *, name, metadata, time_window=None):
    """One selected-member semantic owner shared by materialized and streaming reads."""
    from quant_research_platform.contracts.time import time_window_timestamp_interval
    from quant_research_platform.storage.datasets.reading import (
        iter_declared_parquet_member_batches,
    )

    spec = _evaluation_spec(metadata)
    keys = DAILY_KEYS.get(
        name,
        {
            "metrics": ("timestamp", "signal", "label"),
            "grouped_label_values": ("timestamp", "signal", "label", "group"),
            "ic_summary": ("signal", "label"),
            "group_spread_summary": ("signal", "label"),
        }.get(name),
    )
    seen = (
        _EvaluationDateKeys(primary_key=keys)
        if "timestamp" in keys
        else ExactPrimaryKeyIndex(primary_key=keys)
    )
    filters = None
    if time_window is not None:
        if name.startswith("response_") or "timestamp" not in _EVALUATION_COLUMNS[name]:
            raise ValueError("evaluation_summary_window_not_supported")
        start, stop, closed = time_window_timestamp_interval(time_window)
        if "window" not in metadata:
            raise ValueError("evaluation_science_metadata_missing")
        from quant_research_platform.contracts.time import TimeWindow

        original_start, original_stop, original_closed = time_window_timestamp_interval(
            TimeWindow(**metadata["window"])
        )
        if (
            start < original_start
            or stop > original_stop
            or (stop == original_stop and closed and not original_closed)
        ):
            raise ValueError("evaluation_read_window_outside")
        filters = [
            ("timestamp", ">=", start),
            ("timestamp", "<=" if closed else "<", stop),
        ]

    def check_schema(part):
        if name == "metrics" and part.schema.remove_metadata() != METRICS_SCHEMA:
            if "timestamp" in part.schema.names and part.schema.field(
                "timestamp"
            ).type != pa.timestamp("ns"):
                raise ValueError("evaluation_metrics_timestamp_invalid")
            raise ValueError("evaluation_metrics_schema_invalid")
        if (
            name in DAILY_SCHEMAS
            and part.schema.remove_metadata() != DAILY_SCHEMAS[name]
        ):
            raise ValueError("evaluation_daily_schema_invalid")

    count = 0
    with closing_iterator(
        iter_declared_parquet_member_batches(
            Path(root) / f"{name}.parquet",
            parquet_filters=filters,
            metadata_observer=check_schema,
        )
    ) as chunks:
        for frame in chunks:
            validate_output_frame(
                frame,
                required=_EVALUATION_COLUMNS[name],
                error_name=f"evaluation_{name}",
            )
            if name in DAILY_SCHEMAS:
                validate_daily_frame(frame, name=name, spec=spec, science=metadata)
                table = daily_key_table(frame, name)
            else:
                validate_evaluation_artifact_structure(
                    **{
                        key: frame if key == name else pd.DataFrame(columns=columns)
                        for key, columns in _EVALUATION_COLUMNS.items()
                        if key not in DAILY_SCHEMAS
                    },
                    spec=spec,
                    science=metadata,
                    timestamp_count=metadata["evaluation_diagnostics"][
                        "timestamp_count"
                    ],
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
            if seen.add_table(table):
                raise ValueError(f"duplicate_evaluation_{name}_keys")
            count += len(frame)
            yield frame
    expected = (
        metadata.get("optional_member_rows", {}).get(name)
        if name in DAILY_SCHEMAS
        else metadata["evaluation_diagnostics"][
            {
                "metrics": "metric_rows",
                "grouped_label_values": "grouped_label_value_rows",
                "ic_summary": "ic_summary_rows",
                "group_spread_summary": "group_spread_summary_rows",
            }[name]
        ]
    )
    if time_window is None and count != expected:
        raise ValueError("evaluation_member_row_count_mismatch")


def read_evaluation_outputs(
    root: Path,
    *,
    tables=("metrics", "grouped_label_values", "ic_summary", "group_spread_summary"),
    time_window=None,
):
    """Validate selected members, optionally slicing persisted daily rows."""
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )

    if (
        not tables
        or len(tables) != len(set(tables))
        or any(name not in _EVALUATION_COLUMNS for name in tables)
    ):
        raise ValueError("evaluation_tables_invalid")
    manifest, _ = validate_output_bundle(
        root,
        expected_definition_id="evaluation_outputs",
        selected_files=tuple(f"{name}.parquet" for name in tables),
    )
    metadata = manifest["metadata"]
    result = {}
    for name in tables:
        with closing_iterator(
            _iter_evaluation_member_batches(
                root, name=name, metadata=metadata, time_window=time_window
            )
        ) as chunks:
            frames = list(chunks)
        result[name] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else daily_frame([], name)
            if name in DAILY_SCHEMAS
            else pd.DataFrame(columns=_EVALUATION_COLUMNS[name])
        )
    if {"metrics", "grouped_label_values"} <= result.keys():
        validate_metric_group_counts(
            result["metrics"],
            result["grouped_label_values"],
            spec=_evaluation_spec(metadata),
        )
    validate_daily_counts(
        result.get("metrics"),
        result.get("daily_coverage"),
        result.get("daily_strata"),
        spec=_evaluation_spec(metadata),
        science=metadata,
    )
    validate_response_counts(
        result.get("response_metrics"), result.get("response_groups"), science=metadata
    )
    return manifest, result


def write_parquet(frame: pd.DataFrame, path: Path, *, schema=None) -> Path:
    """Write one member inside its caller-owned atomic bundle staging directory."""
    path = Path(path)
    require_unaliased_path(path, error="invalid_output_write_path")
    with closing_iterator(open_filesystem_file(path, "xb")) as handle:
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        with parquet_table_writer(handle, compression="zstd") as write_table:
            write_table(table)
        handle.flush()
        os.fsync(handle.fileno())
    return path
