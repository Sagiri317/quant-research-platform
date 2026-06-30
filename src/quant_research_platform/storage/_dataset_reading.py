"""Dataset reading implementation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_manifest_validation import (
    _manifest_has_runtime_metadata,
    _validate_manifest_runtime_metadata,
)
from quant_research_platform.storage._dataset_paths import (
    _candidate_data_files,
    _relative_partition_label,
    _require_artifact_root,
    _timestamp_column_from_spec_payload,
)
from quant_research_platform.storage._dataset_schema import (
    _sha256_json,
    _validated_schema,
)
from quant_research_platform.storage._dataset_types import (
    DatasetInspection,
    DatasetReadMode,
)
from quant_research_platform.storage.manifests import read_qrp_json


@dataclass(frozen=True)
class _DatasetReadContext:
    root: Path
    data_root: Path
    spec: dict[str, Any]
    manifest: dict[str, Any]
    all_paths: tuple[Path, ...]
    read_paths: tuple[Path, ...]
    logical_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    timestamp_column: str | None
    time_window: TimeWindow | None
    schema: tuple[tuple[str, str], ...]
    schema_hash: str
    parquet_filters: list[tuple[str, str, Any]] | None


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    context = _build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )

    chunks = _read_dataset_chunks(context)
    if mode is DatasetReadMode.CHUNKED:
        return chunks
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    frames = list(chunks)
    if not frames:
        return pd.DataFrame(columns=list(context.logical_columns))
    return pd.concat(frames, ignore_index=True)


def _build_dataset_read_context(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
) -> _DatasetReadContext:
    root = _require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    spec_payload = read_qrp_json(
        spec_path,
        expected_schema_id="qrp-dataset-spec",
        error="dataset_spec_schema_id_invalid",
    )
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    manifest = read_qrp_json(
        manifest_path,
        expected_schema_id="qrp-dataset-manifest",
        error="dataset_manifest_schema_id_invalid",
    )
    if not _manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    all_paths = tuple(_candidate_data_files(data_root, spec=spec_payload))
    if not all_paths:
        raise ValueError("dataset_data_dir_missing")
    schema = _validated_schema(list(all_paths))
    schema_columns = tuple(name for name, _ in schema)
    schema_hash = _sha256_json(schema)
    _validate_manifest_runtime_metadata(
        manifest,
        data_root=data_root,
        paths=list(all_paths),
        validate_row_counts=True,
    )
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
    if str(manifest["schema_hash"]) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    logical_columns = _resolve_columns_from_schema(
        schema_columns,
        spec=spec_payload,
        columns=columns,
    )
    timestamp_column = _timestamp_column_from_spec_payload(spec_payload)
    physical_columns = _read_columns_for_time_window(
        logical_columns,
        spec=spec_payload,
        time_window=time_window,
    )
    return _DatasetReadContext(
        root=root,
        data_root=data_root,
        spec=spec_payload,
        manifest=manifest,
        all_paths=all_paths,
        read_paths=tuple(
            _prune_paths_for_time_window(
                data_root,
                all_paths,
                spec=spec_payload,
                time_window=time_window,
            )
        ),
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        time_window=time_window,
        schema=schema,
        schema_hash=schema_hash,
        parquet_filters=_parquet_filters_for_time_window(
            timestamp_column=timestamp_column,
            time_window=time_window,
        ),
    )


def _read_dataset_chunks(
    context: _DatasetReadContext,
) -> Iterator[pd.DataFrame]:
    if not context.read_paths:
        yield pd.DataFrame(columns=list(context.logical_columns))
        return
    for path in context.read_paths:
        frame = pd.read_parquet(
            path,
            columns=list(context.physical_columns),
            engine="pyarrow",
            filters=context.parquet_filters,
        )
        filtered = _filter_frame_by_time_window(
            frame,
            spec=context.spec,
            time_window=context.time_window,
        )
        yield filtered.loc[:, list(context.logical_columns)]


def _filter_frame_by_time_window(
    frame: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> pd.DataFrame:
    if time_window is None:
        return frame
    timestamp_column = _timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    if timestamp_column not in frame:
        raise ValueError("time_window_timestamp_column_not_selected")
    values = pd.to_datetime(frame[timestamp_column], errors="coerce")
    dates = values.dt.date
    mask = (dates >= time_window.start) & (dates <= time_window.end)
    return frame.loc[mask.fillna(False)].copy()


def _prune_paths_for_time_window(
    data_root: Path,
    paths: tuple[Path, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[Path, ...]:
    if time_window is None:
        return paths
    timestamp_column = _timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    if partition_columns != ("year",):
        return paths
    years = {
        f"year={year}"
        for year in range(int(time_window.start.year), int(time_window.end.year) + 1)
    }
    return tuple(
        path for path in paths if _relative_partition_label(data_root, path) in years
    )


def _parquet_filters_for_time_window(
    *,
    timestamp_column: str | None,
    time_window: TimeWindow | None,
) -> list[tuple[str, str, Any]] | None:
    if time_window is None:
        return None
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    start = pd.Timestamp(time_window.start)
    exclusive_end = pd.Timestamp(time_window.end) + pd.Timedelta(days=1)
    return [
        (timestamp_column, ">=", start),
        (timestamp_column, "<", exclusive_end),
    ]


def _resolve_columns(
    inspection: DatasetInspection,
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        requested = tuple(str(column) for column in inspection.spec["required_columns"])
    else:
        requested = tuple(str(column) for column in columns)
    missing = [column for column in requested if column not in inspection.columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def _resolve_columns_from_schema(
    schema_columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        requested = tuple(str(column) for column in spec["required_columns"])
    else:
        requested = tuple(str(column) for column in columns)
    missing = [column for column in requested if column not in schema_columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def _read_columns_for_time_window(
    columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[str, ...]:
    if time_window is None:
        return columns
    timestamp_column = _timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    return tuple(dict.fromkeys((*columns, timestamp_column)))
