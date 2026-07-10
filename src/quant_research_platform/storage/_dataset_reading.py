"""Dataset reading implementation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage._dataset_manifest_validation import (
    manifest_has_runtime_metadata,
    validate_manifest_runtime_metadata,
)
from quant_research_platform.storage._dataset_paths import (
    candidate_data_files,
    relative_partition_label,
    require_artifact_root,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage._dataset_schema import (
    sha256_json,
    validate_dataset_manifest_instance_metadata,
    validated_schema,
)
from quant_research_platform.storage._dataset_types import (
    DatasetInspection,
    DatasetReadMode,
)
from quant_research_platform.storage.manifests import read_qrp_json

_DEFAULT_TARGET_CHUNK_BYTES = 64 * 1024 * 1024
_PARQUET_TO_PANDAS_EXPANSION_FACTOR = 4
_VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR = 32
_PHYSICAL_WORKING_SET_PARTS = 4
_PANDAS_PROBE_ROWS = 64


@dataclass(frozen=True)
class DatasetReadContext:
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
    context = build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )

    chunks = iter_dataset_chunks(context)
    if mode is DatasetReadMode.CHUNKED:
        return chunks
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    frames = list(chunks)
    if not frames:
        return pd.DataFrame(columns=list(context.logical_columns))
    return pd.concat(frames, ignore_index=True)


def build_dataset_read_context(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
) -> DatasetReadContext:
    root = require_artifact_root(Path(dataset_root))
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
    validate_dataset_manifest_instance_metadata(manifest)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    all_paths = tuple(candidate_data_files(data_root, spec=spec_payload))
    if not all_paths:
        raise ValueError("dataset_data_dir_missing")
    schema = validated_schema(list(all_paths))
    schema_columns = tuple(name for name, _ in schema)
    schema_hash = sha256_json(schema)
    validate_manifest_runtime_metadata(
        manifest,
        data_root=data_root,
        paths=list(all_paths),
        validate_row_counts=True,
    )
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
    if str(manifest["schema_hash"]) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    logical_columns = resolve_columns_from_schema(
        schema_columns,
        spec=spec_payload,
        columns=columns,
    )
    timestamp_column = timestamp_column_from_spec_payload(spec_payload)
    physical_columns = read_columns_for_time_window(
        logical_columns,
        spec=spec_payload,
        time_window=time_window,
    )
    return DatasetReadContext(
        root=root,
        data_root=data_root,
        spec=spec_payload,
        manifest=manifest,
        all_paths=all_paths,
        read_paths=tuple(
            prune_paths_for_time_window(
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
        parquet_filters=parquet_filters_for_time_window(
            timestamp_column=timestamp_column,
            time_window=time_window,
        ),
    )


def iter_dataset_chunks(
    context: DatasetReadContext,
    *,
    target_chunk_bytes: int | None = None,
) -> Iterator[pd.DataFrame]:
    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    if not context.read_paths:
        yield pd.DataFrame(columns=list(context.logical_columns))
        return
    yielded = False
    for _, frame in iter_dataset_file_batches(
        context,
        columns=context.physical_columns,
        target_chunk_bytes=target_bytes,
    ):
        filtered = filter_frame_by_time_window(
            frame,
            spec=context.spec,
            time_window=context.time_window,
        ).loc[:, list(context.logical_columns)]
        for chunk in _bounded_frame_chunks(
            filtered,
            target_chunk_bytes=target_bytes,
        ):
            yielded = True
            yield chunk
    if not yielded:
        yield pd.DataFrame(columns=list(context.logical_columns))


def iter_dataset_file_batches(
    context: DatasetReadContext,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    for path in context.read_paths:
        parquet_file = pq.ParquetFile(path)
        batch_size = _target_batch_rows(
            parquet_file,
            columns=columns,
            target_chunk_bytes=target_bytes,
        )
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(columns),
            use_threads=False,
        ):
            for frame in _arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=target_bytes,
            ):
                yield path, frame


def dataset_primary_key_schema(
    context: DatasetReadContext,
    primary_key: tuple[str, ...],
) -> pa.Schema:
    parquet_schema = pq.ParquetFile(context.all_paths[0]).schema_arrow
    try:
        return pa.schema(
            [parquet_schema.field(column) for column in primary_key]
        )
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc


def _validated_target_chunk_bytes(value: int | None) -> int:
    if value is None:
        return _DEFAULT_TARGET_CHUNK_BYTES
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid_target_chunk_bytes")
    return value


def _target_batch_rows(
    parquet_file: pq.ParquetFile,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
) -> int:
    column_positions = {
        name: index for index, name in enumerate(parquet_file.schema_arrow.names)
    }
    selected_positions = [column_positions[column] for column in columns]
    maximum_row_bytes = 1
    metadata = parquet_file.metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        if row_group.num_rows <= 0:
            continue
        projected_bytes = sum(
            max(0, int(row_group.column(index).total_uncompressed_size))
            for index in selected_positions
        )
        row_bytes = max(
            1,
            (projected_bytes + row_group.num_rows - 1) // row_group.num_rows,
        )
        maximum_row_bytes = max(maximum_row_bytes, row_bytes)
    selected_fields = tuple(parquet_file.schema_arrow.field(name) for name in columns)
    expansion_factor = (
        _VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR
        if any(_is_variable_width(field.type) for field in selected_fields)
        else _PARQUET_TO_PANDAS_EXPANSION_FACTOR
    )
    conservative_row_bytes = maximum_row_bytes * expansion_factor
    arrow_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    return max(1, arrow_budget // conservative_row_bytes)


def _is_variable_width(data_type: pa.DataType) -> bool:
    return bool(
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_dictionary(data_type)
        or pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_struct(data_type)
    )


def _arrow_batch_to_bounded_pandas(
    batch: pa.RecordBatch,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    pandas_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    arrow_row_bytes = max(1, (batch.nbytes + batch.num_rows - 1) // batch.num_rows)
    expansion = (
        _VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR
        if any(_is_variable_width(field.type) for field in batch.schema)
        else _PARQUET_TO_PANDAS_EXPANSION_FACTOR
    )
    rows_per_slice = max(
        1,
        min(
            batch.num_rows,
            _PANDAS_PROBE_ROWS,
            pandas_budget // max(1, arrow_row_bytes * expansion),
        ),
    )
    start = 0
    while start < batch.num_rows:
        stop = min(batch.num_rows, start + rows_per_slice)
        arrow_slice = batch.slice(start, stop - start)
        frame = arrow_slice.to_pandas(use_threads=False)
        frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
        if frame_bytes > pandas_budget:
            if len(frame) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        yield frame
        start = stop
        if start >= batch.num_rows:
            continue
        measured_row_bytes = max(1, (frame_bytes + len(frame) - 1) // len(frame))
        rows_per_slice = max(
            1,
            min(
                batch.num_rows - start,
                pandas_budget // measured_row_bytes,
            ),
        )


def _bounded_frame_chunks(
    frame: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    if frame.empty:
        return
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    rows_per_chunk = max(
        1,
        min(
            len(frame),
            len(frame) * target_chunk_bytes // max(1, frame_bytes),
        ),
    )
    start = 0
    while start < len(frame):
        stop = min(len(frame), start + rows_per_chunk)
        candidate = frame.iloc[start:stop]
        candidate_bytes = int(
            candidate.memory_usage(index=True, deep=True).sum()
        )
        if candidate_bytes > target_chunk_bytes:
            if len(candidate) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_chunk = max(1, rows_per_chunk // 2)
            continue
        yield candidate.copy()
        start = stop
        if start < len(frame) and candidate_bytes * 2 < target_chunk_bytes:
            rows_per_chunk = min(len(frame) - start, rows_per_chunk * 2)


def filter_frame_by_time_window(
    frame: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> pd.DataFrame:
    if time_window is None:
        return frame
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    if timestamp_column not in frame:
        raise ValueError("time_window_timestamp_column_not_selected")
    values = strict_timestamp_series(
        frame[timestamp_column],
        error="invalid_time_window_timestamp_value",
    )
    start, end, end_inclusive = time_window_timestamp_interval(time_window)
    end_timestamp = pd.Timestamp(end)
    end_mask = values.le(end_timestamp) if end_inclusive else values.lt(end_timestamp)
    mask = values.ge(pd.Timestamp(start)) & end_mask
    return frame.loc[mask].copy()


def prune_paths_for_time_window(
    data_root: Path,
    paths: tuple[Path, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[Path, ...]:
    if time_window is None:
        return paths
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    if partition_columns != ("year",):
        return paths
    start, end = time_window_date_bounds(time_window)
    years = {
        f"year={year}"
        for year in range(int(start.year), int(end.year) + 1)
    }
    return tuple(
        path for path in paths if relative_partition_label(data_root, path) in years
    )


def parquet_filters_for_time_window(
    *,
    timestamp_column: str | None,
    time_window: TimeWindow | None,
) -> list[tuple[str, str, Any]] | None:
    if time_window is None:
        return None
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    start, end, end_inclusive = time_window_timestamp_interval(time_window)
    upper_operator = "<=" if end_inclusive else "<"
    return [
        (timestamp_column, ">=", pd.Timestamp(start)),
        (timestamp_column, upper_operator, pd.Timestamp(end)),
    ]


def resolve_columns(
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


def resolve_columns_from_schema(
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


def read_columns_for_time_window(
    columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[str, ...]:
    if time_window is None:
        return columns
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    return tuple(dict.fromkeys((*columns, timestamp_column)))
