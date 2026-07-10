"""Dataset reading implementation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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


@dataclass(frozen=True)
class DatasetEqualityGroupPlan:
    columns: tuple[str, ...]
    keys: tuple[tuple[Any, ...], ...]
    paths_by_key: dict[tuple[Any, ...], tuple[Path, ...]]


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
        return _empty_frame(context, context.logical_columns)
    return pd.concat(frames, ignore_index=True)


def build_dataset_read_context(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
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
    physical_filters = parquet_filters_for_time_window(
        timestamp_column=timestamp_column,
        time_window=time_window,
    ) or []
    for column, operator, value in row_filters:
        if column not in schema_columns:
            raise ValueError("dataset_filter_column_not_found")
        if operator != "==":
            raise ValueError("unsupported_dataset_physical_filter")
        physical_filters.append((column, operator, value))
    physical_columns = tuple(
        dict.fromkeys(
            (*physical_columns, *(column for column, _, _ in physical_filters))
        )
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
        parquet_filters=physical_filters or None,
    )


def iter_dataset_chunks(
    context: DatasetReadContext,
    *,
    target_chunk_bytes: int | None = None,
) -> Iterator[pd.DataFrame]:
    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    if not context.read_paths:
        yield _empty_frame(context, context.logical_columns)
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
        yield _empty_frame(context, context.logical_columns)


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
        try:
            selected_fields = tuple(
                parquet_file.schema_arrow.field(name) for name in columns
            )
            variable_width = any(
                _is_variable_width(field.type) for field in selected_fields
            )
            batch_size = _target_batch_rows(
                parquet_file,
                columns=columns,
                target_chunk_bytes=target_bytes,
            )
        finally:
            parquet_file.close()
        batches = _iter_filtered_arrow_batches(
            path,
            columns=columns,
            parquet_filters=context.parquet_filters,
            batch_size=batch_size,
            target_chunk_bytes=target_bytes,
            probe_variable_width=variable_width,
        )
        for batch in batches:
            for frame in _arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=target_bytes,
            ):
                yield path, frame


def dataset_primary_key_schema(
    context: DatasetReadContext,
    primary_key: tuple[str, ...],
) -> pa.Schema:
    parquet_file = pq.ParquetFile(context.all_paths[0])
    try:
        parquet_schema = parquet_file.schema_arrow
        return pa.schema(
            [parquet_schema.field(column) for column in primary_key]
        )
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc
    finally:
        parquet_file.close()


def build_dataset_equality_group_plan(
    context: DatasetReadContext,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    target_state_bytes: int,
) -> DatasetEqualityGroupPlan:
    schema_columns = {name for name, _ in context.schema}
    if not columns or any(column not in schema_columns for column in columns):
        raise ValueError("invalid_dataset_equality_group_columns")
    if (
        isinstance(target_state_bytes, bool)
        or not isinstance(target_state_bytes, int)
        or target_state_bytes <= 0
    ):
        raise ValueError("invalid_dataset_equality_group_state_budget")
    filter_columns = tuple(
        column for column, _, _ in context.parquet_filters or ()
    )
    discovery_columns = tuple(dict.fromkeys((*columns, *filter_columns)))
    paths_by_key: dict[tuple[Any, ...], set[Path]] = {}
    state_bytes = 0
    path_order = {path: index for index, path in enumerate(context.read_paths)}
    for path, frame in iter_dataset_file_batches(
        context,
        columns=discovery_columns,
        target_chunk_bytes=target_chunk_bytes,
    ):
        filtered = filter_frame_by_time_window(
            frame,
            spec=context.spec,
            time_window=context.time_window,
        )
        for row in filtered.loc[:, list(columns)].itertuples(index=False, name=None):
            key = tuple(row)
            if any(
                not pd.api.types.is_scalar(value) or bool(pd.isna(value))
                for value in key
            ):
                raise ValueError("invalid_dataset_equality_group_key")
            try:
                hash(key)
            except TypeError as exc:
                raise ValueError("invalid_dataset_equality_group_key") from exc
            is_new_key = key not in paths_by_key
            candidate_paths = paths_by_key.setdefault(key, set())
            is_new_path = path not in candidate_paths
            candidate_paths.add(path)
            state_bytes = _equality_group_plan_bytes(
                state_bytes,
                key=key,
                is_new_key=is_new_key,
                is_new_path=is_new_path,
            )
            if state_bytes > target_state_bytes:
                raise ValueError("dataset_equality_group_index_exceeds_target_bytes")
    ordered_paths = {
        key: tuple(sorted(paths, key=path_order.__getitem__))
        for key, paths in paths_by_key.items()
    }
    return DatasetEqualityGroupPlan(
        columns=columns,
        keys=tuple(sorted(ordered_paths)),
        paths_by_key=ordered_paths,
    )


def dataset_context_for_equality_group(
    context: DatasetReadContext,
    plan: DatasetEqualityGroupPlan,
    key: tuple[Any, ...],
) -> DatasetReadContext:
    if key not in plan.paths_by_key or len(key) != len(plan.columns):
        raise ValueError("unknown_dataset_equality_group_key")
    filters = [*(context.parquet_filters or ())]
    filters.extend(
        (column, "==", value)
        for column, value in zip(plan.columns, key, strict=True)
    )
    physical_columns = tuple(dict.fromkeys((*context.physical_columns, *plan.columns)))
    return replace(
        context,
        read_paths=plan.paths_by_key[key],
        physical_columns=physical_columns,
        parquet_filters=filters,
    )


def _equality_group_plan_bytes(
    current_bytes: int,
    *,
    key: tuple[Any, ...],
    is_new_key: bool,
    is_new_path: bool,
) -> int:
    return (
        current_bytes
        + (
            256 + sum(64 + 2 * len(str(value)) for value in key)
            if is_new_key
            else 0
        )
        + (64 if is_new_path else 0)
    )


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
    selected_fields = tuple(parquet_file.schema_arrow.field(name) for name in columns)
    fixed_row_bytes = tuple(
        _fixed_width_arrow_row_bytes(field.type) for field in selected_fields
    )
    if any(value is None for value in fixed_row_bytes):
        return 1
    selected_positions = _projected_parquet_leaf_positions(
        parquet_file,
        columns,
    )
    maximum_row_bytes = max(1, sum(value or 0 for value in fixed_row_bytes))
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
    conservative_row_bytes = maximum_row_bytes * _PARQUET_TO_PANDAS_EXPANSION_FACTOR
    arrow_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    return max(1, arrow_budget // conservative_row_bytes)


def _projected_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> tuple[int, ...]:
    positions: list[int] = []
    for column in columns:
        matches = [
            index
            for index in range(parquet_file.metadata.num_columns)
            if (
                parquet_file.schema.column(index).path == column
                or parquet_file.schema.column(index).path.startswith(f"{column}.")
            )
        ]
        if not matches:
            raise ValueError("dataset_column_not_found")
        positions.extend(matches)
    return tuple(positions)


def _iter_filtered_arrow_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
    parquet_filters: list[tuple[str, str, Any]] | None,
    batch_size: int,
    target_chunk_bytes: int,
    probe_variable_width: bool,
) -> Iterator[pa.RecordBatch]:
    arrow_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    if not probe_variable_width:
        for batch in _physical_arrow_batches(
            path,
            columns=columns,
            parquet_filters=parquet_filters,
            batch_size=batch_size,
        ):
            _require_arrow_batch_budget(batch, arrow_budget=arrow_budget)
            yield batch
        return

    probe = iter(
        _physical_arrow_batches(
            path,
            columns=columns,
            parquet_filters=parquet_filters,
            batch_size=1,
        )
    )
    try:
        first = next(probe)
    except StopIteration:
        return
    _require_arrow_batch_budget(first, arrow_budget=arrow_budget)
    probe_frame = first.to_pandas(use_threads=False)
    pandas_bytes = int(probe_frame.memory_usage(index=True, deep=True).sum())
    if pandas_bytes > arrow_budget:
        raise ValueError("dataset_row_exceeds_target_chunk_bytes")
    measured_row_bytes = max(1, first.nbytes, pandas_bytes)
    adaptive_rows = max(
        1,
        arrow_budget // measured_row_bytes,
    )
    del probe_frame, first, probe
    for batch in _physical_arrow_batches(
        path,
        columns=columns,
        parquet_filters=parquet_filters,
        batch_size=adaptive_rows,
    ):
        _require_arrow_batch_budget(batch, arrow_budget=arrow_budget)
        yield batch


def _require_arrow_batch_budget(
    batch: pa.RecordBatch,
    *,
    arrow_budget: int,
) -> None:
    if batch.nbytes <= arrow_budget:
        return
    if batch.num_rows == 1:
        raise ValueError("dataset_row_exceeds_target_chunk_bytes")
    raise ValueError("dataset_arrow_batch_exceeds_target_chunk_bytes")


def _physical_arrow_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
    parquet_filters: list[tuple[str, str, Any]] | None,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    parquet_file = pq.ParquetFile(path)
    try:
        row_groups = _matching_row_groups(parquet_file, parquet_filters)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=row_groups,
            columns=list(columns),
            use_threads=False,
        ):
            filtered = _filter_arrow_batch(batch, parquet_filters)
            if filtered.num_rows:
                yield filtered
    finally:
        parquet_file.close()


def _matching_row_groups(
    parquet_file: pq.ParquetFile,
    filters: list[tuple[str, str, Any]] | None,
) -> list[int]:
    indexes = list(range(parquet_file.metadata.num_row_groups))
    if not filters:
        return indexes
    positions = _top_level_parquet_leaf_positions(parquet_file, filters)
    matching: list[int] = []
    for row_group_index in indexes:
        row_group = parquet_file.metadata.row_group(row_group_index)
        if all(
            _row_group_may_match(
                (
                    None
                    if positions[column] is None
                    else row_group.column(positions[column]).statistics
                ),
                operator=operator,
                value=value,
            )
            for column, operator, value in filters
        ):
            matching.append(row_group_index)
    return matching


def _top_level_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    filters: list[tuple[str, str, Any]],
) -> dict[str, int | None]:
    positions: dict[str, int | None] = {}
    for column, _, _ in filters:
        matches = [
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.schema.column(index).path == column
        ]
        positions[column] = matches[0] if len(matches) == 1 else None
    return positions


def _row_group_may_match(
    statistics: Any,
    *,
    operator: str,
    value: Any,
) -> bool:
    if statistics is None or not statistics.has_min_max:
        return True
    minimum = statistics.min
    maximum = statistics.max
    try:
        if operator == ">=":
            return maximum >= value
        if operator == ">":
            return maximum > value
        if operator == "<=":
            return minimum <= value
        if operator == "<":
            return minimum < value
        if operator == "==":
            return minimum <= value <= maximum
    except (TypeError, ValueError):
        return True
    raise ValueError("unsupported_dataset_physical_filter")


def _filter_arrow_batch(
    batch: pa.RecordBatch,
    filters: list[tuple[str, str, Any]] | None,
) -> pa.RecordBatch:
    mask: pa.Array | None = None
    operations = {
        ">=": pc.greater_equal,
        ">": pc.greater,
        "<=": pc.less_equal,
        "<": pc.less,
        "==": pc.equal,
    }
    for column, operator, value in filters or ():
        operation = operations.get(operator)
        if operation is None:
            raise ValueError("unsupported_dataset_physical_filter")
        condition = operation(batch.column(batch.schema.get_field_index(column)), value)
        mask = condition if mask is None else pc.and_kleene(mask, condition)
    return batch if mask is None else batch.filter(mask)


def _empty_frame(
    context: DatasetReadContext,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(context.all_paths[0])
    try:
        schema = parquet_file.schema_arrow
        selected = pa.schema([schema.field(column) for column in columns])
        return pa.Table.from_batches([], schema=selected).to_pandas(use_threads=False)
    finally:
        parquet_file.close()


def _is_variable_width(data_type: pa.DataType) -> bool:
    return _fixed_width_arrow_row_bytes(data_type) is None


def _fixed_width_arrow_row_bytes(data_type: pa.DataType) -> int | None:
    if pa.types.is_null(data_type):
        return 1
    if pa.types.is_boolean(data_type):
        return 1
    if pa.types.is_fixed_size_binary(data_type):
        return int(data_type.byte_width)
    if pa.types.is_decimal(data_type):
        return max(1, int(data_type.bit_width) // 8)
    if (
        pa.types.is_signed_integer(data_type)
        or pa.types.is_unsigned_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_date(data_type)
        or pa.types.is_time(data_type)
        or pa.types.is_timestamp(data_type)
        or pa.types.is_duration(data_type)
    ):
        return max(1, int(data_type.bit_width) // 8)
    return None


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
