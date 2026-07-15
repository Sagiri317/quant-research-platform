"""The sole handle-bound Parquet traversal for dataset storage."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class DatasetFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class DatasetPartMetadata:
    identity: DatasetFileIdentity
    schema: pa.Schema
    row_count: int
    null_columns: frozenset[str]
    file_size_bytes: int
    logical_dtype_payload: bytes


def dataset_file_identity(path: Path) -> DatasetFileIdentity:
    return _identity_from_stat(Path(path).stat())


def inspect_dataset_part(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
) -> DatasetPartMetadata:
    with _opened_parquet(path, expected_identity=expected_identity) as opened:
        parquet_file, identity = opened
        return _part_metadata(parquet_file, identity=identity)


def iter_dataset_part_batches(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None,
    target_chunk_bytes: int,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    row_groups: tuple[int, ...] | None = None,
    source_batch_rows: int | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Decode one part once while binding metadata and bytes to one handle."""

    if isinstance(target_chunk_bytes, bool) or not isinstance(
        target_chunk_bytes, int
    ) or target_chunk_bytes <= 0:
        raise ValueError("invalid_target_chunk_bytes")
    with _opened_parquet(path, expected_identity=expected_identity) as opened:
        parquet_file, identity = opened
        metadata = _part_metadata(
            parquet_file,
            identity=identity,
            require_logical_dtypes=metadata_observer is not None,
        )
        if metadata_observer is not None:
            metadata_observer(metadata)
        selected_columns = columns or tuple(metadata.schema.names)
        try:
            tuple(metadata.schema.field(column) for column in selected_columns)
        except KeyError as exc:
            raise ValueError("dataset_column_not_found") from exc
        selected_row_groups = (
            _matching_row_groups(parquet_file, parquet_filters)
            if row_groups is None
            else list(row_groups)
        )
        batch_rows = source_batch_rows or _target_batch_rows(
            parquet_file,
            columns=selected_columns,
            target_chunk_bytes=target_chunk_bytes,
        )
        for raw_batch in parquet_file.iter_batches(
            batch_size=batch_rows,
            row_groups=selected_row_groups,
            columns=list(selected_columns),
            use_threads=False,
        ):
            filtered = _filter_arrow_batch(raw_batch, parquet_filters)
            yield from _bounded_arrow_batches(
                filtered,
                target_chunk_bytes=target_chunk_bytes,
            )


@contextmanager
def _opened_parquet(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None,
) -> Iterator[tuple[pq.ParquetFile, DatasetFileIdentity]]:
    resolved = Path(path)
    with resolved.open("rb") as handle:
        before = _identity_from_stat(os.fstat(handle.fileno()))
        if expected_identity is not None and before != expected_identity:
            raise ValueError("dataset_file_identity_changed")
        parquet_file = pq.ParquetFile(handle)
        try:
            yield parquet_file, before
        finally:
            after = _identity_from_stat(os.fstat(handle.fileno()))
            parquet_file.close()
            if after != before:
                raise ValueError("dataset_file_identity_changed")


def _part_metadata(
    parquet_file: pq.ParquetFile,
    *,
    identity: DatasetFileIdentity,
    require_logical_dtypes: bool = True,
) -> DatasetPartMetadata:
    schema = parquet_file.schema_arrow
    metadata = schema.metadata or {}
    logical_dtype_payload = metadata.get(b"qrp.logical-dtypes")
    if logical_dtype_payload is None and require_logical_dtypes:
        raise ValueError("dataset_logical_dtypes_missing")
    return DatasetPartMetadata(
        identity=identity,
        schema=schema.remove_metadata(),
        row_count=int(parquet_file.metadata.num_rows),
        null_columns=_parquet_null_columns(parquet_file, tuple(schema.names)),
        file_size_bytes=identity.size,
        logical_dtype_payload=(
            b"" if logical_dtype_payload is None else bytes(logical_dtype_payload)
        ),
    )


def _identity_from_stat(value: os.stat_result) -> DatasetFileIdentity:
    return DatasetFileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        modified_ns=int(value.st_mtime_ns),
    )


def _target_batch_rows(
    parquet_file: pq.ParquetFile,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
) -> int:
    positions = _projected_parquet_leaf_positions(parquet_file, columns)
    maximum_row_bytes = 1
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        if row_group.num_rows <= 0:
            continue
        projected_bytes = sum(
            max(0, int(row_group.column(index).total_uncompressed_size))
            for index in positions
        )
        maximum_row_bytes = max(
            maximum_row_bytes,
            (projected_bytes + int(row_group.num_rows) - 1)
            // int(row_group.num_rows),
        )
    return max(1, target_chunk_bytes // maximum_row_bytes)


def _projected_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> tuple[int, ...]:
    positions: list[int] = []
    for column in columns:
        matches = [
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.schema.column(index).path == column
            or parquet_file.schema.column(index).path.startswith(f"{column}.")
        ]
        if not matches:
            raise ValueError("dataset_column_not_found")
        positions.extend(matches)
    return tuple(positions)


def _parquet_null_columns(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> frozenset[str]:
    result: set[str] = set()
    for column in columns:
        positions = _projected_parquet_leaf_positions(parquet_file, (column,))
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            statistics = tuple(
                row_group.column(position).statistics for position in positions
            )
            if any(
                item is None or item.null_count is None or int(item.null_count) > 0
                for item in statistics
            ):
                result.add(column)
                break
    return frozenset(result)


def _matching_row_groups(
    parquet_file: pq.ParquetFile,
    filters: list[tuple[str, str, Any]] | None,
) -> list[int]:
    indexes = list(range(parquet_file.metadata.num_row_groups))
    if not filters:
        return indexes
    positions = _top_level_parquet_leaf_positions(parquet_file, filters)
    return [
        row_group_index
        for row_group_index in indexes
        if all(
            _row_group_may_match(
                None
                if positions[column] is None
                else parquet_file.metadata.row_group(row_group_index)
                .column(positions[column])
                .statistics,
                operator=operator,
                value=value,
            )
            for column, operator, value in filters
        )
    ]


def _top_level_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    filters: list[tuple[str, str, Any]],
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for column, _, _ in filters:
        matches = [
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.schema.column(index).path == column
        ]
        result[column] = matches[0] if len(matches) == 1 else None
    return result


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


def _bounded_arrow_batches(
    batch: pa.RecordBatch,
    *,
    target_chunk_bytes: int,
) -> Iterator[pa.RecordBatch]:
    if batch.num_rows == 0:
        return
    pending = [batch]
    while pending:
        candidate = pending.pop()
        if candidate.nbytes <= target_chunk_bytes:
            yield candidate
            continue
        if candidate.num_rows == 1:
            raise ValueError("dataset_row_exceeds_target_chunk_bytes")
        left_rows = candidate.num_rows // 2
        pending.append(candidate.slice(left_rows))
        pending.append(candidate.slice(0, left_rows))
