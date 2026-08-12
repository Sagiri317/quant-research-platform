"""The sole handle-bound Parquet traversal for dataset storage."""

from __future__ import annotations

import math
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded


@dataclass(frozen=True, slots=True)
class DatasetFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


def dataset_file_identity_from_stat(value: os.stat_result) -> DatasetFileIdentity:
    return DatasetFileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        modified_ns=int(value.st_mtime_ns),
    )


@dataclass(frozen=True, slots=True)
class DatasetPartMetadata:
    identity: DatasetFileIdentity
    schema: pa.Schema
    row_count: int
    null_columns: frozenset[str]
    file_size_bytes: int
    logical_dtype_payload: bytes


@dataclass(frozen=True, slots=True)
class PreparedMembershipValues:
    values: pa.Array
    max_scalar_bytes: int
    retained_memory_bytes: int


def dataset_file_identity(path: Path) -> DatasetFileIdentity:
    candidate = require_unaliased_path(
        Path(path),
        error="invalid_dataset_part_files",
    )
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or int(before.st_ino) <= 0
        or int(before.st_nlink) != 1
    ):
        raise ValueError("invalid_dataset_part_files")
    identity = dataset_file_identity_from_stat(before)
    try:
        after = candidate.lstat()
    except OSError as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    if (
        not stat.S_ISREG(after.st_mode)
        or int(after.st_nlink) != 1
        or dataset_file_identity_from_stat(after) != identity
    ):
        raise ValueError("invalid_dataset_part_files")
    return identity


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
    operation_headroom_bytes: int | None = None,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    row_groups: tuple[int, ...] | None = None,
    source_batch_rows: int | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
    _previous_output_bytes: list[int] | None = None,
    _operation_headroom_bytes: list[int] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Decode one part once while binding metadata and bytes to one handle."""

    if (
        isinstance(target_chunk_bytes, bool)
        or not isinstance(target_chunk_bytes, int)
        or target_chunk_bytes <= 0
    ):
        raise ValueError("invalid_target_chunk_bytes")
    if parquet_filters and operation_headroom_bytes is None:
        raise DatasetReadCapacityExceeded("dataset_filter_operation_headroom_required")
    if source_batch_rows is not None and (
        isinstance(source_batch_rows, bool)
        or not isinstance(source_batch_rows, int)
        or source_batch_rows <= 0
    ):
        raise ValueError("invalid_source_batch_rows")
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
        previous_output_bytes = (
            [0] if _previous_output_bytes is None else _previous_output_bytes
        )
        initial_operation_headroom = (
            operation_headroom_bytes
            if _operation_headroom_bytes is None
            else _operation_headroom_bytes[0]
        )
        initial_physical_headroom = (
            None
            if initial_operation_headroom is None
            else initial_operation_headroom - previous_output_bytes[0]
        )
        membership = any(
            operator == "in" for _, operator, _ in parquet_filters or ()
        )
        membership_max_scalar_bytes = max(
            (
                _membership_max_scalar_bytes(value)
                for _, operator, value in parquet_filters or ()
                if operator == "in"
            ),
            default=0,
        )
        admitted_batch_rows = _target_batch_rows(
            parquet_file,
            columns=selected_columns,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=(
                initial_physical_headroom if parquet_filters else None
            ),
            membership=membership,
            membership_max_scalar_bytes=membership_max_scalar_bytes,
        )
        batch_rows = (
            admitted_batch_rows
            if source_batch_rows is None
            else min(source_batch_rows, admitted_batch_rows)
        )
        maximum_projected_row_bytes = _maximum_projected_row_bytes(
            parquet_file,
            selected_columns,
        )
        raw_batches = iter(
            parquet_file.iter_batches(
                batch_size=batch_rows,
                row_groups=selected_row_groups,
                columns=list(selected_columns),
                use_threads=False,
            )
        )
        selected_source_rows = sum(
            int(parquet_file.metadata.row_group(index).num_rows)
            for index in selected_row_groups
        )
        pulled_source_rows = 0
        while True:
            if pulled_source_rows >= selected_source_rows:
                break
            next_batch_rows = min(
                batch_rows,
                selected_source_rows - pulled_source_rows,
            )
            source_batch_bytes = next_batch_rows * maximum_projected_row_bytes
            physical_pull_bytes = (
                _filter_working_set_bytes(
                    source_batch_bytes=source_batch_bytes,
                    row_count=next_batch_rows,
                    membership=membership,
                    membership_max_scalar_bytes=membership_max_scalar_bytes,
                )
                if parquet_filters
                else source_batch_bytes
            )
            current_operation_headroom = (
                operation_headroom_bytes
                if _operation_headroom_bytes is None
                else _operation_headroom_bytes[0]
            )
            physical_headroom = (
                None
                if current_operation_headroom is None
                else current_operation_headroom - previous_output_bytes[0]
            )
            if (
                physical_headroom is not None
                and physical_pull_bytes > physical_headroom
            ):
                raise DatasetReadCapacityExceeded(
                    "dataset_working_set_exceeds_task_memory_budget"
                )
            try:
                raw_batch = next(raw_batches)
            except StopIteration:
                break
            pulled_source_rows += raw_batch.num_rows
            filtered = _filter_arrow_batch(
                raw_batch,
                parquet_filters,
                operation_headroom_bytes=physical_headroom,
            )
            del raw_batch
            yield from _bounded_arrow_batches(
                filtered,
                target_chunk_bytes=target_chunk_bytes,
            )
            del filtered


@contextmanager
def _opened_parquet(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None,
) -> Iterator[tuple[pq.ParquetFile, DatasetFileIdentity]]:
    resolved = Path(path)
    path_identity = dataset_file_identity(resolved)
    if expected_identity is not None and path_identity != expected_identity:
        raise ValueError("dataset_file_identity_changed")
    handle = resolved.open("rb")
    try:
        before = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
        if before != path_identity or dataset_file_identity(resolved) != before:
            raise ValueError("dataset_file_identity_changed")
        parquet_file = pq.ParquetFile(handle)
        try:
            yield parquet_file, before
        finally:
            if sys.exception() is not None:
                try:
                    os.fstat(handle.fileno())
                except BaseException:
                    pass
                try:
                    parquet_file.close()
                except BaseException:
                    pass
            else:
                try:
                    after = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
                except BaseException:
                    try:
                        parquet_file.close()
                    except BaseException:
                        pass
                    raise
                parquet_file.close()
                if after != before or dataset_file_identity(resolved) != before:
                    raise ValueError("dataset_file_identity_changed")
    finally:
        if sys.exception() is None:
            handle.close()
            if dataset_file_identity(resolved) != path_identity:
                raise ValueError("dataset_file_identity_changed")
        else:
            try:
                handle.close()
            except BaseException:
                pass
            try:
                dataset_file_identity(resolved)
            except BaseException:
                pass


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


def _target_batch_rows(
    parquet_file: pq.ParquetFile,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    membership: bool = False,
    membership_max_scalar_bytes: int = 0,
) -> int:
    maximum_row_bytes = _maximum_projected_row_bytes(parquet_file, columns)
    target_rows = max(1, target_chunk_bytes // maximum_row_bytes)
    if operation_headroom_bytes is None:
        return target_rows
    low = 1
    high = target_rows
    admitted = 0
    while low <= high:
        candidate = (low + high) // 2
        source_batch_bytes = candidate * maximum_row_bytes
        required_bytes = _filter_working_set_bytes(
            source_batch_bytes=source_batch_bytes,
            row_count=candidate,
            membership=membership,
            membership_max_scalar_bytes=membership_max_scalar_bytes,
        )
        if required_bytes <= operation_headroom_bytes:
            admitted = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    if admitted == 0:
        raise DatasetReadCapacityExceeded(
            "dataset_filter_working_set_exceeds_task_memory_budget"
        )
    return admitted


def _maximum_projected_row_bytes(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
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
            (projected_bytes + int(row_group.num_rows) - 1) // int(row_group.num_rows),
        )
    return maximum_row_bytes


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
        if operator == "in":
            if not isinstance(value, PreparedMembershipValues):
                raise ValueError("invalid_dataset_physical_filter_value")
            values = value.values
            low = 0
            high = len(values)
            while low < high:
                middle = (low + high) // 2
                if values[middle].as_py() < minimum:
                    low = middle + 1
                else:
                    high = middle
            return low < len(values) and values[low].as_py() <= maximum
    except (TypeError, ValueError):
        return True
    raise ValueError("unsupported_dataset_physical_filter")


def _filter_arrow_batch(
    batch: pa.RecordBatch,
    filters: list[tuple[str, str, Any]] | None,
    *,
    operation_headroom_bytes: int | None = None,
) -> pa.RecordBatch:
    if filters and operation_headroom_bytes is None:
        raise DatasetReadCapacityExceeded("dataset_filter_operation_headroom_required")
    operations = {
        ">=": pc.greater_equal,
        ">": pc.greater,
        "<=": pc.less_equal,
        "<": pc.less,
        "==": pc.equal,
    }
    for column, operator, value in filters or ():
        if operator == "in":
            if not isinstance(value, PreparedMembershipValues):
                raise ValueError("invalid_dataset_physical_filter_value")
        elif operator not in operations:
            raise ValueError("unsupported_dataset_physical_filter")
        _require_filter_batch_capacity(
            batch,
            operation_headroom_bytes=operation_headroom_bytes,
            membership=operator == "in",
            membership_max_scalar_bytes=(
                _membership_max_scalar_bytes(value) if operator == "in" else 0
            ),
        )
        if operator == "in":
            column_values = batch.column(batch.schema.get_field_index(column))
            values = value.values
            if len(values) == 0:
                return batch.slice(0, 0)
            row_count = batch.num_rows
            low = pa.repeat(pa.scalar(0, type=pa.int32()), row_count)
            high = pa.repeat(pa.scalar(len(values), type=pa.int32()), row_count)
            maximum_index = pa.scalar(len(values) - 1, type=pa.int32())
            one = pa.scalar(1, type=pa.int32())
            for _ in range(math.ceil(math.log2(len(values) + 1))):
                middle = pc.shift_right(pc.add(low, high), 1)
                candidate_positions = pc.min_element_wise(middle, maximum_index)
                candidates = pc.take(values, candidate_positions)
                at_or_below = pc.less_equal(column_values, candidates)
                next_high = pc.if_else(at_or_below, middle, high)
                next_low = pc.if_else(at_or_below, low, pc.add(middle, one))
                del low, high, middle, candidate_positions, candidates, at_or_below
                low, high = next_low, next_high
            candidate_positions = pc.min_element_wise(low, maximum_index)
            candidates = pc.take(values, candidate_positions)
            condition = pc.and_kleene(
                pc.less(low, pa.scalar(len(values), type=pa.int32())),
                pc.equal(column_values, candidates),
            )
            del column_values, values, low, high, candidate_positions, candidates
            filtered = batch.filter(condition)
            del batch, condition
            batch = filtered
            del filtered
            continue
        condition = operations[operator](
            batch.column(batch.schema.get_field_index(column)), value
        )
        filtered = batch.filter(condition)
        del batch, condition
        batch = filtered
        del filtered
    return batch


def _require_filter_batch_capacity(
    batch: pa.RecordBatch,
    *,
    operation_headroom_bytes: int | None,
    membership: bool,
    membership_max_scalar_bytes: int = 0,
) -> None:
    if operation_headroom_bytes is None:
        return
    required_bytes = _filter_working_set_bytes(
        source_batch_bytes=batch.get_total_buffer_size(),
        row_count=batch.num_rows,
        membership=membership,
        membership_max_scalar_bytes=membership_max_scalar_bytes,
    )
    if required_bytes > operation_headroom_bytes:
        raise DatasetReadCapacityExceeded(
            "dataset_filter_working_set_exceeds_task_memory_budget"
        )


def _filter_working_set_bytes(
    *,
    source_batch_bytes: int,
    row_count: int,
    membership: bool,
    membership_max_scalar_bytes: int = 0,
) -> int:
    boolean_buffer_bytes = 2 * ((row_count + 7) // 8)
    if membership:
        integer_array_bytes = row_count * 4
        candidate_array_bytes = row_count * membership_max_scalar_bytes
        search_index_state_bytes = 8 * integer_array_bytes
        search_condition_state_bytes = 4 * boolean_buffer_bytes
        condition_state_bytes = (
            search_index_state_bytes
            + candidate_array_bytes
            + search_condition_state_bytes
        )
    else:
        condition_state_bytes = boolean_buffer_bytes
    filtered_output_bytes = source_batch_bytes
    publication_state_bytes = boolean_buffer_bytes + filtered_output_bytes
    return source_batch_bytes + max(
        condition_state_bytes,
        publication_state_bytes,
    )


def _membership_max_scalar_bytes(value: PreparedMembershipValues) -> int:
    if not isinstance(value, PreparedMembershipValues):
        raise TypeError("prepared_membership_values_required")
    return value.max_scalar_bytes


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
            raise DatasetReadCapacityExceeded("dataset_row_exceeds_target_chunk_bytes")
        left_rows = candidate.num_rows // 2
        pending.append(candidate.slice(left_rows))
        pending.append(candidate.slice(0, left_rows))
