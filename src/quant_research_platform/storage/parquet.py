"""The sole handle-bound Parquet traversal for dataset storage."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    open_filesystem_file,
)

_DECODED_ROW_BOUNDS_KEY = b"qrp.decoded-row-bounds"
_DECODED_ROW_BOUNDS_SCHEMA_ID = "qrp-dataset-part-decoded-row-bounds"
_PHYSICAL_PAGE_ROWS_KEY = b"qrp.parquet-page-rows"
_PARQUET_MAX_ROW_GROUP_ROWS = 64 * 1024 * 1024
_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES = 5 * 1024 * 1024
_PARQUET_NATIVE_METADATA_PAGE_BYTES = 4096
_PARQUET_METADATA_FIXED_WORKSPACE_BYTES = 2 * 1024 * 1024
_PARQUET_READ_BUFFER_BYTES = 4096


@dataclass(frozen=True, slots=True)
class DatasetPartMetadata:
    identity: DatasetFileIdentity
    schema: pa.Schema
    row_count: int
    file_size_bytes: int
    logical_dtype_payload: bytes


@dataclass(frozen=True, slots=True)
class PreparedMembershipValues:
    values: pa.Array
    max_scalar_bytes: int
    retained_memory_bytes: int


def inspect_dataset_part(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
) -> DatasetPartMetadata:
    with _opened_parquet(path, expected_identity=expected_identity) as opened:
        parquet_file, identity, _handle = opened
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
    _operation_headroom: Callable[[], int] | None = None,
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
        parquet_file, identity, handle = opened
        policy = current_runtime_resource_policy()

        def current_capacity() -> int | None:
            capacity = (
                operation_headroom_bytes
                if _operation_headroom is None
                else _operation_headroom()
            )
            if policy is not None:
                available = policy.working_set_headroom_bytes
                capacity = available if capacity is None else min(capacity, available)
            return capacity

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
        logical_schema = pa.schema(
            [metadata.schema.field(column) for column in selected_columns]
        )
        selected_row_groups = (
            _matching_row_groups(parquet_file, parquet_filters)
            if row_groups is None
            else list(row_groups)
        )
        if not selected_row_groups:
            return
        previous_output_bytes = (
            [0] if _previous_output_bytes is None else _previous_output_bytes
        )
        membership = any(operator == "in" for _, operator, _ in parquet_filters or ())
        membership_max_scalar_bytes = max(
            (
                _membership_max_scalar_bytes(value)
                for _, operator, value in parquet_filters or ()
                if operator == "in"
            ),
            default=0,
        )
        footer_decoded_bounds = (parquet_file.metadata.metadata or {}).get(
            _DECODED_ROW_BOUNDS_KEY
        )
        file_metadata = parquet_file.metadata
        current_dictionary_paths = ()
        for row_group in selected_row_groups:
            dictionary_paths = tuple(
                parquet_file.schema.column(index).path
                for index in _dictionary_encoded_column_positions(
                    parquet_file, selected_columns, row_groups=[row_group]
                )
                if parquet_file.schema.column(index).physical_type == "BYTE_ARRAY"
            )
            if dictionary_paths != current_dictionary_paths:
                parquet_file.reader.open(
                    handle,
                    metadata=file_metadata,
                    read_dictionary=dictionary_paths,
                    buffer_size=_PARQUET_READ_BUFFER_BYTES,
                    pre_buffer=False,
                    arrow_extensions_enabled=True,
                )
                current_dictionary_paths = dictionary_paths
            decode_dictionary = any(
                parquet_file.schema_arrow.field(column).type
                != metadata.schema.field(column).type
                for column in selected_columns
            )
            compact_nested = any(
                parquet_file.schema_arrow.field(column).type
                != metadata.schema.field(column).type
                and not pa.types.is_dictionary(
                    parquet_file.schema_arrow.field(column).type
                )
                for column in selected_columns
            )
            initial_operation_headroom = current_capacity()
            initial_physical_headroom = (
                None
                if initial_operation_headroom is None
                else initial_operation_headroom - previous_output_bytes[0]
            )
            maximum_projected_row_bytes = _maximum_projected_row_bytes(
                parquet_file, selected_columns, row_groups=[row_group]
            )
            maximum_decode_fixed_bytes = _maximum_projected_decode_fixed_bytes(
                parquet_file, selected_columns, row_groups=[row_group]
            )
            selected_source_rows = int(file_metadata.row_group(row_group).num_rows)
            physical_page_rows = _physical_page_rows(parquet_file)
            maximum_page_rows = min(
                selected_source_rows,
                selected_source_rows
                if physical_page_rows is None
                else physical_page_rows,
            )
            projected_buffer_bytes = (
                len(_projected_parquet_leaf_positions(parquet_file, selected_columns))
                * _PARQUET_READ_BUFFER_BYTES
            )
            admitted_batch_rows = _target_batch_rows(
                maximum_row_bytes=maximum_projected_row_bytes,
                fixed_decode_bytes=maximum_decode_fixed_bytes,
                maximum_page_rows=maximum_page_rows,
                projected_buffer_bytes=projected_buffer_bytes,
                target_chunk_bytes=target_chunk_bytes,
                operation_headroom_bytes=initial_physical_headroom,
                filtered=bool(parquet_filters) and not decode_dictionary,
                membership=membership,
                membership_max_scalar_bytes=membership_max_scalar_bytes,
            )
            maximum_batch_rows = (
                admitted_batch_rows
                if source_batch_rows is None
                else min(source_batch_rows, admitted_batch_rows)
            )
            plain_row_upper = (
                _maximum_projected_row_bytes(
                    parquet_file,
                    selected_columns,
                    row_groups=[row_group],
                    output_schema=metadata.schema,
                )
                if decode_dictionary
                else 0
            )
            decode_upper_bytes = plain_row_upper * (2 if compact_nested else 1) + 8 * (
                len(selected_columns) + 1
            )
            if parquet_filters and decode_dictionary:
                decode_upper_bytes += (
                    _filter_working_set_bytes(
                        source_batch_bytes=plain_row_upper,
                        row_count=1,
                        membership=membership,
                        membership_max_scalar_bytes=membership_max_scalar_bytes,
                    )
                    - plain_row_upper
                )
            minimum_decode_bytes = 0
            minimum_native_bytes = (
                maximum_decode_fixed_bytes
                + _maximum_projected_page_decode_bytes(
                    maximum_row_bytes=maximum_projected_row_bytes,
                    maximum_page_rows=maximum_page_rows,
                    projected_buffer_bytes=projected_buffer_bytes,
                    batch_rows=1,
                )
            )
            batch_rows = 1 if decode_dictionary else maximum_batch_rows
            pulled_source_rows = 0
            with closing_iterator(
                parquet_file.iter_batches(
                    batch_size=batch_rows,
                    row_groups=[row_group],
                    columns=list(selected_columns),
                    use_threads=False,
                )
            ) as raw_batches:
                while True:
                    if pulled_source_rows >= selected_source_rows:
                        break
                    current_operation_headroom = current_capacity()
                    physical_headroom = (
                        None
                        if current_operation_headroom is None
                        else current_operation_headroom - previous_output_bytes[0]
                    )
                    if decode_dictionary and minimum_decode_bytes:
                        native_capacity = (
                            None
                            if physical_headroom is None
                            else physical_headroom - minimum_decode_bytes
                        )
                        batch_rows = (
                            1
                            if native_capacity is not None
                            and native_capacity < minimum_native_bytes
                            else min(
                                maximum_batch_rows,
                                _target_batch_rows(
                                    maximum_row_bytes=maximum_projected_row_bytes,
                                    fixed_decode_bytes=maximum_decode_fixed_bytes,
                                    maximum_page_rows=maximum_page_rows,
                                    projected_buffer_bytes=projected_buffer_bytes,
                                    target_chunk_bytes=target_chunk_bytes,
                                    operation_headroom_bytes=native_capacity,
                                ),
                            )
                        )
                    next_batch_rows = min(
                        batch_rows,
                        selected_source_rows - pulled_source_rows,
                    )
                    maximum_page_decode_bytes = (
                        maximum_decode_fixed_bytes
                        + _maximum_projected_page_decode_bytes(
                            maximum_row_bytes=maximum_projected_row_bytes,
                            maximum_page_rows=maximum_page_rows,
                            projected_buffer_bytes=projected_buffer_bytes,
                            batch_rows=next_batch_rows,
                        )
                    )
                    source_batch_bytes = next_batch_rows * maximum_projected_row_bytes
                    physical_pull_bytes = (
                        maximum_page_decode_bytes
                        + max(
                            0,
                            _filter_working_set_bytes(
                                source_batch_bytes=source_batch_bytes,
                                row_count=next_batch_rows,
                                membership=membership,
                                membership_max_scalar_bytes=membership_max_scalar_bytes,
                            )
                            - source_batch_bytes,
                        )
                        if parquet_filters and not decode_dictionary
                        else max(
                            maximum_page_decode_bytes,
                            maximum_decode_fixed_bytes + source_batch_bytes,
                        )
                    )
                    if (
                        physical_headroom is not None
                        and physical_pull_bytes > physical_headroom
                    ):
                        raise DatasetReadCapacityExceeded(
                            "dataset_working_set_exceeds_task_memory_budget"
                        )
                    with ExitStack() as pull_reservation:
                        pull_lease = None
                        if policy is not None:
                            policy.require_working_set(
                                physical_pull_bytes,
                                error="dataset_working_set_exceeds_task_memory_budget",
                            )
                            pull_lease = pull_reservation.enter_context(
                                policy.lease(
                                    "storage.parquet.read",
                                    physical_pull_bytes,
                                )
                            )
                        try:
                            if decode_dictionary:
                                parquet_file.reader.set_batch_size(next_batch_rows)
                            raw_batch = next(raw_batches)
                        except StopIteration:
                            break
                        actual_buffer_bytes = raw_batch.get_total_buffer_size()
                        actual_pull_bytes = actual_buffer_bytes + (
                            _filter_working_set_bytes(
                                source_batch_bytes=actual_buffer_bytes,
                                row_count=raw_batch.num_rows,
                                membership=membership,
                                membership_max_scalar_bytes=membership_max_scalar_bytes,
                            )
                            - actual_buffer_bytes
                            if parquet_filters and not decode_dictionary
                            else 0
                        )
                        if (
                            physical_headroom is not None
                            and actual_pull_bytes > physical_headroom
                        ):
                            del raw_batch
                            raise DatasetReadCapacityExceeded(
                                "dataset_working_set_exceeds_task_memory_budget"
                            )
                        if (
                            pull_lease is not None
                            and actual_pull_bytes > pull_lease.reserved_bytes
                        ):
                            additional = actual_pull_bytes - pull_lease.reserved_bytes
                            policy.require_working_set(
                                additional,
                                error="dataset_working_set_exceeds_task_memory_budget",
                            )
                            pull_lease.grow(additional)
                        pulled_source_rows += raw_batch.num_rows
                        if footer_decoded_bounds is not None:
                            batch_metadata = dict(raw_batch.schema.metadata or {})
                            batch_metadata[_DECODED_ROW_BOUNDS_KEY] = (
                                footer_decoded_bounds
                            )
                            raw_batch = raw_batch.replace_schema_metadata(
                                batch_metadata
                            )
                        if decode_dictionary:
                            native_bytes = (
                                pull_lease.reserved_bytes
                                if pull_lease is not None
                                else max(physical_pull_bytes, actual_pull_bytes)
                            )
                            with closing_iterator(
                                _decoded_dictionary_batches(
                                    raw_batch,
                                    schema=logical_schema,
                                    target_chunk_bytes=target_chunk_bytes,
                                    operation_headroom_bytes=(
                                        None
                                        if physical_headroom is None
                                        else physical_headroom - native_bytes
                                    ),
                                    filters=parquet_filters,
                                    membership=membership,
                                    membership_max_scalar_bytes=membership_max_scalar_bytes,
                                )
                            ) as decoded:
                                observed_decode_bytes = yield from decoded
                            minimum_decode_bytes = max(
                                observed_decode_bytes, decode_upper_bytes
                            )
                            del raw_batch
                            continue
                        filtered = _filter_arrow_batch(
                            raw_batch,
                            parquet_filters,
                            operation_headroom_bytes=physical_headroom,
                        )
                        del raw_batch
                        yield filtered
                        del filtered


def _decoded_dictionary_batches(
    batch: pa.RecordBatch,
    *,
    schema: pa.Schema,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None,
    filters: list[tuple[str, str, Any]] | None,
    membership: bool,
    membership_max_scalar_bytes: int,
) -> Generator[pa.RecordBatch, None, int]:
    bounds = [
        _column_decoded_row_bound(pa.chunked_array([column]), logical_type=field.type)
        for column, field in zip(batch.columns, schema, strict=True)
    ]
    logical_row_bytes = sum(row for row, _fixed in bounds)
    cast_bounds = [
        bound
        for column, field, bound in zip(batch.columns, schema, bounds, strict=True)
        if column.type != field.type
    ]
    cast_fixed_bytes = sum(fixed for _row, fixed in cast_bounds)
    new_row_bytes = sum(row for row, _fixed in cast_bounds)
    compact_nested = any(
        column.type != field.type and not pa.types.is_dictionary(column.type)
        for column, field in zip(batch.columns, schema, strict=True)
    )
    if compact_nested:
        # Nested Arrow casts visit the whole child array behind a sliced parent.
        # Take first so only the admitted rows have children to decode.
        new_row_bytes += 8 + sum(
            _column_decoded_row_bound(
                pa.chunked_array([column]), logical_type=column.type
            )[0]
            for column in batch.columns
        )
    if filters:
        new_row_bytes += (
            _filter_working_set_bytes(
                source_batch_bytes=logical_row_bytes,
                row_count=1,
                membership=membership,
                membership_max_scalar_bytes=membership_max_scalar_bytes,
            )
            - logical_row_bytes
        )
    target_rows = max(1, target_chunk_bytes // logical_row_bytes)
    policy = current_runtime_resource_policy()
    start = 0
    while start < batch.num_rows:
        capacity = operation_headroom_bytes
        if policy is not None:
            capacity = (
                policy.working_set_headroom_bytes
                if capacity is None
                else min(capacity, policy.working_set_headroom_bytes)
            )
        rows = min(target_rows, batch.num_rows - start)
        if capacity is not None:
            rows = min(rows, (capacity - cast_fixed_bytes) // new_row_bytes)
        if rows < 1:
            raise DatasetReadCapacityExceeded(
                "dataset_working_set_exceeds_task_memory_budget"
            )
        with ExitStack() as conversion:
            if policy is not None:
                conversion.enter_context(
                    policy.lease(
                        "storage.parquet.decode",
                        cast_fixed_bytes + rows * new_row_bytes,
                    )
                )
            compact = restored = filtered = None
            try:
                compact = (
                    batch.take(pa.array(range(start, start + rows), type=pa.int64()))
                    if compact_nested
                    else batch.slice(start, rows)
                )
                restored = compact.cast(schema.with_metadata(batch.schema.metadata))
                filtered = _filter_arrow_batch(
                    restored, filters, operation_headroom_bytes=capacity
                )
                yield filtered
            finally:
                compact = restored = filtered = None
        start += rows
    return cast_fixed_bytes + new_row_bytes


@contextmanager
def _opened_parquet(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None,
) -> Iterator[tuple[pq.ParquetFile, DatasetFileIdentity, Any]]:
    resolved = Path(path)
    path_identity = dataset_file_identity(resolved)
    if expected_identity is not None and path_identity != expected_identity:
        raise ValueError("dataset_file_identity_changed")
    handle = open_filesystem_file(resolved)
    try:
        before = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
        if before != path_identity:
            raise ValueError("dataset_file_identity_changed")
        footer_length = _parquet_footer_length(handle)
        with ExitStack() as parquet_stack:
            policy = current_runtime_resource_policy()
            if policy is not None:
                parquet_stack.enter_context(
                    policy.lease(
                        "storage.parquet.metadata",
                        _PARQUET_METADATA_FIXED_WORKSPACE_BYTES + 20 * footer_length,
                    )
                )
                policy.require_working_set(
                    0, error="dataset_working_set_exceeds_task_memory_budget"
                )
            parquet_file = pq.ParquetFile(
                handle,
                thrift_string_size_limit=max(1, footer_length),
                thrift_container_size_limit=max(1, footer_length),
                buffer_size=_PARQUET_READ_BUFFER_BYTES,
                pre_buffer=False,
            )
            try:
                yield parquet_file, before, handle
            finally:
                primary = sys.exception()
                if primary is not None:
                    try:
                        after = dataset_file_identity_from_stat(
                            os.fstat(handle.fileno())
                        )
                        if after != before:
                            raise ValueError("dataset_file_identity_changed")
                    except BaseException as verify_error:
                        primary.add_note(
                            "parquet handle verification also failed: "
                            f"{type(verify_error).__name__}"
                        )
                    try:
                        parquet_file.close()
                    except BaseException as close_error:
                        primary.add_note(
                            "parquet file close also failed: "
                            f"{type(close_error).__name__}"
                        )
                else:
                    try:
                        after = dataset_file_identity_from_stat(
                            os.fstat(handle.fileno())
                        )
                    except BaseException as primary:
                        try:
                            parquet_file.close()
                        except BaseException as close_error:
                            primary.add_note(
                                "parquet file close also failed: "
                                f"{type(close_error).__name__}"
                            )
                        raise
                    parquet_file.close()
                    if after != before:
                        raise ValueError("dataset_file_identity_changed")
    finally:
        primary = sys.exception()
        if primary is None:
            first_error: BaseException | None = None
            try:
                handle.close()
            except BaseException as close_error:
                first_error = close_error
            if first_error is not None:
                raise first_error

        else:
            try:
                handle.close()
            except BaseException as close_error:
                primary.add_note(
                    f"parquet handle close also failed: {type(close_error).__name__}"
                )


def _parquet_footer_length(handle: Any) -> int:
    try:
        position = handle.tell()
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        if file_size < 8:
            raise ValueError("dataset_parquet_footer_invalid")
        handle.seek(-8, os.SEEK_END)
        trailer = handle.read(8)
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "dataset_parquet_footer_invalid":
            raise
        raise ValueError("dataset_parquet_footer_invalid") from exc
    finally:
        try:
            handle.seek(position)
        except (OSError, UnboundLocalError):
            pass
    if len(trailer) != 8 or trailer[4:] != b"PAR1":
        raise ValueError("dataset_parquet_footer_invalid")
    footer_length = int.from_bytes(trailer[:4], "little")
    if footer_length > file_size - 8:
        raise ValueError("dataset_parquet_footer_invalid")
    return footer_length


def _part_metadata(
    parquet_file: pq.ParquetFile,
    *,
    identity: DatasetFileIdentity,
    require_logical_dtypes: bool = True,
) -> DatasetPartMetadata:
    schema = parquet_file.schema.to_arrow_schema()
    metadata = schema.metadata or {}
    logical_dtype_payload = metadata.get(b"qrp.logical-dtypes")
    if logical_dtype_payload is None and require_logical_dtypes:
        raise ValueError("dataset_logical_dtypes_missing")
    return DatasetPartMetadata(
        identity=identity,
        schema=schema.remove_metadata(),
        row_count=int(parquet_file.metadata.num_rows),
        file_size_bytes=identity.size,
        logical_dtype_payload=(
            b"" if logical_dtype_payload is None else bytes(logical_dtype_payload)
        ),
    )


def _target_batch_rows(
    *,
    maximum_row_bytes: int,
    fixed_decode_bytes: int,
    maximum_page_rows: int,
    projected_buffer_bytes: int,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    filtered: bool = False,
    membership: bool = False,
    membership_max_scalar_bytes: int = 0,
) -> int:
    target_rows = max(1, target_chunk_bytes // maximum_row_bytes)
    if operation_headroom_bytes is None:
        return target_rows
    low = 1
    high = target_rows
    admitted = 0
    while low <= high:
        candidate = (low + high) // 2
        page_decode_bytes = fixed_decode_bytes + _maximum_projected_page_decode_bytes(
            maximum_row_bytes=maximum_row_bytes,
            maximum_page_rows=maximum_page_rows,
            projected_buffer_bytes=projected_buffer_bytes,
            batch_rows=candidate,
        )
        source_batch_bytes = candidate * maximum_row_bytes
        logical_required_bytes = fixed_decode_bytes + (
            _filter_working_set_bytes(
                source_batch_bytes=source_batch_bytes,
                row_count=candidate,
                membership=membership,
                membership_max_scalar_bytes=membership_max_scalar_bytes,
            )
            if filtered
            else source_batch_bytes
        )
        required_bytes = (
            page_decode_bytes
            + max(
                0,
                logical_required_bytes - fixed_decode_bytes - source_batch_bytes,
            )
            if filtered
            else max(page_decode_bytes, logical_required_bytes)
        )
        if required_bytes <= operation_headroom_bytes:
            admitted = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    if admitted == 0:
        raise DatasetReadCapacityExceeded(
            "dataset_filter_working_set_exceeds_task_memory_budget"
            if filtered
            else "dataset_working_set_exceeds_task_memory_budget"
        )
    return admitted


def _maximum_projected_row_bytes(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
    *,
    row_groups: list[int] | tuple[int, ...] | None = None,
    output_schema: pa.Schema | None = None,
) -> int:
    decoded_bounds = _decoded_row_bounds(parquet_file)
    selected_groups = (
        range(parquet_file.metadata.num_row_groups)
        if row_groups is None
        else row_groups
    )
    byte_dictionary_positions = {
        index
        for index in _dictionary_encoded_column_positions(
            parquet_file, columns, row_groups=selected_groups
        )
        if parquet_file.schema.column(index).physical_type == "BYTE_ARRAY"
    }
    native_schema = (
        parquet_file.schema_arrow if output_schema is None else output_schema
    )
    native_types = tuple(
        leaf for field in native_schema for leaf in _arrow_leaf_types(field.type)
    )
    dictionary_positions = {
        index
        for index in _projected_parquet_leaf_positions(parquet_file, columns)
        if pa.types.is_dictionary(native_types[index])
    }
    maximum_row_bytes = 0
    for column in columns:
        positions = _projected_parquet_leaf_positions(parquet_file, (column,))
        encoded_positions = dictionary_positions.intersection(positions)
        native_index_bytes = sum(
            native_types[index].index_type.bit_width // 8 + 1
            for index in encoded_positions
        )
        column_row_bytes = max(
            native_index_bytes,
            _column_decoded_row_bound(
                pa.chunked_array([], type=native_schema.field(column).type)
            )[0],
        )
        if decoded_bounds is not None and not encoded_positions:
            column_row_bytes = max(column_row_bytes, decoded_bounds[column][0])
        for row_group_index in selected_groups:
            group = parquet_file.metadata.row_group(row_group_index)
            if group.num_rows <= 0:
                continue
            group_dictionary_positions = set(
                _dictionary_encoded_column_positions(
                    parquet_file, (column,), row_groups=[row_group_index]
                )
            )
            physical_bytes = sum(
                int(group.column(index).total_uncompressed_size)
                for index in positions
                if index not in encoded_positions
            )
            plain_dictionary_bytes = sum(
                int(group.column(index).total_uncompressed_size)
                * (
                    max(1, int(group.column(index).num_values))
                    if parquet_file.schema.column(index).max_repetition_level
                    else 1
                )
                + 9
                for index in positions
                if index in byte_dictionary_positions
                and index in group_dictionary_positions
                and index not in encoded_positions
            )
            column_row_bytes = max(
                column_row_bytes,
                native_index_bytes
                + plain_dictionary_bytes
                + (physical_bytes + group.num_rows - 1) // group.num_rows,
            )
        maximum_row_bytes += column_row_bytes
    return max(1, maximum_row_bytes)


def _arrow_leaf_types(data_type):
    if data_type.num_fields:
        for index in range(data_type.num_fields):
            yield from _arrow_leaf_types(data_type.field(index).type)
    else:
        yield data_type


def _maximum_projected_decode_fixed_bytes(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
    *,
    row_groups: list[int] | tuple[int, ...] | None = None,
) -> int:
    decoded_bounds = _decoded_row_bounds(parquet_file)
    selected_groups = (
        range(parquet_file.metadata.num_row_groups)
        if row_groups is None
        else row_groups
    )
    dictionary_positions = _dictionary_encoded_column_positions(
        parquet_file, columns, row_groups=selected_groups
    )
    native_types = tuple(
        leaf
        for field in parquet_file.schema_arrow
        for leaf in _arrow_leaf_types(field.type)
    )
    native_dictionary_positions = {
        index
        for index in _projected_parquet_leaf_positions(parquet_file, columns)
        if pa.types.is_dictionary(native_types[index])
    }
    dictionary_columns = {
        name
        for name in columns
        if set(_projected_parquet_leaf_positions(parquet_file, (name,))).intersection(
            (*dictionary_positions, *native_dictionary_positions)
        )
    }
    physical_bounds = dict.fromkeys(columns, 0)
    for row_group_index in selected_groups:
        group = parquet_file.metadata.row_group(row_group_index)
        group_bounds = dict.fromkeys(columns, 0)
        for index in native_dictionary_positions.union(
            _dictionary_encoded_column_positions(
                parquet_file, columns, row_groups=[row_group_index]
            )
        ):
            leaf = parquet_file.schema.column(index)
            column = group.column(index)
            # The chunk bounds its dictionary payload even when writer hints lie.
            # Include Arrow offsets, validity, and dictionary-length inspection.
            payload_factor = 4 if leaf.physical_type == "BYTE_ARRAY" else 2
            bound = payload_factor * int(column.total_uncompressed_size)
            if leaf.max_repetition_level:
                bound += 16 * leaf.max_repetition_level * int(column.num_values)
            name = next(
                name
                for name in columns
                if leaf.path == name or leaf.path.startswith(f"{name}.")
            )
            group_bounds[name] += bound
        for name in columns:
            physical_bounds[name] = max(physical_bounds[name], group_bounds[name])
    try:
        return sum(
            physical_bounds[column]
            if column in dictionary_columns
            else 0
            if decoded_bounds is None
            else decoded_bounds[column][1]
            for column in columns
        )
    except KeyError:
        raise DatasetReadCapacityExceeded(
            "dataset_decode_bound_metadata_invalid"
        ) from None


def _dictionary_encoded_column_positions(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
    *,
    row_groups: list[int] | tuple[int, ...] | range | None = None,
) -> tuple[int, ...]:
    selected_groups = (
        range(parquet_file.metadata.num_row_groups)
        if row_groups is None
        else row_groups
    )
    return tuple(
        index
        for index in _projected_parquet_leaf_positions(parquet_file, columns)
        if any(
            {"RLE_DICTIONARY", "PLAIN_DICTIONARY"}.intersection(
                parquet_file.metadata.row_group(group).column(index).encodings
            )
            for group in selected_groups
        )
    )


def _maximum_projected_page_decode_bytes(
    *,
    maximum_row_bytes: int,
    maximum_page_rows: int,
    projected_buffer_bytes: int,
    batch_rows: int,
) -> int:
    if maximum_page_rows <= 0:
        return 0
    return (
        projected_buffer_bytes + min(maximum_page_rows, batch_rows) * maximum_row_bytes
    )


def _physical_page_rows(parquet_file: pq.ParquetFile) -> int | None:
    raw = (parquet_file.metadata.metadata or {}).get(_PHYSICAL_PAGE_ROWS_KEY)
    if raw is None:
        raw = (parquet_file.schema_arrow.metadata or {}).get(_PHYSICAL_PAGE_ROWS_KEY)
    if raw is None:
        return None
    try:
        text = bytes(raw).decode("ascii")
        if not text.isascii() or not text.isdecimal():
            raise ValueError
        value = int(text)
        if value <= 0:
            raise ValueError
        return value
    except (UnicodeDecodeError, ValueError):
        raise DatasetReadCapacityExceeded(
            "dataset_decode_bound_metadata_invalid"
        ) from None


def table_with_decoded_row_bounds(
    table: pa.Table,
    *,
    previous_metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    previous_bounds = (
        {}
        if previous_metadata is None or _DECODED_ROW_BOUNDS_KEY not in previous_metadata
        else _parse_decoded_row_bounds(
            previous_metadata[_DECODED_ROW_BOUNDS_KEY],
            names=table.column_names,
        )
    )
    columns = []
    for name, column in zip(table.column_names, table.columns, strict=True):
        per_row_bytes, fixed_bytes = _column_decoded_row_bound(column)
        previous_per_row, previous_fixed = previous_bounds.get(name, (0, 0))
        columns.append(
            {
                "name": name,
                "per_row_bytes": max(previous_per_row, per_row_bytes),
                "fixed_bytes": max(
                    previous_fixed,
                    fixed_bytes + per_row_bytes,
                ),
            }
        )
    payload = render_qrp_json_object(
        {
            "schema_id": _DECODED_ROW_BOUNDS_SCHEMA_ID,
            "columns": columns,
        }
    ).encode("utf-8")
    metadata = dict(previous_metadata or {})
    metadata.update(table.schema.metadata or {})
    metadata[_DECODED_ROW_BOUNDS_KEY] = payload
    return table.replace_schema_metadata(metadata)


def _table_parquet_page_working_set_bytes(table: pa.Table, *, rows: int) -> int:
    if isinstance(rows, bool) or type(rows) is not int or rows < 0:
        raise ValueError("invalid_parquet_page_rows")
    decoded_bounds = _parse_decoded_row_bounds(
        (table.schema.metadata or {}).get(_DECODED_ROW_BOUNDS_KEY),
        names=table.column_names,
    )
    decoded_bytes = sum(
        fixed + rows * per_row for per_row, fixed in decoded_bounds.values()
    )
    compressed_bound = sum(
        _zstd_compressed_bound(rows * per_row)
        for per_row, _fixed in decoded_bounds.values()
    )
    return _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES + decoded_bytes + compressed_bound


def _parquet_row_group_size(rows: int) -> int:
    if isinstance(rows, bool) or type(rows) is not int or rows < 0:
        raise ValueError("invalid_parquet_row_group_rows")
    return max(1, min(rows, _PARQUET_MAX_ROW_GROUP_ROWS))


def _parquet_row_group_footer_upper_bound(
    schema: pa.Schema,
    *,
    rows: int,
) -> int:
    """Bound retained footer state with statistics and indexes disabled."""

    group_rows = _parquet_row_group_size(rows)
    group_count = max(1, (rows + group_rows - 1) // group_rows)
    variable_bytes = sum(
        len(field.name.encode("utf-8")) + len(str(field.type).encode("utf-8"))
        for field in schema
    ) + sum(len(key) + len(value) for key, value in (schema.metadata or {}).items())
    per_group = _PARQUET_NATIVE_METADATA_PAGE_BYTES * (len(schema) + 1) + variable_bytes
    return group_count * per_group


def _zstd_compressed_bound(source_bytes: int) -> int:
    """Return the codec's exact ZSTD_COMPRESSBOUND for one page payload."""

    return (
        source_bytes
        + (source_bytes >> 8)
        + (((128 << 10) - source_bytes) >> 11 if source_bytes < 128 << 10 else 0)
    )


def parquet_decoded_working_set_bounds(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> tuple[int, int]:
    return (
        _maximum_projected_row_bytes(parquet_file, columns),
        _maximum_projected_decode_fixed_bytes(
            parquet_file,
            columns,
        ),
    )


def record_batch_decoded_working_set_bounds(
    batch: pa.RecordBatch,
) -> tuple[int, int]:
    raw = (batch.schema.metadata or {}).get(_DECODED_ROW_BOUNDS_KEY)
    hinted_row_bytes = hinted_fixed_bytes = 0
    if raw is not None:
        decoded_bounds = _parse_decoded_row_bounds(
            raw,
            names=batch.schema.names,
            exact_names=False,
        )
        hinted_row_bytes = sum(bound[0] for bound in decoded_bounds.values())
        hinted_fixed_bytes = sum(bound[1] for bound in decoded_bounds.values())
    bounds = tuple(
        _column_decoded_row_bound(pa.chunked_array([column]))
        for column in batch.columns
    )
    return (
        max(1, hinted_row_bytes, sum(bound[0] for bound in bounds)),
        max(hinted_fixed_bytes, sum(bound[1] for bound in bounds)),
    )


def _column_decoded_row_bound(
    column: pa.ChunkedArray, *, logical_type: pa.DataType | None = None
) -> tuple[int, int]:
    data_type = column.type if logical_type is None else logical_type
    if pa.types.is_dictionary(column.type) and not pa.types.is_dictionary(data_type):
        return _column_decoded_row_bound(
            pa.chunked_array([chunk.dictionary for chunk in column.chunks]),
            logical_type=data_type,
        )
    if pa.types.is_dictionary(data_type):
        index_bytes = max(1, int(data_type.index_type.bit_width) // 8)
        dictionary_bytes = (
            sum(
                int(chunk.dictionary.nbytes)
                for chunk in column.chunks
                if isinstance(chunk, pa.DictionaryArray)
            )
            if logical_type is None
            else 0
        )
        return index_bytes + 1, dictionary_bytes
    if logical_type is not None and pa.types.is_struct(data_type):
        bounds = [
            _column_decoded_row_bound(
                pa.chunked_array([chunk.field(index) for chunk in column.chunks]),
                logical_type=field.type,
            )
            for index, field in enumerate(data_type)
        ]
        return 1 + sum(row for row, _fixed in bounds), sum(
            fixed for _row, fixed in bounds
        )
    if logical_type is not None and (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
        or pa.types.is_map(data_type)
    ):
        fixed_list = pa.types.is_fixed_size_list(data_type)
        offset_bytes = (
            0 if fixed_list else 8 if pa.types.is_large_list(data_type) else 4
        )
        maximum_length = data_type.list_size if fixed_list else 0
        if not fixed_list:
            for chunk in column.chunks:
                if len(chunk):
                    lengths = pc.subtract(
                        chunk.offsets.slice(1), chunk.offsets.slice(0, len(chunk))
                    )
                    maximum_length = max(maximum_length, int(pc.max(lengths).as_py()))
        if pa.types.is_map(data_type):
            bounds = [
                _column_decoded_row_bound(
                    pa.chunked_array(
                        [getattr(chunk, member) for chunk in column.chunks]
                    ),
                    logical_type=child_type,
                )
                for member, child_type in (
                    ("keys", data_type.key_type),
                    ("items", data_type.item_type),
                )
            ]
        else:
            bounds = [
                _column_decoded_row_bound(
                    pa.chunked_array([chunk.values for chunk in column.chunks]),
                    logical_type=data_type.value_type,
                )
            ]
        return (
            offset_bytes + 1 + maximum_length * sum(row for row, _fixed in bounds),
            offset_bytes + sum(fixed for _row, fixed in bounds),
        )
    if (
        pa.types.is_string(data_type)
        or pa.types.is_binary(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_large_binary(data_type)
    ):
        maximum = 0
        for chunk in column.chunks:
            if len(chunk):
                observed = pc.max(pc.binary_length(chunk)).as_py()
                maximum = max(maximum, 0 if observed is None else int(observed))
        offset_bytes = (
            8
            if pa.types.is_large_string(data_type)
            or pa.types.is_large_binary(data_type)
            else 4
        )
        return maximum + offset_bytes + 1, offset_bytes
    if pa.types.is_null(data_type):
        return 2, 0
    if pa.types.is_boolean(data_type):
        return 2, 0
    try:
        bit_width = data_type.bit_width
    except (AttributeError, ValueError):
        bit_width = None
    if type(bit_width) is int and bit_width > 0:
        return max(1, (bit_width + 7) // 8) + 1, 0
    return max(1, int(column.nbytes)), 0


def _decoded_row_bounds(
    parquet_file: pq.ParquetFile,
) -> dict[str, tuple[int, int]] | None:
    raw = (parquet_file.metadata.metadata or {}).get(_DECODED_ROW_BOUNDS_KEY)
    if raw is None:
        raw = (parquet_file.schema_arrow.metadata or {}).get(_DECODED_ROW_BOUNDS_KEY)
    if raw is None:
        return None
    return _parse_decoded_row_bounds(raw, names=parquet_file.schema_arrow.names)


def _parse_decoded_row_bounds(
    raw: bytes | None,
    *,
    names: list[str],
    exact_names: bool = True,
) -> dict[str, tuple[int, int]]:
    if raw is None:
        raise DatasetReadCapacityExceeded("dataset_decode_bound_metadata_invalid")
    try:
        payload = parse_qrp_json_object(bytes(raw))
        if (
            set(payload) != {"schema_id", "columns"}
            or payload.get("schema_id") != _DECODED_ROW_BOUNDS_SCHEMA_ID
        ):
            raise ValueError
        items = payload["columns"]
        if type(items) is not list:
            raise ValueError
        result: dict[str, tuple[int, int]] = {}
        for item in items:
            if type(item) is not dict or set(item) != {
                "name",
                "per_row_bytes",
                "fixed_bytes",
            }:
                raise ValueError
            name = item["name"]
            per_row = item["per_row_bytes"]
            fixed = item["fixed_bytes"]
            if (
                type(name) is not str
                or not name
                or name in result
                or type(per_row) is not int
                or per_row <= 0
                or type(fixed) is not int
                or fixed < 0
            ):
                raise ValueError
            result[name] = (per_row, fixed)
        expected = set(names)
        if (exact_names and set(result) != expected) or not expected.issubset(result):
            raise ValueError
        return {name: result[name] for name in names}
    except (KeyError, TypeError, ValueError):
        raise DatasetReadCapacityExceeded(
            "dataset_decode_bound_metadata_invalid"
        ) from None


def _projected_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> tuple[int, ...]:
    schema = parquet_file.schema
    paths = tuple(
        schema.column(index).path for index in range(parquet_file.metadata.num_columns)
    )
    positions: list[int] = []
    for column in columns:
        prefix = f"{column}."
        matches = [
            index
            for index, path in enumerate(paths)
            if path == column or path.startswith(prefix)
        ]
        if not matches:
            raise ValueError("dataset_column_not_found")
        positions.extend(matches)
    return tuple(positions)


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
        column_values = batch.column(batch.schema.get_field_index(column))
        scalar = (
            pa.scalar(value, type=column_values.type)
            if pa.types.is_timestamp(column_values.type)
            else value
        )
        condition = operations[operator](column_values, scalar)
        del column_values, scalar
        filtered = batch.filter(condition)
        del batch, condition
        batch = filtered
        del filtered
    return batch


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


def _parquet_frame_bound(
    frame: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
) -> int:
    payload_bytes = pandas_frame_payload_bytes(frame)
    row_count = len(frame)
    column_count = len(frame.columns)
    index_bytes = int(frame.index.memory_usage(deep=True))
    arrow_bytes = payload_bytes + row_count * 9 * column_count
    timestamp_bytes = index_bytes + 9 * row_count if "timestamp" in key_columns else 0
    key_mask_bytes = index_bytes + row_count if key_columns else 0
    return payload_bytes + arrow_bytes + timestamp_bytes + key_mask_bytes


def _parquet_frames_reservation(
    frames: Iterable[tuple[pd.DataFrame, tuple[str, ...]]],
    *,
    owned_lease: MemoryLease | None = None,
    borrowed_input: bool = False,
) -> ExitStack:
    bound = sum(
        _parquet_frame_bound(frame, key_columns=key_columns)
        - (pandas_frame_payload_bytes(frame) if borrowed_input else 0)
        for frame, key_columns in frames
    )
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is None:
        if owned_lease is not None:
            raise RuntimeError("runtime_resource_policy_not_active")
        return stack
    try:
        if owned_lease is None:
            stack.enter_context(policy.lease("storage.parquet.payload", bound))
        else:
            if owned_lease._root is not policy or not owned_lease.active:
                raise RuntimeError("runtime_memory_lease_transfer_mismatch")
            owned_lease.transfer("storage.parquet.payload")
            if bound > owned_lease.reserved_bytes:
                owned_lease.grow(bound - owned_lease.reserved_bytes)
            elif bound < owned_lease.reserved_bytes:
                owned_lease.shrink_to(bound)
            stack.callback(
                _close_lease_preserving_primary,
                owned_lease,
                owner="storage.parquet.payload",
            )
    except BaseException as primary:
        if owned_lease is not None and owned_lease.active:
            try:
                owned_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "artifact output lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    return stack


def _parquet_writer_fixed_reservation(
    *,
    writer_count: int,
    index_bytes: int,
) -> ExitStack:
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is not None:
        stack.enter_context(
            policy.lease(
                "storage.parquet.fixed",
                index_bytes + writer_count * _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
            )
        )
        if policy.operation_headroom_bytes <= 0:
            stack.close()
            raise MemoryError("runtime_memory_budget_exceeded")
    return stack


def _parquet_frame_reservation(
    frame: pd.DataFrame | None,
    *,
    key_columns: tuple[str, ...] = (),
    owned_lease: MemoryLease | None = None,
    borrowed_input: bool = False,
) -> ExitStack:
    if frame is None:
        if owned_lease is not None:
            raise TypeError("invalid_artifact_output_frame")
        return ExitStack()
    return _parquet_frames_reservation(
        ((frame, key_columns),),
        owned_lease=owned_lease,
        borrowed_input=borrowed_input,
    )


def _parquet_table_write_state(
    table: pa.Table,
    *,
    previous_metadata: dict[bytes, bytes] | None = None,
    retained_memory: tuple[int, int] = (0, 0),
) -> tuple[pa.Table, ExitStack, int, tuple[int, int]]:
    persisted = table_with_decoded_row_bounds(
        table,
        previous_metadata=previous_metadata,
    )
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is None:
        return persisted, stack, max(1, persisted.num_rows), (0, 0)
    retained_page_bytes, retained_footer_bytes = retained_memory
    retained_bytes = retained_page_bytes + retained_footer_bytes
    footer_increment = _parquet_row_group_footer_upper_bound(
        persisted.schema,
        rows=persisted.num_rows,
    )
    page_headroom = (
        policy.operation_headroom_bytes + retained_page_bytes - footer_increment
    )
    minimum_rows = 0 if persisted.num_rows == 0 else 1
    minimum_bytes = _table_parquet_page_working_set_bytes(
        persisted,
        rows=minimum_rows,
    )
    if minimum_bytes > page_headroom:
        raise MemoryError("runtime_memory_budget_exceeded")
    batch_rows = max(1, persisted.num_rows)
    working_bytes = minimum_bytes
    low = 1
    high = _parquet_row_group_size(persisted.num_rows)
    while low <= high:
        candidate = (low + high) // 2
        candidate_bytes = _table_parquet_page_working_set_bytes(
            persisted,
            rows=candidate,
        )
        if candidate_bytes <= page_headroom:
            batch_rows = candidate
            working_bytes = candidate_bytes
            low = candidate + 1
        else:
            high = candidate - 1
    next_memory = (
        max(retained_page_bytes, working_bytes),
        retained_footer_bytes + footer_increment,
    )
    next_retained = sum(next_memory)
    stack.enter_context(
        policy.lease(
            "storage.parquet.native",
            next_retained - retained_bytes,
        )
    )
    return persisted, stack, batch_rows, next_memory


@contextmanager
def parquet_table_writer(handle, *, parent: Path, compression: str, schema=None):
    """Own one native stream, its page/footer reservations and primary failure."""
    with (
        handle_preserving_primary(ExitStack()) as state,
        handle_preserving_primary(ExitStack()) as native,
    ):
        state.enter_context(
            _parquet_writer_fixed_reservation(writer_count=1, index_bytes=0)
        )
        policy = current_runtime_resource_policy()
        retained = (
            (_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES, 0) if policy is not None else (0, 0)
        )
        metadata = None
        writer = None

        def write_table(table):
            nonlocal retained, metadata, writer
            persisted = None
            try:
                persisted, reservation, batch_rows, retained = (
                    _parquet_table_write_state(
                        table, previous_metadata=metadata, retained_memory=retained
                    )
                )
                state.enter_context(reservation)
                group_rows = _parquet_row_group_size(persisted.num_rows)
                extra_groups = (persisted.num_rows + batch_rows - 1) // batch_rows - (
                    persisted.num_rows + group_rows - 1
                ) // group_rows
                if policy is not None and extra_groups:
                    extra_footer = extra_groups * _parquet_row_group_footer_upper_bound(
                        persisted.schema, rows=0
                    )
                    state.enter_context(
                        policy.lease("storage.parquet.native", extra_footer)
                    )
                    retained = retained[0], retained[1] + extra_footer
                metadata = dict(persisted.schema.metadata or {})
                require_free_disk_headroom(
                    parent,
                    reserve_bytes=_table_parquet_page_working_set_bytes(
                        persisted, rows=persisted.num_rows
                    )
                    + retained[1],
                )
                if writer is None:
                    writer = native.enter_context(
                        handle_preserving_primary(
                            pq.ParquetWriter(
                                handle,
                                persisted.schema if schema is None else schema,
                                compression=compression,
                                write_batch_size=batch_rows,
                                max_rows_per_page=batch_rows,
                                use_dictionary=False,
                                write_statistics=False,
                                write_page_index=False,
                                write_page_checksum=False,
                            )
                        )
                    )
                writer.write_table(persisted, row_group_size=batch_rows)
            finally:
                persisted = None

        yield write_table
        if writer is not None:
            writer.add_key_value_metadata(metadata)
