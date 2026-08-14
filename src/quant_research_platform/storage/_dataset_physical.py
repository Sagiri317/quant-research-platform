"""The sole handle-bound Parquet traversal for dataset storage."""

from __future__ import annotations

import math
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quant_research_platform.contracts._json import (
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
)
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded

_DECODED_ROW_BOUNDS_KEY = b"qrp.decoded-row-bounds"
_DECODED_ROW_BOUNDS_SCHEMA_ID = "qrp-dataset-part-decoded-row-bounds"
_PHYSICAL_PAGE_ROWS_KEY = b"qrp.parquet-page-rows"
_PARQUET_NATIVE_METADATA_PAGE_BYTES = 4096


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
            if _operation_headroom is None
            else _operation_headroom()
        )
        initial_physical_headroom = (
            None
            if initial_operation_headroom is None
            else initial_operation_headroom - previous_output_bytes[0]
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
        admitted_batch_rows = _target_batch_rows(
            parquet_file,
            columns=selected_columns,
            row_groups=selected_row_groups,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=initial_physical_headroom,
            filtered=bool(parquet_filters),
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
        maximum_decode_fixed_bytes = _maximum_projected_decode_fixed_bytes(
            parquet_file,
            selected_columns,
        )
        maximum_page_decode_bytes = (
            maximum_decode_fixed_bytes
            + _maximum_projected_page_decode_bytes(
                parquet_file,
                selected_columns,
                row_groups=selected_row_groups,
            )
        )
        raw_batches = iter(
            parquet_file.iter_batches(
                batch_size=batch_rows,
                row_groups=selected_row_groups,
                columns=list(selected_columns),
                use_threads=False,
            )
        )
        footer_decoded_bounds = (parquet_file.metadata.metadata or {}).get(
            _DECODED_ROW_BOUNDS_KEY
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
                if parquet_filters
                else max(
                    maximum_page_decode_bytes,
                    maximum_decode_fixed_bytes + source_batch_bytes,
                )
            )
            current_operation_headroom = (
                operation_headroom_bytes
                if _operation_headroom is None
                else _operation_headroom()
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
            with ExitStack() as pull_reservation:
                policy = current_runtime_resource_policy()
                if policy is not None:
                    pull_reservation.enter_context(
                        policy.reserve(
                            "storage.parquet.read",
                            physical_pull_bytes,
                        )
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
                if footer_decoded_bounds is not None:
                    batch_metadata = dict(filtered.schema.metadata or {})
                    batch_metadata[_DECODED_ROW_BOUNDS_KEY] = footer_decoded_bounds
                    filtered = filtered.replace_schema_metadata(batch_metadata)
                yield filtered
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
    row_groups: list[int] | tuple[int, ...] | None = None,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    filtered: bool = False,
    membership: bool = False,
    membership_max_scalar_bytes: int = 0,
) -> int:
    maximum_row_bytes = _maximum_projected_row_bytes(parquet_file, columns)
    selected_row_groups = (
        list(range(parquet_file.metadata.num_row_groups))
        if row_groups is None
        else list(row_groups)
    )
    fixed_decode_bytes = _maximum_projected_decode_fixed_bytes(
        parquet_file,
        columns,
    )
    page_decode_bytes = fixed_decode_bytes + _maximum_projected_page_decode_bytes(
        parquet_file,
        columns,
        row_groups=selected_row_groups,
    )
    if _decoded_row_bounds(parquet_file) is None:
        legacy_rows = max(
            (
                int(parquet_file.metadata.row_group(index).num_rows)
                for index in selected_row_groups
            ),
            default=1,
        )
        if operation_headroom_bytes is None:
            return max(1, legacy_rows)
        legacy_required = page_decode_bytes
        if filtered:
            legacy_required += max(
                0,
                _filter_working_set_bytes(
                    source_batch_bytes=maximum_row_bytes,
                    row_count=1,
                    membership=membership,
                    membership_max_scalar_bytes=membership_max_scalar_bytes,
                )
                - maximum_row_bytes,
            )
        if legacy_required > operation_headroom_bytes:
            raise DatasetReadCapacityExceeded(
                "dataset_working_set_exceeds_task_memory_budget"
            )
        return 1
    target_rows = max(1, target_chunk_bytes // maximum_row_bytes)
    if operation_headroom_bytes is None:
        return target_rows
    low = 1
    high = target_rows
    admitted = 0
    while low <= high:
        candidate = (low + high) // 2
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
                logical_required_bytes
                - fixed_decode_bytes
                - source_batch_bytes,
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
        )
    return admitted


def _maximum_projected_row_bytes(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> int:
    decoded_bounds = _decoded_row_bounds(parquet_file)
    if decoded_bounds is not None:
        try:
            return max(
                1,
                sum(decoded_bounds[column][0] for column in columns),
            )
        except KeyError:
            raise DatasetReadCapacityExceeded(
                "dataset_decode_bound_metadata_invalid"
            ) from None
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


def _maximum_projected_decode_fixed_bytes(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> int:
    decoded_bounds = _decoded_row_bounds(parquet_file)
    if decoded_bounds is None:
        return 0
    try:
        return sum(decoded_bounds[column][1] for column in columns)
    except KeyError:
        raise DatasetReadCapacityExceeded(
            "dataset_decode_bound_metadata_invalid"
        ) from None


def _maximum_projected_page_decode_bytes(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
    *,
    row_groups: list[int] | tuple[int, ...],
) -> int:
    positions = _projected_parquet_leaf_positions(parquet_file, columns)
    page_rows = _physical_page_rows(parquet_file)
    maximum_row_bytes = _maximum_projected_row_bytes(parquet_file, columns)
    maximum = 0
    for row_group_index in row_groups:
        row_group = parquet_file.metadata.row_group(row_group_index)
        rows = int(row_group.num_rows)
        if rows <= 0:
            continue
        compressed_bytes = sum(
            max(0, int(row_group.column(index).total_compressed_size))
            for index in positions
        )
        decoded_bytes = (
            sum(
                max(0, int(row_group.column(index).total_uncompressed_size))
                for index in positions
            )
            if page_rows is None
            else min(rows, page_rows) * maximum_row_bytes
        )
        maximum = max(maximum, compressed_bytes + decoded_bytes)
    return maximum


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


def table_decoded_working_set_bounds(table: pa.Table) -> tuple[int, int]:
    decoded_bounds = _parse_decoded_row_bounds(
        (table.schema.metadata or {}).get(_DECODED_ROW_BOUNDS_KEY),
        names=table.column_names,
    )
    return (
        max(1, sum(bound[0] for bound in decoded_bounds.values())),
        sum(bound[1] for bound in decoded_bounds.values()),
    )


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
    return decoded_bytes + compressed_bound


def _parquet_row_group_footer_upper_bound(schema: pa.Schema) -> int:
    """Bound one row group's footer state with statistics and indexes disabled."""

    variable_bytes = sum(
        len(field.name.encode("utf-8")) + len(str(field.type).encode("utf-8"))
        for field in schema
    ) + sum(len(key) + len(value) for key, value in (schema.metadata or {}).items())
    return _PARQUET_NATIVE_METADATA_PAGE_BYTES * (len(schema) + 1) + variable_bytes


def _zstd_compressed_bound(source_bytes: int) -> int:
    """Return the codec's exact ZSTD_COMPRESSBOUND for one page payload."""

    return source_bytes + (source_bytes >> 8) + (
        ((128 << 10) - source_bytes) >> 11 if source_bytes < 128 << 10 else 0
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
    if raw is not None:
        decoded_bounds = _parse_decoded_row_bounds(
            raw,
            names=batch.schema.names,
            exact_names=False,
        )
        return (
            max(1, sum(bound[0] for bound in decoded_bounds.values())),
            sum(bound[1] for bound in decoded_bounds.values()),
        )
    bounds = tuple(
        _column_decoded_row_bound(pa.chunked_array([column]))
        for column in batch.columns
    )
    return max(1, sum(bound[0] for bound in bounds)), sum(bound[1] for bound in bounds)


def _column_decoded_row_bound(column: pa.ChunkedArray) -> tuple[int, int]:
    data_type = column.type
    if pa.types.is_dictionary(data_type):
        index_bytes = max(1, int(data_type.index_type.bit_width) // 8)
        dictionary_bytes = sum(
            int(chunk.dictionary.nbytes)
            for chunk in column.chunks
            if isinstance(chunk, pa.DictionaryArray)
        )
        return index_bytes + 1, dictionary_bytes
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
