"""The sole handle-bound Parquet traversal for dataset storage."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    DEFAULT_BATCH_ROWS,
)
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    open_filesystem_file,
    verified_file_handle,
)

_PARQUET_MAX_ROW_GROUP_ROWS = 64 * 1024 * 1024


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


def inspect_dataset_part(
    path: Path, *, expected_identity: DatasetFileIdentity | None = None
) -> DatasetPartMetadata:
    with _opened_parquet(path, expected_identity=expected_identity) as opened:
        parquet_file, identity, _handle = opened
        return _part_metadata(parquet_file, identity=identity)


def iter_dataset_part_batches(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    row_groups: tuple[int, ...] | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream native Arrow batches while binding metadata and bytes to one handle."""
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("invalid_batch_rows")
    with _opened_parquet(path, expected_identity=expected_identity) as opened:
        parquet_file, identity, _ = opened
        metadata = _part_metadata(
            parquet_file,
            identity=identity,
            require_logical_dtypes=False,
        )
        if metadata_observer is not None:
            metadata_observer(metadata)
        selected_columns = columns or tuple(metadata.schema.names)
        try:
            schema = pa.schema(
                [metadata.schema.field(column) for column in selected_columns]
            )
        except KeyError as exc:
            raise ValueError("dataset_column_not_found") from exc
        groups = (
            _matching_row_groups(parquet_file, parquet_filters)
            if row_groups is None
            else list(row_groups)
        )
        for group in groups:
            rows = batch_rows
            with closing_iterator(
                parquet_file.iter_batches(
                    batch_size=rows,
                    row_groups=[group],
                    columns=list(selected_columns),
                    use_threads=True,
                )
            ) as batches:
                for batch in batches:
                    if batch.schema.remove_metadata() != schema:
                        batch = batch.cast(schema)
                    filtered = _filter_arrow_batch(batch, parquet_filters)
                    try:
                        yield filtered
                    except GeneratorExit:
                        return


@contextmanager
def _opened_parquet(path: Path, *, expected_identity: DatasetFileIdentity | None):
    resolved = Path(path)
    identity = dataset_file_identity(resolved)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("dataset_file_identity_changed")
    with closing_iterator(open_filesystem_file(resolved)) as handle:
        with verified_file_handle(resolved, handle, expected_identity=identity):
            footer_length = _parquet_footer_length(handle)
            with closing_iterator(
                pq.ParquetFile(
                    handle,
                    thrift_string_size_limit=max(1, footer_length),
                    thrift_container_size_limit=max(1, footer_length),
                    pre_buffer=False,
                )
            ) as parquet:
                yield (parquet, identity, handle)


def _parquet_footer_length(handle: Any) -> int:
    position = handle.tell()
    handle.seek(0, os.SEEK_END)
    file_size = handle.tell()
    if file_size < 8:
        raise ValueError("dataset_parquet_footer_invalid")
    handle.seek(-8, os.SEEK_END)
    trailer = handle.read(8)
    handle.seek(position)
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
        logical_dtype_payload=b""
        if logical_dtype_payload is None
        else bytes(logical_dtype_payload),
    )


def _parquet_row_group_size(rows: int) -> int:
    if isinstance(rows, bool) or type(rows) is not int or rows < 0:
        raise ValueError("invalid_parquet_row_group_rows")
    return max(1, min(rows, _PARQUET_MAX_ROW_GROUP_ROWS))


def _matching_row_groups(
    parquet_file: pq.ParquetFile, filters: list[tuple[str, str, Any]] | None
) -> list[int]:
    indexes = list(range(parquet_file.metadata.num_row_groups))
    if not filters:
        return indexes
    positions = _top_level_parquet_leaf_positions(parquet_file, filters)
    return [
        row_group_index
        for row_group_index in indexes
        if all(
            (
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
        )
    ]


def _top_level_parquet_leaf_positions(
    parquet_file: pq.ParquetFile, filters: list[tuple[str, str, Any]]
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


def _row_group_may_match(statistics: Any, *, operator: str, value: Any) -> bool:
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
    batch: pa.RecordBatch, filters: list[tuple[str, str, Any]] | None
) -> pa.RecordBatch:
    operations = {
        ">=": pc.greater_equal,
        ">": pc.greater,
        "<=": pc.less_equal,
        "<": pc.less,
        "==": pc.equal,
    }
    combined = None
    for column, operator, value in filters or ():
        if operator == "in":
            if not isinstance(value, PreparedMembershipValues):
                raise ValueError("invalid_dataset_physical_filter_value")
        elif operator not in operations:
            raise ValueError("unsupported_dataset_physical_filter")
        if operator == "in":
            column_values = batch.column(batch.schema.get_field_index(column))
            condition = pc.is_in(column_values, value_set=value.values, skip_nulls=True)
            if (
                pa.types.is_floating(column_values.type)
                and pc.any(pc.equal(value.values, 0.0)).as_py()
            ):
                condition = pc.or_(condition, pc.equal(column_values, 0.0))
        else:
            column_values = batch.column(batch.schema.get_field_index(column))
            scalar = value
            if pa.types.is_timestamp(column_values.type) or pa.types.is_duration(
                column_values.type
            ):
                scalar = _physical_filter_scalar(batch.schema.field(column), value)
                if operator == "==":
                    try:
                        scalar = scalar.cast(column_values.type, safe=True)
                    except pa.ArrowInvalid:
                        return batch.slice(0, 0)
            condition = operations[operator](column_values, scalar)
            del column_values, scalar
        combined = condition if combined is None else pc.and_kleene(combined, condition)
        del condition
    return batch if combined is None else pc.filter(batch, combined)


def _physical_filter_scalar(field: pa.Field, value: Any) -> pa.Scalar:
    source_type = None
    if pa.types.is_timestamp(field.type):
        source_type = pa.timestamp(
            value.unit if isinstance(value, pd.Timestamp) else "us", tz=field.type.tz
        )
    elif pa.types.is_duration(field.type):
        source_type = pa.duration(
            value.unit if isinstance(value, pd.Timedelta) else "us"
        )
    return pa.scalar(value, type=source_type)


@contextmanager
def parquet_table_writer(
    handle, *, compression: str, schema=None, statistics=(), **options
):
    """Own the native writer and preserve the primary failure during close."""
    with closing_iterator(ExitStack()) as native:

        def open_writer(writer_schema):
            return native.enter_context(
                closing_iterator(
                    pq.ParquetWriter(
                        handle,
                        writer_schema,
                        compression=compression,
                        use_dictionary=False,
                        write_statistics=list(statistics) if statistics else False,
                        write_page_index=False,
                        write_page_checksum=False,
                        **options,
                    )
                )
            )

        writer = open_writer(schema) if schema is not None else None

        def write_table(table, *, row_group_size=None):
            nonlocal writer
            if writer is None:
                writer = open_writer(table.schema)
            writer.write_table(
                table,
                row_group_size=_parquet_row_group_size(table.num_rows)
                if row_group_size is None
                else row_group_size,
            )

        yield write_table
