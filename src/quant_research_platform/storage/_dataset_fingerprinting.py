"""Dataset fingerprinting implementation."""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterator
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts._json import (
    CanonicalIdentityWriter,
    canonical_json_hash,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_paths import relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    iter_dataset_file_batches,
    open_dataset_session,
    read_columns_for_time_window,
)
from quant_research_platform.storage._dataset_schema import physical_schema_from_schema
from quant_research_platform.storage._dataset_types import DatasetFingerprint


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy | None,
) -> DatasetFingerprint:
    issued_policy = resource_policy or active_runtime_resource_policy()
    with open_dataset_session(
        dataset_root,
        columns=columns,
        time_window=time_window,
        resource_policy=issued_policy,
    ) as context:
        return fingerprint_dataset_session(context, resource_policy=issued_policy)


def fingerprint_dataset_session(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetFingerprint:
    payload_target_bytes = resource_policy.payload_target_bytes
    operation_headroom_bytes = (
        resource_policy.task_memory_budget_bytes
        - context.inventory.retained_memory_bytes
    )
    if operation_headroom_bytes <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    key_columns = tuple(str(column) for column in context.spec["primary_key"])
    hash_columns = tuple(dict.fromkeys((*key_columns, *context.logical_columns)))
    hash_physical_columns = read_columns_for_time_window(
        hash_columns,
        spec=context.spec,
        time_window=context.time_window,
    )
    output_columns = tuple(context.logical_columns)
    projected_schema_hash = canonical_json_hash(
        physical_schema_from_schema(
            pa.schema([context.arrow_schema.field(column) for column in hash_columns])
        )
    )
    row_count = 0
    row_hash_sum = 0
    row_hash_xor = 0
    partition_rows: dict[str, int] = {}
    partition_hash_sum: dict[str, int] = {}
    partition_hash_xor: dict[str, int] = {}

    file_count = 0
    last_observed_path: Path | None = None
    accepted_arrow_bytes: int | None = None
    accepted_arrow_schema: pa.Schema | None = None

    def observe_batch(batch: Any, _frame: pd.DataFrame) -> None:
        nonlocal accepted_arrow_bytes, accepted_arrow_schema
        if accepted_arrow_bytes is not None:
            raise RuntimeError("dataset_fingerprint_batch_observer_out_of_sequence")
        accepted_arrow_bytes = batch.get_total_buffer_size()
        accepted_arrow_schema = batch.schema

    for path, chunk in iter_dataset_file_batches(
        context,
        columns=hash_physical_columns,
        target_chunk_bytes=payload_target_bytes,
        operation_headroom_bytes=operation_headroom_bytes,
        batch_observer=observe_batch,
    ):
        if accepted_arrow_bytes is None or accepted_arrow_schema is None:
            raise RuntimeError("dataset_fingerprint_batch_observer_missing")
        hash_target_bytes = operation_headroom_bytes - accepted_arrow_bytes
        accepted_arrow_bytes = None
        arrow_schema = accepted_arrow_schema
        accepted_arrow_schema = None
        chunk = chunk.loc[:, list(hash_columns)]
        if chunk.empty:
            continue
        if path != last_observed_path:
            file_count += 1
            last_observed_path = path
        chunk_sum = 0
        chunk_xor = 0
        for hashes in _iter_fingerprint_hashes(
            chunk,
            target_bytes=hash_target_bytes,
            arrow_schema=arrow_schema,
            primary_key=key_columns,
        ):
            chunk_sum = (chunk_sum + int(hashes.sum(dtype=np.uint64))) % 2**64
            chunk_xor ^= int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
            del hashes
        partition = relative_partition_label(context.data_root, path)
        row_count += int(len(chunk))
        row_hash_sum = (row_hash_sum + chunk_sum) % 2**64
        row_hash_xor ^= chunk_xor
        partition_rows[partition] = partition_rows.get(partition, 0) + len(chunk)
        partition_hash_sum[partition] = (
            partition_hash_sum.get(partition, 0) + chunk_sum
        ) % 2**64
        partition_hash_xor[partition] = partition_hash_xor.get(partition, 0) ^ chunk_xor

    content_hash = canonical_json_hash(
        {
            "row_hash_sum": row_hash_sum,
            "row_hash_xor": row_hash_xor,
            "row_count": row_count,
            "columns": list(output_columns),
            "hash_columns": list(hash_columns),
        }
    )
    partition_fingerprints = tuple(
        {
            "partition": partition,
            "row_count": int(partition_rows[partition]),
            "content_hash": canonical_json_hash(
                {
                    "row_hash_sum": int(partition_hash_sum[partition]),
                    "row_hash_xor": int(partition_hash_xor[partition]),
                    "row_count": int(partition_rows[partition]),
                    "hash_columns": list(hash_columns),
                }
            ),
        }
        for partition in sorted(partition_rows)
    )
    return DatasetFingerprint(
        schema_hash=projected_schema_hash,
        row_count=row_count,
        key_columns=key_columns,
        content_hash=content_hash,
        partition_fingerprints=partition_fingerprints,
        file_count=file_count,
        column_names=output_columns,
    )


def _iter_fingerprint_hashes(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
    arrow_schema: pa.Schema | None = None,
    primary_key: tuple[str, ...] = (),
) -> Iterator[np.ndarray[Any, np.dtype[np.uint64]]]:
    if target_bytes <= 0:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    input_payload_bytes = pandas_frame_payload_bytes(frame)
    if input_payload_bytes > target_bytes:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    available_bytes = target_bytes - input_payload_bytes
    physical_normalization = any(
        arrow_schema is not None
        and column in arrow_schema.names
        and (
            pa.types.is_map(arrow_schema.field(column).type)
            or (
                column in primary_key
                and (
                    pa.types.is_float32(arrow_schema.field(column).type)
                    or pa.types.is_float64(arrow_schema.field(column).type)
                )
            )
        )
        for column in frame.columns
    )
    full_hash_bytes = len(frame) * 8
    if not physical_normalization and full_hash_bytes <= available_bytes:
        try:
            hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy("uint64")
        except TypeError:
            pass
        else:
            yield hashes
            return

    start = 0
    rows_per_slice = 1
    calibrated = False
    while start < len(frame):
        stop = min(len(frame), start + rows_per_slice)
        current = frame.iloc[start:stop]
        hash_bytes = len(current) * 8
        if hash_bytes > available_bytes:
            if len(current) == 1:
                raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        try:
            output = _normalized_fingerprint_slice(
                current,
                arrow_schema=arrow_schema,
                primary_key=primary_key,
                identity_target_bytes=available_bytes - hash_bytes,
            )
        except ValueError as exc:
            if (
                str(exc) != "dataset_fingerprint_value_exceeds_target_bytes"
                or len(current) == 1
            ):
                raise
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        output_bytes = 0 if output is current else pandas_frame_payload_bytes(output)
        if output_bytes + hash_bytes > available_bytes:
            if len(current) == 1:
                raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        measured_row_bytes = max(
            1,
            (output_bytes + hash_bytes + len(current) - 1) // len(current),
        )
        hashes = pd.util.hash_pandas_object(output, index=False).to_numpy("uint64")
        yield hashes
        del hashes, output, current
        start = stop
        if not calibrated:
            calibrated = True
            rows_per_slice = max(1, available_bytes // measured_row_bytes)
        rows_per_slice = min(rows_per_slice, len(frame) - start)


def _normalized_fingerprint_slice(
    frame: pd.DataFrame,
    *,
    arrow_schema: pa.Schema | None,
    primary_key: tuple[str, ...],
    identity_target_bytes: int,
) -> pd.DataFrame:
    output = frame
    remaining_identity_bytes = identity_target_bytes
    for column in frame.columns:
        data_type = (
            None
            if arrow_schema is None or column not in arrow_schema.names
            else arrow_schema.field(column).type
        )
        requires_physical_normalization = data_type is not None and pa.types.is_map(
            data_type
        )
        requires_float_key_normalization = (
            data_type is not None
            and column in primary_key
            and (pa.types.is_float32(data_type) or pa.types.is_float64(data_type))
        )
        if requires_float_key_normalization:
            zero_mask = frame[column].eq(0)
            if bool(zero_mask.any()):
                if output is frame:
                    output = frame.copy()
                output[column] = frame[column].mask(zero_mask, 0.0)
            continue
        if not requires_physical_normalization:
            try:
                pd.util.hash_pandas_object(frame[column], index=False)
            except TypeError:
                pass
            else:
                continue
        if output is frame:
            output = frame.copy()

        def canonicalize(
            value: Any,
            physical_type: pa.DataType | None = data_type,
        ) -> bytes:
            nonlocal remaining_identity_bytes
            payload = _canonical_nested_value(
                value,
                max_bytes=remaining_identity_bytes,
                data_type=physical_type,
            )
            remaining_identity_bytes -= len(payload)
            return payload

        output[column] = frame[column].map(canonicalize)
    return output


def _canonical_nested_value(
    value: Any,
    *,
    max_bytes: int | None = None,
    data_type: pa.DataType | None = None,
) -> bytes:
    writer = CanonicalIdentityWriter(max_bytes=max_bytes)
    try:
        _write_nested_identity(
            writer,
            value,
            active=set(),
            data_type=data_type,
        )
        return writer.finish()
    except RecursionError as exc:
        raise ValueError("dataset_fingerprint_value_too_deep") from exc
    except ValueError as exc:
        if str(exc) == "canonical_identity_exceeds_max_bytes":
            raise ValueError("dataset_fingerprint_value_exceeds_target_bytes") from exc
        if str(exc) == "canonical_json_recursion_limit":
            raise ValueError("dataset_fingerprint_value_too_deep") from exc
        raise


def _write_nested_identity(
    writer: CanonicalIdentityWriter,
    value: Any,
    *,
    active: set[int],
    data_type: pa.DataType | None = None,
) -> None:
    if data_type is not None and pa.types.is_map(data_type):
        _write_arrow_map_identity(
            writer,
            value,
            data_type=data_type,
            active=active,
        )
    elif type(value) is np.ndarray:
        _with_nested_container(
            value,
            active=active,
            build=lambda: _write_ndarray_identity(writer, value, active=active),
        )
    elif isinstance(value, np.generic):
        with writer.tuple_items(2):
            writer.value("numpy-scalar")
            with writer.tuple_items(2):
                writer.value(value.dtype.str)
                _write_nested_identity(writer, value.item(), active=active)
    elif type(value) in (list, tuple):
        tag = "list" if type(value) is list else "tuple"
        _with_nested_container(
            value,
            active=active,
            build=lambda: _write_sequence_identity(writer, tag, value, active=active),
        )
    elif type(value) is dict:
        _with_nested_container(
            value,
            active=active,
            build=lambda: _write_mapping_identity(writer, value, active=active),
        )
    elif type(value) is bytes:
        _write_tagged_value(writer, "bytes", value)
    elif type(value) is pd.Timestamp:
        _write_tagged_value(writer, "timestamp", value)
    elif type(value) is datetime:
        _write_tagged_value(writer, "datetime", value)
    elif type(value) is date:
        _write_tagged_value(writer, "date", value)
    elif type(value) is Decimal:
        _write_tagged_value(writer, "decimal", value)
    elif type(value) is float:
        _write_tagged_value(writer, "float64-bits", struct.pack(">d", value))
    elif value is None:
        _write_tagged_value(writer, "none", None)
    elif type(value) is bool:
        _write_tagged_value(writer, "bool", value)
    elif type(value) is int:
        _write_tagged_value(writer, "int", value)
    elif type(value) is str:
        _write_tagged_value(writer, "string", value)
    else:
        raise ValueError("unsupported_dataset_fingerprint_value")


def _write_tagged_value(
    writer: CanonicalIdentityWriter, tag: str, value: object
) -> None:
    with writer.tuple_items(2):
        writer.value(tag)
        writer.value(value)


def _write_sequence_identity(
    writer: CanonicalIdentityWriter,
    tag: str,
    value: list[Any] | tuple[Any, ...],
    *,
    active: set[int],
) -> None:
    with writer.tuple_items(2):
        writer.value(tag)
        with writer.tuple_items(len(value)):
            for item in value:
                _write_nested_identity(writer, item, active=active)


def _write_mapping_identity(
    writer: CanonicalIdentityWriter,
    value: dict[Any, Any],
    *,
    active: set[int],
) -> None:
    def emit_key(target: CanonicalIdentityWriter, key: object) -> None:
        _write_nested_identity(target, key, active=active)

    with writer.tuple_items(2):
        writer.value("dict")
        with writer.tuple_items(len(value)):
            with writer.canonical_order(
                len(value),
                emit_key,
                collision_error="dataset_fingerprint_mapping_key_collision",
                mapping=value,
            ) as (keys, order):
                for index in order:
                    with writer.tuple_items(2):
                        _write_nested_identity(writer, keys[index], active=active)
                        _write_nested_identity(
                            writer,
                            value[keys[index]],
                            active=active,
                        )


def _write_ndarray_identity(
    writer: CanonicalIdentityWriter,
    value: np.ndarray,
    *,
    active: set[int],
) -> None:
    with writer.tuple_items(2):
        writer.value("ndarray")
        with writer.tuple_items(3):
            writer.value(value.dtype.str)
            writer.value(tuple(int(item) for item in value.shape))
            _write_ndarray_values_identity(writer, value, active=active)


def _write_ndarray_values_identity(
    writer: CanonicalIdentityWriter,
    value: np.ndarray,
    *,
    active: set[int],
    dimension: int = 0,
    index: tuple[int, ...] = (),
) -> None:
    if dimension == value.ndim:
        item = value[index]
        if isinstance(item, np.generic):
            item = item.item()
        _write_nested_identity(writer, item, active=active)
        return
    with writer.tuple_items(2):
        writer.value("list")
        with writer.tuple_items(value.shape[dimension]):
            for offset in range(value.shape[dimension]):
                _write_ndarray_values_identity(
                    writer,
                    value,
                    active=active,
                    dimension=dimension + 1,
                    index=(*index, offset),
                )


def _write_arrow_map_identity(
    writer: CanonicalIdentityWriter,
    value: Any,
    *,
    data_type: pa.DataType,
    active: set[int],
) -> None:
    if value is None:
        with writer.tuple_items(2):
            writer.value("arrow-map")
            with writer.tuple_items(2):
                writer.value(str(data_type))
                writer.value(None)
        return
    if type(value) not in (list, tuple):
        raise ValueError("unsupported_dataset_fingerprint_value")
    for entry in value:
        if type(entry) not in (list, tuple) or len(entry) != 2:
            raise ValueError("unsupported_dataset_fingerprint_value")

    def build() -> None:
        def emit_entry(target: CanonicalIdentityWriter, index: int) -> None:
            with target.tuple_items(2):
                _write_nested_identity(
                    target,
                    value[index][0],
                    active=active,
                    data_type=data_type.key_type,
                )
                _write_nested_identity(
                    target,
                    value[index][1],
                    active=active,
                    data_type=data_type.item_type,
                )

        with writer.tuple_items(2):
            writer.value("arrow-map")
            with writer.tuple_items(2):
                writer.value(str(data_type))
                with writer.tuple_items(len(value)):
                    with writer.canonical_order(len(value), emit_entry) as order:
                        for index in order:
                            emit_entry(writer, index)

    _with_nested_container(value, active=active, build=build)


def _with_nested_container(
    value: Any, *, active: set[int], build: Callable[[], None]
) -> None:
    identity = id(value)
    if identity in active:
        raise ValueError("dataset_fingerprint_value_cycle")
    active.add(identity)
    try:
        build()
    finally:
        active.remove(identity)
