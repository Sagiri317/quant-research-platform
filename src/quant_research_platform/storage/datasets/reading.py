"""Dataset reading implementation."""

from __future__ import annotations

import math
import struct
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.inventory import (
    DatasetInventoryPaths,
    DatasetSessionInventory,
)
from quant_research_platform.storage.datasets.paths import (
    format_year_partition_label,
    parse_year_partition_label,
    relative_partition_label,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage.datasets.schema import (
    _utf8_payload_length,
    logical_dtype_metadata_bytes,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    validate_dataset_manifest_consumable,
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
    validate_persisted_logical_schema,
)
from quant_research_platform.storage.datasets.types import (
    DatasetReadMode,
)
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    _file_identity_bytes,
    dataset_file_identity,
    filesystem_is_dir,
    filesystem_is_file,
    require_artifact_root,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_bounded_qrp_json
from quant_research_platform.storage.parquet import (
    DatasetPartMetadata,
    PreparedMembershipValues,
    inspect_dataset_part,
    iter_dataset_part_batches,
    record_batch_decoded_working_set_bounds,
)

_PANDAS_FRAME_FIXED_BYTES = 4 * 1024
_PANDAS_OBJECT_CELL_BYTES = 64
_PANDAS_INDEX_BYTES_PER_ROW = 8


@dataclass(frozen=True, slots=True)
class DatasetSession:
    root: Path
    data_root: Path
    spec_bytes: bytes
    spec_identity_bytes: bytes
    manifest_bytes: bytes
    manifest_identity_bytes: bytes
    inventory: DatasetSessionInventory
    logical_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    timestamp_column: str | None
    time_window: TimeWindow | None
    schema_hash: str
    arrow_schema: pa.Schema
    logical_dtype_bytes: bytes
    parquet_filters: tuple[tuple[str, str, Any], ...] | None
    resource_policy: RuntimeResourcePolicy
    owns_inventory: bool
    _lease: MemoryLease
    plan_base_retained_bytes: int
    owns_lease: bool = True
    prepared_filter_bytes: int = 0
    path_selection: int | None = None

    @property
    def all_paths(self) -> DatasetInventoryPaths:
        return DatasetInventoryPaths(self.inventory, selected=False)

    @property
    def read_paths(self) -> DatasetInventoryPaths:
        return DatasetInventoryPaths(
            self.inventory,
            selected=True,
            path_selection=self.path_selection,
        )

    @property
    def spec(self) -> dict[str, Any]:
        return parse_qrp_json_object(self.spec_bytes)

    @property
    def manifest(self) -> dict[str, Any]:
        return parse_qrp_json_object(self.manifest_bytes)

    @property
    def schema(self) -> list[dict[str, Any]]:
        return list(self.manifest["physical_schema"])

    @property
    def logical_dtypes(self) -> dict[str, Any]:
        return parse_logical_dtype_metadata(self.logical_dtype_bytes)

    def metadata_for(self, path: Path) -> DatasetPartMetadata:
        return self.inventory.metadata_for(path)

    def close(self) -> None:
        primary = sys.exception()
        first_error: BaseException | None = None
        if self.owns_inventory:
            try:
                self.inventory.close()
            except BaseException as inventory_error:
                if primary is not None:
                    primary.add_note(
                        "dataset session inventory release also failed: "
                        f"{type(inventory_error).__name__}"
                    )
                else:
                    first_error = inventory_error
        if self.owns_lease:
            try:
                self._lease.close()
            except BaseException as lease_error:
                if primary is not None:
                    primary.add_note(
                        "dataset session lease release also failed: "
                        f"{type(lease_error).__name__}"
                    )
                elif first_error is None:
                    first_error = lease_error
                else:
                    first_error.add_note(
                        "dataset session lease release also failed: "
                        f"{type(lease_error).__name__}"
                    )
        if first_error is not None:
            raise first_error

    def __enter__(self) -> DatasetSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is None:
            self.close()
            return
        try:
            self.close()
        except BaseException as cleanup_error:
            exc_value.add_note(
                f"dataset session cleanup also failed: {type(cleanup_error).__name__}"
            )

    def verify(self) -> None:
        spec_identity = dataset_file_identity(self.root / "spec.json")
        if _file_identity_bytes(spec_identity) != self.spec_identity_bytes:
            raise ValueError("dataset_spec_identity_changed")
        spec_bytes, _ = read_bounded_qrp_json(
            self.root / "spec.json", expected_identity=spec_identity
        )
        if spec_bytes != self.spec_bytes:
            raise ValueError("dataset_spec_identity_changed")
        manifest_identity = dataset_file_identity(
            self.root / "_manifest" / "manifest.json"
        )
        if _file_identity_bytes(manifest_identity) != self.manifest_identity_bytes:
            raise ValueError("dataset_manifest_identity_changed")
        manifest_bytes, _ = read_bounded_qrp_json(
            self.root / "_manifest" / "manifest.json",
            expected_identity=manifest_identity,
        )
        if manifest_bytes != self.manifest_bytes:
            raise ValueError("dataset_manifest_identity_changed")
        self.inventory.verify_filesystem(self.spec)

    @contextmanager
    def filtered_view(self, time_window: TimeWindow) -> Iterator[DatasetSession]:
        if current_runtime_resource_policy() is not self.resource_policy:
            raise ValueError("dataset_session_resource_policy_mismatch")
        parquet_filters = (
            *(self.parquet_filters or ()),
            *(
                parquet_filters_for_time_window(
                    timestamp_column=self.timestamp_column,
                    time_window=time_window,
                )
                or ()
            ),
        )
        selection: int | None = None
        spec = self.spec
        if tuple(spec.get("partition_columns", ())) == ("year",):
            selection = self.inventory.new_path_selection()
            try:
                for path in self.read_paths:
                    if prune_paths_for_time_window(
                        self.data_root,
                        (path,),
                        spec=spec,
                        time_window=time_window,
                    ):
                        self.inventory.select_path(selection, path)
            except BaseException as primary:
                try:
                    self.inventory.release_path_selection(selection)
                except BaseException as cleanup_error:
                    primary.add_note(
                        "dataset filtered-view path cleanup also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                raise
        view = replace(
            self,
            path_selection=selection if selection is not None else self.path_selection,
            time_window=time_window,
            parquet_filters=parquet_filters,
            physical_columns=tuple(
                dict.fromkeys(
                    (
                        *self.physical_columns,
                        *read_columns_for_time_window(
                            self.logical_columns, spec=spec, time_window=time_window
                        ),
                    )
                )
            ),
            owns_inventory=False,
            owns_lease=False,
        )
        try:
            try:
                yield view
            finally:
                _verify_preserving_primary(view)
        finally:
            primary = sys.exception()
            try:
                if selection is not None:
                    self.inventory.release_path_selection(selection)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "dataset filtered-view path cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
    resource_policy: RuntimeResourcePolicy | None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    if mode is DatasetReadMode.CHUNKED:
        captured_policy = current_runtime_resource_policy()
        if resource_policy is None and captured_policy is not None:
            raise RuntimeError("runtime_resource_policy_competing_root")
        bound_policy = resource_policy
        if (
            bound_policy is not None
            and captured_policy is not None
            and bound_policy is not captured_policy
        ):
            raise RuntimeError("runtime_resource_policy_conflict")
        if bound_policy is None:
            bound_policy = captured_policy
        independent_root = resource_policy is None and captured_policy is None

        def activate() -> Iterator[pd.DataFrame]:
            if independent_root and current_runtime_resource_policy() is not None:
                raise RuntimeError("runtime_resource_policy_competing_root")
            if captured_policy is not None:
                with runtime_resource_policy_scope(bound_policy):
                    with open_dataset_session(
                        dataset_root,
                        columns=columns,
                        time_window=time_window,
                        resource_policy=bound_policy,
                    ) as context:
                        yield from iter_dataset_chunks(
                            context,
                            target_chunk_bytes=max(
                                1,
                                context.resource_policy.planning_headroom_bytes,
                            ),
                        )
                return
            with open_dataset_session(
                dataset_root,
                columns=columns,
                time_window=time_window,
                resource_policy=bound_policy,
            ) as context:
                yield from iter_dataset_chunks(
                    context,
                    target_chunk_bytes=max(
                        1,
                        context.resource_policy.planning_headroom_bytes,
                    ),
                )

        return activate()
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    if resource_policy is None and current_runtime_resource_policy() is not None:
        raise RuntimeError("runtime_resource_policy_competing_root")
    with open_dataset_session(
        dataset_root,
        columns=columns,
        time_window=time_window,
        resource_policy=resource_policy,
    ) as context:
        return materialize_dataset_session(
            context,
            target_chunk_bytes=max(
                1,
                context.resource_policy.planning_headroom_bytes,
            ),
        )


def materialize_dataset_session(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    materialized_budget_bytes: int | None = None,
) -> pd.DataFrame:
    """Materialize one validated context under its captured policy."""

    with runtime_resource_policy_scope(context.resource_policy):
        owned = materialize_dataset_session_owned(
            context,
            target_chunk_bytes=target_chunk_bytes,
            materialized_budget_bytes=materialized_budget_bytes,
        )
        try:
            return owned.value
        finally:
            owned.release(owner="storage.dataset.materialized")


def materialize_dataset_session_owned(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    materialized_budget_bytes: int | None = None,
) -> OwnedValue[pd.DataFrame]:
    """Materialize one context while transferring its output lease to the caller."""

    if current_runtime_resource_policy() is not context.resource_policy:
        raise RuntimeError("runtime_resource_policy_not_active")
    return _materialize_dataset_session(
        context,
        target_chunk_bytes=target_chunk_bytes,
        materialized_budget_bytes=materialized_budget_bytes,
    )


def _materialize_dataset_session(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    materialized_budget_bytes: int | None = None,
) -> OwnedValue[pd.DataFrame]:
    """Materialize one validated context through the unique physical iterator."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    configured_budget = _validated_target_chunk_bytes(
        max(1, context.resource_policy.working_set_headroom_bytes)
        if materialized_budget_bytes is None
        else materialized_budget_bytes
    )
    materialized_budget = min(
        configured_budget,
        context.resource_policy.working_set_headroom_bytes,
    )
    if materialized_budget <= 0:
        raise DatasetReadCapacityExceeded(
            "dataset_materialized_result_exceeds_target_bytes"
        )
    operation_headroom = materialized_budget
    dtype_plan = _materialized_dtype_plan(context)
    scan_target_bytes = min(target_bytes, materialized_budget)
    row_count, string_bytes, object_bytes = _scan_materialized_shape(
        context,
        target_chunk_bytes=scan_target_bytes,
        operation_headroom_bytes=operation_headroom,
        string_columns=tuple(
            name for name, (kind, _) in dtype_plan.items() if kind == "string"
        ),
        object_columns=tuple(
            name for name, (kind, _) in dtype_plan.items() if kind == "object"
        ),
    )
    required_payload_bytes = (
        _materialized_buffer_payload_bytes(
            dtype_plan,
            row_count=row_count,
            string_bytes=string_bytes,
        )
        + object_bytes
    )
    materialized_budget = min(
        configured_budget, context.resource_policy.working_set_headroom_bytes
    )
    if required_payload_bytes > materialized_budget:
        raise DatasetReadCapacityExceeded(
            "dataset_materialized_result_exceeds_target_bytes"
        )
    remaining_headroom = materialized_budget - required_payload_bytes
    if remaining_headroom <= 0:
        raise DatasetReadCapacityExceeded(
            "dataset_materialized_result_exceeds_target_bytes"
        )
    second_pass_target_bytes = min(target_bytes, remaining_headroom)
    output_lease = context.resource_policy.lease(
        "storage.materialized.output",
        required_payload_bytes,
    )
    try:
        buffers = _allocate_materialized_buffers(
            dtype_plan,
            row_count=row_count,
            string_bytes=string_bytes,
        )
        _fill_materialized_buffers(
            context,
            buffers=buffers,
            row_count=row_count,
            target_chunk_bytes=second_pass_target_bytes,
            operation_headroom_bytes=operation_headroom - required_payload_bytes,
        )
        output = _materialized_frame(buffers)
        output_bytes = pandas_frame_payload_bytes(output)
        if output_bytes > output_lease.reserved_bytes:
            raise ValueError("dataset_materialized_output_bound_invalid")
        output_lease.shrink_to(output_bytes)
    except BaseException as primary:
        buffers = output = None
        clear_exception_frames(primary)
        try:
            output_lease.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "materialized output lease release also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    return OwnedValue(output, output_lease)


@dataclass(slots=True)
class _MaterializedColumnBuffer:
    name: str
    kind: str
    dtype: object
    values: np.ndarray[Any, Any] | bytearray
    mask: np.ndarray[Any, Any] | bytearray | None = None
    offsets: np.ndarray[Any, Any] | None = None
    data_offset: int = 0
    null_count: int = 0


def _materialized_dtype_plan(
    context: DatasetSession,
) -> dict[str, tuple[str, object]]:
    payload = context.logical_dtypes
    descriptors = {item["name"]: item for item in payload["columns"]}
    overrides = pandas_dtype_overrides_from_logical_payload(
        payload,
        columns=context.logical_columns,
    )
    plan: dict[str, tuple[str, object]] = {}
    for name in context.logical_columns:
        descriptor = descriptors[name]
        kind = str(descriptor["kind"])
        dtype = pd.api.types.pandas_dtype(
            overrides[name] if name in overrides else descriptor["dtype"]
        )
        if kind == "numpy_float" and isinstance(
            dtype,
            (pd.Float32Dtype, pd.Float64Dtype),
        ):
            kind = "nullable_float"
        plan[name] = (kind, dtype)
    return plan


def _scan_materialized_shape(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int,
    string_columns: tuple[str, ...],
    object_columns: tuple[str, ...],
) -> tuple[int, dict[str, int], int]:
    row_count = 0
    string_bytes = dict.fromkeys(string_columns, 0)
    object_bytes = 0
    chunks = iter_dataset_chunks(
        context,
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=operation_headroom_bytes,
    )
    with closing_iterator(chunks):
        for frame in chunks:
            row_count += len(frame)
            for name in string_columns:
                string_bytes[name] += _arrow_string_payload_bytes(frame[name])
            if object_columns:
                object_bytes += _object_payload_bytes(frame, columns=object_columns)
    return row_count, string_bytes, object_bytes


def _materialized_buffer_payload_bytes(
    dtype_plan: Mapping[str, tuple[str, object]],
    *,
    row_count: int,
    string_bytes: Mapping[str, int],
) -> int:
    empty_index_bytes = pandas_frame_payload_bytes(pd.DataFrame())
    result = pandas_frame_payload_bytes(pd.DataFrame(index=pd.RangeIndex(row_count)))
    for name, (kind, dtype) in dtype_plan.items():
        if kind == "categorical":
            categorical = pd.Categorical.from_codes([], dtype=dtype)
            result += row_count * categorical.codes.dtype.itemsize
            result += (
                pandas_frame_payload_bytes(
                    pd.DataFrame({name: pd.Series(categorical)}, copy=False)
                )
                - empty_index_bytes
            )
        elif kind in {"nullable_float", "nullable_integer"}:
            result += row_count * (dtype.numpy_dtype.itemsize + 1)
        elif kind == "nullable_boolean":
            result += row_count * 2
        elif kind == "string":
            result += string_bytes[name] + (row_count + 1) * 8
            result += (row_count + 7) // 8
        elif isinstance(dtype, pd.DatetimeTZDtype):
            result += row_count * 8
        elif kind == "object":
            result += row_count * _PANDAS_OBJECT_CELL_BYTES
        else:
            result += row_count * np.dtype(dtype).itemsize
    return result


def _allocate_materialized_buffers(
    dtype_plan: Mapping[str, tuple[str, object]],
    *,
    row_count: int,
    string_bytes: Mapping[str, int],
) -> dict[str, _MaterializedColumnBuffer]:
    buffers: dict[str, _MaterializedColumnBuffer] = {}
    for name, (kind, dtype) in dtype_plan.items():
        if kind == "categorical":
            code_dtype = pd.Categorical.from_codes([], dtype=dtype).codes.dtype
            values: np.ndarray[Any, Any] | bytearray = np.empty(
                row_count, dtype=code_dtype
            )
            mask: np.ndarray[Any, Any] | bytearray | None = None
            offsets = None
        elif kind in {"nullable_float", "nullable_integer"}:
            values = np.empty(row_count, dtype=dtype.numpy_dtype)
            mask = np.empty(row_count, dtype=bool)
            offsets = None
        elif kind == "nullable_boolean":
            values = np.empty(row_count, dtype=bool)
            mask = np.empty(row_count, dtype=bool)
            offsets = None
        elif kind == "string":
            values = bytearray(string_bytes[name])
            mask = bytearray((row_count + 7) // 8)
            offsets = np.empty(row_count + 1, dtype=np.int64)
            offsets[0] = 0
        elif isinstance(dtype, pd.DatetimeTZDtype):
            values = np.empty(row_count, dtype=f"datetime64[{dtype.unit}]")
            mask = None
            offsets = None
        else:
            values = np.empty(row_count, dtype=dtype)
            mask = None
            offsets = None
        buffers[name] = _MaterializedColumnBuffer(
            name=name,
            kind=kind,
            dtype=dtype,
            values=values,
            mask=mask,
            offsets=offsets,
        )
    return buffers


def _fill_materialized_buffers(
    context: DatasetSession,
    *,
    buffers: Mapping[str, _MaterializedColumnBuffer],
    row_count: int,
    target_chunk_bytes: int,
    operation_headroom_bytes: int,
) -> None:
    cursor = 0
    chunks = iter_dataset_chunks(
        context,
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=operation_headroom_bytes,
    )
    with closing_iterator(chunks):
        for frame in chunks:
            stop = cursor + len(frame)
            if stop > row_count:
                raise ValueError("dataset_materialized_row_count_changed")
            for name, buffer in buffers.items():
                _fill_materialized_column(buffer, frame[name], start=cursor)
            cursor = stop
    if cursor != row_count:
        raise ValueError("dataset_materialized_row_count_changed")
    for buffer in buffers.values():
        if buffer.kind == "string" and buffer.data_offset != len(buffer.values):
            raise ValueError("dataset_materialized_string_size_changed")


def _fill_materialized_column(
    buffer: _MaterializedColumnBuffer,
    series: pd.Series,
    *,
    start: int,
) -> None:
    stop = start + len(series)
    if buffer.kind in {"nullable_float", "nullable_integer", "nullable_boolean"}:
        assert isinstance(buffer.values, np.ndarray)
        assert isinstance(buffer.mask, np.ndarray)
        buffer.mask[start:stop] = series.isna().to_numpy(copy=False)
        buffer.values[start:stop] = series.to_numpy(
            dtype=buffer.values.dtype,
            na_value=0,
            copy=False,
        )
        return
    if buffer.kind == "string":
        _copy_arrow_strings(buffer, series, start=start)
        return
    if buffer.kind == "categorical":
        assert isinstance(buffer.values, np.ndarray)
        buffer.values[start:stop] = series.cat.codes.to_numpy(copy=False)
        return
    assert isinstance(buffer.values, np.ndarray)
    if isinstance(buffer.dtype, pd.DatetimeTZDtype):
        buffer.values.view(np.int64)[start:stop] = series.array.asi8
    else:
        buffer.values[start:stop] = series.to_numpy(copy=False)


def _object_payload_bytes(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> int:
    view = pd.DataFrame({name: frame[name] for name in columns}, copy=False)
    pointer_bytes = len(frame) * len(columns) * np.dtype(object).itemsize
    index_bytes = pandas_frame_payload_bytes(pd.DataFrame(index=frame.index))
    return max(0, pandas_frame_payload_bytes(view) - pointer_bytes - index_bytes)


def _arrow_string_payload_bytes(series: pd.Series) -> int:
    return sum(
        _arrow_string_buffer_view(array)[1] for array in _arrow_string_arrays(series)
    )


def _arrow_string_arrays(series: pd.Series) -> tuple[pa.Array, ...]:
    arrow = series.array.__arrow_array__()
    arrays = tuple(arrow.chunks) if isinstance(arrow, pa.ChunkedArray) else (arrow,)
    if any(
        not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type))
        for array in arrays
    ):
        raise ValueError("dataset_logical_dtype_mismatch")
    return arrays


def _arrow_string_buffer_view(
    array: pa.Array,
) -> tuple[np.ndarray[Any, Any], int]:
    _, offsets_buffer, _ = array.buffers()
    offset_dtype = np.int64 if pa.types.is_large_string(array.type) else np.int32
    offsets = np.frombuffer(
        offsets_buffer,
        dtype=offset_dtype,
        count=len(array) + 1,
        offset=array.offset * np.dtype(offset_dtype).itemsize,
    )
    return offsets, int(offsets[-1] - offsets[0])


def _copy_arrow_strings(
    buffer: _MaterializedColumnBuffer,
    series: pd.Series,
    *,
    start: int,
) -> None:
    assert isinstance(buffer.values, bytearray)
    assert isinstance(buffer.mask, bytearray)
    assert buffer.offsets is not None
    row_cursor = start
    for array in _arrow_string_arrays(series):
        offsets, payload_bytes = _arrow_string_buffer_view(array)
        _, _, data_buffer = array.buffers()
        source_start = int(offsets[0])
        source_stop = int(offsets[-1])
        if buffer.data_offset + payload_bytes > len(buffer.values):
            raise ValueError("dataset_materialized_string_size_changed")
        if payload_bytes:
            assert data_buffer is not None
            buffer.values[buffer.data_offset : buffer.data_offset + payload_bytes] = (
                memoryview(data_buffer)[source_start:source_stop]
            )
        buffer.offsets[row_cursor : row_cursor + len(array) + 1] = (
            offsets - source_start + buffer.data_offset
        )
        valid = array.is_valid().to_numpy(zero_copy_only=False)
        buffer.null_count += len(array) - int(np.count_nonzero(valid))
        packed = np.packbits(valid, bitorder="little")
        destination = np.frombuffer(buffer.mask, dtype=np.uint8)
        byte_offset, bit_offset = divmod(row_cursor, 8)
        first_count = min(len(packed), len(destination) - byte_offset)
        destination[byte_offset : byte_offset + first_count] |= (
            packed[:first_count] << bit_offset
        )
        if bit_offset:
            byte_offset += 1
            second_count = min(len(packed), len(destination) - byte_offset)
            destination[byte_offset : byte_offset + second_count] |= packed[
                :second_count
            ] >> (8 - bit_offset)
        row_cursor += len(array)
        buffer.data_offset += payload_bytes
        buffer.offsets[row_cursor] = buffer.data_offset


def _materialized_frame(
    buffers: Mapping[str, _MaterializedColumnBuffer],
) -> pd.DataFrame:
    columns: dict[str, object] = {}
    for name, buffer in buffers.items():
        if buffer.kind == "nullable_float":
            assert isinstance(buffer.values, np.ndarray)
            assert isinstance(buffer.mask, np.ndarray)
            columns[name] = pd.arrays.FloatingArray(
                buffer.values,
                buffer.mask,
                copy=False,
            )
        elif buffer.kind == "nullable_integer":
            assert isinstance(buffer.values, np.ndarray)
            assert isinstance(buffer.mask, np.ndarray)
            columns[name] = pd.arrays.IntegerArray(
                buffer.values,
                buffer.mask,
                copy=False,
            )
        elif buffer.kind == "nullable_boolean":
            assert isinstance(buffer.values, np.ndarray)
            assert isinstance(buffer.mask, np.ndarray)
            columns[name] = pd.arrays.BooleanArray(
                buffer.values,
                buffer.mask,
                copy=False,
            )
        elif buffer.kind == "string":
            assert isinstance(buffer.values, bytearray)
            assert isinstance(buffer.mask, bytearray)
            assert buffer.offsets is not None
            array = pa.Array.from_buffers(
                pa.large_string(),
                len(buffer.offsets) - 1,
                [
                    pa.py_buffer(buffer.mask),
                    pa.py_buffer(buffer.offsets),
                    pa.py_buffer(buffer.values),
                ],
                null_count=buffer.null_count,
            )
            columns[name] = pd.array(array, dtype=buffer.dtype, copy=False)
        elif buffer.kind == "categorical":
            assert isinstance(buffer.values, np.ndarray)
            columns[name] = pd.Categorical.from_codes(
                buffer.values,
                dtype=buffer.dtype,
                validate=False,
            )
        elif isinstance(buffer.dtype, pd.DatetimeTZDtype):
            columns[name] = pd.array(buffer.values, dtype=buffer.dtype, copy=False)
        elif buffer.kind == "object":
            columns[name] = pd.Series(buffer.values, dtype=object, copy=False)
        else:
            columns[name] = buffer.values
    return pd.DataFrame(columns, copy=False)


@contextmanager
def open_dataset_session(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
    require_consumable: bool = True,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[DatasetSession]:
    root = require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(data_root, error="invalid_dataset_part_files")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    if not filesystem_is_file(spec_path):
        raise ValueError("dataset_spec_missing")
    if not filesystem_is_dir(data_root):
        raise ValueError("dataset_data_dir_missing")
    if not filesystem_is_file(manifest_path):
        raise ValueError("dataset_manifest_missing")
    try:
        spec_identity = dataset_file_identity(spec_path)
        manifest_identity = dataset_file_identity(manifest_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    with runtime_resource_policy_scope(resource_policy) as issued_policy:
        spec_stat_span_bytes = spec_identity.size
        manifest_stat_span_bytes = manifest_identity.size
        spec_identity_bytes = _file_identity_bytes(spec_identity)
        manifest_identity_bytes = _file_identity_bytes(manifest_identity)
        metadata_identity_span_bytes = len(spec_identity_bytes) + len(
            manifest_identity_bytes
        )
        # Keep one contract view plus decode/validation projections admitted.
        metadata_bound = (
            2 * json_record_working_set_bytes(spec_stat_span_bytes)
            + 2 * json_record_working_set_bytes(manifest_stat_span_bytes)
            + metadata_identity_span_bytes
        )
        cap = issued_policy.explicit_memory_cap_bytes
        if cap is None or issued_policy.used_memory_bytes + metadata_bound <= cap:
            issued_policy.require_working_set(
                metadata_bound,
                error="dataset_working_set_exceeds_task_memory_budget",
            )
        session_lease = issued_policy.lease(
            "storage.dataset.session",
            metadata_bound,
        )
        inventory: DatasetSessionInventory | None = None
        session: DatasetSession | None = None
        try:
            spec_bytes, spec_payload = read_bounded_qrp_json(
                spec_path, expected_identity=spec_identity
            )
            if len(spec_bytes) != spec_stat_span_bytes:
                raise ValueError("dataset_metadata_file_identity_changed")
            if spec_payload.get("schema_id") != "qrp-dataset-spec":
                raise ValueError("dataset_spec_schema_id_invalid")
            validate_dataset_spec_payload(spec_payload)
            manifest_bytes, manifest = read_bounded_qrp_json(
                manifest_path, expected_identity=manifest_identity
            )
            if len(manifest_bytes) != manifest_stat_span_bytes:
                raise ValueError("dataset_metadata_file_identity_changed")
            if manifest.get("schema_id") != "qrp-dataset-manifest":
                raise ValueError("dataset_manifest_schema_id_invalid")
            validate_dataset_manifest_payload(manifest, spec=spec_payload)
            if require_consumable:
                validate_dataset_manifest_consumable(manifest)
            inventory = DatasetSessionInventory(
                data_root=data_root,
                resource_policy=issued_policy,
            )
            inventory.discover(spec_payload)
            if inventory.path_count(selected=False) == 0:
                raise ValueError("dataset_data_dir_missing")
            for path in inventory.iter_candidate_paths():
                selected = path in prune_paths_for_time_window(
                    data_root, (path,), spec=spec_payload, time_window=time_window
                )
                inventory.observe(
                    path,
                    inspect_dataset_part(path),
                    selected=selected,
                )
                session_lease.shrink_to(metadata_bound)
            if (
                inventory.content_hash(resource_policy=issued_policy)
                != manifest["content_hash"]
            ):
                raise ValueError("dataset_content_identity_changed")
            if session_lease.reserved_bytes < metadata_bound:
                session_lease.grow(metadata_bound - session_lease.reserved_bytes)
            session = _build_dataset_session(
                root=root,
                data_root=data_root,
                spec_bytes=spec_bytes,
                spec_identity_bytes=spec_identity_bytes,
                spec_payload=spec_payload,
                manifest_bytes=manifest_bytes,
                manifest_identity_bytes=manifest_identity_bytes,
                manifest=manifest,
                inventory=inventory,
                columns=columns,
                time_window=time_window,
                row_filters=row_filters,
                resource_policy=issued_policy,
                owns_inventory=True,
                session_lease=session_lease,
            )
        except BaseException as primary:
            cleanup_ok = True
            if session is not None:
                try:
                    session.close()
                except BaseException as cleanup_error:
                    cleanup_ok = False
                    primary.add_note(
                        "dataset session construction cleanup also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            elif inventory is not None:
                try:
                    inventory.close()
                except BaseException as cleanup_error:
                    cleanup_ok = False
                    primary.add_note(
                        "dataset inventory construction cleanup also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            if cleanup_ok:
                try:
                    session_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "dataset session lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
        assert session is not None
        del spec_payload, manifest
        with session:
            yield session


def dataset_session_from_inventory(
    dataset_root: Path,
    *,
    inventory: DatasetSessionInventory,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetSession:
    """Freeze a writer receipt without reopening already observed Parquet parts."""

    root = require_artifact_root(Path(dataset_root))
    data_root = root / "data"
    spec_path = root / "spec.json"
    manifest_path = root / "_manifest" / "manifest.json"
    require_unaliased_path(data_root, error="invalid_dataset_part_files")
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    try:
        spec_identity = dataset_file_identity(spec_path)
        manifest_identity = dataset_file_identity(manifest_path)
        spec_stat_span_bytes = spec_identity.size
        manifest_stat_span_bytes = manifest_identity.size
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    spec_identity_bytes = _file_identity_bytes(spec_identity)
    manifest_identity_bytes = _file_identity_bytes(manifest_identity)
    metadata_identity_span_bytes = len(spec_identity_bytes) + len(
        manifest_identity_bytes
    )
    metadata_bound = (
        2 * json_record_working_set_bytes(spec_stat_span_bytes)
        + 2 * json_record_working_set_bytes(manifest_stat_span_bytes)
        + metadata_identity_span_bytes
    )
    cap = resource_policy.explicit_memory_cap_bytes
    if cap is None or resource_policy.used_memory_bytes + metadata_bound <= cap:
        resource_policy.require_working_set(
            metadata_bound,
            error="dataset_working_set_exceeds_task_memory_budget",
        )
    session_lease = resource_policy.lease(
        "storage.dataset.session",
        metadata_bound,
    )
    session: DatasetSession | None = None
    try:
        spec_bytes, spec_payload = read_bounded_qrp_json(
            spec_path, expected_identity=spec_identity
        )
        if len(spec_bytes) != spec_stat_span_bytes:
            raise ValueError("dataset_metadata_file_identity_changed")
        manifest_bytes, manifest = read_bounded_qrp_json(
            manifest_path, expected_identity=manifest_identity
        )
        if len(manifest_bytes) != manifest_stat_span_bytes:
            raise ValueError("dataset_metadata_file_identity_changed")
        validate_dataset_manifest_payload(manifest, spec=spec_payload)
        validate_dataset_manifest_consumable(manifest)
        inventory.verify_filesystem(spec_payload)
        session = _build_dataset_session(
            root=root,
            data_root=data_root,
            spec_bytes=spec_bytes,
            spec_identity_bytes=spec_identity_bytes,
            spec_payload=spec_payload,
            manifest_bytes=manifest_bytes,
            manifest_identity_bytes=manifest_identity_bytes,
            manifest=manifest,
            inventory=inventory,
            columns=columns,
            time_window=time_window,
            row_filters=(),
            resource_policy=resource_policy,
            owns_inventory=False,
            session_lease=session_lease,
        )
    except BaseException as primary:
        if session is not None:
            try:
                session.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "dataset session construction cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        else:
            try:
                session_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "dataset session lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    assert session is not None
    return session


def _build_dataset_session(
    *,
    root: Path,
    data_root: Path,
    spec_bytes: bytes,
    spec_identity_bytes: bytes,
    spec_payload: dict[str, Any],
    manifest_bytes: bytes,
    manifest_identity_bytes: bytes,
    manifest: dict[str, Any],
    inventory: DatasetSessionInventory,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...],
    resource_policy: RuntimeResourcePolicy,
    owns_inventory: bool,
    session_lease: MemoryLease,
) -> DatasetSession:
    if inventory.path_count(selected=False) == 0:
        raise ValueError("dataset_data_dir_missing")
    first_path = inventory.path_at(0, selected=False)
    first_metadata = inventory.metadata_for(first_path)
    schema = physical_schema_from_schema(first_metadata.schema)
    logical_dtypes: dict[str, Any] | None = None
    observed_rows = 0
    observed_file_bytes = 0
    observed_partition_rows: dict[str, int] = {}
    for path, metadata in inventory.iter_entries():
        if physical_schema_from_schema(metadata.schema) != schema:
            raise ValueError("dataset_schema_mismatch")
        current_dtypes = parse_logical_dtype_metadata(metadata.logical_dtype_payload)
        validate_persisted_logical_schema(metadata.schema, current_dtypes)
        logical_dtypes = (
            current_dtypes
            if logical_dtypes is None
            else merge_logical_dtype_payloads([logical_dtypes, current_dtypes])
        )
        observed_rows += metadata.row_count
        observed_file_bytes += metadata.file_size_bytes
        partition = relative_partition_label(data_root, path)
        observed_partition_rows[partition] = (
            observed_partition_rows.get(partition, 0) + metadata.row_count
        )
        if (
            tuple(spec_payload["partition_columns"]) == ("year",)
            and path.parent == data_root
            and metadata.row_count != 0
        ):
            raise ValueError("invalid_dataset_part_files")
    assert logical_dtypes is not None
    schema_columns = physical_schema_columns(schema)
    schema_hash = canonical_json_hash(schema)
    if int(manifest["file_count"]) != inventory.path_count(selected=False):
        raise ValueError("manifest_file_count_mismatch")
    if set(manifest["partition_row_counts"]) != set(observed_partition_rows):
        raise ValueError("manifest_partition_row_counts_mismatch")
    if int(manifest["row_count"]) != observed_rows:
        raise ValueError("manifest_row_count_mismatch")
    if dict(sorted(observed_partition_rows.items())) != dict(
        sorted(manifest["partition_row_counts"].items())
    ):
        raise ValueError("manifest_partition_row_counts_mismatch")
    if int(manifest["file_size_bytes"]) != observed_file_bytes:
        raise ValueError("manifest_file_size_bytes_mismatch")
    if manifest["logical_dtypes"] != logical_dtypes:
        raise ValueError("manifest_logical_dtypes_mismatch")
    logical_dtype_bytes = logical_dtype_metadata_bytes(logical_dtypes)
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
    if manifest["physical_schema"] != schema:
        raise ValueError("manifest_physical_schema_mismatch")
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
    physical_columns = tuple(
        dict.fromkeys((*physical_columns, *(column for column, _, _ in row_filters)))
    )
    plan_base_retained_bytes = _dataset_session_plan_base_retained_memory_bytes(
        root=root,
        data_root=data_root,
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        schema_hash=schema_hash,
        arrow_schema_span_bytes=inventory.schema_payload_bytes_for(first_path),
    )
    physical_filters = (
        parquet_filters_for_time_window(
            timestamp_column=timestamp_column,
            time_window=time_window,
        )
        or []
    )
    prepared_filter_bytes = 0
    for column, operator, value in row_filters:
        if column not in schema_columns:
            raise ValueError("dataset_filter_column_not_found")
        if operator not in {"==", "in"}:
            raise ValueError("unsupported_dataset_physical_filter")
        field = first_metadata.schema.field(column)
        if operator == "in":
            if type(value) is not tuple or not value:
                raise ValueError("invalid_dataset_physical_filter_value")
            validated_value = _prepare_in_filter(
                field,
                value,
                resource_policy=resource_policy,
                session_lease=session_lease,
            )
            prepared_filter_bytes += validated_value.retained_memory_bytes
        else:
            validated_value = _validated_physical_filter_value(field, value)
        physical_filters.append(
            (
                column,
                operator,
                validated_value,
            )
        )
    session = DatasetSession(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        spec_identity_bytes=spec_identity_bytes,
        manifest_bytes=manifest_bytes,
        manifest_identity_bytes=manifest_identity_bytes,
        inventory=inventory,
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        time_window=time_window,
        schema_hash=schema_hash,
        arrow_schema=first_metadata.schema,
        logical_dtype_bytes=logical_dtype_bytes,
        parquet_filters=tuple(
            (column, operator, _freeze_filter_value(value))
            for column, operator, value in physical_filters
        )
        or None,
        resource_policy=resource_policy,
        owns_inventory=owns_inventory,
        _lease=session_lease,
        plan_base_retained_bytes=plan_base_retained_bytes,
        prepared_filter_bytes=prepared_filter_bytes,
    )
    session_live_bytes = _dataset_session_retained_memory_bytes(
        inventory_retained_bytes=0,
        spec_bytes=spec_bytes,
        spec_identity_bytes=spec_identity_bytes,
        manifest_bytes=manifest_bytes,
        manifest_identity_bytes=manifest_identity_bytes,
        logical_dtype_bytes=logical_dtype_bytes,
        plan_base_retained_bytes=plan_base_retained_bytes,
        time_window=time_window,
        parquet_filters=tuple(physical_filters) or None,
        prepared_filter_bytes=prepared_filter_bytes,
    )
    if session_live_bytes > session_lease.reserved_bytes:
        session_lease.grow(session_live_bytes - session_lease.reserved_bytes)
    else:
        session_lease.shrink_to(session_live_bytes)
    session.verify()
    return session


def _dataset_session_retained_memory_bytes(
    *,
    inventory_retained_bytes: int,
    spec_bytes: bytes,
    spec_identity_bytes: bytes,
    manifest_bytes: bytes,
    manifest_identity_bytes: bytes,
    logical_dtype_bytes: bytes,
    plan_base_retained_bytes: int = 0,
    time_window: TimeWindow | None = None,
    parquet_filters: tuple[tuple[str, str, Any], ...] | None = None,
    prepared_filter_bytes: int = 0,
) -> int:
    return (
        inventory_retained_bytes
        + 2 * json_record_working_set_bytes(len(spec_bytes))
        + len(spec_identity_bytes)
        + 2 * json_record_working_set_bytes(len(manifest_bytes))
        + len(manifest_identity_bytes)
        + json_record_working_set_bytes(len(logical_dtype_bytes))
        + plan_base_retained_bytes
        + _dataset_session_view_retained_memory_bytes(
            time_window=time_window,
            parquet_filters=parquet_filters,
        )
        + prepared_filter_bytes
    )


def _dataset_session_plan_base_retained_memory_bytes(
    *,
    root: Path,
    data_root: Path,
    logical_columns: tuple[str, ...],
    physical_columns: tuple[str, ...],
    timestamp_column: str | None,
    schema_hash: str,
    arrow_schema_span_bytes: int,
) -> int:
    pointer_bytes = struct.calcsize("P")
    strings = (
        str(root),
        str(data_root),
        *logical_columns,
        *physical_columns,
        schema_hash,
        *((timestamp_column,) if timestamp_column is not None else ()),
    )
    return (
        len(DatasetSession.__slots__) * pointer_bytes
        + (len(logical_columns) + len(physical_columns)) * pointer_bytes
        + sum(_utf8_payload_length(value) for value in strings)
        + arrow_schema_span_bytes
    )


def _dataset_session_view_retained_memory_bytes(
    *,
    time_window: TimeWindow | None,
    parquet_filters: tuple[tuple[str, str, Any], ...] | None,
) -> int:
    pointer_bytes = struct.calcsize("P")
    retained = 0
    if time_window is not None:
        retained += 2 * pointer_bytes
        retained += _utf8_payload_length(time_window.start.isoformat())
        retained += _utf8_payload_length(time_window.end.isoformat())
    for column, operator, value in parquet_filters or ():
        retained += 3 * pointer_bytes
        retained += _utf8_payload_length(column) + _utf8_payload_length(operator)
        if isinstance(value, PreparedMembershipValues):
            continue
        if type(value) is str:
            retained += _utf8_payload_length(value)
        elif type(value) is bytes:
            retained += len(value)
        elif isinstance(value, date | datetime | time):
            retained += _utf8_payload_length(value.isoformat())
        elif isinstance(value, Decimal):
            retained += _utf8_payload_length(str(value))
        else:
            retained += 8
    return retained


def dataset_partition_state_reserve_bytes(context: DatasetSession) -> int:
    """Reserve compact counters for every selected physical partition."""

    pointer_bytes = struct.calcsize("P")
    retained = 0
    previous: str | None = None
    for path in context.read_paths:
        partition = relative_partition_label(context.data_root, path)
        if partition == previous:
            continue
        previous = partition
        retained += _utf8_payload_length(partition) + 12 * pointer_bytes + 32
    return retained


def iter_dataset_chunks(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    with runtime_resource_policy_scope(context.resource_policy):
        yield from _iter_dataset_chunks(
            context,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            batch_observer=batch_observer,
        )


def _iter_dataset_chunks(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    if not context.read_paths:
        try:
            yield _empty_frame(
                context,
                context.logical_columns,
            )
        finally:
            _verify_preserving_primary(context)
        return
    yielded = False
    with closing_iterator(
        iter_dataset_file_batches(
            context,
            columns=context.physical_columns,
            target_chunk_bytes=target_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            batch_observer=batch_observer,
        )
    ) as file_batches:
        for _, frame in file_batches:
            filtered = frame.loc[:, list(context.logical_columns)]
            for chunk in iter_bounded_frame_chunks(
                filtered,
                target_chunk_bytes=target_bytes,
            ):
                yielded = True
                try:
                    yield chunk
                except GeneratorExit:
                    return
    if not yielded:
        try:
            yield _empty_frame(
                context,
                context.logical_columns,
            )
        finally:
            _verify_preserving_primary(context)


def iter_declared_parquet_member_batches(
    path: Path,
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    expected_identity: DatasetFileIdentity,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    dtype_overrides: Mapping[str, object] | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
    _previous_output_bytes: list[int] | None = None,
    _operation_headroom: Callable[[], int] | None = None,
) -> Iterator[pd.DataFrame]:
    """Traverse one frozen Parquet member through the sole Pandas converter."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    previous_output_bytes = (
        [0] if _previous_output_bytes is None else _previous_output_bytes
    )
    live_operation_headroom = _operation_headroom
    policy = current_runtime_resource_policy()
    if live_operation_headroom is None and policy is not None:
        fixed_operation_headroom = (
            operation_headroom_bytes
            if operation_headroom_bytes is not None
            else policy.working_set_headroom_bytes
        )

        def current_operation_headroom(
            active_policy: RuntimeResourcePolicy = policy,
            maximum: int = fixed_operation_headroom,
        ) -> int:
            return min(
                maximum,
                active_policy.working_set_headroom_bytes,
            )

        live_operation_headroom = current_operation_headroom
    with closing_iterator(
        iter_dataset_part_batches(
            Path(path),
            expected_identity=expected_identity,
            target_chunk_bytes=target_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            columns=columns,
            parquet_filters=parquet_filters,
            metadata_observer=metadata_observer,
            _previous_output_bytes=previous_output_bytes,
            _operation_headroom=live_operation_headroom,
        )
    ) as batches:
        for batch in batches:
            with closing_iterator(
                _arrow_batch_to_bounded_pandas(
                    batch,
                    target_chunk_bytes=target_bytes,
                    operation_headroom_bytes=operation_headroom_bytes,
                    dtype_overrides=dtype_overrides,
                    persisted_frame_observer=persisted_frame_observer,
                    batch_observer=batch_observer,
                    _previous_output_bytes=previous_output_bytes,
                    _operation_headroom=live_operation_headroom,
                )
            ) as converted:
                for frame in converted:
                    try:
                        yield frame
                    except GeneratorExit:
                        return
                    finally:
                        frame = None


def iter_dataset_file_batches(
    context: DatasetSession,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches under the session's captured policy."""

    with runtime_resource_policy_scope(context.resource_policy):
        yield from _iter_dataset_file_batches(
            context,
            columns=columns,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            batch_observer=batch_observer,
        )


def _iter_dataset_file_batches(
    context: DatasetSession,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    policy_available = context.resource_policy.working_set_headroom_bytes
    if operation_headroom_bytes is None:
        operation_headroom = policy_available
    else:
        upstream_headroom = _validated_target_chunk_bytes(operation_headroom_bytes)
        operation_headroom = min(
            policy_available,
            upstream_headroom,
        )
    if operation_headroom <= 0:
        raise DatasetReadCapacityExceeded(
            "dataset_working_set_exceeds_task_memory_budget"
        )
    dtype_overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes,
        columns=columns,
    )
    context.verify()
    previous_output_bytes = [0]

    def current_operation_headroom() -> int:
        current = min(
            operation_headroom,
            context.resource_policy.working_set_headroom_bytes,
        )
        if current <= 0:
            raise DatasetReadCapacityExceeded(
                "dataset_working_set_exceeds_task_memory_budget"
            )
        return current

    try:
        for path in context.read_paths:
            metadata = context.metadata_for(path)
            member_headroom = current_operation_headroom()
            with closing_iterator(
                iter_declared_parquet_member_batches(
                    path,
                    expected_identity=metadata.identity,
                    columns=columns,
                    parquet_filters=context.parquet_filters,
                    target_chunk_bytes=target_bytes,
                    operation_headroom_bytes=member_headroom,
                    dtype_overrides=dtype_overrides,
                    batch_observer=batch_observer,
                    _previous_output_bytes=previous_output_bytes,
                    _operation_headroom=current_operation_headroom,
                )
            ) as member_batches:
                while True:
                    try:
                        frame = next(member_batches)
                    except StopIteration:
                        break
                    try:
                        yield path, frame
                    except GeneratorExit:
                        return
    finally:
        _verify_preserving_primary(context)


def _verify_preserving_primary(context: DatasetSession) -> None:
    primary = sys.exception()
    if primary is None or isinstance(primary, GeneratorExit):
        context.verify()
        return
    try:
        context.verify()
    except BaseException as verification_error:
        primary.add_note(
            f"dataset verification also failed: {type(verification_error).__name__}"
        )


def dataset_primary_key_schema(
    context: DatasetSession,
    primary_key: tuple[str, ...],
) -> pa.Schema:
    try:
        return pa.schema([context.arrow_schema.field(column) for column in primary_key])
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc


def _validated_target_chunk_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid_target_chunk_bytes")
    return value


def _freeze_filter_value(value: Any) -> Any:
    if isinstance(value, PreparedMembershipValues):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_filter_value(item) for item in value)
    if isinstance(
        value,
        (
            str,
            bytes,
            bool,
            int,
            float,
            Decimal,
            date,
            datetime,
            time,
            timedelta,
            pd.Timestamp,
            pd.Timedelta,
        ),
    ):
        return value
    raise ValueError("invalid_dataset_physical_filter_value")


def _prepare_in_filter(
    field: pa.Field,
    values: tuple[Any, ...],
    *,
    resource_policy: RuntimeResourcePolicy,
    session_lease: MemoryLease | None = None,
) -> PreparedMembershipValues:
    data_type = field.type
    is_string = pa.types.is_string(data_type) or pa.types.is_large_string(data_type)
    is_binary = pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type)
    max_scalar_bytes = 0
    pointer_bytes = struct.calcsize("P")
    sort_state_bytes = sys.getsizeof([]) + len(values) * pointer_bytes
    timsort_scratch_bytes = (len(values) // 2) * pointer_bytes
    input_state_bytes = sys.getsizeof(values)
    if is_string or is_binary:
        offset_bytes = (
            8
            if pa.types.is_large_string(data_type)
            or pa.types.is_large_binary(data_type)
            else 4
        )
        for value in values:
            if is_string and type(value) is not str:
                raise ValueError("invalid_dataset_physical_filter_value")
            if is_binary and type(value) is not bytes:
                raise ValueError("invalid_dataset_physical_filter_value")
            if is_binary:
                encoded_bytes = len(value)
            else:
                encoded_bytes = _utf8_payload_length(value)
            input_state_bytes += encoded_bytes
            max_scalar_bytes = max(max_scalar_bytes, offset_bytes + encoded_bytes)
    else:
        supported_fixed_width = (
            pa.types.is_boolean(data_type)
            or pa.types.is_integer(data_type)
            or pa.types.is_floating(data_type)
            or pa.types.is_decimal(data_type)
            or pa.types.is_timestamp(data_type)
            or pa.types.is_date(data_type)
            or pa.types.is_time32(data_type)
            or pa.types.is_time64(data_type)
            or pa.types.is_duration(data_type)
            or pa.types.is_fixed_size_binary(data_type)
        )
        if not supported_fixed_width:
            raise ValueError("invalid_dataset_physical_filter_value")
        if pa.types.is_boolean(data_type):
            scalar_width_bytes = 0
        elif pa.types.is_fixed_size_binary(data_type):
            scalar_width_bytes = data_type.byte_width
        else:
            scalar_width_bytes = data_type.bit_width // 8
        for item in values:
            _validated_physical_filter_value(field, item)
        if pa.types.is_boolean(data_type):
            max_scalar_bytes = 1
        elif pa.types.is_fixed_size_binary(data_type):
            max_scalar_bytes = data_type.byte_width
        else:
            max_scalar_bytes = max(1, data_type.bit_width // 8)
    with runtime_resource_policy_scope(resource_policy), ExitStack() as preparation:
        _require_filter_preparation_overlap(
            preflight_bytes=0,
            input_state_bytes=input_state_bytes,
            sort_state_bytes=sort_state_bytes,
            timsort_scratch_bytes=timsort_scratch_bytes,
            resource_policy=resource_policy,
        )
        preparation.enter_context(
            resource_policy.lease(
                "storage.filter.sort",
                sort_state_bytes + timsort_scratch_bytes,
            )
        )
        sorted_values = sorted(values)
        if sorted_values:
            write_position = 1
            for read_position in range(1, len(sorted_values)):
                value = sorted_values[read_position]
                if value != sorted_values[write_position - 1]:
                    sorted_values[write_position] = value
                    write_position += 1
            del sorted_values[write_position:]
        value_count = len(sorted_values)
        validity_bytes = (value_count + 7) // 8
        if is_string or is_binary:
            arrow_bound = offset_bytes * (value_count + 1) + validity_bytes
            arrow_bound += sum(
                len(value) if is_binary else _utf8_payload_length(value)
                for value in sorted_values
            )
        elif pa.types.is_boolean(data_type):
            arrow_bound = 2 * validity_bytes
        else:
            arrow_bound = value_count * scalar_width_bytes + validity_bytes
        _require_filter_preparation_overlap(
            preflight_bytes=arrow_bound,
            input_state_bytes=input_state_bytes,
            sort_state_bytes=0,
            timsort_scratch_bytes=0,
            resource_policy=resource_policy,
        )
        with ExitStack() as arrow_state:
            if session_lease is not None:
                session_lease.grow(arrow_bound)
            else:
                arrow_state.enter_context(
                    resource_policy.lease("storage.filter.arrow", arrow_bound)
                )
            try:
                acquired = pa.array(sorted_values, type=data_type)
            except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
                raise ValueError("invalid_dataset_physical_filter_value") from None
            if acquired.nbytes > arrow_bound:
                raise DatasetReadCapacityExceeded(
                    "dataset_filter_values_exceed_task_memory_budget"
                )
        return PreparedMembershipValues(acquired, max_scalar_bytes, acquired.nbytes)


def _require_filter_preparation_overlap(
    *,
    preflight_bytes: int,
    input_state_bytes: int,
    sort_state_bytes: int,
    timsort_scratch_bytes: int,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if (
        resource_policy.explicit_memory_cap_bytes is not None
        and resource_policy.used_memory_bytes
        + input_state_bytes
        + preflight_bytes
        + sort_state_bytes
        + timsort_scratch_bytes
        > resource_policy.explicit_memory_cap_bytes
    ):
        raise DatasetReadCapacityExceeded(
            "dataset_filter_values_exceed_task_memory_budget"
        )
    # Borrowed input predates capture; only new allocations consume free capacity.
    try:
        resource_policy.require_working_set(
            preflight_bytes + sort_state_bytes + timsort_scratch_bytes,
            error="dataset_filter_values_exceed_task_memory_budget",
        )
    except ValueError:
        raise DatasetReadCapacityExceeded(
            "dataset_filter_values_exceed_task_memory_budget"
        ) from None


def _validated_physical_filter_value(field: pa.Field, value: Any) -> Any:
    frozen = _freeze_filter_value(value)
    data_type = field.type
    if frozen is pd.NaT or isinstance(frozen, np.generic):
        raise ValueError("invalid_dataset_physical_filter_value")
    if isinstance(frozen, (float, Decimal)) and not math.isfinite(frozen):
        raise ValueError("invalid_dataset_physical_filter_value")
    if pa.types.is_boolean(data_type):
        valid = type(frozen) is bool
    elif pa.types.is_integer(data_type):
        valid = type(frozen) is int
    elif pa.types.is_floating(data_type):
        valid = type(frozen) is float
    elif pa.types.is_decimal(data_type):
        valid = type(frozen) is Decimal
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        valid = type(frozen) is str
    elif (
        pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
    ):
        valid = type(frozen) is bytes
    elif pa.types.is_timestamp(data_type):
        valid = type(frozen) in (pd.Timestamp, datetime)
    elif pa.types.is_date(data_type):
        valid = type(frozen) is date
    elif pa.types.is_time32(data_type) or pa.types.is_time64(data_type):
        valid = type(frozen) is time
    elif pa.types.is_duration(data_type):
        valid = type(frozen) in (pd.Timedelta, timedelta)
    else:
        valid = False
    if not valid:
        raise ValueError("invalid_dataset_physical_filter_value")
    try:
        pa.scalar(frozen, type=data_type)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        OverflowError,
    ):
        raise ValueError("invalid_dataset_physical_filter_value") from None
    return frozen


def _empty_frame(
    context: DatasetSession,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    selected = pa.schema([context.arrow_schema.field(column) for column in columns])
    overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes,
        columns=columns,
    )
    frame = _arrow_slice_to_pandas(
        pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in selected],
            schema=selected,
        ),
        dtype_overrides=overrides,
    )
    output = frame.astype(overrides) if overrides else frame
    return output


def _arrow_batch_to_bounded_pandas(
    batch: pa.RecordBatch,
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    dtype_overrides: Mapping[str, object] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
    _previous_output_bytes: list[int] | None = None,
    _operation_headroom: Callable[[], int] | None = None,
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    pandas_target = _validated_target_chunk_bytes(target_chunk_bytes)
    overrides = dict(dtype_overrides or {})
    fixed_operation_headroom = (
        None
        if operation_headroom_bytes is None
        else _validated_target_chunk_bytes(operation_headroom_bytes)
    )
    source_row_bytes, source_fixed_bytes = record_batch_decoded_working_set_bounds(
        batch
    )
    source_row_bytes += (
        len(batch.schema.names) * _PANDAS_OBJECT_CELL_BYTES
        + _PANDAS_INDEX_BYTES_PER_ROW
    )
    categorical_overrides = {
        name: dtype
        for name, value in overrides.items()
        if isinstance(
            dtype := pd.api.types.pandas_dtype(value),
            pd.CategoricalDtype,
        )
    }
    conversion_row_bytes = source_row_bytes
    conversion_fixed_bytes = _PANDAS_FRAME_FIXED_BYTES + (
        source_fixed_bytes if _operation_headroom is None else 0
    )
    if overrides:
        conversion_row_bytes += source_row_bytes
        conversion_fixed_bytes += _PANDAS_FRAME_FIXED_BYTES + sum(
            int(dtype.categories.memory_usage(deep=True))
            for dtype in categorical_overrides.values()
        )
    conversion_row_bytes += len(categorical_overrides) * (
        np.dtype(np.intp).itemsize + 1
    )
    soft_row_bytes = max(
        1,
        (int(batch.nbytes) + batch.num_rows - 1) // batch.num_rows
        + len(batch.schema.names) * _PANDAS_OBJECT_CELL_BYTES
        + _PANDAS_INDEX_BYTES_PER_ROW,
    )
    soft_rows = max(1, pandas_target // soft_row_bytes)
    start = 0
    previous_output_bytes = (
        [0] if _previous_output_bytes is None else _previous_output_bytes
    )
    while start < batch.num_rows:
        current_operation_headroom = (
            fixed_operation_headroom
            if _operation_headroom is None
            else _operation_headroom()
        )
        previous_output_bound = previous_output_bytes[0]
        parent_buffer_bytes = (
            batch.get_total_buffer_size() if _operation_headroom is None else 0
        )
        conversion_budget = (
            None
            if current_operation_headroom is None
            else (
                current_operation_headroom
                - parent_buffer_bytes
                - previous_output_bound
                - conversion_fixed_bytes
            )
        )
        if conversion_budget is not None and conversion_budget < conversion_row_bytes:
            raise DatasetReadCapacityExceeded(
                "dataset_working_set_exceeds_task_memory_budget"
            )
        hard_rows = (
            batch.num_rows - start
            if conversion_budget is None
            else conversion_budget // conversion_row_bytes
        )
        rows_per_slice = max(
            1,
            min(batch.num_rows - start, soft_rows, hard_rows),
        )
        stop = start + rows_per_slice
        conversion_reserve_bytes = (
            conversion_fixed_bytes + (stop - start) * conversion_row_bytes
        )
        policy = current_runtime_resource_policy()
        with ExitStack() as conversion_reservation:
            if policy is not None:
                conversion_reservation.enter_context(
                    policy.lease(
                        "storage.arrow.to_pandas",
                        conversion_reserve_bytes,
                    )
                )
            arrow_slice = batch.slice(start, stop - start)
            frame = _arrow_slice_to_pandas(arrow_slice, dtype_overrides=overrides)
            frame_bytes = pandas_frame_payload_bytes(frame)
            if overrides:
                ordinary_overrides = {
                    name: value
                    for name, value in overrides.items()
                    if name not in categorical_overrides
                }
                output = (
                    frame.astype(ordinary_overrides)
                    if ordinary_overrides
                    else frame.copy()
                )
                for name, dtype in categorical_overrides.items():
                    codes = dtype.categories.get_indexer(frame[name])
                    if np.any((codes == -1) & frame[name].notna().to_numpy()):
                        raise ValueError("dataset_categorical_domain_mismatch")
                    output[name] = pd.Categorical.from_codes(
                        codes,
                        dtype=dtype,
                        validate=False,
                    )
                    del codes
            else:
                output = frame
            output_bytes = pandas_frame_payload_bytes(output)
            actual_working_bytes = frame_bytes + (
                output_bytes if output is not frame else 0
            )
            if (
                current_operation_headroom is not None
                and parent_buffer_bytes + previous_output_bound + actual_working_bytes
                > current_operation_headroom
            ):
                raise DatasetReadCapacityExceeded(
                    "dataset_working_set_exceeds_task_memory_budget"
                )
            if persisted_frame_observer is not None:
                persisted_frame_observer(frame)
            if batch_observer is not None:
                batch_observer(arrow_slice, output)
            del frame
            previous_output_bytes[0] = output_bytes
            try:
                yield output
            except GeneratorExit:
                return
            del output
        start = stop


def _arrow_slice_to_pandas(
    arrow_slice: pa.RecordBatch,
    *,
    dtype_overrides: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    native_types = set()
    for name, dtype in (dtype_overrides or {}).items():
        dtype = pd.api.types.pandas_dtype(dtype)
        data_type = arrow_slice.schema.field(name).type
        if pa.types.is_timestamp(data_type) and pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            expected = pa.Array.from_pandas(pd.Series([], dtype=dtype)).type
            if data_type != expected:
                try:
                    values = arrow_slice.column(name).cast(expected, safe=True)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    raise ValueError("dataset_logical_dtype_mismatch") from None
                arrow_slice = arrow_slice.set_column(
                    arrow_slice.schema.get_field_index(name),
                    name,
                    values,
                )
        if pa.types.is_integer(data_type) or pa.types.is_boolean(data_type):
            native_types.add(data_type)
        if (
            isinstance(dtype, np.dtype)
            and dtype.kind in "biu"
            and arrow_slice.column(name).null_count
        ):
            raise ValueError("dataset_nonnullable_value_missing")
    return arrow_slice.to_pandas(
        use_threads=False,
        types_mapper=(
            lambda data_type: (
                pd.ArrowDtype(data_type) if data_type in native_types else None
            )
        )
        if native_types
        else None,
    )


def iter_bounded_frame_chunks(
    frame: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    if frame.empty:
        return
    target = _validated_target_chunk_bytes(target_chunk_bytes)
    frame_bytes = pandas_frame_payload_bytes(frame)
    rows_per_chunk = max(
        1,
        min(
            len(frame),
            len(frame) * target // max(1, frame_bytes),
        ),
    )
    for start in range(0, len(frame), rows_per_chunk):
        stop = min(len(frame), start + rows_per_chunk)
        yield frame.iloc[start:stop]


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
        format_year_partition_label(year)
        for year in range(int(start.year), int(end.year) + 1)
    }
    return tuple(
        path
        for path in paths
        if path.parent == Path(data_root)
        or format_year_partition_label(
            parse_year_partition_label(relative_partition_label(data_root, path))
        )
        in years
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


def resolve_columns_from_schema(
    schema_columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        requested = tuple(spec["required_columns"])
    else:
        if type(columns) not in (tuple, list):
            raise ValueError("dataset_columns_invalid")
        if not columns:
            raise ValueError("dataset_columns_empty")
        if any(type(column) is not str or not column for column in columns) or len(
            columns
        ) != len(set(columns)):
            raise ValueError("dataset_columns_invalid")
        requested = tuple(columns)
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
