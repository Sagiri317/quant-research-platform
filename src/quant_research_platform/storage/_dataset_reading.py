"""Dataset reading implementation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts._json import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_inventory import (
    DatasetInventoryPaths,
    DatasetSessionInventory,
)
from quant_research_platform.storage._dataset_manifest_validation import (
    manifest_has_runtime_metadata,
)
from quant_research_platform.storage._dataset_paths import (
    format_year_partition_label,
    parse_year_partition_label,
    relative_partition_label,
    require_artifact_root,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    DatasetPartMetadata,
    inspect_dataset_part,
    iter_dataset_part_batches,
)
from quant_research_platform.storage._dataset_schema import (
    logical_dtype_metadata_bytes,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    validate_dataset_manifest_consumable,
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
)
from quant_research_platform.storage._dataset_types import (
    DatasetReadMode,
)
from quant_research_platform.storage.manifests import read_bounded_qrp_json


@dataclass(frozen=True, slots=True)
class DatasetSession:
    root: Path
    data_root: Path
    spec_bytes: bytes
    manifest_bytes: bytes
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
        if self.owns_inventory:
            self.inventory.close()

    def __enter__(self) -> DatasetSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def verify(self) -> None:
        spec_bytes, _ = read_bounded_qrp_json(self.root / "spec.json")
        if spec_bytes != self.spec_bytes:
            raise ValueError("dataset_spec_identity_changed")
        manifest_bytes, _ = read_bounded_qrp_json(
            self.root / "_manifest" / "manifest.json"
        )
        if manifest_bytes != self.manifest_bytes:
            raise ValueError("dataset_manifest_identity_changed")
        self.inventory.verify_filesystem(self.spec)


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
    resource_policy: RuntimeResourcePolicy | None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    if mode is DatasetReadMode.CHUNKED:

        def activate() -> Iterator[pd.DataFrame]:
            with open_dataset_session(
                dataset_root,
                columns=columns,
                time_window=time_window,
                resource_policy=resource_policy,
            ) as context:
                yield from iter_dataset_chunks(
                    context,
                    target_chunk_bytes=context.resource_policy.payload_target_bytes,
                )

        return activate()
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    with open_dataset_session(
        dataset_root,
        columns=columns,
        time_window=time_window,
        resource_policy=resource_policy,
    ) as context:
        return materialize_dataset_session(
            context,
            target_chunk_bytes=context.resource_policy.payload_target_bytes,
            materialized_budget_bytes=context.resource_policy.task_memory_budget_bytes,
        )


def materialize_dataset_session(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    materialized_budget_bytes: int | None = None,
) -> pd.DataFrame:
    """Materialize one validated context through the unique physical iterator."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    materialized_budget = _validated_target_chunk_bytes(
        context.resource_policy.task_memory_budget_bytes
        if materialized_budget_bytes is None
        else materialized_budget_bytes
    )
    materialized_budget -= context.inventory.retained_memory_bytes
    if materialized_budget <= 0:
        raise ValueError("dataset_materialized_result_exceeds_target_bytes")
    dtype_plan = _materialized_dtype_plan(context)
    scan_target_bytes = min(target_bytes, materialized_budget)
    row_count, string_bytes, object_bytes = _scan_materialized_shape(
        context,
        target_chunk_bytes=scan_target_bytes,
        operation_headroom_bytes=materialized_budget,
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
    if required_payload_bytes > materialized_budget:
        raise ValueError("dataset_materialized_result_exceeds_target_bytes")
    remaining_headroom = materialized_budget - required_payload_bytes
    if remaining_headroom <= 0:
        raise ValueError("dataset_materialized_result_exceeds_target_bytes")
    second_pass_target_bytes = min(target_bytes, remaining_headroom)
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
        operation_headroom_bytes=remaining_headroom,
    )
    return _materialized_frame(buffers)


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
    try:
        for frame in chunks:
            row_count += len(frame)
            for name in string_columns:
                string_bytes[name] += _arrow_string_payload_bytes(frame[name])
            if object_columns:
                object_bytes += _object_payload_bytes(frame, columns=object_columns)
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
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
    try:
        for frame in chunks:
            stop = cursor + len(frame)
            if stop > row_count:
                raise ValueError("dataset_materialized_row_count_changed")
            for name, buffer in buffers.items():
                _fill_materialized_column(buffer, frame[name], start=cursor)
            cursor = stop
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
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


def open_dataset_session(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
    require_consumable: bool = True,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> DatasetSession:
    root = require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    spec_bytes, spec_payload = read_bounded_qrp_json(spec_path)
    if spec_payload.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    validate_dataset_spec_payload(spec_payload)
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    manifest_bytes, manifest = read_bounded_qrp_json(manifest_path)
    if manifest.get("schema_id") != "qrp-dataset-manifest":
        raise ValueError("dataset_manifest_schema_id_invalid")
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    if require_consumable:
        validate_dataset_manifest_consumable(manifest)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    issued_policy = resource_policy or active_runtime_resource_policy()
    inventory = DatasetSessionInventory(
        data_root=data_root,
        scratch_parent=root.parent,
        task_memory_budget_bytes=issued_policy.task_memory_budget_bytes,
    )
    try:
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
        return _build_dataset_session(
            root=root,
            data_root=data_root,
            spec_bytes=spec_bytes,
            manifest_bytes=manifest_bytes,
            inventory=inventory,
            columns=columns,
            time_window=time_window,
            row_filters=row_filters,
            require_consumable=require_consumable,
            resource_policy=issued_policy,
            owns_inventory=True,
        )
    except BaseException:
        inventory.close()
        raise


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
    spec_bytes, spec_payload = read_bounded_qrp_json(root / "spec.json")
    manifest_bytes, _ = read_bounded_qrp_json(root / "_manifest" / "manifest.json")
    inventory.verify_filesystem(spec_payload)
    return _build_dataset_session(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        manifest_bytes=manifest_bytes,
        inventory=inventory,
        columns=columns,
        time_window=time_window,
        row_filters=(),
        require_consumable=True,
        resource_policy=resource_policy,
        owns_inventory=False,
    )


def _build_dataset_session(
    *,
    root: Path,
    data_root: Path,
    spec_bytes: bytes,
    manifest_bytes: bytes,
    inventory: DatasetSessionInventory,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...],
    require_consumable: bool,
    resource_policy: RuntimeResourcePolicy,
    owns_inventory: bool,
) -> DatasetSession:
    spec_payload = parse_qrp_json_object(spec_bytes)
    if spec_payload.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    validate_dataset_spec_payload(spec_payload)
    manifest = parse_qrp_json_object(manifest_bytes)
    if manifest.get("schema_id") != "qrp-dataset-manifest":
        raise ValueError("dataset_manifest_schema_id_invalid")
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    if require_consumable:
        validate_dataset_manifest_consumable(manifest)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
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
    physical_filters = (
        parquet_filters_for_time_window(
            timestamp_column=timestamp_column,
            time_window=time_window,
        )
        or []
    )
    for column, operator, value in row_filters:
        if column not in schema_columns:
            raise ValueError("dataset_filter_column_not_found")
        if operator != "==":
            raise ValueError("unsupported_dataset_physical_filter")
        physical_filters.append(
            (
                column,
                operator,
                _validated_physical_filter_value(
                    first_metadata.schema.field(column),
                    value,
                ),
            )
        )
    physical_columns = tuple(
        dict.fromkeys(
            (*physical_columns, *(column for column, _, _ in physical_filters))
        )
    )
    session = DatasetSession(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        manifest_bytes=manifest_bytes,
        inventory=inventory,
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        time_window=time_window,
        schema_hash=schema_hash,
        arrow_schema=first_metadata.schema,
        logical_dtype_bytes=logical_dtype_metadata_bytes(logical_dtypes),
        parquet_filters=tuple(
            (column, operator, _freeze_filter_value(value))
            for column, operator, value in physical_filters
        )
        or None,
        resource_policy=resource_policy,
        owns_inventory=owns_inventory,
    )
    session.verify()
    return session


def iter_dataset_chunks(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    try:
        if not context.read_paths:
            yield _empty_frame(
                context,
                context.logical_columns,
                target_chunk_bytes=target_bytes,
            )
            return
        yielded = False
        for _, frame in iter_dataset_file_batches(
            context,
            columns=context.physical_columns,
            target_chunk_bytes=target_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            batch_observer=batch_observer,
        ):
            filtered = frame.loc[:, list(context.logical_columns)]
            for chunk in _bounded_frame_chunks(
                filtered,
                target_chunk_bytes=target_bytes,
            ):
                yielded = True
                yield chunk
        if not yielded:
            yield _empty_frame(
                context,
                context.logical_columns,
                target_chunk_bytes=target_bytes,
            )
    finally:
        context.verify()


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
) -> Iterator[pd.DataFrame]:
    """Traverse one frozen Parquet member through the sole Pandas converter."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    previous_output_bytes = (
        [0] if _previous_output_bytes is None else _previous_output_bytes
    )
    for batch in iter_dataset_part_batches(
        Path(path),
        expected_identity=expected_identity,
        target_chunk_bytes=target_bytes,
        columns=columns,
        parquet_filters=parquet_filters,
        metadata_observer=metadata_observer,
    ):
        yield from _arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=target_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            dtype_overrides=dtype_overrides,
            persisted_frame_observer=persisted_frame_observer,
            batch_observer=batch_observer,
            _previous_output_bytes=previous_output_bytes,
        )


def iter_dataset_file_batches(
    context: DatasetSession,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    operation_headroom_bytes: int | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    if operation_headroom_bytes is None:
        operation_headroom = (
            context.resource_policy.task_memory_budget_bytes
            - context.inventory.retained_memory_bytes
        )
        if operation_headroom <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    else:
        operation_headroom = _validated_target_chunk_bytes(operation_headroom_bytes)
    dtype_overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes,
        columns=columns,
    )
    context.verify()
    previous_output_bytes = [0]
    try:
        for path in context.read_paths:
            metadata = context.metadata_for(path)
            for frame in iter_declared_parquet_member_batches(
                path,
                expected_identity=metadata.identity,
                columns=columns,
                parquet_filters=context.parquet_filters,
                target_chunk_bytes=target_bytes,
                operation_headroom_bytes=operation_headroom,
                dtype_overrides=dtype_overrides,
                batch_observer=batch_observer,
                _previous_output_bytes=previous_output_bytes,
            ):
                yield path, frame
    finally:
        context.verify()


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
    elif pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        valid = type(frozen) is bytes
    elif pa.types.is_timestamp(data_type):
        valid = type(frozen) in (pd.Timestamp, datetime)
    elif pa.types.is_date(data_type):
        valid = type(frozen) is date
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
    *,
    target_chunk_bytes: int,
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
        )
    )
    output = frame.astype(overrides) if overrides else frame
    if (
        pandas_frame_payload_bytes(frame)
        + (pandas_frame_payload_bytes(output) if output is not frame else 0)
        > target_chunk_bytes
    ):
        raise ValueError("dataset_row_exceeds_target_chunk_bytes")
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
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    pandas_budget = _validated_target_chunk_bytes(target_chunk_bytes)
    overrides = dict(dtype_overrides or {})
    conversion_budget = (
        None
        if operation_headroom_bytes is None
        else _validated_target_chunk_bytes(operation_headroom_bytes)
        - batch.get_total_buffer_size()
    )
    if conversion_budget is not None and conversion_budget <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    rows_per_slice = 1
    calibrated = False
    start = 0
    previous_output_bytes = (
        [0] if _previous_output_bytes is None else _previous_output_bytes
    )
    while start < batch.num_rows:
        current_pandas_budget = (
            pandas_budget
            if conversion_budget is None
            else conversion_budget - previous_output_bytes[0]
        )
        if current_pandas_budget <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        slice_budget = min(pandas_budget, current_pandas_budget)
        stop = min(batch.num_rows, start + rows_per_slice)
        arrow_slice = batch.slice(start, stop - start)
        frame = _arrow_slice_to_pandas(arrow_slice)
        frame_bytes = pandas_frame_payload_bytes(frame)
        if frame_bytes > pandas_budget or (
            conversion_budget is not None and frame_bytes > current_pandas_budget
        ):
            if len(frame) == 1:
                if frame_bytes > pandas_budget:
                    raise ValueError("dataset_row_exceeds_target_chunk_bytes")
                raise ValueError("dataset_working_set_exceeds_task_memory_budget")
            rows_per_slice = max(1, rows_per_slice // 2)
            del frame
            continue
        categorical_overrides = {
            name: dtype
            for name, value in overrides.items()
            if isinstance(
                dtype := pd.api.types.pandas_dtype(value),
                pd.CategoricalDtype,
            )
        }
        if overrides:
            ordinary_overrides = {
                name: value
                for name, value in overrides.items()
                if name not in categorical_overrides
            }
            output = (
                frame.astype(ordinary_overrides) if ordinary_overrides else frame.copy()
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
        if output_bytes > pandas_budget or (
            output is not frame
            and frame_bytes + output_bytes > current_pandas_budget
        ):
            if len(frame) == 1:
                if output_bytes > pandas_budget:
                    raise ValueError("dataset_row_exceeds_target_chunk_bytes")
                raise ValueError("dataset_working_set_exceeds_task_memory_budget")
            rows_per_slice = max(1, rows_per_slice // 2)
            del output, frame
            continue
        measured_bytes = max(frame_bytes, output_bytes)
        measured_row_bytes = max(
            1,
            (measured_bytes + len(frame) - 1) // len(frame),
        )
        if not calibrated:
            calibrated = True
            calibrated_rows = max(
                1,
                min(
                    batch.num_rows - start,
                    slice_budget // measured_row_bytes,
                ),
            )
            if calibrated_rows > rows_per_slice:
                del output, frame
                rows_per_slice = calibrated_rows
                continue
        if persisted_frame_observer is not None:
            persisted_frame_observer(frame)
        if batch_observer is not None:
            batch_observer(arrow_slice, output)
        del frame
        yield output
        previous_output_bytes[0] = output_bytes
        del output
        start = stop
        if start >= batch.num_rows:
            continue
        next_slice_budget = (
            pandas_budget
            if conversion_budget is None
            else conversion_budget - previous_output_bytes[0]
        )
        if next_slice_budget <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        rows_per_slice = max(
            1,
            min(
                batch.num_rows - start,
                min(pandas_budget, next_slice_budget) // measured_row_bytes,
            ),
        )


def _arrow_slice_to_pandas(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
    return arrow_slice.to_pandas(use_threads=False)


def _bounded_frame_chunks(
    frame: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    if frame.empty:
        return
    frame_bytes = pandas_frame_payload_bytes(frame)
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
        candidate_bytes = pandas_frame_payload_bytes(candidate)
        if candidate_bytes > target_chunk_bytes:
            if len(candidate) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_chunk = max(1, rows_per_chunk // 2)
            continue
        yield candidate
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
