"""Dataset fingerprinting implementation."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    _OwnedValue,
    _release_owned_value,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.paths import relative_partition_label
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_file_batches,
    open_dataset_session,
    read_columns_for_time_window,
)
from quant_research_platform.storage.datasets.schema import physical_schema_from_schema
from quant_research_platform.storage.datasets.types import DatasetFingerprint


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy | None,
) -> DatasetFingerprint:
    with open_dataset_session(
        dataset_root,
        columns=columns,
        time_window=time_window,
        resource_policy=resource_policy,
    ) as context:
        return fingerprint_dataset_session(
            context,
            resource_policy=context.resource_policy,
        )


def fingerprint_dataset_session(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetFingerprint:
    if context.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        owned = _fingerprint_dataset_session_owned(
            context,
            resource_policy=resource_policy,
        )
        result = owned.value
        _release_owned_value(owned, owner="storage.dataset.fingerprint.result")
        return result


def _fingerprint_dataset_session_owned(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> _OwnedValue[DatasetFingerprint]:
    try:
        result_lease = resource_policy.lease(
            "storage.dataset.fingerprint.result",
            _fingerprint_operation_bound(context),
        )
    except MemoryError:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget") from None
    try:
        return _fingerprint_dataset_session_owned_with_lease(
            context,
            resource_policy=resource_policy,
            result_lease=result_lease,
        )
    except BaseException:
        _close_lease_preserving_primary(
            result_lease,
            owner="storage.dataset.fingerprint.result",
        )
        raise


def _fingerprint_dataset_session_owned_with_lease(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
    result_lease: MemoryLease,
) -> _OwnedValue[DatasetFingerprint]:
    payload_target_bytes = resource_policy.available_memory_bytes
    operation_headroom_bytes = resource_policy.memory_budget_bytes
    operation_headroom_bytes -= resource_policy.used_memory_bytes
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
    reader_upstream_headroom_bytes = operation_headroom_bytes
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
        operation_headroom_bytes=reader_upstream_headroom_bytes,
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
        partition_hash_xor[partition] = (
            partition_hash_xor.get(partition, 0) ^ chunk_xor
        )

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
    try:
        result = DatasetFingerprint(
            schema_hash=projected_schema_hash,
            row_count=row_count,
            key_columns=key_columns,
            content_hash=content_hash,
            partition_fingerprints=partition_fingerprints,
            file_count=file_count,
            column_names=output_columns,
        )
        result_bytes = dataset_fingerprint_retained_memory_bytes(result)
        if result_bytes > result_lease.reserved_bytes:
            raise ValueError("dataset_fingerprint_result_bound_invalid")
        result_lease.shrink_to(result_bytes)
        return _OwnedValue(result, result_lease)
    except BaseException:
        result = None
        raise


def _fingerprint_operation_bound(context: DatasetSession) -> int:
    pointer_bytes = struct.calcsize("P")
    paths = getattr(context, "read_paths", ())
    path_count = len(paths)
    columns = tuple(getattr(context, "logical_columns", ())) + tuple(
        getattr(context, "physical_columns", ())
    )
    text_bytes = sum(len(str(column).encode("utf-8")) for column in columns)
    metadata_bytes = len(getattr(context, "spec_bytes", b"")) + len(
        getattr(context, "manifest_bytes", b"")
    )
    return (
        metadata_bytes
        + text_bytes * 4
        + pointer_bytes * (4 * len(columns) + 8 * path_count)
        + path_count * 96
        + 1024
    )


def dataset_fingerprint_retained_memory_bytes(value: DatasetFingerprint) -> int:
    return _fingerprint_retained_memory_bytes(
        schema_hash=value.schema_hash,
        content_hash=value.content_hash,
        key_columns=value.key_columns,
        column_names=value.column_names,
        partition_hashes=(
            (str(partition["partition"]), str(partition["content_hash"]))
            for partition in value.partition_fingerprints
        ),
    )


def _fingerprint_retained_memory_bytes(
    *,
    schema_hash: str,
    content_hash: str,
    key_columns: tuple[str, ...],
    column_names: tuple[str, ...],
    partition_hashes: Iterator[tuple[str, str]],
) -> int:
    pointer_bytes = struct.calcsize("P")
    retained = pointer_bytes * (len(key_columns) + len(column_names))
    retained += sum(
        len(text.encode("utf-8"))
        for text in (
            schema_hash,
            content_hash,
            *key_columns,
            *column_names,
        )
    )
    for partition, partition_hash in partition_hashes:
        retained += 7 * pointer_bytes + 16
        retained += len(partition.encode("utf-8"))
        retained += len(partition_hash.encode("utf-8"))
    return retained


def _iter_fingerprint_hashes(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
    arrow_schema: pa.Schema | None = None,
    primary_key: tuple[str, ...] = (),
) -> Iterator[np.ndarray[Any, np.dtype[np.uint64]]]:
    """Hash formal primitive columns in bounded uint64 slices.

    Fingerprints are only a quick reject. General nested/map canonicalization
    is intentionally not part of the persisted dataset contract; exact compare
    remains the authority for any supported complex value.
    """
    if target_bytes <= 0:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    input_payload_bytes = pandas_frame_payload_bytes(frame)
    if input_payload_bytes > target_bytes:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    available_bytes = target_bytes - input_payload_bytes
    if arrow_schema is not None:
        for column in frame.columns:
            if column not in arrow_schema.names:
                continue
            data_type = arrow_schema.field(column).type
            if pa.types.is_nested(data_type) or (
                column in primary_key
                and (
                    pa.types.is_float32(data_type)
                    or pa.types.is_float64(data_type)
                )
            ):
                raise ValueError("unsupported_dataset_fingerprint_value")
    if len(frame) == 0:
        return
    if available_bytes < 8:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    rows_per_slice = max(1, available_bytes // 8)
    for start in range(0, len(frame), rows_per_slice):
        current = frame.iloc[start : start + rows_per_slice]
        try:
            hashes = pd.util.hash_pandas_object(
                current,
                index=False,
            ).to_numpy("uint64", copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported_dataset_fingerprint_value") from exc
        if hashes.nbytes > available_bytes:
            raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
        yield hashes
