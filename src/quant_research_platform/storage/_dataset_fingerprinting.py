"""Dataset fingerprinting implementation."""

from __future__ import annotations

import json
import struct
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.contracts._json import canonical_json_hash
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
    row_count = 0
    row_hash_sum = 0
    row_hash_xor = 0
    partition_rows: dict[str, int] = {}
    partition_hash_sum: dict[str, int] = {}
    partition_hash_xor: dict[str, int] = {}

    file_count = 0
    last_observed_path: Path | None = None
    accepted_arrow_bytes: int | None = None

    def observe_batch(batch: Any, _frame: pd.DataFrame) -> None:
        nonlocal accepted_arrow_bytes
        if accepted_arrow_bytes is not None:
            raise RuntimeError("dataset_fingerprint_batch_observer_out_of_sequence")
        accepted_arrow_bytes = batch.get_total_buffer_size()

    for path, chunk in iter_dataset_file_batches(
        context,
        columns=hash_physical_columns,
        target_chunk_bytes=payload_target_bytes,
        operation_headroom_bytes=operation_headroom_bytes,
        batch_observer=observe_batch,
    ):
        if accepted_arrow_bytes is None:
            raise RuntimeError("dataset_fingerprint_batch_observer_missing")
        hash_target_bytes = operation_headroom_bytes - accepted_arrow_bytes
        accepted_arrow_bytes = None
        chunk = chunk.loc[:, list(hash_columns)]
        if chunk.empty:
            continue
        if path != last_observed_path:
            file_count += 1
            last_observed_path = path
        hashes = pd.util.hash_pandas_object(
            _hashable_fingerprint_frame(
                chunk,
                target_bytes=hash_target_bytes,
            ),
            index=False,
        ).to_numpy("uint64")
        chunk_sum = int(hashes.sum(dtype=np.uint64))
        chunk_xor = int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
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
        schema_hash=context.schema_hash,
        row_count=row_count,
        key_columns=key_columns,
        content_hash=content_hash,
        partition_fingerprints=partition_fingerprints,
        file_count=file_count,
        column_names=output_columns,
    )


def _hashable_fingerprint_frame(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
) -> pd.DataFrame:
    if target_bytes <= 0:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    input_payload_bytes = pandas_frame_payload_bytes(frame)
    if input_payload_bytes > target_bytes:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    output = frame
    for column in frame.columns:
        try:
            pd.util.hash_pandas_object(frame[column], index=False)
        except TypeError:
            if output is frame:
                output = frame.copy()
            output[column] = frame[column].map(_canonical_nested_value)
            if input_payload_bytes + pandas_frame_payload_bytes(output) > target_bytes:
                raise ValueError(
                    "dataset_fingerprint_value_exceeds_target_bytes"
                ) from None
    return output


def _canonical_nested_value(value: Any) -> str:
    return "qrp-nested:" + json.dumps(
        _json_nested_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_nested_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array": [_json_nested_value(item) for item in value.tolist()]}
    if isinstance(value, np.generic):
        return _json_nested_value(value.item())
    if isinstance(value, list):
        return {"list": [_json_nested_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"tuple": [_json_nested_value(item) for item in value]}
    if isinstance(value, dict):
        return {
            "dict": [
                [str(key), _json_nested_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, pd.Timestamp):
        return {"timestamp_ns": int(value.value)}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, float):
        return {"float64_bits": struct.pack(">d", value).hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise ValueError("unsupported_dataset_fingerprint_value")


def partition_keys(fingerprint: DatasetFingerprint) -> tuple[str, ...]:
    return tuple(str(item["partition"]) for item in fingerprint.partition_fingerprints)
