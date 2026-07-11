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

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_paths import relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    DEFAULT_DATASET_PHYSICAL_TARGET_BYTES,
    build_dataset_read_context,
    dataset_categorical_dtypes,
    iter_dataset_file_batches,
    read_columns_for_time_window,
)
from quant_research_platform.storage._dataset_schema import sha256_json
from quant_research_platform.storage._dataset_types import DatasetFingerprint

_FINGERPRINT_PHYSICAL_TARGET_BYTES = DEFAULT_DATASET_PHYSICAL_TARGET_BYTES // 2
_FINGERPRINT_HASH_TARGET_BYTES = (
    DEFAULT_DATASET_PHYSICAL_TARGET_BYTES - _FINGERPRINT_PHYSICAL_TARGET_BYTES
)


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetFingerprint:
    context = build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )
    key_columns = tuple(str(column) for column in context.spec["primary_key"])
    hash_columns = tuple(dict.fromkeys((*key_columns, *context.logical_columns)))
    hash_physical_columns = read_columns_for_time_window(
        hash_columns,
        spec=context.spec,
        time_window=context.time_window,
    )
    output_columns = tuple(context.logical_columns)
    categorical_dtypes, categorical_state_bytes = dataset_categorical_dtypes(
        context.all_paths,
        hash_physical_columns,
    )
    row_count = 0
    row_hash_sum = 0
    row_hash_xor = 0
    partition_rows: dict[str, int] = {}
    partition_hash_sum: dict[str, int] = {}
    partition_hash_xor: dict[str, int] = {}

    observed_paths: set[Path] = set()
    for path, chunk in iter_dataset_file_batches(
        context,
        columns=hash_physical_columns,
        target_chunk_bytes=_FINGERPRINT_PHYSICAL_TARGET_BYTES,
        categorical_dtypes=categorical_dtypes,
        categorical_state_bytes=categorical_state_bytes,
    ):
        chunk = chunk.loc[:, list(hash_columns)]
        if chunk.empty:
            continue
        observed_paths.add(path)
        hashes = pd.util.hash_pandas_object(
            _hashable_fingerprint_frame(
                chunk,
                target_bytes=_FINGERPRINT_HASH_TARGET_BYTES,
            ),
            index=False,
        ).to_numpy("uint64")
        chunk_sum = int(hashes.sum(dtype=np.uint64))
        chunk_xor = int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
        partition = relative_partition_label(context.data_root, path)
        row_count += int(len(chunk))
        row_hash_sum = (row_hash_sum + chunk_sum) % 2**64
        row_hash_xor ^= chunk_xor
        partition_rows[partition] = partition_rows.get(partition, 0) + len(chunk)
        partition_hash_sum[partition] = (
            partition_hash_sum.get(partition, 0) + chunk_sum
        ) % 2**64
        partition_hash_xor[partition] = partition_hash_xor.get(partition, 0) ^ chunk_xor

    content_hash = sha256_json(
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
            "content_hash": sha256_json(
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
        file_count=len(observed_paths),
        column_names=output_columns,
    )


def _hashable_fingerprint_frame(
    frame: pd.DataFrame,
    *,
    target_bytes: int = DEFAULT_DATASET_PHYSICAL_TARGET_BYTES,
) -> pd.DataFrame:
    if target_bytes <= 0:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    retained_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if 2 * retained_bytes > target_bytes:
        raise ValueError("dataset_fingerprint_value_exceeds_target_bytes")
    output = frame.copy()
    for column in output.columns:
        try:
            pd.util.hash_pandas_object(output[column], index=False)
        except TypeError:
            canonical_bytes = 0
            for value in output[column]:
                canonical_bytes += _canonical_nested_value_bound(value)
                if 2 * retained_bytes + 2 * canonical_bytes > target_bytes:
                    raise ValueError(
                        "dataset_fingerprint_value_exceeds_target_bytes"
                    ) from None
            output[column] = output[column].map(_canonical_nested_value)
            retained_bytes += canonical_bytes
    return output


def _canonical_nested_value(value: Any) -> str:
    return "qrp-nested:" + json.dumps(
        _json_nested_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_nested_value_bound(value: Any) -> int:
    return len("qrp-nested:") + _nested_json_bound(value)


def _nested_json_bound(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return 32 + 16 * value.ndim + sum(
            _nested_json_bound(item) + 1 for item in value.flat
        )
    if isinstance(value, np.generic):
        return _nested_json_bound(value.item())
    if isinstance(value, (list, tuple)):
        return 24 + sum(_nested_json_bound(item) + 1 for item in value)
    if isinstance(value, dict):
        return 24 + sum(
            _json_string_bound(str(key)) + _nested_json_bound(item) + 4
            for key, item in value.items()
        )
    if isinstance(value, bytes):
        return 24 + 2 * len(value)
    if isinstance(value, Decimal):
        decimal_tuple = value.as_tuple()
        return 64 + len(decimal_tuple.digits) + int(decimal_tuple.exponent).bit_length()
    if isinstance(value, (pd.Timestamp, datetime, date, float)):
        return 96
    if value is None or isinstance(value, bool):
        return 5
    if isinstance(value, int):
        return max(2, (value.bit_length() * 31 + 102) // 103 + 2)
    if isinstance(value, str):
        return _json_string_bound(value)
    raise ValueError("unsupported_dataset_fingerprint_value")


def _json_string_bound(value: str) -> int:
    encoded = 2
    for character in value:
        codepoint = ord(character)
        if codepoint > 0xFFFF:
            encoded += 12
        elif codepoint < 0x20 or codepoint > 0x7E:
            encoded += 6
        elif character in {'"', "\\"}:
            encoded += 2
        else:
            encoded += 1
    return encoded


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
