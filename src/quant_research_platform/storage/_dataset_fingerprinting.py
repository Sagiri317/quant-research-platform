"""Dataset fingerprinting implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_paths import _relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    _build_dataset_read_context,
    _filter_frame_by_time_window,
    _read_columns_for_time_window,
)
from quant_research_platform.storage._dataset_schema import _sha256_json
from quant_research_platform.storage._dataset_types import DatasetFingerprint


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetFingerprint:
    context = _build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )
    key_columns = tuple(str(column) for column in context.spec["primary_key"])
    hash_columns = tuple(dict.fromkeys((*key_columns, *context.logical_columns)))
    hash_physical_columns = _read_columns_for_time_window(
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

    for path in context.read_paths:
        chunk = pd.read_parquet(
            path,
            columns=list(hash_physical_columns),
            engine="pyarrow",
            filters=context.parquet_filters,
        )
        chunk = _filter_frame_by_time_window(
            chunk,
            spec=context.spec,
            time_window=context.time_window,
        )
        chunk = chunk.loc[:, list(hash_columns)]
        if chunk.empty:
            continue
        file_count += 1
        hashes = pd.util.hash_pandas_object(
            chunk.loc[:, list(hash_columns)],
            index=False,
        ).to_numpy("uint64")
        chunk_sum = int(hashes.sum(dtype=np.uint64))
        chunk_xor = int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
        partition = _relative_partition_label(context.data_root, path)
        row_count += int(len(chunk))
        row_hash_sum = (row_hash_sum + chunk_sum) % 2**64
        row_hash_xor ^= chunk_xor
        partition_rows[partition] = partition_rows.get(partition, 0) + len(chunk)
        partition_hash_sum[partition] = (
            partition_hash_sum.get(partition, 0) + chunk_sum
        ) % 2**64
        partition_hash_xor[partition] = partition_hash_xor.get(partition, 0) ^ chunk_xor

    content_hash = _sha256_json(
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
            "content_hash": _sha256_json(
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


def _partition_keys(fingerprint: DatasetFingerprint) -> tuple[str, ...]:
    return tuple(str(item["partition"]) for item in fingerprint.partition_fingerprints)
