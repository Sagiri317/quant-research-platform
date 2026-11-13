"""Compute streamed dataset summaries and content fingerprints for explicit
selections.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets.paths import relative_partition_label
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_file_batches,
    open_dataset_session,
    projection_columns_for_window,
)
from quant_research_platform.storage.datasets.schema import (
    physical_schema_from_schema,
    validate_dataset_manifest_payload,
)
from quant_research_platform.storage.datasets.types import DatasetFingerprint
from quant_research_platform.storage.exact_keys import _append_arrow_scalar_payload
from quant_research_platform.storage.parquet import DEFAULT_BATCH_ROWS


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetFingerprint:
    context = open_dataset_session(
        dataset_root, columns=columns, time_window=time_window
    )
    return fingerprint_dataset_session(context)


def comparison_schema(context: DatasetSession) -> dict[str, Any]:
    spec = context.spec
    hash_columns = tuple(
        dict.fromkeys((*spec["primary_key"], *context.logical_columns))
    )
    return {
        "frequency": spec["frequency"],
        "grain": spec["grain"],
        **{
            name: {
                column: value
                for column, value in spec.get(name, {}).items()
                if column in hash_columns
            }
            for name in ("column_roles", "column_units", "label_availability_columns")
        },
        "physical_schema": physical_schema_from_schema(
            pa.schema([context.arrow_schema.field(column) for column in hash_columns])
        ),
        "logical_dtypes": {
            descriptor["name"]: {
                key: value
                for key, value in descriptor.items()
                if key != "categories" or descriptor["ordered"]
            }
            for descriptor in context.logical_dtypes["columns"]
            if descriptor["name"] in hash_columns
        },
    }


def fingerprint_dataset_session(context: DatasetSession) -> DatasetFingerprint:
    if context.native:
        validate_dataset_manifest_payload(
            context.manifest, spec=context.spec, logical_columns=()
        )
    key_columns = tuple(str(column) for column in context.spec["primary_key"])
    hash_columns = tuple(dict.fromkeys((*key_columns, *context.logical_columns)))
    hash_physical_columns = projection_columns_for_window(
        hash_columns, spec=context.spec, time_window=context.time_window
    )
    output_columns = tuple(context.logical_columns)
    projected_schema_hash = canonical_json_hash(comparison_schema(context))
    row_count = 0
    row_hash_sum = 0
    row_hash_xor = 0
    partition_rows: dict[str, int] = {}
    partition_hash_sum: dict[str, int] = {}
    partition_hash_xor: dict[str, int] = {}
    file_count = 0
    last_observed_path: Path | None = None
    with closing_iterator(
        iter_dataset_file_batches(
            context,
            columns=hash_physical_columns,
            batch_rows=DEFAULT_BATCH_ROWS,
            validate_parts=True,
        )
    ) as batches:
        for path, chunk in batches:
            chunk = chunk.loc[:, list(hash_columns)]
            if chunk.empty:
                continue
            if path != last_observed_path:
                file_count += 1
                last_observed_path = path
            chunk_sum = 0
            chunk_xor = 0
            for hashes in _iter_fingerprint_hashes(
                chunk, arrow_schema=context.arrow_schema, primary_key=key_columns
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
    arrow_schema: pa.Schema | None = None,
    primary_key: tuple[str, ...] = (),
):
    """Hash one streamed batch; exact comparison remains the final authority."""
    if arrow_schema is not None and any(
        column in primary_key and pa.types.is_floating(arrow_schema.field(column).type)
        for column in frame.columns
        if column in arrow_schema.names
    ):
        schema = pa.schema([arrow_schema.field(column) for column in frame.columns])
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        columns = (*primary_key, *(name for name in frame if name not in primary_key))

        def hashes():
            for position in range(len(frame)):
                payload = bytearray()
                for name in columns:
                    key = name in primary_key
                    values = table.column(name)
                    _append_arrow_scalar_payload(
                        payload,
                        values[position],
                        values.type,
                        allow_null=not key,
                        allow_nonfinite=not key,
                        normalize_float_zero=key,
                    )
                yield int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

        yield np.fromiter(hashes(), dtype=np.uint64, count=len(frame))
        return
    if not len(frame):
        return
    try:
        yield pd.util.hash_pandas_object(frame, index=False).to_numpy(
            "uint64", copy=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported_dataset_fingerprint_value") from exc
