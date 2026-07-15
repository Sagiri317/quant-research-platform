"""Dataset manifest runtime validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.storage._dataset_paths import (
    relative_partition_label,
)
from quant_research_platform.storage._dataset_schema import (
    physical_schema_columns,
    validate_physical_schema_payload,
)


def manifest_has_runtime_metadata(manifest: Mapping[str, Any]) -> bool:
    return all(
        key in manifest
        for key in (
            "columns",
            "physical_schema",
            "schema_hash",
            "partition_row_counts",
            "file_count",
            "row_count",
        )
    )


def validate_manifest_runtime_metadata(
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    paths: list[Path],
) -> None:
    manifest_file_count, _, partition_row_counts = (
        validate_manifest_runtime_metadata_shape(manifest)
    )
    if manifest_file_count != len(paths):
        raise ValueError("manifest_file_count_mismatch")
    actual_partitions = {relative_partition_label(data_root, path) for path in paths}
    if set(partition_row_counts) != actual_partitions:
        raise ValueError("manifest_partition_row_counts_mismatch")


def validate_manifest_runtime_metadata_shape(
    manifest: Mapping[str, Any],
) -> tuple[int, int, dict[str, int]]:
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("manifest_columns_invalid")
    physical_schema = manifest.get("physical_schema")
    validate_physical_schema_payload(physical_schema)
    if tuple(columns) != physical_schema_columns(physical_schema):
        raise ValueError("manifest_columns_mismatch")
    schema_hash = manifest.get("schema_hash")
    if (
        not isinstance(schema_hash, str)
        or len(schema_hash) != 64
        or any(character not in "0123456789abcdef" for character in schema_hash)
    ):
        raise ValueError("manifest_schema_hash_invalid")
    if canonical_json_hash(physical_schema) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    manifest_file_count = manifest.get("file_count")
    manifest_row_count = manifest.get("row_count")
    if type(manifest_file_count) is not int or type(manifest_row_count) is not int:
        raise ValueError("manifest_counts_invalid")
    if manifest_file_count < 0 or manifest_row_count < 0:
        raise ValueError("manifest_counts_invalid")
    partition_row_counts = manifest_partition_row_counts(manifest)
    if sum(partition_row_counts.values()) != manifest_row_count:
        raise ValueError("manifest_row_count_mismatch")
    return manifest_file_count, manifest_row_count, partition_row_counts


def manifest_partition_row_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    value = manifest.get("partition_row_counts")
    if not isinstance(value, dict) or not value:
        raise ValueError("manifest_partition_row_counts_invalid")
    counts: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("manifest_partition_row_counts_invalid")
        if type(item) is not int or item < 0:
            raise ValueError("manifest_partition_row_counts_invalid")
        counts[key] = item
    return dict(sorted(counts.items()))
