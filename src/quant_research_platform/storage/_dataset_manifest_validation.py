"""Dataset manifest runtime validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.storage._dataset_paths import (
    candidate_data_files,
    relative_partition_label,
)
from quant_research_platform.storage._dataset_schema import (
    physical_schema_columns,
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
    validate_physical_schema_payload,
)
from quant_research_platform.storage.manifests import read_qrp_json


def validate_written_artifact(root: Path, accumulator: Any) -> None:
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    spec_payload = read_qrp_json(
        spec_path,
        expected_schema_id="qrp-dataset-spec",
        error="dataset_spec_schema_id_invalid",
    )
    manifest = read_qrp_json(
        manifest_path,
        expected_schema_id="qrp-dataset-manifest",
        error="dataset_manifest_schema_id_invalid",
    )
    validate_dataset_spec_payload(spec_payload)
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    paths = candidate_data_files(data_root, spec=spec_payload)
    if len(paths) != accumulator.file_count:
        raise ValueError("written_file_count_mismatch")
    actual_partitions = sorted(
        {relative_partition_label(data_root, path) for path in paths}
    )
    if actual_partitions != sorted(accumulator.partition_row_counts):
        raise ValueError("written_partition_mismatch")
    validate_manifest_matches_accumulator(manifest, accumulator)


def validate_manifest_matches_accumulator(
    manifest: Mapping[str, Any],
    accumulator: Any,
) -> None:
    if manifest["row_count"] != accumulator.row_count:
        raise ValueError("manifest_row_count_mismatch")
    if manifest["file_count"] != accumulator.file_count:
        raise ValueError("manifest_file_count_mismatch")
    if manifest["file_size_bytes"] != accumulator.file_size_bytes:
        raise ValueError("manifest_file_size_bytes_mismatch")
    if tuple(manifest["columns"]) != accumulator.columns:
        raise ValueError("manifest_columns_mismatch")
    if manifest["physical_schema"] != accumulator._schema:
        raise ValueError("manifest_physical_schema_mismatch")
    if manifest["schema_hash"] != accumulator.schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    if manifest_partition_row_counts(manifest) != accumulator.partition_row_counts:
        raise ValueError("manifest_partition_row_counts_mismatch")
    if sum(manifest_partition_row_counts(manifest).values()) != accumulator.row_count:
        raise ValueError("manifest_row_count_mismatch")


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
    validate_row_counts: bool,
) -> None:
    manifest_file_count, _, partition_row_counts = (
        validate_manifest_runtime_metadata_shape(manifest)
    )
    if manifest_file_count != len(paths):
        raise ValueError("manifest_file_count_mismatch")
    actual_partitions = {relative_partition_label(data_root, path) for path in paths}
    if set(partition_row_counts) != actual_partitions:
        raise ValueError("manifest_partition_row_counts_mismatch")
    if validate_row_counts:
        actual_row_counts = parquet_row_counts_by_partition(data_root, paths)
        if actual_row_counts != partition_row_counts:
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


def parquet_row_counts_by_partition(
    data_root: Path,
    paths: list[Path],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        partition = relative_partition_label(data_root, path)
        row_count = int(pq.ParquetFile(path).metadata.num_rows)
        counts[partition] = counts.get(partition, 0) + row_count
    return dict(sorted(counts.items()))
