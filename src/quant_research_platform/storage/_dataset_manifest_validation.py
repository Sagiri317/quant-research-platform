"""Dataset manifest runtime validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quant_research_platform.storage._dataset_paths import (
    _candidate_data_files,
    _relative_partition_label,
)
from quant_research_platform.storage.manifests import read_qrp_json


def _validate_written_artifact(root: Path, accumulator: Any) -> None:
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
    if not _manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    paths = _candidate_data_files(data_root, spec=spec_payload)
    if len(paths) != accumulator.file_count:
        raise ValueError("written_file_count_mismatch")
    actual_partitions = sorted(
        {_relative_partition_label(data_root, path) for path in paths}
    )
    if actual_partitions != sorted(accumulator.partition_row_counts):
        raise ValueError("written_partition_mismatch")
    _validate_manifest_matches_accumulator(manifest, accumulator)


def _validate_manifest_matches_accumulator(
    manifest: Mapping[str, Any],
    accumulator: Any,
) -> None:
    if int(manifest["row_count"]) != int(accumulator.row_count):
        raise ValueError("manifest_row_count_mismatch")
    if int(manifest["file_count"]) != int(accumulator.file_count):
        raise ValueError("manifest_file_count_mismatch")
    if tuple(str(column) for column in manifest["columns"]) != accumulator.columns:
        raise ValueError("manifest_columns_mismatch")
    if str(manifest["schema_hash"]) != accumulator.schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    if _manifest_partition_row_counts(manifest) != accumulator.partition_row_counts:
        raise ValueError("manifest_partition_row_counts_mismatch")
    if sum(_manifest_partition_row_counts(manifest).values()) != accumulator.row_count:
        raise ValueError("manifest_row_count_mismatch")


def _manifest_has_runtime_metadata(manifest: Mapping[str, Any]) -> bool:
    return all(
        key in manifest
        for key in (
            "columns",
            "schema_hash",
            "partition_row_counts",
            "file_count",
            "row_count",
        )
    )


def _validate_manifest_runtime_metadata(
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    paths: list[Path],
    validate_row_counts: bool,
) -> None:
    manifest_file_count, _, partition_row_counts = (
        _validate_manifest_runtime_metadata_shape(manifest)
    )
    if manifest_file_count != len(paths):
        raise ValueError("manifest_file_count_mismatch")
    actual_partitions = {_relative_partition_label(data_root, path) for path in paths}
    if set(partition_row_counts) != actual_partitions:
        raise ValueError("manifest_partition_row_counts_mismatch")
    if validate_row_counts:
        actual_row_counts = _parquet_row_counts_by_partition(data_root, paths)
        if actual_row_counts != partition_row_counts:
            raise ValueError("manifest_partition_row_counts_mismatch")


def _validate_manifest_runtime_metadata_shape(
    manifest: Mapping[str, Any],
) -> tuple[int, int, dict[str, int]]:
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) for column in columns)
    ):
        raise ValueError("manifest_columns_invalid")
    schema_hash = manifest.get("schema_hash")
    if not isinstance(schema_hash, str) or not schema_hash:
        raise ValueError("manifest_schema_hash_invalid")
    try:
        manifest_file_count = int(manifest["file_count"])
        manifest_row_count = int(manifest["row_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest_counts_invalid") from exc
    if manifest_file_count < 0 or manifest_row_count < 0:
        raise ValueError("manifest_counts_invalid")
    partition_row_counts = _manifest_partition_row_counts(manifest)
    if sum(partition_row_counts.values()) != manifest_row_count:
        raise ValueError("manifest_row_count_mismatch")
    return manifest_file_count, manifest_row_count, partition_row_counts


def _manifest_partition_row_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    value = manifest.get("partition_row_counts")
    if not isinstance(value, dict) or not value:
        raise ValueError("manifest_partition_row_counts_invalid")
    counts: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("manifest_partition_row_counts_invalid")
        try:
            count = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest_partition_row_counts_invalid") from exc
        if count < 0:
            raise ValueError("manifest_partition_row_counts_invalid")
        counts[key] = count
    return dict(sorted(counts.items()))


def _parquet_row_counts_by_partition(
    data_root: Path,
    paths: list[Path],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        partition = _relative_partition_label(data_root, path)
        row_count = int(pq.ParquetFile(path).metadata.num_rows)
        counts[partition] = counts.get(partition, 0) + row_count
    return dict(sorted(counts.items()))
