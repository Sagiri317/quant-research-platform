from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def write_low_level_parquet_artifact(
    artifact_root: Path,
    frame: pd.DataFrame,
    *,
    primary_key: Sequence[str],
    partition_columns: Sequence[str],
) -> None:
    """Construct low-level parquet fixtures for storage-adjacent tests."""

    spec = {
        "schema_id": "qrp-dataset-spec",
        "data_id": Path(artifact_root).name,
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "required_columns": [str(column) for column in frame.columns],
        "column_roles": _column_roles(frame.columns, primary_key),
    }
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data_root = root / "data"
    paths: list[Path] = []
    if "timestamp" in frame and tuple(partition_columns) == ("year",):
        result = frame.copy()
        result["_fixture_year"] = pd.to_datetime(result["timestamp"]).dt.year
        for year, year_frame in result.groupby("_fixture_year"):
            year_root = data_root / f"year={int(year)}"
            year_root.mkdir(parents=True, exist_ok=True)
            path = year_root / "part-000.parquet"
            year_frame.drop(columns=["_fixture_year"]).to_parquet(path, index=False)
            paths.append(path)
    else:
        data_root.mkdir(parents=True, exist_ok=True)
        path = data_root / "part-000.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    _write_runtime_manifest(root, data_root, paths)


def _write_runtime_manifest(root: Path, data_root: Path, paths: list[Path]) -> None:
    schemas: list[tuple[tuple[str, str], ...]] = []
    row_count = 0
    partition_row_counts: dict[str, int] = {}
    for path in sorted(paths):
        parquet_file = pq.ParquetFile(path)
        schemas.append(canonical_schema_from_file(parquet_file))
        partition = relative_partition_label(data_root, path)
        rows = int(parquet_file.metadata.num_rows)
        row_count += rows
        partition_row_counts[partition] = partition_row_counts.get(partition, 0) + rows
    first_schema = schemas[0]
    if any(schema != first_schema for schema in schemas[1:]):
        raise ValueError("dataset_schema_mismatch")
    manifest = {
        "schema_id": "qrp-dataset-manifest",
        "columns": [name for name, _ in first_schema],
        "schema_hash": sha256_json(first_schema),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
        "file_count": len(paths),
        "row_count": row_count,
    }
    manifest_path = root / "_manifest" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _column_roles(
    columns: Sequence[object],
    primary_key: Sequence[str],
) -> dict[str, str]:
    primary_key_set = {str(column) for column in primary_key}
    roles = {}
    for column in columns:
        name = str(column)
        if name == "timestamp":
            roles[name] = "timestamp"
        elif name in primary_key_set:
            roles[name] = "key"
        else:
            roles[name] = "feature"
    return roles


def canonical_schema_from_file(
    parquet_file: pq.ParquetFile,
) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, str(field.type)) for field in parquet_file.schema_arrow)


def relative_partition_label(data_root: Path, path: Path) -> str:
    label = path.parent.relative_to(data_root).as_posix()
    return "unpartitioned" if label == "." else label


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
