from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.storage.datasets import (
    _canonical_schema_from_file,
    _relative_partition_label,
    _sha256_json,
)


def write_low_level_parquet_artifact(
    artifact_root: Path,
    frame: pd.DataFrame,
    *,
    primary_key: Sequence[str],
    partition_columns: Sequence[str],
) -> None:
    """Construct low-level parquet fixtures for storage-adjacent tests."""

    spec = {
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "required_columns": [str(column) for column in frame.columns],
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
        schemas.append(_canonical_schema_from_file(parquet_file))
        partition = _relative_partition_label(data_root, path)
        rows = int(parquet_file.metadata.num_rows)
        row_count += rows
        partition_row_counts[partition] = partition_row_counts.get(partition, 0) + rows
    first_schema = schemas[0]
    if any(schema != first_schema for schema in schemas[1:]):
        raise ValueError("dataset_schema_mismatch")
    manifest = {
        "columns": [name for name, _ in first_schema],
        "schema_hash": _sha256_json(first_schema),
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
