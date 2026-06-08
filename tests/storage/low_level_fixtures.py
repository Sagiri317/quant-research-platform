from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


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
    if "timestamp" in frame and tuple(partition_columns) == ("year",):
        result = frame.copy()
        result["_fixture_year"] = pd.to_datetime(result["timestamp"]).dt.year
        for year, year_frame in result.groupby("_fixture_year"):
            year_root = data_root / f"year={int(year)}"
            year_root.mkdir(parents=True, exist_ok=True)
            year_frame.drop(columns=["_fixture_year"]).to_parquet(
                year_root / "part-000.parquet",
                index=False,
            )
        return
    data_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_root / "part-000.parquet", index=False)
