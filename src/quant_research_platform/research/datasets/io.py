"""ResearchDataset IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.storage.layout import (
    research_dataset_data_root,
    research_dataset_manifest_path,
    research_dataset_spec_path,
)
from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.parquet import read_partitioned_parquet_dataset


def read_research_dataset(
    processed_root: Path,
    dataset_id: str,
    *,
    years: tuple[int, ...] | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    read_research_dataset_spec_payload(processed_root, dataset_id)
    data_root = research_dataset_data_root(processed_root, dataset_id)
    frame = read_partitioned_parquet_dataset(data_root, years=years, columns=columns)
    if "year" in frame.columns:
        frame = frame.drop(columns=["year"])
    return frame


def read_research_dataset_spec_payload(
    processed_root: Path,
    dataset_id: str,
) -> dict[str, Any]:
    return read_manifest(research_dataset_spec_path(processed_root, dataset_id))


def read_research_dataset_manifest(
    processed_root: Path,
    dataset_id: str,
) -> dict[str, Any]:
    return read_manifest(research_dataset_manifest_path(processed_root, dataset_id))


__all__ = [
    "read_research_dataset",
    "read_research_dataset_manifest",
    "read_research_dataset_spec_payload",
]
