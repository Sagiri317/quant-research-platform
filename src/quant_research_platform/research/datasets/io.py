"""ResearchDataset IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
from quant_research_platform.storage.layout import layout_from_processed_market_root


def read_research_dataset(
    processed_root: Path,
    dataset_id: str,
    *,
    years: tuple[int, ...] | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    read_research_dataset_spec_payload(processed_root, dataset_id)
    artifact_root = _research_dataset_root(processed_root, dataset_id)
    frame = read_dataset(
        artifact_root,
        partition_filter=_year_filter(years),
        columns=columns,
    )
    if "year" in frame.columns:
        frame = frame.drop(columns=["year"])
    return frame


def read_research_dataset_spec_payload(
    processed_root: Path,
    dataset_id: str,
) -> dict[str, Any]:
    return inspect_dataset(_research_dataset_root(processed_root, dataset_id)).spec


def read_research_dataset_manifest(
    processed_root: Path,
    dataset_id: str,
) -> dict[str, Any]:
    return inspect_dataset(_research_dataset_root(processed_root, dataset_id)).manifest


def _research_dataset_root(processed_root: Path, dataset_id: str) -> Path:
    root = Path(processed_root)
    return layout_from_processed_market_root(processed_root).research_dataset_root(
        root.name,
        dataset_id,
    )


def _year_filter(years: tuple[int, ...] | None) -> dict[str, tuple[int, ...]] | None:
    if years is None:
        return None
    return {"year": years}


__all__ = [
    "read_research_dataset",
    "read_research_dataset_manifest",
    "read_research_dataset_spec_payload",
]
