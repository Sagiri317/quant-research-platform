"""Portable disk layout for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.contracts.ids import MarketId, SourceId


@dataclass(frozen=True)
class StorageLayout:
    project_root: Path = Path(".")

    def raw_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/raw" / str(source_id)

    def interim_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/interim" / str(source_id)

    def interim_data_root(self, source_id: SourceId, interim_data_id: str) -> Path:
        return self.interim_source_root(source_id) / interim_data_id

    def processed_market_root(self, market_id: MarketId) -> Path:
        return self.project_root / "data/processed" / str(market_id)

    def processed_data_root(
        self,
        market_id: MarketId,
        category: str,
        processed_data_id: str,
    ) -> Path:
        return self.processed_market_root(market_id) / category / processed_data_id

    def research_dataset_root(self, market_id: MarketId, dataset_id: str) -> Path:
        return self.processed_market_root(market_id) / "research_datasets" / dataset_id

    def private_report_root(self, market_id: MarketId, run_type: str) -> Path:
        return self.project_root / "reports/private_real" / str(market_id) / run_type


def processed_data_parquet_root(
    processed_root: Path,
    category: str,
    processed_data_id: str,
) -> Path:
    return processed_root / category / processed_data_id / "data"


def processed_data_manifest_path(
    processed_root: Path,
    category: str,
    processed_data_id: str,
) -> Path:
    return processed_root / category / processed_data_id / "_manifest" / "manifest.json"


def research_dataset_data_root(processed_root: Path, dataset_id: str) -> Path:
    return processed_root / "research_datasets" / dataset_id / "data"


def research_dataset_manifest_path(processed_root: Path, dataset_id: str) -> Path:
    return (
        processed_root
        / "research_datasets"
        / dataset_id
        / "_manifest"
        / "manifest.json"
    )


def research_dataset_spec_path(processed_root: Path, dataset_id: str) -> Path:
    return processed_root / "research_datasets" / dataset_id / "spec.json"


def layout_from_processed_market_root(processed_root: Path) -> StorageLayout:
    """Resolve ``data/processed/<market>`` back to a project storage layout."""

    root = Path(processed_root)
    parts = root.parts
    if len(parts) < 3 or parts[-3:-1] != ("data", "processed"):
        raise ValueError("processed_root_must_be_data_processed_market")
    project_root = Path(*parts[:-3]) if len(parts) > 3 else Path(".")
    return StorageLayout(project_root=project_root)


__all__ = [
    "StorageLayout",
    "layout_from_processed_market_root",
    "processed_data_manifest_path",
    "processed_data_parquet_root",
    "research_dataset_data_root",
    "research_dataset_manifest_path",
    "research_dataset_spec_path",
]
