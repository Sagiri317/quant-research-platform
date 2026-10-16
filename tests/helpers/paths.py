"""Synthetic fixture layout; production operations use explicit paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId, SourceId


@dataclass(frozen=True)
class StorageLayout:
    project_root: Path = Path(".")

    def raw_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/raw" / str(source_id)

    def interim_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/interim" / str(source_id)

    def interim_data_root(self, source_id: SourceId, data_id: str) -> Path:
        return self.interim_source_root(source_id) / data_id

    def processed_market_root(self, market_id: MarketId) -> Path:
        return self.project_root / "data/processed" / str(market_id)

    def processed_data_root(
        self, market_id: MarketId, kind: ProcessedDataKind | str, data_id: str
    ) -> Path:
        return self.processed_market_root(market_id) / str(kind) / data_id

    def research_dataset_root(self, market_id: MarketId, data_id: str) -> Path:
        return self.project_root / "data/research" / str(market_id) / data_id
