"""Research dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True, kw_only=True)
class ResearchDatasetSpec(DataSpec):
    market_id: MarketId
    input_data_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    filter_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()


__all__ = ["ResearchDatasetSpec"]
