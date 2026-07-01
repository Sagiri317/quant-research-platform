"""A-share processed source-id mapping."""

from __future__ import annotations

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    MARKET_BARS_SOURCE_DATA_ID,
)


def market_bars_source_data_id(spec: ProcessedDataSpec) -> str:
    candidates = tuple(
        data_id
        for data_id in spec.source_input_data_ids
        if data_id == MARKET_BARS_SOURCE_DATA_ID
    )
    if len(candidates) != 1:
        raise ValueError("panel_market_bars_source_input_not_unique")
    return candidates[0]


__all__ = ["market_bars_source_data_id"]
