"""A-share processed source-id mapping."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataFrequency, ProcessedDataSpec
from quant_research_platform.sources.csmar import ids as csmar_ids

MARKET_BARS_SOURCE_DATA_ID = csmar_ids.market_bars_data_id(DataFrequency.DAILY)


def market_bars_source_data_id(spec: ProcessedDataSpec) -> str:
    candidates = tuple(
        data_id
        for data_id in spec.source_input_data_ids
        if data_id == MARKET_BARS_SOURCE_DATA_ID
    )
    if len(candidates) != 1:
        raise ValueError("panel_market_bars_source_input_not_unique")
    return candidates[0]


__all__ = ["MARKET_BARS_SOURCE_DATA_ID", "market_bars_source_data_id"]
