"""CSMAR source identifiers."""

from __future__ import annotations

from quant_research_platform.contracts.data import (
    DataFrequency,
    require_supported_data_frequency,
)
from quant_research_platform.contracts.data import (
    market_bars_data_id as platform_market_bars_data_id,
)
from quant_research_platform.contracts.ids import SourceId

SOURCE_ID = SourceId("csmar")
SUPPORTED_MARKET_BAR_FREQUENCIES = (DataFrequency.DAILY,)


def market_bars_data_id(
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    normalized = require_supported_data_frequency(
        frequency,
        supported=SUPPORTED_MARKET_BAR_FREQUENCIES,
        context="csmar market bars",
    )
    return platform_market_bars_data_id(normalized)


__all__ = [
    "SOURCE_ID",
    "SUPPORTED_MARKET_BAR_FREQUENCIES",
    "market_bars_data_id",
]
