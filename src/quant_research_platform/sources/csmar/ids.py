"""CSMAR source identifiers."""

from __future__ import annotations

from quant_research_platform.contracts.data import (
    DataFrequency,
    data_frequency_slug,
    require_supported_data_frequency,
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
    return f"market_bars_{data_frequency_slug(normalized)}"
