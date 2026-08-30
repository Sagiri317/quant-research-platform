"""Stable A-share market identifiers."""

from __future__ import annotations

from quant_research_platform.contracts.data import (
    DataFrequency,
    data_frequency_slug,
    require_supported_data_frequency,
)
from quant_research_platform.contracts.ids import MarketId

MARKET_ID = MarketId("a_share")
SUPPORTED_MARKET_PANEL_FREQUENCIES = (DataFrequency.DAILY,)

DEFAULT_UNIVERSE_ID = "default"
AS_OF_MASK_ID = "as_of"
EXECUTION_MASK_ID = "execution"
EVALUATION_MASK_ID = "evaluation"

PRICE_VOLUME_PROCESSED_DATA_ID = "price_volume"
FUNDAMENTAL_PIT_PROCESSED_DATA_ID = "fundamental_pit"
FUNDAMENTAL_RAW_PROCESSED_DATA_ID = "fundamental_raw"
FORWARD_RETURNS_PROCESSED_DATA_ID = "forward_returns"

PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID = "price_volume_forward_returns"
FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID = "fundamental_forward_returns"


def market_panel_data_id(frequency: DataFrequency | str = DataFrequency.DAILY) -> str:
    normalized = require_supported_panel_frequency(frequency, context="market panel")
    return f"market_panel_{data_frequency_slug(normalized)}"


def market_context_data_id(
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    normalized = require_supported_panel_frequency(frequency, context="market context")
    return f"market_context_{data_frequency_slug(normalized)}"


def require_supported_panel_frequency(
    frequency: DataFrequency | str,
    *,
    context: str,
) -> DataFrequency:
    return require_supported_data_frequency(
        frequency,
        supported=SUPPORTED_MARKET_PANEL_FREQUENCIES,
        context=f"a_share {context}",
    )
