"""A-share market definition."""

from __future__ import annotations

from quant_research_platform.catalog.markets import MarketDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    AS_OF_MASK_ID,
    DEFAULT_UNIVERSE_ID,
    EVALUATION_MASK_ID,
    EXECUTION_MASK_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    MARKET_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
    market_context_data_id,
    market_panel_data_id,
)


def market_definition() -> MarketDefinition:
    return MarketDefinition(
        market_id=MARKET_ID,
        display_name="A-share market implementation",
        processed_spec_ids=(
            market_panel_data_id(),
            market_context_data_id(),
            DEFAULT_UNIVERSE_ID,
            AS_OF_MASK_ID,
            EXECUTION_MASK_ID,
            EVALUATION_MASK_ID,
            PRICE_VOLUME_PROCESSED_DATA_ID,
            FORWARD_RETURNS_PROCESSED_DATA_ID,
        ),
    )


__all__ = ["market_definition"]
