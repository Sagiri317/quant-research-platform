"""A-share market definition."""

from __future__ import annotations

from quant_research_platform.catalog.markets import MarketDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_MARKET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)


def a_share_market_definition() -> MarketDefinition:
    return MarketDefinition(
        market_id=A_SHARE_MARKET_ID,
        display_name="A-share market implementation",
        processed_spec_ids=(
            DAILY_PANEL_PROCESSED_DATA_ID,
            DAILY_CONTEXT_PROCESSED_DATA_ID,
            A_SHARE_DEFAULT_UNIVERSE_ID,
            A_SHARE_AS_OF_MASK_ID,
            A_SHARE_EXECUTION_MASK_ID,
            A_SHARE_EVALUATION_MASK_ID,
            PRICE_VOLUME_PROCESSED_DATA_ID,
            FORWARD_RETURNS_PROCESSED_DATA_ID,
        ),
    )


__all__ = ["a_share_market_definition"]
