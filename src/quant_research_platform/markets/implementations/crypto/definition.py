"""Crypto market definition."""

from __future__ import annotations

from quant_research_platform.catalog.markets import MarketDefinition
from quant_research_platform.markets.implementations.crypto.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
    MICROSTRUCTURE_PROCESSED_DATA_ID,
)


def market_definition() -> MarketDefinition:
    return MarketDefinition(
        market_id=MARKET_ID,
        display_name="Crypto spot market implementation",
        processed_spec_ids=(
            MARKET_PANEL_DATA_ID,
            MICROSTRUCTURE_PROCESSED_DATA_ID,
            FORWARD_RETURNS_PROCESSED_DATA_ID,
        ),
    )


__all__ = ["market_definition"]
