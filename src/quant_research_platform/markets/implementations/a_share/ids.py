"""Stable A-share market identifiers."""

from __future__ import annotations

from quant_research_platform.contracts.ids import MarketId

A_SHARE_MARKET_ID = MarketId("a_share")

DAILY_PANEL_PROCESSED_DATA_ID = "daily_panel"
DAILY_CONTEXT_PROCESSED_DATA_ID = "daily_context"

A_SHARE_DEFAULT_UNIVERSE_ID = "a_share_default"
A_SHARE_AS_OF_MASK_ID = "a_share_as_of"
A_SHARE_EXECUTION_MASK_ID = "a_share_execution"
A_SHARE_EVALUATION_MASK_ID = "a_share_evaluation"

PRICE_VOLUME_PROCESSED_DATA_ID = "price_volume"
FORWARD_RETURNS_PROCESSED_DATA_ID = "forward_returns"

A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID = (
    "a_share_price_volume_forward_returns"
)

__all__ = [
    "A_SHARE_AS_OF_MASK_ID",
    "A_SHARE_DEFAULT_UNIVERSE_ID",
    "A_SHARE_EVALUATION_MASK_ID",
    "A_SHARE_EXECUTION_MASK_ID",
    "A_SHARE_MARKET_ID",
    "A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID",
    "DAILY_CONTEXT_PROCESSED_DATA_ID",
    "DAILY_PANEL_PROCESSED_DATA_ID",
    "FORWARD_RETURNS_PROCESSED_DATA_ID",
    "PRICE_VOLUME_PROCESSED_DATA_ID",
]
