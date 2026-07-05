"""Crypto market ids."""

from __future__ import annotations

from quant_research_platform.contracts.ids import MarketId

MARKET_ID = MarketId("crypto")
MARKET_PANEL_DATA_ID = "market_panel_1m"
MICROSTRUCTURE_PROCESSED_DATA_ID = "microstructure"
FORWARD_RETURNS_PROCESSED_DATA_ID = "forward_returns"
MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID = "microstructure_forward_returns"


def market_panel_data_id() -> str:
    return MARKET_PANEL_DATA_ID


__all__ = [
    "FORWARD_RETURNS_PROCESSED_DATA_ID",
    "MARKET_ID",
    "MARKET_PANEL_DATA_ID",
    "MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID",
    "MICROSTRUCTURE_PROCESSED_DATA_ID",
    "market_panel_data_id",
]
