"""Crypto market ids."""

from __future__ import annotations

from quant_research_platform.contracts.ids import MarketId

MARKET_ID = MarketId("crypto")
MARKET_PANEL_DATA_ID = "market_panel_1m"
MICROSTRUCTURE_PROCESSED_DATA_ID = "microstructure"
FORWARD_RETURNS_PROCESSED_DATA_ID = "forward_returns"
MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID = "microstructure_forward_returns"
_SUPPORTED_QUOTE_ASSETS = ("USDT", "USDC", "FDUSD", "BTC", "ETH")


def market_panel_data_id() -> str:
    return MARKET_PANEL_DATA_ID


def canonical_crypto_asset_id(source_asset_id: str) -> str:
    if not isinstance(source_asset_id, str):
        raise ValueError("unsupported_crypto_source_asset_id")
    symbol = source_asset_id.strip().upper()
    if symbol == "":
        raise ValueError("unsupported_crypto_source_asset_id")
    for quote in _SUPPORTED_QUOTE_ASSETS:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            if base.isalnum():
                return f"{base.lower()}_{quote.lower()}"
    raise ValueError("unsupported_crypto_source_asset_id")


__all__ = [
    "canonical_crypto_asset_id",
    "FORWARD_RETURNS_PROCESSED_DATA_ID",
    "MARKET_ID",
    "MARKET_PANEL_DATA_ID",
    "MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID",
    "MICROSTRUCTURE_PROCESSED_DATA_ID",
    "market_panel_data_id",
]
