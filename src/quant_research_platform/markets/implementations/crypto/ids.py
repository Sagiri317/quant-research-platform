"""Crypto market ids."""

from __future__ import annotations

from quant_research_platform.contracts.ids import MarketId

MARKET_ID = MarketId("crypto")
HFT_PANEL_DATA_ID = "binance_futures_l2_panel_100ms"
HFT_FACTOR_DATA_ID = "binance_futures_microstructure_factors_100ms"
HFT_LABEL_DATA_ID = "binance_futures_mid_labels_100ms"
HFT_RESEARCH_DATASET_ID = "binance_futures_hft_research_100ms"
_SUPPORTED_QUOTE_ASSETS = ("USDT", "USDC", "FDUSD", "BTC", "ETH")


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
