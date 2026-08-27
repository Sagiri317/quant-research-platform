"""Crypto market ids."""

from __future__ import annotations

from quant_research_platform.contracts.ids import MarketId

MARKET_ID = MarketId("crypto")
MARKET_PANEL_DATA_ID = "market_panel_1m"
MICROSTRUCTURE_PROCESSED_DATA_ID = "microstructure"
FORWARD_RETURNS_PROCESSED_DATA_ID = "forward_returns"
MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID = "microstructure_forward_returns"
HFT_PANEL_DATA_ID = "binance_futures_l2_panel_100ms"
HFT_FACTOR_DATA_ID = "binance_futures_microstructure_factors_100ms"
HFT_LABEL_DATA_ID = "binance_futures_mid_labels_100ms"
HFT_RESEARCH_DATASET_ID = "binance_futures_hft_research_100ms"
HFT_LABEL_HORIZONS_MS = (100, 500, 1_000, 5_000)
HFT_MODEL_LABEL_HORIZON_MS = 500
HFT_FACTOR_COLUMNS = (
    "spread",
    "relative_spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice",
    "microprice_deviation",
    "ofi_100ms",
    "signed_trade_flow_100ms",
    "taker_buy_volume_100ms",
    "taker_sell_volume_100ms",
    "return_100ms",
    "realized_volatility_1s",
    "top10_depth_per_spread",
)
HFT_LABEL_COLUMNS = tuple(
    column
    for horizon in HFT_LABEL_HORIZONS_MS
    for column in (
        f"future_mid_return_{horizon}ms",
        f"future_mid_direction_{horizon}ms",
    )
)
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
