"""Crypto microstructure factor bindings."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.research.factors.microstructure.definitions import (
    microstructure_factor_definitions,
)

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


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        microstructure_factor_definitions(
            market_id=MARKET_ID,
            required_input_data_id=MARKET_PANEL_DATA_ID,
        )
    )
