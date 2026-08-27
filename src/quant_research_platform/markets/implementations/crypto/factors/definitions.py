"""Crypto microstructure factor bindings."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_COLUMNS,
    HFT_PANEL_DATA_ID,
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.research.factors.microstructure.definitions import (
    microstructure_factor_definitions,
)


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        microstructure_factor_definitions(
            market_id=MARKET_ID,
            required_input_data_id=MARKET_PANEL_DATA_ID,
        )
    )


_HFT_FACTOR_INPUTS = {
    "spread": ("best_bid_price", "best_ask_price"),
    "relative_spread": ("best_bid_price", "best_ask_price", "mid"),
    "imbalance_1": ("bid_depth_1", "ask_depth_1"),
    "imbalance_5": ("bid_depth_5", "ask_depth_5"),
    "imbalance_10": ("bid_depth_10", "ask_depth_10"),
    "microprice": (
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
    ),
    "microprice_deviation": (
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "mid",
    ),
    "ofi_100ms": ("book_ofi_100ms",),
    "signed_trade_flow_100ms": ("observed_signed_trade_volume_100ms",),
    "taker_buy_volume_100ms": ("observed_taker_buy_volume_100ms",),
    "taker_sell_volume_100ms": ("observed_taker_sell_volume_100ms",),
    "return_100ms": ("mid",),
    "realized_volatility_1s": ("mid",),
    "top10_depth_per_spread": (
        "bid_depth_10",
        "ask_depth_10",
        "best_bid_price",
        "best_ask_price",
    ),
}


def hft_factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        FactorDefinition(
            factor_id=column,
            market_id=MARKET_ID,
            factor_set=FactorSet.CORE,
            input_columns=_HFT_FACTOR_INPUTS[column],
            output_column=column,
            base_input_columns=_HFT_FACTOR_INPUTS[column],
            display_name=column,
            description="Binance Futures public-L2 microstructure factor.",
            family="binance_futures_l2_microstructure",
            grain=DataGrain.RESEARCH_ROW,
            timing_semantics="as_of_futures_transaction_time",
            source_layer="processed",
            required_input_data_id=HFT_PANEL_DATA_ID,
        )
        for column in HFT_FACTOR_COLUMNS
    )
