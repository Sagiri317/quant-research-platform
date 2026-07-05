"""Crypto microstructure factor definitions."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.research.factors.microstructure import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)

MICROSTRUCTURE_FACTOR_FAMILY = "microstructure"
MICROSTRUCTURE_INPUT_COLUMNS = (
    "best_bid",
    "best_ask",
    "bid_quantity",
    "ask_quantity",
    "bid_depth",
    "ask_depth",
    "mid_price",
    "trade_count",
    "signed_volume",
    "volume",
)


def factor_catalog() -> FactorCatalog:
    catalog = FactorCatalog()
    for column in sorted(MICROSTRUCTURE_FACTOR_COLUMNS):
        catalog.register(
            FactorDefinition(
                factor_id=column,
                market_id=MARKET_ID,
                factor_set=FactorSet.CORE,
                input_columns=MICROSTRUCTURE_INPUT_COLUMNS,
                output_column=column,
                display_name=column,
                description="Crypto spot microstructure factor.",
                family=MICROSTRUCTURE_FACTOR_FAMILY,
                required_windows=(5,),
                lookback_window=5,
                grain=DataGrain.ASSET_TIMESTAMP,
                timing_semantics="same_minute_bar_available_after_close",
                null_policy="preserve_nulls_and_zero_denominator_as_null",
                required_input_data_id=MARKET_PANEL_DATA_ID,
                window=5,
            )
        )
    return catalog


__all__ = [
    "MICROSTRUCTURE_FACTOR_FAMILY",
    "MICROSTRUCTURE_INPUT_COLUMNS",
    "factor_catalog",
]
