"""Stable A-share market identifiers and market-only data bindings."""

from __future__ import annotations

from types import MappingProxyType

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.ids import MarketId

MARKET_ID = MarketId("a_share")
SUPPORTED_MARKET_PANEL_FREQUENCIES = (DataFrequency.DAILY,)
MARKET_PANEL_ID = "market_panel_1d"
EXECUTION_MASK_ID = "execution_mask"
PRICE_VOLUME_DATA_ID = "price_volume_facts"
FUNDAMENTAL_PIT_ID = "fundamental_pit"
FUNDAMENTAL_DATA_ID = "fundamental_factors"
EXPECTATIONS_DATA_ID = "expectations_factors"
FORWARD_RETURNS_DATA_ID = "forward_returns"

FORMAL_BENCHMARKS = MappingProxyType(
    {
        "000905": ("csi500", "中证小盘500指数"),
        "000852": ("csi1000", "中证1000指数"),
    }
)
