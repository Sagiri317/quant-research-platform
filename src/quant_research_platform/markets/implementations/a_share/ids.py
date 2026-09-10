"""Stable A-share market identifiers and market-only data bindings."""

from __future__ import annotations

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
RESEARCH_DATASET_IDS = {
    "pv_all": "pv_all",
    "fusion_nf_core": "fusion_nf_core",
    "fusion_nf_expectations": "fusion_nf_expectations",
}


def market_panel_data_id(
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    if DataFrequency(frequency) is not DataFrequency.DAILY:
        raise ValueError("unsupported_a_share_panel_frequency")
    return MARKET_PANEL_ID
