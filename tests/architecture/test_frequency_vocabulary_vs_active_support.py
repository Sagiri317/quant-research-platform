from __future__ import annotations

import pytest


def test_data_frequency_helpers_define_reserved_vocabulary_and_active_support() -> None:
    from quant_research_platform.contracts.data import (
        DataFrequency,
        data_frequency_slug,
        horizon_unit_suffix,
        normalize_data_frequency,
        require_supported_data_frequency,
    )

    assert {item.name for item in DataFrequency} >= {"DAILY", "MINUTE", "TICK"}
    assert normalize_data_frequency("1d") is DataFrequency.DAILY
    assert data_frequency_slug(DataFrequency.DAILY) == "1d"
    assert horizon_unit_suffix(DataFrequency.DAILY) == "d"
    with pytest.raises(ValueError, match="unsupported_data_frequency"):
        require_supported_data_frequency(
            DataFrequency.MINUTE,
            supported=(DataFrequency.DAILY,),
            context="active profile",
        )


def test_current_source_and_market_ids_are_frequency_derived() -> None:
    from quant_research_platform.contracts.data import DataFrequency
    from quant_research_platform.markets.implementations.a_share import (
        ids as market_ids,
    )
    from quant_research_platform.sources.csmar import ids as source_ids

    assert source_ids.SUPPORTED_MARKET_BAR_FREQUENCIES == (DataFrequency.DAILY,)
    assert source_ids.market_bars_data_id(DataFrequency.DAILY) == "market_bars_1d"
    assert market_ids.SUPPORTED_MARKET_PANEL_FREQUENCIES == (DataFrequency.DAILY,)
    assert market_ids.MARKET_PANEL_ID == "market_panel_1d"
