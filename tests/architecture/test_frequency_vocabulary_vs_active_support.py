from __future__ import annotations

from pathlib import Path

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
    from quant_research_platform.contracts.data import (
        market_bars_data_id as contract_market_bars_data_id,
    )
    from quant_research_platform.markets.implementations.a_share import (
        ids as market_ids,
    )
    from quant_research_platform.sources.csmar import ids as source_ids

    assert source_ids.SUPPORTED_MARKET_BAR_FREQUENCIES == (DataFrequency.DAILY,)
    assert contract_market_bars_data_id(DataFrequency.DAILY) == "market_bars_1d"
    assert market_ids.SUPPORTED_MARKET_PANEL_FREQUENCIES == (DataFrequency.DAILY,)
    assert market_ids.market_panel_data_id(DataFrequency.DAILY) == "market_panel_1d"
    assert market_ids.market_context_data_id(DataFrequency.DAILY) == "market_context_1d"
    assert not hasattr(source_ids, "CSMAR" + "_SOURCE_ID")
    assert not hasattr(market_ids, "A_SHARE" + "_MARKET_ID")


def test_regular_tests_do_not_hardcode_active_frequency_literals() -> None:
    allowed = {
        Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"),
        Path("tests/architecture/test_retired_surface_absent.py"),
    }
    forbidden = (
        "market_bars_1d",
        "market_panel_1d",
        "market_context_1d",
        '"1d"',
        "'1d'",
        '"1" + "d"',
        "'1' + 'd'",
    )
    findings: list[str] = []
    for path in sorted(Path("tests").rglob("*.py")):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                findings.append(f"{path.as_posix()}: {token}")

    assert findings == []
