from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.factors import FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.factors.price_volume import (
    CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_CORE_PRESET,
    PRICE_VOLUME_SMOKE_PRESET,
    PRICE_VOLUME_STANDARD_PRESET,
    SMOKE_PRICE_VOLUME_FACTORS,
    calculate_price_volume_factors,
    largest_price_volume_lookback_days,
    price_volume_factor_columns,
    price_volume_factor_definitions,
    price_volume_factor_input_columns,
)


def test_price_volume_standard_preset_defines_catalog_contract() -> None:
    columns = price_volume_factor_columns(preset=PRICE_VOLUME_STANDARD_PRESET)

    assert columns
    assert "mom_252d" in columns
    assert "rev_252d" in columns
    assert "vol_252d" not in columns
    assert "forward_return_20d" not in columns
    assert price_volume_factor_columns() == columns
    assert price_volume_factor_columns(preset=PRICE_VOLUME_CORE_PRESET) == (
        CORE_PRICE_VOLUME_FACTORS
    )
    assert price_volume_factor_columns(preset=PRICE_VOLUME_SMOKE_PRESET) == (
        SMOKE_PRICE_VOLUME_FACTORS
    )
    with pytest.raises(ValueError, match="invalid_price_volume_preset"):
        price_volume_factor_columns(preset="full")


def test_price_volume_factor_definitions_are_output_and_input_driven() -> None:
    definitions = price_volume_factor_definitions(
        market_id=MarketId("a_share"),
        required_processed_data_id="daily_panel",
    )
    columns = price_volume_factor_columns()
    by_id = {definition.factor_id: definition for definition in definitions}

    assert tuple(definition.output_column for definition in definitions) == columns
    assert set(by_id) == set(columns)
    for definition in definitions:
        assert definition.factor_id == definition.output_column
        assert definition.input_columns == price_volume_factor_input_columns(
            definition.factor_id
        )
        assert definition.base_input_columns == definition.input_columns
        assert definition.required_processed_data_id == "daily_panel"
        assert definition.family == "price_volume"
        if definition.factor_id in SMOKE_PRICE_VOLUME_FACTORS:
            assert definition.factor_set is FactorSet.SMOKE
        elif definition.factor_id in CORE_PRICE_VOLUME_FACTORS:
            assert definition.factor_set is FactorSet.CORE
        else:
            assert definition.factor_set is FactorSet.FULL

    assert by_id["rev_1d"].input_columns == ("return_1d",)
    assert by_id["mom_20d"].input_columns == ("adj_close",)
    assert by_id["amihud_20d"].input_columns == ("return_1d", "amount")
    assert by_id["amplitude_20d"].input_columns == (
        "high",
        "low",
        "prev_close",
    )


def test_price_volume_compute_uses_canonical_keys_and_prunes_outputs() -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=260))
    smoke_columns = price_volume_factor_columns(preset="smoke")

    result = calculate_price_volume_factors(panel, factor_columns=smoke_columns)

    assert set(smoke_columns).issubset(result.columns)
    assert not {
        "mom_252d",
        "rev_252d",
        "vol_252d",
        "skew_252d",
        "kurt_252d",
        "amihud_252d",
        "drawdown_252d",
    }.intersection(result.columns)
    assert result.sort_values(["asset_id", "timestamp"]).index.tolist() == list(
        result.index
    )


def test_price_volume_compute_fails_closed_when_required_input_missing() -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5)).drop(
        columns=["amount"]
    )

    with pytest.raises(ValueError, match="missing_price_volume_input_columns: amount"):
        calculate_price_volume_factors(panel, factor_columns=("amihud_20d",))


def test_price_volume_largest_lookback_is_derived_from_requested_columns() -> None:
    columns = price_volume_factor_columns()

    assert largest_price_volume_lookback_days(("mom_3d",)) == 3
    assert largest_price_volume_lookback_days(()) == 0
    assert largest_price_volume_lookback_days(columns) == max(
        int(token[:-1])
        for column in columns
        for token in column.split("_")
        if token.endswith("d") and token[:-1].isdigit()
    )


def _factor_panel_for_dates(dates) -> pd.DataFrame:
    rows = []
    for offset, date in enumerate(pd.to_datetime(dates), start=1):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "asset_a",
                "timestamp": pd.Timestamp(date),
                "adj_close": float(100 + offset),
                "return_1d": 0.001 * offset,
                "amount": 10000.0 + offset,
                "turnover": 0.01 + offset / 100000.0,
                "high": float(101 + offset),
                "low": float(99 + offset),
                "prev_close": float(99 + offset),
                "total_market_cap": float(1_000_000 + offset),
                "float_market_cap": float(500_000 + offset),
            }
        )
    return pd.DataFrame(rows)
