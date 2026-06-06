from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.factors import FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.factors.price_volume import (
    CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_CORE_PRESET,
    PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS,
    PRICE_VOLUME_SMOKE_PRESET,
    PRICE_VOLUME_STANDARD_PRESET,
    SMOKE_PRICE_VOLUME_FACTORS,
    calculate_price_volume_factors,
    largest_price_volume_lookback_days,
    price_volume_factor_columns,
    price_volume_factor_definitions,
    price_volume_factor_input_columns,
    select_price_volume_feature_columns_from_schema,
)
from tests.helpers.synthetic_a_share import make_synthetic_factor_panel


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
            assert definition.factor_set is FactorSet.STANDARD
            assert definition.to_manifest()["factor_set"] == "standard"

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


def test_price_volume_feature_exclusion_is_canonical_key_only() -> None:
    assert PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS == frozenset(
        {"market_id", "asset_id", "timestamp"}
    )
    assert "asset" not in PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS
    assert not {
        "eligible",
        "can_enter",
        "can_exit",
        "is_st",
        "is_pt",
        "limit_status",
    }.intersection(PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS)

    selected, warnings = select_price_volume_feature_columns_from_schema(
        (
            "market_id",
            "asset_id",
            "timestamp",
            "mom_20d",
            "forward_return_20d",
        ),
        expected_columns=(
            "market_id",
            "asset_id",
            "timestamp",
            "mom_20d",
            "forward_return_20d",
        ),
    )

    assert selected == ("mom_20d",)
    assert warnings == (
        "excluded leakage/control column: market_id",
        "excluded leakage/control column: asset_id",
        "excluded leakage/control column: timestamp",
        "excluded leakage/control column: forward_return_20d",
    )


def test_price_volume_factors_handle_grouping_warmup_and_non_finite_edges() -> None:
    panel = make_synthetic_factor_panel(sessions=260)
    asset_a_dates = (
        panel.loc[panel["asset_id"].eq("asset_a"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    asset_b_dates = (
        panel.loc[panel["asset_id"].eq("asset_b"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[0]),
        "adj_close",
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[19]),
        ["amount", "prev_close"],
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[5]),
        ["total_market_cap", "float_market_cap"],
    ] = [0.0, -1.0]
    panel.loc[
        panel["asset_id"].eq("asset_b") & panel["timestamp"].eq(asset_b_dates[7]),
        "float_market_cap",
    ] = np.inf
    shuffled = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)

    factor_columns = (
        "mom_252d",
        "rev_252d",
        "amihud_20d",
        "amplitude_20d",
        "log_float_market_cap",
        "log_total_market_cap",
    )
    result = calculate_price_volume_factors(
        shuffled,
        factor_columns=factor_columns,
    )
    result = result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    asset_a = result[result["asset_id"].eq("asset_a")].reset_index(drop=True)
    asset_b = result[result["asset_id"].eq("asset_b")].reset_index(drop=True)
    expected_asset_b_mom_252d = (
        asset_b.loc[252, "adj_close"] / asset_b.loc[0, "adj_close"] - 1.0
    )

    assert asset_a.loc[:251, "mom_252d"].isna().all()
    assert pd.isna(asset_a.loc[252, "mom_252d"])
    assert pd.isna(asset_a.loc[252, "rev_252d"])
    assert asset_b.loc[:251, "mom_252d"].isna().all()
    assert asset_b.loc[252, "mom_252d"] == pytest.approx(
        expected_asset_b_mom_252d
    )
    assert asset_b.loc[252, "rev_252d"] == pytest.approx(
        -expected_asset_b_mom_252d
    )
    assert pd.isna(asset_a.loc[19, "amihud_20d"])
    assert pd.isna(asset_a.loc[19, "amplitude_20d"])
    assert pd.isna(asset_a.loc[5, "log_total_market_cap"])
    assert pd.isna(asset_a.loc[5, "log_float_market_cap"])
    assert pd.isna(asset_b.loc[7, "log_float_market_cap"])
    assert not np.isinf(result.loc[:, factor_columns].to_numpy(dtype="float64")).any()


def test_price_volume_factor_outputs_sanitize_non_finite_source_values() -> None:
    panel = make_synthetic_factor_panel(sessions=25, assets=("asset_a",))
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    denominator_date = ordered.loc[0, "timestamp"]
    rolling_window_end = ordered.loc[19, "timestamp"]

    panel.loc[panel["timestamp"].eq(denominator_date), "adj_close"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "amount"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "prev_close"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "turnover"] = np.inf
    panel.loc[panel["timestamp"].eq(rolling_window_end), "return_1d"] = np.inf
    panel.loc[panel["timestamp"].eq(rolling_window_end), "high"] = np.inf

    factor_columns = (
        "mom_20d",
        "rev_20d",
        "max_return_20d",
        "amihud_20d",
        "amplitude_20d",
        "intraday_range_20d",
        "turnover_chg_20d",
    )
    result = calculate_price_volume_factors(
        panel.sample(frac=1.0, random_state=17),
        factor_columns=factor_columns,
    ).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    assert pd.isna(result.loc[19, "amihud_20d"])
    assert pd.isna(result.loc[19, "amplitude_20d"])
    assert pd.isna(result.loc[19, "max_return_20d"])
    assert pd.isna(result.loc[19, "intraday_range_20d"])
    assert pd.isna(result.loc[20, "mom_20d"])
    assert pd.isna(result.loc[20, "rev_20d"])
    assert pd.isna(result.loc[20, "turnover_chg_20d"])
    assert not np.isinf(result.loc[:, factor_columns].to_numpy(dtype="float64")).any()


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
