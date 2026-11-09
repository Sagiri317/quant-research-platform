"""Historical extrema and closing-confirmed breaches with independent arithmetic."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)


def level_panel():
    prices = np.r_[60, 70, 100, np.full(17, 90), 80]
    return pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=21),
            "adj_close": prices,
            "close": prices,
            "high": np.r_[np.full(20, 110), 120],
            "low": np.r_[np.full(20, 50), 40],
        }
    )


def test_peak_drawdown_age_and_ordered_leg_retracement_by_hand():
    columns = ("peak_drawdown_20d", "peak_age_20d", "up_leg_retracement_20d")
    actual = calculate_price_volume_factors(level_panel(), factor_columns=columns)
    assert actual.iloc[:-1][list(columns)].isna().all(axis=None)
    assert actual.iloc[-1][list(columns)].to_numpy(float) == pytest.approx(
        [0.2, 18, 0.5]
    )
    panel = level_panel()
    panel.loc[3, ["adj_close", "close"]] = 100
    result = calculate_price_volume_factors(panel, factor_columns=columns)
    assert result.iloc[-1].peak_age_20d == 17
    assert result.iloc[-1].up_leg_retracement_20d == pytest.approx(0.5)
    panel["adj_close"] = np.arange(100, 79, -1)
    result = calculate_price_volume_factors(panel, factor_columns=columns)
    assert np.isnan(result.iloc[-1].up_leg_retracement_20d)
    assert result.iloc[-1].peak_drawdown_20d == pytest.approx(0.2)


def test_adjusted_support_and_failed_breach_are_known_only_after_close():
    columns = (
        "support_distance_20d",
        "resistance_distance_20d",
        "failed_up_breakout_20d",
        "failed_down_breakout_20d",
    )
    panel = level_panel()
    result = calculate_price_volume_factors(panel, factor_columns=columns)
    assert result.iloc[-1][list(columns)].to_numpy(float) == pytest.approx(
        [0.6, 80 / 110 - 1, 1 / 11, 0.2]
    )
    split = panel.copy()
    split.loc[10:, ["high", "low", "close"]] /= 2
    pd.testing.assert_frame_equal(
        result, calculate_price_volume_factors(split, factor_columns=columns)
    )
    split.loc[20, "adj_close"] = 130
    changed = calculate_price_volume_factors(split, factor_columns=columns)
    pd.testing.assert_frame_equal(result.iloc[:-1], changed.iloc[:-1])
    assert changed.iloc[-1].failed_up_breakout_20d == 0


def test_daily_amount_profile_interval_allocation_and_dependency_windows():
    columns = (
        "traded_amount_node_distance_60d",
        "traded_amount_concentration_60d",
        "traded_amount_above_price_60d",
    )
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=62),
            "adj_close": 1.02,
            "close": 1.02,
            "low": 1.0,
            "high": 1.02**2,
            "amount": 100.0,
        }
    )
    actual = calculate_price_volume_factors(panel, factor_columns=columns)
    upper_fraction = (1.02**2 - 1.02) / (1.02**2 - 1)
    assert actual.loc[:59, list(columns)].isna().all(axis=None)
    assert actual.loc[60, list(columns)].to_numpy(float) == pytest.approx(
        [-0.5 * np.log(1.02), upper_fraction, upper_fraction]
    )
    changed = panel.copy()
    changed.loc[60, ["amount", "high", "low"]] = np.nan
    result = calculate_price_volume_factors(changed, factor_columns=columns)
    # Today's missing profile inputs do not invalidate a complete prior profile.
    pd.testing.assert_series_equal(result.loc[60], actual.loc[60])
    assert result.loc[61, list(columns)].isna().all()
    future = panel.copy()
    future.loc[61, ["adj_close", "close", "high", "low"]] = 10000
    pd.testing.assert_frame_equal(
        actual.iloc[:61],
        calculate_price_volume_factors(future, factor_columns=columns).iloc[:61],
    )


def test_flat_bar_amount_profile_is_concentrated_and_zero_amount_undefined():
    column = "traded_amount_concentration_60d"
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=61),
            "adj_close": 2.0,
            "close": 2.0,
            "low": 2.0,
            "high": 2.0,
            "amount": 100,
        }
    )
    assert (
        calculate_price_volume_factors(panel, factor_columns=(column,))[column].iloc[-1]
        == 1
    )
    assert (
        calculate_price_volume_factors(
            panel.assign(amount=0), factor_columns=(column,)
        )[column]
        .isna()
        .all()
    )
