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


@pytest.mark.parametrize("window", (20, 60, 252))
@pytest.mark.parametrize("side", ("support", "resistance"))
def test_rejection_requires_touch_without_crossing_known_reference(window, side):
    prices = np.full(window + 1, 100.0)
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=window + 1),
            "adj_close": prices,
            "close": prices,
            "low": 90.0,
            "high": 110.0,
        }
    )
    column = f"{side}_rejection_{window}d"
    extreme, level, touch, cross = (
        ("low", 90.0, 90.5, 89.9)
        if side == "support"
        else ("high", 110.0, 109.5, 110.1)
    )
    panel.loc[window, extreme] = touch
    actual = calculate_price_volume_factors(panel, factor_columns=(column,))
    assert actual[column].iloc[:-1].isna().all()
    assert actual[column].iloc[-1] == pytest.approx(9.5 / level)
    pd.testing.assert_frame_equal(
        actual,
        calculate_price_volume_factors(
            panel.drop(columns="high" if side == "support" else "low"),
            factor_columns=(column,),
        ),
    )
    split = panel.copy()
    split.loc[window // 2 :, ["close", "low", "high"]] /= 2
    pd.testing.assert_frame_equal(
        actual, calculate_price_volume_factors(split, factor_columns=(column,))
    )
    panel.loc[window, extreme] = cross
    assert (
        calculate_price_volume_factors(panel, factor_columns=(column,))[column].iloc[-1]
        == 0
    )
    panel.loc[window, extreme] = np.nan
    assert pd.isna(
        calculate_price_volume_factors(panel, factor_columns=(column,))[column].iloc[-1]
    )


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
    panel.loc[60, "adj_close"] = np.nan
    assert (
        calculate_price_volume_factors(panel, factor_columns=(column,))[column].iloc[-1]
        == 1
    )


@pytest.mark.parametrize("window", (20, 60, 252))
@pytest.mark.parametrize("side,unused", [("support", "high"), ("resistance", "low")])
def test_level_uses_only_the_relevant_side_and_prior_reference_fields(
    window, side, unused
):
    panel = pd.concat([level_panel().iloc[[0]]] * (window + 1), ignore_index=True)
    panel.timestamp = pd.bdate_range("2024-01-02", periods=window + 1)
    factor = f"{side}_distance_{window}d"
    expected = calculate_price_volume_factors(panel, factor_columns=(factor,))
    panel = panel.drop(columns=unused)
    panel.loc[window, ["close", "low" if side == "support" else "high"]] = np.nan
    actual = calculate_price_volume_factors(panel, factor_columns=(factor,))
    pd.testing.assert_frame_equal(actual, expected)


def test_peak_age_does_not_consume_current_price_or_unrelated_ohlc():
    panel = level_panel().drop(columns=["high", "low", "close"])
    factor = "peak_age_20d"
    panel.loc[20, "adj_close"] = np.nan
    actual = calculate_price_volume_factors(panel, factor_columns=(factor,))
    assert actual[factor].iloc[-1] == 18


@pytest.mark.parametrize("window", (20, 60, 252))
@pytest.mark.parametrize("side", ("up", "down"))
def test_retest_freezes_breakout_level_and_checks_only_observed_path(window, side):
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=window + 6),
            "adj_close": 100.0,
            "close": 100.0,
            "low": 90.0,
            "high": 110.0,
        }
    )
    upward = side == "up"
    panel.loc[window:, ["adj_close", "close"]] = 112 if upward else 88
    panel.loc[window:, ["high", "low"]] = [113.0, 111.0] if upward else [89.0, 87.0]
    touch_side, level = ("low", 110.0) if upward else ("high", 90.0)
    panel.loc[window + 5, touch_side] = 110.5 if upward else 89.5
    factor = f"{side}_break_retest_5_{window}d"
    actual = calculate_price_volume_factors(panel, factor_columns=(factor,))
    assert actual[factor].iloc[-1] == pytest.approx(1.5 / level)
    # Today's opposite side and past touches do not define this frozen reference.
    independent = panel.copy()
    independent.loc[: window + 4, touch_side] = np.nan
    independent.loc[window + 5, "high" if upward else "low"] = np.nan
    independent.loc[window + 4, ["close", "high" if upward else "low"]] = np.nan
    pd.testing.assert_frame_equal(
        actual, calculate_price_volume_factors(independent, factor_columns=(factor,))
    )
    split = panel.copy()
    split.loc[window // 2 :, ["close", "high", "low"]] /= 2
    pd.testing.assert_frame_equal(
        actual, calculate_price_volume_factors(split, factor_columns=(factor,))
    )
    panel.loc[window + 2, ["adj_close", "close"]] = 109 if upward else 91
    assert (
        calculate_price_volume_factors(panel, factor_columns=(factor,))[factor].iloc[-1]
        == 0
    )


def test_newer_known_retest_supersedes_only_older_unknown_candidate():
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=26),
            "adj_close": 99.0,
            "close": 99.0,
            "high": 100.0,
            "low": 98.0,
        }
    )
    frame.loc[24, ["adj_close", "close", "high"]] = [110.0, 110.0, 112.0]
    frame.loc[25, ["adj_close", "close", "low"]] = [105.0, 105.0, 100.5]
    column = "up_break_retest_5_20d"
    baseline = calculate_price_volume_factors(frame, factor_columns=(column,))
    assert baseline[column].iloc[-1] == pytest.approx(0.045)
    frame.loc[0, "high"] = np.nan
    pd.testing.assert_frame_equal(
        baseline, calculate_price_volume_factors(frame, factor_columns=(column,))
    )
    frame.loc[4:24, "high"] = np.nan
    assert pd.isna(
        calculate_price_volume_factors(frame, factor_columns=(column,))[column].iloc[-1]
    )


@pytest.mark.parametrize("side", ("up", "down"))
def test_known_failed_new_candidate_does_not_mask_older_held_level(side):
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=26),
            "adj_close": 99.0,
            "close": 99.0,
            "high": 100.0,
            "low": 98.0,
        }
    )
    frame.loc[20:, "adj_close"] = [105.0, 104.0, 104.0, 104.0, 108.0, 105.0]
    frame["close"] = frame.adj_close
    frame.loc[20:, "high"] = [106.0, 105.0, 105.0, 105.0, 110.0, 106.0]
    frame.loc[25, "low"] = 100.5
    reference = "high"
    if side == "down":
        frame[["adj_close", "close"]] = 200 - frame[["adj_close", "close"]]
        frame[["high", "low"]] = 200 - frame[["low", "high"]].to_numpy()
        reference = "low"
    column = f"{side}_break_retest_5_20d"
    baseline = calculate_price_volume_factors(frame, factor_columns=(column,))
    assert baseline[column].iloc[-1] == pytest.approx(0.045)
    frame.loc[23, reference] = np.nan
    pd.testing.assert_frame_equal(
        baseline, calculate_price_volume_factors(frame, factor_columns=(column,))
    )
