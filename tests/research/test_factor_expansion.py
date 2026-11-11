"""Independent examples for causal factor additions and consumed dependencies."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import _configured_spec
from quant_research_platform.research.factors.market_structure import (
    calculate as structure_factors,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    _breakout_episode,
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.style_structure import (
    calculate as style_factors,
)


@pytest.mark.parametrize("direction", (1, -1))
def test_price_acceleration_matches_quadratic_derivative_and_price_scaling(direction):
    x = np.arange(20, dtype=float)
    log_price = direction * (0.001 * x + 0.0001 * x * x)
    price = np.exp(log_price)
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=20),
            "adj_close": price,
        }
    )
    names = ("price_up_acceleration_20d", "price_down_acceleration_20d")
    result = calculate_price_volume_factors(panel, factor_columns=names)
    expected = 0.0002 / np.sqrt(np.mean(np.diff(log_price) ** 2))
    assert result[names[0 if direction > 0 else 1]].iloc[-1] == pytest.approx(expected)
    assert result[names[1 if direction > 0 else 0]].iloc[-1] == 0
    scaled = calculate_price_volume_factors(
        panel.assign(adj_close=price * 100), factor_columns=names
    )
    np.testing.assert_allclose(result[list(names)], scaled[list(names)], atol=1e-12)


def test_monthly_return_acceleration_uses_twelve_completed_calendar_months():
    dates = pd.date_range("2023-01-31", periods=13, freq="ME").append(
        pd.DatetimeIndex(["2024-02-01", "2024-02-02"])
    )
    prices = np.r_[
        100.0, 100 * np.cumprod(np.r_[np.full(6, 1.01), np.full(6, 1.03)]), 999.0, 1.0
    ]
    panel = pd.DataFrame(dict(asset_id="a", timestamp=dates, adj_close=prices))
    name = "return_acceleration_6m6m"
    result = calculate_price_volume_factors(panel, factor_columns=(name,))
    assert result[name].iloc[:13].isna().all()
    np.testing.assert_allclose(result[name].iloc[-2:], 0.02, atol=1e-14)


def test_breakout_keeps_frozen_boundary_and_restarts_after_reentry():
    price = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 2.0, 1.0])
    direction, age, followthrough = _breakout_episode(price, 2)
    np.testing.assert_allclose(direction[2:], [1, 1, 1, -1, -1])
    np.testing.assert_allclose(age[2:], [1, 2, 3, -1, -2])
    np.testing.assert_allclose(followthrough[2:], [0, 1 / 3, 1 / 6, 0, -0.5])
    assert np.isnan(age[:2]).all()
    extended = _breakout_episode(np.r_[price, 1000], 2)
    for before, after in zip((direction, age, followthrough), extended, strict=True):
        np.testing.assert_allclose(before, after[:-1], equal_nan=True)


def test_breakout_price_view_needs_no_activity_input():
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.date_range("2024-01-01", periods=24),
            "adj_close": np.arange(1.0, 25.0),
        }
    )
    result = calculate_price_volume_factors(
        panel,
        factor_columns=(
            "signed_breakout_duration_20d",
            "signed_breakout_followthrough_20d",
        ),
    )
    assert result.iloc[-1].signed_breakout_duration_20d == 4
    assert result.iloc[-1].signed_breakout_followthrough_20d == 24 / 21 - 1


def test_style_size_needs_no_other_styles_and_uses_prior_exposure():
    dates = pd.bdate_range("2024-01-01", periods=24)
    panel = pd.DataFrame(
        [
            {
                "asset_id": str(asset),
                "timestamp": date,
                "log_total_market_cap": float(asset),
                "period_return": 0.01 * asset,
                "is_trading": True,
                "is_a_share_scope": True,
            }
            for date in dates
            for asset in range(5)
        ]
    )
    name = "style_size_return_trend_20d"
    expected = style_factors(panel, factor_columns=(name,))
    np.testing.assert_allclose(
        expected.loc[expected.timestamp.eq(dates[20]), name], 0.8
    )
    changed = panel.copy()
    changed.loc[changed.timestamp.ge(dates[20]), "log_total_market_cap"] *= -1
    actual = style_factors(changed, factor_columns=(name,))
    pd.testing.assert_frame_equal(
        expected.loc[expected.timestamp.le(dates[20])],
        actual.loc[actual.timestamp.le(dates[20])],
    )
    spec = _configured_spec(
        processed_catalog().get("style_structure_factors"), {"factor_columns": [name]}
    )
    assert spec.processed_input_data_ids == ("market_panel_1d", "price_volume_facts")
    missing = panel.copy()
    missing["period_return"] = missing.period_return.astype("Float64")
    missing.loc[
        missing.timestamp.eq(dates[10]) & missing.asset_id.eq("4"), "period_return"
    ] = pd.NA
    actual = style_factors(missing, factor_columns=(name,))
    np.testing.assert_allclose(actual.loc[actual.timestamp.eq(dates[20]), name], 0.76)


@pytest.mark.parametrize("halt_return", (0.0, np.nan))
def test_style_uses_known_holding_returns_and_prior_scope(halt_return):
    dates = pd.bdate_range("2024-01-01", periods=21)
    panel = pd.DataFrame(
        [
            dict(
                asset_id=asset,
                timestamp=date,
                log_total_market_cap=size,
                period_return=value,
                is_a_share_scope=in_scope,
                is_trading=trading,
            )
            for date in dates
            for asset, size, value, in_scope, trading in (
                ("small", 0.0, halt_return, True, False),
                ("large", 1.0, 0.03, True, True),
                ("outside", -1.0, 0.5, False, True),
            )
        ]
    )
    name = "style_size_return_trend_20d"
    result = style_factors(panel, factor_columns=(name,))
    last = result.loc[result.timestamp.eq(dates[-1]), name]
    if np.isfinite(halt_return):
        # Twenty fixed high-minus-low baskets: 20 * (0.03 - 0).
        np.testing.assert_allclose(last, 0.6)
    else:
        assert last.isna().all()


def test_limit_breadth_needs_no_prices_and_keeps_unknown_out_of_denominator():
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "limit_status": [1.0, -1.0, 0.0, np.nan],
            "filling_flag": 0,
            "is_a_share_scope": True,
        }
    )
    actual = structure_factors(
        frame,
        factor_columns=(
            "market_limit_up_participation",
            "market_limit_down_participation",
        ),
    )
    np.testing.assert_allclose(actual.market_limit_up_participation, 1 / 3)
    np.testing.assert_allclose(actual.market_limit_down_participation, 1 / 3)


def test_closing_failed_break_is_not_a_retest_of_the_frozen_level():
    price = np.r_[np.full(20, 95.0), 110.0, 115.0, 105.0, 99.0, 98.0]
    panel = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=len(price)),
            "adj_close": price,
            "close": price,
            "high": np.r_[np.full(20, 100.0), 111.0, 116.0, 106.0, 100.0, 99.0],
        }
    )
    names = ("failed_close_up_breakout_3_20d", "failed_close_up_breakout_5_20d")
    actual = calculate_price_volume_factors(panel, factor_columns=names)
    assert actual.loc[22, list(names)].eq(0).all()
    np.testing.assert_allclose(
        actual.loc[23, list(names)].to_numpy(float), [0.01, 0.01]
    )
    assert actual.loc[24, list(names)].eq(0).all()
    assert actual.loc[:19, list(names)].isna().all(axis=None)
    missing_unused_high = panel.copy()
    missing_unused_high.loc[21, "high"] = np.nan
    pd.testing.assert_frame_equal(
        actual,
        calculate_price_volume_factors(missing_unused_high, factor_columns=names),
    )
    late = panel.copy()
    late.loc[23, ["adj_close", "close"]] = 102.0
    late.loc[24, ["adj_close", "close"]] = 99.0
    result = calculate_price_volume_factors(late, factor_columns=names)
    assert result.loc[24, names[0]] == 0
    assert np.isclose(result.loc[24, names[1]], 0.01)
