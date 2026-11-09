"""Independent arithmetic for compression, absorption and clearing proxies."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.engine import (
    calculate_factors_for_panel,
    factor_lookback_bars,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_THEME_DESCRIPTIONS,
)
from quant_research_platform.research.protocols.a_share import factor_catalog


def panel(prices):
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=len(prices)).as_unit(
                "ns"
            ),
            "adj_close": prices,
            "adj_open": prices,
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "amount": 1000.0,
            "period_return": pd.Series(prices).pct_change(fill_method=None),
            "turnover": 1.0,
        }
    )


def test_compression_retains_direction_and_unknown_flat_denominator():
    frame = panel(np.ones(60))
    frame["period_return"] = [-0.02, 0.02] * 20 + [0.001] * 20
    names = ("volatility_compression_20_60", "compression_trend_20_60")
    actual = calculate_price_volume_factors(frame, factor_columns=names)
    assert actual[list(names)].iloc[:-1].isna().all(axis=None)
    assert actual.volatility_compression_20_60.iloc[-1] == pytest.approx(0, abs=1e-14)
    assert actual.compression_trend_20_60.iloc[-1] == pytest.approx(1)
    negative = frame.assign(period_return=-frame.period_return)
    assert calculate_price_volume_factors(
        negative, factor_columns=names
    ).compression_trend_20_60.iloc[-1] == pytest.approx(-1)
    flat = frame.assign(period_return=0.0)
    assert (
        calculate_price_volume_factors(flat, factor_columns=names)[list(names)]
        .isna()
        .all(axis=None)
    )


@pytest.mark.parametrize(
    "reverse,expected", [(False, Fraction(1, 20)), (True, Fraction(-1, 20))]
)
def test_breakout_is_signed_excess_beyond_prior_range_per_path(reverse, expected):
    prices = np.arange(1.0, 22.0)
    if reverse:
        prices = prices[::-1]
    frame = panel(prices)
    actual = calculate_price_volume_factors(
        frame, factor_columns=("breakout_efficiency_20d",)
    )
    assert actual.breakout_efficiency_20d.iloc[:20].isna().all()
    assert actual.breakout_efficiency_20d.iloc[-1] == pytest.approx(float(expected))
    frame.loc[5, "adj_close"] = np.nan
    assert (
        calculate_price_volume_factors(
            frame, factor_columns=("breakout_efficiency_20d",)
        )
        .breakout_efficiency_20d.isna()
        .all()
    )


def test_absorption_uses_prior_activity_and_return_scale_without_an_epsilon():
    frame = panel(np.ones(21)).assign(period_return=[0.02] * 20 + [0.01])
    frame.loc[20, "turnover"] = 2.0
    column = "turnover_absorption_20d"
    actual = calculate_price_volume_factors(frame, factor_columns=(column,))
    assert actual[column].iloc[-1] == pytest.approx(1.0)
    frame.loc[20, "period_return"] = 0.04
    assert calculate_price_volume_factors(frame, factor_columns=(column,))[column].iloc[
        -1
    ] == pytest.approx(-2)
    frame.loc[:19, "period_return"] = 0.0
    assert (
        calculate_price_volume_factors(frame, factor_columns=(column,))[column]
        .isna()
        .all()
    )


def test_pressure_rebound_and_impact_have_separate_economic_units():
    frame = panel([100.0] * 19 + [90.0, 95.0])
    frame.loc[20, "turnover"] = 2.0
    columns = (
        "drawdown_turnover_pressure_20d",
        "drawdown_rebound_20d",
        "price_impact_efficiency_20d",
    )
    result = calculate_price_volume_factors(frame, factor_columns=columns)
    # A 5% drawdown, doubled turnover, and a current 90->95 return of 1/18.
    assert result[columns[0]].iloc[-1] == pytest.approx(float(Fraction(1, 10)))
    assert result[columns[1]].iloc[-1] == pytest.approx(float(Fraction(1, 180)))
    # The 20-session close return is -1/20, against 19*1+2 turnover units.
    assert result[columns[2]].iloc[-1] == pytest.approx(float(Fraction(-1, 420)))
    frame.loc[0, "period_return"] = np.nan
    assert calculate_price_volume_factors(frame, factor_columns=columns)[
        columns[1]
    ].iloc[-1] == pytest.approx(float(Fraction(1, 180)))


def test_theme_catalog_windows_future_slices_and_listing_lives():
    names = tuple(PRICE_VOLUME_THEME_DESCRIPTIONS)
    definitions = tuple(d for d in factor_catalog().list() if d.output_column in names)
    assert len(definitions) == len(names)
    assert factor_lookback_bars(definitions) == 252
    assert all(
        d.timing_semantics == "same_session_close_available_at_next_session"
        for d in definitions
    )
    frame = panel(np.linspace(100, 120, 85)).assign(
        listing_date=pd.Timestamp("2020-01-01")
    )
    original = calculate_factors_for_panel(frame, definitions=definitions)
    changed = frame.copy()
    changed.loc[75:, ["adj_close", "turnover", "period_return"]] = [200.0, 50.0, -0.8]
    pd.testing.assert_frame_equal(
        original.iloc[:75],
        calculate_factors_for_panel(changed, definitions=definitions).iloc[:75],
    )
    sliced = calculate_price_volume_factors(
        frame,
        factor_columns=names,
        output_window=TimeWindow(frame.timestamp.iloc[70], frame.timestamp.iloc[74]),
    )
    expected = original.iloc[70:75].drop(columns="market_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(sliced.loc[:, expected.columns], expected)
    frame.loc[75:, "listing_date"] = pd.Timestamp("2024-04-16")
    reset = calculate_price_volume_factors(
        frame, factor_columns=names, history_keys=("asset_id", "listing_date")
    )
    long_windows = [
        d.output_column for d in definitions if factor_lookback_bars((d,)) >= 10
    ]
    assert reset.loc[75:, long_windows].isna().all(axis=None)
    assert np.isnan(reset.opening_gap.iloc[75])
    assert reset.intraday_return.iloc[75] == 0


@pytest.mark.parametrize("window", [60, 252])
def test_position_interactions_use_pre_jump_position_and_separate_input_windows(window):
    frame = panel([20.0] * (window - 1) + [10.0, 15.0])
    frame["turnover"] = np.nan
    frame["period_return"] = np.nan
    frame.loc[window - 20 : window - 1, "turnover"] = 1.0
    frame.loc[window - 20 : window - 1, "period_return"] = [-0.01, 0.01] * 10
    frame.loc[window, ["turnover", "period_return"]] = [2.0, 0.02]
    columns = (
        f"price_position_{window}d",
        "relative_turnover_20d",
        "return_shock_1_20",
        f"low_position_turnover_upshock_{window}d",
        f"high_position_turnover_downshock_{window}d",
    )
    actual = calculate_price_volume_factors(frame, factor_columns=columns)
    shock = 2 * (19 / 20) ** 0.5
    assert actual[columns[0]].iloc[-1] == pytest.approx(0.5)
    assert actual[columns[1]].iloc[-1] == pytest.approx(2)
    assert actual[columns[2]].iloc[-1] == pytest.approx(shock)
    # Yesterday was at the bottom of its range, even though today's price jumped.
    assert actual[columns[3]].iloc[-1] == pytest.approx(shock)
    assert actual[columns[4]].iloc[-1] == 0
    frame.loc[window, "period_return"] = -0.02
    frame.loc[window - 1, "adj_close"] = 20
    frame.loc[0, "adj_close"] = 10
    actual = calculate_price_volume_factors(frame, factor_columns=columns)
    assert actual[columns[3]].iloc[-1] == 0
    assert actual[columns[4]].iloc[-1] == pytest.approx(shock)


def test_opening_gap_and_realized_recovery_use_consistent_adjusted_prices():
    frame = panel([100, 95, 105]).assign(adj_open=[100, 90, 110])
    columns = ("opening_gap", "intraday_return", "down_gap_recovery", "up_gap_reversal")
    actual = calculate_price_volume_factors(frame, factor_columns=columns)
    assert actual.opening_gap.iloc[1] == pytest.approx(-0.1)
    assert actual.intraday_return.iloc[1] == pytest.approx(float(Fraction(1, 18)))
    assert actual.down_gap_recovery.iloc[1] == pytest.approx(float(Fraction(1, 180)))
    assert actual.up_gap_reversal.iloc[2] == pytest.approx(float(Fraction(3, 418)))
    adjusted_scale = frame.assign(
        adj_open=frame.adj_open / 2, adj_close=frame.adj_close / 2
    )
    pd.testing.assert_frame_equal(
        actual, calculate_price_volume_factors(adjusted_scale, factor_columns=columns)
    )
