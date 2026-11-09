"""Independent mathematical checks for the small published-formula subset."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.engine import (
    calculate_factors_for_panel,
    factor_lookback_bars,
    factor_panel_input_columns,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.protocols.a_share import factor_catalog

FACTORS = (
    "alpha101_054",
    "alpha191_046",
    "qlib_rsqr_20d",
    "qlib_resi_20d",
    "qlib_sump_20d",
)


def _panel(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=len(close)),
            "adj_close": close,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def test_alpha101_54_sign_and_common_adjustment_cancel():
    panel = _panel([11.0]).assign(open=10.0, high=12.0, low=8.0)
    expected = -float(Fraction(3, 4) * Fraction(10, 11) ** 5)
    for scale in (1.0, 0.5, 1e70):
        scaled = panel.copy()
        scaled[["open", "high", "low", "close"]] *= scale
        actual = calculate_price_volume_factors(
            scaled, factor_columns=("alpha101_054",)
        )
        assert actual.alpha101_054.iloc[0] == pytest.approx(expected, rel=1e-14)
    for changes in ({"low": 12.0}, {"close": 0.0}, {"open": np.nan}):
        invalid = calculate_price_volume_factors(
            panel.assign(**changes), factor_columns=("alpha101_054",)
        )
        assert invalid.alpha101_054.isna().all()


def test_alpha191_46_four_means_not_endpoint_momentum():
    panel = _panel(np.arange(1.0, 25.0))
    actual = calculate_price_volume_factors(panel, factor_columns=("alpha191_046",))
    assert actual.alpha191_046.iloc[:-1].isna().all()
    # Last 3, 6, 12 and 24 prices have these independently known means.
    assert actual.alpha191_046.iloc[-1] == pytest.approx(
        (23.0 + 21.5 + 18.5 + 12.5) / (4 * 24)
    )
    invalid = calculate_price_volume_factors(
        panel.assign(adj_close=0.0), factor_columns=("alpha191_046",)
    )
    assert invalid.alpha191_046.isna().all()


def test_qlib_ols_has_intercept_and_current_residual():
    panel = _panel([100 + j * j for j in range(1, 21)])
    actual = calculate_price_volume_factors(
        panel, factor_columns=("qlib_rsqr_20d", "qlib_resi_20d")
    )
    assert actual[["qlib_rsqr_20d", "qlib_resi_20d"]].iloc[:-1].isna().all().all()
    # For x=1..20 and y=100+x^2: Sxx=665, Sxy=13965, Syy=310821.
    # The fitted line is 23+21*x, so the last residual is 500-443=57.
    assert actual.qlib_rsqr_20d.iloc[-1] == pytest.approx(
        float(Fraction(13965**2, 665 * 310821)), rel=1e-13
    )
    assert actual.qlib_resi_20d.iloc[-1] == pytest.approx(57 / 500, rel=1e-13)
    linear = calculate_price_volume_factors(
        _panel(np.arange(100.0, 120.0)),
        factor_columns=("qlib_rsqr_20d", "qlib_resi_20d"),
    )
    assert linear.qlib_rsqr_20d.iloc[-1] == pytest.approx(1.0)
    assert linear.qlib_resi_20d.iloc[-1] == pytest.approx(0.0, abs=1e-15)


def test_qlib_degenerate_and_near_constant_windows():
    for close in (np.full(20, 100.0), 100 + np.arange(20) * 1e-6):
        result = calculate_price_volume_factors(
            _panel(close), factor_columns=("qlib_rsqr_20d", "qlib_resi_20d")
        )
        assert result.qlib_rsqr_20d.isna().all()
        assert result.qlib_resi_20d.iloc[-1] == pytest.approx(0.0, abs=1e-15)
    near = calculate_price_volume_factors(
        _panel(100 + np.arange(20) * 1e-5), factor_columns=("qlib_rsqr_20d",)
    )
    assert near.qlib_rsqr_20d.iloc[-1] == pytest.approx(1.0)
    zero = _panel(np.arange(1.0, 21.0))
    zero.loc[19, "adj_close"] = 0.0
    assert (
        calculate_price_volume_factors(zero, factor_columns=("qlib_resi_20d",))
        .qlib_resi_20d.isna()
        .all()
    )


def test_qlib_sump_uses_twenty_differences_and_original_epsilon():
    increments = [3.0, -1.0, 4.0, -1.0] * 5
    panel = _panel([100.0] + list(100.0 + np.cumsum(increments)))
    result = calculate_price_volume_factors(panel, factor_columns=("qlib_sump_20d",))
    assert result.qlib_sump_20d.iloc[:20].isna().all()
    assert result.qlib_sump_20d.iloc[-1] == pytest.approx(35 / (45 + 1e-12))
    tiny = _panel(np.arange(21) * 1e-13)
    assert calculate_price_volume_factors(
        tiny, factor_columns=("qlib_sump_20d",)
    ).qlib_sump_20d.iloc[-1] == pytest.approx(2 / 3)
    flat = calculate_price_volume_factors(
        _panel(np.full(21, 100.0)), factor_columns=("qlib_sump_20d",)
    )
    assert flat.qlib_sump_20d.iloc[-1] == 0.0


@pytest.mark.parametrize(
    ("factor", "preceding"), zip(FACTORS, (0, 23, 19, 19, 20), strict=True)
)
def test_catalog_dependencies_timing_and_minimum_history(factor, preceding):
    definition = factor_catalog().get(f"price_volume_{factor}")
    assert definition.output_column == factor
    assert definition.family == "price_volume"
    assert definition.timing_semantics == "same_session_close_available_at_next_session"
    assert "https://" in definition.description
    assert factor_lookback_bars((definition,)) == preceding
    assert definition.input_columns == (
        ("open", "high", "low", "close") if factor == "alpha101_054" else ("adj_close",)
    )
    panel = _panel(100.0 + np.arange(35) ** 2)
    minimal = panel.loc[:, factor_panel_input_columns((definition,))]
    full = calculate_factors_for_panel(minimal, definitions=(definition,))
    trimmed = calculate_factors_for_panel(
        minimal.tail(preceding + 1), definitions=(definition,)
    )
    assert trimmed[factor].iloc[-1] == pytest.approx(full[factor].iloc[-1])
    if preceding:
        assert (
            calculate_factors_for_panel(
                minimal.tail(preceding), definitions=(definition,)
            )[factor]
            .isna()
            .all()
        )


@pytest.mark.parametrize("factor", FACTORS[1:])
def test_missing_open_session_is_not_compressed_or_partially_accepted(factor):
    panel = _panel(100 + np.arange(35) ** 2)
    panel.loc[20, "adj_close"] = np.nan
    result = calculate_price_volume_factors(
        panel, factor_columns=(factor,), min_fresh_fraction=0.1
    )
    assert result[factor].iloc[20:].isna().all()


def test_future_changes_output_slices_and_listing_lives_are_isolated():
    panel = _panel(100 + np.arange(60) ** 2)
    reference = calculate_price_volume_factors(panel, factor_columns=FACTORS)
    changed = panel.copy()
    changed.loc[40:, ["adj_close", "open", "high", "low", "close"]] *= 2
    actual = calculate_price_volume_factors(changed, factor_columns=FACTORS)
    pd.testing.assert_frame_equal(reference.iloc[:40], actual.iloc[:40])
    window = TimeWindow(
        panel.timestamp.iloc[30].date(), panel.timestamp.iloc[39].date()
    )
    sliced = calculate_price_volume_factors(
        changed, factor_columns=FACTORS, output_window=window
    )
    pd.testing.assert_frame_equal(sliced, reference.iloc[30:40].reset_index(drop=True))
    relisted = panel.assign(listing_date=pd.Timestamp("2024-01-01"))
    relisted.loc[30:, "listing_date"] = panel.timestamp.iloc[30]
    result = calculate_price_volume_factors(
        relisted, factor_columns=FACTORS, history_keys=("asset_id", "listing_date")
    )
    for factor, warmup in zip(FACTORS[1:], (23, 19, 19, 20), strict=True):
        assert result.loc[30 : 30 + warmup - 1, factor].isna().all()
        assert result.loc[30 + warmup, factor] == pytest.approx(
            reference.loc[30 + warmup, factor]
        )


def test_missing_values_affect_only_formulas_that_consume_them():
    panel = _panel(100 + np.arange(30) ** 2)
    baseline = calculate_price_volume_factors(panel, factor_columns=FACTORS)
    missing_open = panel.copy()
    missing_open.loc[29, "open"] = np.nan
    actual = calculate_price_volume_factors(missing_open, factor_columns=FACTORS)
    assert pd.isna(actual.alpha101_054.iloc[-1])
    pd.testing.assert_frame_equal(
        actual[list(FACTORS[1:])], baseline[list(FACTORS[1:])]
    )
    missing_adjusted = panel.copy()
    missing_adjusted.loc[29, "adj_close"] = np.nan
    actual = calculate_price_volume_factors(missing_adjusted, factor_columns=FACTORS)
    pd.testing.assert_series_equal(actual.alpha101_054, baseline.alpha101_054)
    assert actual.loc[29, list(FACTORS[1:])].isna().all()
    assert list(actual.columns) == ["asset_id", "timestamp", *FACTORS]
    assert len(actual) == len(panel)
