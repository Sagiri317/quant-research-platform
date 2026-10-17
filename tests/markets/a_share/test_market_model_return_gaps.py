from fractions import Fraction
from functools import cache

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.market_model import MarketModelRiskState
from quant_research_platform.research.portfolios.risk import (
    FactorRiskState,
    preprocess_risk_styles,
)
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)


@pytest.mark.parametrize("missing_column", ("period_return", "float_market_cap"))
def test_native_market_model_gap_persists_and_recovers_complete_windows(
    tmp_path,
    missing_column,
):
    dates = pd.bdate_range("2020-01-02", periods=953).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    for name in ("stock_master", "stock_info_annual"):
        frames[name]["listing_date"] = dates[0]
    bars = frames["market_bars_1d"]
    bars["period_return"] = np.sin(np.arange(len(bars)) * 0.23) * 0.01
    gap = 640
    bars.loc[
        bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[gap]),
        missing_column,
    ] = np.nan
    inputs = _write_inputs(tmp_path / "source", frames)
    for name in ("market_panel_1d", "market_model_factors"):
        inputs[name] = _prepare(
            tmp_path / "prepared", name, inputs, dates[0], dates[-1]
        )

    def saved(data_id):
        return read_dataset(inputs[data_id])

    panel, factors = saved("market_panel_1d"), saved("market_model_factors")
    keys = ["market_id", "asset_id", "timestamp"]
    pd.testing.assert_frame_equal(factors[keys], panel[keys])
    assert len(factors) == 3 * len(dates)
    assert (
        panel.loc[
            panel.asset_id.eq("000001") & panel.timestamp.eq(dates[gap]),
            missing_column,
        ]
        .isna()
        .all()
    )
    styles = ["beta_252d", "residual_volatility_60d"]
    assert factors.loc[factors.timestamp.eq(dates[gap - 1]), styles].notna().all().all()
    # A single unknown session advances the fixed window without discarding
    # sufficient historical pairs or the other 59 valid residual samples.
    assert factors.loc[factors.timestamp.ge(dates[gap]), styles].notna().all().all()
    risk = FactorRiskState(style_columns=tuple(styles))
    for day in (gap - 1, gap, gap + 310, gap + 311):
        cross = factors.loc[factors.timestamp.eq(dates[day])].assign(
            industry_standard_id="CSRC2012",
            industry_level1_code="A",
            total_market_cap=1000000.0,
            eligible=True,
        )
        prepared = risk.prepare_exposures(cross)
        pd.testing.assert_frame_equal(prepared[styles], cross[styles])
        processed = preprocess_risk_styles(prepared, style_columns=tuple(styles))
        exposures = risk.current_exposures(processed.dropna(subset=styles))
        assert len(exposures) == 3
    # Exact independent equal-cap regression oracle before and after the barrier.
    returns = bars.pivot(
        index="timestamp", columns="source_asset_id", values="period_return"
    )
    market = [
        float(sum(Fraction(value) for value in row) / 3)
        if np.isfinite(row).all()
        else np.nan
        for row in returns.to_numpy()
    ]
    if missing_column == "float_market_cap":
        # The missing current cap leaves today's lagged proxy intact but
        # prevents next-session normalization over only the other members.
        market[gap + 1] = np.nan

    @cache
    def exact_window(asset, end):
        # Adjacent requested dates reuse the identical exact-rational oracle.
        paired = [
            (a, b)
            for a, b in zip(
                market[end - 251 : end + 1],
                returns[asset].iloc[end - 251 : end + 1],
                strict=True,
            )
            if np.isfinite(a) and np.isfinite(b)
        ]
        x = [Fraction(a) for a, _ in paired]
        y = [Fraction(b) for _, b in paired]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        slope = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sum(
            (a - mx) ** 2 for a in x
        )
        residual = (
            float(
                Fraction(returns[asset].iloc[end])
                - my
                - slope * (Fraction(market[end]) - mx)
            )
            if np.isfinite(market[end]) and np.isfinite(returns[asset].iloc[end])
            else None
        )
        return float(slope), residual

    for day in (gap - 1, gap, gap + 1, gap + 311):
        for asset in returns:
            windows = [exact_window(asset, end) for end in range(day - 59, day + 1)]
            slope = windows[-1][0]
            residuals = [residual for _, residual in windows if residual is not None]
            row = factors.loc[
                factors.timestamp.eq(dates[day]) & factors.asset_id.eq(asset)
            ].iloc[0]
            assert row.beta_252d == pytest.approx(float(slope), rel=1e-14)
            assert row.residual_volatility_60d == pytest.approx(
                np.std(residuals), rel=1e-14
            )


@pytest.mark.parametrize("lifecycle", ("new", "relisted", "ineligible"))
def test_missing_return_without_required_weight_does_not_poison_all_a(lifecycle):
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.01,
            float_market_cap=100.0,
            eligible=True,
        )
    )
    for value in (0.01, 0.02, 0.03):
        state.observe(panel.assign(period_return=value))
    panel["period_return"] = [0.04, np.nan]
    if lifecycle == "new":
        panel.loc[1, "asset_id"] = "new"
    elif lifecycle == "relisted":
        panel.loc[1, "listing_date"] = pd.Timestamp("2020-01-01")
    else:
        panel.loc[1, "eligible"] = False
        # Establish a real exclusion, rather than inferring it from missing returns.
        panel["is_st"] = [False, True]
    result = state.observe(panel)
    assert result.beta["a"] == 1.0
    assert result.residual_volatility["a"] == 0.0
    assert np.isnan(result.beta.iloc[1])


def test_required_missing_return_poisoning_survives_market_closure():
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            market_type=[1, 4],
            period_return=0.01,
            float_market_cap=100.0,
            eligible=True,
        )
    )
    for value in (0.01, 0.02, 0.03):
        state.observe(
            panel.assign(period_return=value), open_market_types=frozenset({1, 4})
        )
    state.observe(
        panel.iloc[:1].assign(period_return=np.nan), open_market_types=frozenset({1})
    )
    result = state.observe(
        panel.assign(period_return=0.04), open_market_types=frozenset({1, 4})
    )
    assert result.beta.isna().all()
    assert result.residual_volatility.isna().all()
    result = state.observe(
        panel.assign(period_return=0.05), open_market_types=frozenset({1, 4})
    )
    assert result.beta["a"] == 1.0
    assert np.isnan(result.beta["b"])


@pytest.mark.parametrize(
    "column,value,error",
    (
        ("period_return", np.inf, "numeric"),
        ("period_return", True, "numeric"),
        ("period_return", "invalid", "numeric"),
        ("eligible", None, "eligibility"),
    ),
)
def test_missing_return_fix_preserves_numeric_and_eligibility_boundaries(
    column, value, error
):
    panel = pd.DataFrame(
        dict(
            asset_id=["a"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.01,
            float_market_cap=100.0,
            eligible=True,
        )
    )
    panel[column] = value
    with pytest.raises(ValueError, match=f"market_model_risk_{error}_invalid"):
        MarketModelRiskState().observe(panel)


@pytest.mark.parametrize("missing_asset", ("a", "b"))
def test_required_return_gap_resets_every_asset_without_partial_renormalization(
    missing_asset,
):
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.01,
            float_market_cap=[100.0, 200.0],
        )
    )
    for value in (0.01, 0.02, 0.03):
        state.observe(panel.assign(period_return=value))
    panel.loc[panel.asset_id.eq(missing_asset), "period_return"] = np.nan
    result = state.observe(panel)
    assert result.beta.isna().all()
    assert result.residual_volatility.isna().all()
    assert state.observe(panel.assign(period_return=0.04)).beta.isna().all()
    result = state.observe(panel.assign(period_return=0.05))
    assert result.beta.eq(1.0).all()
    assert result.residual_volatility.eq(0.0).all()


def test_unknown_incumbent_cap_keeps_next_market_pair_unknown():
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "listing_date": pd.Timestamp("2000-01-01"),
            "period_return": 0.01,
            "float_market_cap": 100.0,
            "filling_flag": 0,
            "eligible": True,
            "close": 10.0,
        }
    )
    for value in (0.01, 0.02, 0.03):
        state.observe(panel.assign(period_return=value))
    missing = panel.assign(period_return=0.04)
    missing.loc[0, ["float_market_cap", "close"]] = np.nan
    missing.loc[0, "eligible"] = False
    assert state.observe(missing).beta.eq(1.0).all()
    assert state._market_returns["b"][-1] == pytest.approx(0.04)
    # A real incumbent's absent lagged weight must not disappear via
    # renormalization onto the members whose quotes happen to remain visible.
    assert state.observe(panel.assign(period_return=0.05)).beta.isna().all()
    assert all(np.isnan(values[-1]) for values in state._market_returns.values())
    state.observe(panel.assign(period_return=0.06))
    assert state.observe(panel.assign(period_return=0.07)).beta.notna().all()


def test_market_model_ignores_unrelated_quote_missingness():
    states = [MarketModelRiskState(window=3, residual_window=2) for _ in range(2)]
    base = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "listing_date": pd.Timestamp("2000-01-01"),
            "period_return": 0.01,
            "float_market_cap": 100.0,
            "filling_flag": 0,
            "eligible": True,
        }
    )
    for value in (0.01, 0.02, -0.01, 0.03, 0.04, -0.02):
        current = base.assign(period_return=value)
        expected = states[0].observe(current)
        actual = states[1].observe(
            current.assign(
                close=np.nan,
                adj_close=np.nan,
                volume=np.nan,
                amount=np.nan,
                turnover=np.nan,
            )
        )
        pd.testing.assert_series_equal(actual.beta, expected.beta)
        pd.testing.assert_series_equal(
            actual.residual_volatility, expected.residual_volatility
        )


@pytest.mark.parametrize(
    "scope_fact", ("is_a_share_scope", "is_st", "is_pt", "has_252_trading_sessions")
)
def test_unknown_membership_cannot_drop_lagged_market_member(scope_fact):
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "listing_date": pd.Timestamp("2000-01-01"),
            "period_return": 0.01,
            "float_market_cap": [100.0, 200.0],
            "filling_flag": 0,
            "eligible": True,
            "is_a_share_scope": True,
            "is_st": False,
            "is_pt": False,
            "has_252_trading_sessions": True,
        }
    )
    for value in (0.01, 0.02, 0.03):
        state.observe(panel.assign(period_return=value))
    panel[scope_fact] = panel[scope_fact].astype("boolean")
    panel.loc[0, scope_fact] = pd.NA
    panel.loc[0, "eligible"] = False
    result = state.observe(panel.assign(period_return=0.04))
    assert result.beta.isna().all()
    assert all(np.isnan(values[-1]) for values in state._market_returns.values())


@pytest.mark.parametrize("bad_cap", (np.nan, 0.0, -1.0, np.inf, "invalid", True))
def test_current_cap_gap_preserves_lagged_proxy_then_invalidates_next_pair(bad_cap):
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "listing_date": pd.Timestamp("2000-01-01"),
            "period_return": [0.01, 0.02],
            "float_market_cap": [100.0, 200.0],
            "eligible": True,
        }
    )

    def observe(value, *, missing=False):
        current = panel.assign(period_return=[value, 2 * value])
        if missing:
            current["float_market_cap"] = pd.Series([bad_cap, 900.0], dtype=object)
        return state.observe(current)

    for value in (0.01, 0.02, 0.03):
        observe(value)
    actual = observe(0.04, missing=True)
    # Unequal returns and caps distinguish lagged weighting from both equal
    # weighting and today's available-cap subset. No current cap is consumed.
    expected = float((100 * Fraction(0.04) + 200 * Fraction(0.08)) / 300)
    assert all(values[-1] == expected for values in state._market_returns.values())
    assert actual.beta.to_numpy() == pytest.approx([0.6, 1.2])
    assert actual.residual_volatility.to_numpy() == pytest.approx([0.0, 0.0])
    # Neither a repeated gap nor restored current caps provide the missing
    # previous-session weight. Do not renormalize onto the remaining member.
    for missing in (True, False):
        actual = observe(0.05, missing=missing)
        assert actual.beta.isna().all()
        assert actual.residual_volatility.isna().all()
        assert all(np.isnan(values[-1]) for values in state._market_returns.values())
    actual = observe(0.06)
    expected = float((100 * Fraction(0.06) + 200 * Fraction(0.12)) / 300)
    assert all(values[-1] == expected for values in state._market_returns.values())
    assert actual.beta.isna().all()
    actual = observe(0.07)
    assert actual.beta.to_numpy() == pytest.approx([0.6, 1.2])
    assert actual.residual_volatility.to_numpy() == pytest.approx([0.0, 0.0])


def test_unknown_market_barrier_retains_prior_membership_obligation():
    state = MarketModelRiskState(window=2)
    panel = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "listing_date": pd.Timestamp("2000-01-01"),
            "period_return": 0.01,
            "float_market_cap": 100.0,
            "filling_flag": 0.0,
            "eligible": True,
            "market_type": 1.0,
            "_calendar_segment": 0,
        }
    )
    state.observe(panel, open_market_types=frozenset({1}))
    for segment in (1, 2, 3):
        missing = panel.assign(period_return=0.2)
        missing.loc[
            0, ["market_type", "period_return", "float_market_cap", "filling_flag"]
        ] = np.nan
        missing.loc[0, "eligible"] = False
        missing.loc[0, "_calendar_segment"] = segment
        state.observe(missing, open_market_types=frozenset({1}))
        assert np.isnan(state._market_returns["b"][-1])
    returned = panel.assign(period_return=0.3)
    returned.loc[0, "_calendar_segment"] = 4
    state.observe(returned, open_market_types=frozenset({1}))
    assert np.isnan(state._market_returns["b"][-1])
    state.observe(returned.assign(period_return=0.4), open_market_types=frozenset({1}))
    assert state._market_returns["b"][-1] == pytest.approx(0.4)


@pytest.mark.parametrize("boundary", ("relisting", "calendar_segment"))
def test_cap_gap_cannot_reuse_weight_across_lifecycle_boundary(boundary):
    state = MarketModelRiskState(window=2)
    panel = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "listing_date": pd.Timestamp("2000-01-01"),
            "_calendar_segment": 0,
            "period_return": [0.01, 0.02],
            "float_market_cap": [100.0, 200.0],
            "eligible": True,
        }
    )
    state.observe(panel)
    current = panel.assign(period_return=[0.04, 0.08])
    current.loc[0, "float_market_cap"] = np.nan
    if boundary == "relisting":
        current.loc[0, "listing_date"] = pd.Timestamp("2020-01-01")
    else:
        current.loc[0, "_calendar_segment"] = 1
    state.observe(current)
    if boundary == "relisting":
        assert state._market_returns["b"][-1] == 0.08
    else:
        assert np.isnan(state._market_returns["b"][-1])
    state.observe(current.assign(float_market_cap=[100.0, 200.0]))
    assert np.isnan(state._market_returns["b"][-1])
