"""Specific-risk age follows actual asset-market sessions, including missing data."""

import pandas as pd
import pytest

from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
)


def _cross(day, prefix, multiplier):
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp(day),
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": [f"{prefix}{i}" for i in range(5)],
            "industry_standard_id": "S",
            "industry_level1_code": "I",
            "total_market_cap": 100.0,
            "eligible": True,
            "asset_return": [multiplier * x for x in (-0.02, -0.01, 0, 0.01, 0.02)],
        }
    )


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_specific_risk_distinguishes_market_closure_from_missing_open_observations(
    closed, shared
):
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            factor_window=10,
            specific_window=2,
            minimum_factor_dates=1,
            minimum_specific_dates=1,
            minimum_cross_section=5,
            specific_half_life=1.0,
            annualization=1.0,
        ),
    )
    cache = {} if shared else None
    for index, (prefix, multiplier) in enumerate((("A", 1), ("B", 1), ("A", 2))):
        cross = _cross(f"2023-01-0{index + 2}", prefix, multiplier)
        closed_assets = (
            frozenset(f"A{i}" for i in range(5))
            if closed and index == 1
            else frozenset()
        )
        if cache is not None:
            # A second independent owner can supply the same immutable regression fit.
            fitted, industries = state._fit_return_session(cross)
            cache[index] = (industries, (fitted.factor_returns, fitted.residuals))
        state.observe_return_session(
            cross, closed_assets=closed_assets, shared_fits=cache, fit_key=index
        )
    expected = 0.0004 / 3 + 0.0016 * 2 / 3 if closed else 0.0016
    assert state.estimate().specific_variance["A0"] == pytest.approx(expected)
    assert len(state._residuals["A0"]) == (2 if closed else 1)


@pytest.mark.parametrize("missing_panel", [False, True])
def test_producer_advances_risk_clock_through_whole_open_market_data_gap(
    tmp_path, monkeypatch, missing_panel
):
    from quant_research_platform.research.portfolios import production
    from tests.research.test_portfolio_production import (
        _build_producer,
        _enhanced_inputs,
    )

    estimate = production.FactorRiskState.estimate
    frames, fees, window, candidate = _enhanced_inputs(monkeypatch)
    monkeypatch.setattr(production.FactorRiskState, "estimate", estimate)
    gap = pd.Timestamp("2022-12-20")
    if missing_panel:
        for name in ("panel", "dataset", "execution"):
            frames[name] = frames[name].loc[~frames[name].timestamp.eq(gap)].copy()
    else:
        frames["panel"].loc[frames["panel"].timestamp.eq(gap), "period_return"] = float(
            "nan"
        )
    clocks = []
    original = production.FactorRiskState.observe_return_session

    def observe(state, cross, **kwargs):
        previous = state._session_count
        if not cross.empty and cross.timestamp.eq(gap).all():
            assert cross.asset_return.isna().all()
        original(state, cross, **kwargs)
        assert state._session_count == previous + 1
        if cross.empty:
            assert not state._factor_rows
        clocks.append((state._session_count, dict(state._specific_session_counts)))

    monkeypatch.setattr(production.FactorRiskState, "observe_return_session", observe)
    stream, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        typed_batches=True,
    )
    batches = list(stream)
    assert len(batches) == 4
    sessions = frames["calendar"].loc[
        frames["calendar"].calendar_date.between("2022-12-01", "2023-01-10")
    ]
    expected = len(sessions) - 1
    assert clocks[-1] == (expected, {"a": expected, "b": expected})
