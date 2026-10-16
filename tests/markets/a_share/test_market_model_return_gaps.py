from fractions import Fraction

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


def test_native_market_model_return_gap_persists_and_recovers_complete_windows(
    tmp_path,
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
        "period_return",
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
            "period_return",
        ]
        .isna()
        .all()
    )
    styles = ["beta_252d", "residual_volatility_60d"]
    assert factors.loc[factors.timestamp.eq(dates[gap - 1]), styles].notna().all().all()
    for column, recovery in zip(styles, (252, 311), strict=True):
        assert (
            factors.loc[
                factors.timestamp.between(dates[gap], dates[gap + recovery - 1]), column
            ]
            .isna()
            .all()
        )
        assert (
            factors.loc[factors.timestamp.eq(dates[gap + recovery]), column]
            .notna()
            .all()
        )
    risk = FactorRiskState(style_columns=tuple(styles))
    for day, count in ((gap - 1, 3), (gap, 0), (gap + 310, 0), (gap + 311, 3)):
        cross = factors.loc[factors.timestamp.eq(dates[day])].assign(
            industry_standard_id="CSRC2012",
            industry_level1_code="A",
            total_market_cap=1000000.0,
            eligible=True,
        )
        prepared = risk.prepare_exposures(cross)
        pd.testing.assert_frame_equal(prepared[styles], cross[styles])
        processed = preprocess_risk_styles(prepared, style_columns=tuple(styles))
        if day == gap + 310:
            assert processed.beta_252d.notna().all()
            assert processed.residual_volatility_60d.isna().all()
        exposures = risk.current_exposures(processed.dropna(subset=styles))
        assert len(exposures) == count
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
    for day in (gap - 1, gap + 311):
        for asset in returns:
            residuals = []
            for end in range(day - 59, day + 1):
                x = list(map(Fraction, market[end - 251 : end + 1]))
                y = list(map(Fraction, returns[asset].iloc[end - 251 : end + 1]))
                mx, my = sum(x) / 252, sum(y) / 252
                slope = sum(
                    (a - mx) * (b - my) for a, b in zip(x, y, strict=True)
                ) / sum((a - mx) ** 2 for a in x)
                residuals.append(float(y[-1] - my - slope * (x[-1] - mx)))
            row = factors.loc[
                factors.timestamp.eq(dates[day]) & factors.asset_id.eq(asset)
            ].iloc[0]
            assert row.beta_252d == float(slope)
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
        ("float_market_cap", np.nan, "numeric"),
        ("period_return", np.inf, "numeric"),
        ("period_return", True, "numeric"),
        ("period_return", "invalid", "numeric"),
        ("float_market_cap", 0.0, "key"),
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
