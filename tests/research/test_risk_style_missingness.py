"""Risk transforms use per-style fit domains before joint regressions."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
    HoldingFactorInputs,
    holding_attribution,
    preprocess_risk_styles,
)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([7.0, 7.0, np.nan], [0.0, 0.0, np.nan]),
        ([7.0, np.nan], [np.nan, np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan]),
    ],
)
def test_constant_style_preserves_missingness_and_minimum_count(values, expected):
    frame = pd.DataFrame(
        {"timestamp": pd.Timestamp("2024-01-02"), "eligible": True, "style": values}
    )
    actual = preprocess_risk_styles(frame, style_columns=("style",))
    np.testing.assert_allclose(actual["style"], expected, equal_nan=True)


def test_style_fit_domains_are_independent_by_column_and_day():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 3
            + [pd.Timestamp("2024-01-03")] * 3,
            "eligible": True,
            "a": [1.0, 2.0, 3.0, 7.0, 7.0, np.nan],
            "b": [1.0, 2.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    actual = preprocess_risk_styles(frame, style_columns=("a", "b"))
    full = preprocess_risk_styles(frame.fillna({"b": 3.0}), style_columns=("a", "b"))
    pd.testing.assert_series_equal(actual.a, full.a, check_exact=True)
    np.testing.assert_allclose(actual.a[:3], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    np.testing.assert_allclose(actual.a[3:], [0.0, 0.0, np.nan], equal_nan=True)


def _cross():
    axis = np.arange(24, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": [f"asset_{i}" for i in range(len(axis))],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "total_market_cap": 100.0 + axis,
            "eligible": True,
            "a": axis,
            "b": np.sin(axis),
            "asset_return": 0.01 + 0.002 * axis + 0.003 * np.sin(axis),
        }
    )


def test_joint_risk_fit_uses_per_style_domain_then_complete_cases_and_reconciles():
    cross = _cross()
    cross.loc[23, "b"] = np.nan
    cross.loc[22, "asset_return"] = np.nan
    state = FactorRiskState(
        style_columns=("a", "b"), config=FactorRiskConfig(minimum_cross_section=15)
    )
    expected = preprocess_risk_styles(cross, style_columns=("a", "b"))
    assert len(state.prepare_exposures(cross)) == len(cross)
    fit = state.fit_return_session(cross)
    assert fit is not None
    selected = expected.dropna(subset=["a", "b", "asset_return"]).set_index("asset_id")
    pd.testing.assert_frame_equal(fit.exposures[["a", "b"]], selected[["a", "b"]])
    np.testing.assert_allclose(
        fit.exposures @ fit.factor_returns + fit.residuals,
        fit.asset_returns,
        atol=1e-15,
    )
    assert "asset_23" not in fit.exposures.index
    assert "asset_22" not in fit.exposures.index
    state.observe_return_session(cross)
    np.testing.assert_allclose(
        [state._factor_rows[0][column] for column in fit.factor_returns.index],
        fit.factor_returns,
    )
    notionals = pd.Series(100 / len(fit.exposures), index=fit.exposures.index)
    values, reasons, _ = holding_attribution(
        HoldingFactorInputs(
            decision_at=pd.Timestamp("2024-01-02"),
            entry_at=pd.Timestamp("2024-01-03"),
            end_at=pd.Timestamp("2024-01-04"),
            fitted=fit,
            decision_close=pd.Series(10.0, index=fit.exposures.index),
            entry_open=pd.Series(10.0, index=fit.exposures.index),
            benchmark_weights=pd.DataFrame(),
        ),
        after_notionals=notionals,
        nav_before=100.0,
        nav_post=100.0,
        benchmark_id=None,
        benchmark_return=None,
        holding_returns=fit.asset_returns,
    )
    assert not reasons
    assert sum(
        values[key]
        for key in ("benchmark", "industry", "style", "residual_unexplained")
    ) == pytest.approx(notionals.dot(fit.asset_returns) / 100.0, abs=1e-15)


def test_enhanced_decision_preprocesses_before_joint_style_selection():
    cross = _cross()
    state = FactorRiskState(style_columns=("a", "b"))
    scheduled = pd.DataFrame(
        {
            name: pd.Series(dtype="datetime64[ns]")
            for name in ("timestamp", "entry_at", "available_at")
        }
    )

    def exposures(frame):
        return production._enhanced_decision(
            benchmark_weights=pd.Series(1 / len(frame), index=frame.asset_id),
            alpha=pd.Series(0.0, index=frame.asset_id),
            factor_risk=state,
            current_cross=frame,
            calibration_rows=[],
            calibration_intervals=scheduled,
            trading_sessions=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
            as_of=pd.Timestamp("2024-01-02"),
            risk_style_columns=("a", "b"),
        ).exposures.set_index("asset_id")

    full = exposures(cross)
    cross.loc[23, "b"] = np.nan
    missing = exposures(cross)
    pd.testing.assert_series_equal(
        missing.a, full.a.loc[missing.index], check_exact=True
    )
    assert "asset_23" not in missing.index


@pytest.mark.parametrize(
    "eligible", ([True, True, False, False], [True, False, False, False])
)
def test_constant_projection_requires_two_eligible_values_and_keeps_null(eligible):
    cross = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": eligible,
            "style": [7.0, 7.0, 20.0, np.nan],
        }
    )
    actual = preprocess_risk_styles(cross, style_columns=("style",))
    expected = [0.0, 0.0, 0.0, np.nan] if sum(eligible) == 2 else [np.nan] * 4
    np.testing.assert_allclose(actual["style"], expected, equal_nan=True)
