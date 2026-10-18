"""Ordinary-scale numerical kernels and dependency-local market risk."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.market_model import MarketModelRiskState
from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios.optimizer import (
    OptimizerConfig,
    optimize_portfolio,
)
from quant_research_platform.research.portfolios.risk import _ewma_covariance
from tests.helpers.optimizer import with_cash_asset


def test_public_optimizer_accepts_native_rank_one_roundoff_at_daily_return_scale():
    values = np.array([-1.3267282928093849, 1.107768166657506]) * 0.01
    covariance = _ewma_covariance(
        np.array([values, -values]),
        ages=np.array([1.0, 0.0]),
        half_life=1.0,
        diagonal_shrinkage=0.0,
        annualization_factor=252.0,
        columns=("a", "b"),
    )
    active = np.array([values[1], -values[0]])
    stocks = pd.DataFrame(
        {
            "asset_id": ["one", "two", "three"],
            "expected_alpha": 0.0,
            "current_weight": [0.3 + active[0], 0.3 + active[1], 0.4 - active.sum()],
            "benchmark_weight": [0.3, 0.3, 0.4],
            "specific_variance": 0.0,
            "adv": 1e9,
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
        }
    )
    result = optimize_portfolio(
        with_cash_asset(stocks),
        cash_column="is_cash",
        factor_covariance=covariance,
        style_columns=("a", "b"),
        expected_transaction_cost=lambda buy, sell: (
            0.0,
            np.zeros_like(buy),
            np.zeros_like(sell),
        ),
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        config=OptimizerConfig(active_name_limit=0.1),
    )
    assert np.isfinite(result.weights).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (result.weights >= 0).all()
    assert 0 <= result.predicted_tracking_error < 1e-9


@pytest.mark.parametrize("cap", [np.nan, 0.0, -1.0, np.inf, "invalid", True])
def test_excluded_cap_does_not_destroy_independent_market_beta(cap):
    state = MarketModelRiskState(window=3, residual_window=2)
    for value in (0.0, 0.01, 0.03, -0.02, 0.02):
        state.observe(
            pd.DataFrame(
                dict(
                    asset_id=["good"],
                    listing_date=pd.Timestamp("2000-01-01"),
                    period_return=value,
                    float_market_cap=100.0,
                    eligible=True,
                )
            )
        )
    result = state.observe(
        pd.DataFrame(
            dict(
                asset_id=["good", "excluded"],
                listing_date=pd.Timestamp("2000-01-01"),
                period_return=[0.04, 0.02],
                float_market_cap=[100.0, cap],
                eligible=[True, False],
                is_st=[False, True],
            )
        )
    )
    assert result.beta["good"] == pytest.approx(1.0)


@pytest.mark.parametrize("cap", [np.nan, 0.0, -1.0, np.inf, True])
def test_required_member_bad_cap_keeps_proxy_uncertainty(cap):
    state = MarketModelRiskState(window=2, residual_window=2, min_fresh_fraction=1.0)
    panel = pd.DataFrame(
        dict(
            asset_id=["good", "required"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=[0.01, 0.02],
            float_market_cap=[100.0, 100.0],
            eligible=True,
        )
    )
    for returns in ([0.01, 0.02], [0.03, -0.01], [-0.02, 0.04]):
        panel["period_return"] = returns
        state.observe(panel)
    panel["float_market_cap"] = [100.0, cap]
    result = state.observe(panel)
    assert result.beta.isna().all()
    panel["float_market_cap"] = 100.0
    assert state.observe(panel).beta.isna().all()


@pytest.mark.parametrize("shrinkage", [0.0, 0.2, 1.0])
def test_ordinary_kernels_use_vectorized_paths_and_independent_oracles(shrinkage):
    from quant_research_platform.compute.dataframe import cross_section
    from quant_research_platform.research.features import transforms
    from quant_research_platform.research.models import rank_ensemble
    from quant_research_platform.research.portfolios import risk

    rng = np.random.default_rng(52)
    values = rng.normal(size=(504, 5)) * 0.01
    ages = np.arange(503, -1, -1)
    weights = np.exp2(-ages / 126)
    expected = np.cov(values, rowvar=False, aweights=weights, ddof=0) * 252
    expected = (1 - shrinkage) * expected + shrinkage * np.diag(np.diag(expected))
    actual = risk._ewma_covariance(
        values,
        ages=ages,
        half_life=126,
        diagonal_shrinkage=shrinkage,
        annualization_factor=252,
        columns=tuple("abcde"),
    )
    assert actual.shape == (5, 5)
    assert actual.columns.tolist() == list("abcde")
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-17)
    assert np.linalg.eigvalsh(actual).min() >= 0
    correlation, reason = cross_section._pearson_correlation_result(
        values[:, 0], values[:, 1]
    )
    assert reason is None
    assert correlation == pytest.approx(
        np.corrcoef(values[:, :2].T)[0, 1], rel=2e-14, abs=1e-16
    )
    scores = rng.uniform(-0.5, 0.5, 504)
    labels = 0.01 * scores + rng.normal(0, 0.02, 504)
    design = np.column_stack((np.ones(504), scores))
    expected = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), labels * np.sqrt(weights), rcond=None
    )[0]
    calibration = rank_ensemble._label_calibration(scores, labels, weights)
    np.testing.assert_allclose(calibration, expected, rtol=2e-14, atol=1e-17)
    # Centered ordinary targets may have a zero intercept; that alone is not
    # ill conditioning and must not force arbitrary-precision fitting.
    symmetric_scores = np.array([-0.375, -0.125, 0.125, 0.375])
    np.testing.assert_allclose(
        rank_ensemble._label_calibration(
            symmetric_scores, 0.02 * symmetric_scores, None
        ),
        (0.0, 0.02),
        rtol=2e-14,
        atol=1e-18,
    )
    residual = labels - design @ calibration
    np.testing.assert_allclose(design.T @ (weights * residual), 0, atol=1e-14)
    industries = pd.Series(np.arange(504) % 7)
    sizes = rng.normal(20, 2, 504)
    target = rng.normal(size=504)
    design = np.column_stack((pd.get_dummies(industries).to_numpy(dtype=float), sizes))
    expected = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    residual = transforms._industry_size_residuals(
        target, industries, sizes, rescale=False
    )
    np.testing.assert_allclose(residual, expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(design.T @ residual, 0, atol=2e-11)
    np.testing.assert_allclose(
        transforms._industry_size_residuals(target, industries, sizes, rescale=True),
        residual / np.abs(residual).max(),
        rtol=2e-14,
        atol=1e-16,
    )


def test_roundoff_bound_preserves_materially_indefinite_risk():
    from quant_research_platform.research.portfolios.optimizer import _risk_variances

    active = np.array([0.011077681666575061, 0.013267282928093848])
    covariance = np.array(
        [
            [0.03942865836987627, -0.032921445056149336],
            [-0.032921445056149336, 0.027488166967738002],
        ]
    )
    assert active @ covariance @ active < 0
    assert _risk_variances(active, np.eye(2), covariance, np.zeros(2)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="optimizer_risk_negative"):
        _risk_variances(active, np.eye(2), -np.eye(2) * 1e-13, np.zeros(2))
