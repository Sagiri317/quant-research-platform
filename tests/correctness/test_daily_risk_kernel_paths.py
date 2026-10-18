"""Independent ordinary WLS, covariance, shrinkage and PSD oracles."""

import numpy as np
import pytest

from quant_research_platform.research.portfolios import risk
from tests.correctness.test_portfolio_weight_and_risk_numeric_boundaries import (
    _cross,
    _full_wls_fraction_oracle,
)


def test_weighted_svd_matches_independent_normal_system():
    rng = np.random.default_rng(701)
    frame = _cross()
    frame["total_market_cap"] = np.tile([1.0, 4.0, 16.0, 64.0, 256.0], 20)
    frame["asset_return"] = rng.normal(0, 0.02, len(frame))
    styles = ("style_a", "style_b", "style_c")
    for name in styles:
        frame[name] = rng.normal(size=len(frame))

    for order in (np.arange(len(frame)), rng.permutation(len(frame))):
        current = frame.iloc[order]
        fit = risk.FactorRiskState(style_columns=styles).fit_return_session(current)
        assert fit is not None
        expected, residuals = _full_wls_fraction_oracle(fit.exposures, current)
        np.testing.assert_allclose(fit.factor_returns, expected, rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(fit.residuals, residuals, rtol=2e-13, atol=1e-15)
        aligned = current.set_index("asset_id").reindex(fit.exposures.index)
        normal_residual = fit.exposures.to_numpy().T @ (
            np.sqrt(aligned.total_market_cap.to_numpy()) * fit.residuals.to_numpy()
        )
        np.testing.assert_allclose(normal_residual, 0, atol=2e-13)


@pytest.mark.parametrize("shrinkage", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("zero_columns", [1, 5])
def test_zero_columns_keep_zero_covariance(shrinkage, zero_columns):
    values = np.random.default_rng(52).normal(size=(504, 5)) * 0.01
    values[:, -zero_columns:] = 0
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
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-17)
    assert np.isfinite(actual).all().all()
    assert np.linalg.eigvalsh(actual).min() >= 0
    np.testing.assert_array_equal(actual.to_numpy()[:, -zero_columns:], 0)


@pytest.mark.parametrize("shrinkage", [0.0, 0.2, 1.0])
def test_cancelling_cross_products_keep_zero_covariance(shrinkage):
    actual = risk._ewma_covariance(
        np.array([[0.01, 0.01], [-0.01, 0.01], [0.01, -0.01], [-0.01, -0.01]]),
        ages=np.zeros(4, dtype=int),
        half_life=1.0,
        diagonal_shrinkage=shrinkage,
        annualization_factor=1.0,
        columns=("a", "b"),
    )
    np.testing.assert_allclose(actual, np.eye(2) * 0.0001, rtol=1e-13, atol=1e-18)


def test_constant_nonzero_columns_have_zero_covariance():
    actual = risk._ewma_covariance(
        np.tile([0.01, 0.03], (504, 1)),
        ages=np.arange(504),
        half_life=126.0,
        diagonal_shrinkage=0.2,
        annualization_factor=252.0,
        columns=("a", "b"),
    )
    np.testing.assert_array_equal(actual, np.zeros((2, 2)))


@pytest.mark.parametrize("scale", [1e-6, 1e-4, 1.0])
def test_psd_tolerance_scales_with_covariance_units(scale):
    for matrix in (np.full((3, 3), scale), np.zeros((3, 3))):
        np.testing.assert_array_equal(
            risk._validated_covariance(matrix, error_prefix="factor_risk_covariance"),
            matrix,
        )
    for matrix in (scale * np.array([[1.0, 1.01], [1.01, 1.0]]), np.array([[-scale]])):
        with pytest.raises(ValueError, match="factor_risk_covariance_not_psd"):
            risk._validated_covariance(matrix, error_prefix="factor_risk_covariance")
    with pytest.raises(ValueError, match="factor_risk_covariance_invalid"):
        risk._validated_covariance(
            scale * np.array([[1.0, 1.009], [1.0, 1.0]]),
            error_prefix="factor_risk_covariance",
        )
    almost_symmetric = scale * np.array([[1.0, 0.1 + np.finfo(float).eps], [0.1, 1.0]])
    validated = risk._validated_covariance(
        almost_symmetric, error_prefix="factor_risk_covariance"
    )
    np.testing.assert_array_equal(
        validated, (almost_symmetric + almost_symmetric.T) / 2
    )


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_ewma_weights_decay_for_stored_session_ages(dtype):
    weights = risk._ewma_weights(np.array([3, 2, 1], dtype=dtype), 1.0)
    np.testing.assert_allclose(weights, np.array([1.0, 2.0, 4.0]) / 7, rtol=1e-14)
