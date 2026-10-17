"""Independent ordinary WLS/covariance oracles and exceptional-path boundaries."""

import numpy as np
import pytest

from quant_research_platform.research.portfolios import risk
from tests.correctness.test_portfolio_weight_and_risk_numeric_boundaries import (
    _cross,
    _full_wls_fraction_oracle,
)


def test_ordinary_weighted_svd_matches_exact_normal_system(monkeypatch):
    rng = np.random.default_rng(701)
    frame = _cross()
    frame["total_market_cap"] = np.tile([1.0, 4.0, 16.0, 64.0, 256.0], 20)
    frame["asset_return"] = rng.normal(0, 0.02, len(frame))
    styles = ("style_a", "style_b", "style_c")
    for name in styles:
        frame[name] = rng.normal(size=len(frame))

    def unexpected_exact(*args, **kwargs):
        raise AssertionError("ordinary WLS must avoid arbitrary-precision accumulation")

    monkeypatch.setattr(risk, "_integer_coordinates", unexpected_exact)
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
def test_exact_zero_covariance_products_stay_vectorized(
    monkeypatch, shrinkage, zero_columns
):
    values = np.random.default_rng(52).normal(size=(504, 5)) * 0.01
    values[:, -zero_columns:] = 0
    ages = np.arange(503, -1, -1)
    weights = np.exp2(-ages / 126)
    expected = np.cov(values, rowvar=False, aweights=weights, ddof=0) * 252
    expected = (1 - shrinkage) * expected + shrinkage * np.diag(np.diag(expected))

    def unexpected_exact(*args, **kwargs):
        raise AssertionError("exact-zero products are not cancellation or underflow")

    monkeypatch.setattr(risk, "_integer_coordinates", unexpected_exact)
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
def test_covariance_genuine_underflow_retains_exact_fallback(monkeypatch, shrinkage):
    calls = []
    original = risk._integer_coordinates

    def record(values):
        calls.append(True)
        return original(values)

    monkeypatch.setattr(risk, "_integer_coordinates", record)
    actual = risk._ewma_covariance(
        np.array([[-2e-162, 0.0], [2e-162, 0.0]]),
        ages=np.zeros(2, dtype=int),
        half_life=1.0,
        diagonal_shrinkage=shrinkage,
        annualization_factor=1e300,
        columns=("tiny", "zero"),
    )
    from fractions import Fraction

    expected = float(Fraction(2e-162) ** 2 * Fraction(1e300))
    assert calls
    assert actual.iloc[0, 0] == expected
    assert actual.iloc[1, 1] == 0


def test_diagonal_shrinkage_ignores_discarded_covariance_cancellation(monkeypatch):
    def unexpected_exact(*args, **kwargs):
        raise AssertionError("discarded off-diagonal cancellation is irrelevant")

    monkeypatch.setattr(risk, "_integer_coordinates", unexpected_exact)
    actual = risk._ewma_covariance(
        np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]),
        ages=np.zeros(4, dtype=int),
        half_life=1.0,
        diagonal_shrinkage=1.0,
        annualization_factor=1.0,
        columns=("a", "b"),
    )
    np.testing.assert_allclose(actual, np.eye(2), atol=1e-15)


def test_covariance_nonzero_product_cancellation_retains_exact_fallback(monkeypatch):
    calls = []
    original = risk._integer_coordinates

    def record(values):
        calls.append(True)
        return original(values)

    monkeypatch.setattr(risk, "_integer_coordinates", record)
    actual = risk._ewma_covariance(
        np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]),
        ages=np.zeros(4, dtype=int),
        half_life=1.0,
        diagonal_shrinkage=0.0,
        annualization_factor=1.0,
        columns=("a", "b"),
    )
    assert calls
    np.testing.assert_allclose(actual, np.eye(2), atol=1e-15)


def test_covariance_weighting_underflow_is_not_an_exact_zero_column(monkeypatch):
    calls = []
    original = risk._integer_coordinates

    def record(values):
        calls.append(True)
        return original(values)

    monkeypatch.setattr(risk, "_integer_coordinates", record)
    actual = risk._ewma_covariance(
        np.array([[0.0], [1e-200]]),
        ages=np.array([0, 1020]),
        half_life=1.0,
        diagonal_shrinkage=1.0,
        annualization_factor=1e300,
        columns=("tiny",),
    )
    assert calls
    # Even after annualization this nonzero variance rounds below binary64.
    assert actual.iloc[0, 0] == 0
