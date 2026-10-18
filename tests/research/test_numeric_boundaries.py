from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research import statistics, statistics_production
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
    _ewma_covariance,
)
from tests.helpers.optimizer import with_cash_asset


def _risk_cross(returns, caps, timestamp="2023-01-02"):
    return pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp(timestamp),
            "asset_id": [str(i) for i in range(len(returns))],
            "industry_standard_id": "fixture",
            "industry_level1_code": "one",
            "total_market_cap": caps,
            "asset_return": returns,
        }
    )


@pytest.mark.parametrize("constant", [False, True])
def test_sqrt_cap_weighted_fit_preserves_mean_and_residuals(constant):
    returns = (
        np.full(100, 0.02) if constant else np.tile([0.03, -0.02, 0.01, 0.005], 25)
    )
    caps = np.tile([1e9, 4e9, 9e9, 16e9], 25)
    weights = np.sqrt(caps)
    expected = np.average(returns, weights=weights)
    fitted = FactorRiskState(style_columns=()).fit_return_session(
        _risk_cross(returns, caps)
    )
    assert fitted is not None
    np.testing.assert_allclose(
        fitted.factor_returns, [expected], rtol=1e-13, atol=1e-16
    )
    np.testing.assert_allclose(
        fitted.residuals, returns - expected, rtol=1e-13, atol=1e-16
    )
    if constant:
        assert fitted.residuals.eq(0).all()


@pytest.mark.parametrize("shrinkage", [0.0, 0.2, 1.0])
def test_ewma_covariance_matches_weighted_population_oracle(shrinkage):
    values = np.array([[0.01, -0.02], [-0.02, 0.01], [0.03, 0.01], [-0.01, 0.02]])
    ages = np.array([5, 3, 1, 0])
    weights = 2.0 ** (-ages / 3.0)
    weights /= weights.sum()
    centered = values - np.average(values, axis=0, weights=weights)
    expected = (centered * weights[:, None]).T @ centered * 252
    expected = (1 - shrinkage) * expected + shrinkage * np.diag(np.diag(expected))
    actual = _ewma_covariance(
        values,
        ages=ages,
        half_life=3.0,
        diagonal_shrinkage=shrinkage,
        annualization_factor=252,
        columns=("a", "b"),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-16)
    np.testing.assert_array_equal(actual, actual.T)


def test_bh_by_reuse_individual_hac_and_keep_full_family_order(monkeypatch):
    rng = np.random.default_rng(6107)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=40),
            "a": rng.normal(0.1, 0.3, 40),
            "b": rng.normal(-0.1, 0.4, 40),
            "constant": 0.1,
            "diagnostic": rng.normal(0.0, 0.2, 40),
        }
    )
    planned = ("b", "missing", "a", "constant")
    expected_parts = []
    for method in ("BH", "BY"):
        part = statistics.time_series_inference_summary(
            frame.reindex(columns=["timestamp", *planned]),
            identity_column="timestamp",
            max_lags=2,
            fdr_method=method,
            fdr_alpha=0.05,
            fdr_family_size=4,
        ).rename(columns={"timestamp": "factor_id"})
        part.insert(0, "family_id", "declared")
        part["planned_hypothesis_count"] = 4
        part["defined_hypothesis_count"] = 2
        part["missing_hypothesis_count"] = 2
        expected_parts.append(part)
    diagnostic = statistics.time_series_inference_summary(
        frame[["timestamp", "diagnostic"]],
        identity_column="timestamp",
        max_lags=2,
        fdr_method=None,
        fdr_alpha=0.05,
    ).rename(columns={"timestamp": "factor_id"})
    diagnostic.insert(0, "family_id", "diagnostic_only")
    diagnostic["fdr_method"] = "NOT_APPLICABLE"
    diagnostic["planned_hypothesis_count"] = 1
    diagnostic["defined_hypothesis_count"] = 1
    diagnostic["missing_hypothesis_count"] = 0
    expected_parts.append(diagnostic)
    calls = []
    native = statistics.hac_mean

    def observed(values, **kwargs):
        calls.append(values.name)
        return native(values, **kwargs)

    monkeypatch.setattr(statistics, "hac_mean", observed)
    actual = statistics_production._factor_summary(
        frame,
        factors=("b", "a", "constant", "diagnostic"),
        families={"declared": planned},
        methods=("BH", "BY"),
        max_lags=2,
    )
    assert calls == ["b", "a", "constant", "diagnostic"]
    expected = pd.concat(expected_parts, ignore_index=True)
    expected["input_date_count"] = len(frame)
    continuous = [
        "mean",
        "standard_error",
        "t_statistic",
        "ci_lower",
        "ci_upper",
        "p_value",
        "fdr_adjusted_p_value",
    ]
    continuous = [column for column in continuous if column in actual]
    pd.testing.assert_frame_equal(
        actual.drop(columns=continuous),
        expected.drop(columns=continuous),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual[continuous].isna(), expected[continuous].isna(), check_exact=True
    )
    pd.testing.assert_frame_equal(
        actual[continuous], expected[continuous], check_exact=False, rtol=1e-12, atol=0
    )
    for method in ("BH", "BY"):
        part = actual.loc[actual.fdr_method.eq(method)]
        assert part.factor_id.tolist() == list(planned)
        defined = part.status.eq("DEFINED")
        p_values = np.concatenate([part.loc[defined, "p_value"], [1.0, 1.0]])
        independent = statistics.multipletests(
            p_values, alpha=0.05, method="fdr_bh" if method == "BH" else "fdr_by"
        )[1]
        np.testing.assert_allclose(
            part.loc[defined, "fdr_adjusted_p_value"],
            independent[:2],
            rtol=1e-13,
            atol=0,
        )
        assert part.loc[~defined, "fdr_adjusted_p_value"].isna().all()


@pytest.mark.parametrize("age,half_life", [(30, 10.0), (503, 63.0)])
def test_ewma_history_is_invariant_under_joint_reordering(age, half_life):
    values = np.array([[0.03], [0.01], [0.02]])
    ages = np.array([age, 1, 0])
    weights = 2.0 ** (-ages / half_life)
    weights /= weights.sum()
    mean = np.average(values[:, 0], weights=weights)
    expected = np.average((values[:, 0] - mean) ** 2, weights=weights)
    results = []
    for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):
        results.append(
            _ewma_covariance(
                values[order],
                ages=ages[order],
                half_life=half_life,
                diagonal_shrinkage=0.0,
                annualization_factor=1.0,
                columns=("factor",),
            ).iloc[0, 0]
        )
    np.testing.assert_allclose(results, expected, rtol=1e-13, atol=1e-18)


def test_native_risk_state_keeps_recent_variation_within_factor_window():
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            minimum_cross_section=5,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
            specific_window=504,
            factor_half_life=63.0,
            annualization=1.0,
            diagonal_shrinkage=0.0,
        ),
    )
    first = _risk_cross([0.03] * 5, [1.0] * 5)
    state.observe_return_session(first)
    for _ in range(501):
        state.observe_return_session(first.iloc[:0])
    state.observe_return_session(_risk_cross([0.01] * 5, [1.0] * 5, "2023-01-03"))
    state.observe_return_session(_risk_cross([0.02] * 5, [1.0] * 5, "2023-01-04"))
    estimate = state.estimate()
    np.testing.assert_allclose(estimate.factor_returns.iloc[:, 1], [0.03, 0.01, 0.02])
    weights = 2.0 ** (-np.array([503, 1, 0]) / 63.0)
    mean = np.average([0.03, 0.01, 0.02], weights=weights)
    expected = np.average((np.array([0.03, 0.01, 0.02]) - mean) ** 2, weights=weights)
    assert estimate.factor_covariance.iloc[0, 0] == pytest.approx(expected, rel=1e-13)
    assert len(estimate.specific_variance) == 5
    assert estimate.specific_variance.eq(0.0).all()


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (np.zeros((2, 2)), True),
        (np.full((3, 3), 1e-4), True),
        (np.diag([-np.finfo(float).eps * 1e-4, 1e-4]), True),
        (np.diag([-1e-6, 1e-4]), False),
        (np.diag([1e-8, 1e-4]), True),
        (np.array([[1e-4, 2e-4], [2e-4, 1e-4]]), False),
    ],
)
def test_psd_check_distinguishes_roundoff_from_negative_risk(matrix, expected):
    from quant_research_platform.research.portfolios.risk import _validated_covariance

    if expected:
        np.testing.assert_array_equal(
            _validated_covariance(matrix, error_prefix="factor_risk_covariance"), matrix
        )
    else:
        with pytest.raises(ValueError, match="factor_risk_covariance_not_psd"):
            _validated_covariance(matrix, error_prefix="factor_risk_covariance")


def test_native_rank_one_industry_covariance_reaches_actual_optimizer():
    from quant_research_platform.contracts.portfolios import ExecutionAssumptions
    from quant_research_platform.research.portfolios.optimizer import optimize_portfolio

    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            factor_window=2,
            specific_window=2,
            minimum_cross_section=15,
            minimum_factor_dates=2,
            minimum_specific_dates=2,
            factor_half_life=1.0,
            annualization=1.0,
            diagonal_shrinkage=0.0,
        ),
    )
    frame = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": [f"asset{index}" for index in range(15)],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "total_market_cap": 1.0,
            "asset_return": 0.01,
        }
    )
    state.observe_return_session(frame)
    state.observe_return_session(
        frame.assign(timestamp=pd.Timestamp("2023-01-03"), asset_return=-0.01)
    )
    estimate = state.estimate()
    expected = float(Fraction(8, 9) * Fraction(0.01) ** 2)
    np.testing.assert_allclose(
        estimate.factor_covariance, np.full((3, 3), expected), rtol=2e-15, atol=0
    )
    exposures = state.current_exposures(frame)
    assets = exposures.assign(
        expected_alpha=0.0,
        current_weight=1 / 15,
        benchmark_weight=1 / 15,
        specific_variance=0.0,
        adv=1e9,
    )
    result = optimize_portfolio(
        with_cash_asset(assets),
        cash_column="is_cash",
        factor_covariance=estimate.factor_covariance,
        industry_columns=tuple(estimate.factor_covariance.columns),
        expected_transaction_cost=lambda buy, sell: (
            0.0,
            np.zeros_like(buy),
            np.zeros_like(sell),
        ),
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
    )
    assert np.isfinite(result.weights).all()
    assert (result.weights >= 0.0).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)
    active = result.weights.to_numpy()[:-1] - assets.benchmark_weight.to_numpy()
    assert active.sum() == pytest.approx(0.0, abs=1e-14)
    assert (
        float(assets.expected_alpha.to_numpy() @ result.weights.to_numpy()[:-1]) == 0.0
    )
    assert result.predicted_tracking_error == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    "kind,error",
    [
        ("axis", "optimizer_covariance_columns_invalid"),
        ("finite", "optimizer_covariance_invalid"),
        ("symmetry", "optimizer_covariance_invalid"),
        ("non_psd", "optimizer_covariance_not_psd"),
        ("bound", None),
        ("near_symmetric", None),
        ("duplicate", "optimizer_asset_duplicate"),
    ],
)
def test_optimizer_covariance_boundaries_and_unique_asset_requirement(kind, error):
    from quant_research_platform.contracts.portfolios import ExecutionAssumptions
    from quant_research_platform.research.portfolios.optimizer import optimize_portfolio

    assets = pd.DataFrame(
        {
            "asset_id": ["a", "a" if kind == "duplicate" else "b"],
            "expected_alpha": 0.0,
            "current_weight": 0.5,
            "benchmark_weight": 0.5,
            "specific_variance": 0.0,
            "adv": 1e9,
            "a": 1.0,
            "b": 1.0,
        }
    )
    values = np.eye(2)
    if kind == "finite":
        values[0, 0] = np.nan
    elif kind == "symmetry":
        values[0, 1] = 1.0
    elif kind == "non_psd":
        values[0, 0] = -1e-6
    elif kind == "bound":
        values[0, 0] = -np.finfo(float).eps
    elif kind == "near_symmetric":
        values = np.array([[1.0, 1.0 + np.finfo(float).eps], [1.0, 1.0]])
    columns = ("wrong", "b") if kind == "axis" else ("a", "b")
    covariance = pd.DataFrame(values, index=columns, columns=columns)
    from contextlib import nullcontext

    with pytest.raises(ValueError, match=f"^{error}$") if error else nullcontext():
        result = optimize_portfolio(
            with_cash_asset(assets),
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
        )
    if error is None:
        np.testing.assert_allclose(
            result.weights, [0.5, 0.5, 0.0], rtol=1e-12, atol=1e-14
        )


def test_named_quantile_recipe_preserves_order_and_nulls():
    from quant_research_platform.research.features.transforms import _winsorize

    values = pd.Series([-1.0, 1.0, np.nan, 1.0, 1.0])
    timestamps = pd.Series([0, 0, 0, 1, 1])
    result = _winsorize(values, values.groupby(timestamps))
    np.testing.assert_allclose(
        result.to_numpy(), [-0.98, 0.98, np.nan, 1.0, 1.0], rtol=1e-15
    )
    assert result.dtype == "float64"


def test_optimizer_risk_roundoff_is_bounded_and_rejects_nonfinite():
    from quant_research_platform.research.portfolios.optimizer import _risk_variances

    active = np.array([0.01, -0.01])
    exposure = np.eye(2)
    covariance = np.full((2, 2), 1e-4)
    factor, specific = _risk_variances(active, exposure, covariance, np.zeros(2))
    assert np.isfinite(factor) and factor >= 0 and specific == 0
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="optimizer_risk_nonfinite"):
            _risk_variances(
                np.array([np.inf, -np.inf]), exposure, covariance, np.zeros(2)
            )


@pytest.mark.parametrize(
    "values",
    [
        (-1.3267282928093849, 1.107768166657506),
        (-0.006926123563646536, 0.5015352951547885),
    ],
)
def test_native_zero_shrinkage_has_bounded_nonnegative_neutral_risk(values):
    from quant_research_platform.research.portfolios.optimizer import _risk_variances
    from quant_research_platform.research.portfolios.risk import _validated_covariance

    values = np.asarray(values)
    covariance = _ewma_covariance(
        np.array([values, -values]),
        ages=np.array([1.0, 0.0]),
        half_life=1.0,
        diagonal_shrinkage=0.0,
        annualization_factor=1.0,
        columns=("a", "b"),
    ).to_numpy()
    covariance = _validated_covariance(
        covariance, error_prefix="factor_risk_covariance"
    )
    active = np.array([values[1], -values[0]])
    factor, residual = _risk_variances(active, np.eye(2), covariance, np.zeros(2))
    scale = np.abs(active) @ np.abs(covariance) @ np.abs(active)
    assert 0 <= factor <= 16 * np.finfo(float).eps * scale
    assert residual == 0
