from __future__ import annotations

from decimal import Decimal, localcontext
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


def test_constant_extreme_wls_has_exact_zero_residuals():
    fitted = FactorRiskState(style_columns=()).fit_return_session(
        _risk_cross([1e308] * 100, [1e308] * 100)
    )
    assert fitted is not None
    assert fitted.factor_returns.tolist() == [1e308]
    assert fitted.residuals.eq(0.0).all()


def test_extreme_wls_preserves_sqrt_cap_weighted_mean_and_residuals():
    returns = np.tile([1e308, -1e308, 8e307, -2e307], 25)
    caps = np.tile([1e308, 2.5e307, 6.25e306, 1.5625e306], 25)
    with localcontext() as context:
        context.prec = 100
        weights = [Decimal.from_float(float(cap)).sqrt() for cap in caps]
        expected = float(
            sum(
                weight * Decimal.from_float(float(value))
                for weight, value in zip(weights, returns, strict=True)
            )
            / sum(weights)
        )
        residuals = [
            float(Decimal.from_float(float(value)) - Decimal.from_float(expected))
            for value in returns
        ]
    fitted = FactorRiskState(style_columns=()).fit_return_session(
        _risk_cross(returns, caps)
    )
    assert fitted is not None
    np.testing.assert_allclose(fitted.factor_returns, [expected], rtol=2e-14, atol=0)
    np.testing.assert_allclose(fitted.residuals, residuals, rtol=2e-14, atol=0)
    assert np.isfinite(fitted.residuals).all()


@pytest.mark.parametrize(
    "values,annualization",
    [
        ([-1e154, 1e154], 1.0),
        ([-2e-162, 2e-162], 1.0),
        ([1e100, np.nextafter(1e100, np.inf)], 1.0),
        ([1e308, 1e308], 1.0),
        ([1e308, np.nextafter(1e308, np.inf)], 1e-300),
    ],
)
def test_ewma_variance_matches_fraction_oracle_before_final_rounding(
    values, annualization
):
    exact = [Fraction(float(value)) for value in values]
    mean = sum(exact) / len(exact)
    expected = float(
        sum((value - mean) ** 2 for value in exact)
        / len(exact)
        * Fraction(annualization)
    )
    actual = _ewma_covariance(
        np.asarray(values)[:, None],
        ages=np.zeros(len(values), dtype=int),
        half_life=1.0,
        diagonal_shrinkage=0.0,
        annualization_factor=annualization,
        columns=("factor",),
    )
    assert actual.iloc[0, 0] == pytest.approx(expected, rel=2e-15, abs=0)


def test_ewma_covariance_keeps_subnormal_cross_terms_and_shrinkage():
    values = np.array([[-2e-162, 2e-162], [2e-162, -2e-162]])
    variance = Fraction(float(values[0, 0])) ** 2
    for shrinkage in (0.0, 0.2, 1.0):
        actual = _ewma_covariance(
            values,
            ages=np.array([0, 0]),
            half_life=1.0,
            diagonal_shrinkage=shrinkage,
            annualization_factor=1.0,
            columns=("a", "b"),
        )
        expected = np.array(
            [
                [float(variance), float(-(1 - Fraction(shrinkage)) * variance)],
                [float(-(1 - Fraction(shrinkage)) * variance), float(variance)],
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, actual.T)


def test_ewma_tiny_tail_is_multiplied_before_float64_rounding():
    weight = Fraction(1, 2**2048)
    expected = float(weight / (1 + weight) ** 2 * Fraction(1e308) ** 2)
    actual = _ewma_covariance(
        np.array([[1e308], [0.0]]),
        ages=np.array([2048, 0]),
        half_life=1.0,
        diagonal_shrinkage=0.0,
        annualization_factor=1.0,
        columns=("factor",),
    )
    assert actual.iloc[0, 0] == pytest.approx(expected, rel=2e-15, abs=0)
    assert actual.iloc[0, 0] > 0


def test_specific_variance_keeps_subnormal_residual_mass():
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            factor_window=2,
            specific_window=2,
            minimum_factor_dates=2,
            minimum_specific_dates=2,
            annualization=1.0,
            diagonal_shrinkage=0.0,
        ),
    )
    returns = np.tile([-2e-162, 2e-162], 50)
    for day in ("2023-01-02", "2023-01-03"):
        state.observe_return_session(_risk_cross(returns, np.ones(100), day))
    estimated = state.estimate()
    assert estimated.specific_variance.eq(float(Fraction(2e-162) ** 2)).all()


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
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for method in ("BH", "BY"):
        part = actual.loc[actual.fdr_method.eq(method)]
        assert part.factor_id.tolist() == list(planned)
        defined = part.status.eq("DEFINED")
        p_values = np.concatenate([part.loc[defined, "p_value"], [1.0, 1.0]])
        independent = statistics.multipletests(
            p_values, alpha=0.05, method="fdr_bh" if method == "BH" else "fdr_by"
        )[1]
        np.testing.assert_array_equal(
            part.loc[defined, "fdr_adjusted_p_value"], independent[:2]
        )
        assert part.loc[~defined, "fdr_adjusted_p_value"].isna().all()


@pytest.mark.parametrize("age,half_life", [(3000, 1.0), (503, 0.125)])
def test_ewma_large_old_origin_keeps_recent_variation_under_joint_reordering(
    age, half_life
):
    values = np.array([[1e300], [1.0], [2.0]])
    ages = np.array([age, 1, 0])
    weights = [
        Fraction(2) ** (-int(item) * (1 if half_life == 1 else 8)) for item in ages
    ]
    weights = [weight / sum(weights) for weight in weights]
    exact = [Fraction(float(value)) for value in values[:, 0]]
    mean = sum(weight * value for weight, value in zip(weights, exact, strict=True))
    expected = float(
        sum(
            weight * (value - mean) ** 2
            for weight, value in zip(weights, exact, strict=True)
        )
    )
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
    assert results == [expected] * 3


def test_native_risk_state_keeps_recent_variation_within_factor_window():
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            minimum_cross_section=5,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
            specific_window=504,
            factor_half_life=0.125,
            annualization=1.0,
            diagonal_shrinkage=0.0,
        ),
    )
    first = _risk_cross([1e300] * 5, [1.0] * 5)
    state.observe_return_session(first)
    for _ in range(501):
        state.observe_return_session(first.iloc[:0])
    state.observe_return_session(_risk_cross([1.0] * 5, [1.0] * 5, "2023-01-03"))
    state.observe_return_session(_risk_cross([2.0] * 5, [1.0] * 5, "2023-01-04"))
    estimate = state.estimate()
    assert estimate.factor_returns.iloc[:, 1].tolist() == [1e300, 1.0, 2.0]
    assert estimate.factor_covariance.iloc[0, 0] == float(Fraction(256, 257**2))
    assert len(estimate.specific_variance) == 5
    assert estimate.specific_variance.eq(0.0).all()


def test_unrepresentable_final_wls_residual_fails_before_observation():
    import warnings

    maximum = np.finfo(np.float64).max
    frame = _risk_cross(
        [maximum, -maximum, maximum, -maximum, maximum], [1.0, 4.0, 16.0, 64.0, 256.0]
    )
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            factor_window=1,
            specific_window=1,
            minimum_cross_section=5,
            minimum_factor_dates=1,
            minimum_specific_dates=1,
        ),
    )
    state.observe_return_session(_risk_cross([1.0] * 5, [1.0] * 5))
    before = state.estimate()
    frame["industry_level1_code"] = "new_industry"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="^factor_risk_fit_non_finite$"):
            state.fit_return_session(frame)
        with pytest.raises(ValueError, match="^factor_risk_fit_non_finite$"):
            state.observe_return_session(frame)
    after = state.estimate()
    pd.testing.assert_frame_equal(
        before.factor_returns, after.factor_returns, check_exact=True
    )
    pd.testing.assert_frame_equal(
        before.factor_covariance, after.factor_covariance, check_exact=True
    )
    pd.testing.assert_series_equal(
        before.specific_variance, after.specific_variance, check_exact=True
    )


@pytest.mark.parametrize("maximum,annualization", [(1e100, 1.0), (1e160, 1e-40)])
def test_ewma_cross_moment_cancellation_matches_fraction_oracle(maximum, annualization):
    values = np.array(
        [
            [maximum, maximum],
            [maximum, -maximum],
            [-maximum, maximum],
            [-maximum, -maximum],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    exact = [[Fraction(float(value)) for value in row] for row in values]
    means = [sum(row[column] for row in exact) / len(exact) for column in range(2)]
    expected = np.array(
        [
            [
                float(
                    sum(
                        (row[left] - means[left]) * (row[right] - means[right])
                        for row in exact
                    )
                    / len(exact)
                    * Fraction(annualization)
                )
                for right in range(2)
            ]
            for left in range(2)
        ]
    )
    for order in (np.arange(6), np.arange(6)[::-1], np.array([4, 5, 2, 0, 3, 1])):
        actual = _ewma_covariance(
            values[order],
            ages=np.zeros(6, dtype=int),
            half_life=1.0,
            diagonal_shrinkage=0.0,
            annualization_factor=annualization,
            columns=("a", "b"),
        )
        np.testing.assert_array_equal(actual, expected)


def test_native_risk_state_keeps_float_weight_recent_variation():
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            minimum_cross_section=5,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
            specific_window=504,
            factor_half_life=0.5,
            annualization=1.0,
            diagonal_shrinkage=0.0,
        ),
    )
    first = _risk_cross([1e100] * 5, [1.0] * 5)
    state.observe_return_session(first)
    for _ in range(501):
        state.observe_return_session(first.iloc[:0])
    state.observe_return_session(_risk_cross([1.0] * 5, [1.0] * 5, "2023-01-03"))
    state.observe_return_session(_risk_cross([2.0] * 5, [1.0] * 5, "2023-01-04"))
    estimate = state.estimate()
    assert estimate.factor_covariance.iloc[0, 0] == pytest.approx(
        float(Fraction(4, 25)), rel=2e-15, abs=0
    )
    assert len(estimate.specific_variance) == 5
    assert estimate.specific_variance.eq(0.0).all()


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (np.full((3, 3), 1e100), True),
        (np.diag([-1e-12, 1.0]), True),
        (np.diag([np.nextafter(-1e-12, 0.0), 1.0]), True),
        (np.diag([np.nextafter(-1e-12, -np.inf), 1.0]), False),
        (np.diag([1e-12, 1.0]), True),
        (np.array([[-1e-12, 1e-12], [1e-12, 1.0]]), False),
        (np.diag([-1e-12, -1e-12, 1.0]), True),
        (np.array([[0.0, 1e-12], [1e-12, 0.0]]), True),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), False),
        (np.array([[1e100, 2e100], [2e100, 1e100]]), False),
        (np.array([[1.0, 1.000001], [1.0, 1.0]]), True),
    ],
)
def test_psd_bound_uses_exact_binary64_matrix_and_original_tolerance(matrix, expected):
    from quant_research_platform.research.portfolios.risk import _covariance_is_psd

    assert _covariance_is_psd(matrix) is expected


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
            "asset_return": 1e50,
        }
    )
    state.observe_return_session(frame)
    state.observe_return_session(
        frame.assign(timestamp=pd.Timestamp("2023-01-03"), asset_return=-1e50)
    )
    estimate = state.estimate()
    expected = float(Fraction(8, 9) * Fraction(1e50) ** 2)
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
    assert sum(map(Fraction, active)) == 0
    assert (
        float(assets.expected_alpha.to_numpy() @ result.weights.to_numpy()[:-1]) == 0.0
    )
    assert result.predicted_tracking_error == 0.0


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
        values[0, 0] = np.nextafter(-1e-12, -np.inf)
    elif kind == "bound":
        values[0, 0] = -1e-12
    elif kind == "near_symmetric":
        values = np.array([[1.0, 1.000001], [1.0, 1.0]])
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
        np.testing.assert_array_equal(result.weights, [0.5, 0.5, 0.0])


def test_named_quantile_recipe_keeps_extreme_interpolation_order_and_nulls():
    from quant_research_platform.research.features.transforms import _winsorize

    values = pd.Series([-1e308, 1e308, np.nan, 1e308, 1e308])
    timestamps = pd.Series([0, 0, 0, 1, 1])
    result = _winsorize(values, values.groupby(timestamps))
    np.testing.assert_allclose(
        result.to_numpy(), [-9.8e307, 9.8e307, np.nan, 1e308, 1e308], rtol=1e-15
    )
    assert result.dtype == "float64"


def test_optimizer_blocks_genuinely_negative_risk_inside_psd_admission():
    from quant_research_platform.contracts.portfolios import ExecutionAssumptions
    from quant_research_platform.research.portfolios.optimizer import optimize_portfolio
    from quant_research_platform.research.portfolios.risk import _covariance_is_psd

    covariance = pd.DataFrame([[-1e-12]], index=["style"], columns=["style"])
    assert _covariance_is_psd(covariance.to_numpy())
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "expected_alpha": [1.0, -1.0],
            "current_weight": [0.51, 0.49],
            "benchmark_weight": [0.5, 0.5],
            "specific_variance": 0.0,
            "adv": 1e9,
            "style": [1.0, -1.0],
        }
    )
    with pytest.raises(ValueError, match="optimizer_risk_negative"):
        optimize_portfolio(
            with_cash_asset(assets),
            cash_column="is_cash",
            factor_covariance=covariance,
            style_columns=("style",),
            expected_transaction_cost=lambda buy, sell: (
                0.0,
                np.zeros_like(buy),
                np.zeros_like(sell),
            ),
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        )


def test_optimizer_risk_roundoff_is_bounded_and_rejects_nonfinite():
    from quant_research_platform.research.portfolios.optimizer import _risk_variances

    active = np.array([0.01, -0.01])
    exposure = np.eye(2)
    covariance = np.full((2, 2), 1e100)
    factor, specific = _risk_variances(active, exposure, covariance, np.zeros(2))
    assert np.isfinite(factor) and factor >= 0 and specific == 0
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="optimizer_risk_nonfinite"):
            _risk_variances(
                np.array([1e200, -1e200]), exposure, covariance, np.zeros(2)
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
    from quant_research_platform.research.portfolios.risk import _covariance_is_psd

    values = np.asarray(values)
    covariance = _ewma_covariance(
        np.array([values, -values]),
        ages=np.array([1.0, 0.0]),
        half_life=1.0,
        diagonal_shrinkage=0.0,
        annualization_factor=1.0,
        columns=("a", "b"),
    ).to_numpy()
    assert _covariance_is_psd(covariance)
    active = np.array([values[1], -values[0]])
    factor, residual = _risk_variances(active, np.eye(2), covariance, np.zeros(2))
    scale = np.abs(active) @ np.abs(covariance) @ np.abs(active)
    assert 0 <= factor <= 16 * np.finfo(float).eps * scale
    assert residual == 0
