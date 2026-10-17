from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from arch.bootstrap import MCS, SPA, StepM
from statsmodels.api import OLS, WLS, add_constant
from statsmodels.stats.multitest import multipletests

from quant_research_platform.research.statistics import (
    aligned_candidate_matrix,
    cscv_pbo,
    fama_macbeth_period,
    fdr_adjust,
    hac_mean,
    model_confidence_set,
    probabilistic_and_deflated_sharpe,
    spa_reality_check,
    stationary_bootstrap_mean_ci,
    stepm_superior_models,
    time_series_inference_summary,
)


def test_common_candidate_coverage_is_intersection_without_imputation() -> None:
    frame = pd.DataFrame({"a": [0.1, np.nan, 0.3, 0.4], "b": [0.2, 0.2, np.nan, 0.5]})
    aligned = aligned_candidate_matrix(frame, minimum_observations=2)
    assert aligned.status == "DEFINED"
    assert aligned.observation_count == 2
    np.testing.assert_allclose(aligned.values, [[0.1, 0.2], [0.4, 0.5]])
    assert aligned.dropped_observation_count == 2
    insufficient = aligned_candidate_matrix(frame.iloc[:2], minimum_observations=2)
    assert insufficient.status == "UNDEFINED"
    assert insufficient.reason == "insufficient_common_coverage"
    with pytest.raises(ValueError, match="^candidate_matrix_numeric_required$"):
        aligned_candidate_matrix(pd.DataFrame({"a": ["1", "2"]}))
    unordered = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3]},
        index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
    )
    with pytest.raises(ValueError, match="^candidate_matrix_time_order_invalid$"):
        aligned_candidate_matrix(unordered)


@pytest.mark.parametrize(
    ("method", "owner_method"), (("BH", "fdr_bh"), ("BY", "fdr_by"))
)
def test_hac_and_fdr_delegate_to_statsmodels(method: str, owner_method: str) -> None:
    values = np.array([0.2, -0.1, 0.3, 0.0, 0.4, 0.1], dtype="float64")
    expected_fit = OLS(values, np.ones((len(values), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    result = hac_mean(values, max_lags=1)
    adjusted = fdr_adjust([0.01, 0.04, 0.2], method=method, alpha=0.05)
    expected = multipletests([0.01, 0.04, 0.2], alpha=0.05, method=owner_method)
    assert result.mean == pytest.approx(expected_fit.params[0])
    assert result.standard_error == pytest.approx(expected_fit.bse[0])
    assert result.p_value == pytest.approx(expected_fit.pvalues[0])
    np.testing.assert_array_equal(adjusted.rejected, expected[0])
    np.testing.assert_allclose(adjusted.adjusted_p_values, expected[1])
    with pytest.raises(ValueError, match="^alpha_invalid$"):
        fdr_adjust([0.1], method="BH", alpha=-1.0)


def test_fdr_uses_fixed_family_size_without_persisting_missing_hypotheses() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "observed": [0.1, 0.2, 0.3],
            "missing": [np.nan, np.nan, np.nan],
        }
    )
    result = time_series_inference_summary(
        frame,
        identity_column="timestamp",
        max_lags=1,
        fdr_method="BH",
        fdr_alpha=0.05,
        fdr_family_size=25,
    )
    observed = result.loc[result["timestamp"].eq("observed")].iloc[0]
    missing = result.loc[result["timestamp"].eq("missing")].iloc[0]
    assert observed["fdr_adjusted_p_value"] == pytest.approx(
        multipletests([observed["p_value"], *[1.0] * 24], alpha=0.05, method="fdr_bh")[
            1
        ][0]
    )
    assert missing["status"] == "UNDEFINED"
    assert pd.isna(missing["p_value"])
    assert pd.isna(missing["fdr_adjusted_p_value"])


def test_stationary_bootstrap_and_cscv_are_deterministic_and_bounded() -> None:
    values = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.4, 0.0, 0.2])
    first = stationary_bootstrap_mean_ci(values, block_size=2, repetitions=99, seed=11)
    second = stationary_bootstrap_mean_ci(values, block_size=2, repetitions=99, seed=11)
    returns = np.array(
        [
            [1.101262, 0.338431, -0.539972],
            [-1.260242, -1.894621, 0.018638],
            [-0.810567, -0.872156, -0.22197],
            [-0.051846, -2.276783, 0.925147],
            [-2.026846, 1.859624, 0.590568],
            [-0.471845, 1.342997, 0.020786],
            [0.690921, 0.107227, 1.096531],
            [1.061422, -0.907334, -0.612492],
        ]
    )
    assert first == second
    assert first.mean == pytest.approx(0.1125)
    assert (first.lower, first.upper) == pytest.approx((0.0125, 0.219375))
    longer = stationary_bootstrap_mean_ci(values, block_size=3, repetitions=99, seed=11)
    assert (longer.lower, longer.upper) == pytest.approx((0.018125, 0.194375))
    assert longer.lower < longer.mean < longer.upper
    result = cscv_pbo(returns, slice_count=4)
    assert result.combination_count == 6
    assert result.status == "DEFINED"
    assert result.pbo == pytest.approx(1.0)
    duplicate = cscv_pbo(np.column_stack((values, values)), slice_count=4)
    assert duplicate.status == "UNDEFINED"
    assert duplicate.reason == "duplicate_candidate_series"


def test_psr_dsr_fixed_oracle() -> None:
    returns = np.array([0.02, -0.01, 0.03, 0.01, 0.0, 0.04, -0.02, 0.01])
    result = probabilistic_and_deflated_sharpe(
        returns, benchmark_sharpe=0.0, trial_sharpes=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert result.observed_sharpe == pytest.approx(0.5)
    assert result.psr == pytest.approx(0.8983492642265651)
    assert result.deflated_reference_sharpe == pytest.approx(0.18856566835299468)
    assert result.dsr == pytest.approx(0.7859405033926591)
    more_dispersed = probabilistic_and_deflated_sharpe(
        returns, benchmark_sharpe=0.0, trial_sharpes=[-0.5, 0.0, 0.5, 1.0, 1.5]
    )
    assert more_dispersed.deflated_reference_sharpe > result.deflated_reference_sharpe
    assert more_dispersed.dsr < result.dsr


def test_arch_multiple_comparison_adapters_keep_native_meanings() -> None:
    rng = np.random.default_rng(17)
    benchmark = rng.normal(0.0, 0.01, 64)
    models = pd.DataFrame(
        {
            "a": benchmark + rng.normal(0.001, 0.003, 64),
            "b": benchmark + rng.normal(-0.001, 0.004, 64),
        }
    )
    spa = spa_reality_check(benchmark, models, block_size=20, repetitions=199, seed=17)
    superior = stepm_superior_models(
        benchmark, models, block_size=20, repetitions=199, seed=17
    )
    mcs = model_confidence_set(models, block_size=20, repetitions=199, seed=17)
    direct_spa = SPA(benchmark, models, block_size=20, reps=199, seed=17)
    direct_spa.compute()
    direct_stepm = StepM(benchmark, models, block_size=20, reps=199, seed=17)
    direct_stepm.compute()
    direct_mcs = MCS(models, size=0.05, block_size=20, reps=199, seed=17)
    direct_mcs.compute()
    assert (
        0.0 <= spa.lower_p_value <= spa.consistent_p_value <= spa.upper_p_value <= 1.0
    )
    np.testing.assert_allclose(
        [spa.lower_p_value, spa.consistent_p_value, spa.upper_p_value],
        direct_spa.pvalues[["lower", "consistent", "upper"]],
    )
    assert superior.superior_models == tuple(direct_stepm.superior_models)
    assert mcs.included == tuple(direct_mcs.included)
    assert mcs.excluded == tuple(direct_mcs.excluded)
    pd.testing.assert_frame_equal(mcs.p_values, direct_mcs.pvalues)
    scaled = spa_reality_check(
        benchmark * 1e-20, models * 1e-20, block_size=20, repetitions=199, seed=17
    )
    assert scaled.status == "DEFINED"
    assert scaled.consistent_p_value == pytest.approx(spa.consistent_p_value)
    baseline = np.sin(np.arange(64)) / 100.0
    for candidates in (
        pd.DataFrame({"identical_a": baseline, "identical_b": baseline}),
        pd.DataFrame(
            {
                "identical": baseline,
                "dominated": baseline + 0.05 + np.cos(np.arange(64)) * 0.001,
            }
        ),
    ):
        observed = spa_reality_check(
            baseline, candidates, block_size=20, repetitions=199, seed=17
        )
        step = stepm_superior_models(
            baseline, candidates, block_size=20, repetitions=199, seed=17
        )
        mcs_degenerate = model_confidence_set(
            pd.concat((pd.Series(baseline, name="benchmark"), candidates), axis=1),
            block_size=20,
            repetitions=199,
            seed=17,
        )
        assert observed.status == step.status == mcs_degenerate.status == "UNDEFINED"
        assert (
            observed.reason
            == step.reason
            == mcs_degenerate.reason
            == "constant_loss_differential"
        )
        assert observed.consistent_p_value is None and mcs_degenerate.p_values is None
        assert observed.observation_count == 64
    duplicated_challengers = models.assign(duplicate=models["a"])
    assert (
        spa_reality_check(
            benchmark, duplicated_challengers, block_size=20, repetitions=199, seed=17
        ).status
        == "DEFINED"
    )
    assert (
        model_confidence_set(
            duplicated_challengers, block_size=20, repetitions=199, seed=17
        ).status
        == "UNDEFINED"
    )


def test_multiple_comparison_rejects_disjoint_time_indexes() -> None:
    benchmark = pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2020-01-01", periods=3))
    models = pd.DataFrame(
        {"candidate": [0.0, 0.1, 0.2]}, index=pd.date_range("2030-01-01", periods=3)
    )
    spa = spa_reality_check(benchmark, models, block_size=1, repetitions=9, seed=1)
    stepm = stepm_superior_models(
        benchmark, models, block_size=1, repetitions=9, seed=1
    )
    assert spa.status == "UNDEFINED"
    assert spa.reason == "insufficient_common_coverage"
    assert stepm.status == "UNDEFINED"
    duplicate_non_string_ids = pd.DataFrame([[0.1, 0.2]] * 8, columns=[1, 1])
    with pytest.raises(ValueError, match="^loss_matrix_candidate_id_invalid$"):
        spa_reality_check(
            pd.Series([0.3] * 8),
            duplicate_non_string_ids,
            block_size=1,
            repetitions=9,
            seed=1,
        )


def test_fama_macbeth_uses_period_cross_sections_then_hac() -> None:
    rows: list[dict[str, object]] = []
    for period, slope in enumerate((0.4, 0.5, 0.6, 0.7), start=1):
        for asset, x in enumerate((-2.0, -1.0, 1.0, 2.0), start=1):
            rows.append(
                {
                    "timestamp": pd.Timestamp(2020, period, 1),
                    "asset_id": f"asset-{asset}",
                    "return": 0.1 + slope * x + 0.01 * (asset % 2),
                    "factor": x,
                    "weight": float(asset),
                    "total_market_cap": float(asset),
                }
            )
    from quant_research_platform.research.statistics_production import (
        _fama_macbeth_period_records,
        _fama_macbeth_summary,
    )

    periods = []
    expected_slopes = []
    weighted_slopes = []
    expected_weighted = []
    for _, group in pd.DataFrame(rows).groupby("timestamp", sort=True):
        periods.extend(
            _fama_macbeth_period_records(
                group, return_column="return", factors=("factor",), core_factors=()
            )
        )
        expected_slopes.append(
            OLS(group["return"], add_constant(group[["factor"]])).fit().params["factor"]
        )
        weighted = fama_macbeth_period(
            group,
            return_column="return",
            factor_columns=("factor",),
            weight_column="weight",
        )
        weighted_slopes.append(weighted.coefficients["factor"])
        expected_weighted.append(
            WLS(
                group["return"],
                add_constant(group[["factor"]]),
                weights=group["weight"],
            )
            .fit()
            .params["factor"]
        )
        assert weighted.white_p_value is not None
        assert weighted.white_reason is None
    periods = pd.DataFrame(periods)
    summary = _fama_macbeth_summary(
        periods, factors=("factor",), core_factors=(), max_lags=1
    )
    slope = summary.loc[summary.coefficient_id.eq("factor")].iloc[0]
    assert slope.period_count == 4
    assert slope["mean"] == pytest.approx(np.mean(expected_slopes))
    from scipy.stats import norm

    se = 1 / 16
    expected_mean = 0.549
    assert slope.standard_error == pytest.approx(se)
    assert slope.t_statistic == pytest.approx(expected_mean / se)
    assert slope.p_value == pytest.approx(2 * norm.sf(abs(expected_mean / se)))
    assert (slope.ci_lower, slope.ci_upper) == pytest.approx(
        (expected_mean - norm.ppf(0.975) * se, expected_mean + norm.ppf(0.975) * se)
    )
    lag_zero = _fama_macbeth_summary(
        periods, factors=("factor",), core_factors=(), max_lags=0
    )
    assert lag_zero.loc[lag_zero.coefficient_id.eq("factor"), "standard_error"].iloc[
        0
    ] != pytest.approx(se)
    assert periods.white_p_value.notna().all()
    assert periods.white_reason.isna().all()
    np.testing.assert_allclose(weighted_slopes, expected_weighted)
    duplicate = pd.concat((group, group.iloc[[0]]))
    with pytest.raises(ValueError, match="^fama_macbeth_primary_key_duplicate$"):
        _fama_macbeth_period_records(
            duplicate, return_column="return", factors=("factor",), core_factors=()
        )


@pytest.mark.parametrize("count", (40, 80, 100))
def test_decimal_constant_cscv_is_undefined(count):
    matrix = np.column_stack((np.full(count, 0.1), np.linspace(-0.1, 0.2, count)))
    result = cscv_pbo(matrix, slice_count=10)
    assert result.status == "UNDEFINED"
    assert result.reason == "candidate_sharpe_undefined"


def test_nav_roundoff_is_explicit_and_retains_resolvable_variation():
    from quant_research_platform.research.portfolios.diagnostics import (
        _nav_return_roundoff,
    )
    from quant_research_platform.research.statistics import return_standard_deviation

    nav = 1000 * 1.1 ** np.arange(81)
    values = nav[1:] / nav[:-1] - 1.0
    error = _nav_return_roundoff(nav[:-1], nav[1:], values)
    assert np.ptp(values) > 0
    assert return_standard_deviation(values) > 0
    assert return_standard_deviation(values, roundoff=error) == 0
    with pytest.raises(ValueError, match="sharpe_zero_variance"):
        probabilistic_and_deflated_sharpe(
            values,
            benchmark_sharpe=0.0,
            trial_sharpes=[0.1, 0.2],
            return_roundoff=error,
        )
    varied = values + np.linspace(-1e-12, 1e-12, len(values))
    assert return_standard_deviation(varied, roundoff=error) > 0
    direct = np.array([0.1, np.nextafter(0.1, 1.0)] * 40)
    assert return_standard_deviation(direct) > 0
    matrix = np.column_stack((varied, np.linspace(-0.01, 0.02, len(values))))
    matrix[:40, 0] = values[:40]
    errors = np.column_stack((error, np.zeros(len(values))))
    result = cscv_pbo(matrix, slice_count=10, return_roundoff=errors)
    assert (
        result.status == "UNDEFINED" and result.reason == "candidate_sharpe_undefined"
    )
    result = probabilistic_and_deflated_sharpe(
        np.linspace(-0.1, 0.2, 80), benchmark_sharpe=0.0, trial_sharpes=[0.1] * 10
    )
    assert (
        result.dsr is None and result.dsr_reason == "trial_sharpe_dispersion_undefined"
    )


def test_white_lm_preserves_exact_auxiliary_fit_without_unused_f_warning():
    from scipy.stats import chi2

    from quant_research_platform.research.statistics_production import (
        _fama_macbeth_period_records,
    )

    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    z = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0])
    for day in range(40):
        frame = pd.DataFrame(
            {
                "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "asset_id": list("abcdef"),
                "x": x,
                "z": z,
                "return": 2 + day * x + 3 * z,
                "total_market_cap": 1.0,
            }
        )
        result = fama_macbeth_period(
            frame, return_column="return", factor_columns=("x",)
        )
        auxiliary_y = (3 * z) ** 2
        auxiliary_x = np.column_stack((np.ones(6), x, x * x))
        residual = (
            auxiliary_y
            - auxiliary_x @ np.linalg.lstsq(auxiliary_x, auxiliary_y, rcond=None)[0]
        )
        lm = 6 * (
            1
            - np.dot(residual, residual)
            / np.sum((auxiliary_y - auxiliary_y.mean()) ** 2)
        )
        assert result.white_p_value == pytest.approx(chi2.sf(lm, 2), abs=1e-12)
        records = _fama_macbeth_period_records(
            frame, return_column="return", factors=("x", "z"), core_factors=("x", "z")
        )
        assert records and all(row["status"] == "DEFINED" for row in records)


@pytest.mark.parametrize("exponent", [-700, 0, 700])
def test_sample_risk_and_sharpe_are_invariant_to_binary_units(exponent):
    import warnings

    from quant_research_platform.research.statistics import return_standard_deviation

    scale = 2.0**exponent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert return_standard_deviation([-scale, 0.0, scale]) == scale
        matrix = np.array([[-1.0, 0.0], [0.0, 2.0], [1.0, 4.0]]) * scale
        np.testing.assert_array_equal(
            return_standard_deviation(matrix), [scale, 2 * scale]
        )
        assert return_standard_deviation([scale] * 60) == 0.0
        result = probabilistic_and_deflated_sharpe(
            np.array([0.02, -0.01, 0.03, 0.01, 0.0, 0.04, -0.02, 0.01]) * scale,
            benchmark_sharpe=0.0,
            trial_sharpes=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
    assert result.observed_sharpe == pytest.approx(0.5)
    assert result.psr == pytest.approx(0.8983492642265651)
    assert result.dsr == pytest.approx(0.7859405033926591)


@pytest.mark.parametrize(
    "entry,native", [(spa_reality_check, SPA), (stepm_superior_models, StepM)]
)
@pytest.mark.parametrize("container", [np.asarray, list])
def test_multiple_comparison_aligns_unlabeled_benchmark_to_model_rows(
    entry, native, container
):
    rng = np.random.default_rng(17)
    benchmark = rng.normal(0.0, 0.01, 32)
    models = pd.DataFrame(
        {
            "a": benchmark + rng.normal(0.001, 0.003, 32),
            "b": benchmark + rng.normal(-0.001, 0.004, 32),
        },
        index=pd.date_range("2020-01-01", periods=32),
    )
    expected = native(benchmark, models, block_size=4, reps=99, seed=17)
    expected.compute()
    result = entry(container(benchmark), models, block_size=4, repetitions=99, seed=17)
    assert result.status == "DEFINED"
    assert result.observation_count == 32
    if entry is spa_reality_check:
        np.testing.assert_array_equal(
            [result.lower_p_value, result.consistent_p_value, result.upper_p_value],
            expected.pvalues[["lower", "consistent", "upper"]],
        )
    else:
        assert result.superior_models == tuple(expected.superior_models)
    with pytest.raises(ValueError, match="^loss_matrix_length_mismatch$"):
        entry(container(benchmark[:-1]), models, block_size=4, repetitions=99, seed=17)


def test_cscv_preserves_finite_extreme_scale_against_decimal_oracle():
    from decimal import Decimal, localcontext
    from itertools import combinations

    from quant_research_platform.research.statistics import _candidate_sharpes

    matrix = np.random.default_rng(431).uniform(0.2, 1.2, size=(8, 2))
    slices = np.array_split(np.arange(len(matrix)), 4)

    def oracle(values):
        with localcontext() as context:
            context.prec = 80
            scores = []
            for column in values.T:
                data = [Decimal.from_float(float(value)) for value in column]
                mean = sum(data) / len(data)
                deviation = (
                    sum((value - mean) ** 2 for value in data) / (len(data) - 1)
                ).sqrt()
                scores.append(float(mean / deviation))
            return np.asarray(scores)

    overfit = 0
    for selected in combinations(range(4), 2):
        train = np.concatenate([slices[index] for index in selected])
        test = np.concatenate(
            [slices[index] for index in range(4) if index not in selected]
        )
        winner = int(np.argmax(oracle(matrix[train])))
        overfit += oracle(matrix[test])[winner] < oracle(matrix[test])[1 - winner]
        for scale in (1.0, 1e308):
            np.testing.assert_allclose(
                _candidate_sharpes(matrix[train] * scale, np.zeros_like(matrix[train])),
                oracle(matrix[train] * scale),
                rtol=8 * np.finfo("float64").eps,
                atol=0.0,
            )
    for scale in (1.0, 1e308):
        result = cscv_pbo(matrix * scale, slice_count=4)
        assert result.status == "DEFINED"
        assert result.pbo == overfit / 6


def test_cscv_subnormal_observations_keep_original_roundoff_resolution():
    from quant_research_platform.research.statistics import _candidate_sharpes

    matrix = np.arange(1, 9, dtype="float64").reshape(4, 2) * 1e-310
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert _candidate_sharpes(matrix, np.ones_like(matrix)) is None
    for invalid in (-1.0, np.nan, -np.inf):
        with pytest.raises(ValueError, match="return_roundoff_invalid"):
            cscv_pbo(matrix, slice_count=4, return_roundoff=invalid)


@pytest.mark.parametrize(
    "values,errors,expected",
    [
        ([-1.7e308, 1.7e308], 0.0, 0.0),
        ([-1.7e308, 1.7e308], 1.7e308, None),
        ([1.7e308, 1.7e308], 1.7e308, None),
        ([-1.7e308, -1.7e308], 1.7e308, None),
        ([1e-310, 2e-310], 1.7e308, None),
        ([-1.7e308, 1.7e308], 1e308, 0.0),
    ],
)
def test_cscv_extreme_finite_ratios_and_uncertainty_intervals(values, errors, expected):
    from quant_research_platform.research.statistics import (
        _candidate_sharpes,
        return_standard_deviation,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = _candidate_sharpes(
            np.asarray(values)[:, None], np.full((len(values), 1), errors)
        )
        if expected is None:
            assert result is None
            assert return_standard_deviation(values, roundoff=errors) == 0.0
        else:
            np.testing.assert_array_equal(result, [expected])


def test_return_interval_overlap_preserves_extreme_and_subnormal_separation():
    from decimal import Decimal

    from quant_research_platform.research.statistics import _return_intervals_overlap

    limit = np.finfo("float64").max
    tiny = np.nextafter(0.0, 1.0)
    cases = [
        ([tiny, 2 * tiny, limit], [0.0, 0.0, limit]),
        ([-limit, limit], [limit, limit]),
        ([-limit, -limit], [0.0, 0.0]),
        ([limit, limit], [0.0, 0.0]),
        ([-limit, 0.0], [0.0, limit]),
        ([0.0, limit], [limit, 0.0]),
        ([-limit, 0.0, limit], [limit, 0.0, limit]),
        ([-limit, limit], [np.nextafter(limit, 0.0)] * 2),
    ]
    for values, errors in cases:
        lower = [
            Decimal.from_float(value) - Decimal.from_float(error)
            for value, error in zip(values, errors, strict=True)
        ]
        upper = [
            Decimal.from_float(value) + Decimal.from_float(error)
            for value, error in zip(values, errors, strict=True)
        ]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _return_intervals_overlap(np.asarray(values), np.asarray(errors))
        assert bool(result) == (max(lower) <= min(upper))


@pytest.mark.parametrize("exponent", [-500, 0, 500])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_return_roundoff_separates_intervals_with_equal_rounded_endpoints(
    exponent, sign
):
    from fractions import Fraction

    from quant_research_platform.research.statistics import return_standard_deviation

    values = np.ldexp(np.array([0.5, 1.0, 2.0]) * sign, exponent)
    errors = np.ldexp(np.array([0.5, 0.0, np.nextafter(1.0, 0.0)]), exponent)
    lower = [
        Fraction(float(value)) - Fraction(float(error))
        for value, error in zip(values, errors, strict=True)
    ]
    upper = [
        Fraction(float(value)) + Fraction(float(error))
        for value, error in zip(values, errors, strict=True)
    ]
    assert max(lower) > min(upper)
    assert return_standard_deviation(values, roundoff=errors) == pytest.approx(
        np.std(values, ddof=1), rel=1e-15, abs=0.0
    )
    errors[-1] = np.ldexp(1.0, exponent)
    assert return_standard_deviation(values, roundoff=errors) == 0.0


def test_cscv_keeps_disjoint_uncertainty_intervals_defined():
    matrix = np.tile([[1.0, 0.1], [2.0, 0.3]], (4, 1))
    errors = np.tile([[0.0, 0.0], [np.nextafter(1.0, 0.0), 0.0]], (4, 1))
    result = cscv_pbo(matrix, slice_count=4, return_roundoff=errors)
    assert result.status == "DEFINED"
    assert result.pbo == 0.0
    errors[1::2, 0] = 1.0
    result = cscv_pbo(matrix, slice_count=4, return_roundoff=errors)
    assert result.status == "UNDEFINED"
    assert result.reason == "candidate_sharpe_undefined"


@pytest.mark.parametrize("exponent", (-700, 0, 700))
def test_near_constant_sample_risk_preserves_exact_center(exponent):
    from decimal import Decimal, localcontext

    from quant_research_platform.research.statistics import (
        _candidate_sharpes,
        return_standard_deviation,
    )

    values = np.ldexp(np.array([1.0, np.nextafter(1.0, np.inf)]), exponent)
    with localcontext() as context:
        context.prec = 90
        exact = [Decimal.from_float(value) for value in values]
        mean = sum(exact) / 2
        deviation = ((exact[1] - exact[0]) ** 2 / 2).sqrt()
        expected_sharpe = float(mean / deviation)
    assert float(return_standard_deviation(values)) == pytest.approx(
        float(deviation), rel=2e-15, abs=0
    )
    np.testing.assert_allclose(
        _candidate_sharpes(values[:, None], np.zeros((2, 1))),
        [expected_sharpe],
        rtol=2e-15,
    )


def test_extreme_normal_nav_sharpe_matches_decimal_oracle():
    from decimal import Decimal, localcontext

    from scipy.stats import norm

    from quant_research_platform.research.portfolios.diagnostics import (
        _nav_return_roundoff,
    )

    exponents = [0]
    for _ in range(2):
        exponents.append(exponents[-1] + 1023)
        for _ in range(29):
            exponents.append(exponents[-1] - 51)
    nav = np.ldexp(np.ones(61), exponents)
    values = nav[1:] / nav[:-1] - 1
    assert np.isfinite(nav).all() and (nav >= np.finfo(float).tiny).all()
    np.testing.assert_array_equal(nav[:-1] * (1 + values), nav[1:])
    with localcontext() as context:
        context.prec = 100
        data = [Decimal.from_float(value) for value in values]
        n = Decimal(len(data))
        mean = sum(data) / n
        diffs = [value - mean for value in data]
        m2 = sum(value**2 for value in diffs) / n
        observed = mean / (m2 * n / (n - 1)).sqrt()
        m3 = sum(value**3 for value in diffs) / n
        m4 = sum(value**4 for value in diffs) / n
        skew = (n * (n - 1)).sqrt() / (n - 2) * m3 / (m2 * m2.sqrt())
        kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / m2**2 - 3) + 6) + 3
        variance = (1 - skew * observed + (kurtosis - 1) / 4 * observed**2) / (n - 1)
        expected_psr = norm.cdf(float(observed / variance.sqrt()))
    result = probabilistic_and_deflated_sharpe(
        values,
        benchmark_sharpe=0.0,
        trial_sharpes=[0.1, 0.2, 0.3],
        return_roundoff=_nav_return_roundoff(nav[:-1], nav[1:], values),
    )
    assert result.observed_sharpe == pytest.approx(float(observed), rel=2e-14)
    assert result.psr == pytest.approx(expected_psr, rel=2e-14)


def test_fama_macbeth_preserves_small_factor_units_and_rejects_collinearity():
    from decimal import Decimal, localcontext

    from quant_research_platform.research.statistics_production import (
        _fama_macbeth_period_records,
    )

    x = np.linspace(5e-14, 2e-13, 1000)
    y = 0.01 + 100000000000.0 * x + 0.001 * np.sin(np.arange(len(x)))
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"s{i}" for i in range(len(x))],
            "total_market_cap": 1.0,
            "x": x,
            "y": y,
        }
    )
    with localcontext() as context:
        context.prec = 80
        xx = [Decimal.from_float(value) for value in x]
        yy = [Decimal.from_float(value) for value in y]
        mx, my = (sum(xx) / len(xx), sum(yy) / len(yy))
        slope = sum(((a - mx) * (b - my) for a, b in zip(xx, yy, strict=True))) / sum(
            (a - mx) ** 2 for a in xx
        )
        intercept = my - slope * mx
    for scale in (1.0, 1000000000000.0):
        result = fama_macbeth_period(
            frame.assign(x=x * scale), return_column="y", factor_columns=("x",)
        )
        assert result.coefficients["x"] * scale == pytest.approx(
            float(slope), rel=1e-12
        )
        assert result.coefficients["const"] == pytest.approx(
            float(intercept), rel=1e-12
        )
        assert result.white_p_value is not None
    rows = _fama_macbeth_period_records(
        frame, return_column="y", factors=("x",), core_factors=("x",)
    )
    assert all(row["status"] == "DEFINED" for row in rows)
    with pytest.raises(ValueError, match="fama_macbeth_period_insufficient"):
        fama_macbeth_period(
            frame.assign(z=x * 2), return_column="y", factor_columns=("x", "z")
        )


@pytest.mark.parametrize("capital", (np.nextafter(0.0, 1.0), np.finfo(float).max))
def test_unbounded_nav_roundoff_is_undefined_not_zero_risk(capital):
    from quant_research_platform.research.portfolios.diagnostics import (
        _nav_return_roundoff,
    )
    from quant_research_platform.research.statistics import (
        _candidate_sharpes,
        return_standard_deviation,
    )

    values = np.zeros(4)
    errors = _nav_return_roundoff(np.full(4, capital), np.full(4, capital), values)
    assert np.isposinf(errors).all()
    assert np.isnan(return_standard_deviation(values, roundoff=errors))
    assert _candidate_sharpes(values[:, None], errors[:, None]) is None
    with pytest.raises(ValueError, match="sharpe_zero_variance"):
        probabilistic_and_deflated_sharpe(
            values,
            benchmark_sharpe=0.0,
            trial_sharpes=[0.1, 0.2],
            return_roundoff=errors,
        )


@pytest.mark.parametrize("sign", [1, -1])
def test_common_sample_does_not_hide_infinity_behind_another_missing_candidate(sign):
    frame = pd.DataFrame({"a": [1.0, sign * np.inf, 2.0], "b": [3.0, np.nan, 4.0]})
    with pytest.raises(ValueError, match="candidate_matrix_nonfinite"):
        aligned_candidate_matrix(frame)
    frame.loc[1, "a"] = np.nan
    result = aligned_candidate_matrix(frame)
    assert result.observation_count == 2
    assert result.dropped_observation_count == 1


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_summary_confidence_matches_manual_bartlett_interval(confidence):
    from statistics import NormalDist

    values = np.array([0.02, -0.01, 0.04, 0.01, 0.03, -0.02, 0.05, 0.01])
    centered = values - values.mean()
    standard_error = np.sqrt(centered @ centered + centered[1:] @ centered[:-1]) / len(
        values
    )
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    row = time_series_inference_summary(
        pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=len(values)), "x": values}
        ),
        identity_column="timestamp",
        max_lags=1,
        fdr_method="BY",
        fdr_alpha=0.05,
        confidence=confidence,
    ).iloc[0]
    assert row.standard_error == pytest.approx(standard_error, rel=1e-14)
    np.testing.assert_allclose(
        [row.ci_lower, row.ci_upper],
        values.mean() + np.array([-1, 1]) * critical * standard_error,
        rtol=1e-14,
    )


@pytest.mark.parametrize("values", [[], [np.nan, np.nan], [0.2, 0.2]])
@pytest.mark.parametrize("lag", [-1, True, 1.5])
def test_summary_rejects_invalid_lag_even_without_defined_inference(values, lag):
    with pytest.raises(ValueError, match="hac_max_lags_invalid"):
        time_series_inference_summary(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=len(values)),
                    "x": values,
                }
            ),
            identity_column="timestamp",
            max_lags=lag,
            fdr_method="BY",
            fdr_alpha=0.05,
        )


@pytest.mark.parametrize("confidence", [True, 0.0, 1.0, float("nan")])
def test_empty_summary_still_validates_confidence(confidence):
    with pytest.raises(ValueError, match="confidence_invalid"):
        time_series_inference_summary(
            pd.DataFrame({"timestamp": pd.DatetimeIndex([]), "x": []}),
            identity_column="timestamp",
            max_lags=0,
            fdr_method="BY",
            fdr_alpha=0.05,
            confidence=confidence,
        )


@pytest.mark.parametrize("lag", [-1, True, 1.5])
def test_factor_configuration_rejects_invalid_lag_before_opening_data(tmp_path, lag):
    from quant_research_platform.research.statistics_production import factor_statistics

    with pytest.raises(ValueError, match="factor_statistics_policy_invalid"):
        factor_statistics(
            {
                "dataset": str(tmp_path / "absent"),
                "output": str(tmp_path / "result"),
                "window": {"start": "2024-01-01", "end": "2024-01-02"},
                "evaluation": {"primary_label": "label", "factors": ["x"]},
                "analyses": ["fama_macbeth"],
                "statistics": {"hac_lag": lag, "fama_macbeth_core_factors": ["x"]},
            }
        )
