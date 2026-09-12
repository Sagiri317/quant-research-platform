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
        multipletests(
            [observed["p_value"], *([1.0] * 24)],
            alpha=0.05,
            method="fdr_bh",
        )[1][0]
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
            [-0.810567, -0.872156, -0.221970],
            [-0.051846, -2.276783, 0.925147],
            [-2.026846, 1.859624, 0.590568],
            [-0.471845, 1.342997, 0.020786],
            [0.690921, 0.107227, 1.096531],
            [1.061422, -0.907334, -0.612492],
        ]
    )

    assert first == second
    result = cscv_pbo(returns, slice_count=4, max_combinations=6)
    assert result.combination_count == 6
    assert result.status == "DEFINED"
    assert result.pbo == pytest.approx(1.0)
    duplicate = cscv_pbo(
        np.column_stack((values, values)), slice_count=4, max_combinations=6
    )
    assert duplicate.status == "UNDEFINED"
    assert duplicate.reason == "duplicate_candidate_series"
    with pytest.raises(ValueError, match="^cscv_combination_budget_exceeded$"):
        cscv_pbo(returns, slice_count=8, max_combinations=10)

    class RejectMaterialization:
        shape = (8, 3)

        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            raise AssertionError("matrix_materialized_before_combination_admission")

    with pytest.raises(ValueError, match="^cscv_combination_budget_exceeded$"):
        cscv_pbo(RejectMaterialization(), slice_count=8, max_combinations=10)


def test_psr_dsr_fixed_oracle() -> None:
    returns = np.array([0.02, -0.01, 0.03, 0.01, 0.00, 0.04, -0.02, 0.01])

    result = probabilistic_and_deflated_sharpe(
        returns,
        benchmark_sharpe=0.0,
        trial_sharpes=[0.1, 0.2, 0.3, 0.4, 0.5],
    )

    assert result.observed_sharpe == pytest.approx(0.5)
    assert result.psr == pytest.approx(0.8983492642265651)
    assert result.deflated_reference_sharpe == pytest.approx(0.18856566835299468)
    assert result.dsr == pytest.approx(0.7859405033926591)
    more_dispersed = probabilistic_and_deflated_sharpe(
        returns,
        benchmark_sharpe=0.0,
        trial_sharpes=[-0.5, 0.0, 0.5, 1.0, 1.5],
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

    spa = spa_reality_check(
        benchmark, models, block_size=20, repetitions=10000, seed=17
    )
    superior = stepm_superior_models(
        benchmark, models, block_size=20, repetitions=10000, seed=17
    )
    mcs = model_confidence_set(models, block_size=20, repetitions=10000, seed=17)
    direct_spa = SPA(benchmark, models, block_size=20, reps=10000, seed=17)
    direct_spa.compute()
    direct_stepm = StepM(benchmark, models, block_size=20, reps=10000, seed=17)
    direct_stepm.compute()
    direct_mcs = MCS(models, size=0.05, block_size=20, reps=10000, seed=17)
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
        benchmark * 1e-20, models * 1e-20, block_size=20, repetitions=10000, seed=17
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
            baseline, candidates, block_size=20, repetitions=10000, seed=17
        )
        step = stepm_superior_models(
            baseline, candidates, block_size=20, repetitions=10000, seed=17
        )
        mcs_degenerate = model_confidence_set(
            pd.concat((pd.Series(baseline, name="benchmark"), candidates), axis=1),
            block_size=20,
            repetitions=10000,
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
            benchmark, duplicated_challengers, block_size=20, repetitions=10000, seed=17
        ).status
        == "DEFINED"
    )
    assert (
        model_confidence_set(
            duplicated_challengers, block_size=20, repetitions=10000, seed=17
        ).status
        == "UNDEFINED"
    )


def test_multiple_comparison_rejects_disjoint_time_indexes() -> None:
    benchmark = pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2020-01-01", periods=3))
    models = pd.DataFrame(
        {"candidate": [0.0, 0.1, 0.2]},
        index=pd.date_range("2030-01-01", periods=3),
    )

    spa = spa_reality_check(benchmark, models, block_size=1, repetitions=9, seed=1)
    stepm = stepm_superior_models(
        benchmark, models, block_size=1, repetitions=9, seed=1
    )

    assert spa.status == "UNDEFINED"
    assert spa.reason == "insufficient_common_coverage"
    assert stepm.status == "UNDEFINED"
    duplicate_non_string_ids = pd.DataFrame(
        [[0.1, 0.2]] * 8,
        columns=[1, 1],
    )
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
    assert periods.white_p_value.notna().all()
    assert periods.white_reason.isna().all()
    np.testing.assert_allclose(weighted_slopes, expected_weighted)
    duplicate = pd.concat((group, group.iloc[[0]]))
    with pytest.raises(ValueError, match="^fama_macbeth_primary_key_duplicate$"):
        _fama_macbeth_period_records(
            duplicate, return_column="return", factors=("factor",), core_factors=()
        )


@pytest.mark.parametrize("kind", ["aligned", "finite"])
def test_matrix_cap_includes_conversion_and_live_temporaries(monkeypatch, kind):
    from quant_research_platform.research import statistics
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    rows, columns = 20000, 32
    values = np.arange(rows * columns, dtype="int32").reshape(rows, columns)
    frame = pd.DataFrame(values, columns=[f"c{i}" for i in range(columns)])
    cap = rows * (8 * columns + (2 if kind == "aligned" else 0)) + 65537
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2, explicit_cap_bytes=cap
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("conversion occurred before resource admission")

    with runtime_resource_policy_scope(policy):
        with monkeypatch.context() as patch:
            if kind == "aligned":
                patch.setattr(pd.DataFrame, "astype", forbidden)
            else:
                patch.setattr(np, "asarray", forbidden)
            with pytest.raises(
                MemoryError, match="statistical_matrix_exceeds_runtime_headroom"
            ):
                if kind == "aligned":
                    aligned_candidate_matrix(frame)
                else:
                    statistics._finite_matrix(values, minimum_rows=4, minimum_columns=2)
        assert policy.used_memory_bytes == 0
