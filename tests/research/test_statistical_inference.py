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
    fama_macbeth,
    fdr_adjust,
    hac_mean,
    model_confidence_set,
    probabilistic_and_deflated_sharpe,
    spa_reality_check,
    stationary_block_length,
    stationary_bootstrap_mean_ci,
    stepm_superior_models,
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
    block_values = np.sin(np.arange(64, dtype="float64") / 3.0)
    assert stationary_block_length(block_values) >= 1
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
    benchmark = np.array([0.5, 0.4, 0.6, 0.7, 0.5, 0.6, 0.8, 0.7] * 2)
    models = pd.DataFrame(
        {
            "better": benchmark - 0.2,
            "same": benchmark,
            "worse": benchmark + 0.2,
        }
    )

    spa = spa_reality_check(benchmark, models, block_size=2, repetitions=49, seed=3)
    superior = stepm_superior_models(
        benchmark, models, block_size=2, repetitions=49, seed=3
    )
    mcs = model_confidence_set(models, block_size=2, repetitions=49, seed=3)
    direct_spa = SPA(benchmark, models, block_size=2, reps=49, seed=3)
    direct_spa.compute()
    direct_stepm = StepM(benchmark, models, block_size=2, reps=49, seed=3)
    direct_stepm.compute()
    direct_mcs = MCS(models, size=0.05, block_size=2, reps=49, seed=3)
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

    result = fama_macbeth(
        pd.DataFrame(rows),
        return_column="return",
        factor_columns=("factor",),
        max_lags=1,
    )

    expected_slopes = []
    for _, group in pd.DataFrame(rows).groupby("timestamp", sort=True):
        expected_slopes.append(
            OLS(group["return"], add_constant(group[["factor"]])).fit().params["factor"]
        )
    assert result.period_count == 4
    assert result.summary.loc["factor", "mean"] == pytest.approx(
        np.mean(expected_slopes)
    )
    assert result.white_p_values.notna().all()
    assert result.white_reasons.isna().all()
    weighted = fama_macbeth(
        pd.DataFrame(rows),
        return_column="return",
        factor_columns=("factor",),
        max_lags=1,
        weight_column="weight",
    )
    expected_weighted = []
    for _, group in pd.DataFrame(rows).groupby("timestamp", sort=True):
        expected_weighted.append(
            WLS(
                group["return"],
                add_constant(group[["factor"]]),
                weights=group["weight"],
            )
            .fit()
            .params["factor"]
        )
    assert weighted.summary.loc["factor", "mean"] == pytest.approx(
        np.mean(expected_weighted)
    )
    duplicate = pd.concat((pd.DataFrame(rows), pd.DataFrame(rows).iloc[[0]]))
    with pytest.raises(ValueError, match="^fama_macbeth_primary_key_duplicate$"):
        fama_macbeth(
            duplicate,
            return_column="return",
            factor_columns=("factor",),
            max_lags=1,
        )
