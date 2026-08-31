"""QRP input/status contracts delegating numerical inference to dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, e
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS, SPA, StationaryBootstrap, StepM, optimal_block_length
from scipy.stats import kurtosis, norm, skew
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.multitest import multipletests

from quant_research_platform.contracts.validation import (
    strict_hashable_key_series,
    strict_timestamp_series,
)
from quant_research_platform.runtime.resources import current_runtime_resource_policy


@dataclass(frozen=True)
class AlignedCandidateMatrix:
    values: np.ndarray
    index: pd.Index
    candidate_ids: tuple[str, ...]
    observation_count: int
    dropped_observation_count: int
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None


@dataclass(frozen=True)
class MeanInference:
    mean: float
    standard_error: float
    t_statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    observation_count: int


@dataclass(frozen=True)
class FdrAdjustment:
    rejected: np.ndarray
    adjusted_p_values: np.ndarray
    method: Literal["BH", "BY"]
    alpha: float


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float
    block_size: int
    repetitions: int
    seed: int


def stationary_block_length(values: object) -> int:
    """Return arch's stationary-bootstrap block length, rounded up."""

    data = _finite_vector(values, minimum=3)
    estimate = float(optimal_block_length(data).iloc[0]["stationary"])
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("bootstrap_block_length_undefined")
    return int(np.ceil(estimate))


@dataclass(frozen=True)
class SpaResult:
    lower_p_value: float | None
    consistent_p_value: float | None
    upper_p_value: float | None
    observation_count: int
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None


@dataclass(frozen=True)
class StepMResult:
    superior_models: tuple[str, ...]
    observation_count: int
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None


@dataclass(frozen=True)
class ModelConfidenceSetResult:
    p_values: pd.DataFrame | None
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    observation_count: int
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None


@dataclass(frozen=True)
class SharpeInference:
    observed_sharpe: float
    benchmark_sharpe: float
    psr: float
    deflated_reference_sharpe: float
    dsr: float
    observation_count: int
    trial_count: int


@dataclass(frozen=True)
class PboResult:
    pbo: float | None
    slice_count: int
    combination_count: int
    candidate_count: int
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None


@dataclass(frozen=True)
class FamaMacBethResult:
    period_coefficients: pd.DataFrame
    summary: pd.DataFrame
    white_p_values: pd.Series
    white_reasons: pd.Series
    period_count: int


def aligned_candidate_matrix(
    frame: pd.DataFrame, *, minimum_observations: int = 2
) -> AlignedCandidateMatrix:
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
        raise ValueError("candidate_matrix_invalid")
    if type(minimum_observations) is not int or minimum_observations < 2:
        raise ValueError("candidate_minimum_observations_invalid")
    if (
        frame.columns.has_duplicates
        or any(type(column) is not str or not column for column in frame.columns)
        or not frame.index.is_unique
    ):
        raise ValueError("candidate_matrix_identity_invalid")
    if not isinstance(frame.index, (pd.DatetimeIndex, pd.RangeIndex)) or not (
        frame.index.is_monotonic_increasing
    ):
        raise ValueError("candidate_matrix_time_order_invalid")
    if any(
        not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
        for column in frame.columns
    ):
        raise ValueError("candidate_matrix_numeric_required")
    policy = current_runtime_resource_policy()
    snapshot_bytes = len(frame) * (8 * frame.shape[1] + 2) + 64 * 1024
    if policy is not None and snapshot_bytes > policy.operation_headroom_bytes:
        raise MemoryError("statistical_matrix_exceeds_runtime_headroom")
    numeric = frame.astype("float64")
    complete = numeric.notna().all(axis=1)
    values = numeric.loc[complete].to_numpy(dtype="float64", copy=True)
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("candidate_matrix_nonfinite")
    count = len(values)
    return AlignedCandidateMatrix(
        values=values,
        index=numeric.index[complete].copy(),
        candidate_ids=tuple(str(column) for column in frame.columns),
        observation_count=count,
        dropped_observation_count=len(frame) - count,
        status="DEFINED" if count >= minimum_observations else "UNDEFINED",
        reason=None
        if count >= minimum_observations
        else "insufficient_common_coverage",
    )


def hac_mean(
    values: object, *, max_lags: int, confidence: float = 0.95
) -> MeanInference:
    data = _finite_vector(values, minimum=2)
    if type(max_lags) is not int or max_lags < 0 or max_lags >= len(data):
        raise ValueError("hac_max_lags_invalid")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_invalid")
    fit = sm.OLS(data, np.ones((len(data), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lags}
    )
    interval = fit.conf_int(alpha=1.0 - confidence)[0]
    return MeanInference(
        mean=float(fit.params[0]),
        standard_error=float(fit.bse[0]),
        t_statistic=float(fit.tvalues[0]),
        p_value=float(fit.pvalues[0]),
        confidence_interval=(float(interval[0]), float(interval[1])),
        observation_count=len(data),
    )


def fdr_adjust(
    p_values: object, *, method: Literal["BH", "BY"], alpha: float = 0.05
) -> FdrAdjustment:
    values = _finite_vector(p_values, minimum=1)
    _validate_alpha(alpha)
    if bool(((values < 0.0) | (values > 1.0)).any()) or method not in {"BH", "BY"}:
        raise ValueError("fdr_input_invalid")
    rejected, adjusted, _, _ = multipletests(
        values, alpha=alpha, method="fdr_bh" if method == "BH" else "fdr_by"
    )
    return FdrAdjustment(rejected, adjusted, method, alpha)


def stationary_bootstrap_mean_ci(
    values: object,
    *,
    block_size: int,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapInterval:
    data = _finite_vector(values, minimum=2)
    _validate_bootstrap_configuration(block_size, repetitions, seed)
    if type(confidence) is not float or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_invalid")
    bootstrap = StationaryBootstrap(block_size, data, seed=seed)
    interval = bootstrap.conf_int(
        lambda sample: np.array([np.mean(sample)]),
        reps=repetitions,
        method="percentile",
        size=confidence,
    )
    return BootstrapInterval(
        mean=float(np.mean(data)),
        lower=float(interval[0, 0]),
        upper=float(interval[1, 0]),
        block_size=block_size,
        repetitions=repetitions,
        seed=seed,
    )


def spa_reality_check(
    benchmark_losses: object,
    model_losses: object,
    *,
    block_size: int,
    repetitions: int,
    seed: int,
) -> SpaResult:
    _validate_bootstrap_configuration(block_size, repetitions, seed)
    aligned = _loss_inputs(benchmark_losses, model_losses)
    if aligned.status == "UNDEFINED":
        return SpaResult(
            None,
            None,
            None,
            aligned.observation_count,
            "UNDEFINED",
            aligned.reason,
        )
    benchmark = aligned.values[:, 0]
    models = pd.DataFrame(
        aligned.values[:, 1:],
        index=aligned.index,
        columns=aligned.candidate_ids[1:],
    )
    test = SPA(
        benchmark,
        models,
        block_size=block_size,
        reps=repetitions,
        seed=seed,
    )
    test.compute()
    values = test.pvalues
    return SpaResult(
        lower_p_value=float(values["lower"]),
        consistent_p_value=float(values["consistent"]),
        upper_p_value=float(values["upper"]),
        observation_count=aligned.observation_count,
        status="DEFINED",
        reason=None,
    )


def stepm_superior_models(
    benchmark_losses: object,
    model_losses: pd.DataFrame,
    *,
    block_size: int,
    repetitions: int,
    seed: int,
    alpha: float = 0.05,
) -> StepMResult:
    _validate_bootstrap_configuration(block_size, repetitions, seed)
    _validate_alpha(alpha)
    aligned = _loss_inputs(benchmark_losses, model_losses)
    if aligned.status == "UNDEFINED":
        return StepMResult((), aligned.observation_count, "UNDEFINED", aligned.reason)
    benchmark = aligned.values[:, 0]
    models = pd.DataFrame(
        aligned.values[:, 1:],
        index=aligned.index,
        columns=aligned.candidate_ids[1:],
    )
    test = StepM(
        benchmark,
        models,
        size=alpha,
        block_size=block_size,
        reps=repetitions,
        seed=seed,
    )
    test.compute()
    return StepMResult(
        tuple(str(value) for value in test.superior_models),
        aligned.observation_count,
        "DEFINED",
        None,
    )


def model_confidence_set(
    losses: pd.DataFrame,
    *,
    block_size: int,
    repetitions: int,
    seed: int,
    alpha: float = 0.05,
) -> ModelConfidenceSetResult:
    _validate_bootstrap_configuration(block_size, repetitions, seed)
    _validate_alpha(alpha)
    aligned = aligned_candidate_matrix(losses)
    if aligned.status != "DEFINED":
        return ModelConfidenceSetResult(
            None, (), (), aligned.observation_count, "UNDEFINED", aligned.reason
        )
    frame = pd.DataFrame(
        aligned.values, index=aligned.index, columns=aligned.candidate_ids
    )
    test = MCS(
        frame,
        size=alpha,
        reps=repetitions,
        block_size=block_size,
        seed=seed,
    )
    test.compute()
    return ModelConfidenceSetResult(
        test.pvalues.copy(),
        tuple(str(value) for value in test.included),
        tuple(str(value) for value in test.excluded),
        aligned.observation_count,
        "DEFINED",
        None,
    )


def probabilistic_and_deflated_sharpe(
    returns: object,
    *,
    benchmark_sharpe: float,
    trial_sharpes: object,
) -> SharpeInference:
    data = _finite_vector(returns, minimum=4)
    trials = _finite_vector(trial_sharpes, minimum=2)
    if type(benchmark_sharpe) is not float or not np.isfinite(benchmark_sharpe):
        raise ValueError("sharpe_benchmark_invalid")
    standard_deviation = float(np.std(data, ddof=1))
    if standard_deviation == 0.0:
        raise ValueError("sharpe_zero_variance")
    observed = float(np.mean(data) / standard_deviation)
    sample_skew = float(skew(data, bias=False))
    sample_kurtosis = float(kurtosis(data, fisher=False, bias=False))
    variance = (
        1.0
        - sample_skew * observed
        + ((sample_kurtosis - 1.0) / 4.0) * observed * observed
    ) / (len(data) - 1)
    if variance <= 0.0:
        raise ValueError("sharpe_variance_invalid")
    scale = variance**0.5
    trial_scale = float(np.std(trials, ddof=1))
    if trial_scale == 0.0:
        raise ValueError("trial_sharpe_dispersion_undefined")
    trial_count = len(trials)
    reference = trial_scale * (
        (1.0 - np.euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + np.euler_gamma * norm.ppf(1.0 - 1.0 / (trial_count * e))
    )
    return SharpeInference(
        observed_sharpe=observed,
        benchmark_sharpe=float(benchmark_sharpe),
        psr=float(norm.cdf((observed - benchmark_sharpe) / scale)),
        deflated_reference_sharpe=float(reference),
        dsr=float(norm.cdf((observed - reference) / scale)),
        observation_count=len(data),
        trial_count=trial_count,
    )


def cscv_pbo(returns: object, *, slice_count: int, max_combinations: int) -> PboResult:
    shape = getattr(returns, "shape", None)
    if shape is None:
        shape = np.shape(returns)
    if (
        len(shape) != 2
        or type(slice_count) is not int
        or slice_count < 4
        or slice_count % 2
        or shape[0] % slice_count
    ):
        raise ValueError("cscv_slice_count_invalid")
    combination_count = comb(slice_count, slice_count // 2)
    if type(max_combinations) is not int or combination_count > max_combinations:
        raise ValueError("cscv_combination_budget_exceeded")
    matrix = _finite_matrix(returns, minimum_rows=4, minimum_columns=2)
    if any(
        np.array_equal(matrix[:, left], matrix[:, right])
        for left in range(matrix.shape[1])
        for right in range(left + 1, matrix.shape[1])
    ):
        return _undefined_pbo(
            matrix, slice_count, combination_count, "duplicate_candidate_series"
        )
    slices = np.split(np.arange(len(matrix)), slice_count)
    overfit_count = 0
    all_slice_ids = set(range(slice_count))
    for in_sample_ids in combinations(range(slice_count), slice_count // 2):
        out_sample_ids = tuple(sorted(all_slice_ids.difference(in_sample_ids)))
        in_sample = matrix[np.concatenate([slices[index] for index in in_sample_ids])]
        out_sample = matrix[np.concatenate([slices[index] for index in out_sample_ids])]
        in_scores = _candidate_sharpes(in_sample)
        out_scores = _candidate_sharpes(out_sample)
        if in_scores is None or out_scores is None:
            return _undefined_pbo(
                matrix, slice_count, combination_count, "candidate_sharpe_undefined"
            )
        if len(np.unique(in_scores)) != len(in_scores) or len(
            np.unique(out_scores)
        ) != len(out_scores):
            return _undefined_pbo(
                matrix, slice_count, combination_count, "candidate_ranking_tie"
            )
        selected = int(np.argmax(in_scores))
        rank = 1 + int(np.sum(out_scores < out_scores[selected]))
        fraction = rank / (matrix.shape[1] + 1.0)
        logit = float(np.log(fraction / (1.0 - fraction)))
        overfit_count += int(logit <= 0.0)
    return PboResult(
        pbo=overfit_count / combination_count,
        slice_count=slice_count,
        combination_count=combination_count,
        candidate_count=matrix.shape[1],
        status="DEFINED",
        reason=None,
    )


def fama_macbeth(
    frame: pd.DataFrame,
    *,
    return_column: str,
    factor_columns: tuple[str, ...],
    max_lags: int,
    weight_column: str | None = None,
) -> FamaMacBethResult:
    required = {"timestamp", "asset_id", return_column, *factor_columns}
    if weight_column is not None:
        required.add(weight_column)
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("fama_macbeth_columns_invalid")
    frame = frame.copy()
    frame["timestamp"] = strict_timestamp_series(
        frame["timestamp"], error="fama_macbeth_timestamp_invalid"
    )
    frame["asset_id"] = strict_hashable_key_series(
        frame["asset_id"], error="fama_macbeth_asset_id_invalid"
    )
    if bool(frame.duplicated(["timestamp", "asset_id"], keep=False).any()):
        raise ValueError("fama_macbeth_primary_key_duplicate")
    if not factor_columns or len(set(factor_columns)) != len(factor_columns):
        raise ValueError("fama_macbeth_factors_invalid")
    coefficients: list[pd.Series] = []
    periods: list[object] = []
    white_p_values: list[float] = []
    white_reasons: list[str | None] = []
    for period, group in frame.groupby("timestamp", sort=True):
        y = pd.to_numeric(group[return_column], errors="raise").astype("float64")
        x = sm.add_constant(
            group.loc[:, factor_columns]
            .apply(pd.to_numeric, errors="raise")
            .astype("float64"),
            has_constant="add",
        )
        if (
            len(group) <= x.shape[1]
            or not bool(np.isfinite(y).all())
            or not bool(np.isfinite(x.to_numpy()).all())
        ):
            raise ValueError("fama_macbeth_period_insufficient")
        if weight_column is None:
            fit = sm.OLS(y, x).fit()
        else:
            weights = pd.to_numeric(group[weight_column], errors="raise").astype(
                "float64"
            )
            if bool((weights <= 0.0).any()) or not bool(np.isfinite(weights).all()):
                raise ValueError("fama_macbeth_weights_invalid")
            fit = sm.WLS(y, x, weights=weights).fit()
        coefficients.append(fit.params)
        periods.append(period)
        try:
            white_p_value = float(het_white(fit.resid, x)[1])
        except (AssertionError, ValueError):
            white_p_value = float("nan")
        white_p_values.append(white_p_value)
        white_reasons.append(
            None if np.isfinite(white_p_value) else "white_test_undefined"
        )
    period_coefficients = pd.DataFrame(
        coefficients, index=pd.Index(periods, name="timestamp")
    )
    summary = pd.DataFrame(
        {
            name: {
                "mean": result.mean,
                "standard_error": result.standard_error,
                "t_statistic": result.t_statistic,
                "p_value": result.p_value,
                "ci_lower": result.confidence_interval[0],
                "ci_upper": result.confidence_interval[1],
            }
            for name in period_coefficients.columns
            for result in [
                hac_mean(period_coefficients[name].to_numpy(), max_lags=max_lags)
            ]
        }
    ).T
    return FamaMacBethResult(
        period_coefficients=period_coefficients,
        summary=summary,
        white_p_values=pd.Series(
            white_p_values, index=period_coefficients.index, name="white_p_value"
        ),
        white_reasons=pd.Series(
            white_reasons,
            index=period_coefficients.index,
            name="white_reason",
            dtype="object",
        ),
        period_count=len(period_coefficients),
    )


def _finite_vector(values: object, *, minimum: int) -> np.ndarray:
    data = np.asarray(values, dtype="float64")
    if data.ndim != 1 or len(data) < minimum or not bool(np.isfinite(data).all()):
        raise ValueError("finite_vector_required")
    return data


def _validate_bootstrap_configuration(
    block_size: int, repetitions: int, seed: int
) -> None:
    if (
        type(block_size) is not int
        or block_size <= 0
        or type(repetitions) is not int
        or repetitions <= 0
        or type(seed) is not int
    ):
        raise ValueError("bootstrap_configuration_invalid")


def _validate_alpha(alpha: float) -> None:
    if type(alpha) is not float or not 0.0 < alpha < 1.0:
        raise ValueError("alpha_invalid")


def _finite_matrix(
    values: object, *, minimum_rows: int, minimum_columns: int
) -> np.ndarray:
    shape = getattr(values, "shape", None)
    if shape is None:
        shape = np.shape(values)
    if len(shape) != 2:
        raise ValueError("finite_matrix_required")
    policy = current_runtime_resource_policy()
    snapshot_bytes = int(shape[0]) * int(shape[1]) * 8 + 64 * 1024
    if policy is not None and snapshot_bytes > policy.operation_headroom_bytes:
        raise MemoryError("statistical_matrix_exceeds_runtime_headroom")
    data = np.asarray(values, dtype="float64")
    if (
        data.ndim != 2
        or data.shape[0] < minimum_rows
        or data.shape[1] < minimum_columns
        or not bool(np.isfinite(data).all())
    ):
        raise ValueError("finite_matrix_required")
    return data


def _loss_inputs(
    benchmark_losses: object, model_losses: object
) -> AlignedCandidateMatrix:
    benchmark = (
        benchmark_losses.rename("__benchmark_loss__")
        if isinstance(benchmark_losses, pd.Series)
        else pd.Series(benchmark_losses, name="__benchmark_loss__")
    )
    if not isinstance(model_losses, pd.DataFrame):
        raise ValueError("loss_matrix_candidate_id_invalid")
    models = model_losses.copy()
    if not isinstance(benchmark_losses, pd.Series) and len(benchmark) != len(models):
        raise ValueError("loss_matrix_length_mismatch")
    if "__benchmark_loss__" in models.columns:
        raise ValueError("loss_matrix_candidate_id_invalid")
    if models.columns.has_duplicates or not all(
        type(column) is str and column for column in models.columns
    ):
        raise ValueError("loss_matrix_candidate_id_invalid")
    combined = pd.concat((benchmark, models), axis=1, join="inner")
    return aligned_candidate_matrix(combined, minimum_observations=3)


def _candidate_sharpes(values: np.ndarray) -> np.ndarray | None:
    means = np.mean(values, axis=0)
    standard_deviations = np.std(values, axis=0, ddof=1)
    if bool((standard_deviations == 0.0).any()):
        return None
    return means / standard_deviations


def _undefined_pbo(
    matrix: np.ndarray,
    slice_count: int,
    combination_count: int,
    reason: str,
) -> PboResult:
    return PboResult(
        pbo=None,
        slice_count=slice_count,
        combination_count=combination_count,
        candidate_count=matrix.shape[1],
        status="UNDEFINED",
        reason=reason,
    )
