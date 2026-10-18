"""QRP input/status contracts delegating numerical inference to dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, e
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, norm, skew
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant

from quant_research_platform.compute.dataframe.cross_section import (
    _average_ranks,
    _pearson_correlation_result,
)
from quant_research_platform.contracts.validation import (
    strict_hashable_key_series,
    strict_real_numeric_series,
    strict_timestamp_series,
)


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
    deflated_reference_sharpe: float | None
    dsr: float | None
    observation_count: int
    trial_count: int
    dsr_reason: str | None = None


@dataclass(frozen=True)
class PboResult:
    pbo: float | None
    slice_count: int
    combination_count: int
    candidate_count: int
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None


@dataclass(frozen=True)
class FamaMacBethPeriodResult:
    coefficients: pd.Series
    white_p_value: float | None
    white_reason: str | None
    observation_count: int


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
        or (not frame.index.is_unique)
    ):
        raise ValueError("candidate_matrix_identity_invalid")
    if (
        not isinstance(frame.index, (pd.DatetimeIndex, pd.RangeIndex))
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("candidate_matrix_time_order_invalid")
    if any(
        not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
        for column in frame.columns
    ):
        raise ValueError("candidate_matrix_numeric_required")
    numeric = frame.astype("float64")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("candidate_matrix_nonfinite")
    complete = numeric.notna().all(axis=1)
    values = numeric.loc[complete].to_numpy(dtype="float64", copy=True)
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
    if type(confidence) is not float or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_invalid")
    _, exponent = np.frexp(np.max(np.abs(data)))
    normalized = np.ldexp(data, -int(exponent))
    if float(np.ptp(normalized)) <= 8 * np.finfo("float64").eps * float(
        np.max(np.abs(normalized))
    ):
        raise ValueError("hac_inference_undefined")
    fit = OLS(normalized, np.ones((len(data), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lags}
    )
    mean = float(np.mean(data))
    with np.errstate(over="ignore", under="ignore"):
        standard_error = float(np.ldexp(fit.bse[0], int(exponent)))
    t_statistic = mean / standard_error if standard_error > 0 else np.nan
    critical = float(norm.ppf((1 + confidence) / 2))
    outputs = np.array(
        [
            mean,
            standard_error,
            t_statistic,
            2 * norm.sf(abs(t_statistic)),
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
        dtype="float64",
    )
    if (
        not bool(np.isfinite(outputs).all())
        or outputs[1] <= 0
        or (not 0 <= outputs[3] <= 1)
        or (outputs[4] > outputs[5])
    ):
        raise ValueError("hac_inference_undefined")
    return MeanInference(
        mean=float(outputs[0]),
        standard_error=float(outputs[1]),
        t_statistic=float(outputs[2]),
        p_value=float(outputs[3]),
        confidence_interval=(float(outputs[4]), float(outputs[5])),
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
    from arch.bootstrap import StationaryBootstrap

    bootstrap = StationaryBootstrap(block_size, data, seed=seed)
    interval = bootstrap.conf_int(
        lambda sample: np.array([sample.mean()]),
        reps=repetitions,
        method="percentile",
        size=confidence,
    )
    if not np.isfinite(interval).all() or interval[0, 0] > interval[1, 0]:
        raise ValueError("bootstrap_inference_undefined")
    return BootstrapInterval(
        mean=float(data.mean()),
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
    reason = aligned.reason or _loss_differential_reason(aligned.values, pairwise=False)
    if reason is not None:
        return SpaResult(
            None, None, None, aligned.observation_count, "UNDEFINED", reason
        )
    benchmark = aligned.values[:, 0]
    models = pd.DataFrame(
        aligned.values[:, 1:], index=aligned.index, columns=aligned.candidate_ids[1:]
    )
    from arch.bootstrap import SPA

    test = SPA(benchmark, models, block_size=block_size, reps=repetitions, seed=seed)
    test.compute()
    values = test.pvalues
    if not _positive_finite_variances(test._loss_diff_var) or not _valid_p_values(
        values.to_numpy()
    ):
        return SpaResult(
            None,
            None,
            None,
            aligned.observation_count,
            "UNDEFINED",
            "loss_differential_variance_undefined",
        )
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
    reason = aligned.reason or _loss_differential_reason(aligned.values, pairwise=False)
    if reason is not None:
        return StepMResult((), aligned.observation_count, "UNDEFINED", reason)
    benchmark = aligned.values[:, 0]
    models = pd.DataFrame(
        aligned.values[:, 1:], index=aligned.index, columns=aligned.candidate_ids[1:]
    )
    from arch.bootstrap import StepM

    test = StepM(
        benchmark,
        models,
        size=alpha,
        block_size=block_size,
        reps=repetitions,
        seed=seed,
    )
    test.compute()
    if not _positive_finite_variances(test.spa._loss_diff_var) or not _valid_p_values(
        test.spa.pvalues.to_numpy()
    ):
        return StepMResult(
            (),
            aligned.observation_count,
            "UNDEFINED",
            "loss_differential_variance_undefined",
        )
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
    reason = aligned.reason or _loss_differential_reason(aligned.values, pairwise=True)
    if reason is not None:
        return ModelConfidenceSetResult(
            None, (), (), aligned.observation_count, "UNDEFINED", reason
        )
    frame = pd.DataFrame(
        aligned.values, index=aligned.index, columns=aligned.candidate_ids
    )
    from arch.bootstrap import MCS

    test = MCS(frame, size=alpha, reps=repetitions, block_size=block_size, seed=seed)
    try:
        test.compute()
    except IndexError:
        if _positive_finite_variances(
            test._variances[~np.eye(frame.shape[1], dtype=bool)]
        ):
            raise
        return ModelConfidenceSetResult(
            None,
            (),
            (),
            aligned.observation_count,
            "UNDEFINED",
            "pairwise_loss_variance_undefined",
        )
    if not _positive_finite_variances(
        test._variances[~np.eye(frame.shape[1], dtype=bool)]
    ) or not _valid_p_values(test.pvalues.to_numpy()):
        return ModelConfidenceSetResult(
            None,
            (),
            (),
            aligned.observation_count,
            "UNDEFINED",
            "pairwise_loss_variance_undefined",
        )
    return ModelConfidenceSetResult(
        test.pvalues.copy(),
        tuple(str(value) for value in test.included),
        tuple(str(value) for value in test.excluded),
        aligned.observation_count,
        "DEFINED",
        None,
    )


def _loss_differential_reason(values: np.ndarray, *, pairwise: bool) -> str | None:
    if values.shape[1] < 2:
        return "insufficient_candidate_count"
    scale = float(np.max(np.abs(values)))
    pairs = (
        combinations(range(values.shape[1]), 2)
        if pairwise
        else ((0, index) for index in range(1, values.shape[1]))
    )
    for left, right in pairs:
        difference = values[:, left] - values[:, right]
        if float(np.ptp(difference)) <= np.finfo("float64").eps * scale:
            return "constant_loss_differential"
    return None


def _positive_finite_variances(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and (values > 0).all())


def _valid_p_values(values: np.ndarray) -> bool:
    return bool(
        np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    )


def probabilistic_and_deflated_sharpe(
    returns: object,
    *,
    benchmark_sharpe: float,
    trial_sharpes: object,
    return_roundoff: object = 0.0,
) -> SharpeInference:
    data = _finite_vector(returns, minimum=4)
    trials = np.asarray(trial_sharpes, dtype="float64")
    if trials.ndim != 1 or len(trials) < 1:
        raise ValueError("finite_vector_required")
    if type(benchmark_sharpe) is not float or not np.isfinite(benchmark_sharpe):
        raise ValueError("sharpe_benchmark_invalid")
    observed = float(_return_sharpes(data, roundoff=return_roundoff))
    if not np.isfinite(observed):
        raise ValueError("sharpe_zero_variance")
    _, exponent = np.frexp(np.max(np.abs(data)))
    scaled = np.ldexp(data, -exponent)
    centered = scaled - scaled[0]
    normalized = centered / np.std(centered, ddof=1)
    sample_skew = float(skew(normalized, bias=False))
    sample_kurtosis = float(kurtosis(normalized, fisher=False, bias=False))
    variance = (
        1.0
        - sample_skew * observed
        + (sample_kurtosis - 1.0) / 4.0 * observed * observed
    ) / (len(data) - 1)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("sharpe_variance_invalid")
    scale = variance**0.5
    if len(trials) == 1:
        return SharpeInference(
            observed_sharpe=observed,
            benchmark_sharpe=float(benchmark_sharpe),
            psr=float(norm.cdf((observed - benchmark_sharpe) / scale)),
            deflated_reference_sharpe=None,
            dsr=None,
            observation_count=len(data),
            trial_count=1,
            dsr_reason="insufficient_candidate_count",
        )
    trial_scale = (
        float(return_standard_deviation(trials))
        if np.isfinite(trials).all()
        else np.nan
    )
    trial_count = len(trials)
    reference = trial_scale * (
        (1.0 - np.euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + np.euler_gamma * norm.ppf(1.0 - 1.0 / (trial_count * e))
    )
    return SharpeInference(
        observed_sharpe=observed,
        benchmark_sharpe=float(benchmark_sharpe),
        psr=float(norm.cdf((observed - benchmark_sharpe) / scale)),
        deflated_reference_sharpe=float(reference)
        if np.isfinite(trial_scale) and trial_scale > 0
        else None,
        dsr=float(norm.cdf((observed - reference) / scale))
        if trial_scale > 0.0
        else None,
        observation_count=len(data),
        trial_count=trial_count,
        dsr_reason=None
        if trial_scale > 0.0
        else "trial_sharpe_dispersion_undefined"
        if np.isfinite(trial_scale)
        else "candidate_sharpe_undefined",
    )


def _selected_analyses(config, allowed):
    """Validate explicit analysis selection while retaining the declared default."""
    values = config.get("analyses", list(allowed))
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or value not in allowed for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("statistics_analyses_invalid")
    return tuple(value for value in allowed if value in values)


def cscv_pbo(
    returns: object,
    *,
    slice_count: int,
    return_roundoff: object = 0.0,
) -> PboResult:
    shape = getattr(returns, "shape", None)
    if shape is None:
        shape = np.shape(returns)
    if (
        len(shape) != 2
        or type(slice_count) is not int
        or slice_count < 4
        or slice_count % 2
        or (shape[0] < slice_count)
    ):
        raise ValueError("cscv_slice_count_invalid")
    combination_count = comb(slice_count, slice_count // 2)
    matrix = _finite_matrix(returns, minimum_rows=4, minimum_columns=1)
    if matrix.shape[1] == 1:
        return _undefined_pbo(
            matrix, slice_count, combination_count, "insufficient_candidate_count"
        )
    roundoff = _return_roundoff(matrix, return_roundoff)
    if any(
        np.array_equal(matrix[:, left], matrix[:, right])
        for left in range(matrix.shape[1])
        for right in range(left + 1, matrix.shape[1])
    ):
        return _undefined_pbo(
            matrix, slice_count, combination_count, "duplicate_candidate_series"
        )
    slices = tuple(np.array_split(np.arange(len(matrix)), slice_count))
    if max(len(item) for item in slices) - min(len(item) for item in slices) > 1:
        raise ValueError("cscv_slice_partition_invalid")
    overfit_count = 0
    all_slice_ids = set(range(slice_count))
    # The first half of lexicographic combinations contains slice zero. Each
    # complementary pair has exactly the same two rankings in reverse order.
    for tail in combinations(range(1, slice_count), slice_count // 2 - 1):
        in_sample_ids = (0, *tail)
        out_sample_ids = tuple(sorted(all_slice_ids.difference(in_sample_ids)))
        in_indices = np.concatenate([slices[index] for index in in_sample_ids])
        out_indices = np.concatenate([slices[index] for index in out_sample_ids])
        in_scores = _candidate_ranks(matrix[in_indices], roundoff[in_indices])
        out_scores = _candidate_ranks(matrix[out_indices], roundoff[out_indices])
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
        for training, testing in ((in_scores, out_scores), (out_scores, in_scores)):
            selected = int(np.argmax(training))
            rank = 1 + int(np.sum(testing < testing[selected]))
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


def fama_macbeth_period(
    frame: pd.DataFrame,
    *,
    return_column: str,
    factor_columns: tuple[str, ...],
    weight_column: str | None = None,
) -> FamaMacBethPeriodResult:
    required = {"timestamp", "asset_id", return_column, *factor_columns}
    if weight_column is not None:
        required.add(weight_column)
    if (
        not isinstance(frame, pd.DataFrame)
        or not required.issubset(frame.columns)
        or (not factor_columns)
        or (len(set(factor_columns)) != len(factor_columns))
    ):
        raise ValueError("fama_macbeth_columns_invalid")
    if "intercept" in factor_columns:
        raise ValueError("fama_macbeth_intercept_factor_reserved")
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="fama_macbeth_timestamp_invalid"
    )
    assets = strict_hashable_key_series(
        frame["asset_id"], error="fama_macbeth_asset_id_invalid"
    )
    if timestamps.nunique() != 1 or assets.duplicated().any():
        raise ValueError("fama_macbeth_primary_key_duplicate")
    numeric = (
        frame.loc[:, [return_column, *factor_columns]]
        .apply(pd.to_numeric, errors="raise")
        .astype("float64")
    )
    if weight_column is not None:
        weights = strict_real_numeric_series(
            frame[weight_column], error="fama_macbeth_weights_invalid", errors="coerce"
        )
        numeric[weight_column] = weights.where(np.isfinite(weights) & weights.gt(0))
    complete = numeric.notna().all(axis=1)
    numeric = numeric.loc[complete]
    y = numeric[return_column]
    if len(numeric) <= len(factor_columns) + 1 or not bool(
        np.isfinite(numeric.to_numpy()).all()
    ):
        raise ValueError("fama_macbeth_period_insufficient")
    values = numeric.loc[:, factor_columns].to_numpy(dtype="float64")
    origins = values[0]
    offsets = np.mean(values - origins, axis=0)
    centered = values - origins - offsets
    scales = np.max(np.abs(centered), axis=0)
    x = add_constant(centered / np.where(scales > 0, scales, 1.0), has_constant="add")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("fama_macbeth_period_insufficient")
    target_origin = float(y.iloc[0])
    target_offset = float(np.mean(y - target_origin))
    target_centered = y - target_origin - target_offset
    target_scale = float(np.max(np.abs(target_centered))) or 1.0
    target = (target_centered / target_scale).to_numpy()
    fit = (
        OLS(target, x).fit(method="qr")
        if weight_column is None
        else WLS(target, x, weights=numeric[weight_column]).fit(method="qr")
    )
    # Resolve numerical zero in the regression's own units, before temporal
    # inference can treat floating-point cancellation as coefficient variation.
    roundoff = 32 * np.finfo(float).eps
    normalized_slopes = fit.params[1:].copy()
    normalized_slopes[
        np.abs(normalized_slopes) <= roundoff * np.max(np.abs(target))
    ] = 0.0
    slopes = normalized_slopes * target_scale / scales
    intercept_terms = np.concatenate(
        (
            [target_origin, target_offset, float(fit.params[0]) * target_scale],
            -(origins + offsets) * slopes,
        )
    )
    intercept = float(np.sum(intercept_terms))
    if abs(intercept) <= roundoff * np.sum(np.abs(intercept_terms)):
        intercept = 0.0
    coefficients = pd.Series([intercept, *slopes], index=("intercept", *factor_columns))
    if not np.isfinite(coefficients).all():
        raise ValueError("fama_macbeth_coefficients_unrepresentable")
    residual_scale = float(np.max(np.abs(fit.resid)))
    # A numerically exact fit has no estimable residual-variance diagnostic.
    white_residuals = (
        fit.resid / residual_scale
        if residual_scale * target_scale
        > 32 * np.finfo(float).eps * float(np.max(np.abs(y)))
        else None
    )
    white_p_value = None
    white_response = None if white_residuals is None else np.square(white_residuals)
    if white_response is not None and np.ptp(white_response) > 64 * np.finfo(float).eps:
        try:
            # White LM needs the auxiliary R-squared, not its unused F statistic.
            left, right = np.triu_indices(x.shape[1])
            auxiliary = OLS(
                white_response,
                x[:, left] * x[:, right],
            ).fit()
            white_p_value = float(
                chi2.sf(len(numeric) * auxiliary.rsquared, auxiliary.df_model)
            )
            if not np.isfinite(white_p_value) or not 0 <= white_p_value <= 1:
                white_p_value = None
        except (AssertionError, ValueError):
            white_p_value = None
    return FamaMacBethPeriodResult(
        coefficients=coefficients,
        white_p_value=white_p_value,
        white_reason=None if white_p_value is not None else "white_test_undefined",
        observation_count=len(numeric),
    )


def cross_sectional_rank_ic(
    frame: pd.DataFrame, *, value_columns: tuple[str, ...], label_column: str
) -> pd.DataFrame:
    """Return per-date cross-sectional Spearman correlations."""
    required = {"timestamp", "asset_id", label_column, *value_columns}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("rank_ic_columns_invalid")
    if not value_columns or len(set(value_columns)) != len(value_columns):
        raise ValueError("rank_ic_columns_invalid")
    selected = frame.loc[
        :, ["timestamp", "asset_id", label_column, *value_columns]
    ].copy()
    selected["timestamp"] = strict_timestamp_series(
        selected["timestamp"], error="rank_ic_timestamp_invalid"
    )
    selected["asset_id"] = strict_hashable_key_series(
        selected["asset_id"], error="rank_ic_asset_id_invalid"
    )
    if bool(selected.duplicated(["timestamp", "asset_id"]).any()):
        raise ValueError("rank_ic_primary_key_duplicate")
    for column in (label_column, *value_columns):
        # Integer dtypes already contain only real observations. Keep their exact
        # ordering: conversion to float can turn adjacent integers into ties.
        if not pd.api.types.is_integer_dtype(selected[column].dtype):
            selected[column] = strict_real_numeric_series(
                selected[column], error="rank_ic_numeric_invalid"
            )
    records: list[dict[str, object]] = []
    for timestamp, group in selected.groupby("timestamp", sort=True):
        record: dict[str, object] = {"timestamp": timestamp}
        arrays = {}
        present = {}
        for column in (label_column, *value_columns):
            series = group[column]
            present[column] = series.notna().to_numpy()
            dtype = (
                "uint64"
                if pd.api.types.is_unsigned_integer_dtype(series.dtype)
                else "int64"
                if pd.api.types.is_integer_dtype(series.dtype)
                else "float64"
            )
            # Missing integer slots need a placeholder; the separate mask keeps
            # it out of ranks, including when zero is a genuine observation.
            arrays[column] = series.to_numpy(dtype=dtype, na_value=0)
        labels_all = arrays[label_column]
        label_present = present[label_column]
        label_ranks = {}
        for column in value_columns:
            values_all = arrays[column]
            mask = label_present & present[column]
            values, labels = values_all[mask], labels_all[mask]
            if not np.isfinite(values).all() or not np.isfinite(labels).all():
                record[column] = np.nan
                continue
            key = mask.tobytes()
            if key not in label_ranks:
                label_ranks[key] = _average_ranks(labels)
            correlation, _reason = _pearson_correlation_result(
                _average_ranks(values), label_ranks[key]
            )
            record[column] = np.nan if correlation is None else correlation
        records.append(record)
    return pd.DataFrame.from_records(records, columns=["timestamp", *value_columns])


def time_series_inference_summary(
    frame: pd.DataFrame,
    *,
    identity_column: str,
    max_lags: int,
    fdr_method: Literal["BH", "BY"] | None,
    fdr_alpha: float,
    fdr_family_size: int | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize complete time-series columns with HAC inference and one FDR family."""
    if type(max_lags) is not int or max_lags < 0:
        raise ValueError("hac_max_lags_invalid")
    if type(confidence) is not float or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_invalid")
    if not isinstance(frame, pd.DataFrame) or identity_column not in frame.columns:
        raise ValueError("time_series_inference_columns_invalid")
    value_columns = tuple(
        column for column in frame.columns if column != identity_column
    )
    if not value_columns:
        return pd.DataFrame()
    family_size = len(value_columns) if fdr_family_size is None else fdr_family_size
    if type(family_size) is not int or family_size < len(value_columns):
        raise ValueError("fdr_family_size_invalid")
    strict_timestamp_series(
        frame[identity_column], error="time_series_inference_timestamp_invalid"
    )
    rows: list[dict[str, object]] = []
    for column in value_columns:
        values = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if np.isinf(values).any():
            raise ValueError("time_series_inference_nonfinite")
        valid = values.notna()
        count = int(valid.sum())
        if count != len(values) or count < max(2, max_lags + 1):
            rows.append(
                {
                    identity_column: column,
                    "observation_count": count,
                    "status": "UNDEFINED",
                    "reason": "incomplete_session_clock"
                    if count != len(values)
                    else "hac_lag_exceeds_observations",
                    "mean": float(values.loc[valid].mean()) if count else np.nan,
                    "standard_error": np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
            continue
        try:
            result = hac_mean(
                values.loc[valid], max_lags=max_lags, confidence=confidence
            )
        except ValueError as exc:
            if str(exc) != "hac_inference_undefined":
                raise
            rows.append(
                {
                    identity_column: column,
                    "observation_count": count,
                    "status": "UNDEFINED",
                    "reason": "hac_inference_undefined",
                    "mean": float(values.loc[valid].mean()) if count else np.nan,
                    "standard_error": np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
            continue
        rows.append(
            {
                identity_column: column,
                "observation_count": count,
                "status": "DEFINED",
                "reason": None,
                "mean": result.mean,
                "standard_error": result.standard_error,
                "t_statistic": result.t_statistic,
                "p_value": result.p_value,
                "ci_lower": result.confidence_interval[0],
                "ci_upper": result.confidence_interval[1],
            }
        )
    result = pd.DataFrame(rows)
    return _apply_fdr(result, fdr_method, fdr_alpha, family_size)


def _apply_fdr(result, fdr_method, fdr_alpha, family_size):
    """Apply the full declared family to an existing individual-inference table."""
    if fdr_method is not None:
        _validate_alpha(fdr_alpha)
        if fdr_method not in {"BH", "BY"}:
            raise ValueError("fdr_input_invalid")
    defined = result["status"].eq("DEFINED")
    if fdr_method is None:
        result["fdr_method"] = None
        result["fdr_alpha"] = np.nan
        result["fdr_adjusted_p_value"] = np.nan
    elif bool(defined.any()):
        observed = result.loc[defined, "p_value"].to_numpy(dtype="float64")
        conservative_work = np.concatenate(
            (observed, np.ones(family_size - len(observed), dtype="float64"))
        )
        adjusted = fdr_adjust(conservative_work, method=fdr_method, alpha=fdr_alpha)
        result["fdr_method"] = fdr_method
        result["fdr_alpha"] = fdr_alpha
        result["fdr_adjusted_p_value"] = np.nan
        result.loc[defined, "fdr_adjusted_p_value"] = adjusted.adjusted_p_values[
            : len(observed)
        ]
        result.loc[~defined, "fdr_method"] = fdr_method
        result.loc[~defined, "fdr_alpha"] = fdr_alpha
    else:
        result["fdr_method"] = fdr_method
        result["fdr_alpha"] = fdr_alpha
        result["fdr_adjusted_p_value"] = np.nan
    return result


def _finite_vector(values: object, *, minimum: int) -> np.ndarray:
    data = np.asarray(values, dtype="float64")
    if data.ndim != 1 or len(data) < minimum or (not bool(np.isfinite(data).all())):
        raise ValueError("finite_vector_required")
    return data


def _validate_bootstrap_configuration(
    block_size: int, repetitions: int, seed: int
) -> None:
    if (
        type(block_size) is not int
        or block_size <= 0
        or type(repetitions) is not int
        or (repetitions < 2)
        or (type(seed) is not int)
        or seed < 0
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
    data = np.asarray(values, dtype="float64")
    if (
        data.ndim != 2
        or data.shape[0] < minimum_rows
        or data.shape[1] < minimum_columns
        or (not bool(np.isfinite(data).all()))
    ):
        raise ValueError("finite_matrix_required")
    return data


def _loss_inputs(
    benchmark_losses: object, model_losses: object
) -> AlignedCandidateMatrix:
    if not isinstance(model_losses, pd.DataFrame):
        raise ValueError("loss_matrix_candidate_id_invalid")
    models = model_losses
    benchmark_id = "__benchmark_loss__"
    while benchmark_id in models.columns:
        benchmark_id += "_"
    benchmark = (
        benchmark_losses.rename(benchmark_id)
        if isinstance(benchmark_losses, pd.Series)
        else pd.Series(benchmark_losses, name=benchmark_id)
    )
    if not isinstance(benchmark_losses, pd.Series):
        if len(benchmark) != len(models):
            raise ValueError("loss_matrix_length_mismatch")
        benchmark.index = models.index
    if models.columns.has_duplicates or not all(
        type(column) is str and column for column in models.columns
    ):
        raise ValueError("loss_matrix_candidate_id_invalid")
    combined = pd.concat((benchmark, models), axis=1, join="inner")
    return aligned_candidate_matrix(combined, minimum_observations=3)


def _return_roundoff(values: np.ndarray, roundoff: object) -> np.ndarray:
    errors = np.asarray(roundoff, dtype="float64")
    if (
        errors.shape not in ((), values.shape)
        or np.isnan(errors).any()
        or (errors < 0).any()
    ):
        raise ValueError("return_roundoff_invalid")
    return np.broadcast_to(errors, values.shape)


def _return_intervals_overlap(data: np.ndarray, errors: np.ndarray) -> np.ndarray:
    lower = np.max(data - errors, axis=0)
    upper = np.min(data + errors, axis=0)
    return np.asarray(lower <= upper)


def return_standard_deviation(values: object, *, roundoff: object = 0.0) -> np.ndarray:
    """Sample risk, with zero only when all supplied error intervals overlap.

    Direct observations have zero uncertainty. NAV-derived callers supply their
    arithmetic resolution; that resolution is not an economic return threshold.
    """
    data = np.asarray(values, dtype="float64")
    errors = _return_roundoff(data, roundoff)
    if len(data) < 2:
        return np.full(data.shape[1:], np.nan)
    unresolved = _return_intervals_overlap(data, errors)
    _, exponent = np.frexp(np.max(np.abs(data), axis=0))
    scaled = np.ldexp(data, -exponent)
    result = np.ldexp(
        np.std(scaled - scaled[0], axis=0, ddof=1),
        exponent,
        out=np.zeros(data.shape[1:]),
        where=~unresolved,
    )
    return np.where(np.isfinite(errors).all(axis=0), result, np.nan)


def _return_sharpes(values: object, *, roundoff: object = 0.0) -> np.ndarray:
    """Compute mean/risk in one binary scale without changing candidate coverage."""
    data = np.asarray(values, dtype="float64")
    errors = _return_roundoff(data, roundoff)
    if len(data) < 2:
        return np.full(data.shape[1:], np.nan)
    unresolved = _return_intervals_overlap(data, errors)
    _, exponent = np.frexp(np.max(np.abs(data), axis=0))
    scaled = np.ldexp(data, -exponent)
    differences = scaled - scaled[0]
    deviation = np.std(differences, axis=0, ddof=1)
    return np.divide(
        np.mean(scaled, axis=0),
        deviation,
        out=np.full(data.shape[1:], np.nan),
        where=(deviation > 0) & ~unresolved & np.isfinite(errors).all(axis=0),
    )


def _candidate_ranks(values: np.ndarray, errors: np.ndarray) -> np.ndarray | None:
    scores = _candidate_sharpes(values, errors)
    if scores is None:
        return None
    return np.unique(scores, return_inverse=True)[1]


def _candidate_sharpes(values: np.ndarray, errors: np.ndarray) -> np.ndarray | None:
    result = _return_sharpes(values, roundoff=errors)
    return result if np.isfinite(result).all() else None


def _undefined_pbo(
    matrix: np.ndarray, slice_count: int, combination_count: int, reason: str
) -> PboResult:
    return PboResult(
        pbo=None,
        slice_count=slice_count,
        combination_count=combination_count,
        candidate_count=matrix.shape[1],
        status="UNDEFINED",
        reason=reason,
    )
