"""DataFrame cross-sectional math kernels for long panel inputs."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from math import isfinite, ulp
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
)


def compute_ic_and_grouped_label_values(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute timestamp-signal IC and grouped label values."""

    panel = validate_compute_panel(frame, spec)
    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    ic_records: list[dict[str, object]] = []
    grouped_records: list[dict[str, object]] = []
    for timestamp, timestamp_frame in panel.groupby(spec.timestamp_column, sort=True):
        for signal in spec.signal_columns:
            signal_values = timestamp_frame[signal].to_numpy(
                dtype="float64",
                copy=False,
            )
            label_values = timestamp_frame[spec.label_column].to_numpy(
                dtype="float64",
                copy=False,
            )
            valid_mask = np.isfinite(signal_values) & np.isfinite(label_values)
            signal_valid = signal_values[valid_mask]
            label_valid = label_values[valid_mask]
            n_obs = len(signal_valid)
            ic = np.nan
            rankic = np.nan
            ic_reason = "insufficient_observations"
            rankic_reason = "insufficient_observations"
            if n_obs >= 2:
                ic, ic_reason = _pearson_correlation_result(
                    signal_valid,
                    label_valid,
                )
                signal_ranks = _average_ranks(signal_valid)
                rankic, rankic_reason = _pearson_correlation_result(
                    signal_ranks,
                    _average_ranks(label_valid),
                )
                groups = _assign_groups_from_ranks(signal_ranks, n_groups)
                for group in np.unique(groups):
                    group_mask = groups == group
                    grouped_records.append(
                        {
                            "timestamp": timestamp,
                            "signal": signal,
                            "label": spec.label_column,
                            "group": group,
                            "mean_label_value": float(mean(label_valid[group_mask])),
                            "n_obs": int(group_mask.sum()),
                        }
                    )
            ic_records.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "label": spec.label_column,
                    "ic": ic,
                    "rankic": rankic,
                    "n_obs": n_obs,
                    "ic_reason": ic_reason,
                    "rankic_reason": rankic_reason,
                }
            )

    metrics = pd.DataFrame.from_records(ic_records, columns=EVALUATION_METRICS_COLUMNS)
    metrics["timestamp"] = metrics["timestamp"].astype("datetime64[ns]")
    for column in ("signal", "label", "ic_reason", "rankic_reason"):
        metrics[column] = metrics[column].astype("string")
    for column in ("ic", "rankic"):
        metrics[column] = metrics[column].astype("float64")
    metrics["n_obs"] = metrics["n_obs"].astype("int64")

    grouped = pd.DataFrame.from_records(
        grouped_records,
        columns=EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    )
    grouped["timestamp"] = grouped["timestamp"].astype("datetime64[ns]")
    for column in ("signal", "label"):
        grouped[column] = grouped[column].astype("string")
    grouped["group"] = grouped["group"].astype("int64")
    grouped["mean_label_value"] = grouped["mean_label_value"].astype("float64")
    grouped["n_obs"] = grouped["n_obs"].astype("int64")
    return metrics, grouped


class _CorrelationMoments:
    """Exact sufficient statistics for finite binary floating-point observations."""

    __slots__ = (
        "count",
        "exponent",
        "left_sum",
        "right_sum",
        "left_square",
        "right_square",
        "product",
    )

    def __init__(self) -> None:
        self.count = 0
        self.exponent = 0
        self.left_sum = self.right_sum = 0
        self.left_square = self.right_square = self.product = 0

    def update(self, left: float, right: float) -> None:
        # Use only the binary scale actually observed. Fixed subnormal scaling
        # adds thousands of zero bits to ordinary prices and every product.
        left_numerator, left_denominator = left.as_integer_ratio()
        right_numerator, right_denominator = right.as_integer_ratio()
        left_exponent = left_denominator.bit_length() - 1
        right_exponent = right_denominator.bit_length() - 1
        exponent = max(self.exponent, left_exponent, right_exponent)
        shift = exponent - self.exponent
        if shift:
            self.left_sum <<= shift
            self.right_sum <<= shift
            self.left_square <<= 2 * shift
            self.right_square <<= 2 * shift
            self.product <<= 2 * shift
            self.exponent = exponent
        x = left_numerator << (exponent - left_exponent)
        y = right_numerator << (exponent - right_exponent)
        self.count += 1
        self.left_sum += x
        self.right_sum += y
        self.left_square += x * x
        self.right_square += y * y
        self.product += x * y

    def remove(self, left: float, right: float) -> None:
        """Remove a previously observed pair from the retained rolling window."""
        left_numerator, left_denominator = left.as_integer_ratio()
        right_numerator, right_denominator = right.as_integer_ratio()
        x = left_numerator << (self.exponent - left_denominator.bit_length() + 1)
        y = right_numerator << (self.exponent - right_denominator.bit_length() + 1)
        self.count -= 1
        self.left_sum -= x
        self.right_sum -= y
        self.left_square -= x * x
        self.right_square -= y * y
        self.product -= x * y

    def averages(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
        """Return exact first and second raw moments in observation units."""
        unit = 1 << self.exponent
        return (
            Fraction(self.left_sum, self.count * unit),
            Fraction(self.right_sum, self.count * unit),
            Fraction(self.left_square, self.count * unit * unit),
            Fraction(self.right_square, self.count * unit * unit),
            Fraction(self.product, self.count * unit * unit),
        )

    def result(self) -> tuple[float | None, str | None]:
        # Multiplying centered moments by count avoids division until the ratio.
        return _pearson_correlation_from_moments(
            count=self.count,
            left_m2=self.count * self.left_square - self.left_sum * self.left_sum,
            right_m2=self.count * self.right_square - self.right_sum * self.right_sum,
            covariance=self.count * self.product - self.left_sum * self.right_sum,
        )


def _pearson_correlation_result(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> tuple[float | None, str | None]:
    if len(left_values) < 2:
        return None, "insufficient_observations"
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        return None, "nonfinite_result"
    moments = _CorrelationMoments()
    for left, right in zip(left_values, right_values, strict=True):
        moments.update(float(left), float(right))
    return moments.result()


def _pearson_correlation_from_moments(
    *,
    count: int,
    left_m2: float | int | Fraction,
    right_m2: float | int | Fraction,
    covariance: float | int | Fraction,
) -> tuple[float | None, str | None]:
    if count < 2:
        return None, "insufficient_observations"
    if left_m2 <= 0 or right_m2 <= 0:
        return None, "zero_variance"
    if any(
        isinstance(value, float) and not isfinite(value)
        for value in (left_m2, right_m2, covariance)
    ):
        return None, "nonfinite_result"
    squared = Fraction(covariance) ** 2 / (Fraction(left_m2) * Fraction(right_m2))
    # Decimal avoids intermediate float underflow/overflow; only the result is
    # rounded to binary64. Invalid moments remain outside the correlation domain.
    with localcontext() as context:
        context.prec = 80
        value = float(
            (Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt()
        )
    if covariance < 0:
        value = -value
    if not np.isfinite(value):
        return None, "nonfinite_result"
    return value, None


def _scaled_centered_values(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype="float64")
    reference = values[0]
    with np.errstate(over="ignore"):
        differences = values - reference
    if np.isfinite(differences).all():
        scale = float(np.max(np.abs(differences)))
        if scale == 0:
            return None
        normalized = differences / scale
    else:
        scale = float(np.max(np.abs(values)))
        normalized = values / scale
    centered = normalized - normalized.mean()
    scale = float(np.max(np.abs(centered)))
    if scale == 0:
        return None
    return centered / scale


def _stable_zscore(values: pd.Series) -> pd.Series:
    source = values.to_numpy(dtype="float64", na_value=np.nan)
    output = np.full(len(source), np.nan, dtype="float64")
    valid = np.isfinite(source)
    if int(valid.sum()) < 2:
        return pd.Series(output, index=values.index)
    centered = _scaled_centered_values(source[valid])
    if centered is None:
        output[valid] = 0.0
        return pd.Series(output, index=values.index)
    std = float(np.sqrt(np.dot(centered, centered) / len(centered)))
    if std == 0:
        output[valid] = 0.0
        return pd.Series(output, index=values.index)
    output[valid] = centered / std
    return pd.Series(output, index=values.index)


def _assign_groups_from_ranks(ranks: np.ndarray, n_groups: int) -> np.ndarray:
    group_count = min(n_groups, len(ranks))
    zero_based = np.floor((ranks - 1.0) * group_count / len(ranks))
    return np.clip(zero_based.astype("int64") + 1, 1, group_count)


def _ordinal_rank_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one-based ordinal rank bounds without splitting equal values."""

    _, inverse, counts = np.unique(
        np.asarray(values, dtype="float64"),
        return_inverse=True,
        return_counts=True,
    )
    upper_by_class = np.cumsum(counts, dtype="int64")
    lower_by_class = upper_by_class - counts + 1
    return lower_by_class[inverse], upper_by_class[inverse]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    lower_ranks, upper_ranks = _ordinal_rank_bounds(values)
    return (lower_ranks + upper_ranks) / 2.0


def _summarize_correlation_aggregates(
    *,
    count: int,
    mean: Fraction,
    m2: Fraction,
    positive_count: int,
) -> dict[str, float | int | str | None]:
    if count == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ratio": np.nan,
            "positive_rate": np.nan,
            "valid_count": 0,
            "reason": "no_valid_observations",
            "ratio_reason": "no_valid_observations",
        }
    variance = m2 / count
    with localcontext() as context:
        context.prec = 80
        deviation = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        std = float(deviation)
        ratio = (
            float(Decimal(mean.numerator) / Decimal(mean.denominator) / deviation)
            if count >= 2 and deviation > 0
            else np.nan
        )
    # A nonzero scale that rounds to zero is unrepresentable, not constant.
    if 0 < variance <= (Fraction(ulp(0.0)) / 2) ** 2:
        std = np.nan
    mean = float(mean)
    if not np.isfinite(std):
        ratio = np.nan
    return {
        "mean": mean,
        "std": std,
        "ratio": ratio,
        "positive_rate": positive_count / count,
        "valid_count": count,
        "reason": None,
        "ratio_reason": _correlation_ratio_reason(
            count=count,
            mean=mean,
            std=std,
            ratio=ratio,
        ),
    }


def _add_correlation_aggregate(
    stats: dict[str, int | Fraction],
    prefix: str,
    value: object,
) -> None:
    if pd.isna(value):
        return
    numeric = float(value)
    count = int(stats[f"{prefix}_count"]) + 1
    mean_key = f"{prefix}_mean"
    m2_key = f"{prefix}_m2"
    # Exact updates preserve adjacent binary64 values and tiny variances.
    value = Fraction(numeric)
    delta = value - stats[mean_key]
    mean = stats[mean_key] + delta / count
    stats[mean_key] = mean
    stats[m2_key] += delta * (value - mean)
    stats[f"{prefix}_count"] = count
    stats[f"{prefix}_positive"] += int(numeric > 0)


def _correlation_ratio_reason(
    *,
    count: int,
    mean: float,
    std: float,
    ratio: float,
) -> str | None:
    if np.isfinite(ratio):
        return None
    if count < 2:
        return "insufficient_observations"
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nonfinite_result"
    if std == 0:
        return "zero_variance"
    return "nonfinite_result"
