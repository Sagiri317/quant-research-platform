"""DataFrame cross-sectional math kernels for long panel inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)

IC_RESULT_COLUMNS = [
    "timestamp",
    "signal",
    "label",
    "ic",
    "rankic",
    "n_obs",
    "ic_reason",
    "rankic_reason",
]
IC_SUMMARY_COLUMNS = [
    "signal",
    "label",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "valid_ic_count",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "valid_rankic_count",
    "ic_reason",
    "icir_reason",
    "rankic_reason",
    "rankicir_reason",
]
GROUPED_LABEL_VALUE_COLUMNS = [
    "timestamp",
    "signal",
    "label",
    "group",
    "mean_label_value",
    "n_obs",
]
GROUP_SPREAD_SUMMARY_COLUMNS = [
    "signal",
    "label",
    "top_group_mean_label_value",
    "bottom_group_mean_label_value",
    "top_bottom_label_value_spread_mean",
    "top_bottom_label_value_spread_hit_rate",
    "group_monotonicity_score",
    "group_monotonicity_score_reason",
]


def compute_ic_and_grouped_label_values(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute timestamp-signal IC and grouped label values."""

    panel = validate_compute_panel(frame, spec)
    return compute_ic_and_grouped_label_values_from_validated_panel(
        panel,
        spec,
        n_groups=n_groups,
    )


def compute_ic_and_grouped_label_values_from_validated_panel(
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IC and grouped label values from a validated panel."""

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
                rankic, rankic_reason = _pearson_correlation_result(
                    _average_ranks(signal_valid),
                    _average_ranks(label_valid),
                )
                groups = _assign_groups_array(signal_valid, n_groups)
                for group in np.unique(groups):
                    group_mask = groups == group
                    grouped_records.append(
                        {
                            "timestamp": timestamp,
                            "signal": signal,
                            "label": spec.label_column,
                            "group": group,
                            "mean_label_value": float(label_valid[group_mask].mean()),
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

    metrics = pd.DataFrame.from_records(ic_records, columns=IC_RESULT_COLUMNS)
    metrics["timestamp"] = metrics["timestamp"].astype("datetime64[ns]")
    for column in ("signal", "label", "ic_reason", "rankic_reason"):
        metrics[column] = metrics[column].astype("string")
    for column in ("ic", "rankic"):
        metrics[column] = metrics[column].astype("float64")
    metrics["n_obs"] = metrics["n_obs"].astype("int64")

    grouped = pd.DataFrame.from_records(
        grouped_records,
        columns=GROUPED_LABEL_VALUE_COLUMNS,
    )
    grouped["timestamp"] = grouped["timestamp"].astype("datetime64[ns]")
    for column in ("signal", "label"):
        grouped[column] = grouped[column].astype("string")
    grouped["group"] = grouped["group"].astype("int64")
    grouped["mean_label_value"] = grouped["mean_label_value"].astype("float64")
    grouped["n_obs"] = grouped["n_obs"].astype("int64")
    return metrics, grouped


def _pearson_correlation_result(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> tuple[float | None, str | None]:
    if len(left_values) < 2:
        return None, "insufficient_observations"
    left_centered = _scaled_centered_values(left_values)
    right_centered = _scaled_centered_values(right_values)
    if left_centered is None or right_centered is None:
        return None, "zero_variance"
    return _pearson_correlation_from_moments(
        count=len(left_values),
        left_m2=float(np.dot(left_centered, left_centered)),
        right_m2=float(np.dot(right_centered, right_centered)),
        covariance=float(np.dot(left_centered, right_centered)),
    )


def _pearson_correlation_from_moments(
    *,
    count: int,
    left_m2: float,
    right_m2: float,
    covariance: float,
) -> tuple[float | None, str | None]:
    if count < 2:
        return None, "insufficient_observations"
    if left_m2 <= 0 or right_m2 <= 0:
        return None, "zero_variance"
    denominator = np.sqrt(left_m2 * right_m2)
    if denominator == 0:
        return None, "zero_variance"
    if not np.isfinite(denominator):
        return None, "nonfinite_result"
    value = float(covariance / denominator)
    if not np.isfinite(value):
        return None, "nonfinite_result"
    if abs(value) == np.nextafter(1.0, np.inf):
        value = float(np.copysign(1.0, value))
    return value, None


def _scaled_centered_values(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype="float64")
    reference = values[0]
    differences = values - reference
    if np.isfinite(differences).all():
        centered = differences - differences.mean()
    else:
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            return None
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


def _assign_groups_array(values: np.ndarray, n_groups: int) -> np.ndarray:
    lower_ranks, upper_ranks = _ordinal_rank_bounds(values)
    ranks = (lower_ranks + upper_ranks) / 2.0
    group_count = min(n_groups, len(values))
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


def _safe_ratio(numerator: float, denominator: float, *, valid_count: int) -> float:
    if (
        valid_count < 2
        or not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator == 0
    ):
        return np.nan
    return float(numerator / denominator)


def _summarize_correlation_aggregates(
    *,
    count: int,
    mean: float,
    m2: float,
    positive_count: int,
    minimum: float | None = None,
    maximum: float | None = None,
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
    if minimum is not None and maximum is not None and minimum == maximum:
        std = 0.0
    elif not np.isfinite(mean) or not np.isfinite(m2):
        std = np.nan
    else:
        variance = m2 / count
        std = float(np.sqrt(max(0.0, variance))) if np.isfinite(variance) else np.nan
    ratio = _safe_ratio(mean, std, valid_count=count)
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
    stats: dict[str, float | int],
    prefix: str,
    value: object,
) -> None:
    if pd.isna(value):
        return
    numeric = float(value)
    count = int(stats[f"{prefix}_count"]) + 1
    mean_key = f"{prefix}_mean"
    m2_key = f"{prefix}_m2"
    delta = numeric - float(stats[mean_key])
    mean = float(stats[mean_key]) + delta / count
    stats[mean_key] = mean
    stats[m2_key] += delta * (numeric - mean)
    stats[f"{prefix}_count"] = count
    stats[f"{prefix}_positive"] += int(numeric > 0)
    stats[f"{prefix}_min"] = (
        numeric
        if stats[f"{prefix}_min"] is None
        else min(float(stats[f"{prefix}_min"]), numeric)
    )
    stats[f"{prefix}_max"] = (
        numeric
        if stats[f"{prefix}_max"] is None
        else max(float(stats[f"{prefix}_max"]), numeric)
    )


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
