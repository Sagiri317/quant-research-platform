"""DataFrame cross-sectional math kernels for long panel inputs."""

from __future__ import annotations

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

    return _compute_ic_and_grouped_label_values(frame, spec, n_groups=n_groups)


def _compute_ic_and_grouped_label_values(
    frame, spec, *, n_groups=5, ordinal_signals=None
):
    panel = validate_compute_panel(frame, spec)
    if ordinal_signals is not None:
        keys = [spec.timestamp_column, spec.asset_column]
        ordinal_signals = (
            ordinal_signals.set_index(keys)
            .reindex(pd.MultiIndex.from_frame(panel[keys]))
            .reset_index(drop=True)
        )
    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    ic_records: list[dict[str, object]] = []
    grouped_records: list[dict[str, object]] = []
    for timestamp, timestamp_frame in panel.groupby(spec.timestamp_column, sort=True):
        for label in spec.label_columns:
            label_source = timestamp_frame[label]
            label_values = label_source.to_numpy(dtype="float64", na_value=np.nan)
            label_ranks_by_mask = {}
            for signal in spec.signal_columns:
                signal_source = timestamp_frame[signal]
                reverse_order = (
                    ordinal_signals is not None and signal in ordinal_signals
                )
                signal_order = (
                    ordinal_signals.loc[timestamp_frame.index, signal]
                    if reverse_order
                    else signal_source
                )
                signal_values = signal_source.to_numpy(dtype="float64", na_value=np.nan)
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
                    signal_ranks = _average_ranks(
                        signal_order.loc[valid_mask].to_numpy()
                    )
                    if reverse_order:
                        signal_ranks = -signal_ranks
                    mask_key = valid_mask.tobytes()
                    if mask_key not in label_ranks_by_mask:
                        label_ranks_by_mask[mask_key] = _average_ranks(
                            label_source.loc[valid_mask].to_numpy()
                        )
                    rankic, rankic_reason = _pearson_correlation_result(
                        signal_ranks,
                        label_ranks_by_mask[mask_key],
                    )
                predicted = np.isfinite(signal_values)
                predicted_ranks = _average_ranks(signal_order.loc[predicted].to_numpy())
                if reverse_order:
                    predicted_ranks = len(predicted_ranks) + 1 - predicted_ranks
                groups = _assign_groups_from_ranks(predicted_ranks, n_groups)
                predicted_labels = label_values[predicted]
                for group in np.unique(groups):
                    group_mask = (groups == group) & np.isfinite(predicted_labels)
                    grouped_records.append(
                        {
                            "timestamp": timestamp,
                            "signal": signal,
                            "label": label,
                            "group": group,
                            "mean_label_value": float(
                                np.mean(predicted_labels[group_mask])
                            )
                            if group_mask.any()
                            else np.nan,
                            "n_obs": int(group_mask.sum()),
                        }
                    )
                ic_records.append(
                    {
                        "timestamp": timestamp,
                        "signal": signal,
                        "label": label,
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


def _centered_values(
    values: np.ndarray, paired: np.ndarray | None = None
) -> np.ndarray:
    """Center each observation row relative to an observed origin."""
    values = np.asarray(values, dtype="float64")
    if paired is None:
        differences = values - values[..., :1]
        return differences - differences.mean(axis=-1, keepdims=True)
    origin = np.take_along_axis(values, paired.argmax(axis=-1, keepdims=True), axis=-1)
    differences = np.where(paired, values - origin, 0.0)
    mean = differences.sum(axis=-1, keepdims=True) / np.maximum(
        paired.sum(axis=-1, keepdims=True), 1
    )
    return np.where(paired, differences - mean, 0.0)


def _pearson_correlations(left, right, *, paired=None):
    """Reduce complete or explicitly paired arrays along their last axis."""
    left, right = _centered_values(left, paired), _centered_values(right, paired)
    covariance = np.sum(left * right, axis=-1)
    denominator = np.sqrt(np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1))
    return np.clip(
        np.divide(
            covariance,
            denominator,
            out=np.full_like(covariance, np.nan),
            where=denominator > 0,
        ),
        -1.0,
        1.0,
    )


def _stable_zscore(values: pd.Series) -> pd.Series:
    source = values.to_numpy(dtype="float64", na_value=np.nan)
    output = np.full(len(source), np.nan, dtype="float64")
    valid = np.isfinite(source)
    if int(valid.sum()) < 2:
        return pd.Series(output, index=values.index)
    centered = _centered_values(source[valid])
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
        np.asarray(values),
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
    mean: float,
    m2: float,
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
            "std_reason": "no_valid_observations",
            "ratio_reason": "no_valid_observations",
        }
    std = float(np.sqrt(m2 / count))
    mean = float(mean)
    ratio = mean / std if count >= 2 and std > 0 else np.nan
    return {
        "mean": mean,
        "std": std if np.isfinite(std) else np.nan,
        "std_reason": None if np.isfinite(std) else "nonfinite_result",
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
    stats: dict[str, int | float],
    prefix: str,
    value: object,
) -> None:
    if pd.isna(value):
        return
    numeric = float(value)
    count = int(stats[f"{prefix}_count"]) + 1
    mean_key = f"{prefix}_mean"
    m2_key = f"{prefix}_m2"
    value = numeric
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


def _pearson_correlation_result(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> tuple[float | None, str | None]:
    """Compute Pearson with vectorized centered moments and explicit reasons."""
    left = np.asarray(left_values, dtype="float64")
    right = np.asarray(right_values, dtype="float64")
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("correlation_array_shape_invalid")
    if len(left) < 2:
        return None, "insufficient_observations"
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None, "nonfinite_result"
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None, "zero_variance"
    value = float(_pearson_correlations(left, right))
    if not np.isfinite(value):
        return None, "nonfinite_result"
    return value, None
