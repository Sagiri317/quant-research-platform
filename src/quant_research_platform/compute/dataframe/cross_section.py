"""DataFrame cross-sectional math kernels for long panel inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import validate_compute_panel
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
)


def _compute_ic_and_grouped_label_values(
    frame,
    spec,
    *,
    n_groups=5,
    ordinal_signals=None,
    top_quantile=None,
    include_groups=True,
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
        signals = {}
        for signal in spec.signal_columns:
            source = timestamp_frame[signal]
            values = source.to_numpy(dtype="float64", na_value=np.nan)
            predicted = np.isfinite(values)
            reverse = ordinal_signals is not None and signal in ordinal_signals
            order = (
                ordinal_signals.loc[timestamp_frame.index, signal]
                if reverse
                else source
            )
            codes = _rank_codes(order, predicted)
            if include_groups:
                ranks = _average_ranks_from_codes(codes[predicted])
                if reverse:
                    ranks = len(ranks) + 1 - ranks
                groups = _assign_groups_from_ranks(ranks, n_groups)
            else:
                groups = np.empty(0, dtype="int64")
            domains = {"all": (predicted, int(predicted.sum()))}
            if top_quantile is not None:
                for side, ascending in (("long", reverse), ("short", not reverse)):
                    selected = np.zeros(len(predicted), dtype=bool)
                    selected[predicted] = _select_quantile_boundary(
                        codes[predicted], top_quantile, ascending=ascending
                    )
                    domains[side] = (selected, int(selected.sum()))
            signals[signal] = (
                values,
                predicted,
                codes,
                groups,
                np.unique(groups),
                reverse,
                domains,
                {},
            )
        for label in spec.label_columns:
            label_source = timestamp_frame[label]
            label_values = label_source.to_numpy(dtype="float64", na_value=np.nan)
            observed = np.isfinite(label_values)
            label_codes = _rank_codes(label_source, observed)
            label_ranks_by_mask = {}
            for signal, (
                values,
                predicted,
                codes,
                groups,
                group_ids,
                reverse,
                domains,
                signal_ranks_by_mask,
            ) in signals.items():
                record = {"timestamp": timestamp, "signal": signal, "label": label}
                for side, (selected, n_selected) in domains.items():
                    valid_mask = selected & observed
                    n_obs = int(valid_mask.sum())
                    ic = rankic = np.nan
                    ic_reason = rankic_reason = "insufficient_observations"
                    if n_obs >= 2:
                        ic, ic_reason = _pearson_correlation_result(
                            values[valid_mask], label_values[valid_mask]
                        )
                        # Rerank on the paired side domain, retaining exact tie
                        # classes. Selection never depends on label availability.
                        mask_key = valid_mask.tobytes()
                        if mask_key not in signal_ranks_by_mask:
                            ranks = _average_ranks_from_codes(codes[valid_mask])
                            signal_ranks_by_mask[mask_key] = (
                                -ranks if reverse else ranks
                            )
                        if mask_key not in label_ranks_by_mask:
                            label_ranks_by_mask[mask_key] = _average_ranks_from_codes(
                                label_codes[valid_mask]
                            )
                        rankic, rankic_reason = _pearson_correlation_result(
                            signal_ranks_by_mask[mask_key],
                            label_ranks_by_mask[mask_key],
                        )
                    prefix = "" if side == "all" else f"{side}_"
                    record.update(
                        {
                            f"{prefix}ic": ic,
                            f"{prefix}rankic": rankic,
                            f"{prefix}n_obs": n_obs,
                            f"{prefix}ic_reason": ic_reason,
                            f"{prefix}rankic_reason": rankic_reason,
                        }
                    )
                    if prefix:
                        record[f"{prefix}n_selected"] = n_selected
                ic_records.append(record)
                if include_groups:
                    predicted_labels = label_values[predicted]
                    observed_labels = np.isfinite(predicted_labels)
                    observed_groups = groups[observed_labels]
                    group_count = min(n_groups, len(groups)) + 1
                    group_counts = np.bincount(observed_groups, minlength=group_count)
                    group_sums = np.bincount(
                        observed_groups,
                        weights=predicted_labels[observed_labels],
                        minlength=group_count,
                    )
                    for group in group_ids:
                        grouped_records.append(
                            {
                                "timestamp": timestamp,
                                "signal": signal,
                                "label": label,
                                "group": group,
                                "mean_label_value": float(
                                    group_sums[group] / group_counts[group]
                                )
                                if group_counts[group]
                                else np.nan,
                                "n_obs": int(group_counts[group]),
                            }
                        )
    metrics = _correlation_metrics_frame(ic_records)

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


def _correlation_metrics_frame(records, *, side_reason="not_applicable"):
    """Materialize one shared schema for cross-sectional and time-series results."""
    metrics = pd.DataFrame.from_records(records)
    for side in ("long", "short"):
        for name in (
            "ic",
            "rankic",
            "n_obs",
            "n_selected",
            "ic_reason",
            "rankic_reason",
        ):
            column = f"{side}_{name}"
            if column not in metrics:
                metrics[column] = (
                    side_reason
                    if name.endswith("reason")
                    else 0
                    if name.startswith("n_")
                    else np.nan
                )
    metrics = metrics.reindex(columns=EVALUATION_METRICS_COLUMNS)
    for column in EVALUATION_METRICS_COLUMNS:
        dtype = (
            "datetime64[ns]"
            if column == "timestamp"
            else "string"
            if column in {"signal", "label"} or column.endswith("reason")
            else "int64"
            if column.endswith(("n_obs", "n_selected"))
            else "float64"
        )
        metrics[column] = metrics[column].astype(dtype)
    return metrics


def _select_quantile_boundary(values, quantile, *, ascending):
    """Select floor(N*q), at least one, retaining every boundary tie.

    Inputs are finite values or their ordered tie codes. Partition preserves
    integer ordering and avoids the former full sort plus average-rank pass.
    """
    values = np.asarray(values)
    if not len(values):
        return np.zeros(0, dtype=bool)
    count = max(1, int(len(values) * quantile))
    position = count - 1 if ascending else len(values) - count
    boundary = np.partition(values, position)[position]
    return values <= boundary if ascending else values >= boundary


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


def _rank_codes(values: pd.Series, observed: np.ndarray) -> np.ndarray:
    """Sort observed tie classes once, preserving the source's ordinal precision."""
    codes = np.full(len(values), -1, dtype=np.intp)
    codes[observed] = np.unique(values.loc[observed].to_numpy(), return_inverse=True)[1]
    return codes


def _average_ranks_from_codes(codes: np.ndarray) -> np.ndarray:
    """Rank a subset of ordered tie classes without sorting it again."""
    counts = np.bincount(codes)
    return np.cumsum(counts)[codes] - (counts[codes] - 1) / 2.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return _average_ranks_from_codes(np.unique(values, return_inverse=True)[1])


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


def _winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
) -> pd.Series:
    source = grouped.obj if hasattr(grouped, "obj") else values
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    for positions in (
        grouped.indices if hasattr(grouped, "indices") else grouped
    ).values():
        lower[positions], upper[positions] = _quantile(
            source.iloc[positions], 0.01, 0.99
        )
    return values.clip(
        lower=pd.Series(lower, index=values.index),
        upper=pd.Series(upper, index=values.index),
    )


def _quantile(values: pd.Series, *quantiles: float) -> tuple[float, ...]:
    observed = values.dropna().to_numpy(dtype="float64")
    return (
        tuple(np.quantile(observed, quantiles))
        if len(observed)
        else (np.nan,) * len(quantiles)
    )


def _mad_winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
    *,
    zero_counts: dict[str, int] | None = None,
    column: str | None = None,
) -> pd.Series:
    source = grouped.obj if hasattr(grouped, "obj") else values
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    for positions in (
        grouped.indices if hasattr(grouped, "indices") else grouped
    ).values():
        observed = source.iloc[positions].dropna().to_numpy(dtype="float64")
        if not len(observed):
            continue
        center = np.median(observed)
        mad = np.median(np.abs(observed - center))
        if mad == 0:
            if zero_counts is not None:
                zero_counts[column] += 1
            continue
        bound = 3 * 1.4826 * mad
        lower[positions], upper[positions] = center - bound, center + bound
    return values.clip(
        lower=pd.Series(lower, index=values.index),
        upper=pd.Series(upper, index=values.index),
    )


def _industry_group_codes(
    frame: pd.DataFrame, industry_column: str = "industry_level1_code"
) -> pd.Series:
    """Encode the standard/code pair without conflating literal separators."""
    standard, code = (
        frame[column]
        .astype("string")
        .str.replace("\\", "\\\\", regex=False)
        .str.replace(":", "\\:", regex=False)
        for column in ("industry_standard_id", industry_column)
    )
    return standard.str.cat(code, sep=":")
