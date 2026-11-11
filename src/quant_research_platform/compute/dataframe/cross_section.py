"""DataFrame cross-sectional math kernels for long panel inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import validate_compute_panel
from quant_research_platform.contracts.evaluations import (
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
)
from quant_research_platform.contracts.signals import DEFAULT_N_GROUPS


def leave_one_out_mean(values, group_ids):
    """Use centered peer means without subtracting self from a large total."""
    sides = []
    for step in (1, -1):
        ordered, keys = values.iloc[::step], group_ids.iloc[::step]
        groups = ordered.groupby(keys, sort=False)
        prior = groups.shift()
        count = prior.notna().groupby(keys, sort=False).cumsum()
        anchor = groups.transform("first")
        centered = (prior - anchor).fillna(0).groupby(keys, sort=False).cumsum()
        sides.append(
            (
                (anchor + centered / count.where(count > 0)).iloc[::step],
                count.iloc[::step],
            )
        )
    (left, nleft), (right, nright) = sides
    base = left.fillna(right)
    return base + (right - base).fillna(0) * nright / (nleft + nright)


def _compute_ic_and_grouped_label_values(
    frame,
    spec,
    *,
    n_groups=DEFAULT_N_GROUPS,
    top_quantile=None,
    include_groups=True,
):
    panel = validate_compute_panel(frame, spec)
    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    ic_records, grouped_records = [], []
    for timestamp, timestamp_frame in panel.groupby(spec.timestamp_column, sort=True):
        records, groups = _correlation_records(
            timestamp_frame,
            signal_columns=spec.signal_columns,
            label_columns=spec.label_columns,
            timestamp=timestamp,
            n_groups=n_groups,
            top_quantile=top_quantile,
            include_groups=include_groups,
        )
        ic_records.extend(records)
        grouped_records.extend(groups)
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


def _correlation_records(
    frame,
    *,
    signal_columns,
    label_columns,
    timestamp,
    n_groups=DEFAULT_N_GROUPS,
    top_quantile=None,
    include_groups=True,
):
    """Correlate one declared sample block, sharing ranks across labels."""
    ic_records, grouped_records = [], []
    signals = {}
    for signal in signal_columns:
        source = frame[signal]
        values = source.to_numpy(dtype="float64", na_value=np.nan)
        predicted = np.isfinite(values)
        codes = _rank_codes(source, predicted)
        if include_groups:
            groups = _quantile_group_masks(codes[predicted], n_groups)
        else:
            groups = ()
        domains = {"all": (predicted, int(predicted.sum()))}
        if top_quantile is not None:
            for side, ascending in (("long", False), ("short", True)):
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
            domains,
            {},
        )
    for label in label_columns:
        label_source = frame[label]
        label_values = label_source.to_numpy(dtype="float64", na_value=np.nan)
        observed = np.isfinite(label_values)
        label_codes = _rank_codes(label_source, observed)
        label_ranks_by_mask = {}
        for signal, (
            values,
            predicted,
            codes,
            groups,
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
                        signal_ranks_by_mask[mask_key] = _average_ranks_from_codes(
                            codes[valid_mask]
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
                for group, members in enumerate(groups, start=1):
                    values = predicted_labels[members & observed_labels]
                    grouped_records.append(
                        {
                            "timestamp": timestamp,
                            "signal": signal,
                            "label": label,
                            "group": group,
                            "mean_label_value": float(values.mean())
                            if len(values)
                            else np.nan,
                            "n_obs": len(values),
                        }
                    )
    return ic_records, grouped_records


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


def _quantile_group_masks(values: np.ndarray, n_groups: int) -> tuple[np.ndarray, ...]:
    """Keep the same tied extreme members as side IC and equal-weight targets."""
    low = _select_quantile_boundary(values, 1 / n_groups, ascending=True)
    high = _select_quantile_boundary(values, 1 / n_groups, ascending=False)
    remaining = ~(low | high)
    groups = [low]
    for group in range(2, n_groups):
        selected = (
            remaining
            if group == n_groups - 1
            else remaining
            & _select_quantile_boundary(values, group / n_groups, ascending=True)
        )
        groups.append(selected)
        remaining = remaining & ~selected
    return (*groups, high)


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
