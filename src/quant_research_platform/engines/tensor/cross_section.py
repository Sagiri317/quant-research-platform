"""Tensor-backed cross-sectional IC and group-return diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.engines.dataframe.cross_section import (
    GROUP_SPREAD_SUMMARY_COLUMNS,
    GROUPED_RETURN_COLUMNS,
    IC_RESULT_COLUMNS,
    summarize_group_spread,
)
from quant_research_platform.engines.tensor.layout import PanelTensor
from quant_research_platform.engines.tensor.masks import valid_observation_mask


def evaluate_ic_by_timestamp_tensor(tensor: PanelTensor) -> pd.DataFrame:
    """Compute IC and RankIC from a PanelTensor."""

    records: list[dict[str, object]] = []
    mask = valid_observation_mask(tensor)
    for t_offset, timestamp in enumerate(tensor.layout.timestamps):
        labels = tensor.label_values[t_offset]
        label_rank_cache: dict[bytes, np.ndarray] = {}
        for f_offset, feature in enumerate(tensor.layout.features):
            valid = mask[t_offset, :, f_offset]
            factor = tensor.feature_values[t_offset, valid, f_offset]
            label = labels[valid]
            n_obs = len(factor)
            ic = np.nan
            rankic = np.nan
            if n_obs >= 2:
                ic = _pearson(factor, label)
                rankic = _pearson(
                    _rank_average(factor),
                    _label_ranks_for_mask(labels, valid, label_rank_cache),
                )
            records.append(
                {
                    "timestamp": timestamp,
                    "feature": feature,
                    "label": tensor.layout.label,
                    "ic": ic,
                    "rankic": rankic,
                    "n_obs": n_obs,
                }
            )
    return pd.DataFrame.from_records(records, columns=IC_RESULT_COLUMNS)


def evaluate_ic_and_grouped_returns_tensor(
    tensor: PanelTensor,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IC and grouped returns in one tensor timestamp/feature scan."""

    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    ic_records: list[dict[str, object]] = []
    grouped_records: list[dict[str, object]] = []
    mask = valid_observation_mask(tensor)
    for t_offset, timestamp in enumerate(tensor.layout.timestamps):
        labels = tensor.label_values[t_offset]
        label_rank_cache: dict[bytes, np.ndarray] = {}
        for f_offset, feature in enumerate(tensor.layout.features):
            valid = mask[t_offset, :, f_offset]
            factor = tensor.feature_values[t_offset, valid, f_offset]
            label = labels[valid]
            n_obs = len(factor)
            ic = np.nan
            rankic = np.nan
            if n_obs >= 2:
                ic = _pearson(factor, label)
                rankic = _pearson(
                    _rank_average(factor),
                    _label_ranks_for_mask(labels, valid, label_rank_cache),
                )
                groups = _assign_groups(factor, n_groups)
                _append_grouped_return_records(
                    grouped_records,
                    timestamp=timestamp,
                    feature=feature,
                    label_name=tensor.layout.label,
                    label_values=label,
                    groups=groups,
                )
            ic_records.append(
                {
                    "timestamp": timestamp,
                    "feature": feature,
                    "label": tensor.layout.label,
                    "ic": ic,
                    "rankic": rankic,
                    "n_obs": n_obs,
                }
            )
    return (
        pd.DataFrame.from_records(ic_records, columns=IC_RESULT_COLUMNS),
        pd.DataFrame.from_records(grouped_records, columns=GROUPED_RETURN_COLUMNS),
    )


def analyze_grouped_returns_tensor(
    tensor: PanelTensor,
    n_groups: int = 5,
) -> pd.DataFrame:
    """Compute timestamp-level mean label value by tensor feature rank group."""

    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    records: list[dict[str, object]] = []
    mask = valid_observation_mask(tensor)
    for t_offset, timestamp in enumerate(tensor.layout.timestamps):
        labels = tensor.label_values[t_offset]
        for f_offset, feature in enumerate(tensor.layout.features):
            valid = mask[t_offset, :, f_offset]
            factor = tensor.feature_values[t_offset, valid, f_offset]
            label = labels[valid]
            if len(factor) < 2:
                continue
            groups = _assign_groups(factor, n_groups)
            _append_grouped_return_records(
                records,
                timestamp=timestamp,
                feature=feature,
                label_name=tensor.layout.label,
                label_values=label,
                groups=groups,
            )
    return pd.DataFrame.from_records(records, columns=GROUPED_RETURN_COLUMNS)


def summarize_group_spread_tensor(grouped_returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize tensor grouped-return output with the DataFrame policy."""

    summary = summarize_group_spread(grouped_returns)
    return summary[GROUP_SPREAD_SUMMARY_COLUMNS]


def _assign_groups(values: np.ndarray, n_groups: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(1, len(values) + 1, dtype="float64")
    group_count = min(n_groups, len(values))
    zero_based = np.floor((ranks - 1.0) * group_count / len(ranks))
    return np.clip(zero_based.astype("int64") + 1, 1, group_count)


def _append_grouped_return_records(
    records: list[dict[str, object]],
    *,
    timestamp: object,
    feature: str,
    label_name: str,
    label_values: np.ndarray,
    groups: np.ndarray,
) -> None:
    group_count = int(groups.max())
    counts = np.bincount(groups, minlength=group_count + 1)[1:]
    sums = np.bincount(groups, weights=label_values, minlength=group_count + 1)[1:]
    means = sums / counts
    for group_offset, (mean_return, n_obs) in enumerate(
        zip(means, counts, strict=True),
        start=1,
    ):
        records.append(
            {
                "timestamp": timestamp,
                "feature": feature,
                "label": label_name,
                "group": group_offset,
                "mean_forward_return": float(mean_return),
                "n_obs": int(n_obs),
            }
        )


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _label_ranks_for_mask(
    labels: np.ndarray,
    valid: np.ndarray,
    cache: dict[bytes, np.ndarray],
) -> np.ndarray:
    key = valid.tobytes()
    ranks = cache.get(key)
    if ranks is None:
        ranks = _rank_average(labels[valid])
        cache[key] = ranks
    return ranks


def _rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype="float64")
