"""Tensor-backed cross-sectional math kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    GROUP_SPREAD_SUMMARY_COLUMNS,
    GROUPED_LABEL_VALUE_COLUMNS,
    IC_RESULT_COLUMNS,
    summarize_group_spread,
)
from quant_research_platform.compute.tensor.layout import PanelTensor
from quant_research_platform.compute.tensor.masks import valid_observation_mask


def compute_ic_by_timestamp_tensor(tensor: PanelTensor) -> pd.DataFrame:
    """Compute IC and RankIC from a PanelTensor."""

    records: list[dict[str, object]] = []
    mask = valid_observation_mask(tensor)
    for t_offset, timestamp in enumerate(tensor.layout.timestamps):
        labels = tensor.label_values[t_offset]
        label_rank_cache: dict[bytes, np.ndarray] = {}
        for f_offset, signal in enumerate(tensor.layout.signals):
            valid = mask[t_offset, :, f_offset]
            factor = tensor.signal_values[t_offset, valid, f_offset]
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
                    "signal": signal,
                    "label": tensor.layout.label,
                    "ic": ic,
                    "rankic": rankic,
                    "n_obs": n_obs,
                }
            )
    return pd.DataFrame.from_records(records, columns=IC_RESULT_COLUMNS)


def compute_ic_and_grouped_label_values_tensor(
    tensor: PanelTensor,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """    ?tensor timestamp/signal        ?IC  ?grouped label values ?"""

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
        for f_offset, signal in enumerate(tensor.layout.signals):
            valid = mask[t_offset, :, f_offset]
            factor = tensor.signal_values[t_offset, valid, f_offset]
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
                _append_grouped_label_value_records(
                    grouped_records,
                    timestamp=timestamp,
                    signal=signal,
                    label_name=tensor.layout.label,
                    label_values=label,
                    groups=groups,
                )
            ic_records.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "label": tensor.layout.label,
                    "ic": ic,
                    "rankic": rankic,
                    "n_obs": n_obs,
                }
            )
    return (
        pd.DataFrame.from_records(ic_records, columns=IC_RESULT_COLUMNS),
        pd.DataFrame.from_records(grouped_records, columns=GROUPED_LABEL_VALUE_COLUMNS),
    )


def compute_grouped_label_values_tensor(
    tensor: PanelTensor,
    n_groups: int = 5,
) -> pd.DataFrame:
    """ ?tensor signal rank group     timestamp-level mean label value ?"""

    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    records: list[dict[str, object]] = []
    mask = valid_observation_mask(tensor)
    for t_offset, timestamp in enumerate(tensor.layout.timestamps):
        labels = tensor.label_values[t_offset]
        for f_offset, signal in enumerate(tensor.layout.signals):
            valid = mask[t_offset, :, f_offset]
            factor = tensor.signal_values[t_offset, valid, f_offset]
            label = labels[valid]
            if len(factor) < 2:
                continue
            groups = _assign_groups(factor, n_groups)
            _append_grouped_label_value_records(
                records,
                timestamp=timestamp,
                signal=signal,
                label_name=tensor.layout.label,
                label_values=label,
                groups=groups,
            )
    return pd.DataFrame.from_records(records, columns=GROUPED_LABEL_VALUE_COLUMNS)


def summarize_group_spread_tensor(grouped_label_values: pd.DataFrame) -> pd.DataFrame:
    """    DataFrame policy    ?tensor grouped label-value output ?"""

    summary = summarize_group_spread(grouped_label_values)
    return summary[GROUP_SPREAD_SUMMARY_COLUMNS]


def _assign_groups(values: np.ndarray, n_groups: int) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(1, len(values) + 1, dtype="float64")
    group_count = min(n_groups, len(values))
    zero_based = np.floor((ranks - 1.0) * group_count / len(ranks))
    return np.clip(zero_based.astype("int64") + 1, 1, group_count)


def _append_grouped_label_value_records(
    records: list[dict[str, object]],
    *,
    timestamp: object,
    signal: str,
    label_name: str,
    label_values: np.ndarray,
    groups: np.ndarray,
) -> None:
    group_count = int(groups.max())
    counts = np.bincount(groups, minlength=group_count + 1)[1:]
    sums = np.bincount(groups, weights=label_values, minlength=group_count + 1)[1:]
    means = sums / counts
    for group_offset, (mean_label_value, n_obs) in enumerate(
        zip(means, counts, strict=True),
        start=1,
    ):
        records.append(
            {
                "timestamp": timestamp,
                "signal": signal,
                "label": label_name,
                "group": group_offset,
                "mean_label_value": float(mean_label_value),
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
