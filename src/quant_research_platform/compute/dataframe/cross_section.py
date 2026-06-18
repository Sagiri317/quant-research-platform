"""DataFrame cross-sectional math kernels for long panel inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)

IC_RESULT_COLUMNS = ["timestamp", "signal", "label", "ic", "rankic", "n_obs"]
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
]


def compute_ic_by_timestamp(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
) -> pd.DataFrame:
    """ ?timestamp  ?signal     cross-sectional IC  ?RankIC ?"""

    panel = validate_compute_panel(frame, spec)
    records: list[dict[str, object]] = []
    for timestamp, timestamp_frame in panel.groupby(spec.timestamp_column, sort=True):
        for signal in spec.signal_columns:
            pair = _finite_pair(timestamp_frame, signal, spec.label_column)
            n_obs = len(pair)
            ic = np.nan
            rankic = np.nan
            if n_obs >= 2:
                ic = _pearson_correlation(pair[signal], pair[spec.label_column])
                rankic = _pearson_correlation(
                    pair[signal].rank(method="average"),
                    pair[spec.label_column].rank(method="average"),
                )
            records.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "label": spec.label_column,
                    "ic": ic,
                    "rankic": rankic,
                    "n_obs": n_obs,
                }
            )

    return pd.DataFrame.from_records(records, columns=IC_RESULT_COLUMNS)


def compute_ic_and_grouped_label_values(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """    ?timestamp/signal        ?IC  ?grouped label values ?"""

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
    """             panel     IC  ?grouped label values ?"""

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
            if n_obs >= 2:
                ic = _pearson_correlation_values(signal_valid, label_valid)
                rankic = _pearson_correlation_values(
                    _average_ranks(signal_valid),
                    _average_ranks(label_valid),
                )
                groups = _assign_groups_array(signal_valid, n_groups)
                for group in range(1, int(groups.max()) + 1):
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
                }
            )

    return (
        pd.DataFrame.from_records(ic_records, columns=IC_RESULT_COLUMNS),
        pd.DataFrame.from_records(grouped_records, columns=GROUPED_LABEL_VALUE_COLUMNS),
    )


def compute_grouped_label_values(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    n_groups: int = 5,
) -> pd.DataFrame:
    """ ?signal rank group     timestamp-level mean label value ?"""

    if not isinstance(n_groups, int) or isinstance(n_groups, bool):
        raise TypeError("n_groups must be an integer.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")

    panel = validate_compute_panel(frame, spec)
    records: list[dict[str, object]] = []
    for timestamp, timestamp_frame in panel.groupby(spec.timestamp_column, sort=True):
        for signal in spec.signal_columns:
            pair = _finite_pair(timestamp_frame, signal, spec.label_column)
            if len(pair) < 2:
                continue
            groups = _assign_groups(pair[signal], n_groups).to_numpy()
            label_values = pair[spec.label_column].to_numpy(dtype="float64", copy=False)
            for group in sorted(np.unique(groups)):
                group_mask = groups == group
                records.append(
                    {
                        "timestamp": timestamp,
                        "signal": signal,
                        "label": spec.label_column,
                        "group": int(group),
                        "mean_label_value": float(label_values[group_mask].mean()),
                        "n_obs": int(group_mask.sum()),
                    }
                )

    return pd.DataFrame.from_records(records, columns=GROUPED_LABEL_VALUE_COLUMNS)


def _finite_pair(frame: pd.DataFrame, signal: str, label: str) -> pd.DataFrame:
    pair = frame[[signal, label]].apply(pd.to_numeric, errors="coerce")
    return pair[np.isfinite(pair[signal]) & np.isfinite(pair[label])]


def summarize_ic(
    ic: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """ ?signal    ?IC stability  ?sign diagnostics ?"""

    required_columns = {*segment_columns, *IC_RESULT_COLUMNS}
    missing_columns = sorted(required_columns.difference(ic.columns))
    if missing_columns:
        raise ValueError(f"ic is missing columns: {missing_columns}.")

    output_columns = [*segment_columns, *IC_SUMMARY_COLUMNS]
    if ic.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, object]] = []
    group_columns = [*segment_columns, "signal", "label"]
    for group_key, group in ic.groupby(group_columns, sort=True, dropna=False):
        group_values = _normalize_group_key(group_key, group_columns)
        ic_values = pd.to_numeric(group["ic"], errors="coerce").dropna()
        rankic_values = pd.to_numeric(group["rankic"], errors="coerce").dropna()
        ic_stats = _summarize_correlation_values(ic_values)
        rankic_stats = _summarize_correlation_values(rankic_values)
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                "signal": group_values["signal"],
                "label": group_values["label"],
                "mean_ic": ic_stats["mean"],
                "std_ic": ic_stats["std"],
                "icir": ic_stats["ratio"],
                "positive_ic_rate": ic_stats["positive_rate"],
                "valid_ic_count": ic_stats["valid_count"],
                "mean_rankic": rankic_stats["mean"],
                "std_rankic": rankic_stats["std"],
                "rankicir": rankic_stats["ratio"],
                "positive_rankic_rate": rankic_stats["positive_rate"],
                "valid_rankic_count": rankic_stats["valid_count"],
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)


def summarize_group_spread(
    grouped_label_values: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """ ?signal    ?top-bottom group label-value spread diagnostics ?"""

    required_columns = {*segment_columns, *GROUPED_LABEL_VALUE_COLUMNS}
    missing_columns = sorted(required_columns.difference(grouped_label_values.columns))
    if missing_columns:
        raise ValueError(
            f"grouped_label_values is missing columns: {missing_columns}."
        )

    output_columns = [*segment_columns, *GROUP_SPREAD_SUMMARY_COLUMNS]
    if grouped_label_values.empty:
        return pd.DataFrame(columns=output_columns)

    timestamp_group_columns = [*segment_columns, "signal", "label", "timestamp"]
    ordered = grouped_label_values[
        [*timestamp_group_columns, "group", "mean_label_value"]
    ].copy()
    ordered["mean_label_value"] = pd.to_numeric(
        ordered["mean_label_value"],
        errors="coerce",
    )
    ordered = ordered.sort_values([*timestamp_group_columns, "group"], kind="mergesort")
    ordered["_return_diff_sign"] = np.sign(
        ordered.groupby(timestamp_group_columns, sort=True, dropna=False)[
            "mean_label_value"
        ].diff()
    )
    timestamp_frame = (
        ordered.groupby(
            timestamp_group_columns,
            sort=True,
            dropna=False,
        )
        .agg(
            top_label_value=("mean_label_value", "last"),
            bottom_label_value=("mean_label_value", "first"),
            monotonicity=("_return_diff_sign", "mean"),
        )
        .reset_index()
    )
    if timestamp_frame.empty:
        return pd.DataFrame(columns=output_columns)
    timestamp_frame["monotonicity"] = timestamp_frame["monotonicity"].fillna(0.0)
    timestamp_frame["spread"] = (
        timestamp_frame["top_label_value"] - timestamp_frame["bottom_label_value"]
    )

    summary_group_columns = [*segment_columns, "signal", "label"]
    summary = (
        timestamp_frame.groupby(
            summary_group_columns,
            sort=True,
            dropna=False,
        )
        .agg(
            top_group_mean_label_value=("top_label_value", "mean"),
            bottom_group_mean_label_value=("bottom_label_value", "mean"),
            top_bottom_label_value_spread_mean=("spread", "mean"),
            top_bottom_label_value_spread_hit_rate=(
                "spread",
                lambda values: (values > 0).mean(),
            ),
            group_monotonicity_score=("monotonicity", "mean"),
        )
        .reset_index()
    )
    return summary[output_columns]


def _assign_groups(values: pd.Series, n_groups: int) -> pd.Series:
    groups = _assign_groups_array(values.to_numpy(dtype="float64"), n_groups)
    return pd.Series(groups, index=values.index)


def _pearson_correlation(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(dtype="float64", copy=False)
    right_values = right.to_numpy(dtype="float64", copy=False)
    return _pearson_correlation_values(left_values, right_values)


def _pearson_correlation_values(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> float:
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = np.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _assign_groups_array(values: np.ndarray, n_groups: int) -> np.ndarray:
    ranks = np.empty(len(values), dtype="float64")
    ranks[np.argsort(values, kind="mergesort")] = np.arange(1, len(values) + 1)
    group_count = min(n_groups, len(values))
    zero_based = np.floor((ranks - 1.0) * group_count / len(ranks))
    return np.clip(zero_based.astype("int64") + 1, 1, group_count)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype="float64")


def _safe_ratio(numerator: float, denominator: float, *, valid_count: int) -> float:
    if (
        valid_count < 2
        or not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or np.isclose(denominator, 0.0)
    ):
        return np.nan
    return float(numerator / denominator)


def _summarize_correlation_values(values: pd.Series) -> dict[str, float | int]:
    valid_count = int(len(values))
    if valid_count == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ratio": np.nan,
            "positive_rate": np.nan,
            "valid_count": 0,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "mean": mean,
        "std": std,
        "ratio": _safe_ratio(mean, std, valid_count=valid_count),
        "positive_rate": float((values > 0).mean()),
        "valid_count": valid_count,
    }


def _normalize_group_key(
    group_key: object,
    group_columns: list[str],
) -> dict[str, object]:
    if len(group_columns) == 1:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
    else:
        values = tuple(group_key)
    return dict(zip(group_columns, values, strict=True))
