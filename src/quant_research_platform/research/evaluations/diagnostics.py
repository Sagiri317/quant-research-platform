"""Evaluation diagnostics owned by the research evaluation layer."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    GROUP_SPREAD_SUMMARY_COLUMNS,
    IC_SUMMARY_COLUMNS,
    summarize_ic,
)
from quant_research_platform.compute.dataframe.cross_section import (
    summarize_group_spread as _summarize_group_spread,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
)


def summarize_signal_label_coverage(
    metrics: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
) -> dict[str, int | str | None]:
    """Summarize signal-label observation coverage for evaluation diagnostics."""

    if metrics.empty:
        return {
            "metric_rows": 0,
            "rows_after_signal_label_dropna": 0,
            "timestamp_count": 0,
            "min_timestamp": None,
            "max_timestamp": None,
            "skipped_timestamp_count": 0,
        }
    missing = sorted({timestamp_column, "n_obs"}.difference(metrics.columns))
    if missing:
        raise ValueError(f"missing_evaluation_metric_columns: {', '.join(missing)}")
    timestamps = pd.to_datetime(metrics[timestamp_column], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_evaluation_timestamp_values")
    n_obs = strict_non_negative_count_series(metrics["n_obs"])
    return {
        "metric_rows": int(len(metrics)),
        "rows_after_signal_label_dropna": int(n_obs.sum()),
        "timestamp_count": int(timestamps.nunique()),
        "min_timestamp": date_string(timestamps.min()),
        "max_timestamp": date_string(timestamps.max()),
        "skipped_timestamp_count": int(
            metrics.loc[n_obs.lt(2), timestamp_column].nunique()
        ),
    }


def summarize_group_spread(grouped_label_values: pd.DataFrame) -> pd.DataFrame:
    """Summarize group spread diagnostics with deterministic output columns."""

    if grouped_label_values.empty:
        return pd.DataFrame(columns=GROUP_SPREAD_SUMMARY_COLUMNS)
    return _summarize_group_spread(grouped_label_values)


def summarize_period_ic(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize period IC diagnostics with explicit all-NaN handling."""

    if metrics.empty:
        return pd.DataFrame(columns=IC_SUMMARY_COLUMNS)
    return summarize_ic(metrics)


def build_evaluation_diagnostics(
    *,
    spec: EvaluationSpec,
    filter_metadata: Mapping[str, object],
    rows_after_signal_label_dropna: int,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    n_groups: int,
    timestamp_column: str,
) -> EvaluationDiagnostics:
    coverage = summarize_signal_label_coverage(
        metrics,
        timestamp_column=timestamp_column,
    )
    return EvaluationDiagnostics(
        filter_mode=spec.filter_mode,
        filter_columns_used=tuple(
            str(column) for column in filter_metadata["filter_columns_used"]
        ),
        filter_definition=str(filter_metadata["filter_definition"]),
        rows_input=int(filter_metadata["rows_input"]),
        rows_after_filter_mode=int(filter_metadata["rows_after_filter_mode"]),
        rows_after_signal_label_dropna=rows_after_signal_label_dropna,
        metric_rows=int(coverage["metric_rows"]),
        grouped_label_value_rows=int(len(grouped_label_values)),
        ic_summary_rows=int(len(ic_summary)),
        group_spread_summary_rows=int(len(group_spread_summary)),
        timestamp_count=int(coverage["timestamp_count"]),
        min_timestamp=coverage["min_timestamp"],
        max_timestamp=coverage["max_timestamp"],
        n_groups=n_groups,
        skipped_timestamp_count=int(coverage["skipped_timestamp_count"]),
    )


def date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


__all__ = [
    "summarize_group_spread",
    "summarize_period_ic",
    "summarize_signal_label_coverage",
]
