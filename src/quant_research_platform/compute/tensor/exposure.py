"""Tensor portfolio exposure algorithms."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXPOSURE_SUMMARY_COLUMNS,
)


def summarize_tensor_exposure_from_path(
    exposure_path: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """基于 streamed timestamp exposure path 汇总 Tensor exposure。"""

    output_columns = [*segment_columns, *EXPOSURE_SUMMARY_COLUMNS]
    if exposure_path.empty:
        return pd.DataFrame(columns=output_columns)
    group_columns = [*segment_columns, "feature"]
    exposure_summary = (
        exposure_path.groupby(group_columns, sort=True, dropna=False)
        .agg(
            mean_gross_exposure=("gross_exposure", "mean"),
            mean_net_exposure=("net_exposure", "mean"),
            max_gross_exposure=("gross_exposure", "max"),
            max_abs_net_exposure=("net_exposure", lambda values: values.abs().max()),
            mean_position_count=("position_count", "mean"),
        )
        .reset_index()
    )
    count_summary = (
        returns.groupby(group_columns, sort=True, dropna=False)
        .agg(
            mean_long_count=("long_count", "mean"),
            mean_short_count=("short_count", "mean"),
        )
        .reset_index()
        if not returns.empty
        else pd.DataFrame(
            columns=[*group_columns, "mean_long_count", "mean_short_count"],
        )
    )
    summary = exposure_summary.merge(
        count_summary,
        on=group_columns,
        how="left",
    )
    summary["mean_long_count"] = pd.to_numeric(
        summary["mean_long_count"],
        errors="coerce",
    )
    summary["mean_short_count"] = pd.to_numeric(
        summary["mean_short_count"],
        errors="coerce",
    )
    return summary[output_columns]

def _exposure_record_from_arrays(
    *,
    context: dict[str, object],
    timestamp: object,
    feature: str,
    exposure_values: dict[str, float | int],
    segment_columns: tuple[str, ...],
) -> dict[str, object]:
    return {
        **{column: context[column] for column in segment_columns},
        "timestamp": timestamp,
        "feature": feature,
        **exposure_values,
    }

__all__ = [
    "summarize_tensor_exposure_from_path",
    "_exposure_record_from_arrays",
]
