"""Time-period breakdown metrics built from completed overall evaluation outputs."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.engines.dataframe.cross_section import (
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.engines.dataframe.performance import (
    calculate_cumulative_pnl,
    summarize_performance,
)

SUPPORTED_TIME_BREAKDOWNS = ("year", "quarter", "month")
PERIOD_BREAKDOWN_POLICY = "slice_overall_returns_rebase_period_drawdown"


def build_time_breakdown_metrics(
    metrics: dict[str, pd.DataFrame],
    *,
    time_breakdowns: tuple[str, ...],
    segment_columns: tuple[str, ...] = (),
) -> dict[str, pd.DataFrame]:
    """Build summary metrics by year, quarter, and/or month.

    IC and group spread breakdowns summarize already-computed cross-sectional
    daily outputs. Performance breakdowns slice completed overall portfolio
    returns, turnover, and costs; only drawdown is rebased inside each requested
    period by starting the period-local high-water mark at zero.
    """

    breakdowns: dict[str, pd.DataFrame] = {}
    for period in time_breakdowns:
        _validate_period(period)
        period_segments = (period, *segment_columns)
        ic = _with_period_column(metrics["ic"], period=period)
        grouped_returns = _with_period_column(metrics["grouped_returns"], period=period)
        returns = _with_period_column(metrics["returns"], period=period)
        pnl = calculate_cumulative_pnl(
            returns,
            segment_columns=period_segments,
        )
        pnl = _rebase_period_drawdown_to_zero(pnl, segment_columns=period_segments)
        breakdowns[f"ic_summary_by_{period}"] = summarize_ic(
            ic,
            segment_columns=period_segments,
        )
        breakdowns[f"group_spread_summary_by_{period}"] = summarize_group_spread(
            grouped_returns,
            segment_columns=period_segments,
        )
        breakdowns[f"performance_summary_by_{period}"] = summarize_performance(
            returns,
            pnl,
            segment_columns=period_segments,
        )
    return breakdowns


def _rebase_period_drawdown_to_zero(
    pnl: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Use a zero starting high-water mark for period-local drawdown stats."""

    if pnl.empty:
        return pnl
    result = pnl.copy()
    group_columns = [*segment_columns, "feature"]
    result["high_water_mark"] = result.groupby(group_columns, sort=False)[
        "cumulative_pnl"
    ].transform(lambda values: values.cummax().clip(lower=0.0))
    result["drawdown"] = result["cumulative_pnl"] - result["high_water_mark"]
    return result


def _validate_period(period: str) -> None:
    if period not in SUPPORTED_TIME_BREAKDOWNS:
        raise ValueError(
            f"Unsupported time breakdown {period!r}; "
            f"expected one of {list(SUPPORTED_TIME_BREAKDOWNS)}."
        )


def _with_period_column(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise ValueError("time breakdowns require metric outputs with timestamp.")
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    if timestamp.isna().any():
        raise ValueError("time breakdowns require valid timestamps.")
    if period == "year":
        result[period] = timestamp.dt.year.astype(int)
    elif period == "quarter":
        result[period] = timestamp.dt.to_period("Q").astype(str)
    elif period == "month":
        result[period] = timestamp.dt.to_period("M").astype(str)
    else:
        _validate_period(period)
    return result
