"""基于完整 evaluation output 构造 time-period breakdown metrics。"""

from __future__ import annotations

import pandas as pd

from quant_research_platform.kernels.dataframe.cross_section import (
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.kernels.dataframe.performance import (
    calculate_cumulative_returns,
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
    """按 year、quarter 或 month 构建 summary metrics。

    IC 和 group spread breakdown 汇总已计算的 cross-sectional daily outputs。
    Performance breakdown 切分整体 portfolio returns、turnover 和 costs；
    仅 drawdown 会在每个请求周期内使用零起点 high-water mark 重新基准化。
    """

    breakdowns: dict[str, pd.DataFrame] = {}
    for period in time_breakdowns:
        _validate_period(period)
        period_segments = (period, *segment_columns)
        ic = _with_period_column(metrics["ic"], period=period)
        grouped_returns = _with_period_column(metrics["grouped_returns"], period=period)
        returns = _with_period_column(metrics["returns"], period=period)
        cumulative_returns = calculate_cumulative_returns(
            returns,
            segment_columns=period_segments,
        )
        cumulative_returns = _rebase_period_drawdown_to_zero(
            cumulative_returns,
            segment_columns=period_segments,
        )
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
            cumulative_returns,
            segment_columns=period_segments,
        )
    return breakdowns


def _rebase_period_drawdown_to_zero(
    cumulative_returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...],
) -> pd.DataFrame:
    """使用零起点 high-water mark 计算 period-local drawdown stats。"""

    if cumulative_returns.empty:
        return cumulative_returns
    result = cumulative_returns.copy()
    group_columns = [*segment_columns, "feature"]
    result["high_water_mark"] = result.groupby(group_columns, sort=False)[
        "cumulative_net_return"
    ].transform(lambda values: values.cummax().clip(lower=0.0))
    result["drawdown"] = result["cumulative_net_return"] - result["high_water_mark"]
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
