"""Public helpers for the DataFrame compute backend."""

from __future__ import annotations

from quant_research_platform.compute.dataframe.cross_section import (
    analyze_grouped_returns,
    evaluate_ic_by_timestamp,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.compute.dataframe.performance import (
    calculate_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.compute.dataframe.portfolio import (
    build_weighted_positions,
    calculate_cost_breakdown,
    calculate_turnover,
    simulate_weighted_long_short_returns,
    summarize_exposure,
)

__all__ = [
    "analyze_grouped_returns",
    "build_weighted_positions",
    "calculate_cost_breakdown",
    "calculate_cumulative_returns",
    "calculate_turnover",
    "evaluate_ic_by_timestamp",
    "simulate_weighted_long_short_returns",
    "summarize_exposure",
    "summarize_group_spread",
    "summarize_ic",
    "summarize_performance",
]
