"""Portfolio output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts._membership_validation import (
    validate_string_membership_column,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
    PortfolioSpec,
)

PORTFOLIO_ARTIFACTS = {
    "positions": "positions.parquet",
    "turnover": "turnover.parquet",
    "cost_breakdown": "cost_breakdown.parquet",
    "portfolio_returns": "portfolio_returns.parquet",
    "cumulative_returns": "cumulative_returns.parquet",
    "performance_summary": "performance_summary.parquet",
    "exposure_summary": "exposure_summary.parquet",
    "concentration": "concentration.parquet",
    "period_breakdown": "period_breakdown.parquet",
    "diagnostics": "diagnostics.parquet",
    "capacity_diagnostics": "capacity_diagnostics.parquet",
    "numeric_exposure_diagnostics": "numeric_exposure_diagnostics.parquet",
    "categorical_exposure_diagnostics": "categorical_exposure_diagnostics.parquet",
}
_PORTFOLIO_POSITION_COLUMNS = ("timestamp", "asset", "signal", "position")
_PORTFOLIO_TURNOVER_COLUMNS = ("timestamp", "signal", "turnover")
_PORTFOLIO_COST_BREAKDOWN_COLUMNS = (
    "timestamp",
    "signal",
    "buy_turnover",
    "sell_turnover",
    "turnover",
    "buy_cost",
    "sell_cost",
    "slippage_cost",
    "spread_cost",
    "impact_cost",
    "total_cost",
    "cost_model",
)
_PORTFOLIO_RETURN_COLUMNS = (
    "timestamp",
    "signal",
    "return_source_column",
    "portfolio_mode",
    "long_return",
    "short_return",
    "gross_portfolio_return",
    "turnover",
    "cost",
    "net_portfolio_return",
    "long_count",
    "short_count",
)
_PORTFOLIO_CUMULATIVE_RETURN_COLUMNS = (
    "timestamp",
    "signal",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "cumulative_gross_return",
    "cumulative_net_return",
    "cumulative_cost",
    "high_water_mark",
    "drawdown",
)
_PORTFOLIO_PERFORMANCE_SUMMARY_COLUMNS = (
    "signal",
    "periods",
    "final_cumulative_gross_return",
    "final_cumulative_net_return",
    "mean_net_portfolio_return",
    "volatility_net_portfolio_return",
    "downside_volatility",
    "sharpe_like",
    "sortino_like",
    "calmar_like",
    "best_period_return",
    "worst_period_return",
    "skew",
    "kurtosis",
    "var_95",
    "cvar_95",
    "tail_ratio",
    "positive_return_rate",
    "negative_return_rate",
    "win_rate",
    "loss_rate",
    "profit_factor",
    "payoff_ratio",
    "average_win",
    "average_loss",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "max_drawdown",
    "drawdown_duration",
    "max_drawdown_start",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "mean_turnover",
    "max_turnover",
    "total_cost",
    "mean_cost",
    "cost_to_gross_return",
    "cost_to_net_return",
)
_PORTFOLIO_EXPOSURE_SUMMARY_COLUMNS = (
    "signal",
    "mean_gross_exposure",
    "mean_net_exposure",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "mean_long_count",
    "mean_short_count",
    "mean_position_count",
)
_PORTFOLIO_CONCENTRATION_COLUMNS = (
    "signal",
    "periods",
    "mean_hhi",
    "max_hhi",
    "mean_top_weight",
    "max_top_weight",
    "mean_position_count",
)
_PORTFOLIO_PERIOD_BREAKDOWN_COLUMNS = (
    "signal",
    "year",
    "month",
    "periods",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "turnover",
    "hit_rate",
)
_PORTFOLIO_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "method",
    "universe_count",
    "eligible_count",
    "can_enter_count",
    "can_exit_count",
    "entry_blocked_count",
    "exit_blocked_count",
    "long_count",
    "short_count",
    "fallback_reason",
)


def portfolio_required_columns() -> dict[str, tuple[str, ...]]:
    return {
        "positions": _PORTFOLIO_POSITION_COLUMNS,
        "turnover": _PORTFOLIO_TURNOVER_COLUMNS,
        "cost_breakdown": _PORTFOLIO_COST_BREAKDOWN_COLUMNS,
        "portfolio_returns": _PORTFOLIO_RETURN_COLUMNS,
        "cumulative_returns": _PORTFOLIO_CUMULATIVE_RETURN_COLUMNS,
        "performance_summary": _PORTFOLIO_PERFORMANCE_SUMMARY_COLUMNS,
        "exposure_summary": _PORTFOLIO_EXPOSURE_SUMMARY_COLUMNS,
        "concentration": _PORTFOLIO_CONCENTRATION_COLUMNS,
        "period_breakdown": _PORTFOLIO_PERIOD_BREAKDOWN_COLUMNS,
        "diagnostics": _PORTFOLIO_DIAGNOSTIC_COLUMNS,
        "capacity_diagnostics": PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
        "numeric_exposure_diagnostics": PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
        "categorical_exposure_diagnostics": (
            PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS
        ),
    }


def validate_portfolio_artifact_membership_columns(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    spec: PortfolioSpec,
) -> None:
    if "signal" not in frame.columns:
        return
    signal_ids = {signal.signal_id for signal in spec.signals}
    validate_string_membership_column(
        frame,
        "signal",
        signal_ids,
        "portfolio_artifact_signal_mismatch: " + artifact_name,
    )
    if "return_source_column" in frame.columns:
        validate_string_membership_column(
            frame,
            "return_source_column",
            (spec.return_source_column,),
            "portfolio_return_source_column_mismatch",
        )
