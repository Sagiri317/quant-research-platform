"""Portfolio dataframe output column contracts."""

POSITION_COLUMNS = ["timestamp", "asset", "signal", "position"]
TURNOVER_COLUMNS = ["timestamp", "signal", "turnover"]
RETURNS_COLUMNS = [
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
]
COST_BREAKDOWN_COLUMNS = [
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
]
EXPOSURE_SUMMARY_COLUMNS = [
    "signal",
    "mean_gross_exposure",
    "mean_net_exposure",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "mean_long_count",
    "mean_short_count",
    "mean_position_count",
]
WEIGHT_DIAGNOSTIC_COLUMNS = [
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
]
