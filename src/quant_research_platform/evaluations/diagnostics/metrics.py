"""Registry-owned diagnostic metric declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticMetricSpec:
    policy_family: str
    metric_name: str
    formula_ref: str
    unit: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()
    timing_requirement: str = "post_run_descriptive"
    missing_behavior: str = "emit_unavailable"
    default_enabled: bool = True
    deferred: bool = False


N5_V1_DIAGNOSTIC_METRICS: tuple[DiagnosticMetricSpec, ...] = (
    DiagnosticMetricSpec(
        policy_family="risk_path_proxy",
        metric_name="realized_volatility_proxy",
        formula_ref="std(portfolio/returns.net_return)",
        unit="ratio",
        required_inputs=("returns.net_return",),
    ),
    DiagnosticMetricSpec(
        policy_family="risk_path_proxy",
        metric_name="max_drawdown_proxy",
        formula_ref=(
            "min(compound_wealth(portfolio/returns.net_return) / "
            "running_peak_wealth - 1)"
        ),
        unit="ratio",
        required_inputs=("returns.net_return",),
    ),
    DiagnosticMetricSpec(
        policy_family="risk_path_proxy",
        metric_name="tail_loss_5pct_proxy",
        formula_ref="quantile(portfolio/returns.net_return, 0.05)",
        unit="ratio",
        required_inputs=("returns.net_return",),
    ),
    DiagnosticMetricSpec(
        policy_family="risk_path_proxy",
        metric_name="expected_shortfall_5pct_proxy",
        formula_ref=(
            "mean(portfolio/returns.net_return where net_return <= "
            "quantile(net_return, 0.05))"
        ),
        unit="ratio",
        required_inputs=("returns.net_return",),
    ),
    DiagnosticMetricSpec(
        policy_family="cost_proxy",
        metric_name="mean_modeled_cost",
        formula_ref="mean(portfolio/returns.cost)",
        unit="ratio",
        required_inputs=("returns.cost",),
    ),
    DiagnosticMetricSpec(
        policy_family="cost_proxy",
        metric_name="cost_to_gross_return_ratio",
        formula_ref="sum(abs(cost)) / sum(abs(gross_portfolio_return))",
        unit="ratio",
        required_inputs=("returns.cost", "returns.gross_portfolio_return"),
    ),
    DiagnosticMetricSpec(
        policy_family="cost_proxy",
        metric_name="flat_bps_cost_sensitivity_span_proxy",
        formula_ref=(
            "range(final_cumulative_return(gross_portfolio_return - "
            "turnover * flat_bps / 10000) for flat_bps in diagnostic grid)"
        ),
        unit="ratio",
        required_inputs=("returns.gross_portfolio_return", "returns.turnover"),
    ),
    DiagnosticMetricSpec(
        policy_family="capacity_proxy",
        metric_name="turnover_proxy",
        formula_ref="mean(portfolio/returns.turnover)",
        unit="ratio",
        required_inputs=("returns.turnover",),
        optional_inputs=("adv", "amount", "float_market_cap", "market_cap"),
        missing_behavior="emit_partial",
    ),
    DiagnosticMetricSpec(
        policy_family="capacity_proxy",
        metric_name="liquidity_context_coverage_proxy",
        formula_ref=(
            "non_null_cells(optional_context liquidity columns) / "
            "total_cells(optional_context liquidity columns)"
        ),
        unit="ratio",
        required_inputs=("optional_context.liquidity_columns",),
        missing_behavior="emit_partial",
    ),
    DiagnosticMetricSpec(
        policy_family="capacity_proxy",
        metric_name="median_amount_proxy",
        formula_ref="median(optional_context.amount)",
        unit="currency",
        required_inputs=("optional_context.amount",),
        missing_behavior="emit_partial",
    ),
    DiagnosticMetricSpec(
        policy_family="capacity_proxy",
        metric_name="median_turnover_liquidity_proxy",
        formula_ref="median(optional_context.turnover)",
        unit="ratio",
        required_inputs=("optional_context.turnover",),
        missing_behavior="emit_partial",
    ),
    DiagnosticMetricSpec(
        policy_family="size_proxy",
        metric_name="market_cap_context_coverage_proxy",
        formula_ref=(
            "non_null_cells(optional_context market-cap columns) / "
            "total_cells(optional_context market-cap columns)"
        ),
        unit="ratio",
        required_inputs=("optional_context.market_cap_columns",),
    ),
    DiagnosticMetricSpec(
        policy_family="size_proxy",
        metric_name="median_total_market_cap_proxy",
        formula_ref="median(optional_context.total_market_cap)",
        unit="currency",
        required_inputs=("optional_context.total_market_cap",),
    ),
    DiagnosticMetricSpec(
        policy_family="size_proxy",
        metric_name="median_float_market_cap_proxy",
        formula_ref="median(optional_context.float_market_cap)",
        unit="currency",
        required_inputs=("optional_context.float_market_cap",),
    ),
    DiagnosticMetricSpec(
        policy_family="regime_proxy",
        metric_name="scenario_observation_count",
        formula_ref="count(portfolio/returns rows by optional regime/segment)",
        unit="count",
        required_inputs=("returns.net_return",),
        optional_inputs=("regime", "market_segment", "benchmark_return"),
        timing_requirement="ex_post_descriptive",
        missing_behavior="emit_partial",
    ),
    DiagnosticMetricSpec(
        policy_family="a_share_assumption",
        metric_name="t_plus_one_assumption_recorded",
        formula_ref="metadata/a_share_assumptions",
        unit="flag",
        required_inputs=("run_context.config.a_share_assumptions",),
        missing_behavior="emit_partial",
    ),
)

N5_V1_DIAGNOSTIC_METRICS_BY_NAME = {
    spec.metric_name: spec for spec in N5_V1_DIAGNOSTIC_METRICS
}


__all__ = [
    "DiagnosticMetricSpec",
    "N5_V1_DIAGNOSTIC_METRICS",
    "N5_V1_DIAGNOSTIC_METRICS_BY_NAME",
]
