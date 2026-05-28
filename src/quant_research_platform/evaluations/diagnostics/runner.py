"""Diagnostic runner entry points."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from quant_research_platform.evaluations.diagnostics.inputs import DiagnosticInputBundle
from quant_research_platform.evaluations.diagnostics.metrics import (
    N5_V1_DIAGNOSTIC_METRICS,
    N5_V1_DIAGNOSTIC_METRICS_BY_NAME,
)
from quant_research_platform.evaluations.diagnostics.policies import (
    disabled_policy,
    n5_v1_policies,
)
from quant_research_platform.evaluations.diagnostics.spec import DiagnosticsSpec


class DiagnosticRunner:
    """Produces post-run diagnostic frames without mutating base evaluation output."""

    def __init__(self, spec: DiagnosticsSpec | None = None) -> None:
        self.spec = DiagnosticsSpec.from_request(spec)

    def run_disabled(self, combinations: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if "combination_id" not in combinations.columns:
            raise ValueError("combinations must include combination_id.")
        policy = disabled_policy()
        policies = pd.DataFrame([asdict(policy)])
        summary = pd.DataFrame(
            [
                {
                    "combination_id": str(combination_id),
                    "policy_id": policy.policy_id,
                    "policy_family": policy.policy_family,
                    "decision_usable": False,
                    "materialized_views": "",
                    "missing_inputs": "",
                    "diagnostic_status": policy.diagnostic_status,
                    "status_reason": policy.status_reason,
                    "warning": policy.warning,
                }
                for combination_id in combinations["combination_id"].tolist()
            ]
        )
        metrics = pd.DataFrame(
            columns=[
                "combination_id",
                "policy_id",
                "policy_family",
                "metric_name",
                "metric_value",
                "unit",
                "decision_usable",
                "required_inputs_met",
                "diagnostic_status",
                "status_reason",
                "warning",
            ]
        )
        return {
            "diagnostic_policies": policies,
            "diagnostic_summary": summary,
            "diagnostic_metrics": metrics,
        }

    def run(self, bundle: DiagnosticInputBundle) -> dict[str, pd.DataFrame]:
        combinations = bundle.base_tables.get("combinations", pd.DataFrame())
        if not self.spec.enabled:
            return self.run_disabled(combinations)
        return self._run_enabled(bundle, combinations)

    def _run_enabled(
        self,
        bundle: DiagnosticInputBundle,
        combinations: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if "combination_id" not in combinations.columns:
            raise ValueError("combinations must include combination_id.")
        policies = n5_v1_policies(requested_views=self.spec.materialized_views)
        policy_frame = pd.DataFrame([asdict(policy) for policy in policies])
        metric_frame = _diagnostic_metrics(bundle, combinations)
        summary_frame = _diagnostic_summary(
            combinations,
            policies=policies,
            requested_views=self.spec.materialized_views,
            metrics=metric_frame,
        )
        frames = {
            "diagnostic_policies": policy_frame,
            "diagnostic_summary": summary_frame,
            "diagnostic_metrics": metric_frame,
        }
        conditional = _materialized_views(
            bundle,
            combinations=combinations,
            requested_views=self.spec.materialized_views,
        )
        frames.update(conditional)
        return frames


def _diagnostic_summary(
    combinations: pd.DataFrame,
    *,
    policies,
    requested_views: tuple[str, ...],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    view_text = ",".join(requested_views)
    rows = []
    for combination_id in combinations["combination_id"].tolist():
        for policy in policies:
            policy_metrics = _policy_metrics(metrics, combination_id, policy.policy_id)
            status = _summary_status(policy_metrics)
            missing_inputs = _missing_inputs(policy_metrics)
            rows.append(
                {
                    "combination_id": str(combination_id),
                    "policy_id": policy.policy_id,
                    "policy_family": policy.policy_family,
                    "decision_usable": False,
                    "materialized_views": view_text,
                    "missing_inputs": ",".join(missing_inputs),
                    "diagnostic_status": status,
                    "status_reason": _summary_reason(status, missing_inputs),
                    "warning": _summary_warning(status, missing_inputs),
                }
            )
    return pd.DataFrame.from_records(rows)


def _diagnostic_metrics(
    bundle: DiagnosticInputBundle,
    combinations: pd.DataFrame,
) -> pd.DataFrame:
    returns = bundle.base_tables.get("returns", pd.DataFrame())
    rows: list[dict[str, object]] = []
    for combination_id in combinations["combination_id"].astype(str).tolist():
        combo_returns = (
            returns[returns["combination_id"].astype(str).eq(combination_id)]
            if "combination_id" in returns.columns
            else pd.DataFrame()
        )
        rows.extend(_risk_rows(combination_id, combo_returns))
        rows.extend(_cost_rows(combination_id, combo_returns))
        rows.extend(_capacity_rows(combination_id, combo_returns, bundle))
        rows.extend(_size_rows(combination_id, bundle))
        rows.extend(_regime_rows(combination_id, combo_returns, bundle))
        rows.extend(_a_share_assumption_rows(combination_id, bundle))
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "combination_id",
            "policy_id",
            "policy_family",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            "required_inputs_met",
            "diagnostic_status",
            "status_reason",
            "warning",
        ],
    )


def _risk_rows(combination_id: str, returns: pd.DataFrame) -> list[dict[str, object]]:
    net = _numeric_series(returns, "net_return")
    if net.empty:
        return [
            _metric_row(
                combination_id,
                policy_id="risk_path_proxy_v1",
                metric_name=metric,
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="returns.net_return is missing or empty",
            )
            for metric in (
                "realized_volatility_proxy",
                "max_drawdown_proxy",
                "tail_loss_5pct_proxy",
                "expected_shortfall_5pct_proxy",
            )
        ]
    wealth = (1.0 + net).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    tail_cutoff = net.quantile(0.05)
    tail = net[net.le(tail_cutoff)]
    return [
        _metric_row(
            combination_id,
            policy_id="risk_path_proxy_v1",
            metric_name="realized_volatility_proxy",
            value=float(net.std(ddof=0)),
        ),
        _metric_row(
            combination_id,
            policy_id="risk_path_proxy_v1",
            metric_name="max_drawdown_proxy",
            value=float(drawdown.min()),
        ),
        _metric_row(
            combination_id,
            policy_id="risk_path_proxy_v1",
            metric_name="tail_loss_5pct_proxy",
            value=float(tail_cutoff),
        ),
        _metric_row(
            combination_id,
            policy_id="risk_path_proxy_v1",
            metric_name="expected_shortfall_5pct_proxy",
            value=float(tail.mean()) if not tail.empty else np.nan,
        ),
    ]


def _cost_rows(combination_id: str, returns: pd.DataFrame) -> list[dict[str, object]]:
    costs = _numeric_series(returns, "cost")
    gross = _numeric_series(returns, "gross_portfolio_return")
    turnover = _numeric_series(returns, "turnover")
    sensitivity = _flat_bps_sensitivity_values(returns)
    sensitivity_values = [
        value
        for value in sensitivity.values()
        if isinstance(value, float) and value == value
    ]
    if costs.empty:
        return [
            _metric_row(
                combination_id,
                policy_id="cost_proxy_v1",
                metric_name="mean_modeled_cost",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="returns.cost is missing or empty",
                warning="unmodeled costs are not inferred as zero",
            ),
            _metric_row(
                combination_id,
                policy_id="cost_proxy_v1",
                metric_name="cost_to_gross_return_ratio",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="returns.cost is missing or empty",
                warning="unmodeled costs are not inferred as zero",
            ),
            _metric_row(
                combination_id,
                policy_id="cost_proxy_v1",
                metric_name="flat_bps_cost_sensitivity_span_proxy",
                value=(
                    float(max(sensitivity_values) - min(sensitivity_values))
                    if sensitivity_values
                    else np.nan
                ),
                required_inputs_met=bool(sensitivity_values) and not turnover.empty,
                status=(
                    "ok"
                    if sensitivity_values and not turnover.empty
                    else "unavailable"
                ),
                reason=(
                    "flat bps grid over gross return and turnover"
                    if sensitivity_values and not turnover.empty
                    else "returns.gross_portfolio_return or returns.turnover is missing"
                ),
                warning="flat bps proxy only; no slippage, fill, or impact model",
            ),
        ]
    denominator = gross.abs().sum() if not gross.empty else 0.0
    return [
        _metric_row(
            combination_id,
            policy_id="cost_proxy_v1",
            metric_name="mean_modeled_cost",
            value=float(costs.mean()),
        ),
        _metric_row(
            combination_id,
            policy_id="cost_proxy_v1",
            metric_name="cost_to_gross_return_ratio",
            value=float(costs.abs().sum() / denominator) if denominator else np.nan,
            required_inputs_met=bool(denominator),
            status="ok" if denominator else "unavailable",
            reason="" if denominator else "gross return denominator is unavailable",
        ),
        _metric_row(
            combination_id,
            policy_id="cost_proxy_v1",
            metric_name="flat_bps_cost_sensitivity_span_proxy",
            value=(
                float(max(sensitivity_values) - min(sensitivity_values))
                if sensitivity_values
                else np.nan
            ),
            required_inputs_met=bool(sensitivity_values) and not turnover.empty,
            status=(
                "ok"
                if sensitivity_values and not turnover.empty
                else "unavailable"
            ),
            reason=(
                "flat bps grid over gross return and turnover"
                if sensitivity_values and not turnover.empty
                else (
                    "returns.gross_portfolio_return or returns.turnover is missing"
                )
            ),
            warning="flat bps proxy only; no slippage, fill, or impact model",
        ),
    ]


def _capacity_rows(
    combination_id: str,
    returns: pd.DataFrame,
    bundle: DiagnosticInputBundle,
) -> list[dict[str, object]]:
    turnover = _numeric_series(returns, "turnover")
    rows: list[dict[str, object]] = []
    if turnover.empty:
        rows.append(
            _metric_row(
                combination_id,
                policy_id="capacity_proxy_v1",
                metric_name="turnover_proxy",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="returns.turnover is missing or empty",
                warning="capacity proxy only; no real capacity claim",
            )
        )
    else:
        rows.append(
            _metric_row(
                combination_id,
                policy_id="capacity_proxy_v1",
                metric_name="turnover_proxy",
                value=float(turnover.mean()),
                status="partial",
                reason="ADV/liquidity inputs absent; turnover proxy only",
                warning="descriptive proxy only; no real capacity claim",
            )
        )
    context = _optional_context(bundle)
    liquidity = _liquidity_context_columns(context)
    if not liquidity:
        rows.extend(
            [
                _metric_row(
                    combination_id,
                    policy_id="capacity_proxy_v1",
                    metric_name="liquidity_context_coverage_proxy",
                    value=np.nan,
                    required_inputs_met=False,
                    status="unavailable",
                    reason=(
                        "optional liquidity context unavailable; "
                        "weights or positions were not materialized for diagnostics"
                    ),
                    warning="aggregate descriptive proxy only; no real capacity claim",
                ),
                _metric_row(
                    combination_id,
                    policy_id="capacity_proxy_v1",
                    metric_name="median_amount_proxy",
                    value=np.nan,
                    required_inputs_met=False,
                    status="unavailable",
                    reason="optional_context.amount is missing",
                    warning="aggregate descriptive proxy only; no real capacity claim",
                ),
                _metric_row(
                    combination_id,
                    policy_id="capacity_proxy_v1",
                    metric_name="median_turnover_liquidity_proxy",
                    value=np.nan,
                    required_inputs_met=False,
                    status="unavailable",
                    reason="optional_context.turnover is missing",
                    warning="aggregate descriptive proxy only; no real capacity claim",
                ),
            ]
        )
        return rows
    rows.append(
        _metric_row(
            combination_id,
            policy_id="capacity_proxy_v1",
            metric_name="liquidity_context_coverage_proxy",
            value=_coverage_rate(context, liquidity),
            status="partial",
            reason=(
                "optional context aggregate only; weights or positions were not "
                "materialized for diagnostics"
            ),
            warning="aggregate descriptive proxy only; no real capacity claim",
        )
    )
    rows.append(
        _context_metric_row(
            context,
            combination_id=combination_id,
            policy_id="capacity_proxy_v1",
            metric_name="median_amount_proxy",
            column="amount",
            status="partial",
            missing_status="unavailable",
            warning="aggregate descriptive proxy only; no real capacity claim",
        )
    )
    rows.append(
        _context_metric_row(
            context,
            combination_id=combination_id,
            policy_id="capacity_proxy_v1",
            metric_name="median_turnover_liquidity_proxy",
            column="turnover",
            status="partial",
            missing_status="unavailable",
            warning="aggregate descriptive proxy only; no real capacity claim",
        )
    )
    return rows


def _size_rows(
    combination_id: str,
    bundle: DiagnosticInputBundle,
) -> list[dict[str, object]]:
    context = _optional_context(bundle)
    market_cap = _market_cap_context_columns(context)
    if not market_cap:
        return [
            _metric_row(
                combination_id,
                policy_id="size_proxy_v1",
                metric_name="market_cap_context_coverage_proxy",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="optional market-cap context unavailable",
                warning="descriptive proxy only; not a Barra or formal exposure model",
            ),
            _metric_row(
                combination_id,
                policy_id="size_proxy_v1",
                metric_name="median_total_market_cap_proxy",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="optional_context.total_market_cap is missing",
                warning="descriptive proxy only; not a Barra or formal exposure model",
            ),
            _metric_row(
                combination_id,
                policy_id="size_proxy_v1",
                metric_name="median_float_market_cap_proxy",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="optional_context.float_market_cap is missing",
                warning="descriptive proxy only; not a Barra or formal exposure model",
            ),
        ]
    return [
        _metric_row(
            combination_id,
            policy_id="size_proxy_v1",
            metric_name="market_cap_context_coverage_proxy",
            value=_coverage_rate(context, market_cap),
            status="partial",
            reason="optional context aggregate only; no formal exposure model",
            warning="descriptive proxy only; not a Barra or formal exposure model",
        ),
        _context_metric_row(
            context,
            combination_id=combination_id,
            policy_id="size_proxy_v1",
            metric_name="median_total_market_cap_proxy",
            column="total_market_cap",
            status="partial",
            missing_status="unavailable",
            warning="descriptive proxy only; not a Barra or formal exposure model",
        ),
        _context_metric_row(
            context,
            combination_id=combination_id,
            policy_id="size_proxy_v1",
            metric_name="median_float_market_cap_proxy",
            column="float_market_cap",
            status="partial",
            missing_status="unavailable",
            warning="descriptive proxy only; not a Barra or formal exposure model",
        ),
    ]


def _regime_rows(
    combination_id: str,
    returns: pd.DataFrame,
    bundle: DiagnosticInputBundle,
) -> list[dict[str, object]]:
    del bundle
    return [
        _metric_row(
            combination_id,
            policy_id="regime_proxy_v1",
            metric_name="scenario_observation_count",
            value=float(len(returns)),
            unit="count",
            status="partial",
            reason="no PIT regime or benchmark input supplied",
            warning="ex-post descriptive row; not decision evidence",
        )
    ]


def _a_share_assumption_rows(
    combination_id: str,
    bundle: DiagnosticInputBundle,
) -> list[dict[str, object]]:
    assumptions = bundle.metadata.get("a_share_assumptions")
    if not isinstance(assumptions, dict):
        return [
            _metric_row(
                combination_id,
                policy_id="a_share_assumption_v1",
                metric_name="t_plus_one_assumption_recorded",
                value=np.nan,
                required_inputs_met=False,
                status="unavailable",
                reason="a_share_assumptions metadata unavailable or not applicable",
                warning="A-share assumptions are recorded only when supplied",
            )
        ]
    return [
        _metric_row(
            combination_id,
            policy_id="a_share_assumption_v1",
            metric_name="t_plus_one_assumption_recorded",
            value=1.0 if assumptions.get("t_plus_one") else 0.0,
            status="ok",
            reason="assumption recorded only; not simulated",
            warning="does not simulate orders, paper trading, or live trading",
        )
    ]


def _materialized_views(
    bundle: DiagnosticInputBundle,
    *,
    combinations: pd.DataFrame,
    requested_views: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    requested = set(requested_views)
    if "portfolio_assumptions" in requested:
        views["portfolio_assumptions"] = _portfolio_assumptions(bundle)
    if "weights" in requested:
        views["weights"] = _weights(bundle)
    if "tradability_diagnostics" in requested:
        views["tradability_diagnostics"] = _tradability_diagnostics(bundle)
    if "tradability_summary" in requested:
        views["tradability_summary"] = _tradability_summary(bundle, combinations)
    if "scenario_metrics" in requested:
        views["scenario_metrics"] = _scenario_metrics(bundle, combinations)
    if "portfolio_risk_summary" in requested:
        views["portfolio_risk_summary"] = _portfolio_risk_summary(bundle, combinations)
    if "exposure_capacity_summary" in requested:
        views["exposure_capacity_summary"] = _exposure_capacity_summary(
            bundle,
            combinations,
        )
    return views


def _portfolio_assumptions(bundle: DiagnosticInputBundle) -> pd.DataFrame:
    assumptions = bundle.metadata.get("a_share_assumptions")
    if not isinstance(assumptions, dict) or not assumptions:
        assumptions = {"status": "not_applicable"}
    records = [
        {
            "assumption_id": str(name),
            "assumption_family": "a_share" if name != "status" else "generic",
            "assumption_name": str(name),
            "assumption_value": str(value),
            "unit": "text",
            "diagnostic_status": "ok" if name != "status" else "partial",
            "status_reason": "assumption recorded only; no simulation",
            "warning": "descriptive metadata only",
        }
        for name, value in sorted(assumptions.items())
    ]
    return pd.DataFrame.from_records(records)


def _weights(bundle: DiagnosticInputBundle) -> pd.DataFrame:
    positions = bundle.base_tables.get("positions", pd.DataFrame()).copy()
    if positions.empty:
        return pd.DataFrame(
            columns=["combination_id", "timestamp", "asset", "target_weight"]
        )
    frame = positions.rename(columns={"position": "target_weight"})
    return frame[
        [
            column
            for column in ("combination_id", "timestamp", "asset", "target_weight")
            if column in frame.columns
        ]
    ].copy()


def _tradability_diagnostics(bundle: DiagnosticInputBundle) -> pd.DataFrame:
    diagnostics = bundle.base_tables.get("weight_diagnostics", pd.DataFrame()).copy()
    if diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "combination_id",
                "timestamp",
                "asset",
                "can_enter",
                "can_exit",
                "reason",
                "diagnostic_status",
                "status_reason",
                "warning",
            ]
        )
    records = []
    for _, row in diagnostics.iterrows():
        blocked = bool(row.get("entry_blocked_count", 0)) or bool(
            row.get("exit_blocked_count", 0)
        )
        records.append(
            {
                "combination_id": row.get("combination_id"),
                "timestamp": row.get("timestamp"),
                "asset": "__aggregate__",
                "can_enter": not bool(row.get("entry_blocked_count", 0)),
                "can_exit": not bool(row.get("exit_blocked_count", 0)),
                "reason": "aggregate_blocked_counts" if blocked else "aggregate_ok",
                "diagnostic_status": "partial",
                "status_reason": "aggregate tradability diagnostics only",
                "warning": "not row-level private data",
            }
        )
    return pd.DataFrame.from_records(records)


def _tradability_summary(
    bundle: DiagnosticInputBundle,
    combinations: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = bundle.base_tables.get("weight_diagnostics", pd.DataFrame())
    records = []
    for combination_id in combinations["combination_id"].astype(str):
        frame = (
            diagnostics[diagnostics["combination_id"].astype(str).eq(combination_id)]
            if "combination_id" in diagnostics.columns
            else pd.DataFrame()
        )
        eligible = _numeric_series(frame, "eligible_count").sum()
        blocked = (
            _numeric_series(frame, "entry_blocked_count").sum()
            + _numeric_series(
                frame,
                "exit_blocked_count",
            ).sum()
        )
        records.append(
            {
                "combination_id": combination_id,
                "eligible_count": int(eligible),
                "blocked_count": int(blocked),
                "blocked_rate": float(blocked / eligible) if eligible else np.nan,
                "diagnostic_status": "ok" if eligible else "unavailable",
                "status_reason": "" if eligible else "tradability counts unavailable",
                "warning": "aggregate descriptive proxy only",
            }
        )
    return pd.DataFrame.from_records(records)


def _scenario_metrics(
    bundle: DiagnosticInputBundle,
    combinations: pd.DataFrame,
) -> pd.DataFrame:
    metrics = _diagnostic_metrics(bundle, combinations)
    scenario_rows = metrics[metrics["policy_family"].eq("regime_proxy")].copy()
    scenario_rows["scenario_id"] = "base_ex_post"
    scenario_rows = scenario_rows[
        [
            "combination_id",
            "scenario_id",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            "diagnostic_status",
            "status_reason",
            "warning",
        ]
    ]
    cost_rows = _flat_bps_scenario_rows(bundle, combinations)
    if cost_rows.empty:
        return scenario_rows
    return pd.concat([scenario_rows, cost_rows], ignore_index=True)


def _portfolio_risk_summary(
    bundle: DiagnosticInputBundle,
    combinations: pd.DataFrame,
) -> pd.DataFrame:
    metrics = _diagnostic_metrics(bundle, combinations)
    risk = metrics[metrics["policy_family"].eq("risk_path_proxy")].copy()
    risk["risk_family"] = "path_proxy"
    return risk[
        [
            "combination_id",
            "risk_family",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            "diagnostic_status",
            "status_reason",
            "warning",
        ]
    ]


def _exposure_capacity_summary(
    bundle: DiagnosticInputBundle,
    combinations: pd.DataFrame,
) -> pd.DataFrame:
    metrics = _diagnostic_metrics(bundle, combinations)
    exposure = metrics[
        metrics["policy_family"].isin(("capacity_proxy", "size_proxy"))
    ].copy()
    exposure["exposure_family"] = exposure["metric_name"].map(_exposure_family)
    return exposure[
        [
            "combination_id",
            "exposure_family",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            "diagnostic_status",
            "status_reason",
            "warning",
        ]
    ]


FLAT_BPS_COST_GRID = (0.0, 5.0, 10.0, 25.0)
LIQUIDITY_CONTEXT_EXACT_COLUMNS = ("amount", "turnover")
LIQUIDITY_CONTEXT_PREFIXES = (
    "avg_amount_",
    "turnover_",
    "amihud_",
    "amplitude_",
)
MARKET_CAP_CONTEXT_COLUMNS = ("total_market_cap", "float_market_cap")


def _policy_metrics(
    metrics: pd.DataFrame,
    combination_id: object,
    policy_id: str,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return metrics[
        metrics["combination_id"].astype(str).eq(str(combination_id))
        & metrics["policy_id"].astype(str).eq(policy_id)
    ]


def _summary_status(policy_metrics: pd.DataFrame) -> str:
    if policy_metrics.empty:
        return "ok"
    statuses = set(policy_metrics["diagnostic_status"].astype(str))
    if statuses == {"unavailable"}:
        return "unavailable"
    if statuses.intersection({"partial", "unavailable", "skipped"}):
        return "partial"
    return "ok"


def _missing_inputs(policy_metrics: pd.DataFrame) -> tuple[str, ...]:
    if policy_metrics.empty or "required_inputs_met" not in policy_metrics.columns:
        return ()
    missing: list[str] = []
    failed = policy_metrics[~policy_metrics["required_inputs_met"].astype(bool)]
    for metric_name in failed["metric_name"].astype(str):
        spec = N5_V1_DIAGNOSTIC_METRICS_BY_NAME.get(metric_name)
        if spec is None:
            continue
        missing.extend(spec.required_inputs)
    return tuple(dict.fromkeys(missing))


def _summary_reason(status: str, missing_inputs: tuple[str, ...]) -> str:
    if status == "ok":
        return (
            "post-run descriptive proxy diagnostics; not an alpha proof, "
            "validated risk model, capacity claim, or trading advice"
        )
    if missing_inputs:
        return "missing or partial diagnostic inputs: " + ",".join(missing_inputs)
    return "one or more descriptive proxy diagnostics are partial or unavailable"


def _summary_warning(status: str, missing_inputs: tuple[str, ...]) -> str:
    if status == "ok" and not missing_inputs:
        return ""
    return "descriptive proxy only; not decision evidence or trading advice"


def _optional_context(bundle: DiagnosticInputBundle) -> pd.DataFrame:
    context = bundle.base_tables.get("optional_context", pd.DataFrame())
    return context if isinstance(context, pd.DataFrame) else pd.DataFrame()


def _liquidity_context_columns(context: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in context.columns
        if column in LIQUIDITY_CONTEXT_EXACT_COLUMNS
        or column.startswith(LIQUIDITY_CONTEXT_PREFIXES)
    )


def _market_cap_context_columns(context: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in MARKET_CAP_CONTEXT_COLUMNS if column in context)


def _coverage_rate(context: pd.DataFrame, columns: tuple[str, ...]) -> float:
    if context.empty or not columns:
        return float("nan")
    values = context.loc[:, list(columns)]
    denominator = values.shape[0] * values.shape[1]
    if denominator == 0:
        return float("nan")
    return float(values.notna().sum().sum() / denominator)


def _context_metric_row(
    context: pd.DataFrame,
    *,
    combination_id: str,
    policy_id: str,
    metric_name: str,
    column: str,
    status: str,
    missing_status: str,
    warning: str,
) -> dict[str, object]:
    series = _numeric_series(context, column)
    if series.empty:
        return _metric_row(
            combination_id,
            policy_id=policy_id,
            metric_name=metric_name,
            value=np.nan,
            required_inputs_met=False,
            status=missing_status,
            reason=f"optional_context.{column} is missing or empty",
            warning=warning,
        )
    return _metric_row(
        combination_id,
        policy_id=policy_id,
        metric_name=metric_name,
        value=float(series.median()),
        status=status,
        reason="optional context aggregate only",
        warning=warning,
    )


def _flat_bps_sensitivity_values(returns: pd.DataFrame) -> dict[float, float]:
    gross = _numeric_series(returns, "gross_portfolio_return")
    turnover = _numeric_series(returns, "turnover")
    if gross.empty or turnover.empty:
        return {}
    aligned = pd.concat([gross.rename("gross"), turnover.rename("turnover")], axis=1)
    aligned = aligned.dropna()
    if aligned.empty:
        return {}
    result: dict[float, float] = {}
    for bps in FLAT_BPS_COST_GRID:
        proxy_net = aligned["gross"] - aligned["turnover"].abs() * bps / 10000.0
        result[bps] = float((1.0 + proxy_net).prod() - 1.0)
    return result


def _flat_bps_scenario_rows(
    bundle: DiagnosticInputBundle,
    combinations: pd.DataFrame,
) -> pd.DataFrame:
    returns = bundle.base_tables.get("returns", pd.DataFrame())
    rows: list[dict[str, object]] = []
    for combination_id in combinations["combination_id"].astype(str).tolist():
        combo_returns = (
            returns[returns["combination_id"].astype(str).eq(combination_id)]
            if "combination_id" in returns.columns
            else pd.DataFrame()
        )
        values = _flat_bps_sensitivity_values(combo_returns)
        for bps in FLAT_BPS_COST_GRID:
            value = values.get(bps, np.nan)
            available = isinstance(value, float) and value == value
            rows.append(
                {
                    "combination_id": combination_id,
                    "scenario_id": f"flat_cost_bps_{_format_bps(bps)}",
                    "metric_name": (
                        "final_cumulative_net_return_after_flat_cost_proxy"
                    ),
                    "metric_value": value,
                    "unit": "ratio",
                    "decision_usable": False,
                    "diagnostic_status": "ok" if available else "unavailable",
                    "status_reason": (
                        "flat bps grid over gross return and turnover"
                        if available
                        else (
                            "returns.gross_portfolio_return or returns.turnover "
                            "is missing"
                        )
                    ),
                    "warning": (
                        "flat bps proxy only; no slippage, fill, impact model, "
                        "simulation, or trading advice"
                    ),
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "combination_id",
            "scenario_id",
            "metric_name",
            "metric_value",
            "unit",
            "decision_usable",
            "diagnostic_status",
            "status_reason",
            "warning",
        ],
    )


def _format_bps(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _exposure_family(metric_name: object) -> str:
    name = str(metric_name)
    if name.startswith("market_cap") or "market_cap" in name:
        return "size_proxy"
    if name.startswith("median_amount") or "liquidity" in name:
        return "liquidity_proxy"
    return "capacity_proxy"


def _metric_row(
    combination_id: str,
    *,
    policy_id: str,
    metric_name: str,
    value: object,
    unit: str | None = None,
    decision_usable: bool = False,
    required_inputs_met: bool = True,
    status: str = "ok",
    reason: str = "",
    warning: str = "",
) -> dict[str, object]:
    spec = next(
        metric
        for metric in N5_V1_DIAGNOSTIC_METRICS
        if metric.metric_name == metric_name
    )
    return {
        "combination_id": combination_id,
        "policy_id": policy_id,
        "policy_family": spec.policy_family,
        "metric_name": metric_name,
        "metric_value": value,
        "unit": unit or spec.unit,
        "decision_usable": decision_usable,
        "required_inputs_met": required_inputs_met,
        "diagnostic_status": status,
        "status_reason": reason,
        "warning": warning,
    }


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").dropna()


__all__ = ["DiagnosticRunner"]
