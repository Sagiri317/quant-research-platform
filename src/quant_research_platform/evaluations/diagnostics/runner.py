"""Diagnostic runner entry points."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from quant_research_platform.evaluations.diagnostics.inputs import DiagnosticInputBundle
from quant_research_platform.evaluations.diagnostics.metrics import (
    N5_V1_DIAGNOSTIC_METRICS,
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
) -> pd.DataFrame:
    view_text = ",".join(requested_views)
    rows = []
    for combination_id in combinations["combination_id"].tolist():
        for policy in policies:
            rows.append(
                {
                    "combination_id": str(combination_id),
                    "policy_id": policy.policy_id,
                    "policy_family": policy.policy_family,
                    "decision_usable": False,
                    "materialized_views": view_text,
                    "missing_inputs": "",
                    "diagnostic_status": "ok",
                    "status_reason": (
                        "post-run descriptive proxy diagnostics; not an alpha proof, "
                        "validated risk model, capacity claim, or trading advice"
                    ),
                    "warning": "",
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
        rows.extend(_capacity_rows(combination_id, combo_returns))
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
    ]


def _capacity_rows(
    combination_id: str,
    returns: pd.DataFrame,
) -> list[dict[str, object]]:
    turnover = _numeric_series(returns, "turnover")
    if turnover.empty:
        return [
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
        ]
    return [
        _metric_row(
            combination_id,
            policy_id="capacity_proxy_v1",
            metric_name="turnover_proxy",
            value=float(turnover.mean()),
            status="partial",
            reason="ADV/liquidity inputs absent; turnover proxy only",
            warning="descriptive proxy only; no real capacity claim",
        )
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
                status="partial",
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
    return scenario_rows[
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
    capacity = metrics[metrics["policy_family"].eq("capacity_proxy")].copy()
    capacity["exposure_family"] = "capacity_proxy"
    return capacity[
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
