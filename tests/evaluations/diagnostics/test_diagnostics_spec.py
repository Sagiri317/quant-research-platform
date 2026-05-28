from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.evaluations.diagnostics import (
    DISABLED_DIAGNOSTICS_POLICY_ID,
    DiagnosticRunner,
    DiagnosticsSpec,
)
from quant_research_platform.evaluations.diagnostics.inputs import DiagnosticInputBundle
from quant_research_platform.evaluations.diagnostics.metrics import (
    N5_V1_DIAGNOSTIC_METRICS_BY_NAME,
)


def test_disabled_diagnostics_spec_normalizes_empty_request() -> None:
    spec = DiagnosticsSpec.from_request(None)

    assert spec.enabled is False
    assert spec.policy_ids == ()
    assert spec.materialized_views == ()
    assert spec.to_public_dict() == {
        "enabled": False,
        "policy_ids": [],
        "materialized_views": [],
    }


def test_disabled_diagnostics_runner_emits_required_schema_07_frames() -> None:
    combinations = pd.DataFrame(
        {
            "combination_id": ["combo_a", "combo_b"],
            "feature": ["factor", "factor"],
            "label": ["forward_return", "forward_return"],
            "kernel": ["dataframe", "dataframe"],
            "portfolio_mode": ["long_only", "long_short"],
        }
    )

    frames = DiagnosticRunner(DiagnosticsSpec.disabled()).run_disabled(combinations)

    assert set(frames) == {
        "diagnostic_policies",
        "diagnostic_summary",
        "diagnostic_metrics",
    }
    policies = frames["diagnostic_policies"]
    assert policies["policy_id"].tolist() == [DISABLED_DIAGNOSTICS_POLICY_ID]
    assert policies["diagnostic_status"].tolist() == ["disabled"]

    summary = frames["diagnostic_summary"]
    assert summary["combination_id"].tolist() == ["combo_a", "combo_b"]
    assert summary["policy_id"].tolist() == [
        DISABLED_DIAGNOSTICS_POLICY_ID,
        DISABLED_DIAGNOSTICS_POLICY_ID,
    ]
    assert summary["diagnostic_status"].tolist() == ["disabled", "disabled"]
    assert summary["decision_usable"].tolist() == [False, False]

    metrics = frames["diagnostic_metrics"]
    assert list(metrics.columns) == [
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
    assert metrics.empty


def test_enabled_diagnostics_emit_descriptive_proxy_metrics_and_views() -> None:
    combinations = pd.DataFrame(
        {
            "combination_id": ["combo_a"],
            "feature": ["factor"],
            "label": ["forward_return"],
            "kernel": ["dataframe"],
            "portfolio_mode": ["long_short"],
        }
    )
    returns = pd.DataFrame(
        {
            "combination_id": ["combo_a", "combo_a", "combo_a"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "net_return": [0.01, -0.03, 0.02],
            "gross_portfolio_return": [0.012, -0.028, 0.022],
            "turnover": [0.1, 0.2, 0.3],
            "cost": [0.001, 0.001, 0.001],
        }
    )
    positions = pd.DataFrame(
        {
            "combination_id": ["combo_a"],
            "timestamp": pd.to_datetime(["2026-01-01"]),
            "asset": ["000001"],
            "target_weight": [1.0],
        }
    )
    weight_diagnostics = pd.DataFrame(
        {
            "combination_id": ["combo_a"],
            "timestamp": pd.to_datetime(["2026-01-01"]),
            "eligible_count": [10],
            "entry_blocked_count": [1],
            "exit_blocked_count": [0],
        }
    )

    frames = DiagnosticRunner(
        DiagnosticsSpec.from_request(
            {
                "enabled": True,
                "materialized_views": [
                    "portfolio_assumptions",
                    "weights",
                    "tradability_summary",
                    "scenario_metrics",
                    "portfolio_risk_summary",
                    "exposure_capacity_summary",
                ],
            }
        )
    ).run(
        DiagnosticInputBundle(
            base_tables={
                "combinations": combinations,
                "returns": returns,
                "positions": positions,
                "weight_diagnostics": weight_diagnostics,
            },
            metadata={"a_share_assumptions": {"t_plus_one": True}},
        )
    )

    metrics = frames["diagnostic_metrics"]
    assert {
        "realized_volatility_proxy",
        "max_drawdown_proxy",
        "mean_modeled_cost",
        "turnover_proxy",
        "scenario_observation_count",
        "t_plus_one_assumption_recorded",
    }.issubset(set(metrics["metric_name"]))
    assert frames["portfolio_risk_summary"]["risk_family"].tolist() == [
        "path_proxy",
        "path_proxy",
        "path_proxy",
        "path_proxy",
    ]
    assert (
        frames["exposure_capacity_summary"]["warning"]
        .str.contains("no real capacity claim")
        .any()
    )
    assert frames["scenario_metrics"]["decision_usable"].tolist() == [False]
    assert frames["portfolio_assumptions"]["assumption_name"].tolist() == ["t_plus_one"]


def test_risk_proxy_drawdown_uses_wealth_peak_and_expected_shortfall_formula() -> None:
    combinations = pd.DataFrame(
        {
            "combination_id": ["combo_a"],
            "feature": ["factor"],
            "label": ["forward_return"],
            "kernel": ["dataframe"],
            "portfolio_mode": ["long_only"],
        }
    )
    returns = pd.DataFrame(
        {
            "combination_id": ["combo_a", "combo_a"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "net_return": [1.0, -0.5],
        }
    )

    frames = DiagnosticRunner(DiagnosticsSpec.from_request({"enabled": True})).run(
        DiagnosticInputBundle(
            base_tables={"combinations": combinations, "returns": returns},
            metadata={},
        )
    )

    metrics = frames["diagnostic_metrics"].set_index("metric_name")
    assert metrics.loc["max_drawdown_proxy", "metric_value"] == pytest.approx(-0.5)
    formula = N5_V1_DIAGNOSTIC_METRICS_BY_NAME[
        "expected_shortfall_5pct_proxy"
    ].formula_ref
    assert "where net_return <=" in formula
    assert "mean(returns <=" not in formula


def test_enabled_request_rejects_unknown_materialized_views() -> None:
    with pytest.raises(ValueError, match="unknown diagnostics materialized views"):
        DiagnosticsSpec.from_request(
            {"enabled": True, "materialized_views": ["capacity_curve"]}
        )


def test_enabled_request_rejects_unimplemented_detail_view() -> None:
    with pytest.raises(ValueError, match="diagnostic_view_not_implemented"):
        DiagnosticsSpec.from_request(
            {
                "enabled": True,
                "materialized_views": ["diagnostic_observations"],
            }
        )
