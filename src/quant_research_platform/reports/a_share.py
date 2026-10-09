"""Pure A-share report content projection."""

from __future__ import annotations

from collections.abc import Mapping

from quant_research_platform.contracts.experiments import _model_inference_blocker
from quant_research_platform.contracts.targets import ResearchRunRecord
from quant_research_platform.reports import diagnostic_status, report_status

REPORT_SCHEMA_ID = "qrp-a-share-research-report"

REQUIRED_REPORT_SECTIONS = (
    "Data/Coverage",
    "factor diagnostics",
    "model selection",
    "OOS prediction",
    "portfolio",
    "cost",
    "risk",
    "capacity",
    "attribution",
    "development vs holdout",
    "robustness",
    "readiness/blockers",
    "run observations",
)


def build_a_share_report(
    *, record: ResearchRunRecord, evidence: Mapping[str, object]
) -> dict[str, object]:
    run_payload = record.payload()
    configuration = run_payload["protocol"]["research_configuration"]
    references = evidence.get("references")
    statistics = evidence.get("statistics")
    portfolios = evidence.get("portfolios")
    raw_issues = evidence.get("issues")
    if (
        type(references) is not dict
        or type(statistics) is not dict
        or type(portfolios) is not list
        or type(raw_issues) is not list
    ):
        raise ValueError("a_share_report_evidence_invalid")
    issues = [dict(item) for item in raw_issues if type(item) is dict]
    if len(issues) != len(raw_issues):
        raise ValueError("a_share_report_evidence_invalid")
    portfolio_statistics = evidence["portfolio_statistics"]
    scenario = record.scenario_id
    factor_section = {
        "evaluation": dict(evidence["factor_evaluation"]),
        "inference": dict(evidence["factor_statistics"]),
    }
    portfolio_rows = [dict(item) for item in portfolios if type(item) is dict]
    if len(portfolio_rows) != len(portfolios):
        raise ValueError("a_share_report_evidence_invalid")
    inference_blockers = {
        group: _model_inference_blocker(
            access_window=record.research_window["access"], selection_groups=(group,)
        )
        for group in ("global", "quarterly")
    }
    sections = {
        "Data/Coverage": dict(evidence["coverage"]),
        "factor diagnostics": factor_section,
        "model selection": _identity_section(
            references,
            f"{scenario}_model_selection_refit",
            "model_artifact_identity",
        ),
        "OOS prediction": {
            group: {
                **_identity_section(
                    references,
                    f"{scenario}_{group}_persisted_predictions",
                    "prediction_artifact_identity",
                ),
                "inference_qualification": {
                    "status": "UNDEFINED"
                    if inference_blockers[group] is not None
                    else "ELIGIBLE",
                    "reason": inference_blockers[group],
                },
                "interpretation": (
                    "retrospective_validation_selected_at_development_end"
                    if inference_blockers[group] is not None
                    else "causal_quarterly_validation"
                    if record.research_window["access"] == "development"
                    else "frozen_final_test"
                ),
            }
            for group in ("global", "quarterly")
        },
        "portfolio": {
            "status": "DEFINED" if portfolio_rows else "BLOCKED",
            "development_robustness_groups": [
                group for group, reason in inference_blockers.items() if reason is None
            ]
            if record.research_window["access"] == "development"
            else [],
            "results": portfolio_rows,
        },
        "cost": {
            "status": portfolio_statistics["status"],
            "results": [
                row
                for row in portfolio_statistics.get("performance", [])
                if row["metric_id"] in {"cost", "gross_pnl", "net_pnl"}
            ],
        },
        "risk": {
            "status": portfolio_statistics["status"],
            "exposures": portfolio_statistics.get("exposures", {}),
            "results": [
                row
                for row in portfolio_statistics.get("performance", [])
                if row["metric_id"]
                in {
                    "annualized_volatility",
                    "sharpe",
                    "max_drawdown",
                    "tracking_error",
                    "net_information_ratio",
                    "max_active_drawdown",
                }
            ],
        },
        "capacity": {
            "status": portfolio_statistics["status"],
            "results": portfolio_statistics.get("capacity", []),
            "detail": portfolio_statistics.get("capacity_detail", {}),
        },
        "attribution": {
            "status": portfolio_statistics["status"],
            "results": portfolio_statistics.get("attribution", []),
        },
        "development vs holdout": {
            "status": "DEFINED",
            "requested_access": record.research_window["access"],
            "development_window": run_payload["protocol"].get("development_window"),
            "holdout_window": run_payload["protocol"].get("holdout_window"),
            "frozen_development": run_payload["protocol"].get("frozen_development"),
            "evidence": run_payload["holdout_evidence"],
        },
        "robustness": [
            {"selection_group": group, **row}
            for group, result in statistics.items()
            for row in result.get("robustness", [dict(result)])
        ]
        + portfolio_statistics.get("robustness", [dict(portfolio_statistics)]),
        "readiness/blockers": {
            "source_run_status": record.status.value,
            "evaluation_identity": references.get(f"{scenario}_evaluation"),
            "issues": issues,
        },
    }
    sections["portfolio"]["cash_exposure"] = portfolio_statistics.get(
        "cash_exposure", {}
    )
    sections["portfolio"]["nav_path"] = portfolio_statistics.get("nav_path", {})
    sections["portfolio"]["paired_comparison"] = portfolio_statistics.get(
        "group_comparison", []
    )
    sections["OOS prediction"]["paired_family_rankic"] = statistics.get(
        "quarterly", {}
    ).get("model_group_comparison", [])
    sections["portfolio"]["comparison_interpretation"] = (
        "Frozen final test compares development selection and calibration pipelines; "
        "it does not estimate test-time adaptation. Choosing a winner on this test "
        "does not create independent validation of that winner."
    )
    sections["run observations"] = {
        "run": run_payload["operation_evidence"],
        "model_phases": evidence.get("model_costs", []),
        "accounting": (
            "Shared candidate fits are counted once. Group scoring and final refits "
            "are recorded separately; boundary samples are not full process peaks."
        ),
    }
    for name in ("cost", "risk", "capacity", "attribution"):
        rows = sections[name]["results"]
        if rows:
            sections[name]["status"] = diagnostic_status(rows)
            for row in rows:
                if row["status"] in {"BLOCKED", "FAILED"}:
                    issue = {
                        "section": name,
                        "status": row["status"],
                        "reason": row["reason"],
                    }
                    if issue not in issues:
                        issues.append(issue)
    sections["model selection"]["trial_ledger"] = evidence.get(
        "model_trials", {"status": "UNDEFINED"}
    )
    sections["model selection"]["selections"] = evidence["selections"]
    sections["model selection"]["configurations"] = {
        group: result["model_trials"]
        for group, result in statistics.items()
        if result.get("status") == "DEFINED"
    }
    for group, result in statistics.items():
        if result.get("status") == "DEFINED":
            sections["OOS prediction"][group]["family_comparison"] = result[
                "model_comparison"
            ]
            sections["OOS prediction"][group]["reality_checks"] = result[
                "model_reality_checks"
            ]
    if configuration["stage"] in {"dataset", "evaluation"}:
        for name in ("model selection", "OOS prediction", "robustness"):
            sections[name] = {"status": "UNDEFINED", "reason": "not_requested"}
    if not configuration["portfolios"]:
        for name in ("portfolio", "cost", "risk", "capacity", "attribution"):
            sections[name] = {"status": "UNDEFINED", "reason": "not_requested"}
    sections["development vs holdout"]["stage_selection"] = configuration.get(
        "stage_selection"
    )
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "run_id": record.run_id,
        "scenario_id": scenario,
        "access_window": record.research_window["access"],
        "source_run_status": record.status.value,
        "report_status": report_status(record.status, issues).value,
        "sections": sections,
    }


def _identity_section(
    references: dict[str, object], target_id: str, identity_key: str
) -> dict[str, object]:
    identity = references.get(target_id)
    if identity is None:
        return {"status": "BLOCKED", "reason": "target_unavailable"}
    return {"status": "DEFINED", identity_key: identity}
