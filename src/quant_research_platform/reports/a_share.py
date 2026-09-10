"""Pure A-share report content projection."""

from __future__ import annotations

from collections.abc import Mapping

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
    factor_section = (
        {
            "status": "DEFINED",
            "statistics_identity": statistics["identity"],
            "row_count": statistics["factor_row_count"],
            "defined_rows": statistics["factor_defined_rows"],
            "undefined_rows": statistics["factor_undefined_rows"],
            "results": statistics["factor_diagnostics"],
        }
        if statistics.get("status") == "DEFINED"
        else dict(statistics)
    )
    portfolio_rows = [dict(item) for item in portfolios if type(item) is dict]
    if len(portfolio_rows) != len(portfolios):
        raise ValueError("a_share_report_evidence_invalid")
    sections = {
        "Data/Coverage": dict(evidence["coverage"]),
        "factor diagnostics": factor_section,
        "model selection": _identity_section(
            references,
            f"{scenario}_model_selection_refit",
            "model_artifact_identity",
        ),
        "OOS prediction": _identity_section(
            references,
            f"{scenario}_persisted_predictions",
            "prediction_artifact_identity",
        ),
        "portfolio": {
            "status": "DEFINED" if portfolio_rows else "BLOCKED",
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
        "robustness": statistics.get("robustness", [dict(statistics)])
        + portfolio_statistics.get("robustness", [dict(portfolio_statistics)]),
        "readiness/blockers": {
            "source_run_status": record.status.value,
            "evaluation_identity": references.get(f"{scenario}_evaluation"),
            "issues": issues,
        },
    }
    sections["portfolio"]["nav_path"] = portfolio_statistics.get("nav_path", {})
    sections["run observations"] = run_payload["operation_evidence"]
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
    sections["model selection"]["quarterly_selections"] = evidence[
        "quarterly_selections"
    ]
    if statistics.get("status") == "DEFINED":
        sections["model selection"]["configurations"] = statistics["model_trials"]
        sections["OOS prediction"]["family_comparison"] = statistics["model_comparison"]
        sections["OOS prediction"]["reality_checks"] = statistics[
            "model_reality_checks"
        ]
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
