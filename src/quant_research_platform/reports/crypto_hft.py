"""Crypto HFT report content."""

from __future__ import annotations

from quant_research_platform.reports import diagnostic_status, report_status

REPORT_SCHEMA_ID = "qrp-crypto-hft-report"
REQUIRED_REPORT_SECTIONS = (
    "Data/Coverage",
    "factor diagnostics",
    "models and validation",
    "OOS prediction",
    "maker calibration",
    "portfolio",
    "execution",
    "cost sensitivity",
    "inventory",
    "daily statistics",
    "development vs holdout",
    "readiness/blockers",
    "run observations",
)


def build_crypto_research_report(*, record, evidence):
    statistics = evidence["statistics"]

    def statistical_rows(table, predicate=None):
        if statistics["status"] in {"BLOCKED", "FAILED"}:
            return dict(statistics)
        rows = statistics["tables"][table]
        selected = [row for row in rows if predicate is None or predicate(row)]
        return {
            "status": diagnostic_status(selected)
            if selected and "status" in selected[0]
            else "DEFINED",
            "statistics_identity": statistics["identity"],
            "results": selected,
        }

    evaluations = evidence["evaluations"]
    sections = {
        "Data/Coverage": evidence["data"],
        "factor diagnostics": {
            "status": diagnostic_status(evaluations),
            "evaluations": evaluations,
        },
        "models and validation": evidence["models"],
        "OOS prediction": evidence["predictions"],
        "maker calibration": evidence["calibration"],
        "portfolio": {
            "status": diagnostic_status(evidence["portfolios"]),
            "candidates": evidence["portfolios"],
            "intervals": statistical_rows("crypto_portfolio_intervals"),
        },
        "execution": statistical_rows("crypto_execution_diagnostics"),
        "cost sensitivity": statistical_rows("crypto_cost_sensitivity"),
        "inventory": statistical_rows(
            "crypto_portfolio_statistics", lambda row: "inventory_" in row["metric_id"]
        ),
        "daily statistics": {
            **statistical_rows(
                "crypto_portfolio_statistics",
                lambda row: (
                    row["metric_id"]
                    in {
                        "oos_mean_daily_return",
                        "annualized_volatility",
                        "sharpe",
                        "planned_complete_utc_days",
                        "defined_complete_utc_days",
                        "continuous_max_drawdown",
                    }
                ),
            ),
            "daily_returns": statistical_rows("crypto_portfolio_daily"),
            "policy": dict(record.protocol["portfolio_statistics"]),
            "inference_scope": (
                "daily_economic_descriptives_no_spa_stepm_mcs_psr_dsr_pbo_matrix"
            ),
        },
        "development vs holdout": {
            "status": "DEFINED",
            "access_window": record.research_window["access"],
            "holdout_status": record.protocol["holdout_status"],
            "holdout_window": dict(record.protocol["holdout_window"]),
        },
        "readiness/blockers": {
            "status": "DEFINED"
            if not evidence["issues"]
            else diagnostic_status(evidence["issues"]),
            "build_status": record.status.value,
            "issues": evidence["issues"],
        },
    }
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "run_id": record.run_id,
        "scenario_id": record.scenario_id,
        "access_window": record.research_window["access"],
        "source_run_status": record.status.value,
        "report_status": report_status(record.status, evidence["issues"]).value,
        "sections": {
            name: (
                dict(record.operation_evidence)
                if name == "run observations"
                else sections[name]
            )
            for name in REQUIRED_REPORT_SECTIONS
        },
    }
