"""Crypto HFT report content."""

from __future__ import annotations

from quant_research_platform.reports import diagnostic_status, report_status

REPORT_SCHEMA_ID = "qrp-crypto-hft-report"
REQUIRED_REPORT_SECTIONS = (
    "Data/Coverage",
    "factor diagnostics",
    "models and validation",
    "OOS prediction",
    "model comparison",
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
    run_payload = record.payload()
    configuration = run_payload["protocol"]["research_configuration"]
    statistics = evidence["statistics"]

    def statistical_rows(table, predicate=None):
        selected, states, artifacts = [], [], []
        for item in statistics:
            artifacts.append(
                {key: value for key, value in item.items() if key != "tables"}
            )
            states.append({"status": item["status"]})
            if item["status"] in {"BLOCKED", "FAILED"}:
                continue
            rows = item["tables"][table]
            selected.extend(row for row in rows if predicate is None or predicate(row))
        states.extend({"status": row["status"]} for row in selected if "status" in row)
        return {
            "status": diagnostic_status(states),
            "statistics_artifacts": artifacts,
            "results": selected,
        }

    assets = {
        item["candidate_id"]: item["asset_id"] for item in configuration["portfolios"]
    }
    coverage_groups = {}
    for item in statistics:
        if item["status"] != "DEFINED":
            continue
        bounds = tuple(
            (row["start_at"], row["stop_at"])
            for row in item["tables"]["crypto_portfolio_coverage"]
            if row["kind"] == "covered"
        )
        groups = coverage_groups.setdefault(assets[item["candidate_id"]], {})
        groups.setdefault(bounds, []).append(
            {
                "candidate_id": item["candidate_id"],
                "execution_mode": item["execution_mode"],
            }
        )
    coverage_comparison = [
        {
            "asset_id": asset,
            "status": "DEFINED" if len(groups) == 1 else "UNDEFINED",
            "reason": "" if len(groups) == 1 else "different_coverage_not_comparable",
            "scope": "exact_coverage_and_episode_boundaries_only",
            "groups": [
                {
                    "intervals": [
                        {"start_at": start, "stop_at": stop} for start, stop in bounds
                    ],
                    "candidates": candidates,
                }
                for bounds, candidates in groups.items()
            ],
        }
        for asset, groups in coverage_groups.items()
    ]

    evaluations = evidence["evaluations"]
    sections = {
        "Data/Coverage": {
            **evidence["data"],
            "execution_coverage": statistical_rows("crypto_portfolio_coverage"),
        },
        "factor diagnostics": {
            "status": diagnostic_status(evaluations),
            "evaluations": evaluations,
        },
        "models and validation": evidence["models"],
        "OOS prediction": evidence["predictions"],
        "model comparison": {
            "status": diagnostic_status(evidence["model_comparisons"])
            if evidence["model_comparisons"]
            else "UNDEFINED",
            "reason": "" if evidence["model_comparisons"] else "no_comparable_models",
            "sample_policy": "pairwise_exact_keys_finite_predictions_mature_same_label",
            "reference_model_id": run_payload["protocol"]["research_configuration"][
                "comparison_model_id"
            ],
            "evaluations": evidence["model_comparisons"],
            "portfolio_policy": "full_individual_ledgers_unchanged",
        },
        "maker calibration": evidence["calibrations"],
        "portfolio": {
            "status": diagnostic_status(evidence["portfolios"]),
            "candidates": evidence["portfolios"],
            "intervals": statistical_rows("crypto_portfolio_intervals"),
            "coverage_comparison": coverage_comparison,
        },
        "execution": statistical_rows("crypto_execution_diagnostics"),
        "cost sensitivity": {
            **statistical_rows("crypto_cost_sensitivity"),
            "observed_cost_summary": statistical_rows(
                "crypto_portfolio_statistics",
                lambda row: (
                    row["episode_id"] == "all"
                    and row["metric_id"]
                    in {"fees", "funding", "turnover", "fill_events", "fill_quantity"}
                ),
            ),
        },
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
                        "episode_max_drawdown",
                    }
                ),
            ),
            "daily_returns": statistical_rows("crypto_portfolio_daily"),
            "policy": run_payload["protocol"]["portfolio_statistics"],
            "inference_scope": (
                "daily_economic_descriptives_no_spa_stepm_mcs_psr_dsr_pbo_matrix"
            ),
        },
        "development vs holdout": {
            "status": "DEFINED",
            "access_window": record.research_window["access"],
            "frozen_development": run_payload["protocol"].get("frozen_development"),
            "evidence": run_payload["holdout_evidence"],
            "holdout_window": run_payload["protocol"]["holdout_window"],
        },
        "readiness/blockers": {
            "status": "DEFINED"
            if not evidence["issues"]
            else diagnostic_status(evidence["issues"]),
            "build_status": record.status.value,
            "issues": evidence["issues"],
        },
    }
    if configuration["stage"] == "evaluation":
        for name in ("models and validation", "OOS prediction", "model comparison"):
            sections[name] = {"status": "UNDEFINED", "reason": "not_requested"}
    if not evidence["calibrations"]:
        sections["maker calibration"] = {
            "status": "UNDEFINED",
            "reason": "not_requested",
        }
    if not configuration["portfolios"]:
        for name in (
            "portfolio",
            "execution",
            "cost sensitivity",
            "inventory",
            "daily statistics",
        ):
            sections[name] = {"status": "UNDEFINED", "reason": "not_requested"}
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "run_id": record.run_id,
        "scenario_id": record.scenario_id,
        "access_window": record.research_window["access"],
        "source_run_status": record.status.value,
        "report_status": report_status(record.status, evidence["issues"]).value,
        "sections": {
            name: (
                run_payload["operation_evidence"]
                if name == "run observations"
                else sections[name]
            )
            for name in REQUIRED_REPORT_SECTIONS
        },
    }
