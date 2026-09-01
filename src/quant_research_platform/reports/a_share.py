"""Derived A-share research report projection."""

from __future__ import annotations

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
)


def build_a_share_report(
    *,
    scenario_id: str,
    phase: str,
    data: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_id": "qrp-a-share-research-report",
        "scenario_id": scenario_id,
        "phase": phase,
        "sections": {
            section: data.get(section) for section in REQUIRED_REPORT_SECTIONS
        },
    }
