"""定义 artifact schema 常量和 table path contract。"""

from __future__ import annotations

from pathlib import Path

ARTIFACT_SCHEMA_VERSION = "0.5"
EVALUATION_SET_ARTIFACT_TYPE = "evaluation_set"

EVALUATION_SET_REQUIRED_TABLES = (
    "combinations",
    "ic",
    "ic_summary",
    "grouped_returns",
    "group_spread_summary",
    "performance_summary",
    "positions",
    "turnover",
    "returns",
    "pnl",
    "cost_breakdown",
    "exposure_summary",
    "weight_diagnostics",
    "neutralization_diagnostics",
    "tradability_diagnostics",
)

EVALUATION_SET_METRIC_TABLES = (
    "ic",
    "ic_summary",
    "grouped_returns",
    "group_spread_summary",
    "performance_summary",
)
EVALUATION_SET_PORTFOLIO_TABLES = (
    "positions",
    "turnover",
    "returns",
    "pnl",
    "cost_breakdown",
    "exposure_summary",
)
EVALUATION_SET_DIAGNOSTIC_TABLES = (
    "weight_diagnostics",
    "neutralization_diagnostics",
    "tradability_diagnostics",
)
OPTIONAL_ARTIFACT_GROUPS = (
    "metrics_by_year",
    "metrics_by_quarter",
    "metrics_by_month",
)
BREAKDOWN_PERIODS = ("year", "quarter", "month")
DISPLAY_METADATA_DOWNLOAD_ITEMS = {
    "report.md",
    "comparison.csv",
    "manifest.json",
    "config.json",
    "summary.json",
}


def build_evaluation_set_artifact_paths(
    *,
    summary: dict[str, object],
    output_dir: str | Path,
) -> dict[str, object]:
    """按 evaluation-set schema 构造 portable artifact path contract。"""

    base = Path(output_dir)
    return {
        "manifest": "manifest.json",
        "config": "config.json",
        "summary": "summary.json",
        "combinations": "combinations.csv",
        "metrics": {
            "ic": _relative(base, base / "metrics" / "ic.csv"),
            "ic_summary": _relative(base, base / "metrics" / "ic_summary.csv"),
            "grouped_returns": _relative(
                base,
                base / "metrics" / "grouped_returns.csv",
            ),
            "group_spread_summary": _relative(
                base,
                base / "metrics" / "group_spread_summary.csv",
            ),
            "performance_summary": _relative(
                base,
                base / "metrics" / "performance_summary.csv",
            ),
        },
        **_breakdown_artifact_sections(base, summary),
        "portfolio": {
            "positions": _relative(base, base / "portfolio" / "positions.csv"),
            "turnover": _relative(base, base / "portfolio" / "turnover.csv"),
            "returns": _relative(base, base / "portfolio" / "returns.csv"),
            "pnl": _relative(base, base / "portfolio" / "pnl.csv"),
            "cost_breakdown": _relative(
                base,
                base / "portfolio" / "cost_breakdown.csv",
            ),
            "exposure_summary": _relative(
                base,
                base / "portfolio" / "exposure_summary.csv",
            ),
        },
        "diagnostics": {
            "weight_diagnostics": _relative(
                base,
                base / "diagnostics" / "weight_diagnostics.csv",
            ),
            "neutralization_diagnostics": _relative(
                base,
                base / "diagnostics" / "neutralization_diagnostics.csv",
            ),
            "tradability_diagnostics": _relative(
                base,
                base / "diagnostics" / "tradability_diagnostics.csv",
            ),
        },
        "report": "report.md",
    }


def _breakdown_artifact_sections(
    base: Path,
    summary: dict[str, object],
) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    breakdown_columns = summary.get("breakdown_columns")
    if not isinstance(breakdown_columns, list):
        return sections
    for period in breakdown_columns:
        if not isinstance(period, str):
            continue
        directory = f"metrics_by_{period}"
        sections[directory] = {
            f"ic_summary_by_{period}": _relative(
                base,
                base / directory / f"ic_summary_by_{period}.csv",
            ),
            f"group_spread_summary_by_{period}": _relative(
                base,
                base / directory / f"group_spread_summary_by_{period}.csv",
            ),
            f"performance_summary_by_{period}": _relative(
                base,
                base / directory / f"performance_summary_by_{period}.csv",
            ),
        }
    return sections


def _relative(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix()


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BREAKDOWN_PERIODS",
    "DISPLAY_METADATA_DOWNLOAD_ITEMS",
    "EVALUATION_SET_ARTIFACT_TYPE",
    "EVALUATION_SET_DIAGNOSTIC_TABLES",
    "EVALUATION_SET_METRIC_TABLES",
    "EVALUATION_SET_PORTFOLIO_TABLES",
    "EVALUATION_SET_REQUIRED_TABLES",
    "OPTIONAL_ARTIFACT_GROUPS",
    "build_evaluation_set_artifact_paths",
]
