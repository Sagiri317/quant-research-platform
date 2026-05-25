"""定义 artifact schema 常量和 table path contract。"""

from __future__ import annotations

from pathlib import Path

ARTIFACT_SCHEMA_VERSION = "0.6"
EVALUATION_SET_ARTIFACT_TYPE = "evaluation_set"

EVALUATION_SET_REQUIRED_TABLES = (
    "combinations",
    "ic",
    "ic_summary",
    "grouped_returns",
    "group_spread_summary",
    "performance_summary",
    "returns",
    "cumulative_returns",
    "weights",
    "tradability_diagnostics",
    "label_availability",
    "factor_coverage",
    "tradability_coverage",
    "evaluation_summary",
)

EVALUATION_SET_METRIC_TABLES = (
    "ic",
    "ic_summary",
    "grouped_returns",
    "group_spread_summary",
    "performance_summary",
)
EVALUATION_SET_PORTFOLIO_TABLES = (
    "returns",
    "cumulative_returns",
    "weights",
)
EVALUATION_SET_DIAGNOSTIC_TABLES = (
    "tradability_diagnostics",
)
EVALUATION_SET_INPUT_QUALITY_TABLES = (
    "label_availability",
    "factor_coverage",
    "tradability_coverage",
)
EVALUATION_SET_SUMMARY_TABLES = ("evaluation_summary",)
OPTIONAL_ARTIFACT_GROUPS = (
    "breakdowns/year",
    "breakdowns/quarter",
    "breakdowns/month",
)
BREAKDOWN_PERIODS = ("year", "quarter", "month")
DISPLAY_METADATA_DOWNLOAD_ITEMS = {
    "report.md",
    "comparison.csv",
    "manifest.json",
    "config.json",
    "lineage.json",
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
        "lineage": "lineage.json",
        "summary": "summary.json",
        "dimensions": {
            "combinations": _relative(
                base,
                base / "dimensions" / "combinations.csv",
            ),
        },
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
            "returns": _relative(base, base / "portfolio" / "returns.csv"),
            "cumulative_returns": _relative(
                base,
                base / "portfolio" / "cumulative_returns.csv",
            ),
            "weights": _relative(base, base / "portfolio" / "weights.parquet"),
        },
        "diagnostics": {
            "tradability_diagnostics": _relative(
                base,
                base / "diagnostics" / "tradability_diagnostics.csv",
            ),
        },
        "input_quality": {
            "label_availability": _relative(
                base,
                base / "input_quality" / "label_availability.csv",
            ),
            "factor_coverage": _relative(
                base,
                base / "input_quality" / "factor_coverage.csv",
            ),
            "tradability_coverage": _relative(
                base,
                base / "input_quality" / "tradability_coverage.csv",
            ),
        },
        "summaries": {
            "evaluation_summary": _relative(
                base,
                base / "summaries" / "evaluation_summary.csv",
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
        directory = f"breakdowns/{period}"
        sections[directory] = {
            f"ic_summary_by_{period}": _relative(
                base,
                base / directory / "ic_summary.csv",
            ),
            f"group_spread_summary_by_{period}": _relative(
                base,
                base / directory / "group_spread_summary.csv",
            ),
            f"performance_summary_by_{period}": _relative(
                base,
                base / directory / "performance_summary.csv",
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
    "EVALUATION_SET_INPUT_QUALITY_TABLES",
    "EVALUATION_SET_METRIC_TABLES",
    "EVALUATION_SET_PORTFOLIO_TABLES",
    "EVALUATION_SET_REQUIRED_TABLES",
    "EVALUATION_SET_SUMMARY_TABLES",
    "OPTIONAL_ARTIFACT_GROUPS",
    "build_evaluation_set_artifact_paths",
]
