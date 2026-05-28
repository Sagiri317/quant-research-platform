"""Artifact schema constants and path contract."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.schema_registry import (
    EVALUATION_SET_ARTIFACT_TYPE,
    EVALUATION_SET_SCHEMA_VERSION,
    optional_metadata_files,
    required_metadata_files,
    table_specs,
)

ARTIFACT_SCHEMA_VERSION = EVALUATION_SET_SCHEMA_VERSION
BREAKDOWN_PERIODS = ("year", "quarter", "month")

EVALUATION_SET_REQUIRED_TABLES = tuple(
    spec.key for spec in table_specs() if spec.materialization == "required"
)
EVALUATION_SET_METRIC_TABLES = tuple(
    spec.key for spec in table_specs() if spec.path.startswith("metrics/")
)
EVALUATION_SET_PORTFOLIO_TABLES = tuple(
    spec.key for spec in table_specs() if spec.path.startswith("portfolio/")
)
EVALUATION_SET_DIAGNOSTIC_TABLES = tuple(
    spec.key for spec in table_specs() if spec.path.startswith("diagnostics/")
)
EVALUATION_SET_INPUT_QUALITY_TABLES = tuple(
    spec.key for spec in table_specs() if spec.path.startswith("input_quality/")
)
EVALUATION_SET_SUMMARY_TABLES: tuple[str, ...] = ()
OPTIONAL_ARTIFACT_GROUPS = tuple(f"breakdowns/{period}" for period in BREAKDOWN_PERIODS)
DISPLAY_METADATA_DOWNLOAD_ITEMS = {
    *required_metadata_files(),
    *optional_metadata_files(),
}


def build_evaluation_set_artifact_paths(
    *,
    summary: dict[str, object],
    output_dir: str | Path,
    include_report: bool = True,
    materialized_tables: set[str] | None = None,
) -> dict[str, object]:
    """Build a schema 0.7 portable artifact path contract."""

    del summary, output_dir
    materialized = set(EVALUATION_SET_REQUIRED_TABLES)
    if materialized_tables is not None:
        materialized.update(materialized_tables)
    artifacts: dict[str, object] = {
        "manifest": "manifest.json",
        "summary": "summary.json",
        "run_context": "run_context.json",
    }
    if include_report:
        artifacts["report"] = "report.md"
    for spec in table_specs():
        if spec.materialization != "required" and spec.key not in materialized:
            continue
        _insert_table_path(artifacts, spec.key, spec.path)
    return artifacts


def _insert_table_path(artifacts: dict[str, object], key: str, path: str) -> None:
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"artifact table path must include a group: {path!r}.")
    group = "/".join(parts[:-1])
    section = artifacts.setdefault(group, {})
    if not isinstance(section, dict):
        raise ValueError(f"artifact group conflict: {group}.")
    section[key] = path


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
