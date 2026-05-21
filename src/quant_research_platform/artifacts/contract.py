"""Artifact schema constants and path contracts."""

from quant_research_platform.artifacts.writer import ARTIFACT_SCHEMA_VERSION

OPTIONAL_ARTIFACT_GROUPS = (
    "metrics_by_year",
    "metrics_by_quarter",
    "metrics_by_month",
)

__all__ = ["ARTIFACT_SCHEMA_VERSION", "OPTIONAL_ARTIFACT_GROUPS"]
