"""注册 artifact type contract 的轻量 registry。"""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    BREAKDOWN_PERIODS,
    DISPLAY_METADATA_DOWNLOAD_ITEMS,
    EVALUATION_SET_ARTIFACT_TYPE,
    EVALUATION_SET_DIAGNOSTIC_TABLES,
    EVALUATION_SET_METRIC_TABLES,
    EVALUATION_SET_PORTFOLIO_TABLES,
    EVALUATION_SET_REQUIRED_TABLES,
)


@dataclass(frozen=True)
class ArtifactTypeContract:
    """供 validators 和 docs 复用的稳定 artifact type metadata。"""

    name: str
    version: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tables: tuple[str, ...]
    downloads: tuple[str, ...]
    capabilities: tuple[str, ...]


BREAKDOWN_DOWNLOAD_ITEMS = tuple(
    f"metrics_by_{period}/{table}_by_{period}.csv"
    for period in BREAKDOWN_PERIODS
    for table in (
        "ic_summary",
        "group_spread_summary",
        "performance_summary",
    )
)


EVALUATION_SET_CONTRACT = ArtifactTypeContract(
    name=EVALUATION_SET_ARTIFACT_TYPE,
    version=ARTIFACT_SCHEMA_VERSION,
    inputs=("EvaluationRunPayload",),
    outputs=("manifest.json", "config.json", "summary.json", "report.md"),
    tables=EVALUATION_SET_REQUIRED_TABLES,
    downloads=(
        *tuple(sorted(DISPLAY_METADATA_DOWNLOAD_ITEMS)),
        "combinations.csv",
        *(f"metrics/{table}.csv" for table in EVALUATION_SET_METRIC_TABLES),
        *BREAKDOWN_DOWNLOAD_ITEMS,
        *(f"portfolio/{table}.csv" for table in EVALUATION_SET_PORTFOLIO_TABLES),
        *(f"diagnostics/{table}.csv" for table in EVALUATION_SET_DIAGNOSTIC_TABLES),
    ),
    capabilities=("load", "display", "download", "compare"),
)


def get_artifact_type_contract(name: str) -> ArtifactTypeContract:
    """返回受支持 artifact type 的 contract。"""

    if name == EVALUATION_SET_ARTIFACT_TYPE:
        return EVALUATION_SET_CONTRACT
    raise ValueError(f"unsupported artifact type: {name!r}")


__all__ = [
    "ArtifactTypeContract",
    "EVALUATION_SET_CONTRACT",
    "get_artifact_type_contract",
]
