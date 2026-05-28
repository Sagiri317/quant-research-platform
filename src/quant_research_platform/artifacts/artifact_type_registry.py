"""注册 artifact type contract 的轻量 registry。"""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    DISPLAY_METADATA_DOWNLOAD_ITEMS,
    EVALUATION_SET_ARTIFACT_TYPE,
    EVALUATION_SET_REQUIRED_TABLES,
)
from quant_research_platform.artifacts.schema_registry import table_specs


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


EVALUATION_SET_CONTRACT = ArtifactTypeContract(
    name=EVALUATION_SET_ARTIFACT_TYPE,
    version=ARTIFACT_SCHEMA_VERSION,
    inputs=("EvaluationRunPayload",),
    outputs=(
        "manifest.json",
        "summary.json",
        "run_context.json",
    ),
    tables=EVALUATION_SET_REQUIRED_TABLES,
    downloads=(
        *tuple(sorted(DISPLAY_METADATA_DOWNLOAD_ITEMS)),
        *(spec.path for spec in table_specs()),
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
