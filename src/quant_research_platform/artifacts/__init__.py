"""产物 artifact contract、加载、写入和 inventory 入口。"""

from quant_research_platform.artifacts.artifact_loader import (
    LoadedEvaluationSet,
    load_evaluation_set,
)
from quant_research_platform.artifacts.artifact_schema import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.artifact_writer import (
    ArtifactOutputs,
    build_evaluation_set_manifest,
    write_evaluation_set_artifacts,
)
from quant_research_platform.artifacts.manifest_validation import (
    validate_display_manifest,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactOutputs",
    "LoadedEvaluationSet",
    "build_evaluation_set_manifest",
    "load_evaluation_set",
    "validate_display_manifest",
    "write_evaluation_set_artifacts",
]
