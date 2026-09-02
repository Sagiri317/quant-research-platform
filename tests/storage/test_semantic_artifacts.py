from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.storage.semantic_artifacts import (
    discover_semantic_artifact_instances,
    read_semantic_artifact,
    write_semantic_artifact,
)


def _envelope() -> object:
    return build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-03-31",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )


def _write_artifact(tmp_path: Path) -> Path:
    def writer(staging: Path) -> None:
        (staging / "model.json").write_bytes(b"model")
        (staging / "model.pkl").write_bytes(b"predictor")

    return write_semantic_artifact(tmp_path / "model", _envelope(), writer)


def test_semantic_artifact_member_mutation_is_rejected_by_read_and_discovery(
    tmp_path: Path,
) -> None:
    root = _write_artifact(tmp_path)
    (root / "model.pkl").write_bytes(b"changed")

    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        discover_semantic_artifact_instances(tmp_path / "model")


def test_semantic_artifact_extra_member_is_rejected(tmp_path: Path) -> None:
    root = _write_artifact(tmp_path)
    (root / "extra.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)
