"""Immutable semantic-artifact instances for non-dataset target nodes."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    validate_business_manifest_envelope,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    require_unaliased_path,
    validate_instance_identity,
)
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_internal_artifact_manifest,
)

SEMANTIC_ARTIFACT_SCHEMA_ID = "qrp-semantic-artifact"
MAX_SEMANTIC_ARTIFACT_INSTANCES = 1024


@dataclass(frozen=True)
class SemanticArtifactComparison:
    status: str


def compare_semantic_artifacts(
    old_root: Path,
    candidate_root: Path,
) -> SemanticArtifactComparison:
    try:
        old_envelope = read_semantic_artifact(old_root)
        candidate_envelope = read_semantic_artifact(candidate_root)
        if old_envelope != candidate_envelope:
            return SemanticArtifactComparison("CONTENT_MISMATCH")
        old_files = _artifact_files(Path(old_root))
        candidate_files = _artifact_files(Path(candidate_root))
        if tuple(old_files) != tuple(candidate_files):
            return SemanticArtifactComparison("LAYOUT_MISMATCH")
        for relative in old_files:
            if (Path(old_root) / relative).read_bytes() != (
                Path(candidate_root) / relative
            ).read_bytes():
                return SemanticArtifactComparison("CONTENT_MISMATCH")
    except (OSError, ValueError):
        return SemanticArtifactComparison("INSPECTION_FAILED")
    return SemanticArtifactComparison("MATCH")


def write_semantic_artifact(
    logical_root: Path,
    envelope: BusinessManifestEnvelope,
    writer: Callable[[Path], None],
) -> Path:
    final_root = Path(logical_root) / "instances" / envelope.instance_identity

    def write(staging: Path) -> None:
        writer(staging)
        write_internal_artifact_manifest(
            staging / "manifest.json",
            {
                "schema_id": SEMANTIC_ARTIFACT_SCHEMA_ID,
                "common_manifest": envelope.payload(),
            },
        )

    atomic_write_directory(final_root, write)
    return final_root


def read_semantic_artifact(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> dict[str, object]:
    root = require_unaliased_path(
        Path(instance_root),
        error="semantic_artifact_root_invalid",
    )
    if not root.is_dir() or root.is_symlink():
        raise ValueError("semantic_artifact_root_invalid")
    payload = read_manifest(root / "manifest.json")
    if set(payload) != {"schema_id", "common_manifest"}:
        raise ValueError("semantic_artifact_manifest_invalid")
    if payload["schema_id"] != SEMANTIC_ARTIFACT_SCHEMA_ID:
        raise ValueError("semantic_artifact_manifest_invalid")
    envelope = validate_business_manifest_envelope(payload["common_manifest"])
    if root.name != envelope["instance_identity"]:
        raise ValueError("semantic_artifact_identity_mismatch")
    if (
        expected_identity is not None
        and envelope["instance_identity"] != expected_identity
    ):
        raise ValueError("semantic_artifact_identity_mismatch")
    return envelope


def discover_semantic_artifact_instances(
    logical_root: Path,
    *,
    max_instances: int = MAX_SEMANTIC_ARTIFACT_INSTANCES,
) -> tuple[Path, ...]:
    if type(max_instances) is not int or max_instances <= 0:
        raise ValueError("semantic_artifact_instance_limit_invalid")
    root = require_unaliased_path(
        Path(logical_root),
        error="semantic_artifact_root_invalid",
    )
    if not root.exists():
        return ()
    instances_root = root / "instances"
    require_unaliased_path(instances_root, error="semantic_artifact_root_invalid")
    if not instances_root.exists():
        return ()
    try:
        before = _directory_identity(instances_root)
        names: list[str] = []
        with os.scandir(instances_root) as entries:
            for entry in entries:
                if len(names) >= max_instances:
                    raise ValueError("semantic_artifact_instance_limit_exceeded")
                names.append(entry.name)
        if _directory_identity(instances_root) != before:
            raise ValueError("semantic_artifact_instance_identity_changed")
    except OSError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc

    result: list[Path] = []
    for name in sorted(names):
        try:
            validate_instance_identity(name)
        except ValueError as exc:
            raise ValueError("semantic_artifact_instance_child_invalid") from exc
        entry = instances_root / name
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError("semantic_artifact_instance_child_invalid")
        read_semantic_artifact(entry, expected_identity=name)
        result.append(entry)
    if _directory_identity(instances_root) != before:
        raise ValueError("semantic_artifact_instance_identity_changed")
    return tuple(result)


def _directory_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError("semantic_artifact_root_invalid")
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def _artifact_files(root: Path) -> tuple[Path, ...]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("semantic_artifact_member_invalid")
        if path.is_file():
            files.append(path.relative_to(root))
        elif not path.is_dir():
            raise ValueError("semantic_artifact_member_invalid")
    return tuple(sorted(files, key=lambda value: value.as_posix()))
