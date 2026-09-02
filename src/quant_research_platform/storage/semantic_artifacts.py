"""Immutable semantic-artifact instances for non-dataset target nodes."""

from __future__ import annotations

import hashlib
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
    DatasetFileIdentity,
    dataset_file_identity_from_stat,
    native_filesystem_path,
    open_filesystem_file,
    require_unaliased_path,
    validate_instance_identity,
)
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_manifest,
)

SEMANTIC_ARTIFACT_SCHEMA_ID = "qrp-semantic-artifact"
MAX_SEMANTIC_ARTIFACT_INSTANCES = 1024
MAX_SEMANTIC_ARTIFACT_MEMBERS = 256
_HASH_BUFFER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SemanticArtifactComparison:
    status: str


def compare_semantic_artifacts(
    old_root: Path,
    candidate_root: Path,
) -> SemanticArtifactComparison:
    try:
        old_envelope, old_members = _read_semantic_record(old_root)
        candidate_envelope, candidate_members = _read_semantic_record(candidate_root)
        if old_envelope != candidate_envelope:
            return SemanticArtifactComparison("CONTENT_MISMATCH")
        old_paths = tuple(member["path"] for member in old_members)
        candidate_paths = tuple(member["path"] for member in candidate_members)
        if old_paths != candidate_paths:
            return SemanticArtifactComparison("LAYOUT_MISMATCH")
        if old_members != candidate_members:
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
        members = _member_inventory(staging)
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": SEMANTIC_ARTIFACT_SCHEMA_ID,
                "common_manifest": envelope.payload(),
                "members": list(members),
            },
        )

    atomic_write_directory(final_root, write)
    return final_root


def read_semantic_artifact(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> dict[str, object]:
    envelope, _members = _read_semantic_record(
        instance_root,
        expected_identity=expected_identity,
    )
    return envelope


def _read_semantic_record(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    root = require_unaliased_path(
        Path(instance_root),
        error="semantic_artifact_root_invalid",
    )
    try:
        _directory_identity(root)
    except ValueError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    payload = read_manifest(root / "manifest.json")
    if set(payload) != {"schema_id", "common_manifest", "members"}:
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
    members = _validate_member_payload(payload["members"])
    if members != _member_inventory(root):
        raise ValueError("semantic_artifact_member_integrity_invalid")
    return envelope, members


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
    try:
        value = os.stat(native_filesystem_path(path), follow_symlinks=False)
    except OSError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError("semantic_artifact_root_invalid")
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def _member_inventory(root: Path) -> tuple[dict[str, object], ...]:
    root = require_unaliased_path(
        Path(root),
        error="semantic_artifact_root_invalid",
    )
    try:
        _directory_identity(root)
    except ValueError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    pending = [(root, "")]
    members: list[dict[str, object]] = []
    while pending:
        directory, prefix = pending.pop()
        directory_identity = _directory_identity(directory)
        try:
            with os.scandir(native_filesystem_path(directory)) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
                for entry in children:
                    relative = f"{prefix}/{entry.name}" if prefix else entry.name
                    if entry.is_symlink():
                        raise ValueError("semantic_artifact_member_invalid")
                    child = directory / entry.name
                    if entry.is_dir(follow_symlinks=False):
                        pending.append((child, relative))
                    elif entry.is_file(follow_symlinks=False):
                        if not prefix and entry.name == "manifest.json":
                            continue
                        members.append(_member_record(relative, child))
                        if len(members) > MAX_SEMANTIC_ARTIFACT_MEMBERS:
                            raise ValueError("semantic_artifact_member_limit_exceeded")
                    else:
                        raise ValueError("semantic_artifact_member_invalid")
            if _directory_identity(directory) != directory_identity:
                raise ValueError("semantic_artifact_member_set_changed")
        except OSError as exc:
            raise ValueError("semantic_artifact_member_invalid") from exc
    return tuple(sorted(members, key=lambda item: str(item["path"])))


def _member_record(relative: str, path: Path) -> dict[str, object]:
    identity = _member_identity(path)
    digest = hashlib.sha256()
    try:
        with open_filesystem_file(path) as handle:
            opened = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            if opened != identity:
                raise ValueError("semantic_artifact_member_identity_changed")
            while chunk := handle.read(_HASH_BUFFER_BYTES):
                digest.update(chunk)
            closed = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
    except OSError as exc:
        raise ValueError("semantic_artifact_member_invalid") from exc
    if closed != identity or _member_identity(path) != identity:
        raise ValueError("semantic_artifact_member_identity_changed")
    return {
        "path": relative,
        "size": identity.size,
        "sha256": digest.hexdigest(),
    }


def _validate_member_payload(value: object) -> tuple[dict[str, object], ...]:
    if type(value) is not list or not value:
        raise ValueError("semantic_artifact_member_manifest_invalid")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not dict or set(item) != {"path", "size", "sha256"}:
            raise ValueError("semantic_artifact_member_manifest_invalid")
        relative = item["path"]
        size = item["size"]
        digest = item["sha256"]
        candidate = Path(relative) if type(relative) is str else None
        if (
            candidate is None
            or not relative
            or candidate.is_absolute()
            or candidate.as_posix() != relative
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or type(size) is not int
            or size < 0
            or type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or relative in seen
        ):
            raise ValueError("semantic_artifact_member_manifest_invalid")
        seen.add(relative)
        result.append({"path": relative, "size": size, "sha256": digest})
    return tuple(sorted(result, key=lambda item: str(item["path"])))


def _member_identity(path: Path) -> DatasetFileIdentity:
    try:
        value = os.stat(native_filesystem_path(path), follow_symlinks=False)
    except OSError as exc:
        raise ValueError("semantic_artifact_member_invalid") from exc
    if not stat.S_ISREG(value.st_mode) or int(value.st_ino) <= 0:
        raise ValueError("semantic_artifact_member_invalid")
    return dataset_file_identity_from_stat(value)
