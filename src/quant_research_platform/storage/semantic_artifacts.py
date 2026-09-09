"""Immutable semantic-artifact instances for non-dataset target nodes."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import pyarrow as pa

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.experiments import (
    trial_ledger_semantic_payload,
    validate_trial_ledger_reference,
)
from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.experiments import read_trial_ledger_reference
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity_from_stat,
    native_filesystem_path,
    open_filesystem_file,
    require_unaliased_path,
    validate_instance_identity,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_manifest,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    iter_dataset_part_batches,
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
    *,
    trial_layout: StorageLayout | None = None,
) -> SemanticArtifactComparison:
    try:
        identities = {
            root / "manifest.json": _member_identity(root / "manifest.json")
            for root in (old_root, candidate_root)
        }
        old_envelope, old_members = _read_semantic_record(old_root)
        candidate_envelope, candidate_members = _read_semantic_record(candidate_root)
        if old_envelope != candidate_envelope:
            return SemanticArtifactComparison("CONTENT_MISMATCH")
        old_paths = tuple(member["path"] for member in old_members)
        candidate_paths = tuple(member["path"] for member in candidate_members)
        if old_paths != candidate_paths:
            return SemanticArtifactComparison("LAYOUT_MISMATCH")
        old_trials = read_semantic_artifact_operational_metadata(old_root).get(
            "model_trial_ledgers", []
        )
        candidate_trials = read_semantic_artifact_operational_metadata(
            candidate_root
        ).get("model_trial_ledgers", [])
        if len(old_trials) != len(candidate_trials):
            return SemanticArtifactComparison("CONTENT_MISMATCH")
        for old_ref, candidate_ref in zip(old_trials, candidate_trials, strict=True):
            if trial_layout is None:
                raise ValueError("trial_layout_required")
            old_plan, old_records = read_trial_ledger_reference(trial_layout, old_ref)
            candidate_plan, candidate_records = read_trial_ledger_reference(
                trial_layout, candidate_ref
            )
            if trial_ledger_semantic_payload(
                old_plan, old_records
            ) != trial_ledger_semantic_payload(candidate_plan, candidate_records):
                return SemanticArtifactComparison("CONTENT_MISMATCH")
        for member, candidate_member in zip(
            old_members, candidate_members, strict=True
        ):
            old_path, candidate_path = (
                old_root / member["path"],
                candidate_root / member["path"],
            )
            old_identity, candidate_identity = (
                _member_identity(old_path),
                _member_identity(candidate_path),
            )
            identities[old_path] = old_identity
            identities[candidate_path] = candidate_identity
            if (
                _member_record(member["path"], old_path) != member
                or _member_record(member["path"], candidate_path) != candidate_member
            ):
                raise ValueError("semantic_artifact_member_integrity_invalid")
            if old_path.suffix == ".parquet":
                if not _parquet_members_match(
                    old_path, candidate_path, old_identity, candidate_identity
                ):
                    return SemanticArtifactComparison("CONTENT_MISMATCH")
            else:
                if member != candidate_member:
                    return SemanticArtifactComparison("CONTENT_MISMATCH")
                with (
                    handle_preserving_primary(open_filesystem_file(old_path)) as old,
                    handle_preserving_primary(
                        open_filesystem_file(candidate_path)
                    ) as candidate,
                ):
                    if (
                        dataset_file_identity_from_stat(os.fstat(old.fileno()))
                        != old_identity
                        or dataset_file_identity_from_stat(os.fstat(candidate.fileno()))
                        != candidate_identity
                    ):
                        raise ValueError("semantic_artifact_member_identity_changed")
                    while left := old.read(_HASH_BUFFER_BYTES):
                        if left != candidate.read(_HASH_BUFFER_BYTES):
                            return SemanticArtifactComparison("CONTENT_MISMATCH")
                    if candidate.read(1):
                        return SemanticArtifactComparison("CONTENT_MISMATCH")
                    if (
                        dataset_file_identity_from_stat(os.fstat(old.fileno()))
                        != old_identity
                        or dataset_file_identity_from_stat(os.fstat(candidate.fileno()))
                        != candidate_identity
                    ):
                        raise ValueError("semantic_artifact_member_identity_changed")
            if (
                _member_identity(old_path) != old_identity
                or _member_identity(candidate_path) != candidate_identity
            ):
                raise ValueError("semantic_artifact_member_identity_changed")
        for root in (old_root, candidate_root):
            remaining = set(old_paths)

            def verify(relative: str, _path: Path, remaining=remaining) -> None:
                if relative not in remaining:
                    raise ValueError("semantic_artifact_member_set_changed")
                remaining.remove(relative)

            _visit_member_paths(root, verify)
            if remaining:
                raise ValueError("semantic_artifact_member_set_changed")
        for path, expected_identity in identities.items():
            if _member_identity(path) != expected_identity:
                raise ValueError("semantic_artifact_member_identity_changed")
    except (OSError, ValueError):
        return SemanticArtifactComparison("INSPECTION_FAILED")
    return SemanticArtifactComparison("MATCH")


def _parquet_members_match(old_path, candidate_path, old_identity, candidate_identity):
    # Physical encoding is not semantic identity; reuse the exact row codec.
    with runtime_resource_policy_scope(current_runtime_resource_policy()) as policy:
        schemas = []
        for path, identity in (
            (old_path, old_identity),
            (candidate_path, candidate_identity),
        ):
            with _opened_parquet(path, expected_identity=identity) as opened:
                schema = opened[0].schema_arrow
                metadata = {
                    key: value
                    for key, value in (schema.metadata or {}).items()
                    if key not in {b"qrp.decoded-row-bounds", b"qrp.parquet-page-rows"}
                }
                schemas.append(schema.with_metadata(metadata))
        if not schemas[0].equals(schemas[1], check_metadata=True):
            return False
        payload_target = max(1, policy.operation_headroom_bytes // 8)
        codec = ExactPrimaryKeyIndex(primary_key=(), timestamp_columns=())

        def rows(path, identity):
            with closing_iterator(
                iter_dataset_part_batches(
                    path,
                    expected_identity=identity,
                    target_chunk_bytes=payload_target,
                )
            ) as batches:
                for batch in batches:
                    table = pa.Table.from_batches([batch])
                    for _, _, payload in codec._comparison_payload_rows(
                        table,
                        table,
                        namespace="comparison",
                        payload_target_bytes=payload_target,
                    ):
                        yield payload
                    del batch, table

        missing = object()
        with (
            policy.lease("storage.semantic.comparison.payload", 2 * payload_target),
            closing_iterator(rows(old_path, old_identity)) as old,
            closing_iterator(rows(candidate_path, candidate_identity)) as candidate,
        ):
            return all(
                left == right
                for left, right in zip_longest(old, candidate, fillvalue=missing)
            )


def write_semantic_artifact(
    logical_root: Path,
    envelope: BusinessManifestEnvelope,
    writer: Callable[[Path], None],
    *,
    operational_metadata: Mapping[str, object] | None = None,
    staged_validator: Callable[[Path, tuple[dict[str, object], ...]], None]
    | None = None,
) -> Path:
    final_root = Path(logical_root) / "instances" / envelope.instance_identity

    def write(staging: Path) -> None:
        writer(staging)
        members = _member_inventory(staging)
        if staged_validator is not None:
            staged_validator(staging, members)
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": SEMANTIC_ARTIFACT_SCHEMA_ID,
                "common_manifest": envelope.payload(),
                "members": list(members),
                "operational_metadata": _operational_metadata(
                    dict(operational_metadata or {})
                ),
            },
        )

    atomic_write_directory(final_root, write)
    return final_root


def read_semantic_artifact(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> dict[str, object]:
    envelope, _members = read_semantic_artifact_record(
        instance_root, expected_identity=expected_identity
    )
    return envelope


def read_semantic_artifact_record(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Read one validated semantic envelope and its member inventory."""

    envelope, members = _read_semantic_record(
        instance_root,
        expected_identity=expected_identity,
    )
    return envelope, members


def verify_semantic_artifact_member(
    instance_root: Path,
    *,
    relative_path: str,
    expected: dict[str, object],
) -> None:
    """Verify one already-published member without rescanning its siblings."""

    root = require_unaliased_path(
        Path(instance_root),
        error="semantic_artifact_root_invalid",
    )
    candidate = Path(relative_path)
    if (
        not relative_path
        or candidate.is_absolute()
        or candidate.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("semantic_artifact_member_invalid")
    if _validate_member_payload([expected]) != (expected,):
        raise ValueError("semantic_artifact_member_manifest_invalid")
    member = require_unaliased_path(
        root / Path(*candidate.parts), error="semantic_artifact_member_invalid"
    )
    if _member_record(relative_path, member) != expected:
        raise ValueError("semantic_artifact_member_integrity_invalid")


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
    if set(payload) != {
        "schema_id",
        "common_manifest",
        "members",
        "operational_metadata",
    }:
        raise ValueError("semantic_artifact_manifest_invalid")
    if payload["schema_id"] != SEMANTIC_ARTIFACT_SCHEMA_ID:
        raise ValueError("semantic_artifact_manifest_invalid")
    _operational_metadata(payload["operational_metadata"])
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


def read_semantic_artifact_operational_metadata(
    instance_root: Path,
) -> dict[str, object]:
    read_semantic_artifact_record(instance_root)
    return _operational_metadata(
        read_manifest(instance_root / "manifest.json")["operational_metadata"]
    )


def _operational_metadata(value: object) -> dict[str, object]:
    snapshot = bounded_qrp_json_object_snapshot(value, max_bytes=1024 * 1024)
    if set(snapshot) - {"model_trial_ledgers"}:
        raise ValueError("semantic_artifact_operational_metadata_invalid")
    if "model_trial_ledgers" in snapshot:
        if type(snapshot["model_trial_ledgers"]) is not list:
            raise ValueError("semantic_artifact_operational_metadata_invalid")
        for reference in snapshot["model_trial_ledgers"]:
            validate_trial_ledger_reference(reference)
    assert_internal_artifact_metadata_safe(snapshot)
    return snapshot


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
    before = _directory_identity(instances_root)

    def instance_names() -> tuple[str, ...]:
        names: list[str] = []
        try:
            with os.scandir(native_filesystem_path(instances_root)) as entries:
                for entry in entries:
                    if len(names) >= max_instances:
                        raise ValueError("semantic_artifact_instance_limit_exceeded")
                    names.append(entry.name)
        except OSError as exc:
            raise ValueError("semantic_artifact_root_invalid") from exc
        if _directory_identity(instances_root) != before:
            raise ValueError("semantic_artifact_instance_identity_changed")
        return tuple(sorted(names))

    names = instance_names()

    result: list[Path] = []
    for name in names:
        try:
            validate_instance_identity(name)
        except ValueError as exc:
            raise ValueError("semantic_artifact_instance_child_invalid") from exc
        entry = instances_root / name
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError("semantic_artifact_instance_child_invalid")
        read_semantic_artifact(entry, expected_identity=name)
        result.append(entry)
    if instance_names() != names:
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
    members: list[dict[str, object]] = []

    def record(relative: str, child: Path) -> None:
        members.append(_member_record(relative, child))

    _visit_member_paths(root, record)
    remaining = {str(member["path"]) for member in members}

    def verify(relative: str, _child: Path) -> None:
        if relative not in remaining:
            raise ValueError("semantic_artifact_member_set_changed")
        remaining.remove(relative)

    _visit_member_paths(root, verify)
    if remaining:
        raise ValueError("semantic_artifact_member_set_changed")
    return tuple(sorted(members, key=lambda item: str(item["path"])))


def _visit_member_paths(root: Path, visit: Callable[[str, Path], None]) -> None:
    root = require_unaliased_path(
        Path(root),
        error="semantic_artifact_root_invalid",
    )
    try:
        root_identity = _directory_identity(root)
    except ValueError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    pending = []
    member_count = 0
    try:
        try:
            pending.append(
                (root, "", root_identity, os.scandir(native_filesystem_path(root)))
            )
            while pending:
                directory, prefix, directory_identity, entries = pending[-1]
                entry = next(entries, None)
                if entry is None:
                    pending.pop()
                    entries.close()
                    require_unaliased_path(
                        directory, error="semantic_artifact_member_invalid"
                    )
                    if _directory_identity(directory) != directory_identity:
                        raise ValueError("semantic_artifact_member_set_changed")
                    continue
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                if entry.is_symlink():
                    raise ValueError("semantic_artifact_member_invalid")
                child = directory / entry.name
                if entry.is_dir(follow_symlinks=False):
                    require_unaliased_path(
                        child, error="semantic_artifact_member_invalid"
                    )
                    pending.append(
                        (
                            child,
                            relative,
                            _directory_identity(child),
                            os.scandir(native_filesystem_path(child)),
                        )
                    )
                elif entry.is_file(follow_symlinks=False):
                    if not prefix and entry.name == "manifest.json":
                        continue
                    if member_count >= MAX_SEMANTIC_ARTIFACT_MEMBERS:
                        raise ValueError("semantic_artifact_member_limit_exceeded")
                    member_count += 1
                    visit(relative, child)
                else:
                    raise ValueError("semantic_artifact_member_invalid")
        except BaseException as primary:
            while pending:
                entries = pending.pop()[3]
                try:
                    entries.close()
                except BaseException as close_error:
                    primary.add_note(
                        "semantic inventory close also failed: "
                        f"{type(close_error).__name__}"
                    )
            raise
    except OSError as exc:
        raise ValueError("semantic_artifact_member_invalid") from exc


def _member_record(relative: str, path: Path) -> dict[str, object]:
    identity = _member_identity(path)
    digest = hashlib.sha256()
    try:
        with handle_preserving_primary(open_filesystem_file(path)) as handle:
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
