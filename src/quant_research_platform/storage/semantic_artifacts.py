"""Immutable semantic-artifact instances for non-dataset target nodes."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    MAX_QRP_JSON_BYTES,
    bounded_qrp_json_object_snapshot,
    canonical_identity_bytes,
    render_qrp_json_object,
)
from quant_research_platform.contracts.targets import _operation_evidence
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    is_atomic_directory_staging_name,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.experiments import read_trial_ledger_reference
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    _hash_file_content,
    dataset_file_identity_from_stat,
    filesystem_exists,
    filesystem_is_dir,
    filesystem_scandir,
    filesystem_stat,
    open_filesystem_file,
    path_is_filesystem_alias,
    require_unaliased_path,
    validate_instance_identity,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_bounded_qrp_json,
    write_manifest,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    iter_dataset_part_batches,
)

SEMANTIC_ARTIFACT_SCHEMA_ID = "qrp-semantic-artifact"
_HASH_BUFFER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SemanticArtifactComparison:
    status: str


def compare_semantic_artifacts(
    old_root: Path, candidate_root: Path, *, trial_layout: StorageLayout | None = None
) -> SemanticArtifactComparison:
    try:
        identities = {
            root / "manifest.json": _member_identity(root / "manifest.json")
            for root in (old_root, candidate_root)
        }
        old_operational = {}
        candidate_operational = {}
        member_identities = {}
        old_envelope, old_members = read_semantic_artifact_record(
            old_root,
            operational_metadata=old_operational,
            member_identities=member_identities,
        )
        identities.update(
            (
                (old_root / name, identity)
                for name, identity in member_identities.items()
                if name != "manifest.json"
            )
        )
        member_identities.clear()
        candidate_envelope, candidate_members = read_semantic_artifact_record(
            candidate_root,
            operational_metadata=candidate_operational,
            member_identities=member_identities,
        )
        identities.update(
            (
                (candidate_root / name, identity)
                for name, identity in member_identities.items()
                if name != "manifest.json"
            )
        )
        member_identities.clear()
        if old_envelope != candidate_envelope:
            return SemanticArtifactComparison("CONTENT_MISMATCH")
        old_paths = tuple(member["path"] for member in old_members)
        candidate_paths = tuple(member["path"] for member in candidate_members)
        if old_paths != candidate_paths:
            return SemanticArtifactComparison("LAYOUT_MISMATCH")
        old_trials = old_operational.get("model_trial_ledgers", [])
        candidate_trials = candidate_operational.get("model_trial_ledgers", [])
        if len(old_trials) != len(candidate_trials):
            return SemanticArtifactComparison("CONTENT_MISMATCH")
        for old_ref, candidate_ref in zip(old_trials, candidate_trials, strict=True):
            if trial_layout is None:
                raise ValueError("trial_layout_required")
            old_plan, old_records = read_trial_ledger_reference(trial_layout, old_ref)
            candidate_plan, candidate_records = read_trial_ledger_reference(
                trial_layout, candidate_ref
            )
            if canonical_identity_bytes(
                trial_ledger_semantic_payload(old_plan, old_records)
            ) != canonical_identity_bytes(
                trial_ledger_semantic_payload(candidate_plan, candidate_records)
            ):
                return SemanticArtifactComparison("CONTENT_MISMATCH")
        for member, candidate_member in zip(
            old_members, candidate_members, strict=True
        ):
            old_path, candidate_path = (
                old_root / member["path"],
                candidate_root / member["path"],
            )
            old_identity, candidate_identity = (
                identities[old_path],
                identities[candidate_path],
            )
            if (
                _member_identity(old_path) != old_identity
                or _member_identity(candidate_path) != candidate_identity
            ):
                raise ValueError("semantic_artifact_member_identity_changed")
            same_bytes = False
            if member == candidate_member:
                with closing_iterator(open_filesystem_file(old_path)) as old:
                    with closing_iterator(
                        open_filesystem_file(candidate_path)
                    ) as candidate:
                        if (
                            dataset_file_identity_from_stat(os.fstat(old.fileno()))
                            != old_identity
                            or dataset_file_identity_from_stat(
                                os.fstat(candidate.fileno())
                            )
                            != candidate_identity
                        ):
                            raise ValueError(
                                "semantic_artifact_member_identity_changed"
                            )
                        same_bytes = True
                        chunk_size = _HASH_BUFFER_BYTES
                        while left := old.read(chunk_size):
                            if left != candidate.read(chunk_size):
                                same_bytes = False
                                del left
                                break
                            del left
                        if same_bytes and candidate.read(1):
                            same_bytes = False
                        if (
                            dataset_file_identity_from_stat(os.fstat(old.fileno()))
                            != old_identity
                            or dataset_file_identity_from_stat(
                                os.fstat(candidate.fileno())
                            )
                            != candidate_identity
                        ):
                            raise ValueError(
                                "semantic_artifact_member_identity_changed"
                            )
            if not same_bytes and (
                old_path.suffix != ".parquet"
                or not _parquet_members_match(
                    old_path, candidate_path, old_identity, candidate_identity
                )
            ):
                return SemanticArtifactComparison("CONTENT_MISMATCH")
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
    schemas = []
    for path, identity in (
        (old_path, old_identity),
        (candidate_path, candidate_identity),
    ):
        with _opened_parquet(path, expected_identity=identity) as opened:
            schema = opened[0].schema_arrow
            schemas.append(schema)
    if not schemas[0].equals(schemas[1], check_metadata=True):
        return False
    batch_target = DEFAULT_BATCH_ROWS
    codec = ExactPrimaryKeyIndex(primary_key=())

    def rows(path, identity):
        with closing_iterator(
            iter_dataset_part_batches(
                path, expected_identity=identity, batch_rows=batch_target
            )
        ) as batches:
            for batch in batches:
                table = pa.Table.from_batches([batch])
                for _, _, payload in codec._comparison_payload_rows(
                    table, table, namespace="comparison"
                ):
                    try:
                        yield payload
                    except GeneratorExit:
                        return
                del batch, table

    missing = object()
    with closing_iterator(rows(old_path, old_identity)) as old:
        with closing_iterator(rows(candidate_path, candidate_identity)) as candidate:
            while True:
                left = next(old, missing)
                right = next(candidate, missing)
                if left is missing or right is missing:
                    return left is right
                if left != right:
                    return False
                del left, right


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
        if not members:
            raise ValueError("semantic_artifact_member_manifest_invalid")
        if staged_validator is not None:
            staged_validator(staging, members)
        payload = {
            "schema_id": SEMANTIC_ARTIFACT_SCHEMA_ID,
            "common_manifest": envelope.payload(),
            "members": list(members),
            "operational_metadata": _operational_metadata(
                dict(operational_metadata or {})
            ),
        }
        write_manifest(staging / "manifest.json", payload)

    atomic_write_directory(final_root, write)
    return final_root


def read_semantic_artifact(
    instance_root: Path, *, expected_identity: str | None = None
) -> dict[str, object]:
    envelope, _members = read_semantic_artifact_record(
        instance_root, expected_identity=expected_identity
    )
    return envelope


def verify_semantic_artifact_member(
    instance_root: Path, *, relative_path: str, expected: dict[str, object]
) -> DatasetFileIdentity:
    """Verify one already-published member without rescanning its siblings."""
    root = require_unaliased_path(
        Path(instance_root), error="semantic_artifact_root_invalid"
    )
    candidate = Path(relative_path)
    if (
        not relative_path
        or candidate.anchor
        or candidate.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("semantic_artifact_member_invalid")
    if _validate_member_payload([expected]) != (expected,):
        raise ValueError("semantic_artifact_member_manifest_invalid")
    member = require_unaliased_path(
        root / Path(*candidate.parts), error="semantic_artifact_member_invalid"
    )
    identity = _member_identity(member)
    if _member_record(relative_path, member, identity=identity) != expected:
        raise ValueError("semantic_artifact_member_integrity_invalid")
    return identity


def read_semantic_artifact_record(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
    member_identities: dict[str, DatasetFileIdentity] | None = None,
    expected_member_identities: Mapping[str, DatasetFileIdentity] | None = None,
    operational_metadata: dict[str, object] | None = None,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    root = require_unaliased_path(
        Path(instance_root), error="semantic_artifact_root_invalid"
    )
    try:
        _directory_identity(root)
    except ValueError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    manifest_identity = (
        _member_identity(root / "manifest.json")
        if expected_member_identities is None
        else expected_member_identities["manifest.json"]
    )
    payload = read_bounded_qrp_json(
        root / "manifest.json", expected_identity=manifest_identity
    )[1]
    if set(payload) != {
        "schema_id",
        "common_manifest",
        "members",
        "operational_metadata",
    }:
        raise ValueError("semantic_artifact_manifest_invalid")
    if payload["schema_id"] != SEMANTIC_ARTIFACT_SCHEMA_ID:
        raise ValueError("semantic_artifact_manifest_invalid")
    operational = _operational_metadata(payload["operational_metadata"])
    envelope = validate_business_manifest_envelope(payload["common_manifest"])
    if root.name != envelope["instance_identity"]:
        raise ValueError("semantic_artifact_identity_mismatch")
    if (
        expected_identity is not None
        and envelope["instance_identity"] != expected_identity
    ):
        raise ValueError("semantic_artifact_identity_mismatch")
    members = _validate_member_payload(payload["members"])
    if expected_member_identities is not None and set(expected_member_identities) != {
        "manifest.json",
        *(str(member["path"]) for member in members),
    }:
        raise ValueError("semantic_artifact_member_set_changed")
    if members != _member_inventory(
        root,
        expected_paths={str(member["path"]) for member in members},
        verified_identities=member_identities,
        expected_identities=expected_member_identities,
    ):
        raise ValueError("semantic_artifact_member_integrity_invalid")
    if _member_identity(root / "manifest.json") != manifest_identity:
        raise ValueError("semantic_artifact_member_identity_changed")
    if member_identities is not None:
        member_identities["manifest.json"] = manifest_identity
    if operational_metadata is not None:
        operational_metadata.update(operational)
    return (envelope, members)


def read_semantic_artifact_operational_metadata(
    instance_root: Path,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    read_semantic_artifact_record(instance_root, operational_metadata=metadata)
    return metadata


def _operational_metadata(value: object) -> dict[str, object]:
    snapshot = bounded_qrp_json_object_snapshot(value, max_bytes=MAX_QRP_JSON_BYTES)
    if set(snapshot) - {
        "model_trial_ledgers",
        "model_phase_observations",
        "final_alpha_observations",
    }:
        raise ValueError("semantic_artifact_operational_metadata_invalid")
    if "model_trial_ledgers" in snapshot:
        if type(snapshot["model_trial_ledgers"]) is not list:
            raise ValueError("semantic_artifact_operational_metadata_invalid")
        for reference in snapshot["model_trial_ledgers"]:
            validate_trial_ledger_reference(reference)
    if "model_phase_observations" in snapshot:
        costs = snapshot["model_phase_observations"]
        if (
            type(costs) is not dict
            or set(costs) != {"environments", "phases"}
            or type(costs["environments"]) is not list
            or (type(costs["phases"]) is not list)
            or (not costs["phases"])
        ):
            raise ValueError("semantic_artifact_operational_metadata_invalid")
        for context in costs["environments"]:
            if type(context) is not dict or set(context) != {
                "platform",
                "host_id",
                "native_thread_env",
            }:
                raise ValueError("semantic_artifact_operational_metadata_invalid")
        used_contexts = set()
        for phase in costs["phases"]:
            if (
                type(phase) is not dict
                or set(phase)
                != {
                    "phase",
                    "selection_group",
                    "family_id",
                    "quarter_id",
                    "fit_count",
                    "observations",
                }
                or type(phase["phase"]) is not str
                or phase["phase"]
                not in {"shared_candidate_fit", "group_scoring", "final_refit"}
                or (type(phase["selection_group"]) not in {str, type(None)})
                or (phase["selection_group"] not in {None, "global", "quarterly"})
                or (type(phase["fit_count"]) is not int)
                or (phase["fit_count"] < 0)
                or (type(phase["observations"]) is not dict)
            ):
                raise ValueError("semantic_artifact_operational_metadata_invalid")
            for key in ("family_id", "quarter_id"):
                if phase[key] is not None and (
                    type(phase[key]) is not str or not phase[key]
                ):
                    raise ValueError("semantic_artifact_operational_metadata_invalid")
            observation = dict(phase["observations"])
            environment = observation.get("environment")
            if (
                type(environment) is not dict
                or set(environment) != {"context", "created_at_utc"}
                or type(environment["context"]) is not int
                or (not 0 <= environment["context"] < len(costs["environments"]))
            ):
                raise ValueError("semantic_artifact_operational_metadata_invalid")
            used_contexts.add(environment["context"])
            observation["environment"] = {
                **costs["environments"][environment["context"]],
                "created_at_utc": environment["created_at_utc"],
            }
            _operation_evidence(observation)
        if used_contexts != set(range(len(costs["environments"]))):
            raise ValueError("semantic_artifact_operational_metadata_invalid")
    if "final_alpha_observations" in snapshot:
        _operation_evidence(snapshot["final_alpha_observations"])
    assert_internal_artifact_metadata_safe(snapshot)
    return snapshot


def discover_semantic_artifact_instances(logical_root: Path) -> tuple[Path, ...]:
    root = require_unaliased_path(
        Path(logical_root), error="semantic_artifact_root_invalid"
    )
    if not filesystem_exists(root):
        return ()
    instances_root = root / "instances"
    require_unaliased_path(instances_root, error="semantic_artifact_root_invalid")
    if not filesystem_exists(instances_root):
        return ()
    before = _directory_identity(instances_root)

    def instance_names() -> tuple[str, ...]:
        names: list[str] = []
        try:
            with filesystem_scandir(instances_root) as entries:
                for entry in entries:
                    names.append(entry.name)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("semantic_artifact_root_invalid") from exc
        if _directory_identity(instances_root) != before:
            raise ValueError("semantic_artifact_instance_identity_changed")
        return tuple(sorted(names))

    names = instance_names()
    result: list[Path] = []
    for name in names:
        entry = instances_root / name
        if path_is_filesystem_alias(entry) or not filesystem_is_dir(entry):
            raise ValueError("semantic_artifact_instance_child_invalid")
        if is_atomic_directory_staging_name(name):
            continue
        try:
            validate_instance_identity(name)
        except ValueError as exc:
            raise ValueError("semantic_artifact_instance_child_invalid") from exc
        read_semantic_artifact(entry, expected_identity=name)
        result.append(entry)
    if instance_names() != names:
        raise ValueError("semantic_artifact_instance_identity_changed")
    return tuple(result)


def _directory_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        value = filesystem_stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError("semantic_artifact_root_invalid")
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def _member_inventory(
    root: Path,
    *,
    expected_paths: set[str] | None = None,
    verified_identities: dict[str, DatasetFileIdentity] | None = None,
    expected_identities: Mapping[str, DatasetFileIdentity] | None = None,
) -> tuple[dict[str, object], ...]:
    members: list[dict[str, object]] = []
    identities: dict[str, DatasetFileIdentity] = {}
    encoded_bytes = 0

    def record(relative: str, child: Path) -> None:
        nonlocal encoded_bytes
        if expected_paths is not None and relative not in expected_paths:
            raise ValueError("semantic_artifact_member_integrity_invalid")
        identity = (
            _member_identity(child)
            if expected_identities is None
            else expected_identities[relative]
        )
        identities[relative] = identity
        member = _member_record(relative, child, identity=identity)
        encoded_bytes += len(render_qrp_json_object(member).encode("utf-8"))
        if encoded_bytes > MAX_QRP_JSON_BYTES:
            raise ValueError("semantic_artifact_member_limit_exceeded")
        members.append(member)

    _visit_member_paths(root, record)
    remaining = {str(member["path"]) for member in members}

    def verify(relative: str, _child: Path) -> None:
        if relative not in remaining:
            raise ValueError("semantic_artifact_member_set_changed")
        remaining.remove(relative)
        if _member_identity(_child) != identities[relative]:
            raise ValueError("semantic_artifact_member_identity_changed")

    _visit_member_paths(root, verify)
    if remaining:
        raise ValueError("semantic_artifact_member_set_changed")
    if verified_identities is not None:
        verified_identities.update(identities)
    return tuple(sorted(members, key=lambda item: str(item["path"])))


def _visit_member_paths(root: Path, visit: Callable[[str, Path], None]) -> None:
    root = require_unaliased_path(Path(root), error="semantic_artifact_root_invalid")
    try:
        root_identity = _directory_identity(root)
    except ValueError as exc:
        raise ValueError("semantic_artifact_root_invalid") from exc
    pending = []
    path_bytes = 0
    try:
        try:
            pending.append((root, "", root_identity, filesystem_scandir(root)))
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
                path_bytes += len(relative.encode("utf-8")) + 2
                if path_bytes > MAX_QRP_JSON_BYTES:
                    raise ValueError("semantic_artifact_member_limit_exceeded")
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
                            filesystem_scandir(child),
                        )
                    )
                elif entry.is_file(follow_symlinks=False):
                    if not prefix and entry.name == "manifest.json":
                        continue
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
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("semantic_artifact_member_invalid") from exc


def _member_record(
    relative: str, path: Path, *, identity: DatasetFileIdentity | None = None
) -> dict[str, object]:
    if identity is None:
        identity = _member_identity(path)
    try:
        digest = _hash_file_content(
            path,
            identity,
            "semantic_artifact_member_identity_changed",
            chunk_size=_HASH_BUFFER_BYTES,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("semantic_artifact_member_invalid") from exc
    return {"path": relative, "size": identity.size, "sha256": digest}


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
            or candidate.anchor
            or (candidate.as_posix() != relative)
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or (type(size) is not int)
            or (size < 0)
            or (type(digest) is not str)
            or (len(digest) != 64)
            or any(char not in "0123456789abcdef" for char in digest)
            or (relative in seen)
        ):
            raise ValueError("semantic_artifact_member_manifest_invalid")
        seen.add(relative)
        result.append({"path": relative, "size": size, "sha256": digest})
    return tuple(sorted(result, key=lambda item: str(item["path"])))


def _member_identity(path: Path) -> DatasetFileIdentity:
    try:
        value = filesystem_stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("semantic_artifact_member_invalid") from exc
    if not stat.S_ISREG(value.st_mode) or int(value.st_ino) <= 0:
        raise ValueError("semantic_artifact_member_invalid")
    return dataset_file_identity_from_stat(value)
