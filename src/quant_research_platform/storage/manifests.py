"""Manifest helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    MAX_QRP_JSON_BYTES,
    _bounded_qrp_json_snapshot_text,
    parse_qrp_json_object,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.storage.atomic import atomic_write_text
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    filesystem_is_dir,
    filesystem_is_file,
    filesystem_scandir,
    filesystem_stat,
    open_filesystem_file,
    require_artifact_root,
    require_unaliased_path,
    verified_file_handle,
)


def _qrp_json_file_identity(path: Path) -> DatasetFileIdentity:
    try:
        return dataset_file_identity(path)
    except ValueError as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc


def read_bounded_qrp_json(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
    max_bytes: int = MAX_QRP_JSON_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("dataset_metadata_size_limit_invalid")
    candidate = Path(path)
    identity = _qrp_json_file_identity(candidate)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("dataset_metadata_file_identity_changed")
    if identity.size > max_bytes:
        raise ValueError("dataset_metadata_file_too_large")
    try:
        handle = open_filesystem_file(candidate)
        with closing_iterator(handle):
            with verified_file_handle(
                candidate,
                handle,
                expected_identity=identity,
                error="dataset_metadata_file_identity_changed",
            ):
                raw = handle.read(identity.size + 1)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    except ValueError as exc:
        raise ValueError("dataset_metadata_file_identity_changed") from exc
    if len(raw) > max_bytes:
        raise ValueError("dataset_metadata_file_too_large")
    return (raw, parse_qrp_json_object(raw))


def read_bounded_qrp_json_directory(
    root: Path, *, max_members: int
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Read one bounded directory containing only QRP JSON object files."""
    if type(max_members) is not int or max_members < 0:
        raise ValueError("dataset_metadata_member_limit_invalid")
    candidate = require_unaliased_path(
        Path(root), error="dataset_metadata_directory_invalid"
    )
    if not filesystem_is_dir(candidate):
        raise ValueError("dataset_metadata_directory_invalid")

    def member_names() -> tuple[str, ...]:
        names: list[str] = []
        try:
            with filesystem_scandir(candidate) as entries:
                for entry in entries:
                    if len(names) >= max_members:
                        raise ValueError("dataset_metadata_member_limit_exceeded")
                    if (
                        not entry.is_file(follow_symlinks=False)
                        or Path(entry.name).suffix != ".json"
                        or Path(entry.name).name != entry.name
                    ):
                        raise ValueError("dataset_metadata_directory_member_invalid")
                    names.append(entry.name)
            after = filesystem_stat(candidate, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("dataset_metadata_directory_invalid") from exc
        if _directory_identity(before) != _directory_identity(after):
            raise ValueError("dataset_metadata_directory_identity_changed")
        return tuple(sorted(names))

    try:
        before = filesystem_stat(candidate, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_directory_invalid") from exc
    names = member_names()
    result = []
    for name in names:
        identity = _qrp_json_file_identity(candidate / name)
        result.append(
            (
                name,
                read_bounded_qrp_json(candidate / name, expected_identity=identity)[1],
            )
        )
    if member_names() != names:
        raise ValueError("dataset_metadata_directory_identity_changed")
    return tuple(result)


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def read_manifest(path: Path) -> dict[str, Any]:
    return read_bounded_qrp_json(path)[1]


def read_qrp_json(
    path: Path,
    *,
    expected_schema_id: str,
    error: str,
    expected_identity: DatasetFileIdentity | None = None,
) -> dict[str, Any]:
    payload = read_bounded_qrp_json(path, expected_identity=expected_identity)[1]
    if payload.get("schema_id") != expected_schema_id:
        raise ValueError(error)
    return payload


def read_dataset_manifest_evidence(
    dataset_root: Path,
    *,
    member_identities: dict[str, DatasetFileIdentity] | None = None,
    expected_member_identities: Mapping[str, DatasetFileIdentity] | None = None,
) -> dict[str, Any]:
    root = require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    manifest_path = root / "_manifest" / "manifest.json"
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    if not filesystem_is_file(spec_path) or not filesystem_is_file(manifest_path):
        raise FileNotFoundError("dataset_manifest_evidence_missing")
    paths = {"spec.json": spec_path, "_manifest/manifest.json": manifest_path}
    if expected_member_identities is not None and set(
        expected_member_identities
    ) != set(paths):
        raise ValueError("dataset_metadata_member_set_mismatch")
    identities = {
        name: _qrp_json_file_identity(path)
        if expected_member_identities is None
        else expected_member_identities[name]
        for name, path in paths.items()
    }
    spec = read_qrp_json(
        spec_path,
        expected_identity=identities["spec.json"],
        expected_schema_id="qrp-dataset-spec",
        error="dataset_spec_schema_id_invalid",
    )
    payload = read_qrp_json(
        manifest_path,
        expected_identity=identities["_manifest/manifest.json"],
        expected_schema_id="qrp-dataset-manifest",
        error="dataset_manifest_schema_id_invalid",
    )
    from quant_research_platform.storage.datasets.schema import (
        validate_dataset_manifest_payload,
    )

    validate_dataset_manifest_payload(payload, spec=spec)
    if any(
        (
            _qrp_json_file_identity(path) != identities[name]
            for name, path in paths.items()
        )
    ):
        raise ValueError("dataset_metadata_file_identity_changed")
    if member_identities is not None:
        member_identities.update(identities)
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    snapshot, text = _bounded_qrp_json_snapshot_text(
        payload, max_bytes=MAX_QRP_JSON_BYTES - 1
    )
    _require_schema_id(snapshot)
    return atomic_write_text(path, text + "\n", encoding="utf-8")


def write_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write private artifact metadata without corrupting schema keys.

    Internal manifests are private artifacts, so schema fields such as
    ``source_id`` must remain round-trippable. We still reject path, credential,
    and row-sample payloads that do not belong in schema metadata.
    """
    snapshot, text = _bounded_qrp_json_snapshot_text(
        payload, max_bytes=MAX_QRP_JSON_BYTES - 1
    )
    _require_schema_id(snapshot)
    assert_internal_artifact_metadata_safe(snapshot)
    return atomic_write_text(path, text + "\n", encoding="utf-8")


def _require_schema_id(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("manifest_json_identity_invalid")
    schema_id = payload.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id or "/" in schema_id:
        raise ValueError("manifest_json_identity_invalid")
