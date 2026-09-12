"""Manifest helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.storage.atomic import atomic_write_text
from quant_research_platform.storage.datasets.schema import (
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
)
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    native_filesystem_path,
    open_filesystem_file,
    require_artifact_root,
    require_unaliased_path,
)

_MAX_QRP_JSON_BYTES = 1024 * 1024


def _qrp_json_file_identity(path: Path) -> DatasetFileIdentity:
    try:
        return dataset_file_identity(path)
    except ValueError as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc


def read_bounded_qrp_json(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
    max_bytes: int = _MAX_QRP_JSON_BYTES,
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
        with scratch_owner.handle_preserving_primary(handle):
            before = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            if before != identity:
                raise ValueError("dataset_metadata_file_identity_changed")
            raw = handle.read(max_bytes + 1)
            after = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    except ValueError as exc:
        raise ValueError("dataset_metadata_file_identity_changed") from exc
    if len(raw) > max_bytes:
        raise ValueError("dataset_metadata_file_too_large")
    if before != identity or after != identity:
        raise ValueError("dataset_metadata_file_identity_changed")
    return raw, parse_qrp_json_object(raw)


def read_bounded_qrp_json_directory(
    root: Path,
    *,
    max_members: int,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Read one bounded directory containing only QRP JSON object files."""

    if type(max_members) is not int or max_members < 0:
        raise ValueError("dataset_metadata_member_limit_invalid")
    candidate = require_unaliased_path(
        Path(root), error="dataset_metadata_directory_invalid"
    )
    if not candidate.is_dir():
        raise ValueError("dataset_metadata_directory_invalid")

    def member_names() -> tuple[str, ...]:
        names: list[str] = []
        try:
            with os.scandir(native_filesystem_path(candidate)) as entries:
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
            after = os.stat(native_filesystem_path(candidate), follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("dataset_metadata_directory_invalid") from exc
        if _directory_identity(before) != _directory_identity(after):
            raise ValueError("dataset_metadata_directory_identity_changed")
        return tuple(sorted(names))

    try:
        before = os.stat(native_filesystem_path(candidate), follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_directory_invalid") from exc
    names = member_names()
    result = tuple((name, read_bounded_qrp_json(candidate / name)[1]) for name in names)
    if member_names() != names:
        raise ValueError("dataset_metadata_directory_identity_changed")
    return result


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def read_manifest(path: Path) -> dict[str, Any]:
    return read_bounded_qrp_json(path)[1]


def read_qrp_json(
    path: Path,
    *,
    expected_schema_id: str,
    error: str,
) -> dict[str, Any]:
    payload = read_manifest(path)
    if payload.get("schema_id") != expected_schema_id:
        raise ValueError(error)
    return payload


def read_dataset_manifest_evidence(dataset_root: Path) -> dict[str, Any]:
    root = require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    manifest_path = root / "_manifest" / "manifest.json"
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    if not os.path.isfile(native_filesystem_path(spec_path)) or not os.path.isfile(
        native_filesystem_path(manifest_path)
    ):
        raise FileNotFoundError("dataset_manifest_evidence_missing")
    spec = read_qrp_json(
        spec_path,
        expected_schema_id="qrp-dataset-spec",
        error="dataset_spec_schema_id_invalid",
    )
    payload = read_qrp_json(
        manifest_path,
        expected_schema_id="qrp-dataset-manifest",
        error="dataset_manifest_schema_id_invalid",
    )
    validate_dataset_spec_payload(spec)
    validate_dataset_manifest_payload(payload, spec=spec)
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    snapshot = _bounded_qrp_json_snapshot(payload)
    _require_schema_id(snapshot)
    text = render_qrp_json_object(snapshot) + "\n"
    return atomic_write_text(path, text, encoding="utf-8")


def write_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write private artifact metadata without corrupting schema keys.

    Internal manifests are private artifacts, so schema fields such as
    ``source_id`` must remain round-trippable. We still reject path, credential,
    and row-sample payloads that do not belong in schema metadata.
    """

    snapshot = _bounded_qrp_json_snapshot(payload)
    _require_schema_id(snapshot)
    assert_internal_artifact_metadata_safe(snapshot)
    text = render_qrp_json_object(snapshot) + "\n"
    return atomic_write_text(path, text, encoding="utf-8")


def _require_schema_id(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("manifest_json_identity_invalid")
    schema_id = payload.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id or "/" in schema_id:
        raise ValueError("manifest_json_identity_invalid")


def _bounded_qrp_json_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return bounded_qrp_json_object_snapshot(
        payload,
        max_bytes=_MAX_QRP_JSON_BYTES - 1,
    )
