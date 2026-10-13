"""Manifest helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    _qrp_json_snapshot_text,
    parse_qrp_json_object,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.storage.atomic import atomic_write_text
from quant_research_platform.storage.identity import (
    filesystem_scandir,
    open_filesystem_file,
    require_artifact_root,
    require_unaliased_path,
)


def read_qrp_json_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    candidate = require_unaliased_path(
        Path(path), error="dataset_metadata_file_invalid"
    )
    try:
        with closing_iterator(open_filesystem_file(candidate)) as handle:
            raw = handle.read()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    return raw, parse_qrp_json_object(raw)


def read_bounded_qrp_json_directory(
    root: Path, *, max_members: int
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Read declared JSON members once, retaining ordering and bounded membership."""
    if type(max_members) is not int or max_members < 0:
        raise ValueError("dataset_metadata_member_limit_invalid")
    root = require_unaliased_path(
        Path(root), error="dataset_metadata_directory_invalid"
    )
    names = []
    with filesystem_scandir(root) as entries:
        for entry in entries:
            if len(names) >= max_members:
                raise ValueError("dataset_metadata_member_limit_exceeded")
            if (
                not entry.is_file(follow_symlinks=False)
                or Path(entry.name).suffix != ".json"
            ):
                raise ValueError("dataset_metadata_directory_member_invalid")
            names.append(entry.name)
    return tuple((name, read_qrp_json_file(root / name)[1]) for name in sorted(names))


def read_manifest(path: Path) -> dict[str, Any]:
    return read_qrp_json_file(path)[1]


def read_qrp_json(path: Path, *, expected_schema_id: str, error: str) -> dict[str, Any]:
    payload = read_qrp_json_file(path)[1]
    if payload.get("schema_id") != expected_schema_id:
        raise ValueError(error)
    return payload


def read_dataset_manifest_evidence(dataset_root: Path) -> dict[str, Any]:
    root = require_artifact_root(Path(dataset_root))
    spec = read_qrp_json(
        root / "spec.json",
        expected_schema_id="qrp-dataset-spec",
        error="dataset_spec_schema_id_invalid",
    )
    payload = read_qrp_json(
        root / "_manifest" / "manifest.json",
        expected_schema_id="qrp-dataset-manifest",
        error="dataset_manifest_schema_id_invalid",
    )
    from quant_research_platform.storage.datasets.schema import (
        validate_dataset_manifest_payload,
    )

    validate_dataset_manifest_payload(payload, spec=spec)
    return {**spec, **payload}


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    snapshot, text = _qrp_json_snapshot_text(payload)
    _require_schema_id(snapshot)
    return atomic_write_text(path, text + "\n", encoding="utf-8")


def write_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write private artifact metadata without corrupting schema keys.

    Internal manifests are private artifacts, so schema fields such as
    ``source_id`` and supplied input paths remain round-trippable. Credentials
    and tracebacks are rejected; publication is private, not share-safe export.
    """
    snapshot, text = _qrp_json_snapshot_text(payload)
    _require_schema_id(snapshot)
    assert_internal_artifact_metadata_safe(snapshot)
    return atomic_write_text(path, text + "\n", encoding="utf-8")


def _require_schema_id(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("manifest_json_identity_invalid")
    schema_id = payload.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id or "/" in schema_id:
        raise ValueError("manifest_json_identity_invalid")
