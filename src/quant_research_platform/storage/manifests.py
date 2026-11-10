"""Read strict JSON metadata and atomically publish manifests with credential checks."""

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
    open_filesystem_file,
)


def read_qrp_json_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    candidate = Path(path)
    try:
        with closing_iterator(open_filesystem_file(candidate)) as handle:
            raw = handle.read()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    return raw, parse_qrp_json_object(raw)


def read_manifest(path: Path) -> dict[str, Any]:
    return read_qrp_json_file(path)[1]


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
