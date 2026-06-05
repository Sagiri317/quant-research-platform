"""Manifest helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from quant_research_platform.storage.atomic import atomic_write_text
from quant_research_platform.storage.privacy import sanitize_public_object

_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?i)\b[a-z]:[\\/][^\r\n\t\"'<>|]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\r\n\t\"'<>|]+")
_POSIX_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?<!\w)/(?:mnt|home|Users|tmp|var/tmp|private/tmp)/[^\r\n\t\"'<>|]+"
)
_RAW_DATA_PATH_RE = re.compile(r"(?i)\bdata[\\/]raw[\\/][^\r\n\t\"'<>|]+")
_RAW_EXPORT_FILENAME_RE = re.compile(
    r"(?i)\b[a-z0-9_]+__[^\\/]+__export-\d{8}\.(?:csv|txt)\b"
)
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:token|secret|credential|password|api[_-]?key)\s*[:=]\s*"
    r"[A-Za-z0-9._~+/=-]{8,}"
)
_ROW_SAMPLE_KEYS = {"row", "rows", "sample", "samples", "row_sample", "row_samples"}


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    text = (
        json.dumps(sanitize_public_object(payload), ensure_ascii=False, indent=2)
        + "\n"
    )
    return atomic_write_text(path, text, encoding="utf-8")


def write_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write private artifact metadata without corrupting schema keys.

    Internal manifests are private artifacts, so schema fields such as
    ``source_id`` must remain round-trippable. We still reject path, credential,
    and row-sample payloads that do not belong in schema metadata.
    """

    _assert_internal_artifact_metadata_safe(payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return atomic_write_text(path, text, encoding="utf-8")


def _assert_internal_artifact_metadata_safe(value: Any) -> None:
    unsafe = _unsafe_internal_artifact_reason(value)
    if unsafe is not None:
        raise ValueError(f"unsafe_internal_artifact_metadata: {unsafe}")


def _unsafe_internal_artifact_reason(
    value: Any,
    *,
    key: str | None = None,
) -> str | None:
    if key is not None and key.lower() in _ROW_SAMPLE_KEYS and value:
        return "row_sample"
    if isinstance(value, dict):
        for item_key, item in value.items():
            reason = _unsafe_internal_artifact_reason(str(item_key))
            if reason is not None:
                return reason
            reason = _unsafe_internal_artifact_reason(item, key=str(item_key))
            if reason is not None:
                return reason
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            reason = _unsafe_internal_artifact_reason(item)
            if reason is not None:
                return reason
        return None
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if _WINDOWS_DRIVE_PATH_RE.search(value) or _UNC_PATH_RE.search(value):
            return "absolute_path"
        if _POSIX_PRIVATE_PATH_RE.search(value):
            return "absolute_path"
        if _RAW_DATA_PATH_RE.search(value):
            return "raw_path"
        if _RAW_EXPORT_FILENAME_RE.search(value):
            return "raw_export_filename"
        if _CREDENTIAL_RE.search(value):
            return "credential"
    return None


__all__ = ["read_manifest", "write_internal_artifact_manifest", "write_manifest"]
