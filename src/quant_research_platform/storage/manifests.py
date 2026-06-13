"""Manifest helpers."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_research_platform.runtime.privacy import sanitize_public_object
from quant_research_platform.storage.atomic import atomic_write_text

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


def replace_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically replace an existing private artifact manifest."""

    _assert_internal_artifact_metadata_safe(payload)
    final_path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".tmp-{uuid4().hex[:12]}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return final_path


def write_layer_manifest(
    *,
    layer_root: Path,
    logical_layer: str,
    run_id: str,
    dataset_roots: tuple[Path, ...],
) -> Path:
    root = Path(layer_root)
    datasets: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    total_rows = 0
    date_min_values: list[str] = []
    date_max_values: list[str] = []
    for dataset_root in dataset_roots:
        path = Path(dataset_root) / "_manifest" / "manifest.json"
        if not path.is_file():
            missing.append(_logical_path(path))
            continue
        payload = read_manifest(path)
        summary = {
            "row_count": payload.get("row_count"),
            "date_min": payload.get("date_min"),
            "date_max": payload.get("date_max"),
            "duplicate_primary_key_count": payload.get(
                "duplicate_primary_key_count"
            ),
            "validation_status": payload.get("validation_status"),
        }
        datasets.append(
            {
                "data_id": _data_id_from_manifest(payload, dataset_root),
                "manifest_path": _logical_path(path),
                "manifest_summary": summary,
            }
        )
        total_rows += int(payload.get("row_count") or 0)
        if payload.get("date_min"):
            date_min_values.append(str(payload["date_min"]))
        if payload.get("date_max"):
            date_max_values.append(str(payload["date_max"]))
        validation_status = payload.get("validation_status")
        if validation_status is not None and str(validation_status).lower() not in {
            "pass",
            "ok",
        }:
            failed.append(_data_id_from_manifest(payload, dataset_root))
    status = "pass" if not missing and not failed else "fail"
    payload = {
        "run_id": run_id,
        "logical_layer": logical_layer,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "total_rows": total_rows,
        "date_min": min(date_min_values) if date_min_values else None,
        "date_max": max(date_max_values) if date_max_values else None,
        "missing_dataset_manifests": missing,
        "failed_dataset_validations": failed,
        "layer_validation_status": status,
        "created_at": _utc_now(),
    }
    return write_internal_artifact_manifest(
        root / "_manifests" / "layer_manifest.json",
        payload,
    )


def write_run_manifest(
    *,
    manifests_root: Path,
    run_id: str,
    output_root: Path,
    layer_manifest_paths: tuple[Path, ...],
) -> Path:
    layers = [read_manifest(path) for path in layer_manifest_paths]
    failed = [
        str(layer.get("logical_layer"))
        for layer in layers
        if layer.get("layer_validation_status") != "pass"
    ]
    payload = {
        "run_id": run_id,
        "output_root": _logical_path(output_root),
        "layer_count": len(layers),
        "layers": layers,
        "run_validation_status": "pass" if not failed else "fail",
        "failed_layers": failed,
        "created_at": _utc_now(),
    }
    return write_internal_artifact_manifest(
        Path(manifests_root) / "run_manifest.json",
        payload,
    )


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


def _data_id_from_manifest(payload: dict[str, Any], dataset_root: Path) -> str:
    value = payload.get("data_id")
    if value:
        return str(value)
    return _logical_path(Path(dataset_root))


def _logical_path(path: Path) -> str:
    item = Path(path)
    parts = item.parts
    for marker in (".run", "data", "manifests", "validations", "timings"):
        if marker in parts:
            index = parts.index(marker)
            return Path(*parts[index:]).as_posix()
    if item.is_absolute():
        return item.name
    return item.as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "read_manifest",
    "replace_internal_artifact_manifest",
    "write_internal_artifact_manifest",
    "write_layer_manifest",
    "write_manifest",
    "write_run_manifest",
]
