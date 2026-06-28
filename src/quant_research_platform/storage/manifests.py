"""Manifest helpers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
    evidence_json_value_unsafe_reason,
    qrp_json_identity_reason,
    sanitize_shared_object,
)
from quant_research_platform.runtime.logical_paths import logical_path
from quant_research_platform.storage.atomic import atomic_write_text

_DATASET_LOGICAL_PATH_MARKERS = ("data",)


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_qrp_json(
    path: Path,
    *,
    expected_schema_id: str,
    error: str,
) -> dict[str, Any]:
    payload = read_manifest(path)
    require_qrp_json_identity(
        payload,
        expected_schema_id=expected_schema_id,
        error=error,
    )
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    _require_schema_id(payload)
    text = (
        json.dumps(sanitize_shared_object(payload), ensure_ascii=False, indent=2) + "\n"
    )
    return atomic_write_text(path, text, encoding="utf-8")


def write_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write private artifact metadata without corrupting schema keys.

    Internal manifests are private artifacts, so schema fields such as
    ``source_id`` must remain round-trippable. We still reject path, credential,
    and row-sample payloads that do not belong in schema metadata.
    """

    _require_schema_id(payload)
    assert_internal_artifact_metadata_safe(payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return atomic_write_text(path, text, encoding="utf-8")


def replace_internal_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically replace an existing private artifact manifest."""

    _require_schema_id(payload)
    assert_internal_artifact_metadata_safe(payload)
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


def replace_qrp_json(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically replace a QRP-owned JSON object after share-safety validation."""

    _require_schema_id(payload)
    unsafe = evidence_json_value_unsafe_reason(payload)
    if unsafe is not None:
        raise ValueError(f"unsafe_qrp_json: {unsafe}")
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


def build_layer_summary(
    *,
    layer_root: Path,
    logical_layer: str,
    dataset_roots: tuple[Path, ...],
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    total_rows = 0
    date_min_values: list[str] = []
    date_max_values: list[str] = []
    for dataset_root in dataset_roots:
        path = Path(dataset_root) / "_manifest" / "manifest.json"
        if not path.is_file():
            missing.append(logical_path(path, markers=_DATASET_LOGICAL_PATH_MARKERS))
            continue
        payload = read_manifest(path)
        if payload.get("schema_id") != "qrp-dataset-manifest":
            missing.append(logical_path(path, markers=_DATASET_LOGICAL_PATH_MARKERS))
            continue
        summary = {
            "row_count": payload.get("row_count"),
            "date_min": payload.get("date_min"),
            "date_max": payload.get("date_max"),
            "duplicate_primary_key_count": payload.get("duplicate_primary_key_count"),
            "validation_status": payload.get("validation_status"),
        }
        datasets.append(
            {
                "data_id": _data_id_from_manifest(payload, dataset_root),
                "manifest_path": logical_path(
                    path,
                    markers=_DATASET_LOGICAL_PATH_MARKERS,
                ),
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
    return payload


def build_run_data_summary(
    *,
    layer_summaries: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    layers = [dict(layer) for layer in layer_summaries]
    failed = [
        str(layer.get("logical_layer"))
        for layer in layers
        if layer.get("layer_validation_status") != "pass"
    ]
    payload = {
        "layer_count": len(layers),
        "layers": layers,
        "run_validation_status": "pass" if not failed else "fail",
        "failed_layers": failed,
        "created_at": _utc_now(),
    }
    return payload


def _data_id_from_manifest(payload: dict[str, Any], dataset_root: Path) -> str:
    value = payload.get("data_id")
    if value:
        return str(value)
    return logical_path(Path(dataset_root), markers=_DATASET_LOGICAL_PATH_MARKERS)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_schema_id(payload: object) -> None:
    require_qrp_json_identity(
        payload,
        expected_schema_id=None,
        error="manifest_json_identity_invalid",
    )


def require_qrp_json_identity(
    payload: object,
    *,
    expected_schema_id: str | None,
    error: str,
) -> None:
    reason = qrp_json_identity_reason(payload)
    if reason is not None:
        raise ValueError(f"{error}: {reason}")
    if (
        expected_schema_id is not None
        and isinstance(payload, dict)
        and payload.get("schema_id") != expected_schema_id
    ):
        raise ValueError(error)


__all__ = [
    "build_layer_summary",
    "build_run_data_summary",
    "read_manifest",
    "read_qrp_json",
    "replace_qrp_json",
    "replace_internal_artifact_manifest",
    "require_qrp_json_identity",
    "write_internal_artifact_manifest",
    "write_manifest",
]
