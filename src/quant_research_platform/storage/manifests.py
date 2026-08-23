"""Manifest helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.contracts._json import (
    bounded_qrp_json_object_snapshot,
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.policy.check_status import require_status
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.runtime.logical_paths import logical_path
from quant_research_platform.storage._dataset_paths import (
    require_artifact_root,
    require_unaliased_path,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    dataset_file_identity,
    dataset_file_identity_from_stat,
)
from quant_research_platform.storage._dataset_schema import (
    validate_dataset_manifest_consumable,
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
)
from quant_research_platform.storage._scratch_ownership import (
    remove_owned_scratch,
)
from quant_research_platform.storage.atomic import atomic_write_text

_DATASET_LOGICAL_PATH_MARKERS = ("data",)
_MAX_QRP_JSON_BYTES = 1024 * 1024


def read_bounded_qrp_json(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
) -> tuple[bytes, dict[str, Any]]:
    candidate = require_unaliased_path(
        Path(path), error="dataset_metadata_file_invalid"
    )
    try:
        identity = dataset_file_identity(candidate)
    except (OSError, ValueError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("dataset_metadata_file_identity_changed")
    if identity.size > _MAX_QRP_JSON_BYTES:
        raise ValueError("dataset_metadata_file_too_large")
    try:
        handle = candidate.open("rb")
        with scratch_owner.handle_preserving_primary(handle):
            before = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            if before != identity:
                raise ValueError("dataset_metadata_file_identity_changed")
            raw = handle.read(_MAX_QRP_JSON_BYTES + 1)
            after = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
        current = dataset_file_identity(candidate)
    except OSError as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    except ValueError as exc:
        raise ValueError("dataset_metadata_file_identity_changed") from exc
    if len(raw) > _MAX_QRP_JSON_BYTES:
        raise ValueError("dataset_metadata_file_too_large")
    if before != identity or after != identity or current != identity:
        raise ValueError("dataset_metadata_file_identity_changed")
    return raw, parse_qrp_json_object(raw)


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
    if not spec_path.is_file() or not manifest_path.is_file():
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


def replace_qrp_json(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically replace a QRP-owned JSON object after share-safety validation."""

    snapshot = _bounded_qrp_json_snapshot(payload)
    _require_schema_id(snapshot)
    unsafe = evidence_json_value_unsafe_reason(snapshot)
    if unsafe is not None:
        raise ValueError(f"unsafe_qrp_json: {unsafe}")
    final_path = Path(path)
    text = render_qrp_json_object(snapshot) + "\n"
    require_unaliased_path(final_path.parent, error="invalid_storage_write_path")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    require_unaliased_path(final_path, error="invalid_storage_write_path")
    temp_path = final_path.with_name(
        scratch_owner.owned_scratch_name(".f-", uuid4().hex[:12])
    )
    require_unaliased_path(temp_path, error="invalid_storage_write_path")
    claimed_temp = False
    try:
        handle = scratch_owner.open_owned_scratch_file(
            temp_path,
            encoding="utf-8",
        )
        claimed_temp = True
        with scratch_owner.handle_preserving_primary(handle):
            require_unaliased_path(
                final_path.parent,
                error="invalid_storage_write_path",
            )
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        require_unaliased_path(temp_path, error="invalid_storage_write_path")
        require_unaliased_path(final_path, error="invalid_storage_write_path")
        temp_path.replace(final_path)
    except BaseException as primary:
        if claimed_temp:
            try:
                require_unaliased_path(temp_path, error="invalid_storage_write_path")
                remove_owned_scratch(temp_path)
            except BaseException as cleanup_error:
                primary.add_note(
                    "manifest scratch cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
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
    invalid: list[str] = []
    failed: list[str] = []
    total_rows = 0
    date_min_values: list[str] = []
    date_max_values: list[str] = []
    for dataset_root in dataset_roots:
        path = Path(dataset_root) / "_manifest" / "manifest.json"
        manifest_logical_path = logical_path(
            path,
            markers=_DATASET_LOGICAL_PATH_MARKERS,
        )
        try:
            payload = read_dataset_manifest_evidence(dataset_root)
        except FileNotFoundError:
            missing.append(manifest_logical_path)
            continue
        except (OSError, ValueError):
            invalid.append(manifest_logical_path)
            continue
        data_id = payload["data_id"]
        try:
            validate_dataset_manifest_consumable(payload)
            effective_validation_status = "PASS"
        except ValueError:
            effective_validation_status = "FAIL"
        summary = {
            "row_count": payload.get("row_count"),
            "date_min": payload.get("date_min"),
            "date_max": payload.get("date_max"),
            "partition_count": payload.get("partition_count"),
            "duplicate_primary_key_count": payload.get("duplicate_primary_key_count"),
            "validation_status": effective_validation_status,
        }
        datasets.append(
            {
                "data_id": data_id,
                "manifest_path": manifest_logical_path,
                "manifest_summary": summary,
            }
        )
        total_rows += payload["row_count"]
        if payload.get("date_min"):
            date_min_values.append(payload["date_min"])
        if payload.get("date_max"):
            date_max_values.append(payload["date_max"])
        if effective_validation_status != "PASS":
            failed.append(data_id)
    status = "PASS" if not missing and not invalid and not failed else "FAIL"
    payload = {
        "logical_layer": logical_layer,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "total_rows": total_rows,
        "date_min": min(date_min_values) if date_min_values else None,
        "date_max": max(date_max_values) if date_max_values else None,
        "missing_dataset_manifests": missing,
        "invalid_dataset_manifests": invalid,
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
    failed: list[str] = []
    for layer in layers:
        logical_layer = layer.get("logical_layer")
        if type(logical_layer) is not str or not logical_layer:
            raise ValueError("run_data_summary_layer_invalid")
        status = require_status(layer.get("layer_validation_status"))
        if status != "PASS":
            failed.append(logical_layer)
    payload = {
        "layer_count": len(layers),
        "layers": layers,
        "run_validation_status": "PASS" if not failed else "FAIL",
        "failed_layers": failed,
        "created_at": _utc_now(),
    }
    return payload


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


__all__ = [
    "build_layer_summary",
    "build_run_data_summary",
    "read_manifest",
    "read_dataset_manifest_evidence",
    "read_qrp_json",
    "replace_qrp_json",
    "write_internal_artifact_manifest",
    "write_manifest",
]
