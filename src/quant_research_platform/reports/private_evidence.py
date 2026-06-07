"""Aggregate private evidence report writers."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text
from quant_research_platform.storage.manifests import read_manifest

_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?i)\b[a-z]:[\\/][^\r\n\t\"'<>|]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\r\n\t\"'<>|]+")
_POSIX_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?<!\w)/(?:mnt|home|Users|tmp|var/tmp|private/tmp)/[^\r\n\t\"'<>|]+"
)
_RAW_EXPORT_FILENAME_RE = re.compile(
    r"(?i)\b[a-z0-9_]+__[^\\/]+__export-\d{8}\.(?:csv|txt)\b"
)
_STOCK_CODE_RE = re.compile(
    r"(?<![.\d])(?:[036]\d{5}|688\d{3})(?:\.(?:SZ|SH|BJ))?(?!\d)",
    re.I,
)
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:token|secret|credential|password|api[_-]?key)\s*[:=]\s*"
    r"[A-Za-z0-9._~+/=-]{8,}"
)
_ROW_SAMPLE_RE = re.compile(r"(?i)\b(?:row_sample|row samples?|samples?)\b")
_SOURCE_ASSET_RE = re.compile(r"\bsource_asset_id\b")


def build_validation_summary(
    *,
    run_id: str,
    output_root: Path,
    dataset_manifest_paths: tuple[Path, ...],
    layer_manifest_paths: tuple[Path, ...],
    generated_text_paths: tuple[Path, ...],
) -> dict[str, Any]:
    dataset_failures: list[str] = []
    null_failures: list[str] = []
    duplicate_failures: list[str] = []
    non_finite_failures: list[str] = []
    source_local_leaks: list[str] = []
    date_ranges: list[dict[str, object]] = []
    factor_nan_profile: dict[str, int] = {}
    label_nan_profile: dict[str, int] = {}
    missing_dataset_manifests = [
        _logical_path(path) for path in dataset_manifest_paths if not path.is_file()
    ]
    for path in dataset_manifest_paths:
        if not path.is_file():
            continue
        manifest = read_manifest(path)
        dataset_id = str(manifest.get("dataset_id") or _logical_path(path))
        if str(manifest.get("validation_status", "")).lower() not in {"pass", "ok"}:
            dataset_failures.append(dataset_id)
        null_summary = _int_mapping(manifest.get("null_count_summary"))
        key_columns = tuple(str(item) for item in manifest.get("primary_key") or ())
        if any(null_summary.get(column, 0) > 0 for column in key_columns):
            null_failures.append(dataset_id)
        if int(manifest.get("duplicate_primary_key_count") or 0) > 0:
            duplicate_failures.append(dataset_id)
        non_finite = _int_mapping(manifest.get("non_finite_count_summary"))
        if any(count > 0 for count in non_finite.values()):
            non_finite_failures.append(dataset_id)
        if str(manifest).find("source_asset_id") >= 0 and str(
            manifest.get("logical_layer", "")
        ) in {"processed", "research"}:
            source_local_leaks.append(dataset_id)
        date_ranges.append(
            {
                "dataset_id": dataset_id,
                "date_min": manifest.get("date_min"),
                "date_max": manifest.get("date_max"),
                "partition_count": manifest.get("partition_count"),
            }
        )
        _merge_profile(factor_nan_profile, manifest.get("factor_nan_profile"))
        _merge_profile(label_nan_profile, manifest.get("label_nan_profile"))

    layer_failures: list[str] = []
    missing_layer_manifests = [
        _logical_path(path) for path in layer_manifest_paths if not path.is_file()
    ]
    for path in layer_manifest_paths:
        if not path.is_file():
            continue
        manifest = read_manifest(path)
        if manifest.get("layer_validation_status") != "pass":
            layer_failures.append(str(manifest.get("logical_layer") or path.name))

    safety = _private_safety_scan(generated_text_paths)
    failure_codes: list[str] = []
    if missing_dataset_manifests:
        failure_codes.append("dataset_manifest_missing")
    if dataset_failures:
        failure_codes.append("dataset_validation_failed")
    if null_failures:
        failure_codes.append("primary_key_nulls")
    if duplicate_failures:
        failure_codes.append("duplicate_primary_key")
    if non_finite_failures:
        failure_codes.append("non_finite_values")
    if source_local_leaks:
        failure_codes.append("source_local_key_leakage")
    if missing_layer_manifests:
        failure_codes.append("layer_manifest_missing")
    if layer_failures:
        failure_codes.append("layer_validation_failed")
    if safety["status"] != "PASS":
        failure_codes.append("private_safety_scan_failed")
    return {
        "run_id": run_id,
        "output_root": _logical_path(output_root),
        "status": "PASS" if not failure_codes else "FAIL",
        "failure_codes": failure_codes,
        "required_columns_status": "summarized_by_dataset_manifests",
        "primary_key_null_failures": null_failures,
        "duplicate_primary_key_failures": duplicate_failures,
        "source_local_key_leakage": source_local_leaks,
        "date_ranges": date_ranges,
        "partition_coverage_status": (
            "PASS" if not missing_dataset_manifests else "FAIL"
        ),
        "non_finite_failures": non_finite_failures,
        "factor_nan_profile": factor_nan_profile,
        "label_nan_profile": label_nan_profile,
        "dataset_manifest_completeness": {
            "missing": missing_dataset_manifests,
            "checked_count": len(dataset_manifest_paths),
        },
        "layer_manifest_completeness": {
            "missing": missing_layer_manifests,
            "checked_count": len(layer_manifest_paths),
        },
        "private_safety_scan": safety,
        "created_at": _utc_now(),
    }


def write_validation_summary(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def build_private_evidence_summary(
    *,
    run_id: str,
    date_range: dict[str, str],
    manifests: dict[str, object],
    timings: dict[str, object],
    validation_summary: dict[str, object],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": validation_summary.get("status", "FAIL"),
        "date_range": date_range,
        "output_root": _logical_path(output_root),
        "manifest_status": manifests,
        "timings": timings,
        "validation_summary": validation_summary,
        "privacy_level": "private_aggregate_only",
        "row_level_output_committed": False,
        "source_level_compare_engine": False,
        "created_at": _utc_now(),
    }


def write_private_evidence_summary_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def write_private_evidence_summary_md(path: Path, payload: dict[str, Any]) -> Path:
    lines = [
        "# Private Evidence Summary",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- status: `{payload['status']}`",
        f"- output_root: `{payload['output_root']}`",
        "- source_level_compare_engine: `false`",
        "",
    ]
    return atomic_write_text(path, "\n".join(lines), encoding="utf-8")


def write_run_summary_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def write_run_summary_md(path: Path, payload: dict[str, Any]) -> Path:
    text = "\n".join(
        [
            "# Private Evidence Run Summary",
            "",
            f"- run_id: `{payload['run_id']}`",
            f"- status: `{payload['status']}`",
            f"- output_root: `{payload['output_root']}`",
            f"- validation_status: `{payload['validation_summary']['status']}`",
            "- source_level_compare_engine: `false`",
            "",
        ]
    )
    return atomic_write_text(path, text, encoding="utf-8")


def _private_safety_scan(paths: tuple[Path, ...]) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        reason = _unsafe_json_reason(text) if path.suffix == ".json" else None
        if reason is None and path.suffix != ".json":
            reason = _unsafe_text_reason(text)
        if reason is not None:
            failures.append({"path": _logical_path(path), "reason": reason})
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_count": len(paths),
    }


def _unsafe_text_reason(text: str) -> str | None:
    if _WINDOWS_DRIVE_PATH_RE.search(text) or _UNC_PATH_RE.search(text):
        return "absolute_path"
    if _POSIX_PRIVATE_PATH_RE.search(text):
        return "absolute_path"
    if _RAW_EXPORT_FILENAME_RE.search(text):
        return "vendor_filename"
    if _ROW_SAMPLE_RE.search(text):
        return "row_sample"
    if _STOCK_CODE_RE.search(text):
        return "stock_sample"
    if _SOURCE_ASSET_RE.search(text):
        return "source_asset_id"
    if _CREDENTIAL_RE.search(text):
        return "credential"
    return None


def _unsafe_json_reason(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json"
    return _unsafe_json_value_reason(payload)


def _unsafe_json_value_reason(value: object, *, key: str | None = None) -> str | None:
    if key is not None:
        key_reason = _unsafe_text_reason(key)
        if key_reason is not None:
            return key_reason
    if isinstance(value, dict):
        for item_key, item in value.items():
            reason = _unsafe_json_value_reason(item, key=str(item_key))
            if reason is not None:
                return reason
        return None
    if isinstance(value, list | tuple):
        for item in value:
            reason = _unsafe_json_value_reason(item)
            if reason is not None:
                return reason
        return None
    if isinstance(value, str):
        return _unsafe_text_reason(value)
    return None


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item or 0) for key, item in value.items()}


def _merge_profile(target: dict[str, int], value: object) -> None:
    for key, item in _int_mapping(value).items():
        target[key] = target.get(key, 0) + item


def _logical_path(path: Path) -> str:
    parts = Path(path).parts
    for marker in (".run", "data", "manifests", "validations", "timings"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :]).as_posix()
    return Path(path).as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "build_private_evidence_summary",
    "build_validation_summary",
    "write_private_evidence_summary_json",
    "write_private_evidence_summary_md",
    "write_run_summary_json",
    "write_run_summary_md",
    "write_validation_summary",
]
