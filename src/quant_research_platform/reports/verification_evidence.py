"""Aggregate verification evidence report writers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text
from quant_research_platform.storage.datasets import inspect_dataset
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
_SMOKE_STATUSES = frozenset({"NOT_RUN", "PASS", "FAIL"})
_COMPARE_STATUSES = frozenset({"NOT_APPLICABLE", "PASS", "FAIL"})
_FULL_HORIZON_STATUSES = frozenset({"NOT_RUN", "PASS", "FAIL"})
_COMPARE_EVIDENCE_FIELDS = frozenset({"comparison_path", "verification_scope"})
_FULL_HORIZON_EVIDENCE_FIELDS = frozenset({"evidence_path", "verification_scope"})


@dataclass(frozen=True)
class VerificationEvidenceLayout:
    output_root: Path

    @classmethod
    def from_output_root(cls, output_root: Path) -> VerificationEvidenceLayout:
        return cls(output_root=Path(output_root))

    @property
    def config_root(self) -> Path:
        return self.output_root / "config"

    @property
    def data_root(self) -> Path:
        return self.output_root / "data"

    @property
    def interim_root(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def research_root(self) -> Path:
        return self.data_root / "research"

    @property
    def evidence_root(self) -> Path:
        return self.data_root / "evidence"

    @property
    def manifests_root(self) -> Path:
        return self.output_root / "manifests"

    @property
    def validations_root(self) -> Path:
        return self.output_root / "validations"

    @property
    def timings_root(self) -> Path:
        return self.output_root / "timings"

    @property
    def logs_root(self) -> Path:
        return self.output_root / "logs"

    @property
    def summary_json_path(self) -> Path:
        return self.output_root / "summary.json"

    @property
    def summary_md_path(self) -> Path:
        return self.output_root / "summary.md"


def build_validation_summary(
    *,
    run_id: str,
    output_root: Path,
    dataset_roots: tuple[Path, ...],
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
    missing_dataset_manifests: list[str] = []
    for root in dataset_roots:
        try:
            inspection = inspect_dataset(root)
        except ValueError:
            missing_dataset_manifests.append(_logical_path(root))
            continue
        manifest = inspection.manifest
        if not manifest:
            missing_dataset_manifests.append(_logical_path(root))
            continue
        data_id = str(manifest.get("data_id") or _logical_path(root))
        validation_status = manifest.get("validation_status")
        if validation_status is not None and str(validation_status).lower() not in {
            "pass",
            "ok",
        }:
            dataset_failures.append(data_id)
        null_summary = _int_mapping(manifest.get("null_count_summary"))
        key_columns = tuple(str(item) for item in manifest.get("primary_key") or ())
        if any(null_summary.get(column, 0) > 0 for column in key_columns):
            null_failures.append(data_id)
        if int(manifest.get("duplicate_primary_key_count") or 0) > 0:
            duplicate_failures.append(data_id)
        non_finite = _int_mapping(manifest.get("non_finite_count_summary"))
        if any(count > 0 for count in non_finite.values()):
            non_finite_failures.append(data_id)
        if str(manifest).find("source_asset_id") >= 0 and str(
            manifest.get("logical_layer", "")
        ) in {"processed", "research"}:
            source_local_leaks.append(data_id)
        date_ranges.append(
            {
                "data_id": data_id,
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
            "checked_count": len(dataset_roots),
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


def build_verification_evidence_summary(
    *,
    run_id: str,
    date_range: dict[str, str],
    manifests: dict[str, object],
    timings: dict[str, object],
    validation_summary: dict[str, object],
    output_root: Path,
    smoke_status: str = "NOT_RUN",
    compare_status: str = "NOT_APPLICABLE",
    full_horizon_status: str = "NOT_RUN",
    compare_evidence: dict[str, object] | None = None,
    full_horizon_evidence: dict[str, object] | None = None,
) -> dict[str, Any]:
    statuses = _validate_verification_statuses(
        smoke_status=smoke_status,
        compare_status=compare_status,
        full_horizon_status=full_horizon_status,
        compare_evidence=compare_evidence,
        full_horizon_evidence=full_horizon_evidence,
    )
    return {
        "run_id": run_id,
        "status": _overall_verification_status(
            validation_summary.get("status", "FAIL"),
            statuses,
        ),
        "date_range": date_range,
        "output_root": _logical_path(output_root),
        **statuses,
        "manifest_status": manifests,
        "timings": timings,
        "validation_summary": validation_summary,
        "privacy_level": "private_aggregate_only",
        "row_level_output_committed": False,
        "source_level_compare_engine": False,
        "created_at": _utc_now(),
    }


def write_verification_evidence_summary_json(
    path: Path,
    payload: dict[str, Any],
) -> Path:
    return atomic_write_json(path, payload)


def write_verification_evidence_summary_md(
    path: Path,
    payload: dict[str, Any],
) -> Path:
    lines = [
        "# Verification Evidence Summary",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- status: `{payload['status']}`",
        f"- output_root: `{payload['output_root']}`",
        f"- smoke_status: `{payload['smoke_status']}`",
        f"- compare_status: `{payload['compare_status']}`",
        f"- full_horizon_status: `{payload['full_horizon_status']}`",
        "- source_level_compare_engine: `false`",
        "",
    ]
    return atomic_write_text(path, "\n".join(lines), encoding="utf-8")


def write_run_summary_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def write_run_summary_md(path: Path, payload: dict[str, Any]) -> Path:
    text = "\n".join(
        [
            "# Verification Evidence Run Summary",
            "",
            f"- run_id: `{payload['run_id']}`",
            f"- status: `{payload['status']}`",
            f"- output_root: `{payload['output_root']}`",
            f"- validation_status: `{payload['validation_summary']['status']}`",
            f"- smoke_status: `{payload['smoke_status']}`",
            f"- compare_status: `{payload['compare_status']}`",
            f"- full_horizon_status: `{payload['full_horizon_status']}`",
            "- source_level_compare_engine: `false`",
            "",
        ]
    )
    return atomic_write_text(path, text, encoding="utf-8")


def render_public_safe_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the public-safe evidence payload as Markdown."""

    lines = [
        "# Clean Rewrite 0.1 Public-Safe Parity Evidence",
        "",
        (
            "This evidence records public-safe aggregate validation for QRP "
            "Clean Rewrite `0.1.0`. Public identifiers are "
            "`artifact_id=a_share_private_aggregate_evidence`, "
            "`market_id=a_share`, and contract version `0.1`."
        ),
        "",
        (
            "Machine-readable evidence lives in "
            "`docs/reports/clean-rewrite-0.1-parity-evidence.json`."
        ),
        "",
        "| Area | Rows | Columns | Period | Status |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for area in payload["areas"]:
        status = (
            "human approval required"
            if area["human_approval_required"]
            else "verified aggregate"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(area["area"]).replace("_", " "),
                    str(area["row_count"]),
                    str(area["column_count"]),
                    str(area["period"]),
                    status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Public-safe status:",
            "",
            f"- `privacy_level | {payload['privacy_level']}`",
            f"- `contract_version | {payload['contract_version']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `verification_scope | {payload['verification_scope']}`",
            f"- `evidence_generation_mode | {payload['evidence_generation_mode']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `merge_ready | {str(payload['merge_ready']).lower()}`",
            (
                "- private local aggregate run completed without committed "
                "row-level artifacts"
            ),
        ]
    )
    if not payload["merge_ready"]:
        lines.append("- human approval required before merge-ready status")
    return "\n".join(lines) + "\n"


def render_private_verification_summary_markdown(payload: dict[str, Any]) -> str:
    """Render the sealed dataset inventory summary as public-safe Markdown."""

    lines = [
        "# Clean Rewrite 0.1 Private Verification Summary",
        "",
        (
            "This summary records aggregate-only sealed dataset inventory "
            "for QRP Clean Rewrite `0.1.0`."
        ),
        "",
        (
            "Machine-readable summary lives in "
            "`docs/reports/clean-rewrite-0.1-private-verification-summary.json`."
        ),
        "",
        "| Logical Root | Area | Rows | Columns | Period |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in payload["dataset_inventory"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["logical_root"],
                    item["covered_by_evidence_area"] or "uncovered",
                    str(item["row_count"]),
                    str(item["column_count"]),
                    str(item["period"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Public-safe status:",
            "",
            f"- `privacy_level | {payload['privacy_level']}`",
            f"- `contract_version | {payload['contract_version']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `covered_roots | {len(payload['covered_roots'])}`",
            f"- `uncovered_roots | {len(payload['uncovered_roots'])}`",
            (
                "- `allowed_unverified_roots | "
                f"{len(payload['allowed_unverified_roots'])}`"
            ),
            f"- `merge_ready | {str(payload['merge_ready']).lower()}`",
        ]
    )
    if payload["uncovered_roots"]:
        lines.append("- uncovered dataset roots require review before merge-ready")
    return "\n".join(lines) + "\n"


def _private_safety_scan(paths: tuple[Path, ...]) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    existing_paths = tuple(path for path in paths if path.is_file())
    for path in existing_paths:
        text = path.read_text(encoding="utf-8")
        reason = _unsafe_json_reason(text) if path.suffix == ".json" else None
        if reason is None and path.suffix != ".json":
            reason = _unsafe_text_reason(text)
        if reason is not None:
            failures.append({"path": _logical_path(path), "reason": reason})
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_count": len(existing_paths),
        "requested_count": len(paths),
        "existing_scanned_count": len(existing_paths),
        "missing_count": len(paths) - len(existing_paths),
    }


def _validate_verification_statuses(
    *,
    smoke_status: str,
    compare_status: str,
    full_horizon_status: str,
    compare_evidence: dict[str, object] | None,
    full_horizon_evidence: dict[str, object] | None,
) -> dict[str, object]:
    smoke = smoke_status.upper()
    compare = compare_status.upper()
    full_horizon = full_horizon_status.upper()
    if smoke not in _SMOKE_STATUSES:
        raise ValueError("invalid_smoke_status")
    if compare not in _COMPARE_STATUSES:
        raise ValueError("invalid_compare_status")
    if full_horizon not in _FULL_HORIZON_STATUSES:
        raise ValueError("invalid_full_horizon_status")
    if smoke == "NOT_RUN" and compare != "NOT_APPLICABLE":
        raise ValueError("compare_status_requires_smoke")
    if smoke != "NOT_RUN" and compare == "NOT_APPLICABLE":
        raise ValueError("smoke_status_requires_compare_status")
    if smoke == "FAIL" and compare == "PASS":
        raise ValueError("compare_pass_requires_smoke_pass")
    if compare in {"PASS", "FAIL"} and not compare_evidence:
        raise ValueError("compare_evidence_required")
    if compare == "NOT_APPLICABLE" and compare_evidence:
        raise ValueError("compare_evidence_not_applicable")
    if compare in {"PASS", "FAIL"}:
        _require_evidence_fields(
            compare_evidence,
            required_fields=_COMPARE_EVIDENCE_FIELDS,
            error_name="compare_evidence_missing_fields",
        )
    if full_horizon == "NOT_RUN" and full_horizon_evidence:
        raise ValueError("full_horizon_evidence_not_applicable")
    if full_horizon != "NOT_RUN" and not full_horizon_evidence:
        raise ValueError("full_horizon_evidence_required")
    if full_horizon != "NOT_RUN":
        _require_evidence_fields(
            full_horizon_evidence,
            required_fields=_FULL_HORIZON_EVIDENCE_FIELDS,
            error_name="full_horizon_evidence_missing_fields",
        )
    return {
        "smoke_status": smoke,
        "compare_status": compare,
        "full_horizon_status": full_horizon,
        "compare_evidence": dict(compare_evidence or {}),
        "full_horizon_evidence": dict(full_horizon_evidence or {}),
    }


def _overall_verification_status(
    validation_status: object,
    statuses: dict[str, object],
) -> str:
    if str(validation_status).upper() != "PASS":
        return "FAIL"
    if statuses["smoke_status"] == "FAIL":
        return "FAIL"
    if statuses["compare_status"] == "FAIL":
        return "FAIL"
    if statuses["full_horizon_status"] == "FAIL":
        return "FAIL"
    return "PASS"


def _require_evidence_fields(
    evidence: dict[str, object] | None,
    *,
    required_fields: frozenset[str],
    error_name: str,
) -> None:
    payload = evidence or {}
    missing = [
        field
        for field in sorted(required_fields)
        if not str(payload.get(field, "")).strip()
    ]
    if missing:
        raise ValueError(error_name)


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
    item = Path(path)
    parts = item.parts
    for marker in ("data", "manifests", "validations", "timings"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :]).as_posix()
    if item.is_absolute():
        return item.name
    return item.as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "VerificationEvidenceLayout",
    "build_verification_evidence_summary",
    "build_validation_summary",
    "render_private_verification_summary_markdown",
    "render_public_safe_evidence_markdown",
    "write_verification_evidence_summary_json",
    "write_verification_evidence_summary_md",
    "write_run_summary_json",
    "write_run_summary_md",
    "write_validation_summary",
]
