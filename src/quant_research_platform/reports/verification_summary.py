"""Local verification run summary builders and renderers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_research_platform.policy.share_safety import (
    contains_source_local_key,
    evidence_json_unsafe_reason,
    evidence_text_unsafe_reason,
)
from quant_research_platform.runtime.logical_paths import (
    RUN_RESULT_LOGICAL_PATH_MARKERS,
    dataset_layer_from_path,
    logical_path,
)
from quant_research_platform.storage.datasets import inspect_dataset


def build_validation_summary(
    *,
    run_id: str,
    dataset_roots: tuple[Path, ...],
    layer_summaries: tuple[dict[str, object], ...],
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
            missing_dataset_manifests.append(
                logical_path(root, markers=RUN_RESULT_LOGICAL_PATH_MARKERS)
            )
            continue
        manifest = inspection.manifest
        if not manifest:
            missing_dataset_manifests.append(
                logical_path(root, markers=RUN_RESULT_LOGICAL_PATH_MARKERS)
            )
            continue
        data_id = str(
            manifest.get("data_id")
            or logical_path(root, markers=RUN_RESULT_LOGICAL_PATH_MARKERS)
        )
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
        if dataset_layer_from_path(root) in {
            "processed",
            "research",
        } and contains_source_local_key(manifest):
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
    for summary in layer_summaries:
        if summary.get("layer_validation_status") != "pass":
            layer_failures.append(str(summary.get("logical_layer") or "unknown"))

    safety = _share_safety_scan(generated_text_paths)
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
    if layer_failures:
        failure_codes.append("layer_validation_failed")
    if safety["status"] != "PASS":
        failure_codes.append("share_safety_scan_failed")
    return {
        "run_id": run_id,
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
        "data_layer_summary_completeness": {
            "missing": [],
            "checked_count": len(layer_summaries),
        },
        "share_safety_scan": safety,
        "created_at": _utc_now(),
    }


def render_run_summary_markdown(payload: dict[str, Any]) -> str:
    status = payload["status"]
    lines = [
        "# Local Verification Run Summary",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- overall: `{status['overall']}`",
        f"- validation: `{status['validation']}`",
        f"- smoke: `{status['smoke']}`",
        f"- compare: `{status['compare']}`",
        f"- share_safety: `{status['share_safety']}`",
        f"- full_horizon: `{status['full_horizon']}`",
        f"- validation_matrix: `{status['validation_matrix']}`",
        f"- data_layers: `{len(payload['data']['layers'])}`",
        f"- outputs: `{len(payload['outputs'])}`",
        "",
    ]
    return "\n".join(lines)


def render_review_safe_aggregate_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the review-safe aggregate evidence payload as Markdown."""

    lines = [
        "# QRP Review-Safe Aggregate Evidence",
        "",
        (
            "This evidence records review-safe aggregate validation for the "
            "current QRP research contract. Review identifiers are "
            "`artifact_id=review_safe_aggregate_evidence`, "
            "`market_id=a_share`, and contract version `0.1`."
        ),
        "",
        (
            "Machine-readable evidence is generated by the current local "
            "evidence workflow and should remain review-safe."
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
            "Review-safe status:",
            "",
            f"- `output_scope | {payload['output_scope']}`",
            f"- `schema_id | {payload['schema_id']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `verification_scope | {payload['verification_scope']}`",
            f"- `evidence_generation_mode | {payload['evidence_generation_mode']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `handoff_ready | {str(payload['handoff_ready']).lower()}`",
            ("- local aggregate run completed without committed row-level outputs"),
        ]
    )
    if not payload["handoff_ready"]:
        lines.append("- human approval required before final review status")
    return "\n".join(lines) + "\n"


def render_local_verification_inventory_markdown(payload: dict[str, Any]) -> str:
    """Render the local dataset inventory summary as review-safe Markdown."""

    lines = [
        "# QRP Local Verification Inventory",
        "",
        (
            "This summary records aggregate-only sealed dataset inventory "
            "for the current QRP research contract."
        ),
        "",
        (
            "Machine-readable summary is generated by the current local "
            "evidence workflow and should remain review-safe."
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
            "Review-safe status:",
            "",
            f"- `output_scope | {payload['output_scope']}`",
            f"- `schema_id | {payload['schema_id']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `covered_roots | {len(payload['covered_roots'])}`",
            f"- `uncovered_roots | {len(payload['uncovered_roots'])}`",
            (
                "- `allowed_unverified_roots | "
                f"{len(payload['allowed_unverified_roots'])}`"
            ),
            f"- `handoff_ready | {str(payload['handoff_ready']).lower()}`",
        ]
    )
    if payload["uncovered_roots"]:
        lines.append("- uncovered dataset roots require review before final status")
    return "\n".join(lines) + "\n"


def _share_safety_scan(paths: tuple[Path, ...]) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    existing_paths = tuple(path for path in paths if path.is_file())
    for path in existing_paths:
        text = path.read_text(encoding="utf-8")
        reason = evidence_json_unsafe_reason(text) if path.suffix == ".json" else None
        if reason is None and path.suffix != ".json":
            reason = evidence_text_unsafe_reason(text)
        if reason is not None:
            failures.append(
                {
                    "path": logical_path(
                        path,
                    markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
                    ),
                    "reason": reason,
                }
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_count": len(existing_paths),
        "requested_count": len(paths),
        "existing_scanned_count": len(existing_paths),
        "missing_count": len(paths) - len(existing_paths),
    }


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item or 0) for key, item in value.items()}


def _merge_profile(target: dict[str, int], value: object) -> None:
    for key, item in _int_mapping(value).items():
        target[key] = target.get(key, 0) + item


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "build_validation_summary",
    "render_local_verification_inventory_markdown",
    "render_review_safe_aggregate_evidence_markdown",
    "render_run_summary_markdown",
]
