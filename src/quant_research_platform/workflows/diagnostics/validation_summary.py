"""Workflow-owned validation summary payload builder."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from quant_research_platform.contracts._json import (
    canonical_identity_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.policy.check_status import require_status
from quant_research_platform.policy.share_safety import (
    contains_source_local_key,
    evidence_json_value_unsafe_reason,
    evidence_text_unsafe_reason,
)
from quant_research_platform.runtime.logical_paths import (
    RUN_RESULT_DATASET_LAYERS,
    RUN_RESULT_LOGICAL_PATH_MARKERS,
    dataset_layer_from_path,
    logical_path,
)
from quant_research_platform.storage._dataset_schema import (
    validate_dataset_manifest_consumable,
    validate_dataset_primary_key_null_summary,
)
from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

_LAYER_SUMMARY_KEYS = frozenset(
    {
        "logical_layer",
        "dataset_count",
        "datasets",
        "total_rows",
        "date_min",
        "date_max",
        "missing_dataset_manifests",
        "invalid_dataset_manifests",
        "failed_dataset_validations",
        "layer_validation_status",
        "created_at",
    }
)


def build_validation_summary(
    *,
    run_id: str,
    dataset_roots: tuple[Path, ...],
    layer_summaries: tuple[dict[str, object], ...],
    generated_text_paths: tuple[Path, ...],
) -> dict[str, Any]:
    if not dataset_roots:
        raise ValueError("validation_summary_dataset_roots_required")
    if not layer_summaries:
        raise ValueError("validation_summary_layer_summaries_required")
    if not generated_text_paths:
        raise ValueError("validation_summary_generated_evidence_required")
    if len(set(dataset_roots)) != len(dataset_roots) or len(
        set(generated_text_paths)
    ) != len(generated_text_paths):
        raise ValueError("validation_summary_declaration_duplicate")
    dataset_failures: list[str] = []
    null_failures: list[str] = []
    duplicate_failures: list[str] = []
    non_finite_failures: list[str] = []
    source_local_leaks: list[str] = []
    date_ranges: list[dict[str, object]] = []
    factor_nan_profile: dict[str, int] = {}
    label_nan_profile: dict[str, int] = {}
    missing_dataset_manifests: list[str] = []
    invalid_dataset_manifests: list[str] = []
    observed_layer_datasets: dict[str, list[dict[str, object]]] = {}
    observed_layer_missing: dict[str, list[str]] = {}
    observed_layer_invalid: dict[str, list[str]] = {}
    for root in dataset_roots:
        logical_layer = dataset_layer_from_path(root)
        if logical_layer not in RUN_RESULT_DATASET_LAYERS:
            raise ValueError("validation_summary_dataset_layer_invalid")
        observed_layer_datasets.setdefault(logical_layer, [])
        observed_layer_missing.setdefault(logical_layer, [])
        observed_layer_invalid.setdefault(logical_layer, [])
        manifest_logical_path = logical_path(
            root / "_manifest" / "manifest.json",
            markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
        )
        inspection_failed = False
        manifest_summary_mismatch = False
        try:
            inspection = inspect_dataset(root)
        except (OSError, ValueError) as exc:
            manifest_summary_mismatch = (
                type(exc) is ValueError
                and str(exc) == "dataset_manifest_summary_mismatch"
            )
            try:
                manifest = read_dataset_manifest_evidence(root)
            except FileNotFoundError:
                missing_dataset_manifests.append(manifest_logical_path)
                observed_layer_missing[logical_layer].append(manifest_logical_path)
            except (OSError, ValueError):
                invalid_dataset_manifests.append(manifest_logical_path)
                observed_layer_invalid[logical_layer].append(manifest_logical_path)
            else:
                inspection_failed = True
            if not inspection_failed:
                continue
        else:
            manifest = inspection.manifest
        if not manifest:
            invalid_dataset_manifests.append(manifest_logical_path)
            observed_layer_invalid[logical_layer].append(manifest_logical_path)
            continue
        data_id = manifest["data_id"]
        try:
            validate_dataset_manifest_consumable(manifest)
            manifest_consumable = True
        except ValueError:
            manifest_consumable = False
        if inspection_failed and not manifest_summary_mismatch:
            raise ValueError("validation_summary_dataset_inspection_failed") from None
        if require_status(manifest.get("validation_status")) != "PASS":
            dataset_failures.append(data_id)
        # This existing failure class covers an invalid required PK-null fact;
        # an absent fact is never interpreted as an observed zero count.
        try:
            validate_dataset_primary_key_null_summary(manifest)
        except ValueError:
            null_failures.append(data_id)
        if manifest["duplicate_primary_key_count"] > 0:
            duplicate_failures.append(data_id)
        non_finite = cast(
            Mapping[str, int],
            manifest.get("non_finite_count_summary"),
        )
        if any(count > 0 for count in non_finite.values()):
            non_finite_failures.append(data_id)
        if not manifest_consumable and data_id not in dataset_failures:
            dataset_failures.append(data_id)
        effective_validation_status = "PASS" if manifest_consumable else "FAIL"
        observed_layer_datasets[logical_layer].append(
            {
                "data_id": data_id,
                "manifest_path": manifest_logical_path,
                "manifest_summary": {
                    "row_count": manifest["row_count"],
                    "date_min": manifest.get("date_min"),
                    "date_max": manifest.get("date_max"),
                    "partition_count": manifest["partition_count"],
                    "duplicate_primary_key_count": manifest[
                        "duplicate_primary_key_count"
                    ],
                    "validation_status": effective_validation_status,
                },
            }
        )
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

    (
        layer_failures,
        declared_missing_manifests,
        declared_invalid_manifests,
        declared_failed_datasets,
    ) = _validate_layer_summary_graph(
        layer_summaries,
        observed_layer_datasets=observed_layer_datasets,
        observed_layer_missing=observed_layer_missing,
        observed_layer_invalid=observed_layer_invalid,
    )
    if (
        missing_dataset_manifests != declared_missing_manifests
        or invalid_dataset_manifests != declared_invalid_manifests
        or dataset_failures != declared_failed_datasets
    ):
        raise ValueError("validation_summary_declaration_mismatch")

    safety = _share_safety_scan(generated_text_paths)
    failure_codes: list[str] = []
    if missing_dataset_manifests:
        failure_codes.append("dataset_manifest_missing")
    if invalid_dataset_manifests:
        failure_codes.append("dataset_manifest_invalid")
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
            "PASS"
            if not missing_dataset_manifests and not invalid_dataset_manifests
            else "FAIL"
        ),
        "non_finite_failures": non_finite_failures,
        "factor_nan_profile": factor_nan_profile,
        "label_nan_profile": label_nan_profile,
        "dataset_manifest_completeness": {
            "missing": missing_dataset_manifests,
            "invalid": invalid_dataset_manifests,
            "checked_count": len(dataset_roots),
        },
        "data_layer_summary_completeness": {
            "missing": [],
            "checked_count": len(layer_summaries),
        },
        "share_safety_scan": safety,
        "created_at": _utc_now(),
    }


def _share_safety_scan(paths: tuple[Path, ...]) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    existing_count = 0
    requested_paths = [
        logical_path(path, markers=RUN_RESULT_LOGICAL_PATH_MARKERS) for path in paths
    ]
    if len(requested_paths) != len(set(requested_paths)):
        raise ValueError("validation_summary_evidence_path_duplicate")
    for path, requested_path in zip(paths, requested_paths, strict=True):
        reason: str | None = None
        if not path.is_file():
            reason = "missing_or_non_file"
        else:
            existing_count += 1
            try:
                raw = path.read_bytes()
            except OSError:
                reason = "unreadable"
            else:
                if path.suffix.casefold() == ".json":
                    try:
                        payload = parse_qrp_json_object(raw)
                    except ValueError:
                        reason = "invalid_json"
                    else:
                        reason = evidence_json_value_unsafe_reason(payload)
                else:
                    try:
                        text = raw.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        reason = "invalid_utf8"
                    else:
                        reason = evidence_text_unsafe_reason(text)
        if reason is not None:
            failures.append(
                {
                    "path": requested_path,
                    "reason": reason,
                }
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_count": existing_count,
        "requested_count": len(paths),
        "existing_scanned_count": existing_count,
        "missing_count": len(paths) - existing_count,
        "requested_paths": requested_paths,
    }


def _validate_layer_summary_graph(
    layer_summaries: tuple[dict[str, object], ...],
    *,
    observed_layer_datasets: dict[str, list[dict[str, object]]],
    observed_layer_missing: dict[str, list[str]],
    observed_layer_invalid: dict[str, list[str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    logical_layers = sorted(
        set(observed_layer_datasets)
        .union(observed_layer_missing)
        .union(observed_layer_invalid)
    )
    if len(layer_summaries) != len(logical_layers):
        raise ValueError("validation_summary_layer_declaration_mismatch")
    layer_failures: list[str] = []
    declared_missing: list[str] = []
    declared_invalid: list[str] = []
    declared_failed: list[str] = []
    for summary, logical_layer in zip(layer_summaries, logical_layers, strict=True):
        if type(summary) is not dict or set(summary) != _LAYER_SUMMARY_KEYS:
            raise ValueError("validation_summary_layer_invalid")
        if type(summary["created_at"]) is not str or not summary["created_at"]:
            raise ValueError("validation_summary_layer_invalid")
        require_status(summary["layer_validation_status"])
        datasets = sorted(
            observed_layer_datasets.get(logical_layer, []),
            key=lambda item: str(item["manifest_path"]),
        )
        missing = sorted(observed_layer_missing.get(logical_layer, []))
        invalid = sorted(observed_layer_invalid.get(logical_layer, []))
        failed = [
            str(item["data_id"])
            for item in datasets
            if item["manifest_summary"]["validation_status"] != "PASS"
        ]
        date_mins = [
            str(item["manifest_summary"]["date_min"])
            for item in datasets
            if item["manifest_summary"]["date_min"] is not None
        ]
        date_maxs = [
            str(item["manifest_summary"]["date_max"])
            for item in datasets
            if item["manifest_summary"]["date_max"] is not None
        ]
        expected = {
            "logical_layer": logical_layer,
            "dataset_count": len(datasets),
            "datasets": datasets,
            "total_rows": sum(
                int(item["manifest_summary"]["row_count"]) for item in datasets
            ),
            "date_min": min(date_mins) if date_mins else None,
            "date_max": max(date_maxs) if date_maxs else None,
            "missing_dataset_manifests": missing,
            "invalid_dataset_manifests": invalid,
            "failed_dataset_validations": failed,
            "layer_validation_status": (
                "PASS" if not missing and not invalid and not failed else "FAIL"
            ),
        }
        observed = {key: item for key, item in summary.items() if key != "created_at"}
        try:
            matches = canonical_identity_bytes(observed) == canonical_identity_bytes(
                expected
            )
        except ValueError as exc:
            raise ValueError("validation_summary_layer_invalid") from exc
        if not matches:
            raise ValueError("validation_summary_layer_declaration_mismatch")
        if expected["layer_validation_status"] != "PASS":
            layer_failures.append(logical_layer)
        declared_missing.extend(missing)
        declared_invalid.extend(invalid)
        declared_failed.extend(failed)
    return layer_failures, declared_missing, declared_invalid, declared_failed


def _merge_profile(target: dict[str, int], value: object) -> None:
    profile = cast(Mapping[str, int], value)
    for key, item in profile.items():
        target[key] = target.get(key, 0) + item


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = ["build_validation_summary"]
