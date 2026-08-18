"""Workflow-owned validation summary payload builder."""

from __future__ import annotations

import io
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, cast

from quant_research_platform.artifacts._output_bundle_validation import (
    validate_output_bundle,
)
from quant_research_platform.contracts._json import (
    bounded_rendered_qrp_json_object_size,
    canonical_identity_bytes,
)
from quant_research_platform.policy.check_status import require_status
from quant_research_platform.policy.share_safety import (
    contains_source_local_key,
    evidence_json_value_unsafe_reason,
    evidence_text_stream_unsafe_reason,
)
from quant_research_platform.runtime.logical_paths import (
    RUN_RESULT_DATASET_LAYERS,
    RUN_RESULT_LOGICAL_PATH_MARKERS,
    dataset_layer_from_path,
    logical_path,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_inspection import (
    dataset_inspection_retained_memory_bytes,
    inspect_dataset,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    dataset_file_identity,
    dataset_file_identity_from_stat,
)
from quant_research_platform.storage._dataset_schema import (
    validate_dataset_manifest_consumable,
    validate_dataset_primary_key_null_summary,
)
from quant_research_platform.storage._dataset_types import DatasetInspection
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
    validated_output_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
    _retained_state: ExitStack | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, Any]:
    with (
        runtime_resource_policy_scope(resource_policy) as policy,
        ExitStack() as local_state,
    ):
        return _build_validation_summary(
            run_id=run_id,
            dataset_roots=dataset_roots,
            layer_summaries=layer_summaries,
            validated_output_members=validated_output_members,
            output_bundle_expectations=output_bundle_expectations,
            resource_policy=policy,
            retained_state=_retained_state or local_state,
        )


def _build_validation_summary(
    *,
    run_id: str,
    dataset_roots: tuple[Path, ...],
    layer_summaries: tuple[dict[str, object], ...],
    validated_output_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
    resource_policy: RuntimeResourcePolicy,
    retained_state: ExitStack,
) -> dict[str, Any]:
    if not dataset_roots:
        raise ValueError("validation_summary_dataset_roots_required")
    if not layer_summaries:
        raise ValueError("validation_summary_layer_summaries_required")
    if not validated_output_members:
        raise ValueError("validation_summary_generated_evidence_required")
    if not output_bundle_expectations:
        raise ValueError("validation_summary_output_bundle_expectations_required")
    if len(set(dataset_roots)) != len(dataset_roots) or len(
        {path for path, _identity in validated_output_members}
    ) != len(validated_output_members):
        raise ValueError("validation_summary_declaration_duplicate")
    if len({root for root, *_rest in output_bundle_expectations}) != len(
        output_bundle_expectations
    ):
        raise ValueError("validation_summary_output_bundle_declaration_duplicate")
    inspection_slot_bytes = sys.getsizeof([None] * len(dataset_roots))
    inspection_slot_bytes += len(dataset_roots) * sys.getsizeof((None, None, None))
    if inspection_slot_bytes > resource_policy.remaining_memory_bytes:
        raise ValueError("validation_summary_payload_exceeds_task_memory_budget")
    inspection_results: list[
        tuple[Path, DatasetInspection | None, str | None] | None
    ] = [None] * len(dataset_roots)
    retained_inspection_bytes = inspection_slot_bytes
    for index, root in enumerate(dataset_roots):
        logical_layer = dataset_layer_from_path(root)
        if logical_layer not in RUN_RESULT_DATASET_LAYERS:
            raise ValueError("validation_summary_dataset_layer_invalid")
        try:
            inspection = inspect_dataset(
                root,
                resource_policy=resource_policy,
                retained_payload_bytes=retained_inspection_bytes,
            )
        except (OSError, ValueError):
            try:
                manifest = read_dataset_manifest_evidence(root)
            except FileNotFoundError:
                inspection_results[index] = (root, None, "missing")
            except (OSError, ValueError):
                inspection_results[index] = (root, None, "invalid")
            else:
                if manifest:
                    raise ValueError(
                        "validation_summary_dataset_inspection_failed"
                    ) from None
                inspection_results[index] = (root, None, "invalid")
            continue
        inspection_results[index] = (root, inspection, None)
        retained_inspection_bytes += dataset_inspection_retained_memory_bytes(
            inspection
        )

    dataset_inspections = [
        (root, inspection)
        for result in inspection_results
        if result is not None
        for root, inspection, error in (result,)
        if error is None and inspection is not None
    ]
    summary_headroom = (
        resource_policy.remaining_memory_bytes - retained_inspection_bytes
    )
    if summary_headroom <= 0:
        raise ValueError("validation_summary_payload_exceeds_task_memory_budget")
    summary_bound_bytes = _validation_summary_graph_upper_bound_bytes(
        dataset_roots=dataset_roots,
        dataset_inspections=dataset_inspections,
        layer_summaries=layer_summaries,
        validated_output_members=validated_output_members,
        output_bundle_expectations=output_bundle_expectations,
        max_bytes=summary_headroom,
    )
    try:
        retained_state.enter_context(
            resource_policy.reserve(
                "workflow.validation.summary",
                retained_inspection_bytes + summary_bound_bytes,
            )
        )
    except MemoryError:
        raise ValueError(
            "validation_summary_payload_exceeds_task_memory_budget"
        ) from None

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
    for result in inspection_results:
        assert result is not None
        root, inspection, inspection_error = result
        logical_layer = dataset_layer_from_path(root)
        observed_layer_datasets.setdefault(logical_layer, [])
        observed_layer_missing.setdefault(logical_layer, [])
        observed_layer_invalid.setdefault(logical_layer, [])
        manifest_logical_path = logical_path(
            root / "_manifest" / "manifest.json",
            markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
        )
        if inspection_error is not None:
            failures = (
                missing_dataset_manifests
                if inspection_error == "missing"
                else invalid_dataset_manifests
            )
            observed = (
                observed_layer_missing
                if inspection_error == "missing"
                else observed_layer_invalid
            )
            failures.append(manifest_logical_path)
            observed[logical_layer].append(manifest_logical_path)
            continue
        assert inspection is not None
        manifest = inspection.manifest
        data_id = manifest["data_id"]
        try:
            validate_dataset_manifest_consumable(manifest)
            manifest_consumable = True
        except ValueError:
            manifest_consumable = False
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
        if logical_layer in {
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

    safety = _share_safety_scan(
        validated_output_members,
        output_bundle_expectations=output_bundle_expectations,
    )
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
    summary = {
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
    bounded_rendered_qrp_json_object_size(
        summary,
        lazy_arrays={},
        max_bytes=summary_bound_bytes,
    )
    for root, initial_inspection in dataset_inspections:
        try:
            with resource_policy.reserve(
                "workflow.validation.reinspection",
                dataset_inspection_retained_memory_bytes(initial_inspection),
            ):
                final_inspection = inspect_dataset(
                    root,
                    resource_policy=resource_policy,
                    retained_payload_bytes=0,
                )
        except MemoryError:
            raise ValueError(
                "validation_summary_payload_exceeds_task_memory_budget"
            ) from None
        except (OSError, ValueError):
            raise ValueError("validation_summary_dataset_identity_changed") from None
        if final_inspection != initial_inspection:
            raise ValueError("validation_summary_dataset_identity_changed")
        del final_inspection
    return summary


def _validation_summary_graph_upper_bound_bytes(
    *,
    dataset_roots: tuple[Path, ...],
    dataset_inspections: list[tuple[Path, DatasetInspection]],
    layer_summaries: tuple[dict[str, object], ...],
    validated_output_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
    max_bytes: int,
) -> int:
    manifests = tuple(inspection.manifest for _root, inspection in dataset_inspections)
    layer_bytes = 0
    for summary in layer_summaries:
        layer_bytes += bounded_rendered_qrp_json_object_size(
            summary,
            lazy_arrays={},
            max_bytes=max_bytes - layer_bytes,
        )
    text_characters = sum(len(os.fspath(root)) for root in dataset_roots)
    text_characters += len("/_manifest/manifest.json") * len(dataset_roots)
    text_characters += sum(
        len(os.fspath(path)) for path, _identity in validated_output_members
    )
    text_characters += sum(
        len(os.fspath(root)) + len(run_id) + len(output_id) + len(definition_id)
        for root, run_id, output_id, definition_id in output_bundle_expectations
    )
    dataset_count = len(dataset_roots)
    layer_count = len(layer_summaries)
    member_count = len(validated_output_members)
    profile_entries = sum(
        len(cast(Mapping[str, int], manifest.get(profile)))
        for manifest in manifests
        for profile in ("factor_nan_profile", "label_nan_profile")
    )
    mapping_count = 6 + 3 * dataset_count + layer_count + member_count
    mapping_entries = (
        27
        + 12 * dataset_count
        + 10 * layer_count
        + 2 * member_count
        + profile_entries
    )
    list_count = 14 + 3 * layer_count
    list_slots = (
        32 + 16 * dataset_count + 12 * layer_count + 4 * member_count
    )
    empty_mapping_bytes = sys.getsizeof({})
    mapping_entry_bytes = sys.getsizeof({"": None}) - empty_mapping_bytes
    empty_list_bytes = sys.getsizeof([])
    list_slot_bytes = sys.getsizeof([None]) - empty_list_bytes
    container_bytes = (
        mapping_count * empty_mapping_bytes
        + mapping_entries * mapping_entry_bytes
        + list_count * empty_list_bytes
        + list_slots * list_slot_bytes
    )
    text_bytes = text_characters * 4 + (
        len(dataset_roots) + len(validated_output_members)
    ) * sys.getsizeof("")
    scalar_size = max(sys.getsizeof(None), sys.getsizeof(0), sys.getsizeof(0.0))
    scalar_bytes = scalar_size * (27 + 10 * dataset_count + profile_entries)
    created_at_bytes = sys.getsizeof("0" * 32)
    total = (
        layer_bytes
        + text_bytes
        + container_bytes
        + scalar_bytes
        + created_at_bytes
    )
    if total > max_bytes:
        raise MemoryError("runtime_memory_budget_exceeded")
    return total


def _share_safety_scan(
    members: tuple[tuple[Path, DatasetFileIdentity], ...],
    *,
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
) -> dict[str, object]:
    final_members, failures, existing_count, missing_count = _validated_output_snapshot(
        members, output_bundle_expectations
    )
    scanned_members = tuple(
        member for member in final_members if member[0].suffix.casefold() != ".parquet"
    )
    requested_paths = [
        logical_path(path, markers=RUN_RESULT_LOGICAL_PATH_MARKERS)
        for path, _identity in scanned_members
    ]
    if len(requested_paths) != len(set(requested_paths)):
        raise ValueError("validation_summary_evidence_path_duplicate")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_count": existing_count,
        "requested_count": len(scanned_members),
        "existing_scanned_count": existing_count,
        "missing_count": missing_count,
        "requested_paths": requested_paths,
    }


def _validated_output_snapshot(
    expected_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    expectations: tuple[tuple[Path, str, str, str], ...],
) -> tuple[
    tuple[tuple[Path, DatasetFileIdentity], ...],
    list[dict[str, str]],
    int,
    int,
]:
    observed_all: list[tuple[Path, DatasetFileIdentity]] = []
    failures: list[dict[str, str]] = []
    existing_count = 0
    missing_count = 0
    for (
        root,
        expected_run_id,
        expected_output_id,
        expected_definition_id,
    ) in expectations:
        expected = tuple(
            member for member in expected_members if member[0].parent == root
        )
        manifest_path = root / "manifest.json"
        tentative_observed: tuple[tuple[Path, DatasetFileIdentity], ...] = ()
        tentative_failures: list[dict[str, str]] = []
        tentative_existing_count = 0

        def observe_admitted_bundle(
            payload: dict[str, object],
            observed: tuple[tuple[Path, DatasetFileIdentity], ...],
            expected_definition_id: str = expected_definition_id,
            expected: tuple[tuple[Path, DatasetFileIdentity], ...] = expected,
            manifest_path: Path = manifest_path,
            tentative_failures: list[dict[str, str]] = tentative_failures,
        ) -> None:
            nonlocal tentative_existing_count, tentative_observed
            if payload["output_definition_id"] != expected_definition_id:
                raise ValueError("output_manifest_definition_id_invalid")
            if observed != expected:
                raise ValueError("output_bundle_identity_changed")

            manifest_reason = evidence_json_value_unsafe_reason(payload)
            if manifest_reason is not None:
                tentative_failures.append(
                    {
                        "path": logical_path(
                            manifest_path,
                            markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
                        ),
                        "reason": manifest_reason,
                    }
                )
            scanned = tuple(
                member
                for member in observed
                if member[0].suffix.casefold() != ".parquet"
            )
            for path, identity in scanned:
                if path == manifest_path:
                    continue
                try:
                    reason = _scan_validated_text_evidence(path, identity)
                except UnicodeDecodeError:
                    reason = "invalid_utf8"
                except (OSError, ValueError):
                    reason = "identity_changed_or_unreadable"
                if reason is not None:
                    tentative_failures.append(
                        {
                            "path": logical_path(
                                path,
                                markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
                            ),
                            "reason": reason,
                        }
                    )
            tentative_observed = observed
            tentative_existing_count = len(scanned)

        try:
            validate_output_bundle(
                root,
                expected_run_id=expected_run_id,
                expected_output_id=expected_output_id,
                observer=observe_admitted_bundle,
            )
        except (OSError, ValueError):
            failures.append(
                {
                    "path": logical_path(
                        manifest_path, markers=RUN_RESULT_LOGICAL_PATH_MARKERS
                    ),
                    "reason": "identity_changed_or_unreadable",
                }
            )
            observed_all.extend(expected)
            missing_count += sum(
                path.suffix.casefold() != ".parquet" for path, _identity in expected
            )
            continue
        failures.extend(tentative_failures)
        observed_all.extend(tentative_observed)
        existing_count += tentative_existing_count
    return tuple(observed_all), failures, existing_count, missing_count


def _scan_validated_text_evidence(
    path: Path,
    expected_identity: DatasetFileIdentity,
) -> str | None:
    if dataset_file_identity(path) != expected_identity:
        raise ValueError("validation_summary_evidence_identity_changed")
    try:
        with _opened_evidence(path, expected_identity) as handle:
            try:
                reason = evidence_text_stream_unsafe_reason(
                    _iter_exact_evidence_chunks(handle, expected_identity.size)
                )
            finally:
                primary = sys.exception()
                try:
                    if (
                        dataset_file_identity_from_stat(os.fstat(handle.fileno()))
                        != expected_identity
                    ):
                        raise ValueError("validation_summary_evidence_identity_changed")
                except BaseException:
                    if primary is None:
                        raise
    finally:
        primary = sys.exception()
        try:
            if dataset_file_identity(path) != expected_identity:
                raise ValueError("validation_summary_evidence_identity_changed")
        except BaseException:
            if primary is None:
                raise
    return reason


def _iter_exact_evidence_chunks(
    handle: BinaryIO,
    expected_size: int,
) -> Iterator[memoryview]:
    buffer = bytearray(min(expected_size, io.DEFAULT_BUFFER_SIZE))
    offset = 0
    while offset < expected_size:
        target = memoryview(buffer)[: min(len(buffer), expected_size - offset)]
        count = handle.readinto(target)
        if count is None or count <= 0 or count > len(target):
            raise ValueError("validation_summary_evidence_identity_changed")
        offset += count
        yield target[:count]
    if handle.read(1):
        raise ValueError("validation_summary_evidence_identity_changed")


@contextmanager
def _opened_evidence(
    path: Path,
    expected_identity: DatasetFileIdentity,
) -> Iterator[BinaryIO]:
    handle = path.open("rb")
    try:
        if (
            dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            != expected_identity
        ):
            raise ValueError("validation_summary_evidence_identity_changed")
        yield handle
    finally:
        if sys.exception() is None:
            handle.close()
        else:
            try:
                handle.close()
            except BaseException:
                pass


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
