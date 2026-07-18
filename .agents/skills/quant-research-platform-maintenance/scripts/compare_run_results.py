"""Compare two QRP run roots from their current contracts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_research_platform.artifacts._output_bundle_validation import (  # noqa: E402
    validate_output_bundle,
)
from quant_research_platform.artifacts._output_manifest import (  # noqa: E402
    validate_output_manifest_business_envelope,
)
from quant_research_platform.contracts._json import (  # noqa: E402
    canonical_identity_bytes,
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.contracts.run_id import validate_run_id  # noqa: E402
from quant_research_platform.contracts.workflows import (  # noqa: E402
    validate_schema_inventory_summary,
)
from quant_research_platform.policy.check_status import (  # noqa: E402
    aggregate_validation_matrix_status,
    overall_verification_status,
    require_status,
)
from quant_research_platform.policy.identifiers import (  # noqa: E402
    validate_output_identifier,
)
from quant_research_platform.runtime.logical_paths import (  # noqa: E402
    RUN_RESULT_DATASET_LAYERS,
)
from quant_research_platform.storage.atomic import atomic_write_directory  # noqa: E402

_RUN_KEYS = frozenset(
    {
        "schema_id",
        "workflow_id",
        "run_id",
        "root_ownership",
        "created_at",
        "completed_at",
        "status",
        "code",
        "config",
        "inputs",
        "data",
        "outputs",
        "checks",
        "execution_metrics",
        "logs",
        "summary",
    }
)
_ROOT_OWNERSHIP_KEYS = frozenset(
    {"owner", "workflow_id", "created_by", "created_at", "replace_safe"}
)
_RUN_STATUS_KEYS = frozenset(
    {
        "overall",
        "validation",
        "smoke",
        "compare",
        "share_safety",
        "full_horizon",
        "validation_matrix",
    }
)
_RUN_DATA_KEYS = frozenset(
    {"layer_count", "layers", "run_validation_status", "failed_layers", "created_at"}
)
_LAYER_KEYS = frozenset(
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
_DATASET_INDEX_KEYS = frozenset({"data_id", "manifest_path", "manifest_summary"})
_MANIFEST_SUMMARY_KEYS = frozenset(
    {
        "row_count",
        "date_min",
        "date_max",
        "partition_count",
        "duplicate_primary_key_count",
        "validation_status",
    }
)
_VALIDATION_FAILURE_CODE_ORDER = (
    "dataset_manifest_missing",
    "dataset_manifest_invalid",
    "dataset_validation_failed",
    "primary_key_nulls",
    "duplicate_primary_key",
    "non_finite_values",
    "source_local_key_leakage",
    "layer_validation_failed",
    "share_safety_scan_failed",
)
_VALIDATION_FAILURE_CODES = frozenset(_VALIDATION_FAILURE_CODE_ORDER)
_DATA_VALIDATION_FAILURE_CODES = frozenset(
    {
        "dataset_manifest_missing",
        "dataset_manifest_invalid",
        "dataset_validation_failed",
        "primary_key_nulls",
        "duplicate_primary_key",
        "non_finite_values",
        "layer_validation_failed",
    }
)
_WORKFLOW_ID = "local_verification"
_VALIDATION_CHECK_KEYS = frozenset(
    {
        "run_id",
        "status",
        "failure_codes",
        "required_columns_status",
        "primary_key_null_failures",
        "duplicate_primary_key_failures",
        "source_local_key_leakage",
        "date_ranges",
        "partition_coverage_status",
        "non_finite_failures",
        "factor_nan_profile",
        "label_nan_profile",
        "dataset_manifest_completeness",
        "data_layer_summary_completeness",
        "share_safety_scan",
        "created_at",
    }
)
_SHARE_SAFETY_KEYS = frozenset(
    {
        "status",
        "failures",
        "checked_count",
        "requested_count",
        "existing_scanned_count",
        "missing_count",
        "requested_paths",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


class CompareBlockedError(ValueError):
    """Raised when compare inputs cannot be read as current QRP contracts."""


class _RunDataFacts:
    __slots__ = (
        "status",
        "layer_ids",
        "failed_layer_ids",
        "dataset_ids",
        "failed_dataset_ids",
        "missing_manifest_paths",
        "invalid_manifest_paths",
        "dataset_date_ranges",
        "duplicate_dataset_ids",
        "total_row_count",
    )

    def __init__(
        self,
        *,
        status: str,
        layer_ids: tuple[str, ...],
        failed_layer_ids: tuple[str, ...],
        dataset_ids: tuple[str, ...],
        failed_dataset_ids: tuple[str, ...],
        missing_manifest_paths: tuple[str, ...],
        invalid_manifest_paths: tuple[str, ...],
        dataset_date_ranges: tuple[tuple[str, str | None, str | None, int], ...],
        duplicate_dataset_ids: tuple[str, ...],
        total_row_count: int,
    ) -> None:
        self.status = status
        self.layer_ids = layer_ids
        self.failed_layer_ids = failed_layer_ids
        self.dataset_ids = dataset_ids
        self.failed_dataset_ids = failed_dataset_ids
        self.missing_manifest_paths = missing_manifest_paths
        self.invalid_manifest_paths = invalid_manifest_paths
        self.dataset_date_ranges = dataset_date_ranges
        self.duplicate_dataset_ids = duplicate_dataset_ids
        self.total_row_count = total_row_count


def compare_runs(
    *,
    baseline_root: Path,
    current_root: Path,
    compare_root: Path,
    pairs_file: Path | None = None,
) -> dict[str, Any]:
    baseline_root = Path(baseline_root)
    current_root = Path(current_root)
    compare_root = Path(compare_root)
    compare_root_error = _compare_root_error(
        compare_root,
        baseline_root=baseline_root,
        current_root=current_root,
    )
    if compare_root_error is not None:
        return _blocked_payload(
            baseline_root=baseline_root,
            current_root=current_root,
            reason=compare_root_error,
        )
    try:
        try:
            baseline_run = _read_qrp_json(
                baseline_root / "run.json",
                expected_schema_id="qrp-run",
            )
            current_run = _read_qrp_json(
                current_root / "run.json",
                expected_schema_id="qrp-run",
            )
            _validate_run_payload(baseline_run, run_root=baseline_root)
            _validate_run_payload(current_run, run_root=current_root)
            baseline_outputs = _outputs_by_id(baseline_run)
            current_outputs = _outputs_by_id(current_run)
            baseline_bindings = _bind_declared_outputs(
                baseline_root,
                run_id=baseline_run["run_id"],
                outputs=baseline_outputs,
            )
            current_bindings = _bind_declared_outputs(
                current_root,
                run_id=current_run["run_id"],
                outputs=current_outputs,
            )
            pairs, pairing_mode = _resolve_pairs(
                baseline_run,
                current_run,
                pairs_file=pairs_file,
            )
            _reject_hidden_invalid_declared_outputs(
                baseline_bindings,
                current_bindings,
                pairs=pairs,
            )
            data_results = _compare_data_summaries(
                baseline_run["data"],
                current_run["data"],
            )
            output_results = [
                _compare_output_pair(
                    baseline_root=baseline_root,
                    current_root=current_root,
                    baseline_output_id=pair["baseline_output_id"],
                    current_output_id=pair["current_output_id"],
                    baseline_binding=baseline_bindings[pair["baseline_output_id"]],
                    current_binding=current_bindings[pair["current_output_id"]],
                )
                for pair in pairs
            ]
            statuses = [
                *(result["status"] for result in data_results),
                *(result["status"] for result in output_results),
            ]
            payload = {
                "schema_id": "qrp-compare",
                "inputs": {
                    "baseline_root": baseline_root.name,
                    "current_root": current_root.name,
                },
                "baseline_run": _run_summary(baseline_run),
                "current_run": _run_summary(current_run),
                "pairing": {"mode": pairing_mode, "pairs": pairs},
                "data_results": data_results,
                "output_results": output_results,
                "status_counts": dict(sorted(Counter(statuses).items())),
                "overall_status": aggregate_validation_matrix_status(statuses),
            }
        except CompareBlockedError as exc:
            payload = _blocked_payload(
                baseline_root=baseline_root,
                current_root=current_root,
                reason=str(exc),
            )
        _write_compare_artifacts(compare_root, payload)
    except OSError as exc:
        return _blocked_payload(
            baseline_root=baseline_root,
            current_root=current_root,
            reason=f"compare_output_unavailable:{type(exc).__name__}",
        )
    return payload


def _blocked_payload(
    *,
    baseline_root: Path,
    current_root: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_id": "qrp-compare",
        "inputs": {
            "baseline_root": baseline_root.name,
            "current_root": current_root.name,
        },
        "baseline_run": {},
        "current_run": {},
        "pairing": {"mode": "NOT_RUN", "pairs": []},
        "data_results": [],
        "output_results": [],
        "status_counts": {"BLOCKED": 1},
        "overall_status": "BLOCKED",
        "blocked_reason": reason,
    }


def _compare_root_error(
    compare_root: Path,
    *,
    baseline_root: Path,
    current_root: Path,
) -> str | None:
    resolved_compare = compare_root.resolve(strict=False)
    for input_root in (baseline_root, current_root):
        resolved_input = input_root.resolve(strict=False)
        compare_inside_input = resolved_compare.is_relative_to(resolved_input)
        input_inside_compare = resolved_input.is_relative_to(resolved_compare)
        if compare_inside_input or input_inside_compare:
            return "compare_root_overlaps_input"
    if compare_root.exists():
        return (
            "compare_root_exists"
            if compare_root.is_dir()
            else "compare_root_not_directory"
        )
    return None


def _read_qrp_json(path: Path, *, expected_schema_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompareBlockedError(f"json_missing:{path.name}")
    try:
        payload = parse_qrp_json_object(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CompareBlockedError(f"json_unreadable:{path.name}") from exc
    if payload.get("schema_id") != expected_schema_id:
        raise CompareBlockedError(f"schema_id_mismatch:{path.name}")
    return payload


def _validate_run_payload(payload: dict[str, Any], *, run_root: Path) -> None:
    if set(payload) != _RUN_KEYS:
        raise CompareBlockedError("run_fields_invalid")
    if payload["workflow_id"] != _WORKFLOW_ID:
        raise CompareBlockedError("workflow_id_invalid")
    run_id = _validated_run_id(payload["run_id"], "run_id_invalid")
    created_at = _exact_nonempty_string(payload["created_at"], "run_created_at_invalid")
    _exact_nonempty_string(payload["completed_at"], "run_completed_at_invalid")
    for field in ("code", "config", "inputs", "checks", "execution_metrics", "logs"):
        if type(payload[field]) is not dict:
            raise CompareBlockedError(f"run_{field}_invalid")
    summary = payload["summary"]
    if type(summary) is not dict or set(summary) != {"markdown_path"}:
        raise CompareBlockedError("run_summary_invalid")
    summary_path = _canonical_relative_posix_path(
        summary["markdown_path"], "run_summary_invalid"
    )
    if summary_path != "summary.md":
        raise CompareBlockedError("run_summary_invalid")
    _validate_run_summary_artifact(run_root, summary_path)
    _validate_root_ownership(
        payload["root_ownership"],
        run_created_at=created_at,
    )
    status = _validate_run_status(payload["status"])
    data_facts = _validate_run_data(payload["data"])
    output_evidence_paths = _validate_run_outputs(
        payload["outputs"],
        run_id=run_id,
    )
    _validate_run_status_evidence_bindings(
        status,
        run_id=run_id,
        data_facts=data_facts,
        checks=payload["checks"],
        output_evidence_paths=output_evidence_paths,
    )


def _validate_run_outputs(
    outputs: object,
    *,
    run_id: str,
) -> tuple[str, ...]:
    if type(outputs) is not list:
        raise CompareBlockedError("outputs_invalid")
    output_ids: set[str] = set()
    evidence_paths: list[str] = []
    for item in outputs:
        if type(item) is not dict or "manifest_path" not in item:
            raise CompareBlockedError("output_manifest_fields_invalid")
        persisted_payload = dict(item)
        manifest_path_value = persisted_payload.pop("manifest_path")
        try:
            validated = validate_output_manifest_business_envelope(persisted_payload)
        except ValueError as exc:
            raise CompareBlockedError(str(exc)) from exc
        if validated["run_id"] != run_id:
            raise CompareBlockedError("output_manifest_run_id_mismatch")
        output_id = validated["output_id"]
        manifest_path = _canonical_relative_posix_path(
            manifest_path_value,
            "output_manifest_path_invalid",
        )
        if manifest_path != f"outputs/{output_id}/manifest.json":
            raise CompareBlockedError("output_manifest_path_invalid")
        if output_id in output_ids:
            raise CompareBlockedError("output_id_duplicate")
        output_ids.add(output_id)
        evidence_paths.append(manifest_path)
        evidence_paths.extend(
            f"outputs/{output_id}/{name}"
            for name in validated["files"]
            if PurePosixPath(name).suffix.casefold() != ".parquet"
        )
    return tuple(evidence_paths)


def _validate_root_ownership(
    value: object,
    *,
    run_created_at: str,
) -> None:
    if type(value) is not dict or set(value) != _ROOT_OWNERSHIP_KEYS:
        raise CompareBlockedError("run_root_ownership_invalid")
    if (
        value["owner"] != _WORKFLOW_ID
        or value["workflow_id"] != _WORKFLOW_ID
        or value["created_by"] != "quant_research_platform"
        or type(value["replace_safe"]) is not bool
        or value["replace_safe"] is not True
    ):
        raise CompareBlockedError("run_root_ownership_invalid")
    ownership_created_at = _exact_nonempty_string(
        value["created_at"], "run_root_ownership_invalid"
    )
    if ownership_created_at != run_created_at:
        raise CompareBlockedError("run_root_ownership_invalid")


def _validate_run_status(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != _RUN_STATUS_KEYS:
        raise CompareBlockedError("run_status_invalid")
    for status in value.values():
        try:
            require_status(status)
        except ValueError as exc:
            raise CompareBlockedError("run_status_invalid") from exc
    expected_matrix = aggregate_validation_matrix_status(
        (value["smoke"], value["compare"], value["full_horizon"])
    )
    if value["validation_matrix"] != expected_matrix:
        raise CompareBlockedError("run_status_validation_matrix_mismatch")
    expected_overall = overall_verification_status(
        value["validation"],
        {"validation_matrix_status": expected_matrix},
    )
    if value["overall"] != expected_overall:
        raise CompareBlockedError("run_status_overall_mismatch")
    if value["smoke"] != value["validation"]:
        raise CompareBlockedError("run_status_smoke_validation_mismatch")
    if value["compare"] != "NOT_RUN" or value["full_horizon"] != "NOT_RUN":
        raise CompareBlockedError("run_status_evidence_missing")
    return value


def _validate_run_data(value: object) -> _RunDataFacts:
    if type(value) is not dict or set(value) != _RUN_DATA_KEYS:
        raise CompareBlockedError("run_data_invalid")
    layer_count = _exact_nonnegative_int(value["layer_count"], "run_data_invalid")
    layers = value["layers"]
    failed_layers = _exact_string_list(value["failed_layers"], "run_data_invalid")
    _exact_nonempty_string(value["created_at"], "run_data_invalid")
    if type(layers) is not list or not layers or layer_count != len(layers):
        raise CompareBlockedError("run_data_invalid")
    observed_layers: list[str] = []
    observed_failed: list[str] = []
    dataset_ids: list[str] = []
    failed_dataset_ids: list[str] = []
    missing_manifest_paths: list[str] = []
    invalid_manifest_paths: list[str] = []
    dataset_date_ranges: list[tuple[str, str | None, str | None, int]] = []
    duplicate_dataset_ids: list[str] = []
    total_row_count = 0
    for layer in layers:
        (
            logical_layer,
            status,
            row_count,
            layer_dataset_ids,
            layer_failed_ids,
            layer_missing_paths,
            layer_invalid_paths,
            layer_date_ranges,
            layer_duplicate_ids,
        ) = _validate_run_layer(layer)
        if logical_layer in observed_layers:
            raise CompareBlockedError("run_data_layer_duplicate")
        observed_layers.append(logical_layer)
        if status != "PASS":
            observed_failed.append(logical_layer)
        dataset_ids.extend(layer_dataset_ids)
        failed_dataset_ids.extend(layer_failed_ids)
        missing_manifest_paths.extend(layer_missing_paths)
        invalid_manifest_paths.extend(layer_invalid_paths)
        dataset_date_ranges.extend(layer_date_ranges)
        duplicate_dataset_ids.extend(layer_duplicate_ids)
        total_row_count += row_count
    if observed_layers != sorted(observed_layers):
        raise CompareBlockedError("run_data_layer_order_invalid")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise CompareBlockedError("run_data_dataset_duplicate")
    if failed_layers != observed_failed:
        raise CompareBlockedError("run_data_failed_layers_mismatch")
    try:
        run_status = require_status(value["run_validation_status"])
    except ValueError as exc:
        raise CompareBlockedError("run_data_status_invalid") from exc
    if run_status != ("PASS" if not observed_failed else "FAIL"):
        raise CompareBlockedError("run_data_status_mismatch")
    return _RunDataFacts(
        status=run_status,
        layer_ids=tuple(observed_layers),
        failed_layer_ids=tuple(observed_failed),
        dataset_ids=tuple(dataset_ids),
        failed_dataset_ids=tuple(failed_dataset_ids),
        missing_manifest_paths=tuple(missing_manifest_paths),
        invalid_manifest_paths=tuple(invalid_manifest_paths),
        dataset_date_ranges=tuple(dataset_date_ranges),
        duplicate_dataset_ids=tuple(duplicate_dataset_ids),
        total_row_count=total_row_count,
    )


def _validate_run_status_evidence_bindings(
    status: dict[str, str],
    *,
    run_id: str,
    data_facts: _RunDataFacts,
    checks: object,
    output_evidence_paths: tuple[str, ...],
) -> None:
    if type(checks) is not dict or set(checks) != {"schema_inventory", "validation"}:
        raise CompareBlockedError("run_checks_invalid")
    validation = checks["validation"]
    if type(validation) is not dict or set(validation) != _VALIDATION_CHECK_KEYS:
        raise CompareBlockedError("run_validation_check_invalid")
    if validation["run_id"] != run_id:
        raise CompareBlockedError("run_validation_check_invalid")
    detail_bindings = _validate_validation_evidence_shape(
        validation,
        data_facts=data_facts,
        output_evidence_paths=output_evidence_paths,
    )
    try:
        validation_status = require_status(validation["status"])
    except ValueError as exc:
        raise CompareBlockedError("run_validation_check_invalid") from exc
    try:
        validate_schema_inventory_summary(
            checks["schema_inventory"],
            validation_status=validation_status,
            validation_failure_codes=validation["failure_codes"],
            dataset_count=len(data_facts.dataset_ids),
            total_row_count=data_facts.total_row_count,
        )
    except ValueError as exc:
        raise CompareBlockedError("run_schema_inventory_invalid") from exc
    share_safety = validation["share_safety_scan"]
    try:
        share_safety_status = require_status(share_safety["status"])
    except ValueError as exc:
        raise CompareBlockedError("run_validation_check_invalid") from exc
    failure_codes = _exact_string_list(
        validation["failure_codes"],
        "run_validation_check_invalid",
    )
    if any(code not in _VALIDATION_FAILURE_CODES for code in failure_codes):
        raise CompareBlockedError("run_validation_check_invalid")
    expected_failure_codes = [
        code for code in _VALIDATION_FAILURE_CODE_ORDER if bool(detail_bindings[code])
    ]
    if failure_codes != expected_failure_codes:
        raise CompareBlockedError("run_validation_detail_facts_mismatch")
    safety_failures = share_safety["failures"]
    expected_validation_status = "FAIL" if failure_codes else "PASS"
    if validation_status != expected_validation_status:
        raise CompareBlockedError("run_validation_check_status_mismatch")
    expected_safety_status = "FAIL" if safety_failures else "PASS"
    if share_safety_status != expected_safety_status:
        raise CompareBlockedError("run_share_safety_check_status_mismatch")
    if bool(set(failure_codes) & _DATA_VALIDATION_FAILURE_CODES) != (
        data_facts.status != "PASS"
    ):
        raise CompareBlockedError("run_validation_data_facts_mismatch")
    if ("share_safety_scan_failed" in failure_codes) != (share_safety_status != "PASS"):
        raise CompareBlockedError("run_validation_share_safety_facts_mismatch")
    if status["validation"] != validation_status:
        raise CompareBlockedError("run_status_validation_check_mismatch")
    if status["share_safety"] != share_safety_status:
        raise CompareBlockedError("run_status_share_safety_check_mismatch")
    if data_facts.status != "PASS" and validation_status != "FAIL":
        raise CompareBlockedError("run_status_data_validation_mismatch")
    if share_safety_status != "PASS" and validation_status != "FAIL":
        raise CompareBlockedError("run_status_share_safety_validation_mismatch")


def _validate_validation_evidence_shape(
    validation: dict[str, Any],
    *,
    data_facts: _RunDataFacts,
    output_evidence_paths: tuple[str, ...],
) -> dict[str, object]:
    if validation["required_columns_status"] != "summarized_by_dataset_manifests":
        raise CompareBlockedError("run_validation_check_invalid")
    detail_lists: dict[str, list[str]] = {}
    for field in (
        "primary_key_null_failures",
        "duplicate_primary_key_failures",
        "source_local_key_leakage",
        "non_finite_failures",
    ):
        detail_lists[field] = _exact_string_list(
            validation[field], "run_validation_check_invalid"
        )
    dataset_order = {
        data_id: index for index, data_id in enumerate(data_facts.dataset_ids)
    }
    for details in detail_lists.values():
        if any(
            data_id not in dataset_order for data_id in details
        ) or details != sorted(details, key=dataset_order.__getitem__):
            raise CompareBlockedError("run_validation_detail_facts_mismatch")
    for field in (
        "primary_key_null_failures",
        "non_finite_failures",
    ):
        if not set(detail_lists[field]) <= set(data_facts.failed_dataset_ids):
            raise CompareBlockedError("run_validation_detail_facts_mismatch")
    if detail_lists["duplicate_primary_key_failures"] != list(
        data_facts.duplicate_dataset_ids
    ):
        raise CompareBlockedError("run_validation_detail_facts_mismatch")
    for field in ("factor_nan_profile", "label_nan_profile"):
        _validate_count_mapping(validation[field], "run_validation_check_invalid")
    _validate_validation_date_ranges(
        validation["date_ranges"],
        expected=data_facts.dataset_date_ranges,
    )
    try:
        require_status(validation["partition_coverage_status"])
    except ValueError as exc:
        raise CompareBlockedError("run_validation_check_invalid") from exc

    dataset_completeness = validation["dataset_manifest_completeness"]
    if type(dataset_completeness) is not dict or set(dataset_completeness) != {
        "missing",
        "invalid",
        "checked_count",
    }:
        raise CompareBlockedError("run_validation_check_invalid")
    missing = _exact_string_list(
        dataset_completeness["missing"], "run_validation_check_invalid"
    )
    invalid = _exact_string_list(
        dataset_completeness["invalid"], "run_validation_check_invalid"
    )
    for path in [*missing, *invalid]:
        _canonical_relative_posix_path(path, "run_validation_check_invalid")
    checked_count = _exact_positive_int(
        dataset_completeness["checked_count"], "run_validation_check_invalid"
    )
    if (
        missing != list(data_facts.missing_manifest_paths)
        or invalid != list(data_facts.invalid_manifest_paths)
        or checked_count != len(data_facts.dataset_ids) + len(missing) + len(invalid)
        or validation["partition_coverage_status"]
        != ("FAIL" if missing or invalid else "PASS")
    ):
        raise CompareBlockedError("run_validation_check_invalid")

    layer_completeness = validation["data_layer_summary_completeness"]
    if type(layer_completeness) is not dict or set(layer_completeness) != {
        "missing",
        "checked_count",
    }:
        raise CompareBlockedError("run_validation_check_invalid")
    layer_missing = _exact_string_list(
        layer_completeness["missing"], "run_validation_check_invalid"
    )
    if (
        _exact_positive_int(
            layer_completeness["checked_count"], "run_validation_check_invalid"
        )
        != len(data_facts.layer_ids)
        or layer_missing
    ):
        raise CompareBlockedError("run_validation_check_invalid")
    _exact_nonempty_string(validation["created_at"], "run_validation_check_invalid")
    _validate_share_safety_evidence(
        validation["share_safety_scan"],
        expected_paths=output_evidence_paths,
    )
    return {
        "dataset_manifest_missing": missing,
        "dataset_manifest_invalid": invalid,
        "dataset_validation_failed": data_facts.failed_dataset_ids,
        "primary_key_nulls": detail_lists["primary_key_null_failures"],
        "duplicate_primary_key": detail_lists["duplicate_primary_key_failures"],
        "non_finite_values": detail_lists["non_finite_failures"],
        "source_local_key_leakage": detail_lists["source_local_key_leakage"],
        "layer_validation_failed": data_facts.failed_layer_ids,
        "share_safety_scan_failed": validation["share_safety_scan"]["failures"],
    }


def _validate_validation_date_ranges(
    value: object,
    *,
    expected: tuple[tuple[str, str | None, str | None, int], ...],
) -> None:
    if type(value) is not list or len(value) != len(expected):
        raise CompareBlockedError("run_validation_check_invalid")
    observed: list[tuple[str, str | None, str | None, int]] = []
    for item in value:
        if type(item) is not dict or set(item) != {
            "data_id",
            "date_min",
            "date_max",
            "partition_count",
        }:
            raise CompareBlockedError("run_validation_check_invalid")
        data_id = _exact_nonempty_string(
            item["data_id"], "run_validation_check_invalid"
        )
        date_min, date_max = _canonical_date_range(
            item["date_min"], item["date_max"], "run_validation_check_invalid"
        )
        partition_count = _exact_nonnegative_int(
            item["partition_count"], "run_validation_check_invalid"
        )
        observed.append((data_id, date_min, date_max, partition_count))
    if observed != list(expected):
        raise CompareBlockedError("run_validation_detail_facts_mismatch")


def _validate_share_safety_evidence(
    value: object,
    *,
    expected_paths: tuple[str, ...],
) -> None:
    if type(value) is not dict or set(value) != _SHARE_SAFETY_KEYS:
        raise CompareBlockedError("run_validation_check_invalid")
    try:
        status = require_status(value["status"])
    except ValueError as exc:
        raise CompareBlockedError("run_validation_check_invalid") from exc
    failures = value["failures"]
    if type(failures) is not list:
        raise CompareBlockedError("run_validation_check_invalid")
    observed_failures: set[tuple[str, str]] = set()
    for item in failures:
        if type(item) is not dict or set(item) != {"path", "reason"}:
            raise CompareBlockedError("run_validation_check_invalid")
        path = _canonical_relative_posix_path(
            item["path"], "run_validation_check_invalid"
        )
        reason = _exact_nonempty_string(item["reason"], "run_validation_check_invalid")
        if (path, reason) in observed_failures:
            raise CompareBlockedError("run_validation_check_invalid")
        observed_failures.add((path, reason))
    checked_count = _exact_nonnegative_int(
        value["checked_count"], "run_validation_check_invalid"
    )
    requested_count = _exact_nonnegative_int(
        value["requested_count"], "run_validation_check_invalid"
    )
    existing_count = _exact_nonnegative_int(
        value["existing_scanned_count"], "run_validation_check_invalid"
    )
    missing_count = _exact_nonnegative_int(
        value["missing_count"], "run_validation_check_invalid"
    )
    requested_paths = _exact_string_list(
        value["requested_paths"], "run_validation_check_invalid"
    )
    for path in requested_paths:
        _canonical_relative_posix_path(path, "run_validation_check_invalid")
    if (
        checked_count != existing_count
        or missing_count != requested_count - existing_count
        or existing_count > requested_count
        or requested_count != len(requested_paths)
        or requested_paths != list(expected_paths)
        or any(path not in requested_paths for path, _reason in observed_failures)
        or status != ("FAIL" if failures or missing_count else "PASS")
    ):
        raise CompareBlockedError("run_validation_check_invalid")


def _validate_count_mapping(value: object, error: str) -> None:
    if type(value) is not dict:
        raise CompareBlockedError(error)
    for key, item in value.items():
        _exact_nonempty_string(key, error)
        _exact_nonnegative_int(item, error)


def _validate_run_layer(
    value: object,
) -> tuple[
    str,
    str,
    int,
    list[str],
    list[str],
    list[str],
    list[str],
    list[tuple[str, str | None, str | None, int]],
    list[str],
]:
    if type(value) is not dict or set(value) != _LAYER_KEYS:
        raise CompareBlockedError("run_data_layer_invalid")
    logical_layer = _exact_nonempty_string(
        value["logical_layer"], "run_data_layer_invalid"
    )
    if logical_layer not in RUN_RESULT_DATASET_LAYERS:
        raise CompareBlockedError("run_data_layer_invalid")
    datasets = value["datasets"]
    if type(datasets) is not list:
        raise CompareBlockedError("run_data_layer_invalid")
    dataset_count = _exact_nonnegative_int(
        value["dataset_count"], "run_data_layer_invalid"
    )
    total_rows = _exact_nonnegative_int(value["total_rows"], "run_data_layer_invalid")
    if dataset_count != len(datasets):
        raise CompareBlockedError("run_data_dataset_count_mismatch")
    missing = _exact_string_list(
        value["missing_dataset_manifests"], "run_data_layer_invalid"
    )
    invalid = _exact_string_list(
        value["invalid_dataset_manifests"], "run_data_layer_invalid"
    )
    failed = _exact_string_list(
        value["failed_dataset_validations"], "run_data_layer_invalid"
    )
    for path in [*missing, *invalid]:
        canonical_path = _canonical_relative_posix_path(path, "run_data_layer_invalid")
        if not _manifest_path_belongs_to_layer(canonical_path, logical_layer):
            raise CompareBlockedError("run_data_layer_invalid")
    layer_date_min, layer_date_max = _canonical_date_range(
        value["date_min"],
        value["date_max"],
        "run_data_layer_invalid",
    )
    _exact_nonempty_string(value["created_at"], "run_data_layer_invalid")
    observed_rows = 0
    observed_failed: list[str] = []
    observed_ids: list[str] = []
    observed_paths: list[str] = []
    observed_ranges: list[tuple[str, str | None, str | None, int]] = []
    duplicate_ids: list[str] = []
    observed_date_mins: list[str] = []
    observed_date_maxs: list[str] = []
    for dataset in datasets:
        (
            data_id,
            manifest_path,
            row_count,
            status,
            date_min,
            date_max,
            partition_count,
            duplicate_count,
        ) = _validate_run_dataset(dataset, logical_layer=logical_layer)
        if data_id in observed_ids:
            raise CompareBlockedError("run_data_dataset_duplicate")
        observed_ids.append(data_id)
        observed_paths.append(manifest_path)
        observed_rows += row_count
        if status != "PASS":
            observed_failed.append(data_id)
        if date_min is not None:
            observed_date_mins.append(date_min)
        if date_max is not None:
            observed_date_maxs.append(date_max)
        observed_ranges.append((data_id, date_min, date_max, partition_count))
        if duplicate_count > 0:
            duplicate_ids.append(data_id)
    if observed_paths != sorted(observed_paths):
        raise CompareBlockedError("run_data_dataset_order_invalid")
    if total_rows != observed_rows:
        raise CompareBlockedError("run_data_total_rows_mismatch")
    if failed != observed_failed:
        raise CompareBlockedError("run_data_failed_datasets_mismatch")
    expected_date_min = min(observed_date_mins) if observed_date_mins else None
    expected_date_max = max(observed_date_maxs) if observed_date_maxs else None
    if (layer_date_min, layer_date_max) != (expected_date_min, expected_date_max):
        raise CompareBlockedError("run_data_layer_date_range_mismatch")
    try:
        status = require_status(value["layer_validation_status"])
    except ValueError as exc:
        raise CompareBlockedError("run_data_layer_status_invalid") from exc
    expected_status = (
        "PASS" if not missing and not invalid and not observed_failed else "FAIL"
    )
    if status != expected_status:
        raise CompareBlockedError("run_data_layer_status_mismatch")
    return (
        logical_layer,
        status,
        total_rows,
        observed_ids,
        observed_failed,
        missing,
        invalid,
        observed_ranges,
        duplicate_ids,
    )


def _validate_run_dataset(
    value: object,
    *,
    logical_layer: str,
) -> tuple[str, str, int, str, str | None, str | None, int, int]:
    if type(value) is not dict or set(value) != _DATASET_INDEX_KEYS:
        raise CompareBlockedError("run_data_dataset_invalid")
    data_id = _exact_nonempty_string(value["data_id"], "run_data_dataset_invalid")
    manifest_path = _canonical_relative_posix_path(
        value["manifest_path"], "run_data_dataset_manifest_path_invalid"
    )
    if (
        not _manifest_path_belongs_to_layer(manifest_path, logical_layer)
        or PurePosixPath(manifest_path).parts[-3] != data_id
    ):
        raise CompareBlockedError("run_data_dataset_manifest_path_invalid")
    summary = value["manifest_summary"]
    if type(summary) is not dict or set(summary) != _MANIFEST_SUMMARY_KEYS:
        raise CompareBlockedError("run_data_manifest_summary_invalid")
    row_count = _exact_nonnegative_int(
        summary["row_count"], "run_data_manifest_summary_invalid"
    )
    duplicate_count = _exact_nonnegative_int(
        summary["duplicate_primary_key_count"],
        "run_data_manifest_summary_invalid",
    )
    partition_count = _exact_nonnegative_int(
        summary["partition_count"],
        "run_data_manifest_summary_invalid",
    )
    date_min, date_max = _canonical_date_range(
        summary["date_min"],
        summary["date_max"],
        "run_data_manifest_summary_invalid",
    )
    try:
        status = require_status(summary["validation_status"])
    except ValueError as exc:
        raise CompareBlockedError("run_data_manifest_status_invalid") from exc
    if duplicate_count > 0 and status == "PASS":
        raise CompareBlockedError("run_data_manifest_status_mismatch")
    return (
        data_id,
        manifest_path,
        row_count,
        status,
        date_min,
        date_max,
        partition_count,
        duplicate_count,
    )


def _manifest_path_belongs_to_layer(path: str, logical_layer: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) >= 5
        and parts[:2] == ("data", logical_layer)
        and parts[-2:] == ("_manifest", "manifest.json")
    )


def _validate_run_summary_artifact(run_root: Path, relative_path: str) -> None:
    resolved_root = Path(run_root).resolve(strict=False)
    path = (resolved_root / PurePosixPath(relative_path)).resolve(strict=False)
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise CompareBlockedError("run_summary_missing")


def _resolve_pairs(
    baseline_run: dict[str, Any],
    current_run: dict[str, Any],
    *,
    pairs_file: Path | None,
) -> tuple[list[dict[str, str]], str]:
    if pairs_file is not None:
        payload = _read_qrp_json(
            Path(pairs_file),
            expected_schema_id="qrp-compare-pairs",
        )
        if set(payload) != {"schema_id", "pairs"}:
            raise CompareBlockedError("pairs_file_fields_invalid")
        raw_pairs = payload.get("pairs")
        if not isinstance(raw_pairs, list):
            raise CompareBlockedError("pairs_file_pairs_invalid")
        pairs = [_validate_pair(item) for item in raw_pairs]
        if not pairs:
            raise CompareBlockedError("compare_output_pairs_empty")
        baseline_ids = _output_ids(baseline_run)
        current_ids = _output_ids(current_run)
        baseline_pair_ids = [pair["baseline_output_id"] for pair in pairs]
        current_pair_ids = [pair["current_output_id"] for pair in pairs]
        if len(baseline_pair_ids) != len(set(baseline_pair_ids)) or len(
            current_pair_ids
        ) != len(set(current_pair_ids)):
            raise CompareBlockedError("pairs_file_pair_duplicate")
        if (
            not set(baseline_pair_ids) <= baseline_ids
            or not set(current_pair_ids) <= current_ids
        ):
            raise CompareBlockedError("pairs_file_output_not_declared")
        return pairs, "pairs_file"

    baseline_ids = _output_ids(baseline_run)
    current_ids = _output_ids(current_run)
    if not baseline_ids and not current_ids:
        raise CompareBlockedError("compare_output_pairs_empty")
    if baseline_ids != current_ids:
        raise CompareBlockedError("compare_output_id_topology_mismatch")
    return (
        [
            {
                "baseline_output_id": output_id,
                "current_output_id": output_id,
            }
            for output_id in sorted(baseline_ids)
        ],
        "output_id",
    )


def _validate_pair(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "baseline_output_id",
        "current_output_id",
    }:
        raise CompareBlockedError("pairs_file_pair_invalid")
    baseline_id = value["baseline_output_id"]
    current_id = value["current_output_id"]
    if type(baseline_id) is not str or type(current_id) is not str:
        raise CompareBlockedError("pairs_file_output_id_invalid")
    try:
        validate_output_identifier(baseline_id)
        validate_output_identifier(current_id)
    except ValueError as exc:
        raise CompareBlockedError("pairs_file_output_id_invalid") from exc
    return {
        "baseline_output_id": baseline_id,
        "current_output_id": current_id,
    }


def _output_ids(run_payload: dict[str, Any]) -> set[str]:
    outputs = run_payload.get("outputs")
    if not isinstance(outputs, list):
        raise CompareBlockedError("outputs_invalid")
    result: set[str] = set()
    for item in outputs:
        if type(item) is not dict:
            raise CompareBlockedError("output_index_invalid")
        output_id = item.get("output_id")
        if type(output_id) is not str:
            raise CompareBlockedError("output_id_invalid")
        try:
            validated = validate_output_identifier(output_id)
        except ValueError as exc:
            raise CompareBlockedError("output_id_invalid") from exc
        if validated in result:
            raise CompareBlockedError("output_id_duplicate")
        result.add(validated)
    return result


def _outputs_by_id(run_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["output_id"]: item for item in run_payload["outputs"]}


def _bind_declared_outputs(
    run_root: Path,
    *,
    run_id: str,
    outputs: dict[str, dict[str, Any]],
) -> dict[str, tuple[dict[str, Any] | None, list[str]]]:
    bindings: dict[str, tuple[dict[str, Any] | None, list[str]]] = {}
    for output_id, index in outputs.items():
        failures: list[str] = []
        manifest = _read_output_manifest(
            run_root,
            output_id,
            run_id=run_id,
            failures=failures,
        )
        if not _output_index_matches_manifest(index, manifest):
            failures.append("output_index_manifest_mismatch")
        bindings[output_id] = (manifest, sorted(set(failures)))
    return bindings


def _reject_hidden_invalid_declared_outputs(
    baseline_bindings: dict[str, tuple[dict[str, Any] | None, list[str]]],
    current_bindings: dict[str, tuple[dict[str, Any] | None, list[str]]],
    *,
    pairs: list[dict[str, str]],
) -> None:
    selected_baseline_ids = {pair["baseline_output_id"] for pair in pairs}
    selected_current_ids = {pair["current_output_id"] for pair in pairs}
    for side, bindings, selected_ids in (
        ("baseline", baseline_bindings, selected_baseline_ids),
        ("current", current_bindings, selected_current_ids),
    ):
        for output_id, (_manifest, failures) in bindings.items():
            if failures and output_id not in selected_ids:
                raise CompareBlockedError(
                    f"declared_output_invalid:{side}:{output_id}:{failures[0]}"
                )


def _compare_data_summaries(
    baseline_data: object,
    current_data: object,
) -> list[dict[str, Any]]:
    baseline_layers = _layers_by_id(baseline_data)
    current_layers = _layers_by_id(current_data)
    results: list[dict[str, Any]] = []
    for logical_layer in sorted(set(baseline_layers) | set(current_layers)):
        baseline = baseline_layers.get(logical_layer)
        current = current_layers.get(logical_layer)
        failures: list[str] = []
        if baseline is None or current is None:
            failures.append("data_layer_missing")
        elif _without_transient_fields(baseline) != _without_transient_fields(current):
            failures.append("data_summary_mismatch")
        results.append(
            {
                "logical_layer": logical_layer,
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
            }
        )
    return results


def _layers_by_id(data: object) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        raise CompareBlockedError("run_data_invalid")
    layers = data.get("layers")
    if not isinstance(layers, list):
        raise CompareBlockedError("run_data_layers_invalid")
    result: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if type(layer) is not dict or type(layer.get("logical_layer")) is not str:
            raise CompareBlockedError("run_data_layer_invalid")
        logical_layer = layer["logical_layer"]
        if not logical_layer:
            raise CompareBlockedError("run_data_layer_invalid")
        if logical_layer in result:
            raise CompareBlockedError("run_data_layer_duplicate")
        result[logical_layer] = layer
    return result


def _without_transient_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_transient_fields(item)
            for key, item in sorted(value.items())
            if key != "created_at"
        }
    if isinstance(value, list):
        return [_without_transient_fields(item) for item in value]
    return value


def _compare_output_pair(
    *,
    baseline_root: Path,
    current_root: Path,
    baseline_output_id: str,
    current_output_id: str,
    baseline_binding: tuple[dict[str, Any] | None, list[str]],
    current_binding: tuple[dict[str, Any] | None, list[str]],
) -> dict[str, Any]:
    baseline_manifest, baseline_failures = baseline_binding
    current_manifest, current_failures = current_binding
    failures = [*baseline_failures, *current_failures]
    if baseline_manifest is not None and current_manifest is not None:
        if baseline_manifest.get("output_definition_id") != current_manifest.get(
            "output_definition_id"
        ):
            failures.append("output_definition_id_mismatch")
        if canonical_identity_bytes(
            baseline_manifest.get("metadata")
        ) != canonical_identity_bytes(current_manifest.get("metadata")):
            failures.append("metadata_mismatch")
        baseline_files = baseline_manifest.get("files")
        current_files = current_manifest.get("files")
        if baseline_files != current_files:
            failures.append("file_count_or_names_mismatch")
        elif isinstance(baseline_files, list):
            if _file_hashes(
                baseline_root,
                baseline_output_id,
                baseline_files,
                failures=failures,
            ) != _file_hashes(
                current_root,
                current_output_id,
                current_files,
                failures=failures,
            ):
                failures.append("file_hash_mismatch")
        else:
            failures.append("output_files_invalid")
    return {
        "baseline_output_id": baseline_output_id,
        "current_output_id": current_output_id,
        "status": "PASS" if not failures else "FAIL",
        "failures": sorted(set(failures)),
    }


def _output_index_matches_manifest(
    index: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> bool:
    if index is None or manifest is None:
        return index is None and manifest is None
    persisted_index = dict(index)
    persisted_index.pop("manifest_path")
    return canonical_identity_bytes(persisted_index) == canonical_identity_bytes(
        manifest
    )


def _read_output_manifest(
    run_root: Path,
    output_id: str,
    *,
    run_id: str,
    failures: list[str],
) -> dict[str, Any] | None:
    output_root = run_root / "outputs" / output_id
    path = output_root / "manifest.json"
    if not path.is_file():
        failures.append("output_manifest_missing")
        return None
    try:
        validated, _declared_paths = validate_output_bundle(
            output_root,
            expected_run_id=run_id,
            expected_output_id=output_id,
        )
    except ValueError as exc:
        failures.append(str(exc))
        return None
    return validated


def _file_hashes(
    run_root: Path,
    output_id: str,
    files: list[str],
    *,
    failures: list[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    output_bundle = (run_root / "outputs" / output_id).resolve(strict=False)
    for name in files:
        path = (output_bundle / name).resolve(strict=False)
        if not path.is_relative_to(output_bundle):
            failures.append("output_file_path_invalid")
            continue
        if not path.is_file():
            failures.append("output_file_missing")
            continue
        hashes[name] = _streaming_sha256(path)
    return hashes


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload["run_id"],
        "workflow_id": payload["workflow_id"],
        "status": payload["status"],
    }


def _validated_run_id(value: object, error: str) -> str:
    if type(value) is not str:
        raise CompareBlockedError(error)
    try:
        return validate_run_id(value)
    except ValueError as exc:
        raise CompareBlockedError(error) from exc


def _exact_nonempty_string(value: object, error: str) -> str:
    if type(value) is not str or not value:
        raise CompareBlockedError(error)
    return value


def _canonical_date_range(
    minimum: object,
    maximum: object,
    error: str,
) -> tuple[str | None, str | None]:
    if minimum is None and maximum is None:
        return None, None
    if type(minimum) is not str or type(maximum) is not str:
        raise CompareBlockedError(error)
    try:
        parsed_minimum = date.fromisoformat(minimum)
        parsed_maximum = date.fromisoformat(maximum)
    except ValueError as exc:
        raise CompareBlockedError(error) from exc
    if (
        parsed_minimum.isoformat() != minimum
        or parsed_maximum.isoformat() != maximum
        or parsed_minimum > parsed_maximum
    ):
        raise CompareBlockedError(error)
    return minimum, maximum


def _exact_nonnegative_int(value: object, error: str) -> int:
    if type(value) is not int or value < 0:
        raise CompareBlockedError(error)
    return value


def _exact_positive_int(value: object, error: str) -> int:
    result = _exact_nonnegative_int(value, error)
    if result == 0:
        raise CompareBlockedError(error)
    return result


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _exact_string_list(value: object, error: str) -> list[str]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise CompareBlockedError(error)
    return value


def _canonical_relative_posix_path(value: object, error: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise CompareBlockedError(error)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CompareBlockedError(error)
    return value


def _write_compare_artifacts(compare_root: Path, payload: dict[str, Any]) -> None:
    def _writer(staged_root: Path) -> None:
        (staged_root / "compare.json").write_text(
            render_qrp_json_object(payload) + "\n",
            encoding="utf-8",
        )
        (staged_root / "summary.md").write_text(
            _render_summary(payload),
            encoding="utf-8",
        )

    atomic_write_directory(compare_root, _writer)


def _render_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# QRP Run Comparison",
            "",
            f"- overall_status: `{payload['overall_status']}`",
            f"- data_results: `{len(payload['data_results'])}`",
            f"- output_results: `{len(payload['output_results'])}`",
            "",
        )
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--compare-root", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = compare_runs(
        baseline_root=args.baseline_root,
        current_root=args.current_root,
        compare_root=args.compare_root,
        pairs_file=args.pairs_file,
    )
    print(render_qrp_json_object(payload))
    return 0 if payload["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
