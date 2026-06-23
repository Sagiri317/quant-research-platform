"""Verify a generic QRP handoff zip with one handoff sidecar JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_PARENT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PARENT))

from governance_contract import VALID_STATUS_TOKENS  # noqa: E402
from build_handoff_package import (  # noqa: E402
    FORBIDDEN_PACKAGE_ENTRY_PARTS,
    FORBIDDEN_TASK_KEYS,
    FORBIDDEN_VALIDATION_KEYS,
    VALIDATION_ROW_KEYS,
    handoff_evidence_forbidden_key_failures,
    is_reserved_handoff_evidence_path,
    source_tree_scan_from_clone,
)
from handoff_safety import scan_handoff_zip  # noqa: E402

VALID_STATUSES = set(VALID_STATUS_TOKENS)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ZIP_ENTRIES = {
    "artifact_manifest.json",
    "final-branch.bundle",
    "git_status.txt",
    "task_metadata.json",
    "validation_evidence.json",
}
GENERATION_EVIDENCE = "handoff_evidence/generation.json"
SOURCE_TREE_SCAN_EVIDENCE = "handoff_evidence/source-tree-scan.json"
VERIFICATION_MATRIX_EVIDENCE = "handoff_evidence/verification-matrix.json"
BASELINE_FINAL_COMPARE_EVIDENCE = "handoff_evidence/baseline-final-compare.json"
EXPECTED_FAILURE_REASON = "stale_rank_ensemble_prediction_baseline"
RANK_ENSEMBLE_PREDICTION_ARTIFACTS = (
    "equal_weight_rank_ensemble",
    "ic_weighted_rank_ensemble",
)
OLD_BASELINE_GROUP = "old_baseline"
REFRESHED_PREDICTION_BASELINE_GROUP = "refreshed_prediction_baseline"
BASELINE_FINAL_COMPARE_GROUPS = (
    OLD_BASELINE_GROUP,
    REFRESHED_PREDICTION_BASELINE_GROUP,
)
FORBIDDEN_DUPLICATE_SOURCE_ROOTS = (
    ".agents/",
    "docs/",
    "run_tools/",
    "schemas/",
    "src/",
    "templates/",
    "tests/",
)
TRACKED_HANDOFF_SOURCE_PATHS = (
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_safety.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py",
    ".agents/skills/quant-research-platform-maintenance/templates/handoff/task_metadata.template.json",
    ".agents/skills/quant-research-platform-maintenance/schemas/handoff/task_metadata.schema.json",
    ".agents/skills/quant-research-platform-maintenance/schemas/handoff/sidecar.schema.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    failures = verify_handoff_package(Path(args.zip), Path(args.json))
    if failures:
        print("handoff_package_verify_status=FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("handoff_package_verify_status=PASS")
    print(f"zip_sha256={sha256_file(Path(args.zip))}")
    return 0


def verify_handoff_package(zip_path: Path, json_path: Path) -> list[str]:
    failures: list[str] = []
    try:
        sidecar_text = json_path.read_text(encoding="utf-8")
        sidecar = json.loads(sidecar_text)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"sidecar_json_invalid: {exc}"]
    if not isinstance(sidecar, dict):
        return ["sidecar_must_be_object"]

    safety = scan_handoff_zip(zip_path, json_path=json_path)
    if not safety.ok:
        if any(failure.startswith("sidecar_") for failure in safety.failures):
            failures.append("sidecar_safety_failed")
        failures.extend(f"safety_{failure}" for failure in safety.failures)

    failures.extend(verify_sidecar_shape(sidecar, zip_path))
    failures.extend(verify_sidecar_schema(sidecar))
    try:
        with ZipFile(zip_path) as archive:
            names = archive.namelist()
            name_set = set(names)
            if any("\\" in name for name in names):
                failures.append("zip_paths_not_posix")
            if json_path.name in name_set:
                failures.append("external_json_in_zip")
            missing = sorted(REQUIRED_ZIP_ENTRIES - name_set)
            failures.extend(f"missing_required_entry: {name}" for name in missing)
            failures.extend(
                f"forbidden_entry: {name}"
                for name in sorted(forbidden_package_entries(names))
            )
            failures.extend(
                f"forbidden_handoff_evidence_entry: {name}"
                for name in sorted(names)
                if name.startswith("handoff_evidence/")
                and is_reserved_handoff_evidence_path(name)
            )
            duplicate_source = sorted(
                name
                for name in names
                if name.startswith(FORBIDDEN_DUPLICATE_SOURCE_ROOTS)
            )
            failures.extend(
                f"duplicate_tracked_source_in_zip: {name}"
                for name in duplicate_source
            )
            for required_evidence in (
                GENERATION_EVIDENCE,
                SOURCE_TREE_SCAN_EVIDENCE,
                VERIFICATION_MATRIX_EVIDENCE,
            ):
                if required_evidence not in name_set:
                    failures.append(f"missing_required_entry: {required_evidence}")
            failures.extend(verify_artifact_manifest(archive, sidecar))
            failures.extend(verify_metadata_entries(archive, sidecar))
            failures.extend(verify_handoff_evidence_json(archive, sidecar))
            failures.extend(verify_final_branch_bundle(archive, sidecar))
            failures.extend(verify_git_status(archive))
    except Exception as exc:
        failures.append(f"zip_verification_error: {type(exc).__name__}: {exc}")
    return failures


def verify_sidecar_shape(sidecar: dict[str, object], zip_path: Path) -> list[str]:
    failures: list[str] = []
    allowed = {
        "schema_version",
        "zip_filename",
        "handoff_package_path",
        "handoff_sidecar_path",
        "zip_sha256",
        "artifact_manifest_sha256",
        "branch",
        "base",
        "final_head",
        "generated_at_utc",
        "status_summary",
    }
    extra = sorted(set(sidecar) - allowed)
    failures.extend(f"sidecar_unapproved_field: {key}" for key in extra)
    if sidecar.get("schema_version") != "handoff-sidecar/0.1":
        failures.append("sidecar_schema_version_mismatch")
    if sidecar.get("zip_filename") != zip_path.name:
        failures.append("sidecar_zip_filename_mismatch")
    package_path = sidecar.get("handoff_package_path")
    if (
        not isinstance(package_path, str)
        or "\\" in package_path
        or not package_path.endswith(
        "/" + zip_path.name
        )
    ):
        failures.append("sidecar_handoff_package_path_invalid")
    sidecar_path = sidecar.get("handoff_sidecar_path")
    if (
        not isinstance(sidecar_path, str)
        or "\\" in sidecar_path
        or not sidecar_path.endswith(".json")
    ):
        failures.append("sidecar_handoff_sidecar_path_invalid")
    if sha256_file(zip_path) != sidecar.get("zip_sha256"):
        failures.append("zip_sha256_mismatch")
    for key in ("branch", "base", "final_head", "generated_at_utc"):
        if not sidecar.get(key):
            failures.append(f"sidecar_{key}_required")
    summary = sidecar.get("status_summary")
    if not isinstance(summary, dict):
        failures.append("sidecar_status_summary_required")
        return failures
    required_summary = {
        "validation",
        "handoff_verify",
        "handoff_safety",
        "blocked_count",
        "fail_count",
    }
    missing = sorted(required_summary - set(summary))
    failures.extend(f"sidecar_status_summary_missing: {key}" for key in missing)
    for key in ("validation", "handoff_verify", "handoff_safety"):
        if key in summary and summary[key] not in VALID_STATUSES:
            failures.append(f"sidecar_status_invalid: {key}")
    for key in ("blocked_count", "fail_count"):
        if key in summary and not isinstance(summary[key], int):
            failures.append(f"sidecar_status_count_invalid: {key}")
    return failures


def forbidden_package_entries(names: list[str]) -> list[str]:
    entries: list[str] = []
    for name in names:
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if any(part in FORBIDDEN_PACKAGE_ENTRY_PARTS for part in parts):
            entries.append(name)
            continue
        if is_reserved_handoff_evidence_path(name):
            entries.append(name)
    return entries


def verify_sidecar_schema(sidecar: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if sidecar.get("schema_version") != "handoff-sidecar/0.1":
        failures.append("sidecar_schema_invalid")
    allowed = {
        "schema_version",
        "zip_filename",
        "handoff_package_path",
        "handoff_sidecar_path",
        "zip_sha256",
        "artifact_manifest_sha256",
        "branch",
        "base",
        "final_head",
        "generated_at_utc",
        "status_summary",
    }
    if set(sidecar) - allowed:
        failures.append("sidecar_schema_invalid")
    return failures


def verify_artifact_manifest(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    try:
        raw = archive.read("artifact_manifest.json")
        manifest = json.loads(raw.decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["artifact_manifest_json_invalid"]
    if hashlib.sha256(raw).hexdigest() != sidecar.get("artifact_manifest_sha256"):
        failures.append("artifact_manifest_sha256_mismatch")
    if not isinstance(manifest, dict):
        return failures + ["artifact_manifest_must_be_object"]
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return failures + ["artifact_manifest_entries_required"]
    expected: dict[str, dict[str, object]] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            failures.append("artifact_manifest_entry_invalid")
            continue
        expected[item["path"]] = item
    name_set = set(archive.namelist())
    if set(expected) | {"artifact_manifest.json"} != name_set:
        failures.append("artifact_manifest_zip_entries_mismatch")
    for name, item in expected.items():
        try:
            raw_entry = archive.read(name)
        except KeyError:
            continue
        if len(raw_entry) != item.get("size"):
            failures.append(f"entry_size_mismatch: {name}")
        if hashlib.sha256(raw_entry).hexdigest() != item.get("sha256"):
            failures.append(f"entry_sha256_mismatch: {name}")
    return failures


def verify_metadata_entries(archive: ZipFile, sidecar: dict[str, object]) -> list[str]:
    failures: list[str] = []
    try:
        task_payload = json.loads(archive.read("task_metadata.json").decode("utf-8"))
        validation_payload = json.loads(
            archive.read("validation_evidence.json").decode("utf-8")
        )
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["metadata_json_decode_failed"]
    if not isinstance(task_payload, dict):
        failures.append("task_metadata_must_be_object")
    if not isinstance(validation_payload, dict):
        failures.append("validation_evidence_must_be_object")
    if isinstance(task_payload, dict):
        failures.extend(verify_task_metadata_schema(task_payload))
        failures.extend(
            metadata_consistency_failures("task_metadata", task_payload, sidecar)
        )
    if isinstance(validation_payload, dict):
        failures.extend(verify_validation_evidence_schema(validation_payload))
        failures.extend(
            metadata_consistency_failures(
                "validation_evidence",
                validation_payload,
                sidecar,
            )
        )
        failures.extend(validation_sidecar_status_failures(validation_payload, sidecar))
    return failures


def verify_validation_evidence_schema(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    forbidden = FORBIDDEN_VALIDATION_KEYS.intersection(payload)
    failures.extend(
        f"validation_evidence_forbidden_field: {key}"
        for key in sorted(forbidden)
    )
    rows = payload.get("checks")
    if not isinstance(rows, list):
        failures.append("validation_evidence_checks_missing")
        return failures
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"validation_evidence_check_not_object: {index}")
            continue
        extra = sorted(set(row) - VALIDATION_ROW_KEYS)
        failures.extend(
            f"validation_evidence_row_forbidden_field: {index}:{key}"
            for key in extra
        )
        for key in ("name", "status", "evidence"):
            if key not in row:
                failures.append(f"validation_evidence_row_missing: {index}:{key}")
    return failures


def verify_task_metadata_schema(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    required = {"schema_version", "mode", "slug", "branch", "base", "final_head"}
    missing = sorted(key for key in required if not payload.get(key))
    failures.extend(f"task_metadata_schema_missing: {key}" for key in missing)
    if payload.get("schema_version") != "handoff-task-metadata/0.1":
        failures.append("task_metadata_schema_version_mismatch")
    mode = str(payload["mode"]).strip() if "mode" in payload else ""
    if mode not in {"evidence", "pr_closure"}:
        failures.append("task_metadata_mode_invalid")
    if mode == "pr_closure" and not payload.get("pr_url"):
        failures.append("task_metadata_schema_missing: pr_url")
    forbidden = FORBIDDEN_TASK_KEYS.intersection(payload)
    failures.extend(
        f"task_metadata_forbidden_field: {key}"
        for key in sorted(forbidden)
    )
    if mode == "evidence" and "pr_url" in payload:
        failures.append("task_metadata_evidence_forbidden_field: pr_url")
    return failures


def metadata_consistency_failures(
    label: str,
    payload: dict[str, object],
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    expected = {
        "branch": sidecar.get("branch"),
        "final_head": sidecar.get("final_head"),
        "head_commit_after_task": sidecar.get("final_head"),
        "base": sidecar.get("base"),
        "base_commit_before_task": sidecar.get("base"),
    }
    for key, value in expected.items():
        if value is not None and key in payload and payload[key] != value:
            failures.append(f"{label}_{key}_mismatch")
    return failures


def validation_sidecar_status_failures(
    validation_payload: dict[str, object],
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    status_summary = sidecar.get("status_summary")
    if not isinstance(status_summary, dict):
        return failures
    row_status = validation_status_map(validation_payload)
    validation_status = aggregate_status(row_status.values())
    if status_summary.get("validation") != validation_status:
        failures.append("validation_status_summary_mismatch")
    pairs = {
        "handoff_verify": "handoff_verify",
        "handoff_safety": "handoff_safety_scan",
    }
    for summary_key, check_name in pairs.items():
        if (
            check_name in row_status
            and status_summary.get(summary_key) != row_status[check_name]
        ):
            failures.append(f"status_summary_{summary_key}_mismatch")
    blocked_count = sum(1 for status in row_status.values() if status == "BLOCKED")
    fail_count = sum(1 for status in row_status.values() if status == "FAIL")
    if status_summary.get("blocked_count") != blocked_count:
        failures.append("status_summary_blocked_count_mismatch")
    if status_summary.get("fail_count") != fail_count:
        failures.append("status_summary_fail_count_mismatch")
    return failures


def validation_status_map(payload: dict[str, object]) -> dict[str, str]:
    rows = payload["checks"]
    if not isinstance(rows, list):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        status = str(row.get("status") or "").strip().upper()
        if isinstance(name, str) and status in VALID_STATUSES:
            result[name] = status
    return result


def aggregate_status(statuses: object) -> str:
    values = (
        tuple(statuses)
        if not isinstance(statuses, dict)
        else tuple(statuses.values())
    )
    if not values:
        return "NOT_RUN"
    for status in ("FAIL", "BLOCKED", "NOT_RUN"):
        if status in values:
            return status
    return "PASS"


def verify_handoff_evidence_json(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    for name in archive.namelist():
        if not (name.startswith("handoff_evidence/") and name.endswith(".json")):
            continue
        try:
            payload = json.loads(archive.read(name).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            failures.append(f"handoff_evidence_json_invalid: {name}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"handoff_evidence_must_be_object: {name}")
            continue
        failures.extend(handoff_evidence_forbidden_field_failures(name, payload))
        if name == GENERATION_EVIDENCE:
            failures.extend(verify_generation_evidence(payload, sidecar, archive))
        if name == SOURCE_TREE_SCAN_EVIDENCE:
            failures.extend(verify_source_tree_scan_evidence(payload, sidecar))
        if name.endswith("verification-matrix.json"):
            failures.extend(verify_verification_matrix(payload))
    if VERIFICATION_MATRIX_EVIDENCE not in archive.namelist():
        failures.append("verification_matrix_missing")
    return failures


def handoff_evidence_forbidden_field_failures(
    name: str,
    payload: dict[str, object],
) -> list[str]:
    return handoff_evidence_forbidden_key_failures(name, payload)


def verify_generation_evidence(
    payload: dict[str, object],
    sidecar: dict[str, object],
    archive: ZipFile,
) -> list[str]:
    required = {
        "mode",
        "base",
        "final_head",
        "branch",
        "zip_filename",
        "zip_sha256_source",
        "bundle_sha256",
        "compare_evidence_path",
        "source_tree_scan_path",
        "validation_status",
    }
    failures = [
        f"generation_evidence_missing: {key}"
        for key in sorted(required)
        if key not in payload
    ]
    if payload.get("schema_version") != "handoff-generation/0.1":
        failures.append("generation_evidence_schema_version_mismatch")
    if payload.get("mode") not in {"evidence", "pr_closure"}:
        failures.append("generation_evidence_mode_invalid")
    if payload.get("validation_status") not in VALID_STATUSES:
        failures.append("generation_evidence_validation_status_invalid")
    if payload.get("zip_sha256_source") != "sidecar":
        failures.append("generation_zip_sha256_source_mismatch")
    if payload.get("source_tree_scan_path") != SOURCE_TREE_SCAN_EVIDENCE:
        failures.append("generation_source_tree_scan_path_mismatch")
    compare_evidence_path = payload.get("compare_evidence_path")
    if compare_evidence_path is not None and not isinstance(compare_evidence_path, str):
        failures.append("generation_compare_evidence_path_invalid")
    if isinstance(compare_evidence_path, str) and compare_evidence_path:
        if compare_evidence_path not in archive.namelist():
            failures.append("generation_compare_evidence_missing")
        elif compare_evidence_path == BASELINE_FINAL_COMPARE_EVIDENCE:
            failures.extend(
                verify_baseline_final_compare(
                    json_payload_from_archive(archive, compare_evidence_path),
                    sidecar=sidecar,
                )
            )
    for key in ("base", "branch", "final_head", "zip_filename"):
        if payload.get(key) != sidecar.get(key):
            failures.append(f"generation_evidence_sidecar_mismatch: {key}")
    try:
        bundle_sha256 = hashlib.sha256(archive.read("final-branch.bundle")).hexdigest()
    except KeyError:
        failures.append("generation_bundle_missing")
    else:
        if payload.get("bundle_sha256") != bundle_sha256:
            failures.append("generation_bundle_sha256_mismatch")
    return failures


def json_payload_from_archive(
    archive: ZipFile,
    name: str,
) -> dict[str, object]:
    try:
        payload = json.loads(archive.read(name).decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def verify_baseline_final_compare(
    payload: dict[str, object],
    *,
    sidecar: dict[str, object] | None = None,
) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_version") != "qrp-baseline-final-compare-summary/0.1":
        failures.append("baseline_final_compare_schema_version_mismatch")
    for key in (
        "actual_overall_status",
        "raw_overall_status",
        "policy_adjusted_overall_status",
        "overall_status",
        "old_baseline_compare_status",
        "refreshed_prediction_baseline_compare_status",
        "validation_decision",
        "final_head_full",
    ):
        if key in payload:
            failures.append(f"baseline_final_compare_flat_alias_present: {key}")
    status = payload.get("status")
    if not isinstance(status, dict):
        failures.append("baseline_final_compare_status_missing")
        status = {}
    if status.get("policy_adjusted") != "PASS":
        failures.append("baseline_final_compare_overall_status_not_pass")
    if status.get("validation_decision") != "PASS":
        failures.append("baseline_final_compare_validation_decision_not_pass")
    if "raw" not in status:
        failures.append("baseline_final_compare_raw_status_missing")
    comparison_groups = payload.get("comparison_groups")
    if not isinstance(comparison_groups, dict):
        failures.append("baseline_final_compare_groups_missing")
        comparison_groups = {}
    old_group = comparison_groups.get(OLD_BASELINE_GROUP)
    if not isinstance(old_group, dict) or old_group.get("status") != "FAIL_EXPECTED":
        failures.append("baseline_final_compare_old_status_mismatch")
    refreshed_group = comparison_groups.get(REFRESHED_PREDICTION_BASELINE_GROUP)
    if not isinstance(refreshed_group, dict) or refreshed_group.get("status") != "PASS":
        failures.append("baseline_final_compare_refreshed_status_mismatch")
    if payload.get("contract_failures"):
        failures.append("baseline_final_compare_contract_failures_present")
    if not is_full_sha(payload.get("baseline_commit_full")):
        failures.append("baseline_final_compare_baseline_commit_full_missing")
    if not is_full_sha(payload.get("final_head")):
        failures.append("baseline_final_compare_final_head_missing")
    final_smoke_commit = payload.get("final_smoke_commit")
    if not is_full_sha(final_smoke_commit):
        failures.append("baseline_final_compare_final_smoke_commit_missing")
    if payload.get("git_status_before_smoke") != "clean":
        failures.append("baseline_final_compare_git_status_before_smoke_not_clean")
    if (
        payload.get("git_status_after_smoke")
        != "clean_or_only_ignored_run_outputs"
    ):
        failures.append("baseline_final_compare_git_status_after_smoke_not_clean")
    if sidecar is not None and final_smoke_commit != sidecar.get("final_head"):
        failures.append("baseline_final_compare_final_smoke_commit_mismatch")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, dict):
        return [*failures, "baseline_final_compare_comparisons_missing"]
    grouped = comparisons_by_group(comparisons)
    for group in BASELINE_FINAL_COMPARE_GROUPS:
        if not grouped[group]:
            failures.append(f"baseline_final_compare_missing_required_group: {group}")
    for label, comparison in grouped[OLD_BASELINE_GROUP].items():
        if comparison.get("status") != "FAIL_EXPECTED":
            failures.append(f"baseline_final_compare_old_status_not_expected: {label}")
        if comparison.get("reason_code") != EXPECTED_FAILURE_REASON:
            failures.append(f"baseline_final_compare_old_reason_mismatch: {label}")
        if comparison.get("failed_artifacts") != sorted(
            RANK_ENSEMBLE_PREDICTION_ARTIFACTS
        ):
            failures.append(f"baseline_final_compare_old_artifacts_mismatch: {label}")
    for label, comparison in grouped[REFRESHED_PREDICTION_BASELINE_GROUP].items():
        if comparison.get("status") != "PASS":
            failures.append(
                f"baseline_final_compare_refreshed_status_not_pass: {label}"
            )
            continue
        prediction = comparison.get("prediction_summary", {})
        if not isinstance(prediction, dict):
            failures.append(f"baseline_final_compare_prediction_missing: {label}")
            continue
        for artifact in RANK_ENSEMBLE_PREDICTION_ARTIFACTS:
            if not artifact_family_present(
                prediction.get("artifact_ids", ()),
                artifact,
            ):
                failures.append(
                    "baseline_final_compare_refreshed_missing_artifact: "
                    f"{label}:{artifact}"
                )
        if prediction.get("failed_artifacts"):
            failures.append(
                f"baseline_final_compare_refreshed_failed_artifacts: {label}"
            )
    return failures


def is_full_sha(value: object) -> bool:
    return isinstance(value, str) and bool(FULL_SHA_RE.fullmatch(value))


def comparisons_by_group(
    comparisons: dict[object, object],
) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {
        group: {} for group in BASELINE_FINAL_COMPARE_GROUPS
    }
    for label, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            continue
        group = comparison.get("comparison_group")
        if group in grouped:
            grouped[str(group)][str(label)] = comparison
    return grouped


def artifact_family_present(artifact_ids: object, expected_family: str) -> bool:
    if not isinstance(artifact_ids, list):
        return False
    return any(
        artifact_matches_family(str(artifact_id), expected_family)
        for artifact_id in artifact_ids
    )


def artifact_matches_family(artifact_id: str, expected_family: str) -> bool:
    return bool(
        re.fullmatch(
            rf"{re.escape(expected_family)}_[A-Za-z0-9_]+_[0-9]+d_prediction_[0-9]{{4}}",
            artifact_id,
        )
    )


def verify_source_tree_scan_evidence(
    payload: dict[str, object],
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_version") != "handoff-source-tree-scan/0.1":
        failures.append("source_tree_scan_schema_version_mismatch")
    if payload.get("status") not in VALID_STATUSES:
        failures.append("source_tree_scan_status_invalid")
    if payload.get("status") != "PASS":
        failures.append("source_tree_scan_not_pass")
    for key in (
        "scanned_head",
        "expected_head",
        "finding_count",
        "findings",
        "scan_groups",
    ):
        if key not in payload:
            failures.append(f"source_tree_scan_missing: {key}")
    scan_groups = payload.get("scan_groups")
    if not isinstance(scan_groups, dict):
        failures.append("source_tree_scan_groups_invalid")
    else:
        expected_groups = {
            "retired_terms",
            "retired_governance_paths",
            "active_test_literals",
            "governance_truth",
            "sensitive_negative_literals",
        }
        actual_groups = set(scan_groups)
        if actual_groups != expected_groups:
            failures.append("source_tree_scan_groups_unexpected")
        for group in expected_groups:
            group_payload = scan_groups.get(group)
            if not isinstance(group_payload, dict):
                failures.append(f"source_tree_scan_group_missing: {group}")
                continue
            if (
                "status" not in group_payload
                or "finding_count" not in group_payload
                or "findings" not in group_payload
            ):
                failures.append(f"source_tree_scan_group_shape_invalid: {group}")
                continue
            findings = group_payload.get("findings")
            if not isinstance(findings, list):
                failures.append(f"source_tree_scan_group_findings_invalid: {group}")
                continue
            if group_payload.get("status") != "PASS":
                failures.append(f"source_tree_scan_group_not_pass: {group}")
            if group_payload.get("finding_count") != len(findings):
                failures.append(f"source_tree_scan_group_count_mismatch: {group}")
            if group_payload.get("finding_count") != 0:
                failures.append(f"source_tree_scan_group_findings_present: {group}")
    if payload.get("scanned_head") != payload.get("expected_head"):
        failures.append("source_tree_scan_head_mismatch")
    final_head = sidecar.get("final_head")
    if payload.get("expected_head") != final_head:
        failures.append("source_tree_scan_sidecar_expected_head_mismatch")
    if payload.get("scanned_head") != final_head:
        failures.append("source_tree_scan_sidecar_scanned_head_mismatch")
    if payload.get("scanned_tree") != "final-branch.bundle clone":
        failures.append("source_tree_scan_tree_mismatch")
    if payload.get("finding_count") != 0:
        failures.append("source_tree_scan_findings_present")
    findings = payload.get("findings")
    if isinstance(findings, list) and payload.get("finding_count") != len(findings):
        failures.append("source_tree_scan_finding_count_mismatch")
    return failures


def verify_verification_matrix(payload: dict[str, object]) -> list[str]:
    rows = payload.get("checks")
    if not isinstance(rows, list):
        return ["verification_matrix_checks_missing"]
    by_name = {
        row.get("name"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    failures: list[str] = []
    for name in ("handoff_verify", "handoff_safety_scan"):
        row = by_name.get(name)
        if row is None:
            failures.append(f"verification_matrix_required_missing: {name}")
            continue
        if row.get("required") is not True or row.get("deterministic") is not True:
            failures.append(f"verification_matrix_required_flag_invalid: {name}")
    return failures


def verify_final_branch_bundle(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    try:
        branch = str(sidecar["branch"])
        final_head = str(sidecar["final_head"])
    except KeyError:
        return ["sidecar_branch_or_final_head_missing"]
    with tempfile.TemporaryDirectory(prefix="handoff-verify-") as temp:
        bundle = Path(temp) / "final-branch.bundle"
        try:
            bundle.write_bytes(archive.read("final-branch.bundle"))
        except KeyError:
            return ["missing_final_branch_bundle"]
        clone_dir = Path(temp) / "clone"
        clone = subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle), str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            return ["bundle_clone_failed"]
        failures.extend(verify_tracked_handoff_sources(clone_dir))
        failures.extend(
            verify_payloads_against_tracked_schemas(clone_dir, archive, sidecar)
        )
        head = git_text(clone_dir, "rev-parse", "HEAD").strip()
        show_head = git_text(clone_dir, "show", "-s", "--format=%H", "HEAD").strip()
        if head != final_head or show_head != final_head:
            failures.append("bundle_clone_head_mismatch")
            return failures
        scan = source_tree_scan_from_clone(clone_dir, final_head=final_head)
        if scan.get("status") != "PASS":
            failures.append("source_tree_scan_bundle_not_pass")
        if scan.get("finding_count") != 0:
            failures.append("source_tree_scan_bundle_findings_present")
        findings = scan.get("findings")
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, dict) and isinstance(finding.get("rule"), str):
                    failures.append(f"source_tree_scan_bundle_rule: {finding['rule']}")
        if scan.get("scanned_head") != final_head:
            failures.append("source_tree_scan_bundle_head_mismatch")
        subject = git_text(clone_dir, "show", "-s", "--format=%s", "HEAD").strip()
        if "synthetic" in subject.lower():
            failures.append("bundle_head_synthetic_marker")
        base_commit = sidecar.get("base")
        if base_commit is None:
            failures.append("base_required_for_bundle_context")
    return failures


def verify_tracked_handoff_sources(repo: Path) -> list[str]:
    failures: list[str] = []
    for relative in TRACKED_HANDOFF_SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            failures.append(f"tracked_handoff_source_missing: {relative}")
    schema_expectations = {
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/task_metadata.schema.json"
        ): "handoff-task-metadata/0.1",
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/sidecar.schema.json"
        ): "handoff-sidecar/0.1",
    }
    for relative, schema_version in schema_expectations.items():
        path = repo / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"tracked_schema_json_invalid: {relative}")
            continue
        properties = payload.get("properties")
        version = (
            properties.get("schema_version")
            if isinstance(properties, dict)
            else {}
        )
        if not isinstance(version, dict) or version.get("const") != schema_version:
            failures.append(f"tracked_schema_version_invalid: {relative}")
    return failures


def verify_payloads_against_tracked_schemas(
    repo: Path,
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    try:
        task_payload = json.loads(archive.read("task_metadata.json").decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        failures.append("task_metadata_schema_from_bundle_input_invalid")
        task_payload = {}
    schema_checks = (
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/sidecar.schema.json",
            sidecar,
            "sidecar_schema_from_bundle",
        ),
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/task_metadata.schema.json",
            task_payload,
            "task_metadata_schema_from_bundle",
        ),
    )
    for relative, payload, label in schema_checks:
        schema_path = repo / relative
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"{label}_source_invalid")
            continue
        failures.extend(validate_payload_with_tracked_schema(payload, schema, label))
    return failures


def validate_payload_with_tracked_schema(
    payload: dict[str, object],
    schema: dict[str, object],
    label: str,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return [f"{label}_payload_not_object"]
    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in payload:
                failures.append(f"{label}_required_missing: {key}")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        if schema.get("additionalProperties") is False:
            extra = sorted(set(payload) - set(properties))
            failures.extend(f"{label}_additional_property: {key}" for key in extra)
        for key, raw_rule in properties.items():
            if key not in payload or not isinstance(raw_rule, dict):
                continue
            if "const" in raw_rule and payload[key] != raw_rule["const"]:
                failures.append(f"{label}_const_mismatch: {key}")
            enum = raw_rule.get("enum")
            if isinstance(enum, list) and payload[key] not in enum:
                failures.append(f"{label}_enum_mismatch: {key}")
    not_rule = schema.get("not")
    if isinstance(not_rule, dict):
        forbidden = not_rule.get("required")
        if isinstance(forbidden, list) and all(key in payload for key in forbidden):
            failures.append(f"{label}_forbidden_required_present")
    return failures


def verify_git_status(archive: ZipFile) -> list[str]:
    try:
        status = archive.read("git_status.txt").decode("utf-8").strip()
    except (KeyError, UnicodeDecodeError):
        return ["git_status_invalid"]
    return [] if status == "clean" else ["git_status_not_clean"]


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def contains_local_absolute_path_text(text: str) -> bool:
    markers = (
        "E:/",
        "D:/",
        "C:/",
        "/mnt/data",
        "/home/",
        "/Users/",
        "\\Users\\",
        "\\\\Users\\\\",
    )
    return any(marker in text for marker in markers)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
