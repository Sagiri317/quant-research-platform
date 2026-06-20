"""Verify a generic QRP handoff zip with one external JSON sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from handoff_safety import scan_handoff_zip  # noqa: E402

VALID_STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
REQUIRED_ZIP_ENTRIES = {
    "artifact_manifest.json",
    "final-branch.bundle",
    "git_status.txt",
    "task_metadata.json",
    "validation_evidence.json",
}
DIFF_PATCH_ENTRY = "diff" + ".patch"
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
            if DIFF_PATCH_ENTRY in name_set:
                failures.append("forbidden_entry: " + DIFF_PATCH_ENTRY)
            pr_entries = {"pr_body.md", "review_threads.json"}
            present_pr_entries = pr_entries.intersection(name_set)
            if present_pr_entries and present_pr_entries != pr_entries:
                missing_pr_entries = sorted(pr_entries - present_pr_entries)
                failures.extend(
                    f"pr_closure_missing_required_entry: {name}"
                    for name in missing_pr_entries
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
            if not any(
                name.startswith("handoff_evidence/") and name.endswith(".json")
                for name in names
            ):
                failures.append("missing_handoff_evidence_json")
            failures.extend(verify_artifact_manifest(archive, sidecar))
            failures.extend(verify_metadata_entries(archive, sidecar))
            if pr_entries.issubset(name_set):
                failures.extend(verify_review_thread_closure(archive, sidecar))
                failures.extend(verify_pr_body_statuses(archive, sidecar))
            failures.extend(verify_handoff_evidence_json(archive))
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
        "zip_logical_path",
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
    logical = sidecar.get("zip_logical_path")
    if not isinstance(logical, str) or "\\" in logical or not logical.endswith(
        "/" + zip_path.name
    ):
        failures.append("sidecar_zip_logical_path_invalid")
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


def verify_sidecar_schema(sidecar: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if sidecar.get("schema_version") != "handoff-sidecar/0.1":
        failures.append("sidecar_schema_invalid")
    allowed = {
        "schema_version",
        "zip_filename",
        "zip_logical_path",
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
        failures.extend(
            metadata_consistency_failures(
                "validation_evidence",
                validation_payload,
                sidecar,
            )
        )
        failures.extend(validation_sidecar_status_failures(validation_payload, sidecar))
        failures.extend(validation_command_failures(validation_payload))
    return failures


def verify_task_metadata_schema(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    required = {"schema_version", "slug", "branch", "base", "pr_url"}
    missing = sorted(key for key in required if not payload.get(key))
    failures.extend(f"task_metadata_schema_missing: {key}" for key in missing)
    if payload.get("schema_version") != "handoff-task-metadata/0.1":
        failures.append("task_metadata_schema_version_mismatch")
    if "goal_status" in payload:
        failures.append("task_metadata_goal_status_not_allowed")
    return failures


def metadata_consistency_failures(
    label: str,
    payload: dict[str, object],
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    expected = {
        "branch": sidecar.get("branch"),
        "final_sha": sidecar.get("final_head"),
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
    rows = payload.get("checks", payload.get("validation_table", []))
    if not isinstance(rows, list):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name", row.get("check"))
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


def verify_review_thread_closure(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    del sidecar
    failures: list[str] = []
    try:
        payload = json.loads(archive.read("review_threads.json").decode("utf-8"))
    except (KeyError, json.JSONDecodeError):
        return ["review_threads_json_decode_failed"]
    if not isinstance(payload, dict):
        return ["review_threads_must_be_object"]
    status = str(payload.get("status") or "").strip().upper()
    counts_available = payload.get("counts_available")
    total_count = payload.get("total_count")
    unresolved_count = payload.get("unresolved_count")
    if status == "BLOCKED":
        if counts_available is not False:
            failures.append("review_threads_blocked_counts_available_not_false")
        if total_count is not None or unresolved_count is not None:
            failures.append("review_threads_blocked_counts_must_be_null")
        evidence = str(payload.get("evidence") or "").lower()
        if "counts unknown" not in evidence and "counts are unknown" not in evidence:
            failures.append("review_threads_blocked_counts_unknown_evidence_missing")
    elif status == "PASS":
        if counts_available is not True:
            failures.append("review_threads_pass_counts_available_required")
        if not isinstance(total_count, int) or not isinstance(unresolved_count, int):
            failures.append("review_threads_pass_counts_must_be_int")
    threads = payload.get("threads")
    if not isinstance(threads, list):
        failures.append("review_threads_table_missing")
        threads = []
    required_fields = {
        "thread_id",
        "path",
        "comment_id",
        "classification",
        "resolution_status",
        "evidence",
        "resolved_by_tool",
        "manual_action_required",
    }
    for index, row in enumerate(threads):
        if not isinstance(row, dict):
            failures.append(f"review_thread_row_not_object: {index}")
            continue
        missing = sorted(required_fields.difference(row))
        if missing:
            failures.append(
                "review_thread_row_missing_fields: "
                + str(index)
                + ":"
                + ",".join(missing)
            )
        forbidden_body_fields = {
            "body",
            "comment_body",
            "comment_bodies",
            "comments",
        }.intersection(row)
        if forbidden_body_fields:
            failures.append(
                "review_thread_row_contains_comment_body: "
                + str(index)
                + ":"
                + ",".join(sorted(forbidden_body_fields))
            )
    if (
        status == "PASS"
        and isinstance(total_count, int)
        and len(threads) != total_count
    ):
        failures.append("review_threads_total_count_mismatch")
    return failures


def validation_command_failures(payload: dict[str, object]) -> list[str]:
    rows = payload.get("checks", payload.get("validation_table", []))
    if not isinstance(rows, list):
        return []
    failures: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", row.get("check")) or "").strip()
        status = str(row.get("status") or "").strip().upper()
        command = str(row.get("command") or "").strip()
        if status != "PASS":
            continue
        if not command:
            failures.append(f"validation_command_missing: {name}")
            continue
        if _validation_command_has_placeholder(command):
            failures.append(f"validation_command_placeholder: {name}")
        if contains_local_absolute_path_text(command):
            failures.append(f"validation_command_local_absolute_path: {name}")
    return failures


def _validation_command_has_placeholder(command: str) -> bool:
    lowered = command.lower()
    if "targeted reports/workflows/storage/compute/interfaces suites" in lowered:
        return True
    if "<" in command or ">" in command:
        return True
    return any(
        marker in lowered
        for marker in (
            "<zip>",
            "<json>",
            "<official-rendered-pr-body>",
        )
    )


def verify_pr_body_statuses(archive: ZipFile, sidecar: dict[str, object]) -> list[str]:
    failures: list[str] = []
    try:
        body = archive.read("pr_body.md").decode("utf-8")
        validation = json.loads(
            archive.read("validation_evidence.json").decode("utf-8")
        )
        review_threads = json.loads(
            archive.read("review_threads.json").decode("utf-8")
        )
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["pr_body_validation_inputs_invalid"]
    if not isinstance(validation, dict):
        return ["validation_evidence_must_be_object"]
    if "- PR: ``" in body or "MISSING_PR_URL" in body:
        failures.append("pr_body_pr_url_missing")
    if "{'sha':" in body:
        failures.append("pr_body_commit_list_dict_repr")
    for name, status in validation_status_map(validation).items():
        expected = f"| `{name}` | `{status}` |"
        if expected not in body:
            failures.append(f"pr_body_validation_row_missing: {name}")
    if (
        isinstance(review_threads, dict)
        and review_threads.get("threads")
        and "full evidence: review_threads.json" not in body
    ):
        failures.append("pr_body_review_threads_full_evidence_marker_missing")
    if "- start head: ``" in body or "- base head: ``" in body:
        failures.append("pr_body_base_head_missing")
    if "- review zip: ``" in body:
        failures.append("pr_body_review_zip_missing")
    if "- external evidence JSON: ``" in body:
        failures.append("pr_body_external_json_missing")
    if "zip sidecar | `NOT_RUN`" in body:
        failures.append("pr_body_sidecar_not_run_with_sidecar_present")
    if contains_local_absolute_path_text(body):
        failures.append("pr_body_local_absolute_path")
    summary = sidecar.get("status_summary")
    if isinstance(summary, dict):
        expected = (
            "| zip sidecar | `{}` | validation={}; handoff_verify={}; "
            "handoff_safety={}; blocked_count={}; fail_count={} |"
        ).format(
            aggregate_status(
                str(value)
                for value in summary.values()
                if str(value) in VALID_STATUSES
            ),
            summary.get("validation"),
            summary.get("handoff_verify"),
            summary.get("handoff_safety"),
            summary.get("blocked_count"),
            summary.get("fail_count"),
        )
        if expected not in body:
            failures.append("pr_body_sidecar_summary_row_mismatch")
    if isinstance(review_threads, dict) and review_threads.get("status") == "BLOCKED":
        expected = (
            "| review threads | `BLOCKED` | live GitHub access unavailable; "
            "counts unknown |"
        )
        if expected not in body:
            failures.append("pr_body_review_threads_blocked_counts_unknown_missing")
    return failures


def verify_handoff_evidence_json(archive: ZipFile) -> list[str]:
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
        if name.endswith("verification-matrix.json"):
            failures.extend(verify_verification_matrix(payload))
    if "handoff_evidence/verification-matrix.json" not in archive.namelist():
        failures.append("verification_matrix_missing")
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
        "E:" + "/",
        "D:" + "/",
        "C:" + "/",
        "/" + "mnt/data",
        "/" + "home/",
        "/" + "Users/",
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
