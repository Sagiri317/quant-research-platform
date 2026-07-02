"""Compact receipt schema and validators for QRP handoff tooling.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path

try:
    from quant_research_platform.policy.check_status import (
        VALID_STATUS_TOKENS,
        aggregate_validation_matrix_status as _aggregate_validation_matrix_status,
    )
except ModuleNotFoundError:  # pragma: no cover - copied handoff tools may omit src.
    VALID_STATUS_TOKENS = frozenset({"PASS", "FAIL", "NOT_RUN", "BLOCKED"})

    def _aggregate_validation_matrix_status(statuses: list[str]) -> str:
        if "FAIL" in statuses:
            return "FAIL"
        if "BLOCKED" in statuses:
            return "BLOCKED"
        if "NOT_RUN" in statuses:
            return "NOT_RUN"
        return "PASS"

try:
    from handoff_contract import (
        ADVISORY_REVIEW_RECEIPT_ENTRY,
        CHECK_RECEIPTS_ENTRY,
        FORBIDDEN_HANDOFF_ENTRIES,
        HARD_SCAN_RECEIPT_ENTRY,
        PACKAGE_SMOKE_RECEIPT_ENTRY,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback.
    from .handoff_contract import (
        ADVISORY_REVIEW_RECEIPT_ENTRY,
        CHECK_RECEIPTS_ENTRY,
        FORBIDDEN_HANDOFF_ENTRIES,
        HARD_SCAN_RECEIPT_ENTRY,
        PACKAGE_SMOKE_RECEIPT_ENTRY,
    )


PACKAGE_SMOKE_RECEIPT_NAME = Path(PACKAGE_SMOKE_RECEIPT_ENTRY).name
HARD_SCAN_RECEIPT_NAME = Path(HARD_SCAN_RECEIPT_ENTRY).name
CHECK_RECEIPTS_NAME = Path(CHECK_RECEIPTS_ENTRY).name
ADVISORY_REVIEW_RECEIPT_NAME = Path(ADVISORY_REVIEW_RECEIPT_ENTRY).name

REQUIRED_RECEIPT_FILES = frozenset(
    {
        PACKAGE_SMOKE_RECEIPT_NAME,
        HARD_SCAN_RECEIPT_NAME,
        CHECK_RECEIPTS_NAME,
    }
)
OPTIONAL_RECEIPT_FILES = frozenset({ADVISORY_REVIEW_RECEIPT_NAME})
FORBIDDEN_RECEIPT_FILES = frozenset(Path(entry).name for entry in FORBIDDEN_HANDOFF_ENTRIES)
FORBIDDEN_RECEIPT_KEYS = frozenset(
    {"stdout", "stderr", "raw_output", "raw_grep_output", "grep_hits"}
)
PACKAGE_SMOKE_RECEIPT_KEYS = frozenset(
    {"schema_id", "status", "wheel", "sdist", "install", "import", "content_scan"}
)
PACKAGE_SMOKE_ARTIFACT_KEYS = frozenset({"filename", "sha256"})
PACKAGE_SMOKE_INSTALL_KEYS = frozenset({"target_kind", "no_deps", "status"})
PACKAGE_SMOKE_IMPORT_KEYS = frozenset({"status"})
PACKAGE_SMOKE_CONTENT_SCAN_KEYS = frozenset(
    {"status", "forbidden_member_count", "scanned_artifact_kinds", "examples"}
)
HARD_SCAN_RECEIPT_KEYS = frozenset(
    {
        "schema_id",
        "status",
        "scanned_head",
        "scanner_source_sha256",
        "scan_runs",
        "classified_exceptions",
    }
)
HARD_SCAN_RUN_KEYS = frozenset(
    {
        "scan_id",
        "rule_set_id",
        "match_count",
        "actual_violation_count",
        "deferred_count",
        "negative_fixture_count",
        "false_positive_count",
        "result_sha256",
        "replayable_from_bundle",
    }
)
HARD_SCAN_CLASSIFIED_EXCEPTION_KEYS = frozenset(
    {"scan_id", "classification", "owner", "rationale", "reference"}
)
HARD_SCAN_CLASSIFICATION_VALUES = frozenset(
    {"false_positive", "deferred", "allowed_exception"}
)
CHECK_RECEIPTS_KEYS = frozenset({"schema_id", "status", "checks", "not_run_reason"})
CHECK_RECEIPT_ROW_KEYS = frozenset(
    {
        "schema_id",
        "command_id",
        "scope",
        "command",
        "command_argv",
        "status",
        "exit_code",
        "duration_seconds",
        "counts",
        "stdout_sha256",
        "stderr_sha256",
        "output_summary",
        "failure_excerpt",
        "not_run_reason",
        "blocked_reason",
    }
)
ADVISORY_REVIEW_RECEIPT_KEYS = frozenset(
    {
        "schema_id",
        "status",
        "review_lanes",
        "finding_count",
        "true_issue_count",
        "false_positive_count",
        "re_review_status",
        "findings",
    }
)
ADVISORY_FINDING_KEYS = frozenset(
    {"finding_id", "classification", "resolved", "summary"}
)
ADVISORY_FINDING_CLASSIFICATIONS = frozenset({"true_issue", "false_positive"})
COUNT_KEYS = frozenset(
    {
        "passed",
        "failed",
        "skipped",
        "deselected",
        "errors",
        "xfailed",
        "xpassed",
        "collected",
    }
)


def validate_receipt_file_set(root: Path) -> list[str]:
    failures: list[str] = []
    if not root.is_dir():
        return [f"receipt_dir_missing: {root}"]
    names = {path.name for path in root.glob("*.json")}
    failures.extend(
        f"forbidden_receipt_file: {name}"
        for name in sorted(names & FORBIDDEN_RECEIPT_FILES)
    )
    failures.extend(
        f"required_receipt_missing: {name}"
        for name in sorted(REQUIRED_RECEIPT_FILES - names)
    )
    allowed = REQUIRED_RECEIPT_FILES | OPTIONAL_RECEIPT_FILES
    failures.extend(f"unexpected_receipt_file: {name}" for name in sorted(names - allowed))
    for path in sorted(root.glob("*.json")):
        if path.name not in allowed:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            failures.append(f"receipt_json_invalid: {path.name}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"receipt_not_object: {path.name}")
            continue
        failures.extend(validate_receipt_payload(path.name, payload))
    return failures


def load_receipt_file_set(root: Path) -> dict[str, dict[str, object]]:
    failures = validate_receipt_file_set(root)
    if failures:
        raise SystemExit(";".join(failures))
    receipts: dict[str, dict[str, object]] = {}
    for name, entry in (
        (PACKAGE_SMOKE_RECEIPT_NAME, PACKAGE_SMOKE_RECEIPT_ENTRY),
        (HARD_SCAN_RECEIPT_NAME, HARD_SCAN_RECEIPT_ENTRY),
        (CHECK_RECEIPTS_NAME, CHECK_RECEIPTS_ENTRY),
        (ADVISORY_REVIEW_RECEIPT_NAME, ADVISORY_REVIEW_RECEIPT_ENTRY),
    ):
        path = root / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            assert isinstance(payload, dict)
            receipts[entry] = payload
    return receipts


def validate_receipt_payload(name: str, payload: Mapping[str, object]) -> list[str]:
    failures = forbidden_receipt_key_failures(name, payload)
    if name == PACKAGE_SMOKE_RECEIPT_NAME:
        failures.extend(validate_package_smoke_receipt(payload))
    elif name == HARD_SCAN_RECEIPT_NAME:
        failures.extend(validate_hard_scan_receipt(payload))
    elif name == CHECK_RECEIPTS_NAME:
        failures.extend(validate_check_receipts(payload))
    elif name == ADVISORY_REVIEW_RECEIPT_NAME:
        failures.extend(validate_advisory_review_receipt(payload))
    else:
        failures.append(f"receipt_name_unsupported: {name}")
    return failures


def validate_package_smoke_receipt(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, PACKAGE_SMOKE_RECEIPT_KEYS, "package_smoke")
    if payload.get("schema_id") != "qrp-package-smoke-receipt":
        failures.append("package_smoke_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("package_smoke_status_invalid")
    for key in ("wheel", "sdist"):
        item = payload.get(key)
        if not isinstance(item, Mapping):
            failures.append(f"package_smoke_artifact_invalid: {key}")
            continue
        failures.extend(
            exact_keys(item, PACKAGE_SMOKE_ARTIFACT_KEYS, f"package_smoke_{key}")
        )
        if not isinstance(item.get("filename"), str):
            failures.append(f"package_smoke_filename_invalid: {key}")
        if not sha_or_empty(item.get("sha256")):
            failures.append(f"package_smoke_sha256_invalid: {key}")
    step_statuses: list[str] = []
    install = payload.get("install")
    if not isinstance(install, Mapping):
        failures.append("package_smoke_step_status_invalid: install")
    else:
        failures.extend(exact_keys(install, PACKAGE_SMOKE_INSTALL_KEYS, "package_smoke_install"))
        if install.get("status") not in VALID_STATUS_TOKENS:
            failures.append("package_smoke_step_status_invalid: install")
        else:
            step_statuses.append(str(install.get("status")))
        if not isinstance(install.get("target_kind"), str) or not str(
            install.get("target_kind")
        ).strip():
            failures.append("package_smoke_install_target_kind_invalid")
        if not isinstance(install.get("no_deps"), bool):
            failures.append("package_smoke_install_no_deps_invalid")
    imported = payload.get("import")
    if not isinstance(imported, Mapping):
        failures.append("package_smoke_step_status_invalid: import")
    else:
        failures.extend(exact_keys(imported, PACKAGE_SMOKE_IMPORT_KEYS, "package_smoke_import"))
        if imported.get("status") not in VALID_STATUS_TOKENS:
            failures.append("package_smoke_step_status_invalid: import")
        else:
            step_statuses.append(str(imported.get("status")))
    scan = payload.get("content_scan")
    if not isinstance(scan, Mapping):
        failures.append("package_smoke_content_scan_invalid")
    else:
        failures.extend(
            exact_keys(scan, PACKAGE_SMOKE_CONTENT_SCAN_KEYS, "package_smoke_content_scan")
        )
        scan_status = scan.get("status")
        if scan_status not in VALID_STATUS_TOKENS:
            failures.append("package_smoke_content_scan_status_invalid")
        else:
            step_statuses.append(str(scan_status))
        if not isinstance(scan.get("forbidden_member_count"), int):
            failures.append("package_smoke_forbidden_member_count_invalid")
        elif scan.get("forbidden_member_count", 0) > 0 and scan_status == "PASS":
            failures.append("package_smoke_content_scan_forbidden_members_mismatch")
        if set(scan.get("scanned_artifact_kinds") or ()) != {
            "wheel",
            "sdist",
            "install_target",
        }:
            failures.append("package_smoke_scanned_artifact_kinds_invalid")
        if not isinstance(scan.get("examples"), list):
            failures.append("package_smoke_content_scan_examples_invalid")
    if (
        step_statuses
        and payload.get("status") in VALID_STATUS_TOKENS
        and payload.get("status") != _aggregate_validation_matrix_status(step_statuses)
    ):
        failures.append("package_smoke_status_mismatch")
    return failures


def validate_hard_scan_receipt(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, HARD_SCAN_RECEIPT_KEYS, "hard_scan")
    if payload.get("schema_id") != "qrp-hard-scan-receipt":
        failures.append("hard_scan_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("hard_scan_status_invalid")
    if not isinstance(payload.get("scanned_head"), str) or not str(
        payload.get("scanned_head")
    ).strip():
        failures.append("hard_scan_scanned_head_invalid")
    if not sha(payload.get("scanner_source_sha256")):
        failures.append("hard_scan_scanner_source_sha256_invalid")
    runs = payload.get("scan_runs")
    statuses: list[str] = []
    if not isinstance(runs, list) or not runs:
        failures.append("hard_scan_runs_invalid")
    else:
        for index, row in enumerate(runs):
            if not isinstance(row, Mapping):
                failures.append(f"hard_scan_run_invalid: {index}")
                continue
            failures.extend(exact_keys(row, HARD_SCAN_RUN_KEYS, f"hard_scan_run:{index}"))
            for key in ("scan_id", "rule_set_id", "result_sha256"):
                if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                    failures.append(f"hard_scan_run_missing: {index}:{key}")
            for key in (
                "match_count",
                "actual_violation_count",
                "deferred_count",
                "negative_fixture_count",
                "false_positive_count",
            ):
                if not isinstance(row.get(key), int) or int(row.get(key)) < 0:
                    failures.append(f"hard_scan_run_count_invalid: {index}:{key}")
            if not sha(row.get("result_sha256")):
                failures.append(f"hard_scan_result_sha256_invalid: {index}")
            if row.get("replayable_from_bundle") is not True:
                failures.append(f"hard_scan_receipt_not_replayable: {index}")
            if not isinstance(row.get("actual_violation_count"), int):
                failures.append(f"hard_scan_actual_violation_count_invalid: {index}")
            elif row.get("actual_violation_count") != 0:
                statuses.append("FAIL")
            else:
                statuses.append("PASS")
    exceptions = payload.get("classified_exceptions")
    if not isinstance(exceptions, list):
        failures.append("hard_scan_classified_exceptions_invalid")
    else:
        for index, row in enumerate(exceptions):
            if not isinstance(row, Mapping):
                failures.append(f"hard_scan_classified_exception_invalid: {index}")
                continue
            failures.extend(
                exact_keys(
                    row,
                    HARD_SCAN_CLASSIFIED_EXCEPTION_KEYS,
                    f"hard_scan_classified_exception:{index}",
                )
            )
            for key in HARD_SCAN_CLASSIFIED_EXCEPTION_KEYS:
                if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                    failures.append(
                        f"hard_scan_classified_exception_string_invalid: {index}:{key}"
                    )
            classification = row.get("classification")
            if classification not in HARD_SCAN_CLASSIFICATION_VALUES:
                failures.append(
                    f"hard_scan_classified_exception_classification_invalid: {index}"
                )
            if classification == "actual_violation":
                failures.append(f"hard_scan_classified_exception_actual_violation: {index}")
    if statuses and payload.get("status") != _aggregate_validation_matrix_status(statuses):
        failures.append("hard_scan_status_mismatch")
    return failures


def validate_check_receipts(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(
        payload,
        CHECK_RECEIPTS_KEYS,
        "check_receipts",
        required=frozenset({"schema_id", "status", "checks"}),
    )
    if payload.get("schema_id") != "qrp-check-receipts":
        failures.append("check_receipts_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("check_receipts_status_invalid")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        failures.append("check_receipts_checks_invalid")
        return failures
    if not checks and payload.get("status") == "PASS":
        failures.append("check_receipts_empty_pass_invalid")
    if not checks and payload.get("status") != "NOT_RUN":
        failures.append("check_receipts_empty_status_invalid")
    if not checks and payload.get("status") == "NOT_RUN" and not str(
        payload.get("not_run_reason") or ""
    ).strip():
        failures.append("check_receipts_empty_not_run_reason_missing")
    if checks and "not_run_reason" in payload:
        failures.append("check_receipts_not_run_reason_unexpected")
    statuses: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(checks):
        if not isinstance(row, Mapping):
            failures.append(f"check_receipt_invalid: {index}")
            continue
        failures.extend(validate_check_receipt_row(row, index=index))
        command_id = row.get("command_id")
        if isinstance(command_id, str):
            if command_id in seen:
                failures.append(f"check_receipt_duplicate_command_id: {command_id}")
            seen.add(command_id)
        status = row.get("status")
        if isinstance(status, str) and status in VALID_STATUS_TOKENS:
            statuses.append(status)
    if statuses and payload.get("status") != _aggregate_validation_matrix_status(statuses):
        failures.append("check_receipts_status_mismatch")
    return failures


def validate_check_receipt_row(
    row: Mapping[str, object],
    *,
    index: int = 0,
) -> list[str]:
    failures = exact_keys(
        row,
        CHECK_RECEIPT_ROW_KEYS,
        f"check_receipt:{index}",
        required=frozenset(
            {
                "schema_id",
                "command_id",
                "scope",
                "command",
                "command_argv",
                "status",
                "exit_code",
                "duration_seconds",
                "counts",
                "stdout_sha256",
                "stderr_sha256",
                "output_summary",
                "failure_excerpt",
            }
        ),
    )
    if row.get("schema_id") != "qrp-check-receipt":
        failures.append(f"check_receipt_schema_id_invalid: {index}")
    for key in ("command_id", "scope", "command"):
        if key in row and (
            not isinstance(row.get(key), str) or not str(row.get(key)).strip()
        ):
            failures.append(f"check_receipt_string_invalid: {index}:{key}")
    command_argv = row.get("command_argv")
    if not isinstance(command_argv, list) or not command_argv or not all(
        isinstance(item, str) and item for item in command_argv
    ):
        failures.append(f"check_receipt_command_argv_invalid: {index}")
    elif row.get("command") != shlex.join(command_argv):
        failures.append(f"check_receipt_command_mismatch: {index}")
    status = row.get("status")
    if status not in VALID_STATUS_TOKENS:
        failures.append(f"check_receipt_status_invalid: {index}")
    exit_code = row.get("exit_code")
    if status == "PASS":
        if exit_code != 0:
            failures.append(f"check_receipt_pass_exit_code_invalid: {index}")
        if "not_run_reason" in row or "blocked_reason" in row:
            failures.append(f"check_receipt_reason_unexpected: {index}")
    elif status == "FAIL":
        if not isinstance(exit_code, int) or exit_code == 0:
            failures.append(f"check_receipt_fail_exit_code_invalid: {index}")
        if "not_run_reason" in row or "blocked_reason" in row:
            failures.append(f"check_receipt_reason_unexpected: {index}")
    elif status == "NOT_RUN":
        if exit_code is not None:
            failures.append(f"check_receipt_not_run_exit_code_invalid: {index}")
        if not str(row.get("not_run_reason") or "").strip():
            failures.append(f"check_receipt_not_run_reason_missing: {index}")
        if "blocked_reason" in row:
            failures.append(f"check_receipt_reason_unexpected: {index}")
    elif status == "BLOCKED":
        if exit_code is not None:
            failures.append(f"check_receipt_blocked_exit_code_invalid: {index}")
        if not str(row.get("blocked_reason") or "").strip():
            failures.append(f"check_receipt_blocked_reason_missing: {index}")
        if "not_run_reason" in row:
            failures.append(f"check_receipt_reason_unexpected: {index}")
    counts = row.get("counts")
    if counts is not None:
        if not isinstance(counts, Mapping) or set(counts) != COUNT_KEYS:
            failures.append(f"check_receipt_counts_invalid: {index}")
        elif any(not isinstance(value, int) or value < 0 for value in counts.values()):
            failures.append(f"check_receipt_counts_invalid: {index}")
        else:
            outcome_total = sum(
                int(counts[key])
                for key in COUNT_KEYS
                if key not in {"collected", "deselected"}
            )
            if counts["collected"] == 0 and outcome_total != 0:
                failures.append(f"check_receipt_counts_collected_mismatch: {index}")
            if status == "PASS" and (counts["failed"] != 0 or counts["errors"] != 0):
                failures.append(f"check_receipt_pass_counts_invalid: {index}")
            if status == "NOT_RUN" and any(int(value) != 0 for value in counts.values()):
                failures.append(f"check_receipt_not_run_counts_invalid: {index}")
            if status == "BLOCKED" and any(int(value) != 0 for value in counts.values()):
                failures.append(f"check_receipt_blocked_counts_invalid: {index}")
    duration = row.get("duration_seconds")
    if not isinstance(duration, int | float) or duration < 0:
        failures.append(f"check_receipt_duration_invalid: {index}")
    for key in ("stdout_sha256", "stderr_sha256"):
        if not sha(row.get(key)):
            failures.append(f"check_receipt_sha256_invalid: {index}:{key}")
    for key in ("output_summary", "failure_excerpt"):
        if key in row and not isinstance(row.get(key), str):
            failures.append(f"check_receipt_string_invalid: {index}:{key}")
    return failures


def validate_advisory_review_receipt(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, ADVISORY_REVIEW_RECEIPT_KEYS, "advisory_review")
    if payload.get("schema_id") != "qrp-advisory-review-receipt":
        failures.append("advisory_review_schema_id_invalid")
    for key in ("status", "re_review_status"):
        if payload.get(key) not in VALID_STATUS_TOKENS:
            failures.append(f"advisory_review_status_invalid: {key}")
    for key in ("finding_count", "true_issue_count", "false_positive_count"):
        if not isinstance(payload.get(key), int) or int(payload.get(key)) < 0:
            failures.append(f"advisory_review_count_invalid: {key}")
    if not isinstance(payload.get("review_lanes"), list):
        failures.append("advisory_review_lanes_invalid")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        failures.append("advisory_review_findings_invalid")
        return failures
    true_issue_count = 0
    false_positive_count = 0
    unresolved_true_issue = False
    for index, row in enumerate(findings):
        if not isinstance(row, Mapping):
            failures.append(f"advisory_finding_invalid: {index}")
            continue
        failures.extend(exact_keys(row, ADVISORY_FINDING_KEYS, f"advisory_finding:{index}"))
        for key in ("finding_id", "summary"):
            if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                failures.append(f"advisory_finding_string_invalid: {index}:{key}")
        classification = row.get("classification")
        if classification not in ADVISORY_FINDING_CLASSIFICATIONS:
            failures.append(f"advisory_finding_classification_invalid: {index}")
            continue
        if not isinstance(row.get("resolved"), bool):
            failures.append(f"advisory_finding_resolved_invalid: {index}")
        if classification == "true_issue":
            true_issue_count += 1
            if row.get("resolved") is not True:
                unresolved_true_issue = True
        elif classification == "false_positive":
            false_positive_count += 1
    if isinstance(payload.get("finding_count"), int) and payload.get(
        "finding_count"
    ) != len(findings):
        failures.append("advisory_review_finding_count_mismatch")
    if isinstance(payload.get("true_issue_count"), int) and payload.get(
        "true_issue_count"
    ) != true_issue_count:
        failures.append("advisory_review_true_issue_count_mismatch")
    if isinstance(payload.get("false_positive_count"), int) and payload.get(
        "false_positive_count"
    ) != false_positive_count:
        failures.append("advisory_review_false_positive_count_mismatch")
    if payload.get("status") == "PASS" and payload.get("re_review_status") != "PASS":
        failures.append("advisory_review_pass_re_review_status_mismatch")
    if payload.get("status") == "PASS" and unresolved_true_issue:
        failures.append("advisory_review_pass_with_unresolved_true_issue")
    return failures


def aggregate_receipt_status(receipts: Mapping[str, Mapping[str, object]]) -> str:
    statuses = [
        str(payload.get("status"))
        for payload in receipts.values()
        if str(payload.get("status")) in VALID_STATUS_TOKENS
    ]
    return _aggregate_validation_matrix_status(statuses)


def check_statuses_by_command_id(
    check_receipts: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    rows = check_receipts.get("checks")
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("command_id"), str):
            result[row["command_id"]] = row
    return result


def forbidden_receipt_key_failures(
    name: str,
    payload: object,
    prefix: str = "",
) -> list[str]:
    failures: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            dotted = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FORBIDDEN_RECEIPT_KEYS:
                failures.append(f"receipt_forbidden_raw_field: {name}:{dotted}")
            failures.extend(forbidden_receipt_key_failures(name, value, dotted))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            failures.extend(forbidden_receipt_key_failures(name, value, f"{prefix}.{index}"))
    return failures


def required(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
    prefix: str,
) -> list[str]:
    return [f"{prefix}_missing: {key}" for key in keys if key not in payload]


def exact_keys(
    payload: Mapping[str, object],
    allowed: frozenset[str],
    prefix: str,
    *,
    required: frozenset[str] | None = None,
) -> list[str]:
    required_keys = allowed if required is None else required
    if ":" in prefix:
        label, _, index = prefix.partition(":")
        failures = [
            f"{label}_missing: {index}:{key}"
            for key in sorted(required_keys - set(payload))
        ]
        failures.extend(
            f"{label}_unapproved_field: {index}:{key}"
            for key in sorted(set(payload) - allowed)
        )
        return failures
    failures = [f"{prefix}_missing: {key}" for key in sorted(required_keys - set(payload))]
    failures.extend(
        f"{prefix}_unapproved_field: {key}" for key in sorted(set(payload) - allowed)
    )
    return failures


def sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha_or_empty(value: object) -> bool:
    return value == "" or sha(value)
