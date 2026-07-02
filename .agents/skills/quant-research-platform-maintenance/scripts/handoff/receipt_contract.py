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
    failures = required(payload, ("schema_id", "status", "wheel", "sdist", "install", "import", "content_scan"), "package_smoke")
    if payload.get("schema_id") != "qrp-package-smoke-receipt":
        failures.append("package_smoke_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("package_smoke_status_invalid")
    for key in ("wheel", "sdist"):
        item = payload.get(key)
        if not isinstance(item, Mapping):
            failures.append(f"package_smoke_artifact_invalid: {key}")
            continue
        if not isinstance(item.get("filename"), str):
            failures.append(f"package_smoke_filename_invalid: {key}")
        if not sha_or_empty(item.get("sha256")):
            failures.append(f"package_smoke_sha256_invalid: {key}")
    step_statuses: list[str] = []
    for key in ("install", "import"):
        item = payload.get(key)
        if not isinstance(item, Mapping) or item.get("status") not in VALID_STATUS_TOKENS:
            failures.append(f"package_smoke_step_status_invalid: {key}")
        else:
            step_statuses.append(str(item.get("status")))
    scan = payload.get("content_scan")
    if not isinstance(scan, Mapping):
        failures.append("package_smoke_content_scan_invalid")
    else:
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
    if (
        step_statuses
        and payload.get("status") in VALID_STATUS_TOKENS
        and payload.get("status") != _aggregate_validation_matrix_status(step_statuses)
    ):
        failures.append("package_smoke_status_mismatch")
    return failures


def validate_hard_scan_receipt(payload: Mapping[str, object]) -> list[str]:
    failures = required(
        payload,
        (
            "schema_id",
            "status",
            "scanned_head",
            "scanner_source_sha256",
            "scan_runs",
            "classified_exceptions",
        ),
        "hard_scan",
    )
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
            for key in ("scan_id", "rule_set_id", "result_sha256"):
                if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                    failures.append(f"hard_scan_run_missing: {index}:{key}")
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
    if not isinstance(payload.get("classified_exceptions"), list):
        failures.append("hard_scan_classified_exceptions_invalid")
    if statuses and payload.get("status") != _aggregate_validation_matrix_status(statuses):
        failures.append("hard_scan_status_mismatch")
    return failures


def validate_check_receipts(payload: Mapping[str, object]) -> list[str]:
    failures = required(payload, ("schema_id", "status", "checks"), "check_receipts")
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
    failures: list[str] = []
    for key in (
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
    ):
        if key not in row:
            failures.append(f"check_receipt_missing: {index}:{key}")
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
    elif status == "FAIL":
        if not isinstance(exit_code, int) or exit_code == 0:
            failures.append(f"check_receipt_fail_exit_code_invalid: {index}")
    elif status == "NOT_RUN":
        if not str(row.get("not_run_reason") or "").strip():
            failures.append(f"check_receipt_not_run_reason_missing: {index}")
    elif status == "BLOCKED":
        if not str(row.get("blocked_reason") or "").strip():
            failures.append(f"check_receipt_blocked_reason_missing: {index}")
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
    failures = required(
        payload,
        (
            "schema_id",
            "status",
            "review_lanes",
            "finding_count",
            "true_issue_count",
            "false_positive_count",
            "re_review_status",
            "findings",
        ),
        "advisory_review",
    )
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


def sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def sha_or_empty(value: object) -> bool:
    return value == "" or sha(value)
