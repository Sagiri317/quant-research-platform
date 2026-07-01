#!/usr/bin/env python3
"""Collect compact handoff receipts into the package-ready receipts directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

VALID = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
REQUIRED = {
    "package-smoke-receipt.json",
    "hard-scan-receipt.json",
    "check-receipts.json",
    "private-smoke-status.json",
}
OLD_NAMES = {
    "package-smoke.json",
    "hard-scan-classification.json",
    "test-checks.json",
    "subagent-review-findings.json",
    "subagent-review-reconciliation.json",
    "smoke-compare-status.json",
    "source-tree-scan.json",
    "verification-matrix.json",
}
FORBIDDEN_KEYS = {"stdout", "stderr", "raw_output", "raw_grep_output", "grep_hits"}
COUNT_KEYS = {
    "passed",
    "failed",
    "skipped",
    "deselected",
    "errors",
    "xfailed",
    "xpassed",
    "collected",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--check-jsonl", default=None)
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.check_jsonl:
        write_check_receipts(Path(args.check_jsonl), out_dir / "check-receipts.json")
    for path in input_dir.glob("*.json"):
        if path.name == "check-receipts.json":
            target = out_dir / path.name
        elif path.name in REQUIRED or path.name == "advisory-review-receipt.json":
            target = out_dir / path.name
        else:
            continue
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
    failures = validate_receipts(out_dir)
    index = {
        "schema_id": "qrp-handoff-receipt-index",
        "status": "PASS" if not failures else "FAIL",
        "receipts": sorted(path.name for path in out_dir.glob("*.json")),
        "failures": failures,
    }
    (out_dir / "receipt-index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


def write_check_receipts(source: Path, target: Path) -> None:
    checks = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    statuses = [str(row.get("status")) for row in checks if isinstance(row, dict)]
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": aggregate(statuses),
        "checks": checks,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_receipts(root: Path) -> list[str]:
    failures: list[str] = []
    for name in OLD_NAMES:
        if (root / name).exists():
            failures.append(f"old_receipt_name_present: {name}")
    for name in REQUIRED:
        if not (root / name).is_file():
            failures.append(f"required_receipt_missing: {name}")
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"receipt_json_invalid: {path.name}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"receipt_not_object: {path.name}")
            continue
        schema_id = str(payload.get("schema_id") or "")
        if not schema_id.startswith("qrp-"):
            failures.append(f"receipt_schema_id_invalid: {path.name}")
        validator = {
            "package-smoke-receipt.json": validate_package_smoke_receipt,
            "hard-scan-receipt.json": validate_hard_scan_receipt,
            "check-receipts.json": validate_check_receipts,
            "private-smoke-status.json": validate_private_smoke_status,
            "advisory-review-receipt.json": validate_advisory_review_receipt,
        }.get(path.name, validate_generic_receipt)
        failures.extend(validator(payload))
        failures.extend(forbidden_key_failures(path.name, payload))
    return failures


def validate_generic_receipt(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if payload.get("status") not in VALID:
        failures.append("receipt_status_invalid")
    return failures


def validate_package_smoke_receipt(payload: dict[str, object]) -> list[str]:
    failures = _required(
        payload,
        (
            "schema_id",
            "status",
            "wheel",
            "sdist",
            "install",
            "import",
            "content_scan",
        ),
        "package_smoke",
    )
    if payload.get("schema_id") != "qrp-package-smoke-receipt":
        failures.append("package_smoke_schema_id_invalid")
    if payload.get("status") not in VALID:
        failures.append("package_smoke_status_invalid")
    for key in ("wheel", "sdist"):
        item = payload.get(key)
        if not isinstance(item, dict):
            failures.append(f"package_smoke_artifact_invalid: {key}")
            continue
        if not isinstance(item.get("filename"), str):
            failures.append(f"package_smoke_filename_invalid: {key}")
        if not _sha_or_empty(item.get("sha256")):
            failures.append(f"package_smoke_sha256_invalid: {key}")
    content_scan = payload.get("content_scan")
    if not isinstance(content_scan, dict):
        failures.append("package_smoke_content_scan_invalid")
    else:
        if content_scan.get("status") not in VALID:
            failures.append("package_smoke_content_scan_status_invalid")
        if not isinstance(content_scan.get("forbidden_member_count"), int):
            failures.append("package_smoke_forbidden_member_count_invalid")
        if set(content_scan.get("scanned_artifact_kinds") or ()) != {
            "wheel",
            "sdist",
            "install_target",
        }:
            failures.append("package_smoke_scanned_artifact_kinds_invalid")
    return failures


def validate_hard_scan_receipt(payload: dict[str, object]) -> list[str]:
    failures = _required(
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
    if payload.get("status") not in VALID:
        failures.append("hard_scan_status_invalid")
    if not _sha(payload.get("scanner_source_sha256")):
        failures.append("hard_scan_scanner_source_sha256_invalid")
    runs = payload.get("scan_runs")
    if not isinstance(runs, list) or not runs:
        failures.append("hard_scan_runs_invalid")
    else:
        for index, row in enumerate(runs):
            if not isinstance(row, dict):
                failures.append(f"hard_scan_run_invalid: {index}")
                continue
            for key in ("scan_id", "rule_set_id", "result_sha256"):
                if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                    failures.append(f"hard_scan_run_missing: {index}:{key}")
            if not _sha(row.get("result_sha256")):
                failures.append(f"hard_scan_result_sha256_invalid: {index}")
    if not isinstance(payload.get("classified_exceptions"), list):
        failures.append("hard_scan_classified_exceptions_invalid")
    return failures


def validate_check_receipts(payload: dict[str, object]) -> list[str]:
    failures = _required(payload, ("schema_id", "status", "checks"), "check_receipts")
    if payload.get("schema_id") != "qrp-check-receipts":
        failures.append("check_receipts_schema_id_invalid")
    if payload.get("status") not in VALID:
        failures.append("check_receipts_status_invalid")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        failures.append("check_receipts_checks_invalid")
        return failures
    for index, row in enumerate(checks):
        if not isinstance(row, dict):
            failures.append(f"check_receipt_invalid: {index}")
            continue
        for key in (
            "schema_id",
            "command_id",
            "scope",
            "command",
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
        if "test_count" in row:
            failures.append(f"check_receipt_test_count_forbidden: {index}")
        counts = row.get("counts")
        if counts is not None:
            if not isinstance(counts, dict) or set(counts) != COUNT_KEYS:
                failures.append(f"check_receipt_counts_invalid: {index}")
            elif any(
                not isinstance(value, int) or value < 0
                for value in counts.values()
            ):
                failures.append(f"check_receipt_counts_invalid: {index}")
    return failures


def validate_private_smoke_status(payload: dict[str, object]) -> list[str]:
    failures = _required(
        payload,
        (
            "schema_id",
            "status",
            "command_id",
            "exit_code",
            "run_level_overall",
            "private_smoke_run_level",
            "private_smoke_child_checks",
            "compare",
            "full_horizon",
            "child_statuses",
            "run_json_sha256",
            "row_level_data_included",
        ),
        "private_smoke",
    )
    if payload.get("schema_id") != "qrp-private-smoke-status":
        failures.append("private_smoke_schema_id_invalid")
    for key in (
        "status",
        "run_level_overall",
        "private_smoke_run_level",
        "private_smoke_child_checks",
        "compare",
        "full_horizon",
    ):
        if payload.get(key) not in VALID:
            failures.append(f"private_smoke_status_invalid: {key}")
    if payload.get("row_level_data_included") is not False:
        failures.append("private_smoke_row_level_data_included")
    if payload.get("status") == "NOT_RUN" and not payload.get("not_run_reason"):
        failures.append("private_smoke_not_run_reason_missing")
    if payload.get("status") == "BLOCKED" and not payload.get("blocked_reason"):
        failures.append("private_smoke_blocked_reason_missing")
    return failures


def validate_advisory_review_receipt(payload: dict[str, object]) -> list[str]:
    failures = _required(
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
    if payload.get("status") not in VALID:
        failures.append("advisory_review_status_invalid")
    if payload.get("re_review_status") not in VALID:
        failures.append("advisory_review_re_review_status_invalid")
    if not isinstance(payload.get("findings"), list):
        failures.append("advisory_review_findings_invalid")
    return failures


def _required(
    payload: dict[str, object],
    keys: tuple[str, ...],
    prefix: str,
) -> list[str]:
    return [f"{prefix}_missing: {key}" for key in keys if key not in payload]


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _sha_or_empty(value: object) -> bool:
    return value == "" or _sha(value)


def forbidden_key_failures(name: str, payload: object, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_KEYS:
                failures.append(f"receipt_forbidden_raw_field: {name}:{dotted}")
            failures.extend(forbidden_key_failures(name, value, dotted))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            failures.extend(forbidden_key_failures(name, value, f"{prefix}.{index}"))
    return failures


def aggregate(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUN" in statuses:
        return "NOT_RUN"
    return "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
