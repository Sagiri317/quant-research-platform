"""Compact receipt schema and replay comparators for QRP handoff tooling.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
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
        CHECK_RECEIPTS_ENTRY,
        FORBIDDEN_HANDOFF_ENTRIES,
        HARD_SCAN_RECEIPT_ENTRY,
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        ProofClass,
        replay_command_spec,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback.
    from .handoff_contract import (
        CHECK_RECEIPTS_ENTRY,
        FORBIDDEN_HANDOFF_ENTRIES,
        HARD_SCAN_RECEIPT_ENTRY,
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        ProofClass,
        replay_command_spec,
    )


@dataclass(frozen=True)
class FieldSpec:
    proof_class: ProofClass
    required: bool = True


CHECK_RECEIPTS_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "status": FieldSpec(ProofClass.DERIVED_FROM_REPLAY),
    "checks": FieldSpec(ProofClass.REPLAYED),
}

CHECK_RECEIPT_ROW_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "command_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "scope": FieldSpec(ProofClass.METADATA_VALIDATED),
    "command_argv": FieldSpec(ProofClass.REPLAYED),
    "command": FieldSpec(ProofClass.REPLAYED),
    "status": FieldSpec(ProofClass.REPLAYED),
    "exit_code": FieldSpec(ProofClass.REPLAYED),
    "counts": FieldSpec(ProofClass.REPLAYED),
}

PACKAGE_SMOKE_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "status": FieldSpec(ProofClass.DERIVED_FROM_REPLAY),
    "build": FieldSpec(ProofClass.REPLAYED),
    "install": FieldSpec(ProofClass.REPLAYED),
    "import": FieldSpec(ProofClass.REPLAYED),
    "content_scan": FieldSpec(ProofClass.REPLAYED),
}

PACKAGE_SMOKE_BUILD_FIELD_SPECS = {
    "status": FieldSpec(ProofClass.REPLAYED),
    "artifact_kinds": FieldSpec(ProofClass.REPLAYED),
}

PACKAGE_SMOKE_INSTALL_FIELD_SPECS = {
    "status": FieldSpec(ProofClass.REPLAYED),
    "target_kind": FieldSpec(ProofClass.METADATA_VALIDATED),
    "no_deps": FieldSpec(ProofClass.METADATA_VALIDATED),
}

PACKAGE_SMOKE_IMPORT_FIELD_SPECS = {
    "status": FieldSpec(ProofClass.REPLAYED),
    "module": FieldSpec(ProofClass.METADATA_VALIDATED),
}

PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS = {
    "status": FieldSpec(ProofClass.REPLAYED),
    "forbidden_member_count": FieldSpec(ProofClass.REPLAYED),
    "scanned_artifact_kinds": FieldSpec(ProofClass.REPLAYED),
}

HARD_SCAN_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "status": FieldSpec(ProofClass.DERIVED_FROM_REPLAY),
    "scanned_head": FieldSpec(ProofClass.REPLAYED),
    "scanner_source_sha256": FieldSpec(ProofClass.REPLAYED),
    "scan_runs": FieldSpec(ProofClass.REPLAYED),
}

HARD_SCAN_RUN_FIELD_SPECS = {
    "scan_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "rule_set_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "match_count": FieldSpec(ProofClass.REPLAYED),
    "actual_violation_count": FieldSpec(ProofClass.REPLAYED),
    "result_sha256": FieldSpec(ProofClass.REPLAYED),
    "replayable_from_bundle": FieldSpec(ProofClass.METADATA_VALIDATED),
}

PACKAGE_SMOKE_RECEIPT_NAME = Path(PACKAGE_SMOKE_RECEIPT_ENTRY).name
HARD_SCAN_RECEIPT_NAME = Path(HARD_SCAN_RECEIPT_ENTRY).name
CHECK_RECEIPTS_NAME = Path(CHECK_RECEIPTS_ENTRY).name
REQUIRED_RECEIPT_FILES = frozenset(
    {PACKAGE_SMOKE_RECEIPT_NAME, HARD_SCAN_RECEIPT_NAME, CHECK_RECEIPTS_NAME}
)
FORBIDDEN_RECEIPT_FILES = frozenset(Path(entry).name for entry in FORBIDDEN_HANDOFF_ENTRIES)
FORBIDDEN_RECEIPT_FIELDS = frozenset(
    {
        "stdout",
        "stderr",
        "raw_output",
        "raw_grep_output",
        "grep_hits",
    }
)
COUNT_FIELDS = frozenset(
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
REQUIRED_CHECK_IDS = ("profile-smoke", "code-map")
PACKAGE_ARTIFACT_KINDS = ("wheel", "sdist")
PACKAGE_SCANNED_ARTIFACT_KINDS = ("wheel", "sdist", "install_target")
PACKAGE_INSTALL_TARGET_KIND = "temporary_no_deps_install"
PACKAGE_IMPORT_MODULE = "quant_research_platform"


def allowed_keys(specs: Mapping[str, FieldSpec]) -> frozenset[str]:
    return frozenset(specs)


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
    failures.extend(
        f"unexpected_receipt_file: {name}"
        for name in sorted(names - REQUIRED_RECEIPT_FILES)
    )
    for path in sorted(root.glob("*.json")):
        if path.name not in REQUIRED_RECEIPT_FILES:
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
    ):
        payload = json.loads((root / name).read_text(encoding="utf-8-sig"))
        assert isinstance(payload, dict)
        receipts[entry] = payload
    return receipts


def validate_receipt_payload(name: str, payload: Mapping[str, object]) -> list[str]:
    failures = forbidden_receipt_field_failures(name, payload)
    if name == PACKAGE_SMOKE_RECEIPT_NAME:
        failures.extend(validate_package_smoke_receipt(payload))
    elif name == HARD_SCAN_RECEIPT_NAME:
        failures.extend(validate_hard_scan_receipt(payload))
    elif name == CHECK_RECEIPTS_NAME:
        failures.extend(validate_check_receipts(payload))
    else:
        failures.append(f"receipt_name_unsupported: {name}")
    return failures


def validate_package_smoke_receipt(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, PACKAGE_SMOKE_FIELD_SPECS, "package_smoke")
    if payload.get("schema_id") != "qrp-package-smoke-receipt":
        failures.append("package_smoke_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("package_smoke_status_invalid")
    step_statuses: list[str] = []
    build = payload.get("build")
    if not isinstance(build, Mapping):
        failures.append("package_smoke_build_invalid")
    else:
        failures.extend(exact_keys(build, PACKAGE_SMOKE_BUILD_FIELD_SPECS, "package_smoke_build"))
        _append_step_status(build.get("status"), step_statuses, failures, "package_smoke_build")
        if list(build.get("artifact_kinds") or []) != list(PACKAGE_ARTIFACT_KINDS):
            failures.append("package_smoke_artifact_kinds_invalid")
    install = payload.get("install")
    if not isinstance(install, Mapping):
        failures.append("package_smoke_install_invalid")
    else:
        failures.extend(
            exact_keys(install, PACKAGE_SMOKE_INSTALL_FIELD_SPECS, "package_smoke_install")
        )
        _append_step_status(install.get("status"), step_statuses, failures, "package_smoke_install")
        if install.get("target_kind") != PACKAGE_INSTALL_TARGET_KIND:
            failures.append("package_smoke_install_target_kind_invalid")
        if install.get("no_deps") is not True:
            failures.append("package_smoke_install_no_deps_invalid")
    imported = payload.get("import")
    if not isinstance(imported, Mapping):
        failures.append("package_smoke_import_invalid")
    else:
        failures.extend(
            exact_keys(imported, PACKAGE_SMOKE_IMPORT_FIELD_SPECS, "package_smoke_import")
        )
        _append_step_status(imported.get("status"), step_statuses, failures, "package_smoke_import")
        if imported.get("module") != PACKAGE_IMPORT_MODULE:
            failures.append("package_smoke_import_module_invalid")
    scan = payload.get("content_scan")
    if not isinstance(scan, Mapping):
        failures.append("package_smoke_content_scan_invalid")
    else:
        failures.extend(
            exact_keys(
                scan,
                PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS,
                "package_smoke_content_scan",
            )
        )
        scan_status = scan.get("status")
        _append_step_status(scan_status, step_statuses, failures, "package_smoke_content_scan")
        count = scan.get("forbidden_member_count")
        if not isinstance(count, int) or count < 0:
            failures.append("package_smoke_forbidden_member_count_invalid")
        elif count > 0 and scan_status == "PASS":
            failures.append("package_smoke_content_scan_forbidden_members_mismatch")
        if list(scan.get("scanned_artifact_kinds") or []) != list(
            PACKAGE_SCANNED_ARTIFACT_KINDS
        ):
            failures.append("package_smoke_scanned_artifact_kinds_invalid")
    if (
        step_statuses
        and payload.get("status") in VALID_STATUS_TOKENS
        and payload.get("status") != _aggregate_validation_matrix_status(step_statuses)
    ):
        failures.append("package_smoke_status_mismatch")
    return failures


def validate_hard_scan_receipt(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, HARD_SCAN_FIELD_SPECS, "hard_scan")
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
            failures.extend(exact_keys(row, HARD_SCAN_RUN_FIELD_SPECS, f"hard_scan_run:{index}"))
            for key in ("scan_id", "rule_set_id", "result_sha256"):
                if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                    failures.append(f"hard_scan_run_missing: {index}:{key}")
            for key in ("match_count", "actual_violation_count"):
                if not isinstance(row.get(key), int) or int(row.get(key)) < 0:
                    failures.append(f"hard_scan_run_count_invalid: {index}:{key}")
            if not sha(row.get("result_sha256")):
                failures.append(f"hard_scan_result_sha256_invalid: {index}")
            if row.get("replayable_from_bundle") is not True:
                failures.append(f"hard_scan_receipt_not_replayable: {index}")
            if row.get("actual_violation_count") != 0:
                statuses.append("FAIL")
            else:
                statuses.append("PASS")
    if statuses and payload.get("status") != _aggregate_validation_matrix_status(statuses):
        failures.append("hard_scan_status_mismatch")
    return failures


def validate_check_receipts(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, CHECK_RECEIPTS_FIELD_SPECS, "check_receipts")
    if payload.get("schema_id") != "qrp-check-receipts":
        failures.append("check_receipts_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("check_receipts_status_invalid")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        failures.append("check_receipts_checks_invalid")
        return failures
    rows_by_id: dict[str, Mapping[str, object]] = {}
    statuses: list[str] = []
    for index, row in enumerate(checks):
        if not isinstance(row, Mapping):
            failures.append(f"check_receipt_invalid: {index}")
            continue
        failures.extend(validate_check_receipt_row(row, index=index))
        command_id = row.get("command_id")
        if isinstance(command_id, str):
            if command_id in rows_by_id:
                failures.append(f"check_receipt_duplicate_command_id: {command_id}")
            rows_by_id[command_id] = row
        status = row.get("status")
        if isinstance(status, str) and status in VALID_STATUS_TOKENS:
            statuses.append(status)
    missing = [command_id for command_id in REQUIRED_CHECK_IDS if command_id not in rows_by_id]
    failures.extend(f"check_receipts_missing_command: {command_id}" for command_id in missing)
    extra = sorted(set(rows_by_id) - set(REQUIRED_CHECK_IDS))
    failures.extend(f"check_receipts_unapproved_command: {command_id}" for command_id in extra)
    if statuses and payload.get("status") != _aggregate_validation_matrix_status(statuses):
        failures.append("check_receipts_status_mismatch")
    return failures


def validate_check_receipt_row(
    row: Mapping[str, object],
    *,
    index: int = 0,
) -> list[str]:
    failures = exact_keys(row, CHECK_RECEIPT_ROW_FIELD_SPECS, f"check_receipt:{index}")
    if row.get("schema_id") != "qrp-check-receipt":
        failures.append(f"check_receipt_schema_id_invalid: {index}")
    command_id = row.get("command_id")
    if not isinstance(command_id, str) or not command_id.strip():
        failures.append(f"check_receipt_string_invalid: {index}:command_id")
        command_id = ""
    for key in ("scope", "command"):
        if key in row and (
            not isinstance(row.get(key), str) or not str(row.get(key)).strip()
        ):
            failures.append(f"check_receipt_string_invalid: {index}:{key}")
    command_argv = row.get("command_argv")
    if not isinstance(command_argv, list) or not command_argv or not all(
        isinstance(item, str) and item for item in command_argv
    ):
        failures.append(f"check_receipt_command_argv_invalid: {index}")
    else:
        if row.get("command") != shlex.join(command_argv):
            failures.append(f"check_receipt_command_mismatch: {index}")
        if isinstance(command_id, str) and command_id in REQUIRED_CHECK_IDS:
            spec = replay_command_spec(command_id)
            if tuple(command_argv) != spec.command_argv:
                failures.append(f"check_receipt_command_argv_mismatch: {command_id}")
            if row.get("scope") != spec.scope:
                failures.append(f"check_receipt_scope_mismatch: {command_id}")
    status = row.get("status")
    if status not in {"PASS", "FAIL"}:
        failures.append(f"check_receipt_status_invalid: {index}")
    exit_code = row.get("exit_code")
    if status == "PASS" and exit_code != 0:
        failures.append(f"check_receipt_pass_exit_code_invalid: {index}")
    if status == "FAIL" and (not isinstance(exit_code, int) or exit_code == 0):
        failures.append(f"check_receipt_fail_exit_code_invalid: {index}")
    counts = row.get("counts")
    if command_id == "profile-smoke":
        failures.extend(validate_pytest_counts(counts, status=status, index=index))
    elif command_id == "code-map" and counts is not None:
        failures.append(f"check_receipt_counts_invalid: {index}")
    return failures


def validate_pytest_counts(
    counts: object,
    *,
    status: object,
    index: int,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(counts, Mapping) or set(counts) != COUNT_FIELDS:
        return [f"check_receipt_counts_invalid: {index}"]
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        return [f"check_receipt_counts_invalid: {index}"]
    outcome_total = sum(
        int(counts[key]) for key in COUNT_FIELDS if key not in {"collected", "deselected"}
    )
    if counts["collected"] == 0 and outcome_total != 0:
        failures.append(f"check_receipt_counts_collected_mismatch: {index}")
    if status == "PASS" and (counts["failed"] != 0 or counts["errors"] != 0):
        failures.append(f"check_receipt_pass_counts_invalid: {index}")
    return failures


def normalize_check_receipts(payload: Mapping[str, object]) -> dict[str, object]:
    _raise_if_invalid(validate_check_receipts(payload))
    checks = payload["checks"]
    assert isinstance(checks, list)
    return {
        "schema_id": "qrp-check-receipts",
        "status": str(payload["status"]),
        "checks": sorted(
            (dict(row) for row in checks if isinstance(row, Mapping)),
            key=lambda row: str(row["command_id"]),
        ),
    }


def normalize_package_smoke_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    _raise_if_invalid(validate_package_smoke_receipt(payload))
    return {
        "schema_id": "qrp-package-smoke-receipt",
        "status": str(payload["status"]),
        "build": dict(payload["build"]),  # type: ignore[arg-type]
        "install": dict(payload["install"]),  # type: ignore[arg-type]
        "import": dict(payload["import"]),  # type: ignore[arg-type]
        "content_scan": dict(payload["content_scan"]),  # type: ignore[arg-type]
    }


def normalize_hard_scan_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    scan_runs = payload.get("scan_runs")
    normalized_runs: list[dict[str, object]] = []
    if isinstance(scan_runs, list):
        for row in scan_runs:
            if not isinstance(row, Mapping):
                continue
            normalized_runs.append(
                {
                    "scan_id": row.get("scan_id"),
                    "rule_set_id": row.get("rule_set_id"),
                    "match_count": row.get("match_count"),
                    "actual_violation_count": row.get("actual_violation_count"),
                    "result_sha256": row.get("result_sha256"),
                    "replayable_from_bundle": row.get("replayable_from_bundle"),
                }
            )
    normalized = {
        "schema_id": payload.get("schema_id"),
        "status": payload.get("status"),
        "scanned_head": payload.get("scanned_head"),
        "scanner_source_sha256": payload.get("scanner_source_sha256"),
        "scan_runs": sorted(normalized_runs, key=lambda row: str(row.get("scan_id"))),
    }
    _raise_if_invalid(validate_hard_scan_receipt(normalized))
    return normalized


def compare_check_receipts(
    packaged: Mapping[str, object],
    replayed: Mapping[str, object],
) -> list[str]:
    return _compare_receipts(
        normalize_check_receipts,
        packaged,
        replayed,
        "check_receipts",
    )


def compare_package_smoke_receipt(
    packaged: Mapping[str, object],
    replayed: Mapping[str, object],
) -> list[str]:
    return _compare_receipts(
        normalize_package_smoke_receipt,
        packaged,
        replayed,
        "package_smoke",
    )


def compare_hard_scan_receipt(
    packaged: Mapping[str, object],
    replayed: Mapping[str, object],
) -> list[str]:
    return _compare_receipts(
        normalize_hard_scan_receipt,
        packaged,
        replayed,
        "hard_scan",
    )


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


def forbidden_receipt_field_failures(
    name: str,
    payload: object,
    prefix: str = "",
) -> list[str]:
    failures: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            dotted = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FORBIDDEN_RECEIPT_FIELDS:
                failures.append(f"receipt_forbidden_raw_field: {name}:{dotted}")
            failures.extend(forbidden_receipt_field_failures(name, value, dotted))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            failures.extend(forbidden_receipt_field_failures(name, value, f"{prefix}.{index}"))
    return failures


def exact_keys(
    payload: Mapping[str, object],
    specs: Mapping[str, FieldSpec],
    prefix: str,
) -> list[str]:
    allowed = allowed_keys(specs)
    if ":" in prefix:
        label, _, index = prefix.partition(":")
        failures = [f"{label}_missing: {index}:{key}" for key in sorted(allowed - set(payload))]
        failures.extend(
            f"{label}_unapproved_field: {index}:{key}"
            for key in sorted(set(payload) - allowed)
        )
        return failures
    failures = [f"{prefix}_missing: {key}" for key in sorted(allowed - set(payload))]
    failures.extend(f"{prefix}_unapproved_field: {key}" for key in sorted(set(payload) - allowed))
    return failures


def sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _append_step_status(
    value: object,
    statuses: list[str],
    failures: list[str],
    label: str,
) -> None:
    if value not in VALID_STATUS_TOKENS:
        failures.append(f"{label}_status_invalid")
    else:
        statuses.append(str(value))


def _compare_receipts(
    normalize,
    packaged: Mapping[str, object],
    replayed: Mapping[str, object],
    label: str,
) -> list[str]:
    try:
        packaged_normalized = normalize(packaged)
    except SystemExit as exc:
        return [f"{label}_packaged_invalid: {exc}"]
    try:
        replayed_normalized = normalize(replayed)
    except SystemExit as exc:
        return [f"{label}_replayed_invalid: {exc}"]
    if packaged_normalized != replayed_normalized:
        return [f"{label}_replay_mismatch"]
    return []


def _raise_if_invalid(failures: list[str]) -> None:
    if failures:
        raise SystemExit(";".join(failures))
