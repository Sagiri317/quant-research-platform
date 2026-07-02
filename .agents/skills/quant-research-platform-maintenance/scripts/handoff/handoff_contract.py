"""Handoff package layout and sidecar contract for QRP tooling.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

try:
    from quant_research_platform.policy.check_status import (
        VALID_STATUS_TOKENS,
        aggregate_validation_matrix_status as _aggregate_validation_matrix_status,
    )
except ModuleNotFoundError:  # pragma: no cover - copied handoff tools may omit src.
    VALID_STATUS_TOKENS = frozenset({"PASS", "FAIL", "NOT_RUN", "BLOCKED"})

    def _aggregate_validation_matrix_status(statuses: Iterable[str]) -> str:
        values = list(statuses)
        if "FAIL" in values:
            return "FAIL"
        if "BLOCKED" in values:
            return "BLOCKED"
        if "NOT_RUN" in values:
            return "NOT_RUN"
        return "PASS"


ARTIFACT_MANIFEST_ENTRY = "artifact_manifest.json"
FINAL_BRANCH_BUNDLE_ENTRY = "final-branch.bundle"
TASK_METADATA_ENTRY = "task_metadata.json"
VALIDATION_SUMMARY_ENTRY = "handoff_evidence/validation-summary.json"
PACKAGE_SMOKE_RECEIPT_ENTRY = "handoff_evidence/package-smoke-receipt.json"
HARD_SCAN_RECEIPT_ENTRY = "handoff_evidence/hard-scan-receipt.json"
CHECK_RECEIPTS_ENTRY = "handoff_evidence/check-receipts.json"
ADVISORY_REVIEW_RECEIPT_ENTRY = "handoff_evidence/advisory-review-receipt.json"

REQUIRED_HANDOFF_ENTRIES = frozenset(
    {
        ARTIFACT_MANIFEST_ENTRY,
        FINAL_BRANCH_BUNDLE_ENTRY,
        TASK_METADATA_ENTRY,
        VALIDATION_SUMMARY_ENTRY,
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        HARD_SCAN_RECEIPT_ENTRY,
        CHECK_RECEIPTS_ENTRY,
    }
)
OPTIONAL_HANDOFF_ENTRIES = frozenset({ADVISORY_REVIEW_RECEIPT_ENTRY})

FORBIDDEN_HANDOFF_ENTRIES = frozenset(
    {
        "git_status.txt",
        "receipt-index.json",
        "handoff_evidence/generation.json",
        "handoff_evidence/bundle-reconstruction.json",
        "handoff_evidence/private-smoke-status.json",
        "handoff_evidence/private-validation-status.json",
        "handoff_evidence/package-smoke.json",
        "handoff_evidence/hard-scan-classification.json",
        "handoff_evidence/test-checks.json",
        "handoff_evidence/source-tree-scan.json",
        "handoff_evidence/verification-matrix.json",
        "handoff_evidence/smoke-compare-status.json",
        "handoff_evidence/subagent-review-findings.json",
        "handoff_evidence/subagent-review-reconciliation.json",
    }
)
RETIRED_HANDOFF_ENTRY_NAMES = FORBIDDEN_HANDOFF_ENTRIES

ALLOWED_SIDECAR_KEYS = frozenset(
    {
        "schema_id",
        "zip_filename",
        "zip_sha256",
        "artifact_manifest_sha256",
        "branch",
        "base",
        "final_head",
        "generated_at_utc",
        "status_summary",
        "advisory_reviews",
    }
)
FORBIDDEN_SIDECAR_KEYS = frozenset(
    {
        "handoff_package_path",
        "handoff_sidecar_path",
    }
)
ALLOWED_STATUS_SUMMARY_KEYS = frozenset(
    {
        "validation",
        "scripted_checks",
        "package_smoke",
        "profile_smoke",
        "private_validation",
        "baseline_compare",
        "governance",
        "code_map",
        "handoff_verify",
        "share_safety_scan",
        "validation_blocked_count",
        "validation_fail_count",
    }
)
FORBIDDEN_STATUS_SUMMARY_KEYS = frozenset(
    {
        "tests",
        "private_smoke_run_level",
        "private_smoke_child_checks",
        "handoff_safety",
        "blocked_count",
        "fail_count",
    }
)
MANDATORY_VALIDATION_ROWS = frozenset(
    {
        "scripted_checks",
        "package_smoke",
        "profile_smoke",
        "private_validation",
        "baseline_compare",
        "governance",
        "code_map",
    }
)
VALIDATION_ROW_KEYS = frozenset({"name", "status", "evidence"})


@dataclass(frozen=True)
class EvidenceRef:
    entry: str
    command_id: str | None = None
    reason_kind: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ValidationRowSource:
    entry: str
    command_id: str | None = None
    command_optional: bool = False


VALIDATION_ROW_SOURCE_CONTRACT = {
    "scripted_checks": ValidationRowSource(CHECK_RECEIPTS_ENTRY),
    "package_smoke": ValidationRowSource(PACKAGE_SMOKE_RECEIPT_ENTRY),
    "profile_smoke": ValidationRowSource(
        CHECK_RECEIPTS_ENTRY,
        command_id="profile-smoke",
    ),
    "private_validation": ValidationRowSource(
        CHECK_RECEIPTS_ENTRY,
        command_id="private-validation",
        command_optional=True,
    ),
    "baseline_compare": ValidationRowSource(
        CHECK_RECEIPTS_ENTRY,
        command_id="baseline-compare",
        command_optional=True,
    ),
    "governance": ValidationRowSource(HARD_SCAN_RECEIPT_ENTRY),
    "code_map": ValidationRowSource(CHECK_RECEIPTS_ENTRY, command_id="code-map"),
}

TASK_METADATA_KEYS = frozenset(
    {
        "schema_id",
        "mode",
        "slug",
        "branch",
        "base",
        "final_head",
        "pr_url",
        "task_title",
        "task_id",
        "generated_at_utc",
        "scope_summary",
        "fixed_blockers",
        "open_issues",
        "not_run",
        "blocked",
        "commit_list",
    }
)
TASK_REJECTED_KEYS = frozenset(
    {
        "validation_evidence",
        "handoff_evidence",
        "handoff_markdown_evidence",
        "handoff_package_path",
        "handoff_sidecar_path",
    }
)

HANDOFF_TOOL_SOURCE_PATHS = [
    ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/run_check.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_contract.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/receipt_contract.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/collect_handoff_receipts.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/run_package_smoke.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py",
    ".agents/skills/quant-research-platform-maintenance/templates/handoff/task_metadata.template.json",
]


def exact_package_member_set(*, include_advisory: bool) -> frozenset[str]:
    entries = set(REQUIRED_HANDOFF_ENTRIES)
    if include_advisory:
        entries.add(ADVISORY_REVIEW_RECEIPT_ENTRY)
    return frozenset(entries)


def validate_handoff_package_layout(names: Iterable[str]) -> list[str]:
    members = list(names)
    member_set = set(members)
    failures: list[str] = []
    forbidden = sorted(member_set & FORBIDDEN_HANDOFF_ENTRIES)
    failures.extend(f"forbidden_handoff_entry: {name}" for name in forbidden)
    missing = sorted(REQUIRED_HANDOFF_ENTRIES - member_set)
    failures.extend(f"missing_required_entry: {name}" for name in missing)
    optional_present = member_set & OPTIONAL_HANDOFF_ENTRIES
    expected = exact_package_member_set(
        include_advisory=ADVISORY_REVIEW_RECEIPT_ENTRY in optional_present
    )
    extra = sorted(member_set - expected)
    failures.extend(f"unexpected_handoff_entry: {name}" for name in extra)
    if len(optional_present) > 1:
        failures.append("handoff_optional_entry_set_invalid")
    return failures


def validate_sidecar_contract(payload: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != "qrp-handoff-sidecar":
        failures.append("sidecar_schema_id_invalid")
    missing = sorted(ALLOWED_SIDECAR_KEYS - set(payload))
    failures.extend(f"sidecar_missing: {key}" for key in missing)
    extra = sorted(set(payload) - ALLOWED_SIDECAR_KEYS)
    failures.extend(f"sidecar_unapproved_field: {key}" for key in extra)
    forbidden = sorted(set(payload) & FORBIDDEN_SIDECAR_KEYS)
    failures.extend(f"sidecar_forbidden_field: {key}" for key in forbidden)
    for key in (
        "zip_filename",
        "zip_sha256",
        "artifact_manifest_sha256",
        "branch",
        "base",
        "final_head",
        "generated_at_utc",
    ):
        if key in payload and (
            not isinstance(payload.get(key), str) or not str(payload.get(key)).strip()
        ):
            failures.append(f"sidecar_string_invalid: {key}")
    summary = payload.get("status_summary")
    if not isinstance(summary, Mapping):
        failures.append("sidecar_status_summary_invalid")
    else:
        failures.extend(validate_status_summary_contract(summary))
    advisory_reviews = payload.get("advisory_reviews")
    if not isinstance(advisory_reviews, Mapping):
        failures.append("sidecar_advisory_reviews_invalid")
    elif any(value not in VALID_STATUS_TOKENS for value in advisory_reviews.values()):
        failures.append("sidecar_advisory_reviews_status_invalid")
    return failures


def validate_status_summary_contract(payload: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    missing = sorted(ALLOWED_STATUS_SUMMARY_KEYS - set(payload))
    failures.extend(f"sidecar_status_summary_missing: {key}" for key in missing)
    extra = sorted(set(payload) - ALLOWED_STATUS_SUMMARY_KEYS)
    failures.extend(f"sidecar_status_summary_forbidden: {key}" for key in extra)
    forbidden = sorted(set(payload) & FORBIDDEN_STATUS_SUMMARY_KEYS)
    failures.extend(f"sidecar_status_summary_forbidden: {key}" for key in forbidden)
    for key in ALLOWED_STATUS_SUMMARY_KEYS - {
        "validation_blocked_count",
        "validation_fail_count",
    }:
        if key in payload and payload.get(key) not in VALID_STATUS_TOKENS:
            failures.append(f"sidecar_status_summary_status_invalid: {key}")
    for key in ("validation_blocked_count", "validation_fail_count"):
        if key in payload and (
            not isinstance(payload.get(key), int) or int(payload.get(key)) < 0
        ):
            failures.append(f"sidecar_status_summary_count_invalid: {key}")
    return failures


def validate_task_metadata_contract(
    payload: Mapping[str, object],
    *,
    mode: str | None = None,
) -> list[str]:
    failures: list[str] = []
    allowed = TASK_METADATA_KEYS | TASK_REJECTED_KEYS
    unknown = sorted(set(payload) - allowed)
    failures.extend(f"task_metadata_unapproved_field: {key}" for key in unknown)
    blocked = sorted(set(payload) & TASK_REJECTED_KEYS)
    failures.extend(f"task_metadata_forbidden_field: {key}" for key in blocked)
    if payload.get("schema_id") != "qrp-handoff-task-metadata":
        failures.append("task_metadata_schema_id_invalid")
    effective_mode = str(mode if mode is not None else payload.get("mode") or "")
    if effective_mode not in {"evidence", "pr_closure"}:
        failures.append(f"task_metadata_mode_invalid: {effective_mode}")
    if effective_mode == "evidence" and "pr_url" in payload:
        failures.append("task_metadata_evidence_forbidden_field: pr_url")
    for key in ("slug", "branch", "base", "final_head"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            failures.append(f"task_metadata_required_string_missing: {key}")
    return failures


def validate_validation_summary_contract(
    payload: Mapping[str, object],
    *,
    require_mandatory: bool = True,
) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != "qrp-handoff-validation-summary":
        failures.append("validation_summary_schema_id_invalid")
    rows = payload.get("checks")
    if not isinstance(rows, list):
        failures.append("validation_summary_checks_must_be_list")
        return failures
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"validation_summary_check_not_object: {index}")
            continue
        extra = sorted(set(row) - VALIDATION_ROW_KEYS)
        failures.extend(
            f"validation_summary_row_unapproved_field: {index}:{key}"
            for key in extra
        )
        for key in VALIDATION_ROW_KEYS:
            if key not in row:
                failures.append(f"validation_summary_row_missing: {index}:{key}")
        name = row.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name not in MANDATORY_VALIDATION_ROWS
        ):
            failures.append(f"validation_summary_name_invalid: {index}")
        else:
            if name in seen:
                failures.append(f"validation_summary_duplicate_check: {name}")
            seen.add(name)
        status = row.get("status")
        if status not in VALID_STATUS_TOKENS:
            failures.append(f"validation_summary_status_invalid: {index}:{status}")
        evidence = row.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            failures.append(f"validation_summary_evidence_invalid: {index}")
    if require_mandatory:
        missing = sorted(MANDATORY_VALIDATION_ROWS - seen)
        failures.extend(
            f"validation_summary_mandatory_check_missing: {name}" for name in missing
        )
    return failures


def parse_validation_evidence_ref(evidence: str) -> EvidenceRef:
    text = evidence.strip()
    for prefix, reason_kind in (("not_run: ", "not_run"), ("blocked: ", "blocked")):
        if text.startswith(prefix):
            reason = text.removeprefix(prefix).strip()
            return EvidenceRef("", reason_kind=reason_kind, reason=reason)
    if text in {
        CHECK_RECEIPTS_ENTRY,
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        HARD_SCAN_RECEIPT_ENTRY,
    }:
        return EvidenceRef(text)
    prefix = f"{CHECK_RECEIPTS_ENTRY}#"
    if text.startswith(prefix):
        command_id = text.removeprefix(prefix)
        if command_id and command_id == command_id.strip():
            return EvidenceRef(CHECK_RECEIPTS_ENTRY, command_id=command_id)
    return EvidenceRef("", reason_kind="invalid", reason=text)


def expected_validation_status_map_from_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    check_rows = _check_rows_by_command_id(receipts.get(CHECK_RECEIPTS_ENTRY, {}))
    for row_name, source in VALIDATION_ROW_SOURCE_CONTRACT.items():
        if source.command_id is not None:
            command_row = check_rows.get(source.command_id)
            if command_row is None:
                if source.command_optional:
                    result[row_name] = "NOT_RUN"
                continue
            result[row_name] = _status_or_not_run(command_row.get("status"))
            continue
        receipt = receipts.get(source.entry)
        if receipt is None:
            continue
        result[row_name] = _status_or_not_run(receipt.get("status"))
    return result


def validate_validation_summary_against_receipts(
    validation_summary: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
) -> list[str]:
    failures = validate_validation_summary_contract(validation_summary)
    rows = validation_summary.get("checks")
    if not isinstance(rows, list):
        return failures
    expected_statuses = expected_validation_status_map_from_receipts(receipts)
    check_rows = _check_rows_by_command_id(receipts.get(CHECK_RECEIPTS_ENTRY, {}))
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        status = row.get("status")
        evidence = row.get("evidence")
        if (
            not isinstance(name, str)
            or name not in VALIDATION_ROW_SOURCE_CONTRACT
            or status not in VALID_STATUS_TOKENS
            or not isinstance(evidence, str)
        ):
            continue
        source = VALIDATION_ROW_SOURCE_CONTRACT[name]
        ref = parse_validation_evidence_ref(evidence)
        failures.extend(
            _validate_row_evidence_ref(
                name=name,
                status=str(status),
                source=source,
                ref=ref,
                check_rows=check_rows,
            )
        )
        expected_status = expected_statuses.get(name)
        if expected_status is not None and status != expected_status:
            failures.append(
                f"validation_summary_status_mismatch: {name}:{status}!={expected_status}"
            )
    return failures


def validate_advisory_reviews_against_receipts(
    advisory_reviews: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
) -> list[str]:
    advisory = receipts.get(ADVISORY_REVIEW_RECEIPT_ENTRY)
    expected: dict[str, str] = {}
    if advisory is not None:
        expected = {"review_subagent": _status_or_not_run(advisory.get("status"))}
    if dict(advisory_reviews) != expected:
        return ["advisory_reviews_mismatch"]
    return []


def validate_task_metadata_against_sidecar(
    task_metadata: Mapping[str, object],
    sidecar: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    for key in ("branch", "base", "final_head"):
        if task_metadata.get(key) != sidecar.get(key):
            failures.append(f"task_metadata_sidecar_mismatch: {key}")
    return failures


def validation_row(name: str, status: str, evidence: str) -> dict[str, str]:
    normalized = status if status in VALID_STATUS_TOKENS else "FAIL"
    return {"name": name, "status": normalized, "evidence": evidence}


def _validate_row_evidence_ref(
    *,
    name: str,
    status: str,
    source: ValidationRowSource,
    ref: EvidenceRef,
    check_rows: Mapping[str, Mapping[str, object]],
) -> list[str]:
    failures: list[str] = []
    if ref.reason_kind == "invalid":
        return [f"validation_summary_evidence_syntax_invalid: {name}"]
    if status in {"PASS", "FAIL"} and ref.reason_kind is not None:
        failures.append(f"validation_summary_evidence_status_invalid: {name}")
    if status == "NOT_RUN" and ref.reason_kind not in {None, "not_run"}:
        failures.append(f"validation_summary_evidence_status_invalid: {name}")
    if status == "BLOCKED" and ref.reason_kind not in {None, "blocked"}:
        failures.append(f"validation_summary_evidence_status_invalid: {name}")
    if ref.reason_kind is not None:
        if not ref.reason:
            failures.append(f"validation_summary_evidence_reason_missing: {name}")
        if status == "NOT_RUN" and ref.reason_kind != "not_run":
            failures.append(f"validation_summary_evidence_syntax_invalid: {name}")
        if status == "BLOCKED" and ref.reason_kind != "blocked":
            failures.append(f"validation_summary_evidence_syntax_invalid: {name}")
        if status in {"PASS", "FAIL"}:
            failures.append(f"validation_summary_evidence_syntax_invalid: {name}")
        if source.command_id is not None and not source.command_optional:
            failures.append(f"validation_summary_evidence_source_mismatch: {name}")
        if source.command_optional and source.command_id in check_rows:
            failures.append(f"validation_summary_evidence_source_mismatch: {name}")
        if source.command_optional and source.command_id not in check_rows:
            if status != "NOT_RUN":
                failures.append(
                    f"validation_summary_status_mismatch: {name}:{status}!=NOT_RUN"
                )
            return failures
        return failures
    if ref.entry != source.entry:
        failures.append(f"validation_summary_evidence_source_mismatch: {name}")
        return failures
    if source.command_id is None:
        if ref.command_id is not None:
            failures.append(f"validation_summary_evidence_source_mismatch: {name}")
        return failures
    if ref.command_id != source.command_id:
        if ref.command_id is not None and ref.command_id not in check_rows:
            failures.append(
                f"validation_summary_receipt_edge_missing: {name}:{ref.command_id}"
            )
        else:
            failures.append(f"validation_summary_evidence_source_mismatch: {name}")
        return failures
    command_row = check_rows.get(source.command_id)
    if command_row is None:
        failures.append(
            f"validation_summary_receipt_edge_missing: {name}:{source.command_id}"
        )
    return failures


def _check_rows_by_command_id(
    check_receipts: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    rows = check_receipts.get("checks")
    if not isinstance(rows, list):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("command_id"), str):
            result[str(row["command_id"])] = row
    return result


def _status_or_not_run(value: object) -> str:
    return str(value) if value in VALID_STATUS_TOKENS else "NOT_RUN"


def validation_summary_status_map(payload: Mapping[str, object]) -> dict[str, str]:
    failures = validate_validation_summary_contract(payload)
    if failures:
        raise SystemExit(";".join(failures))
    rows = payload["checks"]
    assert isinstance(rows, list)
    return {str(row["name"]): str(row["status"]) for row in rows if isinstance(row, Mapping)}


def aggregate_validation_summary_status(payload: Mapping[str, object]) -> str:
    statuses = validation_summary_status_map(payload)
    return _aggregate_validation_matrix_status(statuses.values())


def build_sidecar_status_summary(
    status_map: Mapping[str, str],
    *,
    handoff_verify: str = "NOT_RUN",
    share_safety_scan: str = "NOT_RUN",
) -> dict[str, object]:
    mandatory = {name: status_map.get(name, "NOT_RUN") for name in MANDATORY_VALIDATION_ROWS}
    return {
        "validation": _aggregate_validation_matrix_status(mandatory.values()),
        "scripted_checks": mandatory["scripted_checks"],
        "package_smoke": mandatory["package_smoke"],
        "profile_smoke": mandatory["profile_smoke"],
        "private_validation": mandatory["private_validation"],
        "baseline_compare": mandatory["baseline_compare"],
        "governance": mandatory["governance"],
        "code_map": mandatory["code_map"],
        "handoff_verify": handoff_verify,
        "share_safety_scan": share_safety_scan,
        "validation_blocked_count": sum(
            1 for status in mandatory.values() if status == "BLOCKED"
        ),
        "validation_fail_count": sum(1 for status in mandatory.values() if status == "FAIL"),
    }


def posix_member_name_failures(name: str) -> list[str]:
    failures: list[str] = []
    if "\\" in name:
        failures.append(f"zip_member_not_posix: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not name or name.endswith("/"):
        failures.append(f"zip_member_path_invalid: {name}")
    return failures
