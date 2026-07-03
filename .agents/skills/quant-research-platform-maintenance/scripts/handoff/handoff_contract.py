"""Handoff package-level contract for QRP tooling.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

try:
    from quant_research_platform.policy.check_status import (
        VALID_STATUS_TOKENS,
    )
    from quant_research_platform.policy.check_status import (
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


class ProofClass(StrEnum):
    REPLAYED = "REPLAYED"
    DERIVED_FROM_REPLAY = "DERIVED_FROM_REPLAY"
    IDENTITY_BOUND = "IDENTITY_BOUND"
    METADATA_VALIDATED = "METADATA_VALIDATED"


@dataclass(frozen=True)
class FieldSpec:
    proof_class: ProofClass


@dataclass(frozen=True)
class ValidationRowSource:
    receipt_entry: str | None = None
    command_id: str | None = None


@dataclass(frozen=True)
class ReplayCommandSpec:
    command_id: str
    scope: str
    command_argv: tuple[str, ...]
    timeout_seconds: int

    @property
    def command(self) -> str:
        return shlex.join(self.command_argv)


ACTIVE_DENIED_HANDOFF_EVIDENCE_FIELDS = (
    {
        "field_name": "stdout",
        "owner": "handoff receipt contract",
        "risk": "raw command output can expose local paths, private data, or secrets",
        "lifecycle": "deny while REVIEW_SAFE handoff remains compact receipt based",
        "acceptance_gate": "receipt payload validation rejects raw output fields",
    },
    {
        "field_name": "stderr",
        "owner": "handoff receipt contract",
        "risk": "raw error output can expose tracebacks, local paths, or private data",
        "lifecycle": "deny while REVIEW_SAFE handoff remains compact receipt based",
        "acceptance_gate": "receipt payload validation rejects raw output fields",
    },
    {
        "field_name": "raw_output",
        "owner": "handoff receipt contract",
        "risk": "unbounded output bypasses replayable summary contracts",
        "lifecycle": "deny while machine evidence is derived from receipts",
        "acceptance_gate": "receipt payload validation rejects raw output fields",
    },
    {
        "field_name": "raw_log",
        "owner": "handoff receipt contract",
        "risk": "raw logs can contain local paths, row samples, and tracebacks",
        "lifecycle": "deny while logs remain LOCAL_ONLY artifacts",
        "acceptance_gate": "receipt payload validation rejects raw log fields",
    },
    {
        "field_name": "local_absolute_path",
        "owner": "share-safety scanner",
        "risk": "local machine paths are not REVIEW_SAFE",
        "lifecycle": "deny in handoff docs, receipts, sidecars, and bundle text",
        "acceptance_gate": "share-safety and source-tree scans reject local paths",
    },
    {
        "field_name": "row_level_private_data",
        "owner": "share-safety scanner",
        "risk": "row-level samples can disclose private research data",
        "lifecycle": "deny in REVIEW_SAFE evidence",
        "acceptance_gate": "share-safety and source-tree scans reject row samples",
    },
    {
        "field_name": "provider_marker",
        "owner": "share-safety scanner",
        "risk": "provider markers can disclose vendor/source-private context",
        "lifecycle": "deny in REVIEW_SAFE evidence",
        "acceptance_gate": "share-safety and source-tree scans reject provider markers",
    },
    {
        "field_name": "stock_code_like_sample",
        "owner": "share-safety scanner",
        "risk": "stock-code-like samples can disclose row-level market data examples",
        "lifecycle": "deny in REVIEW_SAFE evidence",
        "acceptance_gate": "share-safety and source-tree scans reject stock samples",
    },
    {
        "field_name": "validation_status",
        "owner": "handoff contract",
        "risk": "hand-written validation status can diverge from replayed receipts",
        "lifecycle": "deny outside validation summary rows derived from receipts",
        "acceptance_gate": "validation summary replay comparison rejects status drift",
    },
)


ARTIFACT_MANIFEST_ENTRY = "artifact_manifest.json"
FINAL_BRANCH_BUNDLE_ENTRY = "final-branch.bundle"
TASK_METADATA_ENTRY = "task_metadata.json"
VALIDATION_SUMMARY_ENTRY = "handoff_evidence/validation-summary.json"
PACKAGE_SMOKE_RECEIPT_ENTRY = "handoff_evidence/package-smoke-receipt.json"
HARD_SCAN_RECEIPT_ENTRY = "handoff_evidence/hard-scan-receipt.json"
CHECK_RECEIPTS_ENTRY = "handoff_evidence/check-receipts.json"

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

SIDECAR_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "zip_filename": FieldSpec(ProofClass.IDENTITY_BOUND),
    "zip_sha256": FieldSpec(ProofClass.IDENTITY_BOUND),
}

TASK_METADATA_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "task_name": FieldSpec(ProofClass.METADATA_VALIDATED),
    "branch": FieldSpec(ProofClass.IDENTITY_BOUND),
    "base": FieldSpec(ProofClass.IDENTITY_BOUND),
    "final_head": FieldSpec(ProofClass.IDENTITY_BOUND),
}

ARTIFACT_MANIFEST_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "entries": FieldSpec(ProofClass.IDENTITY_BOUND),
}

ARTIFACT_MANIFEST_ENTRY_FIELD_SPECS = {
    "path": FieldSpec(ProofClass.IDENTITY_BOUND),
    "size": FieldSpec(ProofClass.IDENTITY_BOUND),
    "sha256": FieldSpec(ProofClass.IDENTITY_BOUND),
}

VALIDATION_SUMMARY_FIELD_SPECS = {
    "schema_id": FieldSpec(ProofClass.METADATA_VALIDATED),
    "status": FieldSpec(ProofClass.DERIVED_FROM_REPLAY),
    "rows": FieldSpec(ProofClass.DERIVED_FROM_REPLAY),
}

VALIDATION_ROW_FIELD_SPECS = {
    "name": FieldSpec(ProofClass.METADATA_VALIDATED),
    "status": FieldSpec(ProofClass.DERIVED_FROM_REPLAY),
}

MANDATORY_VALIDATION_ROWS = (
    "scripted_checks",
    "package_smoke",
    "profile_smoke",
    "governance",
    "code_map",
)

VALIDATION_ROW_SOURCE_CONTRACT = {
    "scripted_checks": ValidationRowSource(
        receipt_entry=CHECK_RECEIPTS_ENTRY,
    ),
    "package_smoke": ValidationRowSource(
        receipt_entry=PACKAGE_SMOKE_RECEIPT_ENTRY,
    ),
    "profile_smoke": ValidationRowSource(
        receipt_entry=CHECK_RECEIPTS_ENTRY,
        command_id="profile-smoke",
    ),
    "governance": ValidationRowSource(
        receipt_entry=HARD_SCAN_RECEIPT_ENTRY,
    ),
    "code_map": ValidationRowSource(
        receipt_entry=CHECK_RECEIPTS_ENTRY,
        command_id="code-map",
    ),
}

REPLAY_COMMAND_CONTRACT = {
    "profile-smoke": ReplayCommandSpec(
        command_id="profile-smoke",
        scope="profile-smoke",
        timeout_seconds=900,
        command_argv=(
            "python",
            "tests/tools/run_profile.py",
            "--profile",
            "profile-smoke",
            "--run-dir",
            ".run/handoff-replay/profile-smoke",
        ),
    ),
    "code-map": ReplayCommandSpec(
        command_id="code-map",
        scope="code-map",
        timeout_seconds=300,
        command_argv=(
            "python",
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "verify_code_map_flow.py",
            "--repo-root",
            ".",
        ),
    ),
}

PACKAGE_BUILD_TIMEOUT_SECONDS = 300
PACKAGE_IMPORT_TIMEOUT_SECONDS = 120


def allowed_keys(specs: Mapping[str, FieldSpec]) -> frozenset[str]:
    return frozenset(specs)


def aggregate_handoff_status(statuses: Iterable[str]) -> str:
    return _aggregate_validation_matrix_status(list(statuses))


def validate_handoff_package_layout(names: Iterable[str]) -> list[str]:
    members = list(names)
    member_set = set(members)
    failures: list[str] = []
    missing = sorted(REQUIRED_HANDOFF_ENTRIES - member_set)
    failures.extend(f"missing_required_entry: {name}" for name in missing)
    extra = sorted(member_set - REQUIRED_HANDOFF_ENTRIES)
    failures.extend(f"unexpected_handoff_entry: {name}" for name in extra)
    return failures


def validate_sidecar_contract(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, SIDECAR_FIELD_SPECS, "sidecar")
    if payload.get("schema_id") != "qrp-handoff-sidecar":
        failures.append("sidecar_schema_id_invalid")
    for key in ("zip_filename", "zip_sha256"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            failures.append(f"sidecar_string_invalid: {key}")
    if "zip_sha256" in payload and not sha(payload.get("zip_sha256")):
        failures.append("sidecar_zip_sha256_invalid")
    return failures


def validate_task_metadata_contract(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, TASK_METADATA_FIELD_SPECS, "task_metadata")
    if payload.get("schema_id") != "qrp-handoff-task-metadata":
        failures.append("task_metadata_schema_id_invalid")
    for key in ("task_name", "branch", "base", "final_head"):
        if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
            failures.append(f"task_metadata_required_string_missing: {key}")
    return failures


def validate_artifact_manifest_contract(payload: Mapping[str, object]) -> list[str]:
    failures = exact_keys(payload, ARTIFACT_MANIFEST_FIELD_SPECS, "artifact_manifest")
    if payload.get("schema_id") != "qrp-handoff-artifact-manifest":
        failures.append("artifact_manifest_schema_id_invalid")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        failures.append("artifact_manifest_entries_invalid")
        return failures
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            failures.append(f"artifact_manifest_entry_invalid: {index}")
            continue
        failures.extend(
            exact_keys(
                entry,
                ARTIFACT_MANIFEST_ENTRY_FIELD_SPECS,
                f"artifact_manifest_entry:{index}",
            )
        )
        if not isinstance(entry.get("path"), str) or not str(entry.get("path")).strip():
            failures.append(f"artifact_manifest_path_invalid: {index}")
        if not isinstance(entry.get("size"), int) or int(entry.get("size")) < 0:
            failures.append(f"artifact_manifest_size_invalid: {index}")
        if not sha(entry.get("sha256")):
            failures.append(f"artifact_manifest_sha256_invalid: {index}")
    return failures


def validate_validation_summary_contract(
    payload: Mapping[str, object],
    *,
    require_mandatory: bool = True,
) -> list[str]:
    failures = exact_keys(payload, VALIDATION_SUMMARY_FIELD_SPECS, "validation_summary")
    if payload.get("schema_id") != "qrp-handoff-validation-summary":
        failures.append("validation_summary_schema_id_invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        failures.append("validation_summary_rows_must_be_list")
        return failures
    seen: set[str] = set()
    statuses: list[str] = []
    allowed_rows = set(MANDATORY_VALIDATION_ROWS)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"validation_summary_row_not_object: {index}")
            continue
        failures.extend(
            exact_keys(
                row, VALIDATION_ROW_FIELD_SPECS, f"validation_summary_row:{index}"
            )
        )
        name = row.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name not in allowed_rows
        ):
            failures.append(f"validation_summary_name_invalid: {index}")
        else:
            if name in seen:
                failures.append(f"validation_summary_duplicate_row: {name}")
            seen.add(name)
        status = row.get("status")
        if status not in VALID_STATUS_TOKENS:
            failures.append(f"validation_summary_status_invalid: {index}:{status}")
        else:
            statuses.append(str(status))
    if require_mandatory:
        missing = [name for name in MANDATORY_VALIDATION_ROWS if name not in seen]
        failures.extend(
            f"validation_summary_mandatory_row_missing: {name}" for name in missing
        )
    summary_status = payload.get("status")
    if summary_status not in VALID_STATUS_TOKENS:
        failures.append(f"validation_summary_status_invalid: {summary_status}")
    elif statuses:
        expected = aggregate_handoff_status(statuses)
        if summary_status != expected:
            failures.append(
                f"validation_summary_status_mismatch: {summary_status}!={expected}"
            )
    return failures


def expected_validation_status_map_from_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    check_rows = check_rows_by_command_id(receipts.get(CHECK_RECEIPTS_ENTRY, {}))
    for row_name, source in VALIDATION_ROW_SOURCE_CONTRACT.items():
        if source.command_id is not None:
            command_row = check_rows.get(source.command_id)
            result[row_name] = _status_or_not_run(
                command_row.get("status") if command_row else None
            )
        else:
            assert source.receipt_entry is not None
            receipt = receipts.get(source.receipt_entry)
            result[row_name] = _status_or_not_run(
                receipt.get("status") if receipt else None
            )
    return result


def build_validation_summary_from_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    status_map = expected_validation_status_map_from_receipts(receipts)
    rows = [
        {"name": name, "status": status_map.get(name, "NOT_RUN")}
        for name in MANDATORY_VALIDATION_ROWS
    ]
    payload: dict[str, object] = {
        "schema_id": "qrp-handoff-validation-summary",
        "status": aggregate_handoff_status(str(row["status"]) for row in rows),
        "rows": rows,
    }
    failures = validate_validation_summary_contract(payload)
    if failures:
        raise SystemExit(";".join(failures))
    return payload


def mandatory_validation_failures(
    validation_summary: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    rows = validation_summary.get("rows")
    if not isinstance(rows, list):
        return ["validation_summary_rows_must_be_list"]
    by_name: dict[str, list[object]] = {name: [] for name in MANDATORY_VALIDATION_ROWS}
    observed_statuses: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"validation_summary_row_not_object: {index}")
            continue
        name = row.get("name")
        status = row.get("status")
        if not isinstance(name, str) or name not in MANDATORY_VALIDATION_ROWS:
            failures.append(f"validation_summary_row_unknown: {index}:{name}")
            continue
        by_name.setdefault(name, []).append(status)
        if status not in VALID_STATUS_TOKENS:
            failures.append(f"validation_summary_row_unknown_status:{name}:{status}")
            continue
        observed_statuses.append(str(status))
        if status != "PASS":
            failures.append(f"validation_summary_row_not_pass:{name}:{status}")
    for name in MANDATORY_VALIDATION_ROWS:
        statuses = by_name.get(name, [])
        if not statuses:
            failures.append(f"validation_summary_row_missing:{name}")
        if len(statuses) > 1:
            failures.append(f"validation_summary_row_duplicate:{name}")
    summary_status = validation_summary.get("status")
    if summary_status not in VALID_STATUS_TOKENS:
        failures.append(f"validation_summary_status_unknown:{summary_status}")
    elif observed_statuses:
        expected = aggregate_handoff_status(observed_statuses)
        if summary_status != expected:
            failures.append(
                f"validation_summary_status_mismatch:{summary_status}!={expected}"
            )
    return failures


def validate_validation_summary_against_receipts(
    validation_summary: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
) -> list[str]:
    failures = validate_validation_summary_contract(validation_summary)
    rows = validation_summary.get("rows")
    if not isinstance(rows, list):
        return failures
    expected_statuses = expected_validation_status_map_from_receipts(receipts)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        status = row.get("status")
        if isinstance(name, str) and name in expected_statuses:
            expected = expected_statuses[name]
            if status != expected:
                failures.append(
                    f"validation_summary_status_mismatch: {name}:{status}!={expected}"
                )
    return failures


def check_rows_by_command_id(
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


def replay_command_spec(command_id: str) -> ReplayCommandSpec:
    try:
        return REPLAY_COMMAND_CONTRACT[command_id]
    except KeyError as exc:
        raise KeyError(f"unknown_replay_command: {command_id}") from exc


def exact_keys(
    payload: Mapping[str, object],
    specs: Mapping[str, FieldSpec],
    prefix: str,
) -> list[str]:
    allowed = allowed_keys(specs)
    if ":" in prefix:
        label, _, index = prefix.partition(":")
        failures = [
            f"{label}_missing: {index}:{key}" for key in sorted(allowed - set(payload))
        ]
        failures.extend(
            f"{label}_unapproved_field: {index}:{key}"
            for key in sorted(set(payload) - allowed)
        )
        return failures
    failures = [f"{prefix}_missing: {key}" for key in sorted(allowed - set(payload))]
    failures.extend(
        f"{prefix}_unapproved_field: {key}" for key in sorted(set(payload) - allowed)
    )
    return failures


def posix_member_name_failures(name: str) -> list[str]:
    failures: list[str] = []
    if "\\" in name:
        failures.append(f"zip_member_not_posix: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not name or name.endswith("/"):
        failures.append(f"zip_member_path_invalid: {name}")
    return failures


def sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _status_or_not_run(value: object) -> str:
    return str(value) if value in VALID_STATUS_TOKENS else "NOT_RUN"
