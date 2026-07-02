from __future__ import annotations

import shlex
from pathlib import Path

try:
    from tests.agents.handoff.conftest import _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _load_handoff_module  # type: ignore[no-redef]


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_id": "qrp-check-receipt",
        "command_id": "unit",
        "scope": "unit scope",
        "command_argv": ["python", "-m", "pytest"],
        "command": "python -m pytest",
        "status": "PASS",
        "exit_code": 0,
        "duration_seconds": 1.0,
        "counts": {
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "collected": 1,
        },
        "stdout_sha256": "0" * 64,
        "stderr_sha256": "1" * 64,
        "output_summary": "1 passed",
        "failure_excerpt": "",
    }
    row.update(overrides)
    return row


def _package_smoke_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-package-smoke-receipt",
        "status": "PASS",
        "wheel": {"filename": "demo.whl", "sha256": "0" * 64},
        "sdist": {"filename": "demo.tar.gz", "sha256": "1" * 64},
        "install": {
            "target_kind": "temporary isolated target",
            "no_deps": True,
            "status": "PASS",
        },
        "import": {"status": "PASS"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
            "scanned_artifact_kinds": ["wheel", "sdist", "install_target"],
            "examples": [],
        },
    }
    payload.update(overrides)
    return payload


def _hard_scan_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-hard-scan-receipt",
        "status": "PASS",
        "scanned_head": "1" * 40,
        "scanner_source_sha256": "2" * 64,
        "scan_runs": [
            {
                "scan_id": "allowance_contract",
                "rule_set_id": "source-tree:allowance_contract",
                "match_count": 0,
                "actual_violation_count": 0,
                "deferred_count": 0,
                "negative_fixture_count": 0,
                "false_positive_count": 0,
                "result_sha256": "3" * 64,
                "replayable_from_bundle": True,
            }
        ],
        "classified_exceptions": [],
    }
    payload.update(overrides)
    return payload


def _advisory_finding(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "finding_id": "finding-1",
        "classification": "true_issue",
        "resolved": True,
        "summary": "review-safe aggregate finding",
    }
    payload.update(overrides)
    return payload


def _advisory_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-advisory-review-receipt",
        "status": "PASS",
        "review_lanes": [],
        "finding_count": 0,
        "true_issue_count": 0,
        "false_positive_count": 0,
        "re_review_status": "PASS",
        "findings": [],
    }
    payload.update(overrides)
    return payload


def _failures(row: dict[str, object]) -> list[str]:
    contract = _load_handoff_module("receipt_contract")
    return contract.validate_check_receipt_row(row)


def test_check_receipt_rejects_missing_schema_id() -> None:
    row = _valid_row()
    row.pop("schema_id")

    assert "check_receipt_missing: 0:schema_id" in _failures(row)


def test_check_receipt_rejects_extra_field() -> None:
    assert "check_receipt_unapproved_field: 0:extra_field" in _failures(
        _valid_row(extra_field=True)
    )


def test_check_receipt_rejects_pass_with_nonzero_exit_code() -> None:
    assert "check_receipt_pass_exit_code_invalid: 0" in _failures(
        _valid_row(exit_code=1)
    )


def test_check_receipt_rejects_fail_with_zero_exit_code() -> None:
    assert "check_receipt_fail_exit_code_invalid: 0" in _failures(
        _valid_row(status="FAIL", exit_code=0, failure_excerpt="failed")
    )


def test_check_receipt_rejects_pass_with_failed_count() -> None:
    row = _valid_row()
    counts = dict(row["counts"])
    counts["failed"] = 1
    row["counts"] = counts

    assert "check_receipt_pass_counts_invalid: 0" in _failures(row)


def test_check_receipt_rejects_pass_with_error_count() -> None:
    row = _valid_row()
    counts = dict(row["counts"])
    counts["errors"] = 1
    row["counts"] = counts

    assert "check_receipt_pass_counts_invalid: 0" in _failures(row)


def test_check_receipt_rejects_zero_collected_with_outcome_counts() -> None:
    row = _valid_row()
    counts = dict(row["counts"])
    counts["collected"] = 0
    row["counts"] = counts

    assert "check_receipt_counts_collected_mismatch: 0" in _failures(row)


def test_check_receipt_rejects_not_run_without_reason() -> None:
    assert "check_receipt_not_run_reason_missing: 0" in _failures(
        _valid_row(status="NOT_RUN", exit_code=None, counts=None)
    )


def test_check_receipt_rejects_not_run_success_traces() -> None:
    counts = {
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "collected": 1,
    }

    failures = _failures(
        _valid_row(
            status="NOT_RUN",
            exit_code=0,
            counts=counts,
            not_run_reason="not run in this profile",
        )
    )

    assert "check_receipt_not_run_exit_code_invalid: 0" in failures
    assert "check_receipt_not_run_counts_invalid: 0" in failures


def test_check_receipt_rejects_blocked_without_reason() -> None:
    assert "check_receipt_blocked_reason_missing: 0" in _failures(
        _valid_row(status="BLOCKED", exit_code=None, counts=None)
    )


def test_check_receipt_rejects_blocked_success_traces() -> None:
    counts = {
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "collected": 1,
    }

    failures = _failures(
        _valid_row(
            status="BLOCKED",
            exit_code=0,
            counts=counts,
            blocked_reason="blocked by environment",
        )
    )

    assert "check_receipt_blocked_exit_code_invalid: 0" in failures
    assert "check_receipt_blocked_counts_invalid: 0" in failures


def test_check_receipt_rejects_raw_output_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [
            _valid_row(stdout="raw", stderr="raw", raw_output="raw"),
            _valid_row(command_id="grep", raw_grep_output="raw", grep_hits=[]),
        ],
    }

    failures = contract.validate_receipt_payload("check-receipts.json", payload)

    assert "receipt_forbidden_raw_field: check-receipts.json:checks.0.stdout" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.0.stderr" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.0.raw_output" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.1.raw_grep_output" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.1.grep_hits" in failures


def test_check_receipts_rejects_empty_pass_checks() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = {"schema_id": "qrp-check-receipts", "status": "PASS", "checks": []}

    failures = contract.validate_check_receipts(payload)

    assert "check_receipts_empty_pass_invalid" in failures


def test_check_receipts_rejects_top_level_status_mismatch() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [_valid_row(status="FAIL", exit_code=1, failure_excerpt="failed")],
    }

    failures = contract.validate_check_receipts(payload)

    assert "check_receipts_status_mismatch" in failures


def test_package_smoke_rejects_child_fail_with_top_pass() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _package_smoke_payload(install={"target_kind": "temporary isolated target", "no_deps": True, "status": "FAIL"})

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_status_mismatch" in failures


def test_package_smoke_rejects_forbidden_member_count_with_pass_scan() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _package_smoke_payload(
        content_scan={
            "status": "PASS",
            "forbidden_member_count": 1,
            "scanned_artifact_kinds": ["wheel", "sdist", "install_target"],
            "examples": [],
        }
    )

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_content_scan_forbidden_members_mismatch" in failures


def test_package_smoke_rejects_extra_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _package_smoke_payload(extra_field=True)
    payload["content_scan"] = {
        **payload["content_scan"],  # type: ignore[arg-type]
        "extra_field": True,
    }

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_unapproved_field: extra_field" in failures
    assert "package_smoke_content_scan_unapproved_field: extra_field" in failures


def test_hard_scan_rejects_run_and_exception_extra_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _hard_scan_payload(
        scan_runs=[
            {
                **_hard_scan_payload()["scan_runs"][0],  # type: ignore[index]
                "extra_field": True,
            }
        ],
        classified_exceptions=[
            {
                "scan_id": "allowance_contract",
                "classification": "false_positive",
                "owner": "governance_contract.py",
                "rationale": "review-safe exception",
                "reference": "tests",
                "extra_field": True,
            }
        ],
    )

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_run_unapproved_field: 0:extra_field" in failures
    assert "hard_scan_classified_exception_unapproved_field: 0:extra_field" in failures


def test_hard_scan_rejects_actual_violation_exception() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _hard_scan_payload(
        classified_exceptions=[
            {
                "scan_id": "allowance_contract",
                "classification": "actual_violation",
                "owner": "governance_contract.py",
                "rationale": "actual violations cannot be classified away",
                "reference": "tests",
            }
        ],
    )

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_classified_exception_classification_invalid: 0" in failures
    assert "hard_scan_classified_exception_actual_violation: 0" in failures


def test_hard_scan_rejects_classified_exception_missing_required_context() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _hard_scan_payload(
        classified_exceptions=[
            {
                "scan_id": "allowance_contract",
                "classification": "false_positive",
                "owner": "",
                "rationale": "",
                "reference": "",
            }
        ],
    )

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_classified_exception_string_invalid: 0:owner" in failures
    assert "hard_scan_classified_exception_string_invalid: 0:rationale" in failures
    assert "hard_scan_classified_exception_string_invalid: 0:reference" in failures


def test_advisory_receipt_rejects_extra_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _advisory_payload(
        finding_count=1,
        true_issue_count=1,
        findings=[_advisory_finding(extra_field=True)],
        extra_field=True,
    )

    failures = contract.validate_advisory_review_receipt(payload)

    assert "advisory_review_unapproved_field: extra_field" in failures
    assert "advisory_finding_unapproved_field: 0:extra_field" in failures


def test_advisory_receipt_rejects_pass_with_failed_re_review() -> None:
    contract = _load_handoff_module("receipt_contract")

    failures = contract.validate_advisory_review_receipt(
        _advisory_payload(re_review_status="FAIL")
    )

    assert "advisory_review_pass_re_review_status_mismatch" in failures


def test_advisory_receipt_rejects_count_mismatches() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _advisory_payload(
        finding_count=2,
        true_issue_count=0,
        false_positive_count=1,
        findings=[_advisory_finding()],
    )

    failures = contract.validate_advisory_review_receipt(payload)

    assert "advisory_review_finding_count_mismatch" in failures
    assert "advisory_review_true_issue_count_mismatch" in failures
    assert "advisory_review_false_positive_count_mismatch" in failures


def test_advisory_receipt_rejects_unresolved_true_issue_with_pass() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _advisory_payload(
        finding_count=1,
        true_issue_count=1,
        findings=[_advisory_finding(resolved=False)],
    )

    failures = contract.validate_advisory_review_receipt(payload)

    assert "advisory_review_pass_with_unresolved_true_issue" in failures


def test_check_receipt_rejects_command_display_mismatch() -> None:
    failures = _failures(_valid_row(command="python -m wrong"))

    assert "check_receipt_command_mismatch: 0" in failures


def test_check_receipts_rejects_duplicate_command_id() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [_valid_row(), _valid_row()],
    }

    failures = contract.validate_check_receipts(payload)

    assert "check_receipt_duplicate_command_id: unit" in failures


def test_check_receipt_rejects_missing_or_empty_command_argv() -> None:
    assert "check_receipt_command_argv_invalid: 0" in _failures(
        _valid_row(command_argv=[])
    )
    row = _valid_row()
    row.pop("command_argv")

    failures = _failures(row)

    assert "check_receipt_missing: 0:command_argv" in failures
    assert "check_receipt_command_argv_invalid: 0" in failures


def test_check_receipt_accepts_shlex_joined_command() -> None:
    row = _valid_row(command_argv=["python", "-c", "print('hello world')"])
    row["command"] = shlex.join(row["command_argv"])

    assert "check_receipt_command_mismatch: 0" not in _failures(row)


def test_run_check_writes_command_argv_and_collected_sum(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    receipt_out = tmp_path / "check-receipts.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/quant-research-platform-maintenance/scripts/run_check.py",
            "--id",
            "synthetic-pytest",
            "--scope",
            "unit",
            "--receipt-out",
            str(receipt_out),
            "--",
            sys.executable,
            "-c",
            "print('2 passed, 1 skipped, 3 deselected in 0.01s')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = json.loads(receipt_out.read_text(encoding="utf-8").splitlines()[0])
    assert result.returncode == 0
    assert receipt["command_argv"][0] == sys.executable
    assert receipt["command"]
    assert receipt["counts"]["collected"] == 6
