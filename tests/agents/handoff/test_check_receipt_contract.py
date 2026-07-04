from __future__ import annotations

import shlex

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    _failures,
    _valid_counts,
    _valid_row,
    make_check_receipts_payload,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


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
        _valid_row(status="FAIL", exit_code=0)
    )

@pytest.mark.parametrize(
    "count_overrides",
    (
        {"failed": 1, "collected": 2},
        {"errors": 1, "collected": 2},
    ),
)
def test_check_receipt_rejects_pass_with_failed_or_error_count(
    count_overrides: dict[str, int],
) -> None:
    assert "check_receipt_pass_counts_invalid: 0" in _failures(
        _valid_row(counts=_valid_counts(**count_overrides))
    )

@pytest.mark.parametrize(
    "row_overrides",
    (
        {
            "status": "NOT_RUN",
            "exit_code": None,
            "counts": None,
            "failure_class": "runner_preflight_error",
            "not_run_reason": "no_tests_collected",
        },
        {
            "status": "BLOCKED",
            "exit_code": None,
            "counts": None,
            "failure_class": "command_timeout",
            "blocked_reason": "timeout",
            "timeout_seconds": 1,
        },
    ),
)
def test_check_receipt_accepts_not_run_and_blocked_rows_with_reasons(
    row_overrides: dict[str, object],
) -> None:
    assert _failures(_valid_row(**row_overrides)) == []

def test_check_receipt_requires_failure_class_for_non_pass_rows() -> None:
    failures = _failures(_valid_row(status="FAIL", exit_code=1))

    assert "check_receipt_failure_class_required: 0" in failures

def test_check_receipt_rejects_pass_with_failure_class_or_reason() -> None:
    failures = _failures(
        _valid_row(failure_class="pytest_failed", blocked_reason="timeout")
    )

    assert "check_receipt_pass_failure_class_invalid: 0" in failures
    assert "check_receipt_pass_reason_invalid: 0:blocked_reason" in failures

def test_check_receipt_requires_duration_seconds() -> None:
    failures = _failures(_valid_row(duration_seconds="slow"))

    assert "check_receipt_duration_seconds_invalid: 0" in failures
    row = _valid_row()
    row.pop("duration_seconds")
    assert "check_receipt_missing: 0:duration_seconds" in _failures(row)

def test_check_receipt_rejects_old_trace_fields() -> None:
    failures = _failures(
        _valid_row(
            stdout_sha256="0" * 64,
            stderr_sha256="1" * 64,
            output_summary="old",
            failure_excerpt="old",
        )
    )

    for field in (
        "stdout_sha256",
        "stderr_sha256",
        "output_summary",
        "failure_excerpt",
    ):
        assert f"check_receipt_unapproved_field: 0:{field}" in failures

def test_check_receipts_rejects_missing_or_extra_command() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_check_receipts_payload(checks=[_valid_row("profile-smoke")])

    failures = contract.validate_check_receipts(payload)

    assert "check_receipts_missing_command: code-map" in failures
    extra = _valid_row("profile-smoke")
    extra["command_id"] = "extra"
    payload = make_check_receipts_payload(
        checks=[_valid_row("profile-smoke"), _valid_row("code-map"), extra]
    )
    failures = contract.validate_check_receipts(payload)
    assert "check_receipts_unapproved_command: extra" in failures

def test_check_receipt_rejects_command_display_or_argv_mismatch() -> None:
    assert "check_receipt_command_mismatch: 0" in _failures(
        _valid_row(command="python -m wrong")
    )
    row = _valid_row(command_argv=["python", "-c", "print('hello world')"])
    row["command"] = shlex.join(row["command_argv"])
    failures = _failures(row)
    assert "check_receipt_command_mismatch: 0" not in failures
    assert "check_receipt_command_argv_mismatch: profile-smoke" in failures
