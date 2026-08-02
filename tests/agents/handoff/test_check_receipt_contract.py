from __future__ import annotations

import shlex

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    make_check_receipts_payload,
    make_valid_check_counts,
    make_valid_check_row,
    receipt_row_failures,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


def test_check_receipt_rejects_missing_schema_id() -> None:
    row = make_valid_check_row()
    row.pop("schema_id")

    assert "check_receipt_missing: 0:schema_id" in receipt_row_failures(row)

def test_check_receipt_rejects_extra_field() -> None:
    assert "check_receipt_unapproved_field: 0:extra_field" in receipt_row_failures(
        make_valid_check_row(extra_field=True)
    )

def test_check_receipt_rejects_pass_with_nonzero_exit_code() -> None:
    assert "check_receipt_pass_exit_code_invalid: 0" in receipt_row_failures(
        make_valid_check_row(exit_code=1)
    )

def test_check_receipt_rejects_fail_with_zero_exit_code() -> None:
    assert "check_receipt_fail_exit_code_invalid: 0" in receipt_row_failures(
        make_valid_check_row(status="FAIL", exit_code=0)
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
    assert "check_receipt_pass_counts_invalid: 0" in receipt_row_failures(
        make_valid_check_row(counts=make_valid_check_counts(**count_overrides))
    )


@pytest.mark.parametrize(
    "counts",
    (
        {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "collected": 0,
        },
        make_valid_check_counts(passed=0, skipped=1),
        make_valid_check_counts(passed=0, deselected=1),
        make_valid_check_counts(passed=1, deselected=1, collected=2),
    ),
    ids=(
        "zero-collected",
        "no-passed-body",
        "fully-deselected",
        "partially-deselected",
    ),
)
def test_profile_smoke_pass_requires_executed_body_and_zero_deselection(
    counts: dict[str, int],
) -> None:
    assert "check_receipt_pass_counts_invalid: 0" in receipt_row_failures(
        make_valid_check_row(counts=counts)
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
    assert receipt_row_failures(make_valid_check_row(**row_overrides)) == []

def test_check_receipt_requires_failure_class_for_non_pass_rows() -> None:
    failures = receipt_row_failures(make_valid_check_row(status="FAIL", exit_code=1))

    assert "check_receipt_failure_class_required: 0" in failures

def test_check_receipt_rejects_pass_with_failure_class_or_reason() -> None:
    failures = receipt_row_failures(
        make_valid_check_row(failure_class="pytest_failed", blocked_reason="timeout")
    )

    assert "check_receipt_pass_failure_class_invalid: 0" in failures
    assert "check_receipt_pass_reason_invalid: 0:blocked_reason" in failures

def test_check_receipt_requires_duration_seconds() -> None:
    failures = receipt_row_failures(make_valid_check_row(duration_seconds="slow"))

    assert "check_receipt_duration_seconds_invalid: 0" in failures
    row = make_valid_check_row()
    row.pop("duration_seconds")
    assert "check_receipt_missing: 0:duration_seconds" in receipt_row_failures(row)


@pytest.mark.parametrize("duration", (float("nan"), float("inf"), float("-inf")))
def test_check_receipt_rejects_nonfinite_duration(duration: float) -> None:
    assert "check_receipt_duration_seconds_invalid: 0" in receipt_row_failures(
        make_valid_check_row(duration_seconds=duration)
    )


def test_check_receipt_rejects_bool_for_integer_fields() -> None:
    assert "check_receipt_pass_exit_code_invalid: 0" in receipt_row_failures(
        make_valid_check_row(exit_code=False)
    )
    counts = make_valid_check_counts()
    counts["passed"] = True
    assert "check_receipt_counts_invalid: 0" in receipt_row_failures(
        make_valid_check_row(counts=counts)
    )


def test_check_receipt_rejects_nonfinite_timeout() -> None:
    failures = receipt_row_failures(
        make_valid_check_row(
            status="BLOCKED",
            exit_code=None,
            counts=None,
            failure_class="command_timeout",
            blocked_reason="timeout",
            timeout_seconds=float("inf"),
        )
    )

    assert "check_receipt_timeout_seconds_invalid: 0" in failures


def test_check_receipt_requires_exact_command_argv_list() -> None:
    row = make_valid_check_row()
    row["command_argv"] = tuple(row["command_argv"])

    assert "check_receipt_command_argv_invalid: 0" in receipt_row_failures(row)

def test_check_receipts_rejects_missing_or_extra_command() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_check_receipts_payload(checks=[])

    failures = contract.validate_check_receipts(payload)

    assert "check_receipts_missing_command: profile-smoke" in failures
    extra = make_valid_check_row("profile-smoke")
    extra["command_id"] = "extra"
    payload = make_check_receipts_payload(
        checks=[
            make_valid_check_row("profile-smoke"),
            extra,
        ]
    )
    failures = contract.validate_check_receipts(payload)
    assert "check_receipts_unapproved_command: extra" in failures

def test_check_receipt_rejects_command_display_or_argv_mismatch() -> None:
    assert "check_receipt_command_mismatch: 0" in receipt_row_failures(
        make_valid_check_row(command="python -m wrong")
    )
    row = make_valid_check_row(command_argv=["python", "-c", "print('hello world')"])
    row["command"] = shlex.join(row["command_argv"])
    failures = receipt_row_failures(row)
    assert "check_receipt_command_mismatch: 0" not in failures
    assert "check_receipt_command_argv_mismatch: profile-smoke" in failures
