from __future__ import annotations

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    make_hard_scan_receipt_payload,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


def test_hard_scan_rejects_old_classification_and_count_fields() -> None:
    contract = load_handoff_script_module("receipt_contract")
    run = {
        **make_hard_scan_receipt_payload()["scan_runs"][0],  # type: ignore[index]
        "deferred_count": 0,
        "negative_fixture_count": 0,
        "false_positive_count": 0,
    }
    payload = make_hard_scan_receipt_payload(
        scan_runs=[run],
        classified_exceptions=[],
    )

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_unapproved_field: classified_exceptions" in failures
    assert "hard_scan_run_unapproved_field: 0:deferred_count" in failures
    assert "hard_scan_run_unapproved_field: 0:negative_fixture_count" in failures
    assert "hard_scan_run_unapproved_field: 0:false_positive_count" in failures

@pytest.mark.parametrize(
    ("payload_overrides", "run_overrides", "expected"),
    (
        (
            {"classified_exceptions": []},
            {},
            "hard_scan_unapproved_field: classified_exceptions",
        ),
        ({"extra": True}, {}, "hard_scan_unapproved_field: extra"),
        ({}, {"deferred_count": 0}, "hard_scan_run_unapproved_field: 0:deferred_count"),
        (
            {},
            {"negative_fixture_count": 0},
            "hard_scan_run_unapproved_field: 0:negative_fixture_count",
        ),
        (
            {},
            {"false_positive_count": 0},
            "hard_scan_run_unapproved_field: 0:false_positive_count",
        ),
        ({}, {"extra": True}, "hard_scan_run_unapproved_field: 0:extra"),
    ),
)
def test_normalize_hard_scan_rejects_raw_extra_fields_before_projection(
    payload_overrides: dict[str, object],
    run_overrides: dict[str, object],
    expected: str,
) -> None:
    contract = load_handoff_script_module("receipt_contract")
    run = {
        **make_hard_scan_receipt_payload()["scan_runs"][0],  # type: ignore[index]
        **run_overrides,
    }
    payload = make_hard_scan_receipt_payload(scan_runs=[run], **payload_overrides)

    with pytest.raises(SystemExit) as excinfo:
        contract.normalize_hard_scan_receipt(payload)

    assert expected in str(excinfo.value)

def test_hard_scan_rejects_actual_violation_with_top_pass() -> None:
    contract = load_handoff_script_module("receipt_contract")
    run = {
        **make_hard_scan_receipt_payload()["scan_runs"][0],  # type: ignore[index]
        "actual_violation_count": 1,
    }
    payload = make_hard_scan_receipt_payload(scan_runs=[run])

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_status_mismatch" in failures
