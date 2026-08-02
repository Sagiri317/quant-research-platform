from __future__ import annotations

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    make_hard_scan_receipt_payload,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


@pytest.mark.parametrize(
    ("payload_overrides", "run_overrides", "expected"),
    (
        ({"extra": True}, {}, "hard_scan_unapproved_field: extra"),
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


def test_hard_scan_requires_exact_list_and_integer_counts() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_hard_scan_receipt_payload()
    payload["scan_runs"][0]["match_count"] = False

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_run_count_invalid: 0:match_count" in failures
    payload["scan_runs"] = tuple(payload["scan_runs"])
    assert "hard_scan_runs_invalid" in contract.validate_hard_scan_receipt(payload)
