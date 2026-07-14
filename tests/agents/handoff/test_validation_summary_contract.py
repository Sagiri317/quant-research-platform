from __future__ import annotations

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    make_valid_validation_summary_rows,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


def test_artifact_manifest_requires_exact_list_and_integer_size() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    payload = {
        "schema_id": "qrp-handoff-artifact-manifest",
        "entries": [
            {"path": "final-branch.bundle", "size": True, "sha256": "0" * 64}
        ],
    }

    failures = handoff_contract.validate_artifact_manifest_contract(payload)

    assert "artifact_manifest_size_invalid: 0" in failures
    payload["entries"] = tuple(payload["entries"])
    assert "artifact_manifest_entries_invalid" in (
        handoff_contract.validate_artifact_manifest_contract(payload)
    )


@pytest.mark.parametrize("row_name", ("private_validation", "baseline_compare"))
def test_validation_summary_rejects_deleted_rows(row_name: str) -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    rows = [
        *make_valid_validation_summary_rows(),
        {"name": row_name, "status": "NOT_RUN"},
    ]
    payload = {"schema_id": "qrp-handoff-validation-summary", "rows": rows}

    failures = handoff_contract.validate_validation_summary_contract(payload)

    assert f"validation_summary_name_invalid: {len(rows) - 1}" in failures

def test_task_metadata_rejects_deleted_mode_field() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    payload = {
        "schema_id": "qrp-handoff-task-metadata",
        "task_name": "handoff-test",
        "mode": "evidence",
        "branch": "feature/test",
        "base": "1" * 40,
        "final_head": "2" * 40,
    }

    failures = handoff_contract.validate_task_metadata_contract(payload)

    assert "task_metadata_unapproved_field: mode" in failures

def test_proof_class_has_no_absence_declaration() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")

    assert "ABSENCE_DECLARATION" not in handoff_contract.ProofClass.__members__
