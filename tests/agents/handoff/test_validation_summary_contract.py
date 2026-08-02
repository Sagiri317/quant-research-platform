from __future__ import annotations

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


def test_validation_summary_rejects_unknown_row() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    rows = [
        *make_valid_validation_summary_rows(),
        {"name": "unknown", "status": "NOT_RUN"},
    ]
    payload = {"schema_id": "qrp-handoff-validation-summary", "rows": rows}

    failures = handoff_contract.validate_validation_summary_contract(payload)

    assert f"validation_summary_name_invalid: {len(rows) - 1}" in failures
