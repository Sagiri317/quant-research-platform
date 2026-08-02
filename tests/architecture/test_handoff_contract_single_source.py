from __future__ import annotations

from tests.helpers.maintenance_script_loader import load_handoff_script_module


def test_exact_schema_helpers_have_one_contract_owner() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    receipt_contract = load_handoff_script_module("receipt_contract")

    assert receipt_contract.exact_keys is handoff_contract.exact_keys
    assert receipt_contract.sha is handoff_contract.sha


def test_validation_rows_and_replay_commands_are_closed() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    receipt_contract = load_handoff_script_module("receipt_contract")

    assert handoff_contract.MANDATORY_VALIDATION_ROWS == (
        "scripted_checks",
        "package_smoke",
        "profile_smoke",
        "governance",
    )
    assert set(handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT) == set(
        handoff_contract.MANDATORY_VALIDATION_ROWS
    )
    assert receipt_contract.REQUIRED_CHECK_IDS == ("profile-smoke",)
    assert set(handoff_contract.REPLAY_COMMAND_CONTRACT) == {
        "profile-smoke",
    }
    spec = handoff_contract.REPLAY_COMMAND_CONTRACT["profile-smoke"]
    assert spec.command_id == "profile-smoke"
    assert spec.scope == "profile-smoke"


def test_generated_field_specs_keep_explicit_proof_classes() -> None:
    handoff_contract = load_handoff_script_module("handoff_contract")
    receipt_contract = load_handoff_script_module("receipt_contract")

    spec_tables = (
        handoff_contract.SIDECAR_FIELD_SPECS,
        handoff_contract.TASK_METADATA_FIELD_SPECS,
        handoff_contract.ARTIFACT_MANIFEST_FIELD_SPECS,
        handoff_contract.ARTIFACT_MANIFEST_ENTRY_FIELD_SPECS,
        handoff_contract.VALIDATION_SUMMARY_FIELD_SPECS,
        handoff_contract.VALIDATION_ROW_FIELD_SPECS,
        receipt_contract.CHECK_RECEIPTS_FIELD_SPECS,
        receipt_contract.CHECK_RECEIPT_ROW_FIELD_SPECS,
        receipt_contract.PACKAGE_SMOKE_FIELD_SPECS,
        receipt_contract.HARD_SCAN_FIELD_SPECS,
    )

    assert all(
        table
        and all(
            isinstance(spec, handoff_contract.FieldSpec)
            and isinstance(spec.proof_class, handoff_contract.ProofClass)
            for spec in table.values()
        )
        for table in spec_tables
    )
