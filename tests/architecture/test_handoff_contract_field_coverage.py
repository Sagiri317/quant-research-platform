from __future__ import annotations

from collections.abc import Mapping

try:
    from tests.agents.handoff.conftest import _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _load_handoff_module  # type: ignore[no-redef]


def _specs(module: object, name: str) -> Mapping[str, object]:
    value = getattr(module, name)
    assert isinstance(value, Mapping)
    return value


def _assert_field_specs_have_proof(specs: Mapping[str, object]) -> None:
    for field, spec in specs.items():
        assert field
        assert getattr(spec, "proof_class")


def test_handoff_generated_fields_have_proof_classes() -> None:
    handoff_contract = _load_handoff_module("handoff_contract")

    assert set(handoff_contract.ProofClass.__members__) == {
        "REPLAYED",
        "DERIVED_FROM_REPLAY",
        "IDENTITY_BOUND",
        "METADATA_VALIDATED",
    }
    assert set(handoff_contract.TASK_METADATA_FIELD_SPECS) == {
        "schema_id",
        "task_name",
        "branch",
        "base",
        "final_head",
    }
    assert set(handoff_contract.MANDATORY_VALIDATION_ROWS) == {
        "scripted_checks",
        "package_smoke",
        "profile_smoke",
        "governance",
        "code_map",
    }

    for name in (
        "SIDECAR_FIELD_SPECS",
        "TASK_METADATA_FIELD_SPECS",
        "ARTIFACT_MANIFEST_FIELD_SPECS",
        "ARTIFACT_MANIFEST_ENTRY_FIELD_SPECS",
        "VALIDATION_SUMMARY_FIELD_SPECS",
        "VALIDATION_ROW_FIELD_SPECS",
    ):
        _assert_field_specs_have_proof(_specs(handoff_contract, name))

    for row_name, source in handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT.items():
        assert row_name in handoff_contract.MANDATORY_VALIDATION_ROWS
        assert source.proof_class


def test_receipt_generated_fields_have_proof_classes() -> None:
    receipt_contract = _load_handoff_module("receipt_contract")

    for name in (
        "CHECK_RECEIPTS_FIELD_SPECS",
        "CHECK_RECEIPT_ROW_FIELD_SPECS",
        "PACKAGE_SMOKE_FIELD_SPECS",
        "PACKAGE_SMOKE_BUILD_FIELD_SPECS",
        "PACKAGE_SMOKE_INSTALL_FIELD_SPECS",
        "PACKAGE_SMOKE_IMPORT_FIELD_SPECS",
        "PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS",
        "HARD_SCAN_FIELD_SPECS",
        "HARD_SCAN_RUN_FIELD_SPECS",
    ):
        _assert_field_specs_have_proof(_specs(receipt_contract, name))


def test_receipt_allowed_keys_are_derived_from_field_specs() -> None:
    receipt_contract = _load_handoff_module("receipt_contract")

    for spec_name in (
        "CHECK_RECEIPTS_FIELD_SPECS",
        "CHECK_RECEIPT_ROW_FIELD_SPECS",
        "PACKAGE_SMOKE_FIELD_SPECS",
        "PACKAGE_SMOKE_BUILD_FIELD_SPECS",
        "PACKAGE_SMOKE_INSTALL_FIELD_SPECS",
        "PACKAGE_SMOKE_IMPORT_FIELD_SPECS",
        "PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS",
        "HARD_SCAN_FIELD_SPECS",
        "HARD_SCAN_RUN_FIELD_SPECS",
    ):
        specs = _specs(receipt_contract, spec_name)
        assert receipt_contract.allowed_keys(specs) == frozenset(specs)


def test_status_pass_proof_classes_are_replayed_or_derived() -> None:
    handoff_contract = _load_handoff_module("handoff_contract")
    receipt_contract = _load_handoff_module("receipt_contract")

    allowed = {
        handoff_contract.ProofClass.REPLAYED,
        handoff_contract.ProofClass.DERIVED_FROM_REPLAY,
    }
    forbidden = {
        handoff_contract.ProofClass.METADATA_VALIDATED,
    }
    status_specs = [
        handoff_contract.VALIDATION_ROW_FIELD_SPECS["status"],
        receipt_contract.CHECK_RECEIPTS_FIELD_SPECS["status"],
        receipt_contract.CHECK_RECEIPT_ROW_FIELD_SPECS["status"],
        receipt_contract.PACKAGE_SMOKE_FIELD_SPECS["status"],
        receipt_contract.PACKAGE_SMOKE_BUILD_FIELD_SPECS["status"],
        receipt_contract.PACKAGE_SMOKE_INSTALL_FIELD_SPECS["status"],
        receipt_contract.PACKAGE_SMOKE_IMPORT_FIELD_SPECS["status"],
        receipt_contract.PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS["status"],
        receipt_contract.HARD_SCAN_FIELD_SPECS["status"],
    ]

    for spec in status_specs:
        assert spec.proof_class in allowed
        assert spec.proof_class not in forbidden

    assert set(handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT) == set(
        handoff_contract.MANDATORY_VALIDATION_ROWS
    )
