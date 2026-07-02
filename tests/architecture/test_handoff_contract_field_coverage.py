from __future__ import annotations

from collections.abc import Mapping

try:
    from tests.architecture.conftest import _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _load_handoff_module  # type: ignore[no-redef]


def _specs(module: object, name: str) -> Mapping[str, object]:
    value = getattr(module, name)
    assert isinstance(value, Mapping)
    return value


def _assert_field_specs_have_proof(specs: Mapping[str, object]) -> None:
    for field, spec in specs.items():
        assert field
        assert spec.proof_class
        assert not hasattr(spec, "required")


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
        assert source.receipt_entry
        assert (
            row_name
            in handoff_contract.expected_validation_status_map_from_receipts(
                {
                    handoff_contract.CHECK_RECEIPTS_ENTRY: {
                        "schema_id": "qrp-check-receipts",
                        "status": "PASS",
                        "checks": [],
                    },
                    handoff_contract.PACKAGE_SMOKE_RECEIPT_ENTRY: {
                        "schema_id": "qrp-package-smoke-receipt",
                        "status": "PASS",
                    },
                    handoff_contract.HARD_SCAN_RECEIPT_ENTRY: {
                        "schema_id": "qrp-hard-scan-receipt",
                        "status": "PASS",
                    },
                }
            )
        )
        assert not hasattr(source, "proof_class")


def test_receipt_generated_fields_have_proof_classes() -> None:
    handoff_contract = _load_handoff_module("handoff_contract")
    receipt_contract = _load_handoff_module("receipt_contract")

    assert receipt_contract.FieldSpec is handoff_contract.FieldSpec

    for name in (
        "CHECK_RECEIPTS_FIELD_SPECS",
        "CHECK_RECEIPT_ROW_FIELD_SPECS",
        "PACKAGE_SMOKE_FIELD_SPECS",
        "PACKAGE_SMOKE_BUILD_FIELD_SPECS",
        "PACKAGE_SMOKE_WHEEL_EXTRACT_FIELD_SPECS",
        "PACKAGE_SMOKE_IMPORT_FIELD_SPECS",
        "PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS",
        "HARD_SCAN_FIELD_SPECS",
        "HARD_SCAN_RUN_FIELD_SPECS",
    ):
        specs = _specs(receipt_contract, name)
        _assert_field_specs_have_proof(specs)
        for spec in specs.values():
            assert isinstance(spec, handoff_contract.FieldSpec)


def test_receipt_allowed_keys_are_derived_from_field_specs() -> None:
    handoff_contract = _load_handoff_module("handoff_contract")
    receipt_contract = _load_handoff_module("receipt_contract")

    assert receipt_contract.allowed_keys is handoff_contract.allowed_keys

    for spec_name in (
        "CHECK_RECEIPTS_FIELD_SPECS",
        "CHECK_RECEIPT_ROW_FIELD_SPECS",
        "PACKAGE_SMOKE_FIELD_SPECS",
        "PACKAGE_SMOKE_BUILD_FIELD_SPECS",
        "PACKAGE_SMOKE_WHEEL_EXTRACT_FIELD_SPECS",
        "PACKAGE_SMOKE_IMPORT_FIELD_SPECS",
        "PACKAGE_SMOKE_CONTENT_SCAN_FIELD_SPECS",
        "HARD_SCAN_FIELD_SPECS",
        "HARD_SCAN_RUN_FIELD_SPECS",
    ):
        specs = _specs(receipt_contract, spec_name)
        assert handoff_contract.allowed_keys(specs) == frozenset(specs)


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
        receipt_contract.PACKAGE_SMOKE_WHEEL_EXTRACT_FIELD_SPECS["status"],
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
    assert all(
        source.receipt_entry
        for source in handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT.values()
    )
    assert all(
        not hasattr(source, "proof_class")
        for source in handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT.values()
    )
