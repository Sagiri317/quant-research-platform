from __future__ import annotations

import ast
from pathlib import Path

HANDOFF_ROOT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff"
)
SCRIPT_ROOT = Path(".agents/skills/quant-research-platform-maintenance/scripts")


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_handoff_contract_modules_are_present_and_own_contracts() -> None:
    handoff_contract = HANDOFF_ROOT / "handoff_contract.py"
    receipt_contract = HANDOFF_ROOT / "receipt_contract.py"
    governance_contract = SCRIPT_ROOT / "governance_contract.py"

    assert handoff_contract.is_file()
    assert receipt_contract.is_file()
    assert "REQUIRED_HANDOFF_ENTRIES" in _source(handoff_contract)
    assert "ALLOWED_SIDECAR_KEYS" in _source(handoff_contract)
    assert "validate_receipt_payload" in _source(receipt_contract)
    assert "REQUIRED_RECEIPT_FILES" in _source(receipt_contract)
    governance = _source(governance_contract)
    assert "REQUIRED_HANDOFF_ENTRIES" not in governance
    assert "ALLOWED_SIDECAR_KEYS" not in governance
    assert "REQUIRED_RECEIPT_FILES" not in governance
    assert "validate_receipt_payload" not in governance


def test_collector_builder_verifier_do_not_own_receipt_validators() -> None:
    local_validator_names = {
        "validate_package_smoke_receipt",
        "validate_hard_scan_receipt",
        "validate_check_receipts",
        "validate_advisory_review_receipt",
    }
    offenders: list[str] = []
    for path in (
        HANDOFF_ROOT / "collect_handoff_receipts.py",
        HANDOFF_ROOT / "build_handoff_package.py",
        HANDOFF_ROOT / "verify_handoff_package.py",
    ):
        tree = ast.parse(_source(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in local_validator_names:
                offenders.append(f"{path}:{node.name}")

    assert offenders == []


def test_retired_execution_symbols_are_absent_from_execution_path() -> None:
    execution_paths = [
        HANDOFF_ROOT / "build_handoff_package.py",
        HANDOFF_ROOT / "collect_handoff_receipts.py",
        HANDOFF_ROOT / "verify_handoff_package.py",
        HANDOFF_ROOT / "self_test.py",
    ]
    retired_symbols = (
        "GENERATION_EVIDENCE",
        "BUNDLE_RECONSTRUCTION_EVIDENCE",
        "PRIVATE_SMOKE_STATUS_EVIDENCE",
        "PACKAGE_SMOKE_EVIDENCE",
        "HARD_SCAN_CLASSIFICATION_EVIDENCE",
        "TEST_CHECKS_EVIDENCE",
        "BUILDER_OWNED_HANDOFF_EVIDENCE",
        "verify_private_smoke_status",
        "verify_test_checks_evidence",
        "verify_generation_evidence",
        "verify_bundle_reconstruction_evidence",
        "verify_git_status",
    )
    found = [
        f"{path}:{symbol}"
        for path in execution_paths
        for symbol in retired_symbols
        if symbol in _source(path)
    ]

    assert found == []


def test_retired_files_and_schema_contracts_are_absent() -> None:
    assert not (HANDOFF_ROOT / "collect_private_smoke_status.py").exists()
    assert not (
        Path(".agents/skills/quant-research-platform-maintenance/schemas/handoff")
        / "sidecar.schema.json"
    ).exists()
    assert not (
        Path(".agents/skills/quant-research-platform-maintenance/schemas/handoff")
        / "task_metadata.schema.json"
    ).exists()
    self_test = _source(HANDOFF_ROOT / "self_test.py")
    assert "task_metadata.template.json" not in self_test
    assert "handoff_package_path" not in self_test
    assert "handoff_sidecar_path" not in self_test


def test_retired_entry_names_are_only_contract_or_negative_test_owned() -> None:
    handoff_contract = _source(HANDOFF_ROOT / "handoff_contract.py")
    governance_contract = _source(SCRIPT_ROOT / "governance_contract.py")
    assert "FORBIDDEN_HANDOFF_ENTRIES" in handoff_contract
    assert "RETIRED_HANDOFF_ENTRY_NAMES" in governance_contract
    execution_combined = "\n".join(
        _source(path)
        for path in (
            HANDOFF_ROOT / "build_handoff_package.py",
            HANDOFF_ROOT / "collect_handoff_receipts.py",
            HANDOFF_ROOT / "verify_handoff_package.py",
        )
    )
    for retired in (
        "private-smoke-status.json",
        "private-validation-status.json",
        "generation.json",
        "bundle-reconstruction.json",
        "git_status.txt",
        "receipt-index.json",
    ):
        assert retired not in execution_combined
