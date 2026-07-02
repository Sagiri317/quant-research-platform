from __future__ import annotations

import ast
import tomllib
from pathlib import Path

HANDOFF_ROOT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff"
)
SCRIPT_ROOT = Path(".agents/skills/quant-research-platform-maintenance/scripts")
TEMPLATE_ROOT = Path(
    ".agents/skills/quant-research-platform-maintenance/templates/handoff"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_defs(path: Path) -> set[str]:
    tree = ast.parse(_source(path), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _class_defs(path: Path) -> set[str]:
    tree = ast.parse(_source(path), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _assigned_names(path: Path) -> set[str]:
    tree = ast.parse(_source(path), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(_source(path), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _argparse_flags(path: Path) -> set[str]:
    tree = ast.parse(_source(path), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def test_retired_handoff_files_are_absent() -> None:
    assert not (HANDOFF_ROOT / "proof_contract.py").exists()
    assert not (HANDOFF_ROOT / "collect_handoff_receipts.py").exists()
    assert not (HANDOFF_ROOT / "run_package_smoke.py").exists()
    assert not (SCRIPT_ROOT / "run_check.py").exists()
    assert not (TEMPLATE_ROOT / "task_metadata.template.json").exists()


def test_builder_exposes_only_scripted_handoff_cli() -> None:
    builder = _source(HANDOFF_ROOT / "build_handoff_package.py")
    flags = _argparse_flags(HANDOFF_ROOT / "build_handoff_package.py")

    assert '"--task"' not in builder
    assert '"--receipts-dir"' not in builder
    assert "--task" not in flags
    assert "--receipts-dir" not in flags
    for flag in (
        "--repo-root",
        "--out-dir",
        "--task-name",
        "--base-ref",
        "--mode",
        "--verify",
    ):
        assert flag in builder
        assert flag in flags


def test_receipt_contract_is_payload_only() -> None:
    receipt_contract = HANDOFF_ROOT / "receipt_contract.py"
    receipt_defs = _function_defs(receipt_contract)
    source = _source(receipt_contract)

    assert "validate_receipt_file_set" not in receipt_defs
    assert "load_receipt_file_set" not in receipt_defs
    assert "REQUIRED_RECEIPT_FILES" not in source
    assert "receipt_dir_missing" not in source
    assert 'glob("*.json")' not in source


def test_replay_evidence_does_not_import_retired_package_smoke_cli() -> None:
    replay = HANDOFF_ROOT / "replay_evidence.py"

    assert "run_package_smoke" not in _imported_modules(replay)
    assert "from run_package_smoke import" not in _source(replay)
    assert "_build_package_artifacts" in _function_defs(replay)
    assert "_install_package_smoke_target" in _function_defs(replay)
    assert "_scan_package_artifacts" in _function_defs(replay)
    assert "_build_replayed_package_smoke_receipt" in _function_defs(replay)


def test_handoff_readme_does_not_list_retired_package_smoke_cli() -> None:
    readme = _source(HANDOFF_ROOT / "README.md")

    assert "run_package_smoke.py" not in readme


def test_dev_dependencies_include_package_build_backend_frontend() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "build>=1.5,<2" in pyproject["project"]["optional-dependencies"]["dev"]


def test_verifier_has_no_parallel_contract_functions() -> None:
    verifier_defs = _function_defs(HANDOFF_ROOT / "verify_handoff_package.py")

    assert "verify_receipts" not in verifier_defs
    assert "verify_validation_summary_payload" not in verifier_defs


def test_advisory_machine_contract_is_absent() -> None:
    combined = "\n".join(
        _source(path)
        for path in (
            HANDOFF_ROOT / "handoff_contract.py",
            HANDOFF_ROOT / "receipt_contract.py",
            HANDOFF_ROOT / "build_handoff_package.py",
            HANDOFF_ROOT / "verify_handoff_package.py",
            HANDOFF_ROOT / "replay_evidence.py",
        )
        if path.exists()
    )

    for retired in (
        "advisory-review-receipt.json",
        "advisory_reviews",
        "ADVISORY_REVIEW",
        "validate_advisory_review_receipt",
        "validate_advisory_reviews_against_receipts",
    ):
        assert retired not in combined


def test_proof_class_is_only_defined_by_handoff_contract() -> None:
    offenders = [
        path
        for path in HANDOFF_ROOT.glob("*.py")
        if path.name != "handoff_contract.py" and "class ProofClass" in _source(path)
    ]

    assert "class ProofClass" in _source(HANDOFF_ROOT / "handoff_contract.py")
    assert offenders == []


def test_replay_command_contract_is_only_handoff_contract_owned() -> None:
    owner = HANDOFF_ROOT / "handoff_contract.py"
    assert "REPLAY_COMMAND_CONTRACT" in _source(owner)
    offenders = [
        path
        for path in (
            HANDOFF_ROOT / "build_handoff_package.py",
            HANDOFF_ROOT / "verify_handoff_package.py",
            HANDOFF_ROOT / "replay_evidence.py",
            HANDOFF_ROOT / "receipt_contract.py",
        )
        if "REPLAY_COMMAND_CONTRACT" in _source(path)
    ]

    assert offenders == []


def test_receipt_field_specs_are_only_receipt_contract_owned() -> None:
    receipt_contract = HANDOFF_ROOT / "receipt_contract.py"
    assert "FieldSpec" in _class_defs(receipt_contract)
    for path in (
        HANDOFF_ROOT / "build_handoff_package.py",
        HANDOFF_ROOT / "verify_handoff_package.py",
        HANDOFF_ROOT / "replay_evidence.py",
    ):
        assert "FieldSpec" not in _class_defs(path)


def test_execution_modules_do_not_define_contract_tables() -> None:
    disallowed_suffixes = (
        "_KEYS",
        "_FIELD_SPECS",
        "_PROOF",
        "_PROOFS",
        "_RECEIPT_KEYS",
    )
    offenders: list[str] = []
    for path in (
        HANDOFF_ROOT / "build_handoff_package.py",
        HANDOFF_ROOT / "verify_handoff_package.py",
        HANDOFF_ROOT / "replay_evidence.py",
    ):
        for name in _assigned_names(path):
            if name.endswith(disallowed_suffixes):
                offenders.append(f"{path.name}:{name}")

    assert offenders == []


def test_contract_modules_own_their_respective_boundaries() -> None:
    handoff_contract = _source(HANDOFF_ROOT / "handoff_contract.py")
    receipt_contract = _source(HANDOFF_ROOT / "receipt_contract.py")

    assert "EvidenceGraph" in handoff_contract
    assert "VALIDATION_ROW_SOURCE_CONTRACT" in handoff_contract
    assert "normalize_check_receipts" in receipt_contract
    assert "compare_check_receipts" in receipt_contract
    assert "normalize_package_smoke_receipt" in receipt_contract
    assert "compare_package_smoke_receipt" in receipt_contract
    assert "normalize_hard_scan_receipt" in receipt_contract
    assert "compare_hard_scan_receipt" in receipt_contract
