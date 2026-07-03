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


def _class_source(path: Path, class_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_source = ast.get_source_segment(source, node)
            assert class_source is not None
            return class_source
    raise AssertionError(f"class not found: {class_name}")


def _function_source(path: Path, function_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_source = ast.get_source_segment(source, node)
            assert function_source is not None
            return function_source
    raise AssertionError(f"function not found: {function_name}")


def _called_names_in_function(path: Path, function_name: str) -> set[str]:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _name(*parts: str) -> str:
    return "_".join(parts)


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
    retired_package_smoke = "run_" + "package_" + "smoke.py"

    assert not (HANDOFF_ROOT / "proof_contract.py").exists()
    assert not (HANDOFF_ROOT / "collect_handoff_receipts.py").exists()
    assert not (HANDOFF_ROOT / retired_package_smoke).exists()
    assert not (SCRIPT_ROOT / "run_check.py").exists()
    assert not (TEMPLATE_ROOT / "task_metadata.template.json").exists()


def test_builder_exposes_only_scripted_handoff_cli() -> None:
    builder = _source(HANDOFF_ROOT / "build_handoff_package.py")
    flags = _argparse_flags(HANDOFF_ROOT / "build_handoff_package.py")

    assert '"--task"' not in builder
    assert '"--receipts-dir"' not in builder
    assert '"--mode"' not in builder
    assert '"--verify"' not in builder
    assert "pr_closure" not in builder
    assert "--task" not in flags
    assert "--receipts-dir" not in flags
    assert "--mode" not in flags
    assert "--verify" not in flags
    assert flags == {
        "--repo-root",
        "--out-dir",
        "--task-name",
        "--base-ref",
    }


def test_receipt_contract_is_payload_only() -> None:
    receipt_contract = HANDOFF_ROOT / "receipt_contract.py"
    receipt_defs = _function_defs(receipt_contract)
    source = _source(receipt_contract)

    assert "validate_receipt_file_set" not in receipt_defs
    assert "load_receipt_file_set" not in receipt_defs
    assert "check_statuses_by_command_id" not in receipt_defs
    assert "REQUIRED_RECEIPT_FILES" not in source
    assert "receipt_dir_missing" not in source
    assert 'glob("*.json")' not in source


def test_verify_governance_is_not_a_handoff_receipt_writer() -> None:
    verifier = SCRIPT_ROOT / "verify_governance.py"
    source = _source(verifier)
    flags = _argparse_flags(verifier)

    assert "--hard-scan-suite" not in flags
    assert "--receipt-out" not in flags
    assert "hard_scan_receipt_from_clone" not in source
    assert "receipt_path.write_text" not in source
    assert "qrp-hard-scan-receipt" not in source


def test_governance_contract_does_not_construct_hard_scan_receipts() -> None:
    governance_contract = SCRIPT_ROOT / "governance_contract.py"
    defs = _function_defs(governance_contract)
    names = _assigned_names(governance_contract)
    source = _source(governance_contract)

    assert "hard_scan_receipt_from_clone" not in defs
    assert "hard_scan_receipt_from_bundle" not in defs
    assert "source_tree_scan_payload" not in defs
    assert "source_tree_scan_from_clone" not in defs
    assert "HARD_SCAN_CLASSIFICATIONS" not in names
    assert "SUBAGENT_REVIEW_SEVERITIES" not in source
    assert "SUBAGENT_REVIEW_CLASSIFICATIONS" not in source
    assert "source_tree_scan_groups" in defs


def test_replay_evidence_does_not_import_retired_package_smoke_cli() -> None:
    replay = HANDOFF_ROOT / "replay_evidence.py"
    retired_module = "run_" + "package_" + "smoke"

    assert retired_module not in _imported_modules(replay)
    assert f"from {retired_module} import" not in _source(replay)
    assert "_build_package_artifacts" in _function_defs(replay)
    retired_extract = "_" + _name("in" + "stall", "package", "smoke", "target")
    assert retired_extract not in _function_defs(replay)
    assert "_extract_wheel_smoke_target" in _function_defs(replay)
    assert "_scan_package_artifacts" in _function_defs(replay)
    assert "_build_replayed_package_smoke_receipt" in _function_defs(replay)


def test_replay_evidence_owns_hard_scan_receipt_construction() -> None:
    replay = HANDOFF_ROOT / "replay_evidence.py"
    source = _source(replay)

    assert "hard_scan_receipt_from_clone" not in source
    assert "source_tree_scan_groups" in source
    assert "build_replayed_hard_scan_receipt" in _function_defs(replay)


def test_normalize_hard_scan_receipt_validates_before_canonicalization() -> None:
    receipt_contract = HANDOFF_ROOT / "receipt_contract.py"
    source = _source(receipt_contract)
    tree = ast.parse(source, filename=str(receipt_contract))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_hard_scan_receipt"
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None

    validation_index = function_source.index("validate_hard_scan_receipt(payload)")
    canonical_index = function_source.index("normalized_runs")

    assert validation_index < canonical_index


def test_handoff_readme_does_not_list_retired_package_smoke_cli() -> None:
    readme = _source(HANDOFF_ROOT / "README.md")
    retired_package_smoke = "run_" + "package_" + "smoke.py"

    assert retired_package_smoke not in readme


def test_dev_dependencies_include_package_build_backend_frontend() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "build>=1.5,<2" in pyproject["project"]["optional-dependencies"]["dev"]


def test_verifier_has_no_parallel_contract_functions() -> None:
    verifier_defs = _function_defs(HANDOFF_ROOT / "verify_handoff_package.py")
    source = _source(HANDOFF_ROOT / "verify_handoff_package.py")

    assert "verify_receipts" not in verifier_defs
    assert "verify_validation_summary_payload" not in verifier_defs
    assert "to_text" not in verifier_defs
    assert "def errors" not in source


def test_verifier_uses_single_result_inspection_and_replay_path() -> None:
    verifier = HANDOFF_ROOT / "verify_handoff_package.py"
    verifier_defs = _function_defs(verifier)
    classes = _class_defs(verifier)
    source = _source(verifier)

    assert "HandoffVerificationResult" in classes
    assert "HandoffPackagePayload" in classes
    assert "verify_handoff_package_result" in verifier_defs
    assert _name("verify", "handoff", "package", "envelope", "result") not in (
        verifier_defs
    )
    assert "load_handoff_package_payload" in verifier_defs
    assert _name("load", "handoff", "package", "payload", "once") not in (
        verifier_defs
    )
    assert "_validate_handoff_package_payload" in verifier_defs
    assert "verify_bundle_and_replay_once" in verifier_defs
    assert "scan_clone_tree_for_share_safety" in verifier_defs
    assert "compare_replayed_handoff_payloads" in verifier_defs
    assert "main" in verifier_defs

    payload_class_source = _class_source(verifier, "HandoffPackagePayload")
    for field_name in (
        "zip_path",
        "json_path",
        "zip_sha256",
        "member_names",
        "duplicate_member_names",
        "raw_bundle",
        "sidecar_payload",
        "task_metadata",
        "validation_summary",
        "receipts",
        "load_failures",
        "share_safety_failures",
    ):
        assert f"    {field_name}:" in payload_class_source

    for retired_class in ("Handoff" + "Safety" + "Result",):
        assert retired_class not in classes
        assert retired_class not in source

    for retired in (
        _name("verify", "handoff", "package"),
        _name("scan", "handoff", "zip"),
        _name("scan", "git", "bundle"),
        _name("verify", "manifest", "hashes"),
        _name("load", "handoff", "payloads"),
        _name("load", "json", "member"),
        _name("inspect", "handoff", "zip", "once"),
        "_" + _name("inspect", "handoff", "package", "envelope"),
    ):
        assert retired not in verifier_defs

    result_calls = _called_names_in_function(verifier, "verify_handoff_package_result")
    assert "load_handoff_package_payload" in result_calls
    assert "_validate_handoff_package_payload" in result_calls
    assert _name("scan", "handoff", "zip") not in result_calls
    assert _name("scan", "git", "bundle") not in result_calls

    validator_calls = _called_names_in_function(
        verifier, "_validate_handoff_package_payload"
    )
    assert "verify_bundle_and_replay_once" in validator_calls

    payload_source = _function_source(verifier, "load_handoff_package_payload")
    assert _name("scan", "bundle", "safety") not in payload_source
    assert _name("scan", "git", "bundle") not in payload_source


def test_handoff_contract_has_no_retired_verified_path_surfaces() -> None:
    handoff_contract = HANDOFF_ROOT / "handoff_contract.py"
    defs = _function_defs(handoff_contract)
    classes = _class_defs(handoff_contract)
    source = _source(handoff_contract)

    assert "pr_closure" not in source
    assert "ABSENCE_DECLARATION" not in source
    assert "EvidenceGraph" not in classes
    assert "validation_summary_status_map" not in defs
    assert "aggregate_validation_summary_status" not in defs
    assert "private_validation" not in source
    assert "baseline_compare" not in source


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


def test_current_docs_do_not_describe_retired_handoff_inputs() -> None:
    docs = [
        Path("docs/architecture/engineering-contract.md"),
        Path(".agents/briefs/user-principles.md"),
        Path(
            ".agents/skills/quant-research-platform-maintenance/references/"
            "handoff-and-evidence.md"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/references/"
            "verification-and-gates.md"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/references/"
            "tooling-and-portability.md"
        ),
        Path(".agents/skills/quant-research-platform-maintenance/SKILL.md"),
        HANDOFF_ROOT / "README.md",
    ]
    combined = "\n".join(_source(path) for path in docs)

    for retired in (
        "Builder 只接受 task metadata 和 receipts-dir",
        "builder 只读取 task metadata 和 receipts-dir",
        "--mode evidence",
        "pr_closure",
        "task_metadata.mode",
        "EvidenceGraph",
        "ABSENCE_DECLARATION",
        "private_validation",
        "baseline_compare",
        "SUBAGENT_REVIEW_SEVERITIES",
        "SUBAGENT_REVIEW_CLASSIFICATIONS",
        "--hard-scan-suite",
        "--receipt-out",
        "advisory-review-receipt",
        "advisory_reviews",
        "薄" + "包" + "装",
        "thin " + "wrapper",
        "compatibility " + "wrapper",
        "兼容" + "薄" + "包" + "装",
        "只是" + "薄" + "包" + "装",
        "wrapper " + "保留",
        "task metadata JSON 输入",
        "receipts-dir 输入",
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


def test_field_spec_and_allowed_keys_are_only_handoff_contract_owned() -> None:
    handoff_contract = HANDOFF_ROOT / "handoff_contract.py"
    receipt_contract = HANDOFF_ROOT / "receipt_contract.py"

    assert "FieldSpec" in _class_defs(handoff_contract)
    assert "required" not in _class_source(handoff_contract, "FieldSpec")
    assert "FieldSpec" not in _class_defs(receipt_contract)
    assert "def allowed_keys" in _source(handoff_contract)
    assert "def allowed_keys" not in _source(receipt_contract)

    for path in (
        HANDOFF_ROOT / "build_handoff_package.py",
        HANDOFF_ROOT / "verify_handoff_package.py",
        HANDOFF_ROOT / "replay_evidence.py",
    ):
        assert "FieldSpec" not in _class_defs(path)
        assert "def allowed_keys" not in _source(path)


def test_validation_row_source_is_receipt_entry_only() -> None:
    handoff_contract = HANDOFF_ROOT / "handoff_contract.py"
    source = _class_source(handoff_contract, "ValidationRowSource")

    assert "proof_class" not in source
    assert "receipt_entry" in source
    assert "command_id" in source


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

    assert "EvidenceGraph" not in handoff_contract
    assert "VALIDATION_ROW_SOURCE_CONTRACT" in handoff_contract
    assert "FieldSpec" not in _class_defs(HANDOFF_ROOT / "receipt_contract.py")
    assert "normalize_check_receipts" in receipt_contract
    assert "compare_check_receipts" in receipt_contract
    assert "normalize_package_smoke_receipt" in receipt_contract
    assert "compare_package_smoke_receipt" in receipt_contract
    assert "normalize_hard_scan_receipt" in receipt_contract
    assert "compare_hard_scan_receipt" in receipt_contract


def test_builder_uses_verifier_result_without_separate_scan_status() -> None:
    builder = HANDOFF_ROOT / "build_handoff_package.py"
    source = _source(builder)
    defs = _function_defs(builder)

    assert "scan_status" not in defs
    assert "_validate_handoff_package_payload" in source
    assert "load_handoff_package_payload" in source
    assert _name("verify", "handoff", "package", "envelope", "result") not in source
    assert "verify_handoff_package_result" not in source
    assert _name("scan", "handoff", "zip") not in source


def test_mutation_tests_do_not_patch_global_replay_function() -> None:
    source = _source(Path("tests/agents/handoff/test_handoff_package_mutations.py"))

    assert ".build_machine_receipts =" not in source
    assert "monkeypatch.setattr(verifier, \"build_machine_receipts\"" not in source


def test_mutation_manifest_refresh_reads_each_member_once() -> None:
    source = _function_source(
        Path("tests/agents/handoff/test_handoff_package_mutations.py"),
        "mutate_handoff_zip_and_refresh_hashes",
    )

    assert "raw = archive.read(name)" in source
    assert source.count("archive.read(name)") == 1
