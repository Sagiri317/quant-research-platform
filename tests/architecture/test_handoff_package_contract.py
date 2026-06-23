from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

HANDOFF_ROOT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff"
)
GOVERNANCE_CONTRACT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "governance_contract.py"
)


def test_handoff_package_contract_uses_evidence_mode_outputs() -> None:
    builder = (HANDOFF_ROOT / "build_handoff_package.py").read_text(encoding="utf-8")
    verifier = (HANDOFF_ROOT / "verify_handoff_package.py").read_text(encoding="utf-8")

    assert "handoff_evidence/" in builder
    assert "handoff_evidence/" in verifier
    assert "handoff_evidence/generation.json" in builder
    assert "handoff_evidence/generation.json" in verifier
    assert "handoff_evidence/source-tree-scan.json" in builder
    assert "handoff_evidence/source-tree-scan.json" in verifier
    assert "handoff_generation.json" not in builder
    assert "handoff_generation.json" not in verifier
    source_tree_scan = _load_handoff_builder().source_tree_scan_groups(Path("."))
    assert source_tree_scan["retired_text"]["findings"] == []
    assert source_tree_scan["retired_governance_paths"]["findings"] == []
    assert source_tree_scan["retired_path_names"]["findings"] == []
    assert source_tree_scan["python_symbols"]["findings"] == []
    assert source_tree_scan["synthesized_strings"]["findings"] == []
    assert source_tree_scan["structured_keys"]["findings"] == []
    assert source_tree_scan["governance_truth"]["findings"] == []
    assert "governance_truth" in verifier
    assert '".run/handoffs"' in builder
    assert '"evidence"' in builder
    assert '"pr_closure"' in builder


def test_pr_closure_only_entries_are_not_required_for_evidence_mode() -> None:
    builder = _load_handoff_builder()
    verifier = _load_handoff_verifier()

    assert verifier.REQUIRED_ZIP_ENTRIES.isdisjoint(
        builder.RESERVED_HANDOFF_EVIDENCE_STEMS
    )


def test_source_tree_scan_synthesizes_static_string_fragments(tmp_path: Path) -> None:
    builder = _load_handoff_builder()
    sample = tmp_path / "sample.py"
    source = 'VALUE = "al" + "pha"\nADJACENT = "be" "ta"\n'
    sample.write_text(source, encoding="utf-8")

    findings = builder.source_tree_synthesized_string_findings(
        Path("sample.py"),
        sample,
        source.splitlines(),
        set(),
        set(),
        scan_group="synthesized_strings",
        patterns={
            "synthetic_alpha": re.compile(r"\balpha\b"),
            "synthetic_beta": re.compile(r"\bbeta\b"),
        },
    )

    observed = {
        (finding["rule"], finding["scan_mode"])
        for finding in findings
    }
    assert ("synthetic_alpha", "ast_static_string") in observed
    assert ("synthetic_alpha", "token_static_string") in observed
    assert ("synthetic_beta", "token_static_string") in observed


def test_source_tree_scan_rule_allowlist_is_definition_scoped() -> None:
    builder = _load_handoff_builder()
    relative = Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "build_handoff_package.py"
    )
    rule = next(iter(builder.SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST[relative]))

    assert not builder.source_tree_path_allowed(
        relative,
        scan_group="retired_text",
        rule=rule,
        line_text='"rule-definition-example"',
        line_number=10,
        rule_definition_lines=set(),
    )
    assert builder.source_tree_path_allowed(
        relative,
        scan_group="retired_text",
        rule=rule,
        line_text='"rule-definition-example"',
        line_number=10,
        rule_definition_lines={10},
    )


def test_source_tree_scan_rejects_retired_governance_path_resurrection(
    tmp_path: Path,
) -> None:
    builder = _load_handoff_builder()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    (repo / "PROJECT_STATUS.md").write_text("retired status\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired governance path")

    groups = builder.source_tree_scan_groups(repo)

    assert groups["retired_governance_paths"]["status"] == "FAIL"
    assert groups["retired_governance_paths"]["findings"] == [
        {"path": "PROJECT_STATUS.md", "rule": "retired_governance_exact_path"}
    ]


def test_source_tree_scan_rejects_retired_stage_path_names(tmp_path: Path) -> None:
    builder = _load_handoff_builder()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    path = repo / "tests" / "architecture" / "test_CLEAN_REWRITE_fake.py"
    path.parent.mkdir(parents=True)
    path.write_text("def test_placeholder():\n    pass\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired stage path")

    groups = builder.source_tree_scan_groups(repo)

    assert groups["retired_path_names"]["status"] == "FAIL"
    assert groups["retired_path_names"]["findings"] == [
        {
            "path": "tests/architecture/test_CLEAN_REWRITE_fake.py",
            "rule": "clean_rewrite",
        }
    ]


def test_source_tree_scan_rejects_retired_structured_keys(tmp_path: Path) -> None:
    builder = _load_handoff_builder()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    path = repo / "docs" / "metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"CLEAN_REWRITE_FLAG": true}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired structured key")

    groups = builder.source_tree_scan_groups(repo)

    assert groups["structured_keys"]["status"] == "FAIL"
    assert groups["structured_keys"]["findings"] == [
        {
            "path": "docs/metadata.json",
            "line": 0,
            "key_path": "CLEAN_REWRITE_FLAG",
            "key": "CLEAN_REWRITE_FLAG",
            "rule": "clean_rewrite",
        }
    ]


def test_source_tree_scan_independently_rejects_python_symbols_and_synthesized_strings(
    tmp_path: Path,
) -> None:
    builder = _load_handoff_builder()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    path = repo / "docs" / "architecture" / "hack_fixture.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "class QRPCleanRewriteHelper:",
                "    pass",
                'adjacent = "QRP " "Clean Rewrite"',
                'joined = "".join(["clean", "-rewrite"])',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired symbol fixture")

    groups = builder.source_tree_scan_groups(repo)

    assert groups["python_symbols"]["status"] == "FAIL"
    assert groups["synthesized_strings"]["status"] == "FAIL"
    assert {
        (finding["path"], finding["symbol"], finding["rule"])
        for finding in groups["python_symbols"]["findings"]
    } == {
        (
            "docs/architecture/hack_fixture.py",
            "QRPCleanRewriteHelper",
            "clean_rewrite",
        )
    }
    observed = {
        (finding["path"], finding["rule"], finding["scan_mode"])
        for finding in groups["synthesized_strings"]["findings"]
    }
    assert (
        "docs/architecture/hack_fixture.py",
        "clean_rewrite",
        "token_static_string",
    ) in observed
    assert (
        "docs/architecture/hack_fixture.py",
        "clean_rewrite",
        "ast_join_string",
    ) in observed


def test_handoff_source_tree_rules_import_governance_contract() -> None:
    builder = _load_handoff_builder()
    contract = _load_governance_contract()

    assert tuple(
        path.as_posix() + ("/" if path.is_dir() else "")
        for path in builder.SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS
    ) == contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    assert builder.source_tree_forbidden_patterns().keys() == (
        contract.source_tree_forbidden_patterns().keys()
    )


def test_public_safe_text_checks_use_runtime_privacy_single_source() -> None:
    agent_safety = Path("src/quant_research_platform/agents/safety.py").read_text(
        encoding="utf-8"
    )
    handoff_safety = (HANDOFF_ROOT / "handoff_safety.py").read_text(encoding="utf-8")
    report_safety = Path(
        "src/quant_research_platform/reports/verification_evidence.py"
    ).read_text(encoding="utf-8")

    assert "public_text_unsafe_reason" in agent_safety
    assert "public_text_unsafe_reason" in handoff_safety
    assert "public_text_unsafe_reason" in report_safety
    forbidden_local_fact_sources = {
        "_UNSAFE_TEXT_PATTERNS",
        "DRIVE_ABSOLUTE_RE",
        "POSIX_USER_ABSOLUTE_RE",
        "ABSOLUTE_REPO_PRIVATE_PATH_RE",
        "WINDOWS_USER_ABSOLUTE_RE",
        "STOCK_CODE_RE",
        "CREDENTIAL_RE",
        "VENDOR_MARKER_RE",
        "RAW_TRACEBACK_RE",
    }
    combined = agent_safety + "\n" + handoff_safety + "\n" + report_safety

    assert [name for name in forbidden_local_fact_sources if name in combined] == []


def _load_handoff_builder():
    path = HANDOFF_ROOT / "build_handoff_package.py"
    spec = importlib.util.spec_from_file_location("handoff_builder_contract_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_handoff_verifier():
    path = HANDOFF_ROOT / "verify_handoff_package.py"
    spec = importlib.util.spec_from_file_location(
        "handoff_verifier_contract_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_governance_contract():
    spec = importlib.util.spec_from_file_location(
        "governance_contract_contract_test",
        GOVERNANCE_CONTRACT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
