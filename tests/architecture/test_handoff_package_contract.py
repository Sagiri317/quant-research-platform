from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

from tests.helpers.governance_fixtures import (
    adjacent_string_assignment,
    derived_identifier,
    forms_matching_pattern,
    joined_string_assignment,
    rule_form,
    structured_key_document,
    text_rule_example,
)

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


def test_handoff_package_contract_uses_evidence_mode_outputs(
    live_source_tree_scan: dict[str, dict[str, object]],
) -> None:
    builder = (HANDOFF_ROOT / "build_handoff_package.py").read_text(encoding="utf-8")
    verifier = (HANDOFF_ROOT / "verify_handoff_package.py").read_text(encoding="utf-8")

    assert "normalized_handoff_evidence_path" in builder
    assert "stage_handoff_evidence" in builder
    assert "handoff_evidence/" in verifier
    contract = _load_governance_contract()
    assert contract.GENERATION_EVIDENCE == "handoff_evidence/generation.json"
    assert contract.SOURCE_TREE_SCAN_EVIDENCE == (
        "handoff_evidence/source-tree-scan.json"
    )
    assert "GENERATION_EVIDENCE" in builder
    assert "handoff_evidence/generation.json" in verifier
    assert "SOURCE_TREE_SCAN_EVIDENCE" in builder
    assert "handoff_evidence/source-tree-scan.json" in verifier
    assert "handoff_generation.json" not in builder
    assert "handoff_generation.json" not in verifier
    source_tree_scan = live_source_tree_scan
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
    assert "baseline-final" not in builder
    assert "baseline-final" not in verifier
    assert "old_baseline" not in verifier
    assert "refreshed_prediction_baseline" not in verifier
    assert "def aggregate_component_status" not in verifier
    assert "def aggregate_validation_matrix_status" not in verifier
    assert "def aggregate_status" not in builder
    assert "BLOCKED_DUPLICATE_SOURCE_ROOTS =" not in verifier
    assert "FORBIDDEN_PATH_SEGMENTS =" not in verifier
    assert "PACKAGE_FORBIDDEN_PATH_PARTS" in verifier
    assert "BLOCKED_DUPLICATE_SOURCE_ROOT_PREFIXES" in verifier


def test_pr_closure_only_entries_are_not_required_for_evidence_mode() -> None:
    contract = _load_governance_contract()
    verifier = _load_handoff_verifier()

    assert verifier.REQUIRED_ZIP_ENTRIES.isdisjoint(
        contract.RESERVED_HANDOFF_EVIDENCE_STEMS
    )


def test_handoff_negative_token_is_derived_from_governance_contract() -> None:
    contract = _load_governance_contract()

    assert rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_review_bundle",
    )


def test_source_tree_scan_synthesizes_static_string_fragments(tmp_path: Path) -> None:
    contract = _load_governance_contract()
    sample = tmp_path / "sample.py"
    source = 'VALUE = "al" + "pha"\nADJACENT = "be" "ta"\n'
    sample.write_text(source, encoding="utf-8")

    findings = contract.source_tree_synthesized_string_findings(
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

    observed = {(finding["rule"], finding["scan_mode"]) for finding in findings}
    assert ("synthetic_alpha", "ast_static_string") in observed
    assert ("synthetic_alpha", "token_static_string") in observed
    assert ("synthetic_beta", "token_static_string") in observed


def test_source_tree_scan_rule_allowlist_is_definition_scoped() -> None:
    contract = _load_governance_contract()
    relative = Path("tests/architecture/test_retired_surface_absent.py")
    rule = next(iter(contract.SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST[relative]))

    assert not contract.source_tree_path_allowed(
        relative,
        scan_group="retired_text",
        rule=rule,
        line_text='"rule-definition-example"',
        line_number=10,
        rule_definition_lines=set(),
    )
    assert contract.source_tree_path_allowed(
        relative,
        scan_group="retired_text",
        rule=rule,
        line_text='"rule-definition-example"',
        line_number=10,
        rule_definition_lines={10},
    )


def test_source_tree_allowlist_entries_have_owner_metadata_and_coverage() -> None:
    contract = _load_governance_contract()

    entries = contract.SOURCE_TREE_ALLOWLIST_ENTRIES

    assert entries
    for entry in entries:
        assert Path(entry.path).is_file(), entry.path
        assert str(entry.owner).strip()
        assert str(entry.rationale).strip()
        assert str(entry.lifecycle).strip()
        assert entry.covered_by
        for covered_by in entry.covered_by:
            test_path, _separator, test_name = covered_by.partition("::")
            assert Path(test_path).is_file(), covered_by
            assert test_path.startswith("tests/")
            if test_name:
                text = Path(test_path).read_text(encoding="utf-8")
                assert f"def {test_name}" in text


def test_source_tree_allowlist_does_not_reference_retired_runtime_privacy_path(
) -> None:
    contract = _load_governance_contract()

    assert not any(
        entry.path == Path("src/quant_research_platform/runtime/privacy.py")
        for entry in contract.SOURCE_TREE_ALLOWLIST_ENTRIES
    )


def test_share_safety_retired_inventory_is_owner_allowlisted() -> None:
    contract = _load_governance_contract()
    share_safety = Path("src/quant_research_platform/policy/share_safety.py")
    entries = {
        (entry.path, entry.rule_name, entry.owner)
        for entry in contract.SOURCE_TREE_ALLOWLIST_ENTRIES
    }

    assert (
        share_safety,
        "forbidden_private_real",
        "share-safety-policy",
    ) in entries
    assert (
        share_safety,
        "forbidden_private_data",
        "share-safety-policy",
    ) in entries


def test_source_tree_scan_rejects_retired_literal_in_ordinary_src_file(
    tmp_path: Path,
) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    bad_source = repo / "src" / "quant_research_platform" / "policy" / "bad.py"
    bad_source.parent.mkdir(parents=True)
    retired_literal = rule_form(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_private_real",
    )
    bad_source.write_text(f"VALUE = {retired_literal!r}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired source literal")

    groups = contract.source_tree_scan_groups(repo)

    assert {
        (finding["path"], finding["rule"])
        for finding in groups["retired_text"]["findings"]
    } == {
        (
            "src/quant_research_platform/policy/bad.py",
            "forbidden_private_real",
        )
    }


def test_source_tree_allowlist_facts_are_structured_not_bare_dicts() -> None:
    source = GOVERNANCE_CONTRACT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GOVERNANCE_CONTRACT))
    bare_fact_sources: list[str] = []
    target_names = {
        "SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST",
        "SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST",
        "SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        for target in targets:
            if target in target_names and isinstance(node.value, ast.Dict | ast.Set):
                bare_fact_sources.append(target)

    assert bare_fact_sources == []


def test_structured_source_tree_allowlist_coverage_points_to_current_tests() -> None:
    contract = _load_governance_contract()
    covered = {
        coverage
        for entry in contract.SOURCE_TREE_ALLOWLIST_ENTRIES
        for coverage in entry.covered_by
    }

    assert any(
        item.startswith("tests/architecture/test_retired_surface_absent.py::")
        for item in covered
    )
    assert any(
        item.startswith("tests/agents/handoff/test_handoff_tools.py::")
        for item in covered
    )


def test_unregistered_retired_negative_test_location_is_not_allowlisted() -> None:
    contract = _load_governance_contract()

    assert not contract.source_tree_path_allowed(
        Path("tests/architecture/test_retired_new_surface.py"),
        scan_group="retired_text",
        rule="forbidden_pr_body",
        line_text=repr(
            text_rule_example(
                contract.RETIRED_PUBLIC_SURFACE_RULES,
                "forbidden_pr_body",
            )
        ),
        line_number=10,
        rule_definition_lines={10},
    )


def test_source_tree_scan_rejects_retired_literal_in_ordinary_test(
    tmp_path: Path,
) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    bad_test = repo / "tests" / "workflows" / "test_accidental_retired_literal.py"
    bad_test.parent.mkdir(parents=True)
    retired_literal = text_rule_example(
        contract.RETIRED_PUBLIC_SURFACE_RULES,
        "forbidden_pr_body",
    )
    bad_test.write_text(f"VALUE = {retired_literal!r}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add accidental retired literal")

    groups = contract.source_tree_scan_groups(repo)

    assert {
        (finding["path"], finding["rule"])
        for finding in groups["retired_text"]["findings"]
    } == {
        (
            "tests/workflows/test_accidental_retired_literal.py",
            "forbidden_pr_body",
        )
    }


def test_owner_approved_negative_fixture_uses_structured_allowlist() -> None:
    contract = _load_governance_contract()
    relative = Path("tests/architecture/test_retired_surface_absent.py")

    entry = next(
        item
        for item in contract.SOURCE_TREE_ALLOWLIST_ENTRIES
        if item.path == relative and item.rule_name == "forbidden_render_pr_body"
    )

    assert entry.owner == "architecture-governance-negative-tests"
    assert contract.source_tree_path_allowed(
        relative,
        scan_group="retired_text",
        rule="forbidden_render_pr_body",
        line_text=repr(
            text_rule_example(
                contract.RETIRED_PUBLIC_SURFACE_RULES,
                "forbidden_render_pr_body",
            )
        ),
        line_number=10,
        rule_definition_lines={10},
    )


def test_source_tree_scan_rejects_retired_governance_path_resurrection(
    tmp_path: Path,
) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    retired_path = contract.RETIRED_GOVERNANCE_EXACT_PATHS[0]
    (repo / retired_path).write_text("retired status\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired governance path")

    groups = contract.source_tree_scan_groups(repo)

    assert groups["retired_governance_paths"]["status"] == "FAIL"
    assert groups["retired_governance_paths"]["findings"] == [
        {"path": retired_path, "rule": "retired_governance_exact_path"}
    ]


def test_source_tree_scan_rejects_retired_stage_path_names(tmp_path: Path) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    retired_component = rule_form(contract.RETIRED_NAME_RULES, "clean_rewrite")
    relative = f"tests/architecture/test_{retired_component}_fixture.py"
    path = repo / relative
    path.parent.mkdir(parents=True)
    path.write_text("def test_placeholder():\n    pass\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired stage path")

    groups = contract.source_tree_scan_groups(repo)

    assert groups["retired_path_names"]["status"] == "FAIL"
    assert groups["retired_path_names"]["findings"] == [
        {
            "path": relative,
            "rule": "clean_rewrite",
        }
    ]


def test_source_tree_scan_rejects_retired_structured_keys(tmp_path: Path) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    path = repo / "docs" / "metadata.json"
    path.parent.mkdir(parents=True)
    retired_key = forms_matching_pattern(
        contract.RETIRED_NAME_RULES,
        "clean_rewrite",
        contract.structured_key_forbidden_patterns()["clean_rewrite"],
    )[0]
    path.write_text(structured_key_document((retired_key,)), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired structured key")

    groups = contract.source_tree_scan_groups(repo)

    assert groups["structured_keys"]["status"] == "FAIL"
    assert groups["structured_keys"]["findings"] == [
        {
            "path": "docs/metadata.json",
            "line": 0,
            "key_path": retired_key,
            "key": retired_key,
            "rule": "clean_rewrite",
        }
    ]


def test_source_tree_scan_independently_rejects_synthesized_strings(
    tmp_path: Path,
) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    path = repo / "tools" / "hack_synthesized.py"
    path.parent.mkdir(parents=True)
    retired_value = rule_form(contract.RETIRED_NAME_RULES, "clean_rewrite")
    path.write_text(
        "\n".join(
            [
                adjacent_string_assignment(retired_value, variable="adjacent"),
                joined_string_assignment(retired_value, variable="joined"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired symbol fixture")

    groups = contract.source_tree_scan_groups(repo)

    assert groups["python_symbols"]["status"] == "PASS"
    assert groups["synthesized_strings"]["status"] == "FAIL"
    assert groups["retired_text"]["status"] == "PASS"
    observed = {
        (finding["path"], finding["rule"], finding["scan_mode"])
        for finding in groups["synthesized_strings"]["findings"]
    }
    assert (
        "tools/hack_synthesized.py",
        "clean_rewrite",
        "token_static_string",
    ) in observed
    assert (
        "tools/hack_synthesized.py",
        "clean_rewrite",
        "ast_join_string",
    ) in observed


def test_source_tree_scan_independently_rejects_python_symbols(
    tmp_path: Path,
) -> None:
    contract = _load_governance_contract()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "governance-contract@example.invalid")
    _git(repo, "config", "user.name", "Governance Contract Test")
    path = repo / "tools" / "hack_symbol.py"
    path.parent.mkdir(parents=True)
    symbol = derived_identifier(
        contract.RETIRED_NAME_RULES,
        "clean_rewrite",
        suffix="_fixture",
    )
    path.write_text(
        "\n".join(
            [
                f"{symbol} = None",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add retired symbol fixture")

    groups = contract.source_tree_scan_groups(repo)

    assert groups["python_symbols"]["status"] == "FAIL"
    assert groups["synthesized_strings"]["status"] == "PASS"
    assert {
        (finding["path"], finding["symbol"], finding["rule"])
        for finding in groups["python_symbols"]["findings"]
    } == {("tools/hack_symbol.py", symbol, "clean_rewrite")}


def test_handoff_source_tree_rules_import_governance_contract() -> None:
    builder_source = (HANDOFF_ROOT / "build_handoff_package.py").read_text(
        encoding="utf-8"
    )
    contract = _load_governance_contract()

    assert (
        tuple(
            path.as_posix() + ("/" if path.is_dir() else "")
            for path in contract.SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS
        )
        == contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    )
    assert "source_tree_scan_payload" in builder_source
    assert "source_tree_forbidden_patterns" not in builder_source
    assert "source_tree_sensitive_negative_patterns" not in builder_source


def test_handoff_builder_does_not_own_sensitive_negative_rules() -> None:
    builder_path = HANDOFF_ROOT / "build_handoff_package.py"
    source = builder_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(builder_path))
    local_owners: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "source_tree_sensitive_negative_patterns":
                local_owners.append(node.name)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith(
                    "SOURCE_TREE_SENSITIVE_NEGATIVE"
                ):
                    local_owners.append(target.id)

    assert local_owners == []
    assert "source_tree_sensitive_negative_patterns" not in source
    assert "source_tree_scan_payload" in source
    assert "from governance_contract import" in source


def test_share_safe_text_checks_use_policy_share_safety_single_source() -> None:
    agent_safety = Path("src/quant_research_platform/agents/safety.py").read_text(
        encoding="utf-8"
    )
    handoff_safety = (HANDOFF_ROOT / "verify_handoff_package.py").read_text(
        encoding="utf-8"
    )
    report_safety = Path(
        "src/quant_research_platform/reports/verification_summary.py"
    ).read_text(encoding="utf-8")

    assert "unsafe_for_share_reason" in agent_safety
    assert "unsafe_for_share_reason" in handoff_safety
    assert "evidence_text_unsafe_reason" in report_safety
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
    sys.modules[spec.name] = module
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
    sys.modules[spec.name] = module
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
