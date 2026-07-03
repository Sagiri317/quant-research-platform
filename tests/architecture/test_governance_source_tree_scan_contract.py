from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path, PurePosixPath

import pytest

from tests.architecture.governance_scan_helpers import (
    GOVERNANCE_NEGATIVE_FIXTURE_FILES,
    RETIRED_NAME_RULES,
)
from tests.helpers.governance_fixtures import source_tree_sensitive_rule_summary
from tests.helpers.governance_git_repos import init_governance_git_repo


def test_source_tree_allowlist_retired_occurrences_are_exact_inventory() -> None:
    from governance_contract import (
        SOURCE_TREE_ALLOWED_OCCURRENCES,
        validate_source_tree_allowed_occurrences,
    )

    assert (
        validate_source_tree_allowed_occurrences(SOURCE_TREE_ALLOWED_OCCURRENCES) == []
    )
    field_names = {field.name for field in fields(SOURCE_TREE_ALLOWED_OCCURRENCES[0])}
    assert field_names == {
        "path",
        "rule_id",
        "exact_line",
        "owner",
        "rationale",
        "lifecycle",
        "covered_by",
        "py_owner",
    }
    assert "allowed_context" not in field_names
    assert "line_regex" not in field_names
    assert "pattern_name" not in field_names
    assert all(
        not hasattr(entry, "scan_group") for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )
    assert all(
        not hasattr(entry, "literal") for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )
    assert all(
        not hasattr(entry, "context_kind") for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )
    assert all(
        not hasattr(entry, "max_occurrences")
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )
    assert all(
        ":" in entry.rule_id and entry.rule_id.split(":", 1)[0]
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )
    assert all(
        entry.owner and entry.rationale and entry.lifecycle
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )
    assert all(entry.covered_by for entry in SOURCE_TREE_ALLOWED_OCCURRENCES)
    assert all(
        entry.py_owner if entry.path.suffix == ".py" else not entry.py_owner
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
    )


def test_source_tree_scan_input_owns_tracked_file_inventory(tmp_path: Path) -> None:
    from governance_contract import build_source_tree_scan_input

    repo = tmp_path / "repo"
    init_governance_git_repo(
        repo,
        {
            "src/example.py": "VALUE = 1\n",
            "data/raw/private.csv": "id,value\n",
        },
    )

    scan_input = build_source_tree_scan_input(repo)

    assert scan_input.clone_dir == repo
    assert Path("src/example.py") in scan_input.tracked_paths
    assert Path("src/example.py") in scan_input.tracked_text_files
    assert Path("data/raw/private.csv") in scan_input.tracked_paths
    assert Path("data/raw/private.csv") not in scan_input.tracked_text_files
    assert scan_input.text_files[Path("src/example.py")].text == "VALUE = 1\n"
    assert scan_input.text_files[Path("src/example.py")].ast_tree is not None


def test_source_tree_scan_groups_keeps_path_high_level_entrypoint(
    tmp_path: Path,
) -> None:
    from governance_contract import source_tree_scan_groups

    repo = tmp_path / "repo"
    init_governance_git_repo(repo, {"src/example.py": "VALUE = 1\n"})

    scan_groups = source_tree_scan_groups(repo)

    assert "retired_text" in scan_groups
    assert "synthesized_strings" in scan_groups


def test_source_tree_allowlist_rejects_invalid_exact_inventory_shape() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        validate_source_tree_allowed_occurrences,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    valid_kwargs = {
        "path": Path("tests/agents/test_verify_governance.py"),
        "rule_id": f"retired_text:{rule.name}",
        "exact_line": f'x = "{rule.example}"',
        "owner": "architecture-governance-negative-tests",
        "rationale": "test-only exact occurrence inventory",
        "lifecycle": "test-only",
        "covered_by": (
            "tests/architecture/test_governance_source_tree_scan_contract.py",
        ),
        "py_owner": "def:test_func",
    }

    cases = [
        ("source_tree_allowed_occurrence_missing_exact_line", {"exact_line": ""}),
        (
            "source_tree_allowed_occurrence_exact_line_rule_mismatch",
            {"exact_line": "x = 'other'"},
        ),
        (
            "source_tree_allowed_occurrence_exact_line_rule_overmatched",
            {"exact_line": f'x = "{rule.example} {rule.example}"'},
        ),
        ("source_tree_allowed_occurrence_missing_py_owner", {"py_owner": ""}),
        (
            "source_tree_allowed_occurrence_non_python_rejects_py_owner",
            {"path": Path("docs/architecture/governance.md")},
        ),
        ("source_tree_allowed_occurrence_missing_owner", {"owner": ""}),
        ("source_tree_allowed_occurrence_missing_rationale", {"rationale": ""}),
        ("source_tree_allowed_occurrence_missing_lifecycle", {"lifecycle": ""}),
        (
            "source_tree_allowed_occurrence_missing_covered_by",
            {"covered_by": ()},
        ),
    ]

    for expected, overrides in cases:
        entry = SourceTreeAllowedOccurrence(**(valid_kwargs | overrides))
        failures = validate_source_tree_allowed_occurrences((entry,))
        assert any(expected in item for item in failures), (expected, failures)

def test_source_tree_allowlist_rejects_invalid_path_and_rule() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        validate_source_tree_allowed_occurrences,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    valid_kwargs = {
        "path": Path("tests/agents/test_verify_governance.py"),
        "rule_id": f"retired_text:{rule.name}",
        "exact_line": f'x = "{rule.example}"',
        "owner": "architecture-governance-negative-tests",
        "rationale": "test-only exact occurrence inventory",
        "lifecycle": "test-only",
        "covered_by": (
            "tests/architecture/test_governance_source_tree_scan_contract.py",
        ),
        "py_owner": "def:test_func",
    }

    cases = [
        (
            "source_tree_allowed_occurrence_absolute_path",
            {"path": Path("/tmp/fixture.py")},
        ),
        ("source_tree_allowed_occurrence_parent_path", {"path": Path("../fixture.py")}),
        (
            "source_tree_allowed_occurrence_windows_backslash_path",
            {"path": "tests\\bad.py"},
        ),
        (
            "source_tree_allowed_occurrence_unknown_rule",
            {"rule_id": "retired_text:unknown_retired_rule"},
        ),
    ]

    for expected, overrides in cases:
        entry = SourceTreeAllowedOccurrence(**(valid_kwargs | overrides))
        failures = validate_source_tree_allowed_occurrences((entry,))
        assert any(expected in item for item in failures), (expected, failures)

def test_source_tree_allowed_occurrence_normalizes_string_path() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]

    entry = SourceTreeAllowedOccurrence(
        path="tests/agents/test_verify_governance.py",
        rule_id=f"retired_text:{rule.name}",
        exact_line=f'x = "{rule.example}"',
        owner="architecture-governance-negative-tests",
        rationale="test-only string path normalization contract",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_source_tree_scan_contract.py",),
        py_owner="def:test_func",
    )

    assert entry.path == PurePosixPath("tests/agents/test_verify_governance.py")

def test_source_tree_allowlist_rejects_retired_terms_namespace() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        validate_source_tree_allowed_occurrences,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    entry = SourceTreeAllowedOccurrence(
        path=Path("tests/agents/test_verify_governance.py"),
        rule_id=f"retired_terms:{rule.name}",
        exact_line=f'x = "{rule.example}"',
        owner="architecture-governance-negative-tests",
        rationale="retired_terms is not a real source-tree scan group",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_source_tree_scan_contract.py",),
        py_owner="def:test_func",
    )

    failures = validate_source_tree_allowed_occurrences((entry,))

    assert any(
        "source_tree_allowed_occurrence_unknown_rule" in item for item in failures
    )

def test_source_tree_allowlist_rejects_old_retired_occurrence_dsl() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    literal = rule.example
    common = {
        "path": Path("tests/agents/test_verify_governance.py"),
        "rule_id": f"retired_text:{rule.name}",
        "exact_line": f'x = "{literal}"',
        "owner": "architecture-governance-negative-tests",
        "rationale": "test-only exact occurrence inventory",
        "lifecycle": "test-only",
        "covered_by": (
            "tests/architecture/test_governance_source_tree_scan_contract.py",
        ),
        "py_owner": "def:test_func",
    }

    for old_field in (
        {"allowed_context": f"line_regex:^{literal}$"},
        {"allowed_context": "python_ast:test_func"},
        {"allowed_context": (f"python_ast:test_func;line_regex:^{literal}$")},
        {"line_regex": f"^{literal}$"},
        {"pattern_name": rule.name},
    ):
        with pytest.raises(TypeError):
            SourceTreeAllowedOccurrence(**(common | old_field))

    assert getattr(SourceTreeAllowedOccurrence, "allowed_context", None) is None
    assert getattr(SourceTreeAllowedOccurrence, "line_regex", None) is None
    assert getattr(SourceTreeAllowedOccurrence, "pattern_name", None) is None

def test_source_tree_path_allowed_does_not_consume_literal_occurrences() -> None:
    from governance_contract import source_tree_path_allowed

    for scan_group in (
        "retired_text",
        "synthesized_strings",
        "active_test_literals",
        "sensitive_negative_literals",
    ):
        assert not source_tree_path_allowed(
            Path("tests/agents/test_verify_governance.py"),
            scan_group=scan_group,
            rule="profile_runner_retired_qrp_managed_repo_roots",
            line_text="x = retired_literal",
            line_number=1,
            rule_definition_lines={1},
            definition_contexts={"def:test_func": {1}},
        )
    with pytest.raises(ValueError):
        source_tree_path_allowed(
            Path("tests/agents/test_verify_governance.py"),
            scan_group="retired_terms",
            rule="profile_runner_retired_qrp_managed_repo_roots",
            line_text='x = "not_a_literal_occurrence"',
            line_number=1,
            rule_definition_lines={1},
        )

def test_source_tree_scan_rejects_unregistered_profile_runner_retired_text(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    repo = tmp_path / "repo"
    retired_text = PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example
    init_governance_git_repo(
        repo,
        {"tests/tools/test_run_profile.py": f"{retired_text}\n"},
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_retired_text_findings(scan_input)

    assert any(
        finding["path"] == "tests/tools/test_run_profile.py"
        and finding["rule"] == PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name
        for finding in findings
    )

def test_governance_negative_fixtures_do_not_hardcode_rule_inventory() -> None:
    from governance_contract import (
        RETIRED_PUBLIC_SURFACE_RULES,
        STALE_GOVERNANCE_TRUTH_RULES,
        retired_name_patterns,
        stale_governance_truth_patterns,
    )

    rule_names = {
        rule.name for rule in (*RETIRED_NAME_RULES, *RETIRED_PUBLIC_SURFACE_RULES)
    } | {rule.name for rule in STALE_GOVERNANCE_TRUTH_RULES}
    patterns = {
        **retired_name_patterns(),
        **stale_governance_truth_patterns(),
    }
    hits: list[str] = []
    for path in GOVERNANCE_NEGATIVE_FIXTURE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if value in rule_names:
                continue
            matched = sorted(
                rule_name
                for rule_name, pattern in patterns.items()
                if pattern.search(value)
            )
            if matched:
                hits.append(f"{path}:{node.lineno}: {','.join(matched)}")

    assert hits == [], "hardcoded_governance_fixture: " + "; ".join(hits)

def test_source_tree_sensitive_negative_rules_are_contract_owned() -> None:
    from governance_contract import (
        SOURCE_TREE_SENSITIVE_NEGATIVE_RULES,
        source_tree_sensitive_negative_patterns,
        source_tree_sensitive_negative_rules,
    )

    rules = source_tree_sensitive_negative_rules()
    patterns = source_tree_sensitive_negative_patterns()
    summary = source_tree_sensitive_rule_summary(rules)

    assert rules == SOURCE_TREE_SENSITIVE_NEGATIVE_RULES
    assert rules
    assert set(patterns) == {rule.name for rule in rules}
    assert {item[0] for item in summary} == set(patterns)
    assert {rule.scan_group for rule in rules} == {"sensitive_negative_literals"}
    assert all(rule.category for rule in rules)
    assert all(rule.definition_sources for rule in rules)
