from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path, PurePosixPath

import pytest

from tests.architecture.governance_scan_helpers import (
    GOVERNANCE_NEGATIVE_FIXTURE_FILES,
    RETIRED_NAME_RULES,
)
from tests.helpers.governance_fixtures import (
    forms_matching_pattern,
    python_symbol_fixture,
    rule_form,
    source_tree_sensitive_rule_summary,
    structured_key_document,
    structured_value_document,
    synthesized_string_assignment,
    text_rule_example,
)
from tests.helpers.governance_git_repos import init_governance_git_repo
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

verify_governance = load_maintenance_script_module("verify_governance")
governance_contract = load_maintenance_script_module("governance_contract")


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


def test_source_tree_scan_input_owns_pattern_bundles(tmp_path: Path) -> None:
    from governance_contract import build_source_tree_scan_input

    repo = tmp_path / "repo"
    init_governance_git_repo(repo, {"src/example.py": "VALUE = 1\n"})

    scan_input = build_source_tree_scan_input(repo)
    field_names = {field.name for field in fields(scan_input)}

    assert {
        "path_name_patterns",
        "python_symbol_patterns",
        "structured_key_patterns",
        "forbidden_patterns",
        "governance_truth_patterns",
        "sensitive_negative_patterns",
        "active_test_literal_patterns",
        "allowed_occurrence_patterns",
    } <= field_names
    assert scan_input.path_name_patterns
    assert scan_input.python_symbol_patterns
    assert scan_input.structured_key_patterns
    assert scan_input.forbidden_patterns
    assert scan_input.governance_truth_patterns
    assert scan_input.sensitive_negative_patterns
    assert scan_input.active_test_literal_patterns
    assert scan_input.allowed_occurrence_patterns


def test_source_tree_scan_groups_keeps_path_high_level_entrypoint(
    tmp_path: Path,
) -> None:
    from governance_contract import source_tree_scan_groups

    repo = tmp_path / "repo"
    init_governance_git_repo(repo, {"src/example.py": "VALUE = 1\n"})

    scan_groups = source_tree_scan_groups(repo)

    assert "retired_text" in scan_groups
    assert "synthesized_strings" in scan_groups


def test_source_tree_string_scanners_use_source_iter_names() -> None:
    import governance_contract

    assert hasattr(governance_contract, "iter_synthesized_string_findings_from_source")
    assert not hasattr(governance_contract, "source_tree_" + "split_string_findings")
    assert not hasattr(
        governance_contract,
        "source_tree_" + "synthesized_string_findings",
    )


def test_source_tree_allowlist_rejects_invalid_exact_inventory_shape() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        validate_source_tree_allowed_occurrences,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    valid_kwargs = {
        "path": Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
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
        "path": Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
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
        path="tests/architecture/test_governance_source_tree_scan_contract.py",
        rule_id=f"retired_text:{rule.name}",
        exact_line=f'x = "{rule.example}"',
        owner="architecture-governance-negative-tests",
        rationale="test-only string path normalization contract",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_source_tree_scan_contract.py",),
        py_owner="def:test_func",
    )

    assert entry.path == PurePosixPath(
        "tests/architecture/test_governance_source_tree_scan_contract.py"
    )


def test_source_tree_allowlist_rejects_retired_terms_namespace() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        validate_source_tree_allowed_occurrences,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    entry = SourceTreeAllowedOccurrence(
        path=Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
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
        "path": Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
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
            Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
            scan_group=scan_group,
            rule="profile_runner_retired_qrp_managed_repo_roots",
            line_text="x = retired_literal",
            line_number=1,
            rule_definition_lines={1},
            definition_contexts={"def:test_func": {1}},
        )
    with pytest.raises(ValueError):
        source_tree_path_allowed(
            Path("tests/architecture/test_governance_source_tree_scan_contract.py"),
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


def test_retired_governance_exact_paths_existing_in_worktree_fail(
    tmp_path: Path,
) -> None:
    contract = governance_contract

    for retired in contract.RETIRED_GOVERNANCE_EXACT_PATHS:
        repo = tmp_path / retired.replace("/", "_")
        path = repo / retired
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retired path resurrected\n", encoding="utf-8")

        findings = contract.retired_governance_path_findings(repo)

        assert findings
        assert findings[0]["path"] == retired


def test_retired_governance_prefix_existing_in_worktree_fails(
    tmp_path: Path,
) -> None:
    contract = governance_contract
    prefix = contract.RETIRED_GOVERNANCE_PREFIXES[0]
    relative = prefix + "test.md"
    retired = tmp_path / relative
    retired.parent.mkdir(parents=True)
    retired.write_text("retired report resurrected\n", encoding="utf-8")

    findings = contract.retired_governance_path_findings(tmp_path)

    assert findings == [
        {
            "path": relative,
            "rule": "retired_governance_prefix",
        }
    ]


def test_path_name_gate_fails_retired_stage_paths(tmp_path: Path) -> None:
    module = verify_governance
    contract = governance_contract
    retired_component = rule_form(contract.RETIRED_NAME_RULES, "clean_rewrite")
    cases = (
        f"tests/architecture/test_{retired_component}_fixture.py",
        contract.RETIRED_GOVERNANCE_EXACT_PATHS[1],
        contract.RETIRED_GOVERNANCE_PREFIXES[0] + "fixture.md",
    )
    for relative in cases:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("temporary retired naming fixture\n", encoding="utf-8")

    result = module.Result()

    module.check_forbidden_path_names(result, tmp_path)

    for relative in cases:
        assert any(relative in error for error in result.errors), (
            relative,
            result.errors,
        )


def test_source_tree_retired_scan_is_clean_for_current_worktree() -> None:
    contract = governance_contract

    scan_input = contract.build_source_tree_scan_input(Path("."))
    assert contract.source_tree_retired_text_findings(scan_input) == []


def test_broad_retired_occurrence_allowlist_cannot_hide_literal(
    tmp_path: Path,
) -> None:
    contract = governance_contract
    rule = contract.PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    repo = tmp_path / "repo"
    relative_path = "tests/architecture/test_governance_source_tree_scan_contract.py"
    path = repo / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "def test_func():",
                f'    x = "prefix {rule.example} suffix"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError):
        contract.SourceTreeAllowedOccurrence(
            path=Path(
                "tests/architecture/test_governance_source_tree_scan_contract.py"
            ),
            rule_id=f"retired_text:{rule.name}",
            exact_line=f'x = "prefix {rule.example} suffix"',
            allowed_context=(f"python_ast:test_func;line_regex:^.+{rule.example}.+$"),
            owner="governance-negative-tests",
            rationale="old DSL must not be constructible",
            lifecycle="test-only",
            covered_by=(
                "tests/architecture/test_governance_source_tree_scan_contract.py",
            ),
            py_owner="def:test_func",
        )
    init_governance_git_repo(
        repo,
        {relative_path: path.read_text(encoding="utf-8")},
    )

    scan_input = contract.build_source_tree_scan_input(repo)
    findings = contract.source_tree_retired_text_findings(scan_input)

    assert any(
        finding["path"]
        == "tests/architecture/test_governance_source_tree_scan_contract.py"
        and finding["line"] == 2
        and finding["rule"] == rule.name
        for finding in findings
    ), findings


def test_retired_occurrence_duplicate_exact_line_overmatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = governance_contract
    rule = contract.PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    exact_line = f'x = "prefix {rule.example} suffix"'
    repo = tmp_path / "repo"
    path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
    relative_path = "tests/architecture/test_governance_source_tree_scan_contract.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(["def test_func():", f"    {exact_line}", f"    {exact_line}"])
        + "\n",
        encoding="utf-8",
    )
    entry = contract.SourceTreeAllowedOccurrence(
        path=Path(relative_path),
        rule_id=f"retired_text:{rule.name}",
        exact_line=exact_line,
        owner="governance-negative-tests",
        rationale="proves duplicate exact retired occurrence is overmatched",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_source_tree_scan_contract.py",),
        py_owner="def:test_func",
    )
    monkeypatch.setattr(
        contract,
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
        contract.SOURCE_TREE_ALLOWED_OCCURRENCES + (entry,),
    )
    init_governance_git_repo(
        repo,
        {relative_path: path.read_text(encoding="utf-8")},
    )

    scan_input = contract.build_source_tree_scan_input(repo)
    findings = contract.source_tree_retired_text_findings(scan_input)

    assert any(
        finding["path"]
        == "tests/architecture/test_governance_source_tree_scan_contract.py"
        and finding["line"] == 3
        and finding["rule"] == rule.name
        and finding.get("reason") == "source_tree_allowed_occurrence_overmatched"
        for finding in findings
    ), findings


def test_retired_occurrence_stale_exact_line_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = governance_contract
    rule = contract.PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    repo = tmp_path / "repo"
    path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
    relative_path = "tests/architecture/test_governance_source_tree_scan_contract.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(["def test_func():", f'    x = "{rule.example}"']) + "\n",
        encoding="utf-8",
    )
    stale_line = f'y = "{rule.example}"'
    stale = contract.SourceTreeAllowedOccurrence(
        path=Path(relative_path),
        rule_id=f"retired_text:{rule.name}",
        exact_line=stale_line,
        owner="governance-negative-tests",
        rationale="proves stale exact retired occurrence allowlist is reported",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_source_tree_scan_contract.py",),
        py_owner="def:test_func",
    )
    monkeypatch.setattr(
        contract,
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
        contract.SOURCE_TREE_ALLOWED_OCCURRENCES + (stale,),
    )
    init_governance_git_repo(
        repo,
        {relative_path: path.read_text(encoding="utf-8")},
    )

    scan_input = contract.build_source_tree_scan_input(repo)
    findings = contract.source_tree_retired_text_findings(scan_input)

    assert any(
        finding["path"]
        == "tests/architecture/test_governance_source_tree_scan_contract.py"
        and finding["rule"] == rule.name
        and finding.get("reason") == "source_tree_allowed_occurrence_stale"
        for finding in findings
    ), findings


def test_source_tree_retired_text_negative_sources_scan_all_retired_rules(
    tmp_path: Path,
) -> None:
    contract = governance_contract
    repo = tmp_path / "repo"
    relative_path = "tests/architecture/test_governance_source_tree_scan_contract.py"
    path = repo / relative_path
    path.parent.mkdir(parents=True)
    direct_literals = {
        "retired_public_path_symbol": "public_" + "path",
        "retired_private_summary_symbol": "private_" + "summary",
        "clean_" + "rewrite": rule_form(
            contract.RETIRED_NAME_RULES,
            "clean_" + "rewrite",
        ),
    }
    path.write_text(
        "\n".join(f"# {literal}" for literal in direct_literals.values()) + "\n",
        encoding="utf-8",
    )
    init_governance_git_repo(
        repo,
        {relative_path: path.read_text(encoding="utf-8")},
    )

    scan_input = contract.build_source_tree_scan_input(repo)
    findings = contract.source_tree_retired_text_findings(scan_input)

    observed = {(finding["path"], finding["rule"]) for finding in findings}
    assert {(relative_path, rule) for rule in direct_literals} <= observed


def test_verify_governance_synthesized_strings_scan_negative_test_sources(
    tmp_path: Path,
) -> None:
    module = verify_governance
    relative_path = "tests/architecture/test_governance_source_tree_scan_contract.py"
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                '_BAD = "public_" "path"',
                '_BAD2 = "private_" "summary"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_synthesized_strings(
        result,
        tmp_path,
        [relative_path],
    )

    assert any("retired_public_path_symbol" in error for error in result.errors)
    assert any("retired_private_summary_symbol" in error for error in result.errors)


def test_python_symbol_gate_fails_retired_stage_and_surface_names(
    tmp_path: Path,
) -> None:
    module = verify_governance
    contract = governance_contract
    path = tmp_path / "tests" / "sample_symbols.py"
    path.parent.mkdir(parents=True)
    expected_rules = (
        "clean_" + "rewrite",
        "current_" + "qrp_status",
        "retired_execution_mode",
        "retired_daily_panel",
        "retired_model_algorithm_collection",
        "retired_model_family",
    )
    source, _symbols = python_symbol_fixture(
        (*contract.RETIRED_NAME_RULES, *contract.RETIRED_PUBLIC_SURFACE_RULES),
        expected_rules,
    )
    path.write_text(source, encoding="utf-8")
    result = module.Result()

    module.check_python_symbol_names(result, tmp_path, ["tests/sample_symbols.py"])

    for rule in expected_rules:
        assert any(rule in error for error in result.errors), (rule, result.errors)


def test_python_symbol_gate_allows_declared_current_runtime_symbols(
    tmp_path: Path,
) -> None:
    module = verify_governance
    cases = {
        "src/quant_research_platform/catalog/profiles.py": (
            "_a_share_csmar_source_id = 'source'\n"
        ),
    }

    for relative, text in cases.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    result = module.Result()

    module.check_python_symbol_names(result, tmp_path, list(cases))

    assert result.errors == []


def test_synthesized_string_gate_fails_retired_stage_names(tmp_path: Path) -> None:
    module = verify_governance
    contract = governance_contract
    path = tmp_path / "tests" / "sample_split_strings.py"
    path.parent.mkdir(parents=True)
    cases = {
        "clean_" + "rewrite": rule_form(
            contract.RETIRED_NAME_RULES,
            "clean_" + "rewrite",
        ),
        "old_feature_branch_current_truth": text_rule_example(
            contract.STALE_GOVERNANCE_TRUTH_RULES,
            "old_feature_branch_current_truth",
        ),
        "current_" + "qrp_status": rule_form(
            contract.RETIRED_NAME_RULES,
            "current_" + "qrp_status",
        ),
        "open_" + "issues_register": rule_form(
            contract.RETIRED_NAME_RULES,
            "open_" + "issues_register",
        ),
    }
    path.write_text(
        "\n".join(
            synthesized_string_assignment(value, variable=f"value_{index}")
            for index, value in enumerate(cases.values())
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_synthesized_strings(
        result,
        tmp_path,
        ["tests/sample_split_strings.py"],
    )

    for rule in cases:
        assert any(rule in error for error in result.errors), (rule, result.errors)


def test_structured_key_gate_fails_retired_keys_and_allows_values(
    tmp_path: Path,
) -> None:
    module = verify_governance
    contract = governance_contract
    retired_keys = forms_matching_pattern(
        contract.RETIRED_NAME_RULES,
        "clean_rewrite",
        contract.structured_key_forbidden_patterns()["clean_rewrite"],
    )
    bad = tmp_path / "config.json"
    bad.write_text(structured_key_document(retired_keys), encoding="utf-8")
    good = tmp_path / "allowed.json"
    good.write_text(structured_value_document(retired_keys), encoding="utf-8")
    result = module.Result()

    module.check_structured_key_names(result, tmp_path, ["config.json", "allowed.json"])

    for retired_key in retired_keys:
        assert any(
            "config.json" in error and retired_key in error for error in result.errors
        )
    assert not any("allowed.json" in error for error in result.errors)
