from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from tests.helpers.governance_fixtures import (
    forms_matching_pattern,
    python_symbol_fixture,
    representative_name_forms,
    rule_form,
    rule_forms,
    structured_key_document,
    structured_value_document,
    synthesized_string_assignment,
    text_rule_example,
)

REPO_ROOT = Path.cwd()
VERIFY_GOVERNANCE = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "verify_governance.py"
)
GOVERNANCE_CONTRACT = VERIFY_GOVERNANCE.with_name("governance_contract.py")


def _load_governance_module():
    spec = importlib.util.spec_from_file_location(
        "verify_governance",
        VERIFY_GOVERNANCE,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "governance_contract",
        GOVERNANCE_CONTRACT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_governance_allows_sanitization_negative_fixtures_only() -> None:
    module = _load_governance_module()
    share_safety_runtime_fixture = Path(
        "tests/policy/test_share_safety_runtime_edges.py"
    )
    privacy_fixture_lines = share_safety_runtime_fixture.read_text(
        encoding="utf-8",
    ).splitlines()
    local_path_line = next(
        line for line in privacy_fixture_lines if "private.csv" in line
    )
    secret_line = next(line for line in privacy_fixture_lines if "token=" in line)

    assert module._is_allowed_local_path_fixture(
        "tests/policy/test_share_safety_runtime_edges.py",
        local_path_line,
    )
    assert module._is_allowed_secret_fixture(
        "tests/policy/test_share_safety_runtime_edges.py",
        secret_line,
    )
    assert not module._is_allowed_local_path_fixture(
        "tests/architecture/test_share_safe_evidence.py",
        '    "/mnt/",',
    )
    assert not module._is_allowed_local_path_fixture(
        "src/quant_research_platform/example.py",
        'path = "outside-run/private-data"',
    )
    assert not module._is_allowed_secret_fixture(
        "src/quant_research_platform/example.py",
        '"credential": "redacted"',
    )


def test_active_doc_scan_collects_non_empty_clean_tree_file_set() -> None:
    module = _load_governance_module()
    contract = _load_contract_module()

    paths = module.active_doc_files(Path("."))

    assert "AGENTS.md" in paths
    assert "README.md" in paths
    assert set(contract.RETIRED_GOVERNANCE_EXACT_PATHS).isdisjoint(paths)
    assert any(path.startswith(".agents/") for path in paths)
    assert (
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/README.md"
        in paths
    )
    assert any(path.startswith("docs/architecture/") for path in paths)
    assert any(path.startswith("docs/code-map/") for path in paths)
    assert any(path.startswith("docs/api/") for path in paths)
    assert any(path.startswith("docs/data/") for path in paths)
    assert "data/README.md" in paths


def test_active_doc_scan_cli_collects_non_empty_clean_tree_file_set() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_GOVERNANCE),
            "--governance-only",
            "--scan-scope",
            "active-docs",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = __import__("json").loads(result.stdout)

    assert payload["status"] == "PASS"
    assert payload["scan_scope"] == "active-docs"
    assert payload["checked_count"] > 0
    assert payload["checked_files"]


def test_governance_only_requires_explicit_scan_scope() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_GOVERNANCE),
            "--governance-only",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--scan-scope" in result.stderr


def test_removed_active_doc_flags_are_not_accepted() -> None:
    for flag in ("--allow-active-docs", "--scan-active-docs"):
        result = subprocess.run(
            [
                sys.executable,
                str(VERIFY_GOVERNANCE),
                "--governance-only",
                "--scan-scope",
                "active-docs",
                flag,
                "--json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr


def test_active_doc_scan_empty_file_set_is_failure() -> None:
    module = _load_governance_module()
    result = module.Result()

    module.check_scan_non_empty(result, [], scan_scope="active-docs")

    assert not result.ok
    assert "checked_files_empty: active-docs" in result.errors


def test_retired_governance_exact_paths_existing_in_worktree_fail(
    tmp_path: Path,
) -> None:
    contract = _load_contract_module()

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
    contract = _load_contract_module()
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


def test_active_doc_scan_fails_stale_current_truth(tmp_path: Path) -> None:
    module = _load_governance_module()
    contract = _load_contract_module()
    stale_cases = (
        *(
            text_rule_example(contract.STALE_GOVERNANCE_TRUTH_RULES, rule.name)
            for rule in contract.STALE_GOVERNANCE_TRUTH_RULES
        ),
        f"{rule_form(contract.RETIRED_NAME_RULES, 'clean_rewrite')} is current.",
        f"{rule_form(contract.RETIRED_NAME_RULES, 'draft_pr')} is current.",
    )

    for index, text in enumerate(stale_cases):
        doc = tmp_path / f"doc-{index}.md"
        doc.write_text(text, encoding="utf-8")
        result = module.Result()

        module.check_text_safety(result, tmp_path, [doc.name])

        assert not result.ok, text


def test_active_doc_scan_fails_retired_public_surface_terms(tmp_path: Path) -> None:
    module = _load_governance_module()
    contract = _load_contract_module()
    cases = representative_name_forms(contract.RETIRED_PUBLIC_SURFACE_RULES)

    for rule, form in cases.items():
        doc = tmp_path / f"{rule}.md"
        doc.write_text(f"{form} is active.", encoding="utf-8")
        result = module.Result()

        module.check_text_safety(result, tmp_path, [doc.name])

        assert any(rule in error for error in result.errors), (rule, result.errors)


def test_active_doc_forbidden_patterns_include_retired_public_surface() -> None:
    contract = _load_contract_module()

    patterns = contract.active_doc_forbidden_patterns()
    cases = representative_name_forms(contract.RETIRED_PUBLIC_SURFACE_RULES)

    for rule, form in cases.items():
        assert patterns[rule].search(form)


def test_retired_naming_rules_cover_required_derived_forms() -> None:
    contract = _load_contract_module()
    selected_rules = (
        "project_status",
        "current_qrp_status",
        "clean_rewrite",
        "retired_execution_mode",
    )
    all_rules = (
        *contract.RETIRED_NAME_RULES,
        *contract.RETIRED_PUBLIC_SURFACE_RULES,
    )

    patterns = contract.retired_name_patterns()

    for rule in selected_rules:
        forms = rule_forms(all_rules, rule)
        assert rule in patterns
        for form in forms:
            assert patterns[rule].search(form), (rule, form)
    assert not patterns["clean_rewrite"].search("clean")


def test_governance_contract_exposes_fixed_rule_helpers() -> None:
    contract = _load_contract_module()

    for name in (
        "RETIRED_NAME_RULES",
        "RETIRED_PUBLIC_SURFACE_RULES",
        "VERSION_SURFACE_POLICY",
        "LEGITIMATE_RUNTIME_SYMBOL_ALLOWLIST",
        "NEGATIVE_TEST_LITERAL_ALLOWLIST",
        "RULE_DEFINITION_ALLOWLIST",
        "PACKAGE_FORBIDDEN_PATH_PARTS",
        "VALID_STATUS_TOKENS",
        "retired_name_patterns",
        "retired_public_surface_patterns",
        "active_doc_forbidden_patterns",
        "source_tree_forbidden_patterns",
        "path_name_forbidden_patterns",
        "python_symbol_forbidden_patterns",
        "structured_key_forbidden_patterns",
        "synthesized_string_forbidden_patterns",
        "is_allowed_rule_definition",
        "is_allowed_negative_test_literal",
        "is_allowed_runtime_symbol",
        "is_allowed_version_surface",
    ):
        assert hasattr(contract, name), name


def test_changed_files_only_and_active_doc_scan_use_distinct_inputs() -> None:
    module = _load_governance_module()
    changed_only = ["README.md"]
    active_docs = module.active_doc_files(Path("."))

    assert changed_only != active_docs
    assert len(active_docs) > len(changed_only)


def test_changed_files_with_base_ref_includes_untracked_files(tmp_path: Path) -> None:
    module = _load_governance_module()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "governance@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Governance Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    new_file = tmp_path / "tests" / "architecture" / "new_gate.py"
    new_file.parent.mkdir(parents=True)
    new_file.write_text("VALUE = 1\n", encoding="utf-8")

    paths = module.changed_files(tmp_path, base)

    assert "tests/architecture/new_gate.py" in paths


def test_retired_status_documents_are_not_required_active_docs() -> None:
    contract = _load_contract_module()

    active_roots = set(contract.ACTIVE_GOVERNANCE_DOC_ROOTS)
    assert set(contract.RETIRED_GOVERNANCE_EXACT_PATHS).isdisjoint(active_roots)
    assert all(
        not any(root.startswith(prefix) for root in active_roots)
        for prefix in contract.RETIRED_GOVERNANCE_PREFIXES
    )


def test_path_name_gate_fails_retired_stage_paths(tmp_path: Path) -> None:
    module = _load_governance_module()
    contract = _load_contract_module()
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


def test_python_symbol_gate_fails_retired_stage_and_surface_names(
    tmp_path: Path,
) -> None:
    module = _load_governance_module()
    contract = _load_contract_module()
    path = tmp_path / "tests" / "sample_symbols.py"
    path.parent.mkdir(parents=True)
    expected_rules = (
        "clean_rewrite",
        "current_qrp_status",
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
    module = _load_governance_module()
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
    module = _load_governance_module()
    contract = _load_contract_module()
    path = tmp_path / "tests" / "sample_split_strings.py"
    path.parent.mkdir(parents=True)
    cases = {
        "clean_rewrite": rule_form(
            contract.RETIRED_NAME_RULES,
            "clean_rewrite",
        ),
        "old_feature_branch_current_truth": text_rule_example(
            contract.STALE_GOVERNANCE_TRUTH_RULES,
            "old_feature_branch_current_truth",
        ),
        "current_qrp_status": rule_form(
            contract.RETIRED_NAME_RULES,
            "current_qrp_status",
        ),
        "open_issues_register": rule_form(
            contract.RETIRED_NAME_RULES,
            "open_issues_register",
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


def test_synthesized_string_gate_allows_current_version_surface(
    tmp_path: Path,
) -> None:
    module = _load_governance_module()
    path = tmp_path / "tests" / "sample_version_strings.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                'label = "QRP" " 0.1"',
                'api = "/api/" "0.1"',
                'version = "0." "1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_synthesized_strings(
        result,
        tmp_path,
        ["tests/sample_version_strings.py"],
    )

    assert result.errors == []


def test_structured_key_gate_fails_retired_keys_and_allows_values(
    tmp_path: Path,
) -> None:
    module = _load_governance_module()
    contract = _load_contract_module()
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
            "config.json" in error and retired_key in error
            for error in result.errors
        )
    assert not any("allowed.json" in error for error in result.errors)


def test_version_surface_policy_keeps_legal_tokens_out_of_retired_gate(
    tmp_path: Path,
) -> None:
    module = _load_governance_module()
    contract = _load_contract_module()
    legal_doc = tmp_path / "legal.md"
    legal_doc.write_text(
        "\n".join(
            [
                "QRP 0.1 is the current contract version.",
                "GET /api/0.1/catalog/factors",
                'definition_version = "0.1"',
                'workflow_version = "0.1"',
                'package = "handoff-sidecar/0.1"',
                'project_version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.Result()
    module.check_text_safety(result, tmp_path, ["legal.md"])

    assert result.errors == []
    assert "/api/0.1" in contract.VERSION_SURFACE_POLICY["allowed"]
    assert "/api/v0.1" in contract.VERSION_SURFACE_POLICY["forbidden"]


def test_version_surface_policy_rejects_legacy_v_api_prefix(tmp_path: Path) -> None:
    module = _load_governance_module()
    path = tmp_path / "docs" / "api.md"
    path.parent.mkdir(parents=True)
    path.write_text("GET /api/v0.1/catalog/factors\n", encoding="utf-8")
    result = module.Result()

    module.check_text_safety(result, tmp_path, ["docs/api.md"])

    assert any("forbidden_versioned_api_prefix" in error for error in result.errors)
