from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.governance_fixtures import (
    representative_name_forms,
    rule_form,
    rule_forms,
    text_rule_example,
)
from tests.helpers.governance_git_repos import (
    init_governance_git_repo,
)
from tests.helpers.governance_text_fixtures import unsafe_assignment_text
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

REPO_ROOT = Path.cwd()
VERIFY_GOVERNANCE = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "verify_governance.py"
)
verify_governance = load_maintenance_script_module("verify_governance")
governance_contract = load_maintenance_script_module("governance_contract")

def test_governance_allows_sanitization_negative_fixtures_only() -> None:
    module = verify_governance
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
    forged_key = "".join(chr(code) for code in (116, 111, 107, 101, 110))
    assert not module._is_allowed_secret_fixture(
        "tests/policy/test_share_safety_runtime_edges.py",
        f'accidental = "{forged_key}=not_allowed"',
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

def test_governance_only_allows_exact_policy_literal_fixture_only() -> None:
    module = verify_governance

    assert module.is_governance_only_allowed_path(
        Path("."),
        "tests/policy/test_share_safety_runtime_edges.py",
    )
    assert not module.is_governance_only_allowed_path(
        Path("."),
        "tests/policy/test_share_safety.py",
    )

def test_active_doc_scan_collects_non_empty_clean_tree_file_set() -> None:
    module = verify_governance
    contract = governance_contract

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

@pytest.mark.parametrize(
    ("relative", "payload_parts", "expected_rule"),
    (
        (
            "src/quant_research_platform/private_payload.py",
            ("VALUE = '", "vendor_", "marker=raw-provider", "'\n"),
            "sensitive_provider_marker",
        ),
        (
            "docs/architecture/private-payload.md",
            ("private ", "vendor_", "marker=raw-provider", "\n"),
            "sensitive_provider_marker",
        ),
        (
            "README.md",
            ("sample ", "600", "000.SH", "\n"),
            "sensitive_stock_code_sample",
        ),
    ),
)
def test_source_tree_sensitive_scan_covers_configured_text_roots(
    tmp_path: Path,
    relative: str,
    payload_parts: tuple[str, ...],
    expected_rule: str,
) -> None:
    contract = governance_contract
    repo = tmp_path / relative.replace("/", "_").replace(".", "_")
    init_governance_git_repo(repo, {relative: "".join(payload_parts)})

    scan_input = contract.build_source_tree_scan_input(repo)
    findings = contract.source_tree_sensitive_negative_literal_findings(scan_input)

    assert any(
        finding["path"] == relative and finding["rule"] == expected_rule
        for finding in findings
    ), findings

def test_forged_retired_profile_env_does_not_skip_repo_file_negative_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = verify_governance
    repo = tmp_path
    forged = repo / "forge-managed-root"
    forged.mkdir(parents=True)
    payload = forged / "private-negative.py"
    payload.write_text(unsafe_assignment_text(), encoding="utf-8")
    monkeypatch.setenv("QRP_MANAGED_REPO_ROOTS", "forge-managed-root")

    result = module.Result()
    paths = ["forge-managed-root/private-negative.py"]

    module.check_text_safety(result, repo, paths)

    assert result.checked_files == ["forge-managed-root/private-negative.py"]
    assert any("possible secret-like assignment" in error for error in result.errors)

def test_unmanaged_repo_file_negative_payload_is_still_detected(tmp_path: Path) -> None:
    module = verify_governance
    path = tmp_path / "docs" / "architecture" / "private-negative.md"
    path.parent.mkdir(parents=True)
    path.write_text(unsafe_assignment_text(), encoding="utf-8")
    result = module.Result()

    module.check_text_safety(
        result,
        tmp_path,
        ["docs/architecture/private-negative.md"],
    )

    assert result.checked_files == ["docs/architecture/private-negative.md"]
    assert any("possible secret-like assignment" in error for error in result.errors)

def test_active_doc_scan_fails_stale_current_truth(tmp_path: Path) -> None:
    module = verify_governance
    contract = governance_contract
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
    module = verify_governance
    contract = governance_contract
    cases = representative_name_forms(contract.RETIRED_PUBLIC_SURFACE_RULES)

    for rule, form in cases.items():
        doc = tmp_path / f"{rule}.md"
        doc.write_text(f"{form} is active.", encoding="utf-8")
        result = module.Result()

        module.check_text_safety(result, tmp_path, [doc.name])

        assert any(rule in error for error in result.errors), (rule, result.errors)

def test_active_doc_forbidden_patterns_include_retired_public_surface() -> None:
    contract = governance_contract

    patterns = contract.active_doc_forbidden_patterns()
    cases = representative_name_forms(contract.RETIRED_PUBLIC_SURFACE_RULES)

    for rule, form in cases.items():
        assert patterns[rule].search(form)

def test_retired_naming_rules_cover_required_derived_forms() -> None:
    contract = governance_contract
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
    contract = governance_contract

    for name in (
        "RETIRED_NAME_RULES",
        "RETIRED_PUBLIC_SURFACE_RULES",
        "VERSION_SURFACE_POLICY",
        "LEGITIMATE_RUNTIME_SYMBOL_ALLOWLIST",
        "STALE_GOVERNANCE_TRUTH_TEST_SOURCES",
        "PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES",
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
        "is_allowed_runtime_symbol",
        "is_allowed_version_surface",
    ):
        assert hasattr(contract, name), name

def test_retired_status_documents_are_not_required_active_docs() -> None:
    contract = governance_contract

    active_roots = set(contract.ACTIVE_GOVERNANCE_DOC_ROOTS)
    assert set(contract.RETIRED_GOVERNANCE_EXACT_PATHS).isdisjoint(active_roots)
    assert all(
        not any(root.startswith(prefix) for root in active_roots)
        for prefix in contract.RETIRED_GOVERNANCE_PREFIXES
    )
