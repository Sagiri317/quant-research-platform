from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.governance_git_repos import (
    run_governance_script,
)
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

def _unsafe_assignment_text() -> str:
    key = "".join(chr(code) for code in (116, 111, 107, 101, 110))
    return f'credential = "{key}=unsafe"\n'

@pytest.mark.integration
@pytest.mark.slow
def test_active_doc_scan_cli_collects_non_empty_clean_tree_file_set() -> None:
    result = run_governance_script(
        REPO_ROOT,
        str(VERIFY_GOVERNANCE),
        "--governance-only",
        "--scan-scope",
        "active-docs",
        "--json",
    )
    result.check_returncode()

    payload = __import__("json").loads(result.stdout)

    assert payload["status"] == "PASS"
    assert payload["scan_scope"] == "active-docs"
    assert payload["checked_count"] > 0
    assert payload["checked_files"]

@pytest.mark.integration
def test_governance_only_requires_explicit_scan_scope() -> None:
    result = run_governance_script(
        REPO_ROOT,
        str(VERIFY_GOVERNANCE),
        "--governance-only",
        "--json",
    )

    assert result.returncode != 0
    assert "--scan-scope" in result.stderr

@pytest.mark.integration
def test_removed_active_doc_flags_are_not_accepted() -> None:
    for flag in ("--allow-active-docs", "--scan-active-docs"):
        result = run_governance_script(
            REPO_ROOT,
            str(VERIFY_GOVERNANCE),
            "--governance-only",
            "--scan-scope",
            "active-docs",
            flag,
            "--json",
        )

        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr

def test_active_doc_scan_empty_file_set_is_failure() -> None:
    module = verify_governance
    result = module.Result()

    module.check_scan_non_empty(result, [], scan_scope="active-docs")

    assert not result.ok
    assert "checked_files_empty: active-docs" in result.errors
