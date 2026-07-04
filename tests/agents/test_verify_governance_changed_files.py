from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.governance_git_repos import (
    init_governance_git_repo,
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

def test_changed_files_only_and_active_doc_scan_use_distinct_inputs() -> None:
    module = verify_governance
    changed_only = ["README.md"]
    active_docs = module.active_doc_files(Path("."))

    assert changed_only != active_docs
    assert len(active_docs) > len(changed_only)

@pytest.mark.integration
def test_changed_files_with_base_ref_includes_untracked_files(tmp_path: Path) -> None:
    module = verify_governance
    base = init_governance_git_repo(tmp_path, {"README.md": "base\n"})
    new_file = tmp_path / "tests" / "architecture" / "new_gate.py"
    new_file.parent.mkdir(parents=True)
    new_file.write_text("VALUE = 1\n", encoding="utf-8")

    paths = module.changed_files(tmp_path, base)

    assert "tests/architecture/new_gate.py" in paths
