from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
VERIFY_GOVERNANCE = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "verify_governance.py"
)


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


def test_governance_allows_sanitization_negative_fixtures_only() -> None:
    module = _load_governance_module()
    privacy_fixture_lines = Path("tests/runtime/test_privacy_01.py").read_text(
        encoding="utf-8"
    ).splitlines()
    local_path_line = next(
        line for line in privacy_fixture_lines if "private.csv" in line
    )
    secret_line = next(line for line in privacy_fixture_lines if "token=" in line)

    assert module._is_allowed_local_path_fixture(
        "tests/runtime/test_privacy_01.py",
        local_path_line,
    )
    assert module._is_allowed_secret_fixture(
        "tests/runtime/test_privacy_01.py",
        secret_line,
    )
    assert not module._is_allowed_local_path_fixture(
        "tests/architecture/test_public_safe_evidence.py",
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

    paths = module.active_doc_files(Path("."))

    assert "AGENTS.md" in paths
    assert "README.md" in paths
    assert "PROJECT_STATUS.md" not in paths
    assert ".agents/briefs/current-qrp-status.md" not in paths
    assert any(path.startswith(".agents/") for path in paths)
    assert any(path.startswith("docs/architecture/") for path in paths)
    assert any(path.startswith("docs/code-map/") for path in paths)
    assert any(path.startswith("docs/api/") for path in paths)
    assert any(path.startswith("docs/data/") for path in paths)
    assert "data/README.md" in paths


def test_active_doc_scan_empty_file_set_is_failure() -> None:
    module = _load_governance_module()
    result = module.Result()

    module.check_active_doc_scan_non_empty(result, [])

    assert not result.ok
    assert "active-doc scan found no files" in result.errors


def test_active_doc_scan_fails_stale_current_truth(tmp_path: Path) -> None:
    module = _load_governance_module()
    stale_cases = (
        "当前状态来自 feature/qrp-clean-rewrite。",
        "clean rewrite 是 current state。",
        "The draft PR is current truth.",
        "." + "run official tool source",
    )

    for index, text in enumerate(stale_cases):
        doc = tmp_path / f"doc-{index}.md"
        doc.write_text(text, encoding="utf-8")
        result = module.Result()

        module.check_text_safety(result, tmp_path, [doc.name])

        assert not result.ok, text


def test_changed_files_only_and_active_doc_scan_use_distinct_inputs() -> None:
    module = _load_governance_module()
    changed_only = ["README.md"]
    active_docs = module.active_doc_files(Path("."))

    assert changed_only != active_docs
    assert len(active_docs) > len(changed_only)


def test_retired_status_documents_are_not_required_active_docs() -> None:
    module = _load_governance_module()

    assert "PROJECT_STATUS.md" not in module.BASE_GOVERNANCE_PREFIXES
    assert all(
        "current-qrp-status.md" not in prefix
        for prefix in module.ACTIVE_DOCS_PREFIXES
    )
    assert all(
        "clean-rewrite-0.1" not in prefix
        for prefix in module.ACTIVE_DOCS_PREFIXES
    )
