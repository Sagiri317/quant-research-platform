from __future__ import annotations

import importlib.util
import subprocess
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
    retired = tmp_path / "docs" / "reports" / "clean-rewrite-0.1-test.md"
    retired.parent.mkdir(parents=True)
    retired.write_text("retired report resurrected\n", encoding="utf-8")

    findings = contract.retired_governance_path_findings(tmp_path)

    assert findings == [
        {
            "path": "docs/reports/clean-rewrite-0.1-test.md",
            "rule": "retired_governance_prefix",
        }
    ]


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


def test_active_doc_scan_fails_retired_public_surface_terms(tmp_path: Path) -> None:
    module = _load_governance_module()
    cases = {
        "retired_execution_mode": "ExecutionMode is the current public mode.",
        "retired_daily_panel": "daily-panel is the current dataset.",
        "retired_model_family": "model_family is the current model selector.",
        "retired_backtest_spec": "BacktestSpec is the active backtest contract.",
        "retired_prediction_bundle": "prediction_bundle is the active artifact.",
        "retired_csmar_source_alias": "CSMAR_SOURCE_ID is the active source alias.",
        "retired_model_algorithm_collection": (
            "MODEL_ALGORITHMS is the active algorithm list."
        ),
    }

    for rule, text in cases.items():
        doc = tmp_path / f"{rule}.md"
        doc.write_text(text, encoding="utf-8")
        result = module.Result()

        module.check_text_safety(result, tmp_path, [doc.name])

        assert any(rule in error for error in result.errors), (rule, result.errors)


def test_active_doc_forbidden_patterns_include_retired_public_surface() -> None:
    contract = _load_contract_module()

    patterns = contract.active_doc_forbidden_patterns()

    assert patterns["retired_execution_mode"].search("ExecutionMode")
    assert patterns["retired_daily_panel"].search("daily-panel")
    assert patterns["retired_model_family"].search("model_family")


def test_retired_naming_rules_cover_required_derived_forms() -> None:
    contract = _load_contract_module()
    examples = {
        "project_status": [
            "PROJECT_STATUS.md",
            "project_status",
            "PROJECT_STATUS",
            "ProjectStatus",
            "project-status",
        ],
        "current_qrp_status": [
            "current-qrp-status",
            "current_qrp_status",
            "CurrentQrpStatus",
            "currentQrpStatus",
        ],
        "clean_rewrite": [
            "clean-rewrite",
            "clean_rewrite",
            "clean rewrite",
            "CleanRewrite",
            "cleanRewrite",
        ],
        "retired_execution_mode": [
            "ExecutionMode",
            "execution_mode",
            "execution-mode",
            "--execution-mode",
        ],
    }

    patterns = contract.retired_name_patterns()

    for rule, forms in examples.items():
        assert rule in patterns
        for form in forms:
            assert patterns[rule].search(form), (rule, form)
    assert not patterns["clean_rewrite"].search("clean")


def test_changed_files_only_and_active_doc_scan_use_distinct_inputs() -> None:
    module = _load_governance_module()
    changed_only = ["README.md"]
    active_docs = module.active_doc_files(Path("."))

    assert changed_only != active_docs
    assert len(active_docs) > len(changed_only)


def test_retired_status_documents_are_not_required_active_docs() -> None:
    contract = _load_contract_module()

    assert "PROJECT_STATUS.md" not in contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    assert all(
        "current-qrp-status.md" not in prefix
        for prefix in contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    )
    assert all(
        "clean-rewrite-0.1" not in prefix
        for prefix in contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    )
