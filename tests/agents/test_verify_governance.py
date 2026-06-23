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
            "CLEAN_REWRITE",
            "Clean Rewrite",
            "Clean-Rewrite",
            "QRP Clean Rewrite",
            "QrpCleanRewrite",
            "QRPCleanRewrite",
            "qrp-clean-rewrite",
            "qrp_clean_rewrite",
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

    assert "PROJECT_STATUS.md" not in contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    assert all(
        "current-qrp-status.md" not in prefix
        for prefix in contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    )
    assert all(
        "clean-rewrite-0.1" not in prefix
        for prefix in contract.ACTIVE_GOVERNANCE_DOC_ROOTS
    )


def test_path_name_gate_fails_retired_stage_paths(tmp_path: Path) -> None:
    module = _load_governance_module()
    cases = (
        "tests/architecture/test_CLEAN_REWRITE_fake.py",
        "tests/contracts/test_clean_rewrite_fake.py",
        "docs/clean-rewrite-guide.md",
        "docs/reports/clean-rewrite-0.1-test.md",
        ".agents/briefs/current-qrp-status.md",
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
    path = tmp_path / "tests" / "sample_symbols.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "import pytest",
                "class CleanRewriteFixture:",
                "    pass",
                "class QRPCleanRewriteHelper:",
                "    pass",
                "class QrpCleanRewriteHelper:",
                "    pass",
                "class ExecutionMode:",
                "    pass",
                "def test_current_qrp_status():",
                "    pass",
                "def test_CLEAN_REWRITE_fake_symbol():",
                "    pass",
                "@pytest.fixture",
                "def daily_panel():",
                "    return None",
                "def clean_rewrite_fixture():",
                "    return None",
                "clean_rewrite_fixture_value = None",
                "CLEAN_REWRITE_FLAG = False",
                "MODEL_ALGORITHMS = ()",
                "model_family = 'linear'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_python_symbol_names(result, tmp_path, ["tests/sample_symbols.py"])

    expected_rules = (
        "clean_rewrite",
        "current_qrp_status",
        "retired_execution_mode",
        "retired_daily_panel",
        "retired_model_algorithm_collection",
        "retired_model_family",
    )
    for rule in expected_rules:
        assert any(rule in error for error in result.errors), (rule, result.errors)


def test_python_symbol_gate_allows_current_runtime_symbols(tmp_path: Path) -> None:
    module = _load_governance_module()
    cases = {
        "src/quant_research_platform/catalog/profiles.py": (
            "_a_share_csmar_source_id = 'source'\n"
        ),
        "src/quant_research_platform/markets/implementations/a_share/"
        "research_scenarios.py": "model_algorithms = ()\n",
        "src/quant_research_platform/reports/verification_evidence.py": (
            "merge_ready = False\n"
        ),
        "src/quant_research_platform/markets/implementations/a_share/"
        "verification/private_aggregate.py": "merge_ready = False\n",
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
    path = tmp_path / "tests" / "sample_split_strings.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                'stage = "clean" "-rewrite"',
                'display = "QRP " "Clean Rewrite"',
                'flag = "CLEAN" "_REWRITE"',
                'title = "Clean" "-Rewrite"',
                'branch = "feature/" + "qrp-clean-rewrite"',
                'status_doc = "-".join(["current", "qrp", "status"])',
                'register = "-".join(["open", "issues", "register"])',
                'snake = "_".join(["clean", "rewrite"])',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_synthesized_strings(result, tmp_path, ["tests/sample_split_strings.py"])

    expected_rules = (
        "clean_rewrite",
        "old_feature_branch_current_truth",
        "current_qrp_status",
        "open_issues_register",
    )
    for rule in expected_rules:
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

    module.check_synthesized_strings(result, tmp_path, ["tests/sample_version_strings.py"])

    assert result.errors == []


def test_structured_key_gate_fails_retired_keys_and_allows_values(
    tmp_path: Path,
) -> None:
    module = _load_governance_module()
    bad = tmp_path / "config.json"
    bad.write_text(
        (
            "{\n"
            '  "clean_rewrite": true,\n'
            '  "CleanRewrite": true,\n'
            '  "CLEAN_REWRITE_FLAG": true\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    good = tmp_path / "allowed.json"
    good.write_text(
        (
            "{\n"
            '  "retired_name": "clean-rewrite",\n'
            '  "retired_branch": "feature/qrp-clean-rewrite"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    result = module.Result()

    module.check_structured_key_names(result, tmp_path, ["config.json", "allowed.json"])

    assert any("config.json" in error and "clean_rewrite" in error for error in result.errors)
    assert any("config.json" in error and "CleanRewrite" in error for error in result.errors)
    assert any("config.json" in error and "CLEAN_REWRITE_FLAG" in error for error in result.errors)
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
                "QRP 0.1 是当前合同版本。",
                "GET /api/0.1/catalog/factors",
                'contract_version = "0.1"',
                'schema_version = "0.1"',
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
