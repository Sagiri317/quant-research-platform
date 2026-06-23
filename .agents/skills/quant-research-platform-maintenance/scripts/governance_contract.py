"""Shared QRP governance contract rules for repository maintenance scripts.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import tomllib
import tokenize
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextRule:
    name: str
    pattern: str
    flags: int = 0

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True)
class NameRule:
    name: str
    forms: tuple[str, ...]


ACTIVE_GOVERNANCE_DOC_ROOTS = (
    "AGENTS.md",
    "README.md",
    ".agents/briefs/",
    ".agents/skills/quant-research-platform-maintenance/SKILL.md",
    ".agents/skills/quant-research-platform-maintenance/references/",
    "docs/architecture/",
    "docs/code-map/",
    "docs/api/",
    "docs/data/",
    "data/README.md",
)

RETIRED_GOVERNANCE_EXACT_PATHS = (
    "PROJECT_STATUS.md",
    ".agents/briefs/current-qrp-status.md",
    "docs/architecture/open-issues-register.md",
)

RETIRED_GOVERNANCE_PREFIXES = (
    "docs/reports/clean-rewrite-0.1-",
)

RETIRED_NAME_RULES = (
    NameRule(
        "project_status",
        (
            "PROJECT_STATUS.md",
            "project_status",
            "PROJECT_STATUS",
            "ProjectStatus",
            "project-status",
        ),
    ),
    NameRule(
        "current_qrp_status",
        (
            "current-qrp-status",
            "current_qrp_status",
            "CurrentQrpStatus",
            "currentQrpStatus",
        ),
    ),
    NameRule(
        "open_issues_register",
        (
            "open-issues-register",
            "open_issues_register",
            "OpenIssuesRegister",
            "openIssuesRegister",
        ),
    ),
    NameRule(
        "clean_rewrite",
        (
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
        ),
    ),
    NameRule(
        "merge_ready",
        (
            "merge-ready",
            "merge_ready",
            "merge ready",
            "MergeReady",
            "mergeReady",
        ),
    ),
    NameRule(
        "pre_merge",
        (
            "pre-merge",
            "pre_merge",
            "pre merge",
            "PreMerge",
            "preMerge",
        ),
    ),
    NameRule(
        "draft_pr",
        (
            "draft PR",
            "draft_pr",
            "draft-pr",
            "DraftPr",
            "draftPr",
        ),
    ),
)

STALE_GOVERNANCE_TRUTH_RULES = (
    TextRule(
        "old_feature_branch_current_truth",
        r"feature/(?:qrp-clean-rewrite|agent-governance-redesign)",
    ),
    TextRule(
        "run_official_tool_source",
        r"(?:\.run\s+official|official\s+\.run)",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_status_source",
        r"(?:PROJECT_STATUS\.md|current-qrp-status\.md)",
    ),
)

VERSION_SURFACE_POLICY = {
    "allowed": (
        "QRP 0.1",
        "0.1",
        "0.1.0",
        "/api/0.1",
        'contract_version="0.1"',
        'schema_version="0.1"',
        "definition_version",
        "workflow_version",
        "handoff-sidecar/0.1",
        "handoff-package/0.1",
    ),
    "forbidden": (
        "/api/v0.1",
    ),
}

VERSION_SURFACE_FORBIDDEN_RULES = (
    TextRule(
        "forbidden_versioned_api_prefix",
        r"(?<![A-Za-z0-9_./-])/api/v0\.1(?![A-Za-z0-9_.-])",
    ),
)

RETIRED_PUBLIC_SURFACE_RULES = (
    NameRule(
        "retired_execution_mode",
        ("ExecutionMode", "execution_mode", "execution-mode", "--execution-mode"),
    ),
    NameRule(
        "retired_daily_panel",
        ("daily_panel", "daily-panel", "DailyPanel"),
    ),
    NameRule(
        "retired_daily_market",
        ("daily_market", "daily-market", "DailyMarket"),
    ),
    NameRule(
        "retired_daily_context",
        ("daily_context", "daily-context", "DailyContext"),
    ),
    NameRule(
        "retired_a_share_daily_bar",
        ("a_share_daily_bar", "a-share-daily-bar", "AShareDailyBar"),
    ),
    NameRule(
        "retired_backtest_spec",
        ("BacktestSpec", "backtest_spec", "backtest-spec"),
    ),
    NameRule(
        "retired_prediction_bundle",
        ("prediction_bundle", "prediction-bundle", "PredictionBundle"),
    ),
    NameRule(
        "retired_model_family",
        ("model_family", "model-family", "ModelFamily"),
    ),
    NameRule(
        "retired_csmar_source_alias",
        ("CSMAR_SOURCE_ID", "csmar_source_id"),
    ),
    NameRule(
        "retired_a_share_market_alias",
        ("A_SHARE_MARKET_ID", "a_share_market_id"),
    ),
    NameRule(
        "retired_model_algorithm_collection",
        ("MODEL_ALGORITHMS", "model_algorithms"),
    ),
    NameRule("retired_a_share_market_definition", ("a_share_market_definition",)),
    NameRule("retired_a_share_field_catalog", ("a_share_field_catalog",)),
    NameRule("retired_a_share_processed_catalog", ("a_share_processed_catalog",)),
    NameRule("retired_a_share_factor_catalog", ("a_share_factor_catalog",)),
    NameRule("retired_a_share_label_catalog", ("a_share_label_catalog",)),
    TextRule("retired_build_a_share", r"\bbuild_a_share_[A-Za-z0-9_]+\b"),
    TextRule("retired_default_a_share", r"\bdefault_a_share_[A-Za-z0-9_]+\b"),
    TextRule("retired_workflow_module", r"\bcontracts\.workflow_run\b"),
    NameRule("retired_a_share_daily_bar_interim", ("a_share_daily_bar_interim",)),
    NameRule("retired_daily_backtest", ("daily_backtest", "daily-backtest")),
    NameRule("retired_daily_rebalance", ("daily_rebalance", "daily-rebalance")),
    NameRule("retired_daily_open", ("daily_open", "daily-open")),
    NameRule("retired_dry_run_flag", ("--dry-run",)),
    NameRule("forbidden_private_real", ("private_real",)),
    NameRule("forbidden_private_data", ("private_data",)),
    NameRule("forbidden_review_evidence", ("review_evidence",)),
    NameRule("forbidden_review_bundle", ("review_bundle",)),
    NameRule("forbidden_review_package", ("review_package",)),
    NameRule("forbidden_review_zip", ("review_zip",)),
    NameRule("forbidden_review_sidecar", ("review_sidecar",)),
    NameRule("forbidden_review_zip_path", ("review_zip_path",)),
    NameRule("forbidden_external_evidence_json_path", ("external_evidence_json_path",)),
    NameRule("forbidden_final_sha", ("final_sha",)),
    NameRule("forbidden_final_head_full", ("final_head_full",)),
    NameRule("forbidden_validation_table", ("validation_table",)),
    NameRule("forbidden_review_threads", ("review_threads",)),
    TextRule("forbidden_review_threads_json", r"\breview_threads\.json\b"),
    NameRule("forbidden_task_review_threads", ("task_review_threads",)),
    NameRule("forbidden_verify_review_thread_closure", ("verify_review_thread_closure",)),
    TextRule("forbidden_diff_patch", r"\bdiff\.patch\b"),
    TextRule("forbidden_pr_body", r"\bpr_body(?:\.md)?\b"),
    NameRule("forbidden_goal_status", ("goal_status",)),
    TextRule(
        "forbidden_render_pr_body",
        r"\brender_pr_body(?:_from_validation_evidence)?(?:\.py)?\b",
    ),
)

RETIRED_PUBLIC_SYMBOL_RULE_NAMES = {
    "retired_execution_mode",
    "retired_daily_panel",
    "retired_daily_market",
    "retired_daily_context",
    "retired_a_share_daily_bar",
    "retired_backtest_spec",
    "retired_prediction_bundle",
    "retired_model_family",
    "retired_csmar_source_alias",
    "retired_a_share_market_alias",
    "retired_model_algorithm_collection",
}

LEGITIMATE_RUNTIME_SYMBOL_ALLOWLIST = {
    Path("src/quant_research_platform/catalog/profiles.py"): {
        "retired_csmar_source_alias": {"_a_share_csmar_source_id"},
    },
    Path(
        "src/quant_research_platform/markets/implementations/a_share/"
        "research_scenarios.py"
    ): {
        "retired_model_algorithm_collection": {"model_algorithms"},
    },
    Path("tests/architecture/test_model_algorithm_single_source.py"): {
        "retired_model_algorithm_collection": {
            "test_model_algorithms_live_only_in_supervised_model_facts",
        },
    },
    Path("tests/research/test_filter_modes.py"): {
        "retired_model_algorithm_collection": {
            "test_required_model_algorithms_fit_and_predict",
        },
    },
    Path("tests/research/test_model_prediction_semantic_names.py"): {
        "retired_model_algorithm_collection": {
            "test_all_model_algorithms_fit_and_predict_synthetic_data",
        },
    },
    Path("src/quant_research_platform/reports/verification_evidence.py"): {
        "merge_ready": {"merge_ready"},
    },
    Path(
        "src/quant_research_platform/markets/implementations/a_share/verification/"
        "private_aggregate.py"
    ): {
        "merge_ready": {"merge_ready"},
    },
}

NEGATIVE_TEST_SYMBOL_ALLOWLIST: dict[Path, dict[str, set[str]]] = {}

NEGATIVE_TEST_LITERAL_ALLOWLIST = {
    "stale_governance_truth": (
        "tests/agents/test_verify_governance.py",
        "tests/architecture/test_governance_contract_rg_gate.py",
    ),
    "retired_name": (
        "tests/agents/test_verify_governance.py",
        "tests/architecture/test_governance_contract_rg_gate.py",
        "tests/architecture/test_handoff_package_contract.py",
    ),
    "retired_path_policy_docs": (
        "AGENTS.md",
        ".agents/skills/quant-research-platform-maintenance/references/verification-and-gates.md",
    ),
}

RULE_DEFINITION_ALLOWLIST = (
    ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py",
)

PACKAGE_FORBIDDEN_PATH_PARTS = (
    "data/raw",
    "data/interim",
    "data/processed",
    "data/research",
    "reports/private_research",
    "reports/private_real",
    "reports/private_data",
    "diff.patch",
    "pr_body.md",
    "review_threads",
    "review_evidence",
    "review_bundle",
    "review_package",
    "review_zip_path",
    "secrets",
    "credentials",
    ".env",
    "auth.json",
    ".codex",
    "vendor raw",
)

VALID_STATUS_TOKENS = ("PASS", "FAIL", "NOT_RUN", "BLOCKED", "TIMEOUT")

RETIRED_PATH_POLICY_MARKERS = tuple(RETIRED_GOVERNANCE_EXACT_PATHS) + (
    "docs/reports/clean-rewrite",
)

PRIVACY_NEGATIVE_TEST_PATHS = (
    "tests/runtime/test_privacy_01.py",
    "tests/agents/handoff/test_handoff_tools.py",
    "tests/agents/handoff/test_package_safety_check.py",
)

STALE_NEGATIVE_TEST_PATHS = (
    "tests/agents/test_verify_governance.py",
    "tests/architecture/test_governance_contract_rg_gate.py",
)

SCANNER_DEFINITION_MARKERS = (
    "SECRET_PATTERN",
    "LOCAL_PATH_PATTERN",
    "STALE_STATUS_PATTERNS",
    "ACTIVE_DOC_FORBIDDEN_PATTERNS",
    "STALE_STATUS_MARKERS",
    "STOCK_SAMPLE_PATTERN",
    "CREDENTIAL_RE",
    "VENDOR_MARKER_RE",
    "RAW_TRACEBACK_RE",
    "PRIVATE_RUNTIME_PAYLOAD_RE",
    "RAW_EXPORT_FILENAME_RE",
    "re.compile(",
    "token = re.compile",
    "secret-like assignment",
    "Finish the official review-fix handoff package",
)

SCANNER_SCRIPT_DEFINITION_MARKERS = (
    "QRP Clean Rewrite",
    "clean rewrite",
    "clean-rewrite",
    "feature/qrp-clean-rewrite",
    "feature/agent-governance-redesign",
    "feature/(?:qrp-clean-rewrite|agent-governance-redesign)",
    "draft PR",
    "pre-merge",
    "merge-ready",
    ".run\\s+official",
    "official\\s+\\.run",
    ".run official",
    "official .run",
    "PROJECT_STATUS\\.md",
    "current-qrp-status\\.md",
    "PROJECT_STATUS.md",
    "PROJECT_STATUS",
    "project_status",
    "ProjectStatus",
    "project-status",
    "current-qrp-status.md",
    "current_qrp_status",
    "CurrentQrpStatus",
    "currentQrpStatus",
    "open-issues-register",
    "open_issues_register",
    "OpenIssuesRegister",
    "openIssuesRegister",
    "docs/reports/clean-rewrite-0.1-",
    "clean_rewrite",
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
    "merge_ready",
    "MergeReady",
    "mergeReady",
    "pre_merge",
    "PreMerge",
    "preMerge",
    "draft_pr",
    "DraftPr",
    "draftPr",
    "ExecutionMode",
    "execution_mode",
    "execution-mode",
    "--execution-mode",
    "daily_panel",
    "daily-panel",
    "DailyPanel",
    "daily_market",
    "daily-market",
    "DailyMarket",
    "daily_context",
    "daily-context",
    "DailyContext",
    "a_share_daily_bar",
    "a-share-daily-bar",
    "AShareDailyBar",
    "BacktestSpec",
    "backtest_spec",
    "backtest-spec",
    "prediction_bundle",
    "prediction-bundle",
    "PredictionBundle",
    "model_family",
    "model-family",
    "ModelFamily",
    "CSMAR_SOURCE_ID",
    "csmar_source_id",
    "A_SHARE_MARKET_ID",
    "a_share_market_id",
    "MODEL_ALGORITHMS",
    "model_algorithms",
    "E:/",
    "D:/",
    "C:/",
    "\\Users\\",
    "\\\\Users\\\\",
    "/Users/",
    "/home/",
    "/mnt/",
    "/mnt/data",
    "vendor_marker",
    "raw_vendor",
    "Traceback (most recent call last):",
    "runtime_payload",
)

CODE_MAP_FORBIDDEN_TERMS = (
    "QRP Clean Rewrite",
    "clean rewrite",
    "clean-rewrite",
    "raw_private",
    "interim_private",
    "processed_private",
    "reports/private_research",
    "/api/v0.1",
    "/api/pipeline-runs",
    "/api/runs",
)

GOVERNANCE_CONTRACT_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/"
    "governance_contract.py"
)
BUILD_HANDOFF_PACKAGE_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "build_handoff_package.py"
)
HANDOFF_SAFETY_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "handoff_safety.py"
)
HANDOFF_SELF_TEST_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "self_test.py"
)
VERIFY_HANDOFF_PACKAGE_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "verify_handoff_package.py"
)
PACKAGE_SAFETY_CHECK_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/"
    "package_safety_check.py"
)
VERIFY_GOVERNANCE_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/"
    "verify_governance.py"
)

SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST = {
    HANDOFF_SAFETY_SCRIPT: {
        "forbidden_diff_patch": {'"diff.patch",'},
        "forbidden_pr_body": {'"pr_body.md",'},
        "forbidden_review_threads": {'"review_threads",'},
        "forbidden_private_real": {'"reports/private_real",'},
        "forbidden_private_data": {'"reports/private_data",'},
        "forbidden_review_evidence": {'"review_evidence",'},
        "forbidden_review_bundle": {'"review_bundle",'},
        "forbidden_review_package": {'"review_package",'},
        "forbidden_review_zip_path": {'"review_zip_path",'},
    },
    VERIFY_HANDOFF_PACKAGE_SCRIPT: {
        "forbidden_final_head_full": {'"final_head_full",'},
    },
    PACKAGE_SAFETY_CHECK_SCRIPT: {
        "forbidden_diff_patch": {'"diff.patch",'},
        "forbidden_pr_body": {'"pr_body.md",'},
        "forbidden_review_threads": {'"review_threads",'},
        "forbidden_private_real": {
            'for path_name in ("private_research", "private_real", "private_data")',
            '"reports/private_real",',
        },
        "forbidden_private_data": {
            'for path_name in ("private_research", "private_real", "private_data")',
            '"reports/private_data",',
        },
        "forbidden_review_evidence": {'"review_evidence",'},
        "forbidden_review_bundle": {'"review_bundle",'},
        "forbidden_review_package": {'"review_package",'},
        "forbidden_review_zip_path": {'"review_zip_path",'},
    },
    Path("src/quant_research_platform/runtime/privacy.py"): {
        "forbidden_private_real": {'"private_real",'},
        "forbidden_private_data": {'"private_data",'},
    },
    Path("tests/architecture/test_retired_surface_absent.py"): {
        "retired_a_share_market_definition": {'"a_share_market_definition",'},
        "retired_a_share_field_catalog": {'"a_share_field_catalog",'},
        "retired_a_share_processed_catalog": {'"a_share_processed_catalog",'},
        "retired_a_share_factor_catalog": {'"a_share_factor_catalog",'},
        "retired_a_share_label_catalog": {'"a_share_label_catalog",'},
        "retired_build_a_share": {
            '"build_a_share_research_dataset_catalog",',
            '"build_a_share_research_dataset_specs",',
            '"build_a_share_model_catalog",',
            '"build_a_share_model_specs",',
            '"build_a_share_prediction_catalog",',
            '"build_a_share_prediction_specs",',
            '"build_a_share_portfolio_catalog",',
            '"build_a_share_portfolio_specs",',
        },
        "retired_default_a_share": {
            '"default_a_share_prediction_smoke_id",',
            '"default_a_share_prediction_oos_period",',
        },
        "retired_workflow_module": {
            'importlib.util.find_spec("quant_research_platform.contracts.workflow_run")',
        },
        "retired_model_algorithm_collection": {
            'assert not hasattr(contracts_models, "MODEL_ALGORITHMS")',
        },
        "retired_csmar_source_alias": {
            'assert not hasattr(source_ids, "CSMAR_SOURCE_ID")',
        },
        "retired_a_share_market_alias": {
            'assert not hasattr(market_ids, "A_SHARE_MARKET_ID")',
        },
        "forbidden_render_pr_body": {
            '".agents/skills/quant-research-platform-maintenance/scripts/render_pr_body_from_validation_evidence.py"',
            '"render_pr_body_from_validation_evidence.py"',
        },
    },
    Path("tests/agents/handoff/test_handoff_tools.py"): {
        "forbidden_private_data": {
            '"reports/private_data/a_share/run/summary.json",',
        },
        "forbidden_private_real": {
            '"reports/private_real/a_share/run/summary.json",',
            '"retired path reports/private_real/a_share/run\\n",',
        },
    },
    Path("tests/agents/handoff/test_package_safety_check.py"): {
        "forbidden_private_data": {
            '"reports/private_data/a_share/run/summary.json",',
        },
        "forbidden_private_real": {
            '"reports/private_real/a_share/run/summary.json",',
        },
    },
}

SOURCE_TREE_ACTIVE_TEST_LITERAL_PATHS = {
    Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"),
}

SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS = tuple(
    Path(root.rstrip("/")) for root in ACTIVE_GOVERNANCE_DOC_ROOTS
)

SOURCE_TREE_RETIRED_TERM_ROOTS = SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS + (
    Path("src/quant_research_platform/interfaces"),
    Path("src/quant_research_platform/agents"),
    Path("src/quant_research_platform/reports"),
    Path(".agents/skills/quant-research-platform-maintenance/scripts/handoff"),
    PACKAGE_SAFETY_CHECK_SCRIPT,
)

SOURCE_TREE_RETIRED_PATH_POLICY_MARKERS = RETIRED_PATH_POLICY_MARKERS

SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST = {
    BUILD_HANDOFF_PACKAGE_SCRIPT: {
        "active_market_bars_data_id",
        "active_market_panel_data_id",
        "active_market_context_data_id",
        "active_frequency_slug",
    },
    Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"): {
        "active_market_bars_data_id",
        "active_market_panel_data_id",
        "active_market_context_data_id",
        "active_frequency_slug",
    },
}

SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST = {
    BUILD_HANDOFF_PACKAGE_SCRIPT: {rule.name for rule in RETIRED_PUBLIC_SURFACE_RULES},
    Path("tests/architecture/test_retired_surface_absent.py"): {
        "retired_csmar_source_alias",
        "retired_a_share_market_alias",
        "retired_model_algorithm_collection",
        "forbidden_render_pr_body",
    },
}

SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES = {
    GOVERNANCE_CONTRACT_SCRIPT: {
        "RETIRED_NAME_RULES",
        "STALE_GOVERNANCE_TRUTH_RULES",
        "VERSION_SURFACE_POLICY",
        "VERSION_SURFACE_FORBIDDEN_RULES",
        "RETIRED_PUBLIC_SURFACE_RULES",
        "RETIRED_PUBLIC_SYMBOL_RULE_NAMES",
        "LEGITIMATE_RUNTIME_SYMBOL_ALLOWLIST",
        "NEGATIVE_TEST_SYMBOL_ALLOWLIST",
        "NEGATIVE_TEST_LITERAL_ALLOWLIST",
        "RULE_DEFINITION_ALLOWLIST",
        "PACKAGE_FORBIDDEN_PATH_PARTS",
        "RETIRED_PATH_POLICY_MARKERS",
        "SCANNER_DEFINITION_MARKERS",
        "SCANNER_SCRIPT_DEFINITION_MARKERS",
        "CODE_MAP_FORBIDDEN_TERMS",
        "SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST",
        "SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST",
        "SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST",
        "SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST",
        "active_doc_forbidden_patterns",
        "path_name_forbidden_patterns",
        "python_symbol_forbidden_patterns",
        "structured_key_forbidden_patterns",
        "synthesized_string_forbidden_patterns",
        "source_tree_forbidden_patterns",
        "stale_fixture_markers",
        "is_allowed_stale_or_retired_fixture",
        "is_allowed_rule_definition",
        "is_allowed_negative_test_literal",
        "is_allowed_runtime_symbol",
        "is_allowed_version_surface",
    },
    BUILD_HANDOFF_PACKAGE_SCRIPT: {
        "RESERVED_HANDOFF_EVIDENCE_STEMS",
        "FORBIDDEN_PACKAGE_ENTRY_PARTS",
        "TASK_CONTROL_KEYS",
        "TASK_METADATA_KEYS",
        "TASK_INPUT_KEYS",
        "FORBIDDEN_TASK_KEYS",
        "FORBIDDEN_VALIDATION_KEYS",
        "FORBIDDEN_HANDOFF_EVIDENCE_KEYS",
        "HANDOFF_EVIDENCE_FORBIDDEN_KEY_COVERAGE_CASES",
        "source_tree_forbidden_patterns",
        "source_tree_active_test_literal_patterns",
        "source_tree_governance_truth_patterns",
        "source_tree_sensitive_negative_patterns",
        "sanitize_handoff_text",
    },
    HANDOFF_SAFETY_SCRIPT: {
        "FORBIDDEN_PATH_SEGMENTS",
        "PRIVATE_RUN_PATH_RE",
        "unsafe_text_reason",
        "is_scanner_definition_line",
    },
    HANDOFF_SELF_TEST_SCRIPT: {"main", "_zip_and_sidecar_with_leak"},
    VERIFY_HANDOFF_PACKAGE_SCRIPT: {
        "contains_local_absolute_path_text",
    },
    PACKAGE_SAFETY_CHECK_SCRIPT: {
        "BLOCKED_PATH_PARTS",
        "_skip_pattern_line",
    },
    VERIFY_GOVERNANCE_SCRIPT: {
        "LOCAL_PATH_PATTERN",
        "SECRET_PATTERN",
        "_skip_regex_definition_line",
        "_is_allowed_local_path_fixture",
        "_is_allowed_secret_fixture",
        "check_text_safety",
        "check_markdown_paths",
    },
}

SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST = {
    GOVERNANCE_CONTRACT_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
        "sensitive_stock_code_sample",
    },
    BUILD_HANDOFF_PACKAGE_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    VERIFY_GOVERNANCE_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    PACKAGE_SAFETY_CHECK_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    HANDOFF_SAFETY_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    VERIFY_HANDOFF_PACKAGE_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    HANDOFF_SELF_TEST_SCRIPT: {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path("tests/agents/handoff/test_handoff_tools.py"): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path("tests/agents/handoff/test_package_safety_check.py"): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path("tests/runtime/test_privacy_01.py"): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
        "sensitive_stock_code_sample",
    },
}


def normalize_repo_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def has_path_prefix(path: str, prefix: str) -> bool:
    path = normalize_repo_path(path)
    prefix = normalize_repo_path(prefix)
    return path == prefix.rstrip("/") or path.startswith(prefix)


def is_retired_governance_path(path: str | Path) -> bool:
    normalized = normalize_repo_path(path)
    return normalized in RETIRED_GOVERNANCE_EXACT_PATHS or any(
        normalized.startswith(prefix) for prefix in RETIRED_GOVERNANCE_PREFIXES
    )


def retired_governance_path_findings(
    repo: Path,
    *,
    tracked_paths: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    found: dict[str, str] = {}
    tracked = tracked_paths if tracked_paths is not None else git_tracked_paths(repo)
    for path in tracked:
        normalized = normalize_repo_path(path)
        if normalized in RETIRED_GOVERNANCE_EXACT_PATHS:
            found[normalized] = "retired_governance_exact_path"
        elif any(
            normalized.startswith(prefix) for prefix in RETIRED_GOVERNANCE_PREFIXES
        ):
            found[normalized] = "retired_governance_prefix"
    for exact in RETIRED_GOVERNANCE_EXACT_PATHS:
        if (repo / exact).exists():
            found[exact] = "retired_governance_exact_path"
    for prefix in RETIRED_GOVERNANCE_PREFIXES:
        parent = repo / Path(prefix).parent
        stem = Path(prefix).name
        if not parent.exists():
            continue
        for path in parent.glob(stem + "*"):
            if path.is_file():
                found[normalize_repo_path(path.relative_to(repo))] = (
                    "retired_governance_prefix"
                )
    return [
        {"path": path, "rule": rule}
        for path, rule in sorted(found.items())
    ]


def git_tracked_paths(repo: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ()
    return tuple(
        normalize_repo_path(line)
        for line in result.stdout.splitlines()
        if line.strip()
    )


def compile_text_rules(rules: tuple[TextRule, ...]) -> dict[str, re.Pattern[str]]:
    return {rule.name: rule.compile() for rule in rules}


def name_rule_patterns(
    rules: tuple[NameRule | TextRule, ...],
) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for rule in rules:
        if isinstance(rule, TextRule):
            patterns[rule.name] = rule.compile()
            continue
        alternatives = "|".join(_bounded_literal_pattern(form) for form in rule.forms)
        patterns[rule.name] = re.compile(alternatives)
    return patterns


def name_rule_patterns_for_surface(
    rules: tuple[NameRule | TextRule, ...],
    *,
    pattern_builder,
    include_text_rules: bool = True,
) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for rule in rules:
        if isinstance(rule, TextRule):
            if include_text_rules:
                patterns[rule.name] = rule.compile()
            continue
        alternatives = "|".join(
            pattern for form in rule.forms if (pattern := pattern_builder(form))
        )
        if alternatives:
            patterns[rule.name] = re.compile(alternatives)
    return patterns


def retired_name_patterns() -> dict[str, re.Pattern[str]]:
    return name_rule_patterns(RETIRED_NAME_RULES + RETIRED_PUBLIC_SURFACE_RULES)


def stale_governance_truth_patterns() -> dict[str, re.Pattern[str]]:
    patterns = compile_text_rules(STALE_GOVERNANCE_TRUTH_RULES)
    patterns.update(name_rule_patterns(RETIRED_NAME_RULES))
    return patterns


def retired_public_surface_patterns() -> dict[str, re.Pattern[str]]:
    return name_rule_patterns(RETIRED_PUBLIC_SURFACE_RULES)


def retired_public_symbol_rules() -> tuple[NameRule | TextRule, ...]:
    return tuple(
        rule
        for rule in RETIRED_PUBLIC_SURFACE_RULES
        if rule.name in RETIRED_PUBLIC_SYMBOL_RULE_NAMES
    )


def version_surface_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    return compile_text_rules(VERSION_SURFACE_FORBIDDEN_RULES)


def active_doc_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    patterns = stale_governance_truth_patterns()
    patterns.update(retired_public_surface_patterns())
    patterns.update(version_surface_forbidden_patterns())
    return patterns


def path_name_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    return name_rule_patterns_for_surface(
        RETIRED_NAME_RULES,
        pattern_builder=_path_literal_pattern,
    )


def python_symbol_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    return name_rule_patterns_for_surface(
        RETIRED_NAME_RULES + retired_public_symbol_rules(),
        pattern_builder=_symbol_literal_pattern,
        include_text_rules=False,
    )


def synthesized_string_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    patterns = stale_governance_truth_patterns()
    patterns.update(version_surface_forbidden_patterns())
    return patterns


def structured_key_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    return python_symbol_forbidden_patterns()


def source_tree_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    patterns = stale_governance_truth_patterns()
    patterns.update(retired_public_surface_patterns())
    patterns.update(version_surface_forbidden_patterns())
    return patterns


def path_name_forbidden_rule_ids() -> tuple[str, ...]:
    return tuple(path_name_forbidden_patterns())


def code_map_forbidden_terms() -> tuple[str, ...]:
    return CODE_MAP_FORBIDDEN_TERMS


def stale_fixture_markers() -> tuple[str, ...]:
    markers: list[str] = [
        "feature/qrp-clean-rewrite",
        "feature/agent-governance-redesign",
        ".run official",
        "official .run",
        *VERSION_SURFACE_POLICY["forbidden"],
    ]
    for rule in RETIRED_NAME_RULES + RETIRED_PUBLIC_SURFACE_RULES:
        if isinstance(rule, TextRule):
            markers.append(rule.pattern)
        else:
            markers.extend(rule.forms)
    return tuple(dict.fromkeys(markers))


def is_retired_path_policy_line(path: str | Path, line: str) -> bool:
    return is_negative_test_allowlisted(path, "retired_path_policy_docs") and any(
        marker in line for marker in RETIRED_PATH_POLICY_MARKERS
    )


def is_allowed_stale_or_retired_fixture(path: str | Path, line: str) -> bool:
    if is_governance_rule_definition_path(path):
        return True
    if is_retired_path_policy_line(path, line):
        return True
    return (
        normalize_repo_path(path) in STALE_NEGATIVE_TEST_PATHS
        or is_negative_test_allowlisted(path, "stale_governance_truth")
        or is_negative_test_allowlisted(path, "retired_name")
    ) and any(marker in line for marker in stale_fixture_markers())


def is_scanner_definition_line(path: str | Path, line: str) -> bool:
    normalized = normalize_repo_path(path)
    if any(marker in line for marker in SCANNER_DEFINITION_MARKERS):
        return True
    scanner_source = normalized.startswith(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
    ) or normalized.startswith("run_tools/") or normalized.endswith(
        (
            "build_handoff_package.py",
            "handoff_safety.py",
            "package_safety_check.py",
            "self_test.py",
            "verify_handoff_package.py",
            "verify_governance.py",
        )
    )
    if scanner_source and any(
        marker in line for marker in SCANNER_SCRIPT_DEFINITION_MARKERS
    ):
        return True
    return normalized.endswith(".py") and line.lstrip().startswith(("r\"", "r'"))


def _path_literal_pattern(form: str) -> str:
    escaped = re.escape(form)
    if form.startswith("--"):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_-])"
    if form.endswith(".md"):
        return rf"(?<![A-Za-z0-9.]){escaped}(?![A-Za-z0-9.])"
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def _symbol_literal_pattern(form: str) -> str:
    if " " in form or "-" in form or "/" in form or "." in form:
        return ""
    return re.escape(form)


def _bounded_literal_pattern(form: str) -> str:
    escaped = re.escape(form)
    if form.startswith("--"):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_-])"
    if form.endswith(".md"):
        return rf"(?<![A-Za-z0-9_./-]){escaped}(?![A-Za-z0-9_./-])"
    return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"


def is_negative_test_allowlisted(path: str | Path, category: str) -> bool:
    normalized = normalize_repo_path(path)
    if category == "governance_rule_definition":
        return normalized in RULE_DEFINITION_ALLOWLIST
    return normalized in NEGATIVE_TEST_LITERAL_ALLOWLIST.get(category, ())


def is_allowed_rule_definition(path: str | Path, rule_id: str | None = None) -> bool:
    del rule_id
    return normalize_repo_path(path) in RULE_DEFINITION_ALLOWLIST


def is_allowed_negative_test_literal(
    path: str | Path,
    rule_id: str | None = None,
) -> bool:
    if rule_id is None:
        return any(
            normalize_repo_path(path) in paths
            for paths in NEGATIVE_TEST_LITERAL_ALLOWLIST.values()
        )
    return any(
        normalize_repo_path(path) in paths
        for category, paths in NEGATIVE_TEST_LITERAL_ALLOWLIST.items()
        if category == rule_id or rule_id.startswith(category)
    )


def is_governance_rule_definition_path(path: str | Path) -> bool:
    return is_allowed_rule_definition(path)


def is_allowed_runtime_symbol(path: str | Path, symbol: str, rule: str) -> bool:
    normalized = Path(normalize_repo_path(path))
    return symbol in LEGITIMATE_RUNTIME_SYMBOL_ALLOWLIST.get(normalized, {}).get(
        rule,
        set(),
    )


def is_allowed_negative_test_symbol(path: str | Path, symbol: str, rule: str) -> bool:
    normalized = Path(normalize_repo_path(path))
    return symbol in NEGATIVE_TEST_SYMBOL_ALLOWLIST.get(normalized, {}).get(
        rule,
        set(),
    )


def is_allowed_version_surface(path: str | Path, version: str) -> bool:
    del path
    return version in VERSION_SURFACE_POLICY["allowed"]


def structured_keys_from_text(
    path: str | Path,
    text: str,
) -> list[dict[str, object]]:
    suffix = Path(normalize_repo_path(path)).suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _structured_keys_from_mapping(payload)
    if suffix == ".toml":
        try:
            payload = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return []
        return _structured_keys_from_mapping(payload)
    if suffix in {".yaml", ".yml"}:
        return _yaml_like_keys(text)
    return []


def _structured_keys_from_mapping(
    payload: object,
    *,
    key_path: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            current = (*key_path, key_text)
            findings.append(
                {
                    "line": 0,
                    "key_path": ".".join(current),
                    "key": key_text,
                }
            )
            findings.extend(_structured_keys_from_mapping(value, key_path=current))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            findings.extend(
                _structured_keys_from_mapping(
                    item,
                    key_path=(*key_path, str(index)),
                )
            )
    return findings


def _yaml_like_keys(text: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    stack: list[tuple[int, str]] = []
    key_re = re.compile(
        r"^(?P<indent>\s*)(?P<quote>['\"]?)"
        r"(?P<key>[A-Za-z0-9_.-]+)(?P=quote)\s*:"
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        match = key_re.match(line)
        if not match:
            continue
        indent = len(match.group("indent").replace("\t", "    "))
        key = match.group("key")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        findings.append(
            {
                "line": line_number,
                "key_path": ".".join(item for _indent, item in stack),
                "key": key,
            }
        )
    return findings


def python_synthesized_strings(source: str) -> list[tuple[int, str, str]]:
    values: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                value = static_string_value(node)
                if value is not None:
                    item = (getattr(node, "lineno", 0), value, "ast_static_string")
                    if item not in seen:
                        seen.add(item)
                        values.append(item)
            if isinstance(node, ast.Call):
                value = join_call_string_value(node)
                if value is not None:
                    item = (getattr(node, "lineno", 0), value, "ast_join_string")
                    if item not in seen:
                        seen.add(item)
                        values.append(item)
    for item in token_synthesized_strings(source):
        if item not in seen:
            seen.add(item)
            values.append(item)
    return values


def token_synthesized_strings(source: str) -> list[tuple[int, str, str]]:
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in {tokenize.NL, tokenize.NEWLINE}
        ]
    except tokenize.TokenError:
        return []
    values: list[tuple[int, str, str]] = []
    index = 0
    while index < len(tokens):
        lexical_item = tokens[index]
        if lexical_item.type != tokenize.STRING:
            index += 1
            continue
        pieces: list[str] = []
        line_number = lexical_item.start[0]
        cursor = index
        expect_string = True
        saw_operator_or_adjacent = False
        while cursor < len(tokens):
            current = tokens[cursor]
            if expect_string:
                if current.type != tokenize.STRING:
                    break
                try:
                    pieces.append(str(ast.literal_eval(current.string)))
                except (SyntaxError, ValueError):
                    break
                expect_string = False
                cursor += 1
                continue
            if current.type == tokenize.OP and current.string == "+":
                saw_operator_or_adjacent = True
                expect_string = True
                cursor += 1
                continue
            if current.type == tokenize.STRING:
                saw_operator_or_adjacent = True
                try:
                    pieces.append(str(ast.literal_eval(current.string)))
                except (SyntaxError, ValueError):
                    break
                cursor += 1
                continue
            break
        if len(pieces) >= 2 and saw_operator_or_adjacent and not expect_string:
            values.append((line_number, "".join(pieces), "token_static_string"))
            index = cursor
            continue
        index += 1
    return values


def static_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = static_string_value(node.left)
        right = static_string_value(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def join_call_string_value(node: ast.Call) -> str | None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "join":
        return None
    if not isinstance(node.func.value, ast.Constant) or not isinstance(
        node.func.value.value,
        str,
    ):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    sequence = node.args[0]
    if not isinstance(sequence, (ast.List, ast.Tuple)):
        return None
    pieces: list[str] = []
    for item in sequence.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        pieces.append(item.value)
    return node.func.value.value.join(pieces)
