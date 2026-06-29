"""Shared QRP governance contract rules for repository maintenance scripts.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    from quant_research_platform.policy.check_status import (
        VALID_STATUS_TOKENS,
        aggregate_validation_matrix_status,
    )
except ModuleNotFoundError:  # pragma: no cover - copied handoff tools may omit src.
    _policy_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "quant_research_platform"
        / "policy"
        / "check_status.py"
    )
    if _policy_path.exists():
        _policy_spec = importlib.util.spec_from_file_location(
            "qrp_policy_check_status",
            _policy_path,
        )
        if _policy_spec is None or _policy_spec.loader is None:
            raise
        _policy_module = importlib.util.module_from_spec(_policy_spec)
        _policy_spec.loader.exec_module(_policy_module)
        VALID_STATUS_TOKENS = _policy_module.VALID_STATUS_TOKENS
        aggregate_validation_matrix_status = (
            _policy_module.aggregate_validation_matrix_status
        )
    else:
        raise


@dataclass(frozen=True)
class TextRule:
    name: str
    pattern: str
    flags: int = 0
    example: str = ""

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True)
class SourceTreeSensitiveNegativeRule:
    name: str
    category: str
    scan_group: str
    pattern: str
    flags: int = 0
    allowed_sources: tuple[Path, ...] = ()

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True)
class SourceTreeAllowlistEntry:
    path: Path
    scan_group: str
    rule_name: str
    literal: str = ""
    pattern_name: str = ""
    owner: str = ""
    rationale: str = ""
    lifecycle: str = ""
    covered_by: tuple[str, ...] = ()


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
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/README.md",
    "docs/architecture/",
    "docs/code-map/",
    "docs/api/",
    "docs/data/",
    "data/README.md",
    ".gitignore",
)

RETIRED_GOVERNANCE_EXACT_PATHS = (
    "PROJECT_STATUS.md",
    ".agents/briefs/current-qrp-status.md",
    "docs/architecture/open-issues-register.md",
)

RETIRED_GOVERNANCE_PREFIXES = ("docs/reports/clean-rewrite-0.1-",)

# Intentional retired-surface negative rules for the run-output architecture.
# These paths and names must not reappear outside explicit governance rules or
# negative tests.
RUN_OUTPUT_ARCHITECTURE_RETIRED_PATHS = (
    "src/quant_research_platform/runtime/privacy.py",
    "src/quant_research_platform/contracts/privacy.py",
    "src/quant_research_platform/contracts/reports.py",
    "src/quant_research_platform/catalog/reports.py",
    "src/quant_research_platform/catalog/profiles.py",
    "src/quant_research_platform/reports/catalog.py",
    "src/quant_research_platform/reports/report_artifacts.py",
    "src/quant_research_platform/reports/execution_metrics_artifacts.py",
    "src/quant_research_platform/research/predictions/artifacts.py",
    "src/quant_research_platform/workflows/verification_summary.py",
    (
        "src/quant_research_platform/markets/implementations/a_share/"
        "verification/private_aggregate.py"
    ),
    (
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "handoff_safety.py"
    ),
    (
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "package_safety_check.py"
    ),
)

RUN_OUTPUT_ARCHITECTURE_RETIRED_NAME_RULES = (
    TextRule("PrivacyLevel", r"\bPrivacyLevel\b"),
    TextRule("ReportSpec", r"\bReportSpec\b"),
    TextRule("ReportDefinition", r"\bReportDefinition\b"),
    TextRule("ReportCatalog", r"\bReportCatalog\b"),
    TextRule("report_id", r"\breport_id\b"),
    TextRule("report_definition_id", r"\breport_definition_id\b"),
    TextRule("private_boundary_required", r"\bprivate_boundary_required\b"),
    TextRule("private_research_report_root", r"\bprivate_research_report_root\b"),
    TextRule("write_report_artifacts", r"\bwrite_[a-zA-Z0-9_]*_report_artifacts\b"),
    TextRule("report_private", r"\b[a-zA-Z0-9_]*_report_private\b"),
    TextRule("prediction_artifacts_private", r"\bprediction_artifacts_private\b"),
    TextRule("report_artifacts", r"\breport_artifacts\b"),
    TextRule("private_aggregate", r"\bprivate_aggregate\b"),
)

RESULT_ARCHITECTURE_RETIRED_TEXT_RULES = (
    TextRule(
        "retired_result_review_safety_tests",
        (
            r"(?<![A-Za-z0-9_])(?:test_public_safe_evidence"
            r"(?:_data_contract)?(?:\.py)?|public_safe_evidence)"
            r"(?![A-Za-z0-9_])"
        ),
    ),
    TextRule("retired_result_run_outputs", r"\brun_outputs\b", re.IGNORECASE),
    TextRule(
        "retired_result_outputs_local_layout",
        r"\boutputs[/\\]local\b",
        re.IGNORECASE,
    ),
    TextRule("retired_result_run_output_type", r"\bRunOutput[A-Za-z0-9_]*\b"),
    TextRule(
        "retired_result_review_evidence_root",
        r"\breview_evidence_root\b",
    ),
    TextRule(
        "retired_result_local_output_root",
        r"(?:\boutput_root\b|--output-root)",
    ),
    TextRule(
        "retired_result_compare_specialization",
        r"\bbaseline[-_]final(?:[-_][A-Za-z0-9_]+)?\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_result_run_level_path",
        (
            r"(?:run_config\.json|run_manifest\.json|validation_summary\.json|"
            r"evidence_summary\.json|timings\.json|timings\.csv|summary\.json|"
            r"data/evidence|layer_manifest\.json|config/run_config\.json|"
            r"manifests/run_manifest\.json|validations/validation_summary\.json|"
            r"timings/timings\.json|\.qrp-local-verification\.json)"
        ),
        re.IGNORECASE,
    ),
    TextRule(
        "retired_result_verification_evidence",
        r"\bverification[ _-]evidence\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_result_verification_evidence_symbol",
        r"\b(?:verification_evidence|build_verification_evidence_payload|"
        r"compare_verification_evidence|evidence_run|evidence_app)\b",
    ),
    TextRule(
        "retired_result_local_evidence_interim_source",
        r"\bLOCAL_EVIDENCE_INTERIM_SOURCE\b",
    ),
    TextRule("retired_result_layer_manifest", r"\blayer_manifest\b"),
    TextRule(
        "retired_result_review_safe_evidence",
        r"\b(?:public_safe_evidence|runtime privacy|report artifact)\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_review_safe_public_wording",
        r"\bpublic[-_ ]safe\b|\bpublic evidence\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_review_safe_public_record",
        r"\bpublic_record\b",
    ),
    TextRule(
        "retired_review_safe_private_public_axis",
        r"\b(?:private/public|private-derived public)\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_review_safe_public_evidence_patterns",
        r"\bFORBIDDEN_PUBLIC_EVIDENCE_PATTERNS\b",
    ),
    TextRule(
        "retired_local_verification_evidence_window",
        r"\b(?:evidence_start|evidence_end|evidence_years)\b",
    ),
    TextRule(
        "retired_local_verification_evidence_range",
        r"\bout_of_evidence_range\b",
    ),
    TextRule(
        "retired_local_verification_evidence_range_surface",
        r"\b(?:outside_evidence_range|evidence_range)\b",
    ),
    TextRule(
        "retired_local_verification_evidence_output",
        r"\b(?:evidence-output|evidence_output)\b",
    ),
    TextRule(
        "retired_local_verification_output_surface",
        r"\b(?:(?<!local-)verification-output|(?<!local_)verification_output)\b",
    ),
    TextRule(
        "retired_result_evidence_root_text",
        r"\bevidence root\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_output_artifact_root_error",
        r"\boutput_must_use_artifact_root\b",
    ),
    TextRule(
        "retired_verification_summary_contract_version_text",
        r"contract version `0\.1`",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_generated_json_schema_version_identity_text",
        r"\bschema_version\s*=\s*\"0\.1\"",
    ),
    TextRule(
        "retired_generated_json_contract_version_identity_text",
        r"\bcontract_version\s*=\s*\"0\.1\"",
    ),
    TextRule(
        "retired_local_evidence_workflow_text",
        r"\blocal evidence workflow\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_api_private_evidence_run_text",
        r"\bqrp-private evidence run\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_evidence_run_text",
        r"\bevidence run\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_evidence_artifacts_text",
        r"\b(?:local )?evidence artifacts\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_private_evidence_text",
        r"\bprivate evidence\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_handoff_safe_evidence_text",
        r"\bhandoff-safe\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_share_safe_evidence_text",
        r"\bshare-safe evidence\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_share_safe_path_text",
        r"\bshare-safe path\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_public_evidence_symbol",
        r"\b(?:public_[A-Za-z0-9_]*evidence|evidence[A-Za-z0-9_]*public|"
        r"allows_public_[A-Za-z0-9_]*evidence)\b",
    ),
    TextRule(
        "retired_public_summary_symbol",
        r"\b(?:public_summary|to_public_summary(?:_payload)?)\b",
    ),
    TextRule(
        "retired_public_payload_symbol",
        r"\b(?:public_payload|public_payloads|guarded_public_output)\b",
    ),
    TextRule(
        "retired_private_summary_symbol",
        r"\bprivate_summary\b",
    ),
    TextRule(
        "retired_public_path_symbol",
        r"\bpublic_path\b",
    ),
    TextRule(
        "retired_public_output_text",
        r"\b(?:public output|test_public_output_writers)\b",
        re.IGNORECASE,
    ),
    TextRule(
        "retired_public_private_axis_text",
        r"\b(?:private\s*/\s*public|public\s*/\s*private)\b",
        re.IGNORECASE,
    ),
)

RESULT_ARCHITECTURE_NEGATIVE_TEST_SOURCES = {
    Path("tests/agents/test_compare_run_results.py"),
    Path("tests/agents/test_verify_governance.py"),
    Path("tests/agents/handoff/test_handoff_package_verifier.py"),
    Path("tests/agents/handoff/test_handoff_tools.py"),
    Path("tests/architecture/test_governance_contract_rg_gate.py"),
    Path("tests/architecture/test_handoff_package_contract.py"),
    Path("tests/interfaces/test_private_cli_01.py"),
    Path("tests/policy/test_share_safety_runtime_edges.py"),
    Path("tests/storage/test_storage_01.py"),
    Path("tests/workflows/test_local_verification.py"),
    Path("tests/workflows/test_local_verification_contract.py"),
}

RUN_OUTPUT_ARCHITECTURE_NEGATIVE_TEST_SOURCES = (
    ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py",
    "tests/architecture/test_no_legacy_names.py",
)

LAYER_BOUNDARY_IMPORT_RULES = (
    {
        "root": "policy",
        "forbidden_prefixes": (
            "quant_research_platform.agents",
            "quant_research_platform.artifacts",
            "quant_research_platform.catalog",
            "quant_research_platform.compute",
            "quant_research_platform.contracts",
            "quant_research_platform.interfaces",
            "quant_research_platform.markets",
            "quant_research_platform.reports",
            "quant_research_platform.research",
            "quant_research_platform.runtime",
            "quant_research_platform.sources",
            "quant_research_platform.storage",
            "quant_research_platform.workflows",
        ),
    },
    {
        "root": "catalog",
        "forbidden_prefixes": (
            "quant_research_platform.sources",
            "quant_research_platform.markets.implementations",
            "quant_research_platform.materializers",
        ),
    },
    {
        "root": "research",
        "forbidden_prefixes": (
            "quant_research_platform.reports",
            "quant_research_platform.artifacts",
        ),
    },
    {
        "root": "reports",
        "forbidden_prefixes": (
            "quant_research_platform.research",
            "quant_research_platform.artifacts",
            "quant_research_platform.markets.implementations",
            "quant_research_platform.storage.atomic",
        ),
    },
    {
        "root": "artifacts",
        "forbidden_prefixes": (
            "quant_research_platform.research",
            "quant_research_platform.reports",
            "quant_research_platform.markets.implementations",
        ),
    },
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
        example="feature/qrp-clean-rewrite",
    ),
    TextRule(
        "run_official_tool_source",
        r"(?:\.run\s+official|official\s+\.run)",
        re.IGNORECASE,
        ".run official",
    ),
    TextRule(
        "retired_status_source",
        r"(?:PROJECT_STATUS\.md|current-qrp-status\.md)",
        example="PROJECT_STATUS.md",
    ),
)

ACTIVE_DOC_HISTORY_NARRATIVE_RULES = (
    TextRule(
        "historical_temporary_tool_narrative",
        r"(?:旧的临时\s+run tools?|old temporary\s+run tools?)",
        re.IGNORECASE,
        "旧的临时 run tools",
    ),
    TextRule(
        "historical_evidence_narrative",
        r"(?:历史(?:阶段)?\s*evidence|historical\s+evidence)",
        re.IGNORECASE,
        "历史 evidence",
    ),
    TextRule(
        "retired_handoff_artifact_narrative",
        r"(?:旧\s*(?:handoff|package|sidecar|交接产物)|old\s+handoff artifacts?)",
        re.IGNORECASE,
        "旧 handoff artifacts",
    ),
    TextRule(
        "historical_rename_narrative",
        r"(?:历史上(?:曾经)?(?:叫|称为)|historically\s+(?:called|named))",
        re.IGNORECASE,
        "历史上叫",
    ),
    TextRule(
        "historical_report_narrative",
        r"(?:历史\s*reports?|旧\s*reports?)",
        re.IGNORECASE,
        "历史 reports",
    ),
)

VERSION_SURFACE_POLICY = {
    "allowed": (
        "QRP 0.1",
        "0.1",
        "0.1.0",
        "/api/0.1",
        "definition_version",
        "workflow_version",
        "qrp-handoff-sidecar",
        "handoff-package/0.1",
    ),
    "forbidden": ("/api/v0.1",),
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
    TextRule(
        "retired_build_a_share",
        r"\bbuild_a_share_[A-Za-z0-9_]+\b",
        example="build_a_share_example",
    ),
    TextRule(
        "retired_default_a_share",
        r"\bdefault_a_share_[A-Za-z0-9_]+\b",
        example="default_a_share_example",
    ),
    TextRule(
        "retired_workflow_module",
        r"\bcontracts\.workflow_run\b",
        example="contracts.workflow_run",
    ),
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
    TextRule(
        "forbidden_review_bundle_text",
        r"\breview bundle\b",
        re.IGNORECASE,
        "review bundle",
    ),
    TextRule(
        "forbidden_review_package_text",
        r"\breview package\b",
        re.IGNORECASE,
        "review package",
    ),
    TextRule(
        "forbidden_review_zip_text",
        r"\breview zip\b",
        re.IGNORECASE,
        "review zip",
    ),
    TextRule(
        "forbidden_review_sidecar_text",
        r"\breview sidecar\b",
        re.IGNORECASE,
        "review sidecar",
    ),
    NameRule("forbidden_review_zip_path", ("review_zip_path",)),
    NameRule("forbidden_external_evidence_json_path", ("external_evidence_json_path",)),
    NameRule("forbidden_final_sha", ("final_sha",)),
    NameRule("forbidden_final_head_full", ("final_head_full",)),
    NameRule("forbidden_validation_table", ("validation_table",)),
    NameRule("forbidden_review_threads", ("review_threads",)),
    TextRule(
        "forbidden_review_threads_json",
        r"\breview_threads\.json\b",
        example="review_threads.json",
    ),
    NameRule("forbidden_task_review_threads", ("task_review_threads",)),
    NameRule(
        "forbidden_verify_review_thread_closure",
        ("verify_review_thread_closure",),
    ),
    TextRule(
        "forbidden_diff_patch",
        r"\bdiff\.patch\b",
        example="diff.patch",
    ),
    TextRule(
        "forbidden_pr_body",
        r"\bpr_body(?:\.md)?\b",
        example="pr_body.md",
    ),
    NameRule("forbidden_goal_status", ("goal_status",)),
    TextRule(
        "forbidden_render_pr_body",
        r"\brender_pr_body(?:_from_validation_evidence)?(?:\.py)?\b",
        example="render_pr_body_from_validation_evidence.py",
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

BLOCKED_DUPLICATE_SOURCE_ROOT_PREFIXES = (
    ".agents/",
    "docs/",
    "run_tools/",
    "schemas/",
    "src/",
    "templates/",
    "tests/",
)

PACKAGE_FORBIDDEN_SUFFIXES = (
    ".arrow",
    ".csv",
    ".db",
    ".feather",
    ".orc",
    ".parquet",
    ".pickle",
    ".pkl",
    ".sqlite",
    ".sqlite3",
)

RETIRED_PATH_POLICY_MARKERS = tuple(RETIRED_GOVERNANCE_EXACT_PATHS) + (
    "docs/reports/clean-rewrite",
)

PRIVACY_NEGATIVE_TEST_PATHS = (
    "tests/policy/test_share_safety.py",
    "tests/policy/test_share_safety_runtime_edges.py",
    "tests/storage/test_manifest_helper_edges.py",
    "tests/agents/handoff/test_handoff_tools.py",
    "tests/agents/handoff/test_handoff_package_verifier.py",
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
    ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py"
)
BUILD_HANDOFF_PACKAGE_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "build_handoff_package.py"
)
HANDOFF_SELF_TEST_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py"
)
VERIFY_HANDOFF_PACKAGE_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "verify_handoff_package.py"
)
VERIFY_GOVERNANCE_SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py"
)

def _source_tree_allowlist_entry(
    *,
    path: Path,
    scan_group: str,
    rule_name: str,
    owner: str,
    rationale: str,
    lifecycle: str,
    covered_by: tuple[str, ...],
    literal: str = "",
    pattern_name: str = "",
) -> SourceTreeAllowlistEntry:
    return SourceTreeAllowlistEntry(
        path=path,
        scan_group=scan_group,
        rule_name=rule_name,
        literal=literal,
        pattern_name=pattern_name or rule_name,
        owner=owner,
        rationale=rationale,
        lifecycle=lifecycle,
        covered_by=covered_by,
    )


SOURCE_TREE_ALLOWLIST_ENTRIES = (
    _source_tree_allowlist_entry(
        path=VERIFY_HANDOFF_PACKAGE_SCRIPT,
        scan_group="retired_occurrence",
        rule_name="forbidden_final_head_full",
        literal='"final_head_full",',
        owner="handoff-package-verifier",
        rationale="verifier rejects the retired validation metadata field",
        lifecycle="current negative contract; remove when field rule is retired",
        covered_by=(
            "tests/agents/handoff/test_handoff_tools.py::"
            "test_verify_rejects_validation_row_extra_fields",
        ),
    ),
    _source_tree_allowlist_entry(
        path=VERIFY_HANDOFF_PACKAGE_SCRIPT,
        scan_group="retired_occurrence",
        rule_name="retired_review_safe_public_wording",
        literal='if summary.get("public_safe") is not True:',
        owner="handoff-package-verifier",
        rationale=(
            "verifier enforces the user-required portable_summary public_safe "
            "boolean contract"
        ),
        lifecycle="remove only if portable_summary public_safe field is renamed",
        covered_by=(
            "tests/agents/handoff/test_handoff_tools.py::"
            "test_verify_smoke_compare_rejects_weak_portable_summary_shape",
        ),
    ),
    _source_tree_allowlist_entry(
        path=Path("src/quant_research_platform/policy/share_safety.py"),
        scan_group="retired_occurrence",
        rule_name="forbidden_private_real",
        literal='"private_real",',
        owner="share-safety-policy",
        rationale=(
            "share-safety policy keeps retired local-only report path names as "
            "private rejection inventory"
        ),
        lifecycle=(
            "remove only if retired local-only path rejection inventory is "
            "removed or renamed"
        ),
        covered_by=(
            "tests/architecture/test_retired_surface_absent.py::"
            "test_retired_local_only_report_paths_remain_share_unsafe",
        ),
    ),
    _source_tree_allowlist_entry(
        path=Path("src/quant_research_platform/policy/share_safety.py"),
        scan_group="retired_occurrence",
        rule_name="forbidden_private_data",
        literal='"private_data",',
        owner="share-safety-policy",
        rationale=(
            "share-safety policy keeps retired local-only report path names as "
            "private rejection inventory"
        ),
        lifecycle=(
            "remove only if retired local-only path rejection inventory is "
            "removed or renamed"
        ),
        covered_by=(
            "tests/architecture/test_retired_surface_absent.py::"
            "test_retired_local_only_report_paths_remain_share_unsafe",
        ),
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_retired_surface_absent.py"),
            scan_group="retired_occurrence",
            rule_name=rule_name,
            literal=literal,
            owner="architecture-governance-negative-tests",
            rationale="owner-approved retired public surface absence assertion",
            lifecycle="current negative test fixture; update with governance rule",
            covered_by=(
                "tests/architecture/test_retired_surface_absent.py::"
                "test_retired_constant_fixtures_are_derived_from_governance_contract",
            ),
        )
        for rule_name, literal in (
            ("retired_a_share_market_definition", '"a_share_market_definition",'),
            ("retired_a_share_field_catalog", '"a_share_field_catalog",'),
            ("retired_a_share_processed_catalog", '"a_share_processed_catalog",'),
            ("retired_a_share_factor_catalog", '"a_share_factor_catalog",'),
            ("retired_a_share_label_catalog", '"a_share_label_catalog",'),
            (
                "retired_workflow_module",
                'importlib.util.find_spec("quant_research_platform.contracts.workflow_run")',
            ),
            (
                "retired_model_algorithm_collection",
                'assert not hasattr(contracts_models, "MODEL_ALGORITHMS")',
            ),
            (
                "retired_csmar_source_alias",
                'assert not hasattr(source_ids, "CSMAR_SOURCE_ID")',
            ),
            (
                "retired_a_share_market_alias",
                'assert not hasattr(market_ids, "A_SHARE_MARKET_ID")',
            ),
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_retired_surface_absent.py"),
            scan_group="retired_occurrence",
            rule_name="retired_build_a_share",
            literal=literal,
            owner="architecture-governance-negative-tests",
            rationale="owner-approved retired A-share builder surface assertion",
            lifecycle="current negative test fixture; update with governance rule",
            covered_by=(
                "tests/architecture/test_retired_surface_absent.py::"
                "test_retired_owner_local_function_names_are_absent",
            ),
        )
        for literal in (
            '"build_a_share_research_dataset_catalog",',
            '"build_a_share_research_dataset_specs",',
            '"build_a_share_model_catalog",',
            '"build_a_share_model_specs",',
            '"build_a_share_prediction_catalog",',
            '"build_a_share_prediction_specs",',
            '"build_a_share_portfolio_catalog",',
            '"build_a_share_portfolio_specs",',
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_retired_surface_absent.py"),
            scan_group="retired_occurrence",
            rule_name="retired_default_a_share",
            literal=literal,
            owner="architecture-governance-negative-tests",
            rationale="owner-approved retired A-share default surface assertion",
            lifecycle="current negative test fixture; update with governance rule",
            covered_by=(
                "tests/architecture/test_retired_surface_absent.py::"
                "test_retired_owner_local_function_names_are_absent",
            ),
        )
        for literal in (
            '"default_a_share_prediction_smoke_id",',
            '"default_a_share_prediction_oos_period",',
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_retired_surface_absent.py"),
            scan_group="retired_occurrence",
            rule_name="forbidden_render_pr_body",
            literal=literal,
            owner="architecture-governance-negative-tests",
            rationale="owner-approved handoff retired renderer absence assertion",
            lifecycle="current negative test fixture; update with governance rule",
            covered_by=(
                "tests/architecture/test_retired_surface_absent.py::"
                "test_retired_pr_body_renderer_is_absent",
            ),
        )
        for literal in (
            '".agents/skills/quant-research-platform-maintenance/scripts/render_pr_body_from_validation_evidence.py"',
            '"render_pr_body_from_validation_evidence.py"',
        )
    ),
    _source_tree_allowlist_entry(
        path=Path("tests/agents/handoff/test_handoff_tools.py"),
        scan_group="retired_occurrence",
        rule_name="forbidden_private_data",
        literal='"reports/private_data/a_share/run/summary.json",',
        owner="handoff-package-safety-tests",
        rationale="malicious handoff path payload must remain rejected",
        lifecycle="current negative test fixture; update with package path rules",
        covered_by=(
            "tests/agents/handoff/test_handoff_tools.py::"
            "test_handoff_package_verifier_rejects_retired_local_output_paths",
        ),
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/agents/handoff/test_handoff_tools.py"),
            scan_group="retired_occurrence",
            rule_name="forbidden_private_real",
            literal=literal,
            owner="handoff-package-safety-tests",
            rationale="malicious handoff path payload must remain rejected",
            lifecycle="current negative test fixture; update with package path rules",
            covered_by=(
                "tests/agents/handoff/test_handoff_tools.py::"
                "test_handoff_package_verifier_rejects_retired_local_output_path_text",
            ),
        )
        for literal in (
            '"reports/private_real/a_share/run/summary.json",',
            '"retired path reports/private_real/a_share/run\\n",',
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=BUILD_HANDOFF_PACKAGE_SCRIPT,
            scan_group="active_test_literals",
            rule_name=rule_name,
            owner="handoff-builder-active-fixtures",
            rationale="builder fixture uses current market data identifiers",
            lifecycle="current active literal; remove if fixture changes",
            covered_by=(
                "tests/architecture/test_handoff_package_contract.py::"
                "test_handoff_source_tree_rules_import_governance_contract",
            ),
        )
        for rule_name in (
            "active_market_bars_data_id",
            "active_market_panel_data_id",
            "active_market_context_data_id",
            "active_frequency_slug",
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"),
            scan_group="active_test_literal_path",
            rule_name=rule_name,
            owner="architecture-active-vocabulary-tests",
            rationale="owner-approved active data-frequency vocabulary assertions",
            lifecycle="current active test fixture; update with frequency contract",
            covered_by=(
                "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
            ),
        )
        for rule_name in (
            "active_market_bars_data_id",
            "active_market_panel_data_id",
            "active_market_context_data_id",
            "active_frequency_slug",
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_governance_contract_rg_gate.py"),
            scan_group="active_test_literal_path",
            rule_name=rule_name,
            owner="architecture-governance-negative-tests",
            rationale="owner-approved retired/current vocabulary gate fixture",
            lifecycle="current governance test fixture; update with rule owner",
            covered_by=("tests/architecture/test_governance_contract_rg_gate.py",),
        )
        for rule_name in (
            "retired_local_verification_evidence_range_surface",
            "retired_local_verification_output_surface",
            "retired_result_evidence_root_text",
        )
    ),
    *(
        _source_tree_allowlist_entry(
            path=GOVERNANCE_CONTRACT_SCRIPT,
            scan_group="retired_rule_definition",
            rule_name=rule.name,
            owner="governance-contract-rule-source",
            rationale="rule names are defined only in the governance contract",
            lifecycle="current rule definition; remove with retired rule",
            covered_by=(
                "tests/architecture/test_handoff_package_contract.py::"
                "test_source_tree_scan_rule_allowlist_is_definition_scoped",
            ),
        )
        for rule in RESULT_ARCHITECTURE_RETIRED_TEXT_RULES
    ),
    *(
        _source_tree_allowlist_entry(
            path=Path("tests/architecture/test_retired_surface_absent.py"),
            scan_group="retired_rule_definition",
            rule_name=rule_name,
            owner="architecture-governance-negative-tests",
            rationale="test derives retired rule names from governance contract",
            lifecycle="current negative test fixture; update with governance rule",
            covered_by=(
                "tests/architecture/test_handoff_package_contract.py::"
                "test_source_tree_scan_rule_allowlist_is_definition_scoped",
            ),
        )
        for rule_name in (
            "retired_csmar_source_alias",
            "retired_a_share_market_alias",
            "retired_model_algorithm_collection",
            "forbidden_render_pr_body",
        )
    ),
)

SOURCE_TREE_ACTIVE_TEST_LITERAL_PATHS = {
    entry.path
    for entry in SOURCE_TREE_ALLOWLIST_ENTRIES
    if entry.scan_group == "active_test_literal_path"
}

SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS = tuple(
    Path(root.rstrip("/")) for root in ACTIVE_GOVERNANCE_DOC_ROOTS
)

SOURCE_TREE_RETIRED_TERM_ROOTS = SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS + (
    Path("src/quant_research_platform/interfaces"),
    Path("src/quant_research_platform/agents"),
    Path("src/quant_research_platform/policy"),
    Path("src/quant_research_platform/reports"),
    Path("tests/workflows"),
    Path(".agents/skills/quant-research-platform-maintenance/scripts/handoff"),
)

SOURCE_TREE_RETIRED_PATH_POLICY_MARKERS = RETIRED_PATH_POLICY_MARKERS

def _source_tree_grouped_rule_allowlist(
    scan_group: str,
) -> dict[Path, set[str]]:
    result: dict[Path, set[str]] = {}
    for entry in SOURCE_TREE_ALLOWLIST_ENTRIES:
        if entry.scan_group != scan_group:
            continue
        result.setdefault(entry.path, set()).add(entry.rule_name)
    return result


def _source_tree_retired_occurrence_allowlist() -> dict[Path, dict[str, set[str]]]:
    result: dict[Path, dict[str, set[str]]] = {}
    for entry in SOURCE_TREE_ALLOWLIST_ENTRIES:
        if entry.scan_group != "retired_occurrence":
            continue
        result.setdefault(entry.path, {}).setdefault(entry.rule_name, set()).add(
            entry.literal
        )
    return result


SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST = (
    _source_tree_retired_occurrence_allowlist()
)
SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST = {
    path: rules
    for path, rules in _source_tree_grouped_rule_allowlist(
        "active_test_literals"
    ).items()
} | {
    path: rules
    for path, rules in _source_tree_grouped_rule_allowlist(
        "active_test_literal_path"
    ).items()
}
SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST = (
    _source_tree_grouped_rule_allowlist("retired_rule_definition")
)

SOURCE_TREE_SENSITIVE_NEGATIVE_RULES = (
    SourceTreeSensitiveNegativeRule(
        name="sensitive_local_only_path",
        category="local_only_path_text",
        scan_group="sensitive_negative_literals",
        pattern=(
            r"(?i)(?:[A-Za-z]:[\\/][^\n]*(?:"
            r"data[\\/](?:raw|interim|processed|research)"
            r"|reports[\\/]private_research"
            r")"
            r"|/(?:Users|home|workspace|workspaces|mnt/data)[^\n]*(?:"
            r"data/(?:raw|interim|processed|research)"
            r"|reports/private_research"
            r")"
            r"|[A-Za-z]:[\\/][^\n]*(?:private|Users)"
            r"|/(?:Users|home)[^\n]*(?:private|data)"
            r"|\\\\[^\n]*(?:server|share|private))"
        ),
        allowed_sources=(
            GOVERNANCE_CONTRACT_SCRIPT,
            VERIFY_GOVERNANCE_SCRIPT,
            VERIFY_HANDOFF_PACKAGE_SCRIPT,
            HANDOFF_SELF_TEST_SCRIPT,
        ),
    ),
    SourceTreeSensitiveNegativeRule(
        name="sensitive_secret",
        category="secret_like_assignment",
        scan_group="sensitive_negative_literals",
        pattern=(
            r"(?i)(?:api[_-]?key|secret|token|password|cookie)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{8,}"
        ),
        allowed_sources=(
            GOVERNANCE_CONTRACT_SCRIPT,
            VERIFY_GOVERNANCE_SCRIPT,
            VERIFY_HANDOFF_PACKAGE_SCRIPT,
            HANDOFF_SELF_TEST_SCRIPT,
        ),
    ),
    SourceTreeSensitiveNegativeRule(
        name="sensitive_provider_marker",
        category="provider_marker_text",
        scan_group="sensitive_negative_literals",
        pattern=(
            r"(?i)\b(?:vendor_marker|vendor_payload|raw_vendor|provider_marker"
            r"|provider_name=raw-provider|source_name=raw-source"
            r"|data_provider=raw-provider|vendor_name=raw-provider)\b"
        ),
        allowed_sources=(
            GOVERNANCE_CONTRACT_SCRIPT,
            VERIFY_GOVERNANCE_SCRIPT,
            VERIFY_HANDOFF_PACKAGE_SCRIPT,
            HANDOFF_SELF_TEST_SCRIPT,
        ),
    ),
    SourceTreeSensitiveNegativeRule(
        name="sensitive_stock_code_sample",
        category="stock_code_sample_text",
        scan_group="sensitive_negative_literals",
        pattern=r"\b[03468]\d{5}\.(?:SZ|SH|BJ)\b",
        allowed_sources=(GOVERNANCE_CONTRACT_SCRIPT,),
    ),
)

SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES = {
    GOVERNANCE_CONTRACT_SCRIPT: {
        "RETIRED_NAME_RULES",
        "RESULT_ARCHITECTURE_RETIRED_TEXT_RULES",
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
        "SOURCE_TREE_ALLOWLIST_ENTRIES",
        "SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST",
        "SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST",
        "SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST",
        "SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST",
        "SOURCE_TREE_SENSITIVE_NEGATIVE_RULES",
        "active_doc_forbidden_patterns",
        "path_name_forbidden_patterns",
        "python_symbol_forbidden_patterns",
        "structured_key_forbidden_patterns",
        "synthesized_string_forbidden_patterns",
        "source_tree_forbidden_patterns",
        "source_tree_sensitive_negative_rules",
        "source_tree_sensitive_negative_patterns",
        "stale_fixture_markers",
        "is_allowed_stale_or_retired_fixture",
        "is_allowed_rule_definition",
        "is_allowed_negative_test_literal",
        "is_allowed_runtime_symbol",
        "is_allowed_version_surface",
    },
    HANDOFF_SELF_TEST_SCRIPT: {"main"},
    VERIFY_HANDOFF_PACKAGE_SCRIPT: {
        "contains_local_absolute_path_text",
        "unsafe_text_reason",
        "is_scanner_definition_line",
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

SOURCE_TREE_ALLOWLIST_ENTRIES += tuple(
    _source_tree_allowlist_entry(
        path=path,
        scan_group="sensitive_negative_literals",
        rule_name=rule_name,
        owner=owner,
        rationale=rationale,
        lifecycle="current safety negative fixture; update with share-safety rules",
        covered_by=covered_by,
    )
    for path, owner, rationale, covered_by, rule_names in (
        (
            GOVERNANCE_CONTRACT_SCRIPT,
            "governance-contract-rule-source",
            "sensitive negative rules are defined only in governance contract",
            (
                "tests/architecture/test_handoff_package_contract.py::"
                "test_source_tree_allowlist_entries_have_owner_metadata_and_coverage",
            ),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
                "sensitive_stock_code_sample",
            ),
        ),
        (
            VERIFY_GOVERNANCE_SCRIPT,
            "governance-verifier",
            "verifier consumes sensitive fixtures from governance contract",
            (
                "tests/agents/test_verify_governance.py::"
                "test_active_doc_scan_fails_retired_public_surface_terms",
            ),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
            ),
        ),
        (
            VERIFY_HANDOFF_PACKAGE_SCRIPT,
            "handoff-package-verifier",
            "handoff verifier consumes sensitive fixtures from share-safety policy",
            (
                "tests/agents/handoff/test_handoff_package_verifier.py::"
                "test_handoff_package_verifier_still_rejects_regular_evidence_paths",
            ),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
            ),
        ),
        (
            HANDOFF_SELF_TEST_SCRIPT,
            "handoff-self-test",
            "self-test carries intentional sensitive negative fixtures",
            (
                "tests/agents/handoff/test_handoff_package_verifier.py::"
                "test_handoff_package_verifier_still_rejects_regular_evidence_paths",
            ),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
            ),
        ),
        (
            Path("tests/agents/handoff/test_handoff_tools.py"),
            "handoff-package-safety-tests",
            "handoff tests carry malicious package payload fixtures",
            (
                "tests/agents/handoff/test_handoff_tools.py::"
                "test_handoff_package_verifier_delegates_text_privacy_semantics",
            ),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
            ),
        ),
        (
            Path("tests/agents/handoff/test_handoff_package_verifier.py"),
            "handoff-package-safety-tests",
            "handoff verifier tests carry malicious package payload fixtures",
            (
                "tests/agents/handoff/test_handoff_package_verifier.py::"
                "test_handoff_package_verifier_still_rejects_regular_evidence_paths",
            ),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
            ),
        ),
        (
            Path("tests/policy/test_share_safety.py"),
            "share-safety-policy-tests",
            "policy test carries intentional local-only path negative fixture",
            ("tests/policy/test_share_safety.py",),
            ("sensitive_local_only_path",),
        ),
        (
            Path("tests/policy/test_share_safety_runtime_edges.py"),
            "share-safety-policy-tests",
            "policy runtime edge tests carry intentional sensitive fixtures",
            ("tests/policy/test_share_safety_runtime_edges.py",),
            (
                "sensitive_local_only_path",
                "sensitive_secret",
                "sensitive_provider_marker",
                "sensitive_stock_code_sample",
            ),
        ),
        (
            Path("tests/storage/test_manifest_helper_edges.py"),
            "storage-manifest-safety-tests",
            "manifest tests carry local-only path and provider marker fixtures",
            ("tests/storage/test_manifest_helper_edges.py",),
            (
                "sensitive_local_only_path",
                "sensitive_provider_marker",
            ),
        ),
    )
    for rule_name in rule_names
)

SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST = (
    _source_tree_grouped_rule_allowlist("sensitive_negative_literals")
)


def normalize_repo_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def package_member_name_failures(name: str) -> tuple[str, ...]:
    """Return package member path policy failures from the governance contract."""

    failures: list[str] = []
    if "\\" in name:
        failures.append("windows_backslash_zip_path")
    if name.startswith("/") or (
        len(name) >= 3
        and name[0].isalpha()
        and name[1] == ":"
        and name[2] in {"/", "\\"}
    ):
        failures.append("absolute_zip_path")
    normalized = name.replace("\\", "/").lower()
    parts = tuple(part for part in PurePosixPath(normalized).parts if part)
    if ".." in parts:
        failures.append("path_traversal_zip_path")
    if PurePosixPath(normalized).suffix in PACKAGE_FORBIDDEN_SUFFIXES:
        failures.append("forbidden_suffix")
    if any(
        _contains_package_path_part(normalized, part)
        for part in PACKAGE_FORBIDDEN_PATH_PARTS
    ):
        failures.append("forbidden_path_segment")
    if len(parts) >= 2 and parts[0] == ".run":
        run_root = parts[1]
        if run_root in {"private", "smoke"} or (
            run_root.startswith("s") and run_root[1:].isdigit()
        ):
            failures.append("forbidden_path_segment")
    return tuple(dict.fromkeys(failures))


def _contains_package_path_part(path: str, part: str) -> bool:
    parts = tuple(item for item in path.split("/") if item)
    needle = tuple(item for item in part.lower().split("/") if item)
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle
        for index in range(len(parts) - len(needle) + 1)
    )


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
    return [{"path": path, "rule": rule} for path, rule in sorted(found.items())]


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
        normalize_repo_path(line) for line in result.stdout.splitlines() if line.strip()
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
    patterns.update(compile_text_rules(ACTIVE_DOC_HISTORY_NARRATIVE_RULES))
    patterns.update(compile_text_rules(RESULT_ARCHITECTURE_RETIRED_TEXT_RULES))
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
    patterns = python_symbol_forbidden_patterns()
    patterns.update(
        {
            "retired_json_schema_version": re.compile(r"^schema_version$"),
            "retired_json_contract_version": re.compile(r"^contract_version$"),
        }
    )
    return patterns


def source_tree_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    patterns = stale_governance_truth_patterns()
    patterns.update(retired_public_surface_patterns())
    patterns.update(version_surface_forbidden_patterns())
    patterns.update(compile_text_rules(RESULT_ARCHITECTURE_RETIRED_TEXT_RULES))
    return patterns


def source_tree_sensitive_negative_rules() -> tuple[
    SourceTreeSensitiveNegativeRule,
    ...,
]:
    return SOURCE_TREE_SENSITIVE_NEGATIVE_RULES


def source_tree_sensitive_negative_patterns() -> dict[str, re.Pattern[str]]:
    return {
        rule.name: rule.compile() for rule in source_tree_sensitive_negative_rules()
    }


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
    scanner_source = (
        normalized.startswith(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
        )
        or normalized.startswith("run_tools/")
        or normalized.endswith(
            (
                "build_handoff_package.py",
                "self_test.py",
                "verify_handoff_package.py",
                "verify_governance.py",
            )
        )
    )
    if scanner_source and any(
        marker in line for marker in SCANNER_SCRIPT_DEFINITION_MARKERS
    ):
        return True
    return normalized.endswith(".py") and line.lstrip().startswith(('r"', "r'"))


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


def iter_path_name_findings(paths: object) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    patterns = path_name_forbidden_patterns()
    for raw_path in paths:
        path = normalize_repo_path(raw_path)
        for rule, pattern in patterns.items():
            if pattern.search(path):
                findings.append({"path": path, "rule": rule})
    return findings


def python_symbols_from_source(source: str) -> list[dict[str, object]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    symbols: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(
                {
                    "line": node.lineno,
                    "symbol": node.name,
                    "symbol_kind": type(node).__name__,
                }
            )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _assignment_target_names(target):
                    symbols.append(
                        {
                            "line": getattr(
                                target,
                                "lineno",
                                getattr(node, "lineno", 0),
                            ),
                            "symbol": name,
                            "symbol_kind": "Assign",
                        }
                    )
        elif isinstance(node, ast.AnnAssign):
            for name in _assignment_target_names(node.target):
                symbols.append(
                    {
                        "line": getattr(
                            node.target,
                            "lineno",
                            getattr(node, "lineno", 0),
                        ),
                        "symbol": name,
                        "symbol_kind": "AnnAssign",
                    }
                )
    return symbols


def iter_python_symbol_findings(
    path: str | Path,
    source: str,
) -> list[dict[str, object]]:
    if is_governance_rule_definition_path(path):
        return []
    findings: list[dict[str, object]] = []
    for item in python_symbols_from_source(source):
        symbol = str(item["symbol"])
        for rule, pattern in python_symbol_forbidden_patterns().items():
            if not pattern.search(symbol):
                continue
            if is_allowed_runtime_symbol(path, symbol, rule):
                continue
            if is_allowed_negative_test_symbol(path, symbol, rule):
                continue
            findings.append(
                {
                    "path": normalize_repo_path(path),
                    "line": int(item["line"]),
                    "symbol_kind": str(item["symbol_kind"]),
                    "symbol": symbol,
                    "rule": rule,
                }
            )
    return findings


def iter_structured_key_findings(
    path: str | Path,
    text: str,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for item in structured_keys_from_text(path, text):
        key = str(item["key"])
        key_path = str(item["key_path"])
        line_number = int(item["line"])
        for rule, pattern in structured_key_forbidden_patterns().items():
            if not pattern.search(key):
                continue
            findings.append(
                {
                    "path": normalize_repo_path(path),
                    "line": line_number,
                    "key_path": key_path,
                    "key": key,
                    "rule": rule,
                }
            )
    if Path(normalize_repo_path(path)).suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        for key_path, value in _schema_id_values(payload):
            if re.search(r"/(?:v)?\d+(?:\.\d+)+$", value):
                findings.append(
                    {
                        "path": normalize_repo_path(path),
                        "line": 0,
                        "key_path": key_path,
                        "key": "schema_id",
                        "rule": "versioned_schema_id",
                    }
                )
    return findings


def _schema_id_values(
    payload: object,
    *,
    key_path: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            current = (*key_path, str(key))
            if key == "schema_id" and isinstance(value, str):
                values.append((".".join(current), value))
            values.extend(_schema_id_values(value, key_path=current))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            values.extend(_schema_id_values(value, key_path=(*key_path, str(index))))
    return values


def iter_synthesized_string_findings(
    path: str | Path,
    source: str,
    *,
    patterns: dict[str, re.Pattern[str]] | None = None,
    allow_negative_tests: bool = True,
    skip_rule_definition: bool = True,
) -> list[dict[str, object]]:
    if skip_rule_definition and is_governance_rule_definition_path(path):
        return []
    if allow_negative_tests and (
        is_negative_test_allowlisted(path, "stale_governance_truth")
        or is_negative_test_allowlisted(path, "retired_name")
    ):
        return []
    active_patterns = patterns or synthesized_string_forbidden_patterns()
    findings: list[dict[str, object]] = []
    for line_number, value, scan_mode in python_synthesized_strings(source):
        for rule, pattern in active_patterns.items():
            if not pattern.search(value):
                continue
            findings.append(
                {
                    "path": normalize_repo_path(path),
                    "line": line_number,
                    "rule": rule,
                    "scan_mode": scan_mode,
                    "synthesized_value": value,
                }
            )
    return findings


def _assignment_target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in target.elts:
            names.extend(_assignment_target_names(item))
        return names
    return []


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


def retired_run_output_architecture_paths() -> tuple[str, ...]:
    return RUN_OUTPUT_ARCHITECTURE_RETIRED_PATHS


def retired_run_output_architecture_name_patterns() -> dict[str, re.Pattern[str]]:
    return {
        rule.name: rule.compile() for rule in RUN_OUTPUT_ARCHITECTURE_RETIRED_NAME_RULES
    }


def is_run_output_architecture_negative_test(path: Path | str) -> bool:
    normalized = Path(path).as_posix()
    return normalized in RUN_OUTPUT_ARCHITECTURE_NEGATIVE_TEST_SOURCES


def layer_boundary_import_rules() -> tuple[dict[str, object], ...]:
    return LAYER_BOUNDARY_IMPORT_RULES


# Handoff package contract and source-tree scan ownership.
VALID_STATUSES = set(VALID_STATUS_TOKENS)
GENERATION_EVIDENCE = "handoff_evidence/generation.json"
SOURCE_TREE_SCAN_EVIDENCE = "handoff_evidence/source-tree-scan.json"
VERIFICATION_MATRIX_EVIDENCE = "handoff_evidence/verification-matrix.json"
BUILDER_OWNED_HANDOFF_EVIDENCE = {
    GENERATION_EVIDENCE,
    SOURCE_TREE_SCAN_EVIDENCE,
    VERIFICATION_MATRIX_EVIDENCE,
}
RESERVED_HANDOFF_EVIDENCE_STEMS = {
    "diff.patch",
    "external_evidence_json_path",
    "final_sha",
    "pr_body",
    "pr_body.md",
    "review_threads",
    "review_threads.json",
    "review_evidence",
    "review_bundle",
    "review_package",
    "review_sidecar",
    "review_zip_path",
    "validation_table",
}
TASK_PROVIDED_HANDOFF_EVIDENCE_RESERVED_STEMS = RESERVED_HANDOFF_EVIDENCE_STEMS | {
    "generation",
    "source-tree-scan",
    "task_metadata",
    "validation_evidence",
    "verification-matrix",
}
BLOCKED_PACKAGE_ENTRY_PARTS = {
    "diff.patch",
    "pr_body.md",
    "review_threads.json",
}
HANDOFF_TOOL_SOURCE_PATHS = [
    ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py",
    ".agents/skills/quant-research-platform-maintenance/templates/handoff/task_metadata.template.json",
    ".agents/skills/quant-research-platform-maintenance/schemas/handoff/task_metadata.schema.json",
    ".agents/skills/quant-research-platform-maintenance/schemas/handoff/sidecar.schema.json",
]
TASK_CONTROL_KEYS = {
    "validation_evidence",
    "handoff_evidence",
    "handoff_markdown_evidence",
}
TASK_METADATA_KEYS = {
    "schema_id",
    "mode",
    "slug",
    "branch",
    "base",
    "final_head",
    "pr_url",
    "task_title",
    "task_id",
    "generated_at_utc",
    "handoff_package_path",
    "handoff_sidecar_path",
    "scope_summary",
    "fixed_blockers",
    "open_issues",
    "not_run",
    "blocked",
    "commit_list",
    "source_tree_scan",
}
TASK_INPUT_KEYS = TASK_METADATA_KEYS | TASK_CONTROL_KEYS
BLOCKED_TASK_KEYS = {
    "diff.patch",
    "external_evidence_json_path",
    "final_head_full",
    "final_sha",
    "goal_status",
    "pr_body",
    "review_bundle",
    "review_evidence",
    "review_package",
    "review_sidecar",
    "review_threads",
    "review_threads_json",
    "review_threads_path",
    "review_zip",
    "review_zip_path",
    "sidecar_logical_path",
    "validation_table",
    "zip_logical_path",
}
TASK_REJECTED_KEYS = BLOCKED_TASK_KEYS
BLOCKED_VALIDATION_KEYS = {
    "external_evidence_json_path",
    "final_head_full",
    "final_sha",
    "goal_status",
    "review_threads",
    "review_zip_path",
    "validation_table",
}
HANDOFF_EVIDENCE_BLOCKED_KEY_COVERAGE_CASES = (
    "final_sha",
    "review_zip_path",
    "external_evidence_json_path",
    "validation_table",
    "check",
    "pr_body_path",
    "review_threads_path",
    "review_threads_json",
    "goal_status",
)
BLOCKED_HANDOFF_EVIDENCE_KEYS = (
    BLOCKED_TASK_KEYS
    | {"check", "pr_body_path", "pr_url"}
    | set(HANDOFF_EVIDENCE_BLOCKED_KEY_COVERAGE_CASES)
)
VALIDATION_ROW_KEYS = {"name", "status", "evidence"}
MANDATORY_VALIDATION_CHECKS = frozenset(
    {
        "tests",
        "smoke",
        "compare",
        "handoff_verify",
        "handoff_safety_scan",
        "governance",
        "code_map",
    }
)
BUILDER_OWNED_VALIDATION_CHECKS = frozenset(
    {"handoff_verify", "handoff_safety_scan"}
)


def source_tree_scan_payload(
    *,
    bundle_path: Path,
    branch: str,
    final_head: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="handoff-source-tree-scan-") as temp:
        clone_dir = Path(temp) / "repo"
        subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle_path), str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return source_tree_scan_from_clone(clone_dir, final_head=final_head)


def source_tree_scan_from_clone(
    clone_dir: Path,
    *,
    final_head: str,
) -> dict[str, object]:
    scanned_head = _governance_git_text(clone_dir, "rev-parse", "HEAD").strip()
    scan_groups = source_tree_scan_groups(clone_dir)
    findings = [
        finding for group in scan_groups.values() for finding in group["findings"]
    ]
    failures = []
    if scanned_head != final_head:
        failures.append("source_tree_scan_head_mismatch")
    failures.extend(finding["rule"] for finding in findings)
    return {
        "schema_id": "qrp-source-tree-scan-result",
        "status": "PASS" if not failures else "FAIL",
        "scanned_head": scanned_head,
        "expected_head": final_head,
        "scanned_tree": "final-branch.bundle clone",
        "forbidden_rule_count": len(source_tree_forbidden_patterns()),
        "finding_count": len(findings),
        "findings": findings[:50],
        "failures": failures[:50],
        "scan_groups": {
            group: {
                **payload,
                "findings": payload["findings"][:50],
            }
            for group, payload in scan_groups.items()
        },
    }


def source_tree_findings(clone_dir: Path) -> list[dict[str, object]]:
    groups = source_tree_scan_groups(clone_dir)
    return [
        *groups["retired_text"]["findings"],
        *groups["python_symbols"]["findings"],
        *groups["synthesized_strings"]["findings"],
    ]


def source_tree_scan_groups(clone_dir: Path) -> dict[str, dict[str, object]]:
    retired_text_findings = source_tree_retired_text_findings(clone_dir)
    retired_governance_path_findings_ = source_tree_retired_governance_path_findings(
        clone_dir
    )
    retired_path_name_findings = source_tree_retired_path_name_findings(clone_dir)
    python_symbol_findings = source_tree_python_symbol_findings(clone_dir)
    synthesized_string_findings = source_tree_synthesized_string_scan_findings(
        clone_dir
    )
    active_test_literal_findings = source_tree_active_test_literal_findings(clone_dir)
    structured_key_findings = source_tree_structured_key_findings(clone_dir)
    governance_truth_findings = source_tree_governance_truth_findings(clone_dir)
    sensitive_negative_literal_findings = (
        source_tree_sensitive_negative_literal_findings(clone_dir)
    )
    return {
        "retired_text": {
            "status": "PASS" if not retired_text_findings else "FAIL",
            "finding_count": len(retired_text_findings),
            "findings": retired_text_findings,
        },
        "retired_governance_paths": {
            "status": "PASS" if not retired_governance_path_findings_ else "FAIL",
            "finding_count": len(retired_governance_path_findings_),
            "findings": retired_governance_path_findings_,
        },
        "retired_path_names": {
            "status": "PASS" if not retired_path_name_findings else "FAIL",
            "finding_count": len(retired_path_name_findings),
            "findings": retired_path_name_findings,
        },
        "python_symbols": {
            "status": "PASS" if not python_symbol_findings else "FAIL",
            "finding_count": len(python_symbol_findings),
            "findings": python_symbol_findings,
        },
        "synthesized_strings": {
            "status": "PASS" if not synthesized_string_findings else "FAIL",
            "finding_count": len(synthesized_string_findings),
            "findings": synthesized_string_findings,
        },
        "active_test_literals": {
            "status": "PASS" if not active_test_literal_findings else "FAIL",
            "finding_count": len(active_test_literal_findings),
            "findings": active_test_literal_findings,
        },
        "structured_keys": {
            "status": "PASS" if not structured_key_findings else "FAIL",
            "finding_count": len(structured_key_findings),
            "findings": structured_key_findings,
        },
        "governance_truth": {
            "status": "PASS" if not governance_truth_findings else "FAIL",
            "finding_count": len(governance_truth_findings),
            "findings": governance_truth_findings,
        },
        "sensitive_negative_literals": {
            "status": "PASS" if not sensitive_negative_literal_findings else "FAIL",
            "finding_count": len(sensitive_negative_literal_findings),
            "findings": sensitive_negative_literal_findings,
        },
    }


def source_tree_retired_governance_path_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    tracked = tuple(path.as_posix() for path in source_tree_tracked_paths(clone_dir))
    return retired_governance_path_findings(clone_dir, tracked_paths=tracked)


def source_tree_retired_path_name_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    return iter_path_name_findings(
        relative.as_posix() for relative in source_tree_tracked_paths(clone_dir)
    )


def source_tree_python_symbol_findings(clone_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if relative.suffix.lower() != ".py":
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        findings.extend(iter_python_symbol_findings(relative, source))
    return findings


def source_tree_structured_key_findings(clone_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if relative.suffix.lower() not in {".json", ".toml", ".yaml", ".yml"}:
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings.extend(iter_structured_key_findings(relative, text))
    return findings


def source_tree_retired_text_findings(clone_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if not source_tree_should_scan_retired_text(relative):
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        for rule, pattern in source_tree_forbidden_patterns().items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                if source_tree_path_allowed(
                    relative,
                    scan_group="retired_text",
                    rule=rule,
                    line_text=line.strip(),
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
    return findings


def source_tree_should_scan_retired_text(relative: Path) -> bool:
    if is_negative_test_allowlisted(relative.as_posix(), "retired_name"):
        return False
    return any(
        relative == root or root in relative.parents
        for root in SOURCE_TREE_RETIRED_TERM_ROOTS
    )


def source_tree_synthesized_string_scan_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if relative.suffix.lower() != ".py":
            continue
        if is_negative_test_allowlisted(
            relative.as_posix(),
            "stale_governance_truth",
        ) or is_negative_test_allowlisted(relative.as_posix(), "retired_name"):
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        findings.extend(
            source_tree_synthesized_string_findings(
                relative,
                path,
                lines,
                skip_lines,
                rule_definition_lines,
                scan_group="synthesized_strings",
                patterns=source_tree_forbidden_patterns(),
            )
        )
    return findings


def source_tree_split_string_findings(
    relative: Path,
    path: Path,
    lines: list[str],
    skip_lines: set[int],
    rule_definition_lines: set[int],
) -> list[dict[str, object]]:
    return source_tree_synthesized_string_findings(
        relative,
        path,
        lines,
        skip_lines,
        rule_definition_lines,
        scan_group="synthesized_strings",
        patterns=source_tree_forbidden_patterns(),
    )


def source_tree_synthesized_string_findings(
    relative: Path,
    path: Path,
    lines: list[str],
    skip_lines: set[int],
    rule_definition_lines: set[int],
    *,
    scan_group: str,
    patterns: dict[str, re.Pattern[str]],
) -> list[dict[str, object]]:
    if path.suffix != ".py":
        return []
    source = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[dict[str, object]] = []
    for item in iter_synthesized_string_findings(
        relative,
        source,
        patterns=patterns,
        allow_negative_tests=False,
        skip_rule_definition=False,
    ):
        line_number = int(item["line"])
        if line_number in skip_lines:
            continue
        line_text = lines[line_number - 1].strip() if line_number else ""
        if source_tree_path_allowed(
            relative,
            scan_group=scan_group,
            rule=str(item["rule"]),
            line_text=line_text,
            line_number=line_number,
            rule_definition_lines=rule_definition_lines,
        ):
            continue
        findings.append(
            {
                **item,
                "path": relative.as_posix(),
            }
        )
    return findings


def source_tree_python_synthesized_strings(
    tree: ast.AST,
    source: str,
) -> list[tuple[int, str, str]]:
    del tree
    return [
        (
            int(finding["line"]),
            str(finding["synthesized_value"]),
            str(finding["scan_mode"]),
        )
        for finding in iter_synthesized_string_findings(
            "source_tree_python_synthesized_strings.py",
            source,
            allow_negative_tests=False,
            skip_rule_definition=False,
        )
    ]


def source_tree_governance_rule_definition_lines(
    relative: Path,
    path: Path,
) -> set[int]:
    block_names = SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES.get(relative)
    if not block_names or path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.iter_child_nodes(tree):
        name = source_tree_definition_name(node)
        if name not in block_names:
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if start and end:
            lines.update(range(start, end + 1))
    return lines


def source_tree_definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def source_tree_allowlist_definition_lines(
    relative: Path,
    lines: list[str],
) -> set[int]:
    del relative, lines
    return set()


def source_tree_active_test_literal_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if relative.parts[:1] != ("tests",) and relative != Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "build_handoff_package.py"
        ):
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        for rule, pattern in source_tree_active_test_literal_patterns(
            clone_dir
        ).items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                if source_tree_path_allowed(
                    relative,
                    scan_group="active_test_literals",
                    rule=rule,
                    line_text=line.strip(),
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
        findings.extend(
            source_tree_synthesized_string_findings(
                relative,
                path,
                lines,
                skip_lines,
                rule_definition_lines,
                scan_group="active_test_literals",
                patterns=source_tree_active_test_literal_patterns(clone_dir),
            )
        )
    return findings


def source_tree_active_test_literal_path_allowed(relative: Path) -> bool:
    return bool(SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST.get(relative))


def source_tree_governance_truth_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_active_governance_doc_files(clone_dir):
        path = clone_dir / relative
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for rule, pattern in source_tree_governance_truth_patterns().items():
            for line_number, line in enumerate(lines, start=1):
                if not pattern.search(line):
                    continue
                if source_tree_retired_path_policy_line(relative, line):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
    return findings


def source_tree_retired_path_policy_line(relative: Path, line: str) -> bool:
    return is_retired_path_policy_line(relative.as_posix(), line)


def source_tree_active_governance_doc_files(clone_dir: Path) -> tuple[Path, ...]:
    tracked = set(source_tree_tracked_text_files(clone_dir))
    result: list[Path] = []
    for root in SOURCE_TREE_GOVERNANCE_TRUTH_ROOTS:
        path = clone_dir / root
        if path.is_file() and root in tracked:
            result.append(root)
            continue
        if path.is_dir():
            result.extend(
                relative
                for relative in sorted(tracked)
                if relative.is_relative_to(root)
                and relative.suffix.lower() in {".md", ".txt"}
                and (clone_dir / relative).is_file()
            )
    return tuple(sorted(set(result)))


def source_tree_governance_truth_patterns() -> dict[str, re.Pattern[str]]:
    return active_doc_forbidden_patterns()


def source_tree_active_test_literal_patterns(
    repo_root: Path | None = None,
) -> dict[str, re.Pattern[str]]:
    ensure_runtime_src_path(repo_root)
    from quant_research_platform.contracts.data import (
        DataFrequency,
        data_frequency_slug,
    )
    from quant_research_platform.markets.implementations.a_share.ids import (
        market_context_data_id,
        market_panel_data_id,
    )

    market_bars_data_id = discover_market_bars_data_id_helper(repo_root)
    frequency = DataFrequency.DAILY
    daily_slug = re.escape(data_frequency_slug(frequency))
    return {
        "active_market_bars_data_id": re.compile(
            rf"\b{re.escape(market_bars_data_id(frequency))}\b"
        ),
        "active_market_panel_data_id": re.compile(
            rf"\b{re.escape(market_panel_data_id(frequency))}\b"
        ),
        "active_market_context_data_id": re.compile(
            rf"\b{re.escape(market_context_data_id(frequency))}\b"
        ),
        "active_frequency_slug": re.compile(
            rf"(?:[\"']{daily_slug}[\"']|"
            rf"(?<![A-Za-z0-9_]){daily_slug}(?![A-Za-z0-9_]))"
        ),
        "retired_local_verification_evidence_range_surface": re.compile(
            r"\b(?:outside_evidence_range|evidence_range)\b"
        ),
        "retired_local_verification_output_surface": re.compile(
            r"\b(?:(?<!local-)verification-output|(?<!local_)verification_output)\b"
        ),
        "retired_result_evidence_root_text": re.compile(
            r"\bevidence root\b",
            re.IGNORECASE,
        ),
    }


def discover_market_bars_data_id_helper(repo_root: Path | None) -> object:
    src_root = ensure_runtime_src_path(repo_root)
    source_root = src_root / "quant_research_platform" / "sources"
    for path in sorted(source_root.rglob("ids.py")) if source_root.exists() else ():
        if "market_bars_data_id" not in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ):
            continue
        module_name = ".".join(path.relative_to(src_root).with_suffix("").parts)
        module = importlib.import_module(module_name)
        helper = getattr(module, "market_bars_data_id", None)
        if callable(helper):
            return helper
    raise RuntimeError("active market bars data id helper unavailable")


def ensure_runtime_src_path(repo_root: Path | None = None) -> Path:
    search_roots = []
    if repo_root is not None:
        search_roots.append(repo_root)
    script_dir = Path(__file__).resolve().parent
    search_roots.extend([Path.cwd(), *script_dir.parents])
    for parent in search_roots:
        src_root = parent / "src"
        if (src_root / "quant_research_platform").is_dir():
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))
            return src_root
    raise RuntimeError(
        "active literal source helpers unavailable: "
        "expected src/quant_research_platform under repo root"
    )


def source_tree_sensitive_negative_literal_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if relative.parts[:1] not in {(".agents",), ("tests",)}:
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        for rule, pattern in source_tree_sensitive_negative_patterns().items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                if source_tree_path_allowed(
                    relative,
                    scan_group="sensitive_negative_literals",
                    rule=rule,
                    line_text=line.strip(),
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
        findings.extend(
            source_tree_synthesized_string_findings(
                relative,
                path,
                lines,
                skip_lines,
                rule_definition_lines,
                scan_group="sensitive_negative_literals",
                patterns=source_tree_sensitive_negative_patterns(),
            )
        )
    return findings


def source_tree_tracked_text_files(clone_dir: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for relative in source_tree_tracked_paths(clone_dir):
        text_suffixes = {".py", ".md", ".json", ".toml", ".yaml", ".yml"}
        if relative.suffix.lower() not in text_suffixes:
            continue
        result.append(relative)
    return tuple(result)


def source_tree_tracked_paths(clone_dir: Path) -> tuple[Path, ...]:
    raw = _governance_git_text(clone_dir, "ls-files")
    return tuple(Path(line) for line in raw.splitlines() if line.strip())


def source_tree_path_allowed(
    relative: Path,
    *,
    scan_group: str,
    rule: str | None = None,
    line_text: str | None = None,
    line_number: int | None = None,
    rule_definition_lines: set[int] | None = None,
) -> bool:
    in_governance_rule_definition = (
        line_number is not None
        and rule_definition_lines is not None
        and line_number in rule_definition_lines
    )
    if scan_group in {"retired_text", "synthesized_strings", "retired_terms"}:
        if (
            rule is not None
            and rule.startswith("retired_result_")
            and relative in RESULT_ARCHITECTURE_NEGATIVE_TEST_SOURCES
        ):
            return True
        allowed_by_rule = SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST.get(relative, {})
        if rule is None or line_text is None:
            return False
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line_text):
            if is_allowed_runtime_symbol(relative, token, rule):
                return True
        if is_retired_path_policy_line(relative, line_text):
            return True
        if line_text in allowed_by_rule.get(rule, set()):
            return True
        allowed_rules = SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST.get(
            relative,
            set(),
        )
        return rule in allowed_rules and in_governance_rule_definition
    if scan_group == "active_test_literals":
        allowed_rules = SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST.get(relative, set())
        if rule not in allowed_rules:
            return False
        if relative in SOURCE_TREE_ACTIVE_TEST_LITERAL_PATHS:
            return True
        return in_governance_rule_definition
    if scan_group == "sensitive_negative_literals":
        allowed = SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST.get(
            relative,
            set(),
        )
        if isinstance(allowed, set):
            if rule not in allowed:
                return False
            if relative.parts[:1] == ("tests",):
                return True
            return in_governance_rule_definition
        if rule is None or line_text is None:
            return False
        return any(term in line_text for term in allowed.get(rule, set()))
    raise ValueError(f"unsupported_source_tree_scan_group: {scan_group}")


def is_reserved_handoff_evidence_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    for part in parts:
        if part in RESERVED_HANDOFF_EVIDENCE_STEMS:
            return True
        if part.endswith(".json") and part[:-5] in RESERVED_HANDOFF_EVIDENCE_STEMS:
            return True
    return False


def is_task_provided_handoff_evidence_blocked_path(name: str) -> bool:
    if name in BUILDER_OWNED_HANDOFF_EVIDENCE:
        return True
    parts = PurePosixPath(name).parts
    for part in parts:
        if part in TASK_PROVIDED_HANDOFF_EVIDENCE_RESERVED_STEMS:
            return True
        suffix = PurePosixPath(part).suffix
        if (
            suffix in {".json", ".md"}
            and part[: -len(suffix)] in TASK_PROVIDED_HANDOFF_EVIDENCE_RESERVED_STEMS
        ):
            return True
    return False


def normalized_handoff_evidence_path(raw_name: str) -> str:
    name = _normalize_governance_zip_path(raw_name)
    if not name.startswith("handoff_evidence/"):
        name = "handoff_evidence/" + name
    if not name.endswith(".json"):
        name += ".json"
    return name


def normalized_handoff_markdown_evidence_path(raw_name: str) -> str:
    name = _normalize_governance_zip_path(raw_name)
    if not name.startswith("handoff_evidence/"):
        name = "handoff_evidence/" + name
    if not name.endswith(".md"):
        name += ".md"
    return name


def handoff_evidence_blocked_key_failures(
    evidence_path: str,
    payload: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    collect_handoff_evidence_blocked_key_failures(
        evidence_path,
        payload,
        failures,
    )
    return failures


def collect_handoff_evidence_blocked_key_failures(
    evidence_path: str,
    payload: object,
    failures: list[str],
    *,
    key_path: tuple[str, ...] = (),
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            current_path = (*key_path, key_text)
            if key_text in BLOCKED_HANDOFF_EVIDENCE_KEYS:
                dotted = ".".join(current_path)
                failures.append(
                    f"handoff_evidence_forbidden_field: {evidence_path}: {dotted}"
                )
            collect_handoff_evidence_blocked_key_failures(
                evidence_path,
                value,
                failures,
                key_path=current_path,
            )
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            collect_handoff_evidence_blocked_key_failures(
                evidence_path,
                item,
                failures,
                key_path=(*key_path, str(index)),
            )


def validation_with_status_overrides(
    validation_evidence: dict[str, object],
    overrides: dict[str, str],
) -> dict[str, object]:
    payload = json.loads(_stable_governance_json(validation_evidence))
    rows = payload.get("checks")
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    validate_validation_evidence_contract(payload, require_mandatory=False)
    for check_name in BUILDER_OWNED_VALIDATION_CHECKS:
        set_check_status(rows, check_name, overrides.get(check_name, "NOT_RUN"))
    for check_name, status in overrides.items():
        set_check_status(rows, check_name, status)
    validate_validation_evidence_contract(payload)
    return payload


def enrich_validation_evidence(
    validation_evidence: dict[str, object],
    *,
    handoff_package_path: str,
    handoff_sidecar_path: str,
) -> dict[str, object]:
    payload = json.loads(_stable_governance_json(validation_evidence))
    validate_validation_evidence_contract(payload)
    payload["handoff_package_path"] = handoff_package_path
    payload["handoff_sidecar_path"] = handoff_sidecar_path
    rows = payload.get("checks")
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    return payload


def validate_validation_evidence_contract(
    payload: dict[str, object],
    *,
    require_mandatory: bool = True,
) -> None:
    failures = validation_evidence_contract_failures(
        payload,
        require_mandatory=require_mandatory,
    )
    if failures:
        raise SystemExit(";".join(failures))


def validation_evidence_contract_failures(
    payload: dict[str, object],
    *,
    require_mandatory: bool = True,
) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != "qrp-handoff-validation-evidence":
        failures.append("validation_evidence_schema_id_invalid")
    blocked = sorted(set(payload) & BLOCKED_VALIDATION_KEYS)
    failures.extend(f"validation_evidence_forbidden_field: {key}" for key in blocked)
    rows = payload.get("checks")
    if not isinstance(rows, list):
        failures.append("validation_evidence_checks_must_be_list")
        return failures
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"validation_evidence_check_not_object: {index}")
            continue
        if "check" in row:
            failures.append(f"validation_evidence_row_forbidden_field: {index}:check")
        extra = sorted(set(row) - VALIDATION_ROW_KEYS)
        if extra:
            failures.append(
                f"validation_evidence_row_unapproved_field: {index}:"
                + ",".join(extra)
            )
        for key in ("name", "status", "evidence"):
            if key not in row:
                failures.append(f"validation_evidence_row_missing: {index}:{key}")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            failures.append(f"validation_evidence_name_invalid: {index}")
        else:
            normalized_name = name.strip()
            if normalized_name in seen:
                failures.append(
                    f"validation_evidence_duplicate_check: {normalized_name}"
                )
            seen.add(normalized_name)
        if "status" in row:
            status = row.get("status")
            if not isinstance(status, str) or status not in VALID_STATUSES:
                failures.append(
                    f"validation_evidence_status_invalid: {index}:{row.get('status')}"
                )
        if "evidence" in row:
            evidence = row.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                failures.append(f"validation_evidence_evidence_invalid: {index}")
    if require_mandatory:
        missing = sorted(MANDATORY_VALIDATION_CHECKS - seen)
        failures.extend(
            f"validation_evidence_mandatory_check_missing: {name}"
            for name in missing
        )
    return failures


def set_check_status(
    rows: list[object],
    check_name: str,
    status: str,
) -> None:
    details_key = "evidence"
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if name == check_name:
            raw_status = row.get("status")
            current_status = raw_status if isinstance(raw_status, str) else ""
            current_evidence = str(row.get(details_key) or "")
            row["status"] = status
            generated_missing_evidence = current_evidence.startswith(
                "task validation_evidence missing; "
            )
            if current_status != status or not generated_missing_evidence:
                row[details_key] = validation_status_evidence(check_name, status)
            return
    rows.append(
        {
            "name": check_name,
            "status": status,
            details_key: validation_status_evidence(check_name, status),
        }
    )


def validation_status_evidence(check_name: str, status: str) -> str:
    if check_name == "handoff_verify":
        return f"verify_handoff_package.py completed with {status}"
    if check_name == "handoff_safety_scan":
        return f"verify_handoff_package.py safety scan completed with {status}"
    return f"{check_name}={status}"


def verification_matrix_payload(
    validation_evidence: dict[str, object],
) -> dict[str, object]:
    status_map = validation_status_map(validation_evidence)
    checks = []
    for name in ("handoff_verify", "handoff_safety_scan"):
        checks.append(
            {
                "name": name,
                "required": True,
                "deterministic": True,
                "status": status_map.get(name, "NOT_RUN"),
                "evidence": validation_status_evidence(
                    name,
                    status_map.get(name, "NOT_RUN"),
                ),
            }
        )
    return {
        "schema_id": "qrp-handoff-verification-matrix",
        "checks": checks,
    }


def validation_status_map(payload: dict[str, object]) -> dict[str, str]:
    validate_validation_evidence_contract(payload)
    rows = payload["checks"]
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    result: dict[str, str] = {}
    for row in rows:
        name = str(row["name"]).strip()
        status = str(row["status"])
        result[name] = status
    return result


def mandatory_validation_status_map(payload: dict[str, object]) -> dict[str, str]:
    status_map = validation_status_map(payload)
    return {name: status_map[name] for name in sorted(MANDATORY_VALIDATION_CHECKS)}


def aggregate_mandatory_validation_status(payload: dict[str, object]) -> str:
    statuses = mandatory_validation_status_map(payload).values()
    return aggregate_validation_matrix_status(statuses)


def _stable_governance_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _normalize_governance_zip_path(value: str) -> str:
    result = value.replace("\\", "/").lstrip("/")
    if not result or ".." in Path(result).parts:
        raise SystemExit(f"invalid_zip_path: {value}")
    return result


def _governance_git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
