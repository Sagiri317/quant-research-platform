"""Shared QRP governance contract rules for repository maintenance scripts.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import ast
import hashlib
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
    definition_sources: tuple[Path, ...] = ()

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True, eq=False)
class SourceTreeAllowedOccurrence:
    path: PurePosixPath
    rule_id: str
    exact_line: str
    owner: str = ""
    rationale: str = ""
    lifecycle: str = ""
    covered_by: tuple[str, ...] = ()
    py_owner: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _source_tree_posix_path(self.path))


def _source_tree_posix_path(path: str | Path | PurePosixPath) -> PurePosixPath:
    if isinstance(path, Path):
        raw = path.as_posix()
    elif isinstance(path, PurePosixPath):
        raw = path.as_posix()
    else:
        raw = str(path)
    return PurePosixPath(raw)


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
            r"(?<![A-Za-z0-9_-])(?:run_config\.json|run_manifest\.json|"
            r"validation_summary\.json|evidence_summary\.json|timings\.json|"
            r"timings\.csv|summary\.json|data/evidence|layer_manifest\.json|"
            r"config/run_config\.json|manifests/run_manifest\.json|"
            r"validations/validation_summary\.json|timings/timings\.json|"
            r"\.qrp-local-verification\.json)(?![A-Za-z0-9_-])"
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

PROFILE_RUNNER_RETIRED_LITERALS = (
    "QRP_MANAGED_REPO_ROOTS",
    "MANAGED_ROOTS_ENV",
    "managed_roots_env",
    "managed_repo_roots",
    "filter_managed_repo_paths",
    "_is_managed_repo_path",
    "--output-dir",
    "--tmp-dir",
    "profile-output",
)

PROFILE_RUNNER_RETIRED_TEXT_RULES = (
    TextRule(
        "profile_runner_retired_qrp_managed_repo_roots",
        r"(?<![A-Za-z0-9_])QRP_MANAGED_REPO_ROOTS(?![A-Za-z0-9_])",
        example="QRP_MANAGED_REPO_ROOTS",
    ),
    TextRule(
        "profile_runner_retired_managed_roots_env_symbol",
        r"(?<![A-Za-z0-9_])MANAGED_ROOTS_ENV(?![A-Za-z0-9_])",
        example="MANAGED_ROOTS_ENV",
    ),
    TextRule(
        "profile_runner_retired_managed_roots_env_field",
        r"(?<![A-Za-z0-9_])managed_roots_env(?![A-Za-z0-9_])",
        example="managed_roots_env",
    ),
    TextRule(
        "profile_runner_retired_managed_repo_roots",
        r"(?<![A-Za-z0-9_])managed_repo_roots(?![A-Za-z0-9_])",
        example="managed_repo_roots",
    ),
    TextRule(
        "profile_runner_retired_filter_managed_repo_paths",
        r"(?<![A-Za-z0-9_])filter_managed_repo_paths(?![A-Za-z0-9_])",
        example="filter_managed_repo_paths",
    ),
    TextRule(
        "profile_runner_retired_is_managed_repo_path",
        r"(?<![A-Za-z0-9_])_is_managed_repo_path(?![A-Za-z0-9_])",
        example="_is_managed_repo_path",
    ),
    TextRule(
        "profile_runner_retired_output_dir_flag",
        r"(?<![A-Za-z0-9_-])--output-dir(?![A-Za-z0-9_-])",
        example="--output-dir",
    ),
    TextRule(
        "profile_runner_retired_tmp_dir_flag",
        r"(?<![A-Za-z0-9_-])--tmp-dir(?![A-Za-z0-9_-])",
        example="--tmp-dir",
    ),
    TextRule(
        "profile_runner_retired_profile_output_text",
        r"(?<![A-Za-z0-9_-])profile-output(?![A-Za-z0-9_-])",
        example="profile-output",
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
SOURCE_TREE_RETIRED_TEXT_NEGATIVE_TEST_SOURCES = {
    Path("tests/agents/test_verify_governance.py"),
    Path("tests/architecture/test_governance_contract_rg_gate.py"),
    Path("tests/architecture/test_handoff_package_contract.py"),
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
            "quant_research_platform.sources",
        ),
    },
    {
        "root": "reports",
        "forbidden_prefixes": (
            "quant_research_platform.compute",
            "quant_research_platform.research",
            "quant_research_platform.artifacts",
            "quant_research_platform.workflows",
            "quant_research_platform.interfaces",
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
    {
        "root": "agents",
        "forbidden_prefixes": (
            "quant_research_platform.storage",
            "quant_research_platform.sources",
            "quant_research_platform.markets.implementations",
        ),
    },
    {
        "root": "compute",
        "forbidden_prefixes": (
            "quant_research_platform.reports",
            "quant_research_platform.artifacts",
            "quant_research_platform.workflows",
            "quant_research_platform.interfaces",
        ),
    },
    {
        "root": "interfaces",
        "forbidden_prefixes": (
            "quant_research_platform.sources",
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

STALE_GOVERNANCE_TRUTH_TEST_SOURCES = (
    "tests/agents/test_verify_governance.py",
    "tests/architecture/test_governance_contract_rg_gate.py",
)

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

PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES = (
    "tests/policy/test_share_safety.py",
    "tests/policy/test_share_safety_runtime_edges.py",
    "tests/storage/test_manifest_helper_edges.py",
    "tests/agents/handoff/test_handoff_tools.py",
    "tests/agents/handoff/test_handoff_package_verifier.py",
)

GOVERNANCE_TEST_EXCEPTION_PATHS = (
    "tests/policy/test_share_safety_runtime_edges.py",
)

LOCAL_PATH_TEXT_SAFETY_FIXTURE_SOURCES = (
    *PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES,
    "tests/architecture/test_share_safe_evidence.py",
    "tests/interfaces/test_api_cli_01_contract.py",
    "tests/storage/test_storage_01.py",
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


def _source_tree_allowed_occurrence(
    *,
    path: Path,
    rule_id: str,
    exact_line: str,
    owner: str,
    rationale: str,
    lifecycle: str,
    covered_by: tuple[str, ...],
    py_owner: str = "",
) -> SourceTreeAllowedOccurrence:
    return SourceTreeAllowedOccurrence(
        path=path,
        rule_id=rule_id,
        exact_line=exact_line,
        owner=owner,
        rationale=rationale,
        lifecycle=lifecycle,
        covered_by=covered_by,
        py_owner=py_owner,
    )


def _exact_line(*parts: str) -> str:
    return "".join(parts)


SOURCE_TREE_ALLOWED_OCCURRENCES = (
    _source_tree_allowed_occurrence(
        path=Path("src/quant_research_platform/policy/share_safety.py"),
        rule_id="retired_text:forbidden_private_real",
        exact_line='"private_real",',
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
        py_owner="assign:_RETIRED_LOCAL_ONLY_REPORT_PATH_NAMES",
    ),
    _source_tree_allowed_occurrence(
        path=Path("src/quant_research_platform/policy/share_safety.py"),
        rule_id="retired_text:forbidden_private_data",
        exact_line='"private_data",',
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
        py_owner="assign:_RETIRED_LOCAL_ONLY_REPORT_PATH_NAMES",
    ),
    *(
        _source_tree_allowed_occurrence(
            path=GOVERNANCE_CONTRACT_SCRIPT,
            rule_id=f"retired_text:{rule.name}",
            exact_line=f'"{rule.example}",',
            owner="governance-contract-rule-source",
            rationale="profile runner retired literal inventory line is owned here",
            lifecycle="current hard-cutover rule; remove only if rule is retired",
            covered_by=(
                "tests/architecture/test_governance_contract_rg_gate.py::"
                "test_profile_runner_retired_text_rules_are_contract_owned",
            ),
            py_owner="assign:PROFILE_RUNNER_RETIRED_LITERALS",
        )
        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
    ),
    *(
        _source_tree_allowed_occurrence(
            path=GOVERNANCE_CONTRACT_SCRIPT,
            rule_id=f"retired_text:{rule.name}",
            exact_line=f'r"{rule.pattern}",',
            owner="governance-contract-rule-source",
            rationale="profile runner retired regex rule line is owned here",
            lifecycle="current hard-cutover rule; remove only if rule is retired",
            covered_by=(
                "tests/architecture/test_governance_contract_rg_gate.py::"
                "test_profile_runner_retired_text_rules_are_contract_owned",
            ),
            py_owner="assign:PROFILE_RUNNER_RETIRED_TEXT_RULES",
        )
        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
    ),
    *(
        _source_tree_allowed_occurrence(
            path=GOVERNANCE_CONTRACT_SCRIPT,
            rule_id=f"retired_text:{rule.name}",
            exact_line=f'example="{rule.example}",',
            owner="governance-contract-rule-source",
            rationale="profile runner retired example line is owned here",
            lifecycle="current hard-cutover rule; remove only if rule is retired",
            covered_by=(
                "tests/architecture/test_governance_contract_rg_gate.py::"
                "test_profile_runner_retired_text_rules_are_contract_owned",
            ),
            py_owner="assign:PROFILE_RUNNER_RETIRED_TEXT_RULES",
        )
        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
    ),
    *(
        _source_tree_allowed_occurrence(
            path=Path("tests/agents/test_verify_governance.py"),
            rule_id=f"retired_text:{rule.name}",
            exact_line=f'monkeypatch.setenv("{rule.example}", "forge-managed-root")',
            owner="governance-negative-tests",
            rationale="forged environment negative test proves the env is ignored",
            lifecycle="current negative fixture; update with profile runner rule",
            covered_by=(
                "tests/agents/test_verify_governance.py::"
                "test_forged_retired_profile_env_does_not_skip_repo_file_negative_payload",
            ),
            py_owner=(
                "def:test_forged_retired_profile_env_does_not_skip_repo_file_"
                "negative_payload"
            ),
        )
        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
        if rule.name == "profile_runner_retired_qrp_managed_repo_roots"
    ),
    *(
        _source_tree_allowed_occurrence(
            path=Path("tests/architecture/test_governance_contract_rg_gate.py"),
            rule_id=f"retired_text:{rule_name}",
            exact_line=line_text,
            owner="architecture-governance-negative-tests",
            rationale="architecture gate carries an exact retired fixture line",
            lifecycle="current negative fixture; update with profile runner rule",
            covered_by=("tests/architecture/test_governance_contract_rg_gate.py",),
            py_owner=f"def:{function_name}",
        )
        for rule_name, literal, function_name, line_text in (
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example,
                "test_profile_runner_retired_text_allowlist_is_precise",
                f'if rule.example == "{PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example}"',
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example,
                "test_profile_runner_retired_text_allowlist_is_precise",
                (
                    f'assert all("{PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example}" '
                    "in entry.exact_line for entry in agent_entries)"
                ),
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example,
                "test_retired_profile_literal_allowlist_is_context_scoped_in_python_file",
                (
                    "'    monkeypatch.setenv("
                    f'"{PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example}", ' + "'"
                ),
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example,
                "test_retired_profile_literal_allowlist_is_context_scoped_in_python_file",
                (
                    '"    # '
                    + PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example
                    + ' must not be broadly allowlisted",'
                ),
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[6].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[6].example,
                "test_profile_runner_retired_cli_flags_are_rejected",
                (
                    f'for flag in ("{PROFILE_RUNNER_RETIRED_TEXT_RULES[6].example}", '
                    f'"{PROFILE_RUNNER_RETIRED_TEXT_RULES[7].example}"):'
                ),
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[7].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[7].example,
                "test_profile_runner_retired_cli_flags_are_rejected",
                (
                    f'for flag in ("{PROFILE_RUNNER_RETIRED_TEXT_RULES[6].example}", '
                    f'"{PROFILE_RUNNER_RETIRED_TEXT_RULES[7].example}"):'
                ),
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example,
                "test_profile_runner_env_is_not_forwarded_to_pytest_subprocess",
                (
                    "monkeypatch.setenv("
                    f'"{PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example}", '
                    '"forge-managed-root")'
                ),
            ),
            (
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].name,
                PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example,
                "test_profile_runner_env_is_not_forwarded_to_pytest_subprocess",
                (
                    f'assert "{PROFILE_RUNNER_RETIRED_TEXT_RULES[0].example}" '
                    'not in captured["env"]'
                ),
            ),
        )
    ),
)

SOURCE_TREE_ALLOWED_OCCURRENCES += (
    _source_tree_allowed_occurrence(
        path=Path("tests/architecture/test_governance_contract_rg_gate.py"),
        rule_id="retired_text:retired_result_run_level_path",
        exact_line='assert rule.search("summary.json")',
        owner="architecture-governance-negative-tests",
        rationale=(
            "negative test proves the retired result path rule still catches "
            "the exact summary JSON filename"
        ),
        lifecycle=(
            "current negative fixture; remove if run-level retired path rule "
            "changes"
        ),
        covered_by=(
            "tests/architecture/test_governance_contract_rg_gate.py::"
            "test_retired_run_level_path_rule_ignores_validation_summary_substring",
        ),
        py_owner=(
            "def:test_retired_run_level_path_rule_ignores_validation_summary_substring"
        ),
    ),
    _source_tree_allowed_occurrence(
        path=Path("tests/architecture/test_governance_contract_rg_gate.py"),
        rule_id="retired_text:retired_result_run_level_path",
        exact_line='assert rule.search("run/summary.json")',
        owner="architecture-governance-negative-tests",
        rationale=(
            "negative test proves the retired result path rule still catches "
            "the run-level summary JSON path"
        ),
        lifecycle=(
            "current negative fixture; remove if run-level retired path rule "
            "changes"
        ),
        covered_by=(
            "tests/architecture/test_governance_contract_rg_gate.py::"
            "test_retired_run_level_path_rule_ignores_validation_summary_substring",
        ),
        py_owner=(
            "def:test_retired_run_level_path_rule_ignores_validation_summary_substring"
        ),
    ),
    _source_tree_allowed_occurrence(
        path=Path("tests/architecture/test_governance_contract_rg_gate.py"),
        rule_id="retired_text:retired_result_run_level_path",
        exact_line='assert rule.search("summary.json.bak")',
        owner="architecture-governance-negative-tests",
        rationale=(
            "negative test proves backup-suffixed retired result files stay "
            "caught"
        ),
        lifecycle=(
            "current negative fixture; remove if run-level retired path rule "
            "changes"
        ),
        covered_by=(
            "tests/architecture/test_governance_contract_rg_gate.py::"
            "test_retired_run_level_path_rule_ignores_validation_summary_substring",
        ),
        py_owner=(
            "def:test_retired_run_level_path_rule_ignores_validation_summary_substring"
        ),
    ),
)

_MIGRATED_SOURCE_TREE_LITERAL_OCCURRENCE_ROWS = (
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "synthesized_strings:profile_runner_retired_qrp_managed_repo_roots",
        '"\\n".join(',
        "def:test_retired_profile_literal_allowlist_is_context_scoped_in_python_file",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "synthesized_strings:profile_runner_retired_qrp_managed_repo_roots",
        (
            "'    monkeypatch.setenv("
            '"QRP_MANAGED_REPO_ROOTS", '
            + "'"
        ),
        "def:test_retired_profile_literal_allowlist_is_context_scoped_in_python_file",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_market_bars_data_id",
        _exact_line(
            "assert source_ids.market_bars_data_id(DataFrequency.DAILY) ",
            '== "market_bars_1d"',
        ),
        "def:test_current_source_and_market_ids_are_frequency_derived",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_market_bars_data_id",
        '"market_bars_1d",',
        "def:test_regular_tests_do_not_hardcode_active_frequency_literals",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_market_panel_data_id",
        _exact_line(
            "assert market_ids.market_panel_data_id(DataFrequency.DAILY) ",
            '== "market_panel_1d"',
        ),
        "def:test_current_source_and_market_ids_are_frequency_derived",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_market_panel_data_id",
        '"market_panel_1d",',
        "def:test_regular_tests_do_not_hardcode_active_frequency_literals",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_market_context_data_id",
        _exact_line(
            "assert market_ids.market_context_data_id(DataFrequency.DAILY) ",
            '== "market_context_1d"',
        ),
        "def:test_current_source_and_market_ids_are_frequency_derived",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_market_context_data_id",
        '"market_context_1d",',
        "def:test_regular_tests_do_not_hardcode_active_frequency_literals",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_frequency_slug",
        'assert normalize_data_frequency("1d") is DataFrequency.DAILY',
        "def:test_data_frequency_helpers_define_reserved_vocabulary_and_active_support",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_frequency_slug",
        'assert data_frequency_slug(DataFrequency.DAILY) == "1d"',
        "def:test_data_frequency_helpers_define_reserved_vocabulary_and_active_support",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_frequency_slug",
        "'\"1d\"',",
        "def:test_regular_tests_do_not_hardcode_active_frequency_literals",
    ),
    (
        "tests/architecture/test_frequency_vocabulary_vs_active_support.py",
        "active_test_literals:active_frequency_slug",
        "\"'1d'\",",
        "def:test_regular_tests_do_not_hardcode_active_frequency_literals",
    ),
    (
        "tests/agents/test_verify_governance.py",
        "retired_text:clean_rewrite",
        _exact_line(
            'retired_component = rule_form(contract.RETIRED_NAME_RULES, "clean_',
            'rewrite")',
        ),
        "def:test_path_name_gate_fails_retired_stage_paths",
    ),
    (
        "tests/agents/test_verify_governance.py",
        "retired_text:clean_rewrite",
        _exact_line('"clean_', 'rewrite",'),
        "def:test_structured_key_gate_fails_retired_keys_and_allows_values",
    ),
    (
        "tests/agents/test_verify_governance.py",
        "retired_text:clean_rewrite",
        _exact_line(
            'contract.structured_key_forbidden_patterns()["clean_', 'rewrite"],'
        ),
        "def:test_structured_key_gate_fails_retired_keys_and_allows_values",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line(
            'retired_component = rule_form(contract.RETIRED_NAME_RULES, "clean_',
            'rewrite")',
        ),
        "def:test_source_tree_scan_rejects_retired_stage_path_names",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line('"rule": "clean_', 'rewrite",'),
        "def:test_source_tree_scan_rejects_retired_stage_path_names",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line('"clean_', 'rewrite",'),
        "def:test_source_tree_scan_rejects_retired_structured_keys",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line(
            'contract.structured_key_forbidden_patterns()["clean_', 'rewrite"],'
        ),
        "def:test_source_tree_scan_rejects_retired_structured_keys",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line('"rule": "clean_', 'rewrite",'),
        "def:test_source_tree_scan_rejects_retired_structured_keys",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line(
            'retired_value = rule_form(contract.RETIRED_NAME_RULES, "clean_',
            'rewrite")',
        ),
        "def:test_source_tree_scan_independently_rejects_synthesized_strings",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line('"clean_', 'rewrite",'),
        "def:test_source_tree_scan_independently_rejects_python_symbols",
    ),
    (
        "tests/architecture/test_handoff_package_contract.py",
        "retired_text:clean_rewrite",
        _exact_line('} == {("tools/hack_symbol.py", symbol, "clean_', 'rewrite")}'),
        "def:test_source_tree_scan_independently_rejects_python_symbols",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_evidence_range_surface",
        _exact_line(
            '"retired_local_verification_evidence_range_surface": ',
            '("outside_evidence_range"),',
        ),
        "def:test_result_architecture_rules_cover_hard_cut_examples",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_evidence_range_surface",
        '"outside_evidence_range",',
        "def:test_active_docs_reject_retired_review_safety_names",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_evidence_range_surface",
        '"evidence_range",',
        "def:test_active_docs_reject_retired_review_safety_names",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_evidence_range_surface",
        "\"    label = 'evidence_range'\",",
        "def:test_source_tree_rejects_retired_local_verification_surface_terms",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_output_surface",
        '"retired_local_verification_output_surface": "verification-output",',
        "def:test_result_architecture_rules_cover_hard_cut_examples",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_output_surface",
        '"verification-output",',
        "def:test_active_docs_reject_retired_review_safety_names",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_output_surface",
        '"verification_output",',
        "def:test_active_docs_reject_retired_review_safety_names",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_output_surface",
        "\"    run_root = Path('verification-output/s1')\",",
        "def:test_source_tree_rejects_retired_local_verification_surface_terms",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_local_verification_output_surface",
        "\"    run_id = 'verification_output'\",",
        "def:test_source_tree_rejects_retired_local_verification_surface_terms",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_result_evidence_root_text",
        '"retired_result_evidence_root_text": "evidence root",',
        "def:test_result_architecture_rules_cover_hard_cut_examples",
    ),
    (
        "tests/architecture/test_governance_contract_rg_gate.py",
        "active_test_literals:retired_result_evidence_root_text",
        '"evidence root",',
        "def:test_active_docs_reject_retired_review_safety_names",
    ),
    (
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('return "vendor', '_marker"'),
        "def:unsafe_text_reason",
    ),
    (
        ".agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py",
        "sensitive_negative_literals:sensitive_secret",
        _exact_line("tok", "en = re.compile("),
        "def:check_markdown_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"zip_path": "E', ':/quant-workspace/private/handoff.zip",'),
        "def:test_handoff_package_verifier_scans_json_sidecar_absolute_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('\'{"path": "E', ":/private\"}\\n',"),
        "def:test_handoff_package_verifier_still_rejects_regular_evidence_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('".run\\\\\\\\priv', 'ate\\\\\\\\s1",'),
        "def:test_handoff_package_verifier_rejects_local_run_paths_in_sidecar",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/work', "space/qrp/data/r", 'esearch/a_share/demo",'),
        "def:test_handoff_verifier_rejects_posix_local_workspace_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/work', "spaces/qrp/reports/priv", 'ate_research/a_share/run",'),
        "def:test_handoff_verifier_rejects_posix_local_workspace_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/h', 'ome/user/private.txt",'),
        "def:test_handoff_verifier_rejects_posix_local_workspace_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/Us', 'ers/user/private.txt",'),
        "def:test_handoff_verifier_rejects_posix_local_workspace_paths",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('"provider_name', '=raw-provider",'),
        "def:test_handoff_package_verifier_rejects_private_provider_markers",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('"source_name', '=raw-source",'),
        "def:test_handoff_package_verifier_rejects_private_provider_markers",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('"data_provider', '=raw-provider",'),
        "def:test_handoff_package_verifier_rejects_private_provider_markers",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('"vendor_name=', 'raw-provider",'),
        "def:test_handoff_package_verifier_rejects_private_provider_markers",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('"raw_v', 'endor=export",'),
        "def:test_handoff_package_verifier_rejects_private_provider_markers",
    ),
    (
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('"vendor_', 'payload={}",'),
        "def:test_handoff_package_verifier_rejects_private_provider_markers",
    ),
    (
        "tests/policy/test_share_safety.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            'text = "created at C',
            ":/Us",
            "ers/example/project/data",
            '/raw/source/file.csv"',
        ),
        "def:test_share_safety_rejects_local_path_and_sanitizes_it",
    ),
    (
        "tests/policy/test_share_safety.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('assert "C', ":/Us", 'ers" not in sanitize_shared_text(text)'),
        "def:test_share_safety_rejects_local_path_and_sanitizes_it",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            'text = r"C:',
            '\\Users\\DELL\\secret.csv \\\\serv',
            "er\\sha",
            "re\\raw.xlsx /mnt",
            "/data/private.csv /h",
            "ome/user/private.csv /Us",
            'ers/name/private.csv /tmp/x"  # noqa: E501',
        ),
        "def:test_sanitizer_redacts_windows_unc_and_posix_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('assert "\\\\\\\\serv', 'er" not in sanitized'),
        "def:test_sanitizer_redacts_windows_unc_and_posix_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            'text = "/work',
            "space/qrp/data/r",
            'esearch/a_share/demo/_manifest/manifest.json"',
        ),
        "def:test_sanitizer_redacts_absolute_private_repo_path_as_one_token",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/work', "space/qrp/data/r", 'esearch/a_share/demo",'),
        "def:test_sanitizer_redacts_repo_private_absolute_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/work', "spaces/qrp/reports/priv", 'ate_research/a_share/run",'),
        "def:test_sanitizer_redacts_repo_private_absolute_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            'r"C:',
            "\\repo\\data\\research\\a_",
            'share\\demo",',
        ),
        "def:test_sanitizer_redacts_repo_private_absolute_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            'r"D:',
            "\\repo\\reports\\priv",
            "ate_research\\a_",
            'share\\run",',
        ),
        "def:test_sanitizer_redacts_repo_private_absolute_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line('"/work', "space/qrp/reports/priv", 'ate_research/a_share/run",'),
        "def:test_sanitizer_rejects_local_output_paths",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            '"path": r"D:',
            "\\quant-workspace\\data\\processed\\a_",
            'share",',
        ),
        "def:test_sanitizer_recurses_public_objects",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            '({"path": "/work',
            "space/qrp/data/r",
            'esearch/a_share/demo"}, "local_only_path"),',
        ),
        "def:test_internal_artifact_metadata_safety_is_owned_by_runtime",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_secret",
        _exact_line(
            '"data',
            "/raw/csmar/export.csv stock code 6000",
            "00.SH CSMAR vendor tok",
            'en=abcdef123456"  # noqa: E501',
        ),
        "def:test_sanitizer_redacts_shared_markers_stock_codes_and_credentials",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_secret",
        _exact_line('"nested": {"secret": "pass', 'word=abcdef123456"},'),
        "def:test_sanitizer_recurses_public_objects",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_secret",
        _exact_line('({"credential": "tok', 'en=abcdef123456"}, "credential"),'),
        "def:test_internal_artifact_metadata_safety_is_owned_by_runtime",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('("vendor', '_marker=raw-provider", "provider_source_marker"),'),
        "def:test_unsafe_for_share_reason_covers_share_safety_markers",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('("provider_name', '=raw-provider", "provider_source_marker"),'),
        "def:test_unsafe_for_share_reason_covers_share_safety_markers",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('("source_name', '=raw-source", "provider_source_marker"),'),
        "def:test_unsafe_for_share_reason_covers_share_safety_markers",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('("data_provider', '=raw-provider", "provider_source_marker"),'),
        "def:test_unsafe_for_share_reason_covers_share_safety_markers",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_provider_marker",
        _exact_line('("vendor_name=', 'raw-provider", "provider_source_marker"),'),
        "def:test_unsafe_for_share_reason_covers_share_safety_markers",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_stock_code_sample",
        _exact_line(
            '"data',
            "/raw/csmar/export.csv stock code 6000",
            "00.SH CSMAR vendor tok",
            'en=abcdef123456"  # noqa: E501',
        ),
        "def:test_sanitizer_redacts_shared_markers_stock_codes_and_credentials",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_stock_code_sample",
        _exact_line('"8300', '00.BJ",'),
        "def:test_stock_code_detector_covers_a_share_exchange_suffixes",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_stock_code_sample",
        _exact_line('"4300', '47.BJ",'),
        "def:test_stock_code_detector_covers_a_share_exchange_suffixes",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_stock_code_sample",
        _exact_line('"6000', '00.SH",'),
        "def:test_stock_code_detector_covers_a_share_exchange_suffixes",
    ),
    (
        "tests/policy/test_share_safety_runtime_edges.py",
        "sensitive_negative_literals:sensitive_stock_code_sample",
        _exact_line('"rows": ["0000', '01.SZ"],'),
        "def:test_sanitizer_recurses_public_objects",
    ),
    (
        "tests/storage/test_manifest_helper_edges.py",
        "sensitive_negative_literals:sensitive_local_only_path",
        _exact_line(
            '({"path": "/work',
            "space/qrp/data/r",
            'esearch/a_share/demo"}, "local_only_path"),',
        ),
        "def:test_internal_manifest_rejects_private_or_row_level_metadata",
    ),
)
SOURCE_TREE_ALLOWED_OCCURRENCES += tuple(
    _source_tree_allowed_occurrence(
        path=Path(path),
        rule_id=rule_id,
        exact_line=exact_line,
        owner="source-tree-exact-occurrence-migration",
        rationale="migrated literal fixture from path-level source-tree allowance",
        lifecycle="current exact occurrence fixture; update when fixture line changes",
        covered_by=("tests/architecture/test_governance_contract_rg_gate.py",),
        py_owner=py_owner,
    )
    for (
        path,
        rule_id,
        exact_line,
        py_owner,
    ) in _MIGRATED_SOURCE_TREE_LITERAL_OCCURRENCE_ROWS
)

_SOURCE_TREE_RETIRED_RULE_DEFINITION_RULES = {
    rule.name
    for rule in (
        *STALE_GOVERNANCE_TRUTH_RULES,
        *RETIRED_NAME_RULES,
        *RETIRED_PUBLIC_SURFACE_RULES,
        *VERSION_SURFACE_FORBIDDEN_RULES,
        *RESULT_ARCHITECTURE_RETIRED_TEXT_RULES,
        *PROFILE_RUNNER_RETIRED_TEXT_RULES,
    )
}

SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST = {
    GOVERNANCE_CONTRACT_SCRIPT: {
        rule.name for rule in RESULT_ARCHITECTURE_RETIRED_TEXT_RULES
    },
    Path("tests/agents/test_verify_governance.py"): (
        _SOURCE_TREE_RETIRED_RULE_DEFINITION_RULES
    ),
    Path("tests/architecture/test_governance_contract_rg_gate.py"): (
        _SOURCE_TREE_RETIRED_RULE_DEFINITION_RULES
    ),
    Path("tests/architecture/test_handoff_package_contract.py"): (
        _SOURCE_TREE_RETIRED_RULE_DEFINITION_RULES
    ),
    Path("tests/architecture/test_retired_surface_absent.py"): {
        "retired_csmar_source_alias",
        "retired_a_share_market_alias",
        "retired_model_algorithm_collection",
        "forbidden_render_pr_body",
    },
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
        definition_sources=(
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
        definition_sources=(
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
        definition_sources=(
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
        definition_sources=(GOVERNANCE_CONTRACT_SCRIPT,),
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
        "STALE_GOVERNANCE_TRUTH_TEST_SOURCES",
        "PRIVACY_TEXT_SAFETY_FIXTURE_SOURCES",
        "RULE_DEFINITION_ALLOWLIST",
        "PACKAGE_FORBIDDEN_PATH_PARTS",
        "RETIRED_PATH_POLICY_MARKERS",
        "SCANNER_DEFINITION_MARKERS",
        "SCANNER_SCRIPT_DEFINITION_MARKERS",
        "CODE_MAP_FORBIDDEN_TERMS",
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
        "_MIGRATED_SOURCE_TREE_LITERAL_OCCURRENCE_ROWS",
        "SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST",
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
    Path("tests/agents/test_verify_governance.py"): {
        "test_active_doc_scan_fails_stale_current_truth",
        "test_retired_naming_rules_cover_required_derived_forms",
        "test_python_symbol_gate_fails_retired_stage_and_surface_names",
        "test_synthesized_string_gate_fails_retired_stage_names",
        "test_source_tree_retired_text_negative_sources_scan_all_retired_rules",
        "test_version_surface_policy_keeps_legal_tokens_out_of_retired_gate",
        "test_version_surface_policy_rejects_legacy_v_api_prefix",
    },
    Path("tests/architecture/test_governance_contract_rg_gate.py"): {
        "ARCHITECTURE_GATE_NEGATIVE_LITERAL_ALLOWLIST",
        "test_source_tree_path_allowed_does_not_consume_literal_occurrences",
        "test_active_tree_has_no_retired_terms",
        "test_active_tree_has_no_stale_qrp_owned_version_tokens",
        "test_result_architecture_rules_cover_hard_cut_examples",
        "test_active_docs_reject_retired_review_safety_names",
        "test_verify_governance_uses_current_local_only_roots",
        "test_gitignore_is_active_doc_scanned_for_retired_result_paths",
        "test_gitignore_rejects_outputs_local_layout",
        "test_source_tree_rejects_retired_local_verification_surface_terms",
        "test_research_foundation_semantic_closure_terms_are_retired",
        "test_approved_writer_signatures_do_not_accept_arbitrary_physical_roots",
    },
    Path("tests/architecture/test_handoff_package_contract.py"): {
        "test_handoff_package_contract_uses_evidence_mode_outputs",
    },
}


def validate_source_tree_allowed_occurrences(
    entries: tuple[SourceTreeAllowedOccurrence, ...],
    *,
    patterns_by_rule_id: Mapping[str, re.Pattern[str]] | None = None,
) -> list[str]:
    patterns = (
        dict(patterns_by_rule_id)
        if patterns_by_rule_id is not None
        else source_tree_allowed_occurrence_patterns(repo_root=Path("."))
    )
    failures: list[str] = []
    for index, entry in enumerate(entries):
        prefix = f"source_tree_allowed_occurrence_invalid:{index}:{entry.path}"
        failures.extend(
            f"{prefix}:{failure}"
            for failure in source_tree_allowed_occurrence_path_failures(entry.path)
        )
        if not str(entry.rule_id).strip():
            failures.append(f"{prefix}:source_tree_allowed_occurrence_missing_rule_id")
        elif entry.rule_id not in patterns:
            failures.append(f"{prefix}:source_tree_allowed_occurrence_unknown_rule")
        elif (
            not entry.rule_id.startswith("synthesized_strings:")
            and str(entry.exact_line).strip()
        ):
            match_count = len(patterns[entry.rule_id].findall(entry.exact_line))
            if match_count == 0:
                failures.append(
                    f"{prefix}:source_tree_allowed_occurrence_exact_line_rule_mismatch"
                )
            elif match_count > 1:
                failures.append(
                    f"{prefix}:source_tree_allowed_occurrence_exact_line_rule_overmatched"
                )
        if not str(entry.exact_line).strip():
            failures.append(
                f"{prefix}:source_tree_allowed_occurrence_missing_exact_line"
            )
        path_suffix = entry.path.suffix
        if path_suffix == ".py":
            if not str(entry.py_owner).strip():
                failures.append(
                    f"{prefix}:source_tree_allowed_occurrence_missing_py_owner"
                )
        elif entry.py_owner:
            failures.append(
                f"{prefix}:source_tree_allowed_occurrence_non_python_rejects_py_owner"
            )
        if not str(entry.owner).strip():
            failures.append(f"{prefix}:source_tree_allowed_occurrence_missing_owner")
        if not str(entry.rationale).strip():
            failures.append(
                f"{prefix}:source_tree_allowed_occurrence_missing_rationale"
            )
        if not str(entry.lifecycle).strip():
            failures.append(
                f"{prefix}:source_tree_allowed_occurrence_missing_lifecycle"
            )
        if not entry.covered_by:
            failures.append(
                f"{prefix}:source_tree_allowed_occurrence_missing_covered_by"
            )
    return failures


def source_tree_allowed_occurrence_path_failures(
    path: str | Path | PurePosixPath,
) -> list[str]:
    if isinstance(path, Path):
        raw = path.as_posix()
    elif isinstance(path, PurePosixPath):
        raw = path.as_posix()
    else:
        raw = str(path)
    failures: list[str] = []
    if not raw or raw == ".":
        failures.append("source_tree_allowed_occurrence_missing_path")
        return failures
    if "\\" in raw:
        failures.append("source_tree_allowed_occurrence_windows_backslash_path")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", raw):
        failures.append("source_tree_allowed_occurrence_absolute_path")
    normalized = raw.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        failures.append("source_tree_allowed_occurrence_parent_path")
    if "." in parts or PurePosixPath(normalized).as_posix() != normalized:
        failures.append("source_tree_allowed_occurrence_not_posix_normalized")
    return failures


def source_tree_allowed_occurrence_patterns(
    *,
    repo_root: Path | None = None,
) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for namespace in ("retired_text", "synthesized_strings"):
        patterns.update(
            {
                f"{namespace}:{rule}": pattern
                for rule, pattern in source_tree_forbidden_patterns().items()
            }
        )
    patterns.update(
        {
            f"sensitive_negative_literals:{rule}": pattern
            for rule, pattern in source_tree_sensitive_negative_patterns().items()
        }
    )
    if repo_root is not None:
        patterns.update(
            {
                f"active_test_literals:{rule}": pattern
                for rule, pattern in source_tree_active_test_literal_patterns(
                    repo_root
                ).items()
            }
        )
    return patterns


def _source_tree_allowed_occurrence_entries(
    relative: Path,
    *,
    scan_group: str,
    rule: str,
) -> tuple[SourceTreeAllowedOccurrence, ...]:
    rule_id = f"{scan_group}:{rule}"
    relative_key = _source_tree_posix_path(relative)
    return tuple(
        entry
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
        if entry.path == relative_key and entry.rule_id == rule_id
    )


def _source_tree_allowed_occurrence_match_status(
    entry: SourceTreeAllowedOccurrence,
    *,
    relative: Path,
    rule_id: str,
    line_text: str,
    line_number: int | None,
    owner_ranges: dict[str, tuple[frozenset[int], ...]],
) -> str:
    relative_key = _source_tree_posix_path(relative)
    if entry.path != relative_key or entry.rule_id != rule_id:
        return "no_match"
    if line_text.strip() != entry.exact_line:
        return "no_match"
    if relative_key.suffix != ".py":
        return "match"
    ranges = owner_ranges.get(entry.py_owner, ())
    if len(ranges) > 1:
        return "ambiguous_owner"
    if len(ranges) == 0:
        return "stale_owner"
    if line_number is None or line_number not in ranges[0]:
        return "no_match"
    return "match"


def _source_tree_retired_rule_definition_allowed(
    relative: Path,
    *,
    rule: str,
    line_text: str,
    line_number: int | None,
    rule_definition_lines: set[int] | None,
    definition_contexts: dict[str, set[int]] | None = None,
) -> bool:
    del line_text, definition_contexts
    allowed_rules = SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST.get(relative, set())
    return (
        rule in allowed_rules
        and line_number is not None
        and rule_definition_lines is not None
        and line_number in rule_definition_lines
    )


def _source_tree_matching_allowed_occurrence_entries(
    relative: Path,
    *,
    scan_group: str,
    rule: str,
    line_text: str,
    line_number: int | None,
    owner_ranges: dict[str, tuple[frozenset[int], ...]] | None = None,
) -> tuple[
    tuple[SourceTreeAllowedOccurrence, ...],
    tuple[SourceTreeAllowedOccurrence, ...],
    tuple[SourceTreeAllowedOccurrence, ...],
]:
    matches: list[SourceTreeAllowedOccurrence] = []
    ambiguous: list[SourceTreeAllowedOccurrence] = []
    stale_owner: list[SourceTreeAllowedOccurrence] = []
    rule_id = f"{scan_group}:{rule}"
    for entry in _source_tree_allowed_occurrence_entries(
        relative,
        scan_group=scan_group,
        rule=rule,
    ):
        match = _source_tree_allowed_occurrence_match_status(
            entry,
            relative=relative,
            rule_id=rule_id,
            line_text=line_text,
            line_number=line_number,
            owner_ranges=owner_ranges or {},
        )
        if match == "match":
            matches.append(entry)
        elif match == "ambiguous_owner":
            ambiguous.append(entry)
        elif match == "stale_owner":
            stale_owner.append(entry)
    return tuple(matches), tuple(ambiguous), tuple(stale_owner)


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
    patterns.update(compile_text_rules(PROFILE_RUNNER_RETIRED_TEXT_RULES))
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
    return source_tree_forbidden_patterns()


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
    patterns.update(profile_runner_retired_patterns())
    return patterns


def profile_runner_retired_patterns() -> dict[str, re.Pattern[str]]:
    return compile_text_rules(PROFILE_RUNNER_RETIRED_TEXT_RULES)


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
    allowlisted_negative_test = (
        normalize_repo_path(path) in STALE_GOVERNANCE_TRUTH_TEST_SOURCES
    )
    if not allowlisted_negative_test:
        return False
    if any(marker in line for marker in stale_fixture_markers()):
        return True
    return any(
        pattern.search(line) for pattern in active_doc_forbidden_patterns().values()
    )


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
    return False


def is_allowed_rule_definition(path: str | Path, rule_id: str | None = None) -> bool:
    del rule_id
    return normalize_repo_path(path) in RULE_DEFINITION_ALLOWLIST


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
    allow_negative_tests: bool = False,
    skip_rule_definition: bool = True,
) -> list[dict[str, object]]:
    del allow_negative_tests
    if skip_rule_definition and is_governance_rule_definition_path(path):
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


# Handoff retired surface names for source-tree governance scanning.
VALID_STATUSES = set(VALID_STATUS_TOKENS)
RETIRED_HANDOFF_ENTRY_NAMES = frozenset(
    {
        "git_status.txt",
        "receipt-index.json",
        "handoff_evidence/generation.json",
        "handoff_evidence/bundle-reconstruction.json",
        "handoff_evidence/private-smoke-status.json",
        "handoff_evidence/private-validation-status.json",
        "handoff_evidence/package-smoke.json",
        "handoff_evidence/hard-scan-classification.json",
        "handoff_evidence/test-checks.json",
        "handoff_evidence/source-tree-scan.json",
        "handoff_evidence/verification-matrix.json",
        "handoff_evidence/smoke-compare-status.json",
        "handoff_evidence/subagent-review-findings.json",
        "handoff_evidence/subagent-review-reconciliation.json",
    }
)
HARD_SCAN_CLASSIFICATIONS = frozenset(
    {
        "actual_violation",
        "allowed_generated_evidence",
        "negative_test_fixture",
        "existing_base_issue",
        "false_positive",
        "valuable_deferred_feature_no_change",
        "test_white_box_allowed",
    }
)
SUBAGENT_REVIEW_SEVERITIES = frozenset({"P0", "P1", "P2", "P3", "INFO"})
SUBAGENT_REVIEW_CLASSIFICATIONS = frozenset(
    {
        "true_issue_to_fix",
        "false_positive",
        "valuable_deferred_feature_no_change",
        "existing_base_issue_no_change",
        "blocked_approval",
        "no_findings",
    }
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


def hard_scan_receipt_from_clone(
    clone_dir: Path,
    *,
    final_head: str,
) -> dict[str, object]:
    scanned_head = _governance_git_text(clone_dir, "rev-parse", "HEAD").strip()
    scan_groups = source_tree_scan_groups(clone_dir)
    scan_runs: list[dict[str, object]] = []
    classified_exceptions: list[dict[str, object]] = []
    for group_name, group_payload in sorted(scan_groups.items()):
        findings = list(group_payload["findings"])
        actual_violations = len(findings)
        result_payload = {
            "scan_id": group_name,
            "findings": findings,
            "finding_count": len(findings),
        }
        scan_runs.append(
            {
                "scan_id": group_name,
                "rule_set_id": f"source-tree:{group_name}",
                "match_count": len(findings),
                "actual_violation_count": actual_violations,
                "deferred_count": 0,
                "negative_fixture_count": 0,
                "false_positive_count": 0,
                "result_sha256": _sha256_text(_stable_governance_json(result_payload)),
                "replayable_from_bundle": True,
            }
        )
        for finding in findings:
            if isinstance(finding, Mapping):
                classified_exceptions.append(
                    {
                        "scan_id": group_name,
                        "classification": "actual_violation",
                        "owner": str(finding.get("path") or group_name),
                        "rationale": str(finding.get("reason") or finding.get("rule")),
                        "reference": "governance_contract.py",
                    }
                )
    return {
        "schema_id": "qrp-hard-scan-receipt",
        "status": "PASS"
        if scanned_head == final_head
        and all(run["actual_violation_count"] == 0 for run in scan_runs)
        else "FAIL",
        "scanned_head": scanned_head,
        "scanner_source_sha256": sha256_file(Path(__file__)),
        "scan_runs": scan_runs,
        "classified_exceptions": classified_exceptions,
    }


def hard_scan_receipt_from_bundle(
    *,
    bundle_path: Path,
    branch: str,
    final_head: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="handoff-hard-scan-") as temp:
        clone_dir = Path(temp) / "repo"
        subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle_path), str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return hard_scan_receipt_from_clone(clone_dir, final_head=final_head)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_tree_scan_groups(clone_dir: Path) -> dict[str, dict[str, object]]:
    tracked_paths = source_tree_tracked_paths(clone_dir)
    tracked_text_files = source_tree_tracked_text_files_from_paths(tracked_paths)
    retired_text_findings = source_tree_retired_text_findings(
        clone_dir,
        tracked_text_files=tracked_text_files,
    )
    retired_governance_path_findings_ = source_tree_retired_governance_path_findings(
        clone_dir,
        tracked_paths=tracked_paths,
    )
    retired_path_name_findings = source_tree_retired_path_name_findings(
        clone_dir,
        tracked_paths=tracked_paths,
    )
    python_symbol_findings = source_tree_python_symbol_findings(
        clone_dir,
        tracked_text_files=tracked_text_files,
    )
    synthesized_string_findings = source_tree_synthesized_string_scan_findings(
        clone_dir,
        tracked_text_files=tracked_text_files,
    )
    active_test_literal_findings = source_tree_active_test_literal_findings(
        clone_dir,
        tracked_text_files=tracked_text_files,
    )
    structured_key_findings = source_tree_structured_key_findings(
        clone_dir,
        tracked_text_files=tracked_text_files,
    )
    governance_truth_findings = source_tree_governance_truth_findings(
        clone_dir,
        tracked_text_files=tracked_text_files,
    )
    sensitive_negative_literal_findings = (
        source_tree_sensitive_negative_literal_findings(
            clone_dir,
            tracked_text_files=tracked_text_files,
        )
    )
    allowance_contract_findings = source_tree_allowance_contract_findings(clone_dir)
    return {
        "allowance_contract": {
            "status": "PASS" if not allowance_contract_findings else "FAIL",
            "finding_count": len(allowance_contract_findings),
            "findings": allowance_contract_findings,
        },
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


def source_tree_allowance_contract_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    tracked_paths = {
        _source_tree_posix_path(path) for path in source_tree_tracked_paths(clone_dir)
    }
    governance_allowlist_owner_path = Path(
        "tests/architecture/test_governance_contract_rg_gate.py"
    )
    enforce_missing_allowance_entries = (
        clone_dir / governance_allowlist_owner_path
    ).is_file()
    patterns = source_tree_allowed_occurrence_patterns(repo_root=clone_dir)
    for failure in validate_source_tree_allowed_occurrences(
        SOURCE_TREE_ALLOWED_OCCURRENCES,
        patterns_by_rule_id=patterns,
    ):
        findings.append(
            {
                "path": "",
                "line": 0,
                "rule": "source_tree_allowed_occurrence_contract",
                "reason": failure,
            }
        )
    for entry in SOURCE_TREE_ALLOWED_OCCURRENCES:
        path = clone_dir / Path(entry.path.as_posix())
        if entry.path not in tracked_paths or not path.exists():
            if not enforce_missing_allowance_entries:
                continue
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": 0,
                    "rule": entry.rule_id,
                    "reason": "source_tree_allowed_occurrence_stale",
                }
            )
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        owner_ranges = source_tree_python_owner_ranges(path)
        if entry.py_owner:
            ranges = owner_ranges.get(entry.py_owner, ())
            if len(ranges) > 1:
                findings.append(
                    {
                        "path": entry.path.as_posix(),
                        "line": 0,
                        "rule": entry.rule_id,
                        "reason": "source_tree_allowed_occurrence_ambiguous_owner",
                    }
                )
                continue
            if len(ranges) == 0:
                findings.append(
                    {
                        "path": entry.path.as_posix(),
                        "line": 0,
                        "rule": entry.rule_id,
                        "reason": "source_tree_allowed_occurrence_stale",
                    }
                )
                continue
        exact_lines = {
            line_number
            for line_number, line in enumerate(lines, start=1)
            if line.strip() == entry.exact_line
        }
        if entry.py_owner and entry.py_owner in owner_ranges:
            exact_lines &= set(owner_ranges[entry.py_owner][0])
        if not exact_lines:
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": 0,
                    "rule": entry.rule_id,
                    "reason": "source_tree_allowed_occurrence_stale",
                }
            )
            continue
        if len(exact_lines) > 1:
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": min(exact_lines),
                    "rule": entry.rule_id,
                    "reason": "source_tree_allowed_occurrence_overmatched",
                }
            )
            continue
        unreachable_reason = source_tree_allowed_occurrence_unreachable_reason(
            clone_dir,
            entry,
            path,
            lines,
            next(iter(exact_lines)),
            owner_ranges,
        )
        if unreachable_reason:
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": next(iter(exact_lines)),
                    "rule": entry.rule_id,
                    "reason": unreachable_reason,
                }
            )
    return findings


def source_tree_allowed_occurrence_unreachable_reason(
    clone_dir: Path,
    entry: SourceTreeAllowedOccurrence,
    path: Path,
    lines: list[str],
    line_number: int,
    owner_ranges: dict[str, tuple[frozenset[int], ...]],
) -> str:
    if ":" not in entry.rule_id:
        return "source_tree_allowed_occurrence_unreachable"
    scan_group, rule = entry.rule_id.split(":", 1)
    relative = Path(entry.path.as_posix())
    line_text = lines[line_number - 1].strip()
    if scan_group == "retired_text":
        if rule not in source_tree_retired_text_patterns_for(relative):
            return "source_tree_allowed_occurrence_unreachable"
        return ""
    if scan_group == "synthesized_strings":
        if entry.path.suffix != ".py":
            return "source_tree_allowed_occurrence_unreachable"
        source = path.read_text(encoding="utf-8", errors="ignore")
        pattern = source_tree_forbidden_patterns().get(rule)
        if pattern is None:
            return "source_tree_allowed_occurrence_unreachable"
        candidates = iter_synthesized_string_findings(
            relative,
            source,
            patterns={rule: pattern},
            skip_rule_definition=False,
        )
        if not any(
            int(candidate.get("line", 0)) == line_number
            and candidate.get("rule") == rule
            for candidate in candidates
        ):
            return "source_tree_allowed_occurrence_unreachable"
        matched_entries, ambiguous_entries, stale_owner_entries = (
            _source_tree_matching_allowed_occurrence_entries(
                relative,
                scan_group=scan_group,
                rule=rule,
                line_text=line_text,
                line_number=line_number,
                owner_ranges=owner_ranges,
            )
        )
        if ambiguous_entries or stale_owner_entries or not matched_entries:
            return "source_tree_allowed_occurrence_unreachable"
        return ""
    if scan_group == "active_test_literals":
        if relative.parts[:1] != ("tests",) and relative != Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "build_handoff_package.py"
        ):
            return "source_tree_allowed_occurrence_unreachable"
        if rule not in source_tree_active_test_literal_patterns(clone_dir):
            return "source_tree_allowed_occurrence_unreachable"
        return ""
    if scan_group == "sensitive_negative_literals":
        if relative.parts[:1] not in {(".agents",), ("tests",)}:
            return "source_tree_allowed_occurrence_unreachable"
        if rule not in source_tree_sensitive_negative_patterns():
            return "source_tree_allowed_occurrence_unreachable"
        return ""
    return "source_tree_allowed_occurrence_unreachable"


def source_tree_retired_governance_path_findings(
    clone_dir: Path,
    *,
    tracked_paths: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    paths = (
        tracked_paths
        if tracked_paths is not None
        else source_tree_tracked_paths(clone_dir)
    )
    tracked = tuple(path.as_posix() for path in paths)
    return retired_governance_path_findings(clone_dir, tracked_paths=tracked)


def source_tree_retired_path_name_findings(
    clone_dir: Path,
    *,
    tracked_paths: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    paths = (
        tracked_paths
        if tracked_paths is not None
        else source_tree_tracked_paths(clone_dir)
    )
    return iter_path_name_findings(relative.as_posix() for relative in paths)


def source_tree_python_symbol_findings(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    for relative in files:
        if relative.suffix.lower() != ".py":
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        findings.extend(iter_python_symbol_findings(relative, source))
    return findings


def source_tree_structured_key_findings(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    for relative in files:
        if relative.suffix.lower() not in {".json", ".toml", ".yaml", ".yml"}:
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings.extend(iter_structured_key_findings(relative, text))
    return findings


def source_tree_retired_text_findings(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    known_patterns = source_tree_forbidden_patterns()
    scanned_files = {_source_tree_posix_path(path) for path in files}
    occurrence_entries = {
        entry: 0
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
        if (
            entry.path in scanned_files
            and entry.rule_id.startswith("retired_text:")
            and entry.rule_id.split(":", 1)[1] in known_patterns
        )
    }
    for relative in files:
        patterns = source_tree_retired_text_patterns_for(relative)
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        owner_ranges = source_tree_python_owner_ranges(path)
        for rule, pattern in patterns.items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                line_text = line.strip()
                if _source_tree_retired_rule_definition_allowed(
                    relative,
                    rule=rule,
                    line_text=line_text,
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                matched_entries = _source_tree_matching_allowed_occurrence_entries(
                    relative,
                    scan_group="retired_text",
                    rule=rule,
                    line_text=line_text,
                    line_number=line_number,
                    owner_ranges=owner_ranges,
                )
                matched_entries, ambiguous_entries, stale_owner_entries = (
                    matched_entries
                )
                if ambiguous_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_ambiguous_owner",
                        }
                    )
                    continue
                if stale_owner_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_stale",
                        }
                    )
                    continue
                available_entry = next(
                    (
                        entry
                        for entry in matched_entries
                        if occurrence_entries.get(entry, 0) < 1
                    ),
                    None,
                )
                if available_entry is not None:
                    occurrence_entries[available_entry] += 1
                    continue
                if matched_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_overmatched",
                        }
                    )
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
    for entry, count in occurrence_entries.items():
        if count:
            continue
        findings.append(
            {
                "path": entry.path.as_posix(),
                "line": 0,
                "rule": entry.rule_id.split(":", 1)[1],
                "reason": "source_tree_allowed_occurrence_stale",
            }
        )
    return findings


def source_tree_should_scan_retired_text(relative: Path) -> bool:
    return any(
        relative == root or root in relative.parents
        for root in SOURCE_TREE_RETIRED_TERM_ROOTS
    )


def source_tree_retired_text_patterns_for(
    relative: Path,
) -> dict[str, re.Pattern[str]]:
    if source_tree_should_scan_retired_text(relative):
        return source_tree_forbidden_patterns()
    patterns = dict(profile_runner_retired_patterns())
    if relative in SOURCE_TREE_RETIRED_TEXT_NEGATIVE_TEST_SOURCES:
        patterns.update(source_tree_forbidden_patterns())
    return patterns


def source_tree_synthesized_string_scan_findings(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    for relative in files:
        if relative.suffix.lower() != ".py":
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
                None,
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
        None,
        scan_group="synthesized_strings",
        patterns=source_tree_forbidden_patterns(),
    )


def source_tree_synthesized_string_findings(
    relative: Path,
    path: Path,
    lines: list[str],
    skip_lines: set[int],
    rule_definition_lines: set[int],
    definition_contexts: dict[str, set[int]] | None = None,
    *,
    scan_group: str,
    patterns: dict[str, re.Pattern[str]],
    occurrence_entries: dict[SourceTreeAllowedOccurrence, int] | None = None,
    report_stale: bool = True,
    scan_modes: frozenset[str] | None = None,
) -> list[dict[str, object]]:
    if path.suffix != ".py":
        return []
    source = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[dict[str, object]] = []
    owner_ranges = source_tree_python_owner_ranges(path)
    if occurrence_entries is None:
        relative_key = _source_tree_posix_path(relative)
        occurrence_entries = {
            entry: 0
            for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
            if entry.path == relative_key and entry.rule_id.startswith(f"{scan_group}:")
        }
    for item in iter_synthesized_string_findings(
        relative,
        source,
        patterns=patterns,
        allow_negative_tests=False,
        skip_rule_definition=False,
    ):
        if scan_modes is not None and str(item["scan_mode"]) not in scan_modes:
            continue
        line_number = int(item["line"])
        if line_number in skip_lines:
            continue
        line_text = lines[line_number - 1].strip() if line_number else ""
        matched_entries, ambiguous_entries, stale_owner_entries = (
            _source_tree_matching_allowed_occurrence_entries(
                relative,
                scan_group=scan_group,
                rule=str(item["rule"]),
                line_text=line_text,
                line_number=line_number,
                owner_ranges=owner_ranges,
            )
        )
        if ambiguous_entries:
            findings.append(
                {
                    **item,
                    "path": relative.as_posix(),
                    "reason": "source_tree_allowed_occurrence_ambiguous_owner",
                }
            )
            continue
        if stale_owner_entries:
            findings.append(
                {
                    **item,
                    "path": relative.as_posix(),
                    "reason": "source_tree_allowed_occurrence_stale",
                }
            )
            continue
        available_entry = next(
            (
                entry
                for entry in matched_entries
                if occurrence_entries.get(entry, 0) < 1
            ),
            None,
        )
        if available_entry is not None:
            occurrence_entries[available_entry] += 1
            continue
        if matched_entries:
            findings.append(
                {
                    **item,
                    "path": relative.as_posix(),
                    "reason": "source_tree_allowed_occurrence_overmatched",
                }
            )
            continue
        if _source_tree_retired_rule_definition_allowed(
            relative,
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
    if report_stale:
        for entry, count in occurrence_entries.items():
            if count:
                continue
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": 0,
                    "rule": entry.rule_id.split(":", 1)[1],
                    "reason": "source_tree_allowed_occurrence_stale",
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
        decorator_lines = [
            getattr(decorator, "lineno", 0)
            for decorator in getattr(node, "decorator_list", ())
        ]
        start = min(
            [line for line in decorator_lines if line] or [getattr(node, "lineno", 0)]
        )
        decorator_end_lines = [
            getattr(decorator, "end_lineno", getattr(decorator, "lineno", 0))
            for decorator in getattr(node, "decorator_list", ())
        ]
        end = max(
            [line for line in decorator_end_lines if line]
            + [getattr(node, "end_lineno", start)]
        )
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


def source_tree_python_owner_ranges(
    path: Path,
) -> dict[str, tuple[frozenset[int], ...]]:
    if path.suffix != ".py" or not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return {}
    ranges: dict[str, list[frozenset[int]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = f"def:{node.name}"
        elif isinstance(node, ast.ClassDef):
            key = f"class:{node.name}"
        else:
            continue
        decorator_lines = [
            getattr(decorator, "lineno", 0)
            for decorator in getattr(node, "decorator_list", ())
        ]
        start = min(
            [line for line in decorator_lines if line] or [getattr(node, "lineno", 0)]
        )
        decorator_end_lines = [
            getattr(decorator, "end_lineno", getattr(decorator, "lineno", 0))
            for decorator in getattr(node, "decorator_list", ())
        ]
        end = max(
            [line for line in decorator_end_lines if line]
            + [getattr(node, "end_lineno", start)]
        )
        if start and end:
            ranges.setdefault(key, []).append(frozenset(range(start, end + 1)))
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = source_tree_definition_name(node)
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if name and start and end:
            ranges.setdefault(f"assign:{name}", []).append(
                frozenset(range(start, end + 1))
            )
    return {key: tuple(value) for key, value in ranges.items()}


def source_tree_allowlist_definition_lines(
    relative: Path,
    lines: list[str],
) -> set[int]:
    block_names = set(SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES.get(relative, set()))
    if not block_names:
        return set()
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError:
        return set()
    result: set[int] = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = source_tree_definition_name(node)
        if name not in block_names:
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if start and end:
            result.update(range(start, end + 1))
    return result


def source_tree_active_test_literal_findings(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    for relative in files:
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
        owner_ranges = source_tree_python_owner_ranges(path)
        relative_key = _source_tree_posix_path(relative)
        occurrence_entries = {
            entry: 0
            for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
            if entry.path == relative_key
            and entry.rule_id.startswith("active_test_literals:")
        }
        active_patterns = source_tree_active_test_literal_patterns(clone_dir)
        for rule, pattern in active_patterns.items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                line_text = line.strip()
                matched_entries, ambiguous_entries, stale_owner_entries = (
                    _source_tree_matching_allowed_occurrence_entries(
                        relative,
                        scan_group="active_test_literals",
                        rule=rule,
                        line_text=line_text,
                        line_number=line_number,
                        owner_ranges=owner_ranges,
                    )
                )
                if ambiguous_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_ambiguous_owner",
                        }
                    )
                    continue
                if stale_owner_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_stale",
                        }
                    )
                    continue
                available_entry = next(
                    (
                        entry
                        for entry in matched_entries
                        if occurrence_entries.get(entry, 0) < 1
                    ),
                    None,
                )
                if available_entry is not None:
                    occurrence_entries[available_entry] += 1
                    continue
                if matched_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_overmatched",
                        }
                    )
                    continue
                if _source_tree_retired_rule_definition_allowed(
                    relative,
                    rule=rule,
                    line_text=line_text,
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
                None,
                scan_group="active_test_literals",
                patterns=active_patterns,
                occurrence_entries=occurrence_entries,
                report_stale=False,
                scan_modes=frozenset({"ast_static_string", "token_static_string"}),
            )
        )
        for entry, count in occurrence_entries.items():
            if count:
                continue
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": 0,
                    "rule": entry.rule_id.split(":", 1)[1],
                    "reason": "source_tree_allowed_occurrence_stale",
                }
            )
    return findings


def source_tree_governance_truth_findings(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_active_governance_doc_files(
        clone_dir,
        tracked_text_files=tracked_text_files,
    ):
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


def source_tree_active_governance_doc_files(
    clone_dir: Path,
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> tuple[Path, ...]:
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    tracked = set(files)
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
    try:
        from quant_research_platform.contracts.data import (
            DataFrequency,
            data_frequency_slug,
        )
        from quant_research_platform.markets.implementations.a_share.ids import (
            market_context_data_id,
            market_panel_data_id,
        )

        market_bars_data_id = discover_market_bars_data_id_helper(repo_root)
    except (ModuleNotFoundError, RuntimeError):
        if repo_root is not None:
            return {}
        raise
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
    primary_src_root = ensure_runtime_src_path(repo_root)
    script_dir = Path(__file__).resolve().parent
    candidate_src_roots = [primary_src_root]
    candidate_src_roots.extend(
        parent / "src"
        for parent in script_dir.parents
        if (parent / "src" / "quant_research_platform").is_dir()
    )
    for src_root in dict.fromkeys(candidate_src_roots):
        source_root = src_root / "quant_research_platform" / "sources"
        for path in sorted(source_root.rglob("ids.py")) if source_root.exists() else ():
            if "market_bars_data_id" not in path.read_text(
                encoding="utf-8",
                errors="ignore",
            ):
                continue
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))
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
    *,
    tracked_text_files: tuple[Path, ...] | None = None,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    files = (
        tracked_text_files
        if tracked_text_files is not None
        else source_tree_tracked_text_files(clone_dir)
    )
    for relative in files:
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
        owner_ranges = source_tree_python_owner_ranges(path)
        relative_key = _source_tree_posix_path(relative)
        occurrence_entries = {
            entry: 0
            for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
            if entry.path == relative_key
            and entry.rule_id.startswith("sensitive_negative_literals:")
        }
        sensitive_patterns = source_tree_sensitive_negative_patterns()
        for rule, pattern in sensitive_patterns.items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                line_text = line.strip()
                matched_entries, ambiguous_entries, stale_owner_entries = (
                    _source_tree_matching_allowed_occurrence_entries(
                        relative,
                        scan_group="sensitive_negative_literals",
                        rule=rule,
                        line_text=line_text,
                        line_number=line_number,
                        owner_ranges=owner_ranges,
                    )
                )
                if ambiguous_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_ambiguous_owner",
                        }
                    )
                    continue
                if stale_owner_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_stale",
                        }
                    )
                    continue
                available_entry = next(
                    (
                        entry
                        for entry in matched_entries
                        if occurrence_entries.get(entry, 0) < 1
                    ),
                    None,
                )
                if available_entry is not None:
                    occurrence_entries[available_entry] += 1
                    continue
                if matched_entries:
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule": rule,
                            "reason": "source_tree_allowed_occurrence_overmatched",
                        }
                    )
                    continue
                if _source_tree_retired_rule_definition_allowed(
                    relative,
                    rule=rule,
                    line_text=line_text,
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
                None,
                scan_group="sensitive_negative_literals",
                patterns=sensitive_patterns,
                occurrence_entries=occurrence_entries,
                report_stale=False,
                scan_modes=frozenset({"ast_static_string", "token_static_string"}),
            )
        )
        for entry, count in occurrence_entries.items():
            if count:
                continue
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": 0,
                    "rule": entry.rule_id.split(":", 1)[1],
                    "reason": "source_tree_allowed_occurrence_stale",
                }
            )
    return findings


def source_tree_tracked_text_files(clone_dir: Path) -> tuple[Path, ...]:
    return source_tree_tracked_text_files_from_paths(
        source_tree_tracked_paths(clone_dir)
    )


def source_tree_tracked_text_files_from_paths(
    tracked_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    result: list[Path] = []
    for relative in tracked_paths:
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
    definition_contexts: dict[str, set[int]] | None = None,
) -> bool:
    in_governance_rule_definition = (
        line_number is not None
        and rule_definition_lines is not None
        and line_number in rule_definition_lines
    )
    if in_governance_rule_definition and definition_contexts is not None:
        allowed_blocks = SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES.get(relative, set())
        in_allowed_definition_owner = False
        for owner, lines in definition_contexts.items():
            owner_name = owner.split(":", 1)[1] if ":" in owner else owner
            if owner_name in allowed_blocks and line_number in lines:
                in_allowed_definition_owner = True
                break
        in_governance_rule_definition = in_allowed_definition_owner
    if scan_group == "sensitive_negative_literals":
        return False
    if scan_group in {"synthesized_strings", "active_test_literals"}:
        return False
    if scan_group == "retired_text":
        if rule is None or line_text is None:
            return False
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line_text):
            if is_allowed_runtime_symbol(relative, token, rule):
                return True
        if is_retired_path_policy_line(relative, line_text):
            return True
        allowed_rules = SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST.get(
            relative,
            set(),
        )
        return rule in allowed_rules and in_governance_rule_definition
    raise ValueError(f"unsupported_source_tree_scan_group: {scan_group}")


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
