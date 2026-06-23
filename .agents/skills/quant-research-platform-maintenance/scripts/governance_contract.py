"""Shared QRP governance contract rules for repository maintenance scripts.

This module is an internal tooling fact source. It is not a public API.
"""

from __future__ import annotations

import re
import subprocess
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

NEGATIVE_TEST_ALLOWLIST = {
    "stale_governance_truth": (
        "tests/agents/test_verify_governance.py",
        "tests/architecture/test_clean_rewrite_rg_gate.py",
    ),
    "retired_name": (
        "tests/agents/test_verify_governance.py",
        "tests/architecture/test_clean_rewrite_rg_gate.py",
        "tests/architecture/test_handoff_package_contract.py",
    ),
    "governance_rule_definition": (
        ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py",
    ),
    "retired_path_policy_docs": (
        "AGENTS.md",
        ".agents/skills/quant-research-platform-maintenance/references/verification-and-gates.md",
    ),
}

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


def retired_name_patterns() -> dict[str, re.Pattern[str]]:
    return name_rule_patterns(RETIRED_NAME_RULES + RETIRED_PUBLIC_SURFACE_RULES)


def stale_governance_truth_patterns() -> dict[str, re.Pattern[str]]:
    patterns = compile_text_rules(STALE_GOVERNANCE_TRUTH_RULES)
    patterns.update(name_rule_patterns(RETIRED_NAME_RULES))
    return patterns


def retired_public_surface_patterns() -> dict[str, re.Pattern[str]]:
    return name_rule_patterns(RETIRED_PUBLIC_SURFACE_RULES)


def _bounded_literal_pattern(form: str) -> str:
    escaped = re.escape(form)
    if form.startswith("--"):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_-])"
    if form.endswith(".md"):
        return rf"(?<![A-Za-z0-9_./-]){escaped}(?![A-Za-z0-9_./-])"
    return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"


def is_negative_test_allowlisted(path: str | Path, category: str) -> bool:
    normalized = normalize_repo_path(path)
    return normalized in NEGATIVE_TEST_ALLOWLIST.get(category, ())


def is_governance_rule_definition_path(path: str | Path) -> bool:
    return is_negative_test_allowlisted(path, "governance_rule_definition")
