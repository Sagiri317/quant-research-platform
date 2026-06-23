#!/usr/bin/env python3
"""QRP 治理层检查脚本。

只使用 Python standard library，目标兼容 Win11、macOS、Linux、WSL。
默认只检查 Git changed files，不扫描 private data roots。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from governance_contract import (  # noqa: E402
    ACTIVE_GOVERNANCE_DOC_ROOTS,
    PACKAGE_FORBIDDEN_PATH_PARTS,
    RETIRED_GOVERNANCE_EXACT_PATHS,
    RETIRED_GOVERNANCE_PREFIXES,
    is_governance_rule_definition_path,
    is_negative_test_allowlisted,
    is_retired_governance_path,
    retired_governance_path_findings,
    stale_governance_truth_patterns,
)

BASE_GOVERNANCE_PREFIXES = (
    "AGENTS.md",
    "README.md",
    ".agents/",
)

ACTIVE_DOCS_PREFIXES = (
    "docs/architecture/",
    "docs/code-map/",
    "docs/api/",
    "docs/data/",
    "data/README.md",
)

ACTIVE_DOC_SCAN_ROOTS = ACTIVE_GOVERNANCE_DOC_ROOTS

BUSINESS_PREFIXES = (
    "src/",
    "tests/",
    "pyproject.toml",
    "environment.yml",
    "data/",
    "reports/private_research/",
)

GOVERNANCE_TEST_PREFIXES = (
    "tests/agents/",
    "tests/architecture/",
    "tests/reports/",
)

REPORTS_EVIDENCE_RENDERER_EXCEPTIONS = {
    "src/quant_research_platform/reports/verification_evidence.py",
}

RETIRED_DOC_DELETION_PATHS = set(RETIRED_GOVERNANCE_EXACT_PATHS)

RETIRED_DOC_DELETION_PREFIXES = RETIRED_GOVERNANCE_PREFIXES
RETIRED_PATH_POLICY_MARKERS = tuple(RETIRED_GOVERNANCE_EXACT_PATHS) + (
    "docs/reports/clean-rewrite",
)

PRIVATE_ROOTS = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "data/research/",
    "reports/private_research/",
)

BLOCKED_EXTENSIONS = {
    ".parquet",
    ".csv",
    ".pkl",
    ".pickle",
    ".feather",
    ".arrow",
    ".zip",
    ".bundle",
}

REQUIRED_AGENTS_TERMS = (
    (".agents/",),
    (".run/",),
    ("PASS",),
    ("NOT_RUN",),
    ("private safety", "private data", "私有"),
    ("current-contract-only",),
)

STALE_STATUS_PATTERNS = tuple(stale_governance_truth_patterns().values())

LOCAL_PATH_PATTERN = re.compile(r"((?<![A-Za-z])[A-Za-z]:[\\/]|/Users/[^\s`]+|/home/[^\s`]+|/mnt/(?:data|[a-z])/[^\s`]+)")  # noqa: E501
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|cookie)\s*[:=]\s*['\"]?[^\s'\"]+"
)

TEXT_SUFFIXES = {".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml"}
PRIVACY_NEGATIVE_TEST_PATHS = {
    "tests/runtime/test_privacy_01.py",
    "tests/agents/handoff/test_handoff_tools.py",
    "tests/agents/handoff/test_package_safety_check.py",
}
STALE_NEGATIVE_TEST_PATHS = {
    "tests/agents/test_verify_governance.py",
    "tests/architecture/test_clean_rewrite_rg_gate.py",
}


@dataclass
class Result:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def run_git(repo: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def normalize(path: str) -> str:
    return path.replace("\\", "/")


def changed_files(repo: Path, base_ref: str | None) -> list[str]:
    if base_ref:
        code, out, err = run_git(repo, ["diff", "--name-only", f"{base_ref}...HEAD"])
        if code != 0:
            raise RuntimeError(f"git diff against {base_ref} failed: {err.strip()}")
        names = out.splitlines()
        for args in (["diff", "--name-only", "--cached"], ["diff", "--name-only"]):
            sub_code, sub_out, _ = run_git(repo, args)
            if sub_code == 0:
                names.extend(sub_out.splitlines())
    else:
        code, out, err = run_git(repo, ["status", "--short"])
        if code != 0:
            raise RuntimeError(f"git status failed: {err.strip()}")
        names = []
        for line in out.splitlines():
            if not line.strip():
                continue
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            names.append(path)
    return sorted({normalize(name) for name in names if name})


def active_doc_files(repo: Path) -> list[str]:
    files: list[str] = []
    for root_name in ACTIVE_DOC_SCAN_ROOTS:
        root = repo / root_name
        if root.is_file():
            files.append(normalize(root_name))
            continue
        if root.is_dir():
            files.extend(
                normalize(str(path.relative_to(repo)))
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
            )
    return sorted(set(files))


def check_scan_non_empty(
    result: Result,
    paths: list[str],
    *,
    scan_scope: str,
) -> None:
    if not paths:
        result.errors.append(f"checked_files_empty: {scan_scope}")


def check_checked_files_non_empty(result: Result, *, scan_scope: str) -> None:
    if not result.checked_files:
        result.errors.append(f"checked_files_empty: {scan_scope}")


def allowed_governance_prefixes() -> tuple[str, ...]:
    return BASE_GOVERNANCE_PREFIXES + ACTIVE_DOCS_PREFIXES


def is_governance_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(prefix)
        for prefix in allowed_governance_prefixes()
    )


def check_governance_only(
    result: Result,
    repo: Path,
    paths: list[str],
) -> None:
    for path in paths:
        allowed = is_governance_only_allowed_path(repo, path)
        if path.startswith(PRIVATE_ROOTS):
            result.errors.append(f"private root changed or included: {path}")
        if path.startswith(".codex/"):
            result.errors.append(
                f"repo-local .codex change is not allowed by default: {path}"
            )
        if path.startswith(".claude/") or path == "CLAUDE.md":
            result.errors.append(
                f"agent-specific project adapter must remain draft-only: {path}"
            )
        if path.startswith(BUSINESS_PREFIXES) and not allowed:
            result.errors.append(
                "business/dependency path changed in governance-only mode: "
                f"{path}"
            )
        if not allowed:
            result.errors.append(f"path is outside governance-only allow list: {path}")
        if (
            Path(path).suffix.lower() in BLOCKED_EXTENSIONS
            and not is_retired_doc_deletion(repo, path)
        ):
            result.errors.append(f"blocked extension in changed path: {path}")


def is_governance_only_allowed_path(
    repo: Path,
    path: str,
) -> bool:
    if is_governance_path(path):
        return True
    if path.startswith(GOVERNANCE_TEST_PREFIXES):
        return True
    if path in REPORTS_EVIDENCE_RENDERER_EXCEPTIONS:
        return True
    return is_retired_doc_deletion(repo, path)


def is_retired_doc_deletion(repo: Path, path: str) -> bool:
    if not is_retired_governance_path(path):
        return False
    return not (repo / path).exists()


def check_retired_governance_paths(result: Result, repo: Path) -> None:
    for finding in retired_governance_path_findings(repo):
        result.errors.append(
            f"retired_governance_path_exists: {finding['path']}:{finding['rule']}"
        )


def iter_text_files(repo: Path, paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for name in paths:
        p = repo / name
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            out.append(p)
    return out


def _skip_regex_definition_line(rel: str, line: str) -> bool:
    markers = (
        "SECRET_PATTERN",
        "LOCAL_PATH_PATTERN",
        "STALE_STATUS_PATTERNS",
        "STALE_STATUS_MARKERS",
        "STOCK_SAMPLE_PATTERN",
        "re.compile(",
        "token = re.compile",
        "secret-like assignment",
        "Finish the official review-fix handoff package",
    )
    if any(marker in line for marker in markers):
        return True
    if rel.startswith(".agents/skills/quant-research-platform-maintenance/scripts/"):
        scanner_markers = (
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
            "merge_ready",
            "MergeReady",
            "mergeReady",
            "pre_merge",
            "PreMerge",
            "preMerge",
            "draft_pr",
            "DraftPr",
            "draftPr",
            "E:/",
            "D:/",
            "C:/",
            "\\Users\\",
            "\\\\Users\\\\",
            "/Users/",
            "/home/",
            "/mnt/",
            "/mnt/data",
        )
        if any(marker in line for marker in scanner_markers):
            return True
    return rel.endswith(".py") and line.lstrip().startswith(("r\"", "r'"))


def _is_allowed_local_path_fixture(rel: str, line: str) -> bool:
    if rel in PRIVACY_NEGATIVE_TEST_PATHS and LOCAL_PATH_PATTERN.search(line):
        return True
    if (
        rel == "tests/architecture/test_public_safe_evidence.py"
        and LOCAL_PATH_PATTERN.search(line)
    ):
        return True
    if rel in {
        "tests/interfaces/test_api_cli_01_contract.py",
        "tests/storage/test_storage_01.py",
    } and LOCAL_PATH_PATTERN.search(line):
        return True
    return False


def _is_allowed_secret_fixture(rel: str, line: str) -> bool:
    if rel in PRIVACY_NEGATIVE_TEST_PATHS and SECRET_PATTERN.search(line):
        return True
    if rel == "tests/architecture/test_public_safe_evidence.py" and '"token="' in line:
        return True
    if rel in {
        "tests/interfaces/test_api_cli_01_contract.py",
        "tests/storage/test_storage_01.py",
    } and '"token"' in line and "token=" in line:
        return True
    return False


def _is_allowed_stale_fixture(rel: str, line: str) -> bool:
    if is_governance_rule_definition_path(rel):
        return True
    if is_negative_test_allowlisted(rel, "retired_path_policy_docs") and any(
        marker in line for marker in RETIRED_PATH_POLICY_MARKERS
    ):
        return True
    return (
        rel in STALE_NEGATIVE_TEST_PATHS
        or is_negative_test_allowlisted(rel, "stale_governance_truth")
        or is_negative_test_allowlisted(rel, "retired_name")
    ) and any(
        marker in line
        for marker in (
            "feature/qrp-clean-rewrite",
            "feature/agent-governance-redesign",
            "clean rewrite",
            "clean-rewrite",
            "draft PR",
            "pre-merge",
            "merge-ready",
            ".run official",
            "official .run",
            "PROJECT_STATUS",
            "current-qrp-status",
            "project_status",
            "ProjectStatus",
            "project-status",
            "PROJECT_STATUS.md",
            "open-issues-register",
            "open_issues_register",
            "OpenIssuesRegister",
            "openIssuesRegister",
            "current_qrp_status",
            "CurrentQrpStatus",
            "currentQrpStatus",
            "clean_rewrite",
            "CleanRewrite",
            "cleanRewrite",
            "merge_ready",
            "MergeReady",
            "pre_merge",
            "PreMerge",
            "draft_pr",
            "DraftPr",
        )
    )


def _should_scan_stale_governance_truth(rel: str) -> bool:
    if rel.startswith(("src/", "tests/")):
        return rel in STALE_NEGATIVE_TEST_PATHS or is_negative_test_allowlisted(
            rel,
            "stale_governance_truth",
        )
    if rel.startswith(".agents/skills/quant-research-platform-maintenance/scripts/"):
        return False
    return True


def check_text_safety(result: Result, repo: Path, paths: list[str]) -> None:
    for p in iter_text_files(repo, paths):
        rel = normalize(str(p.relative_to(repo)))
        result.checked_files.append(rel)
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.warnings.append(f"cannot decode as utf-8: {rel}")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _skip_regex_definition_line(rel, line):
                continue
            if SECRET_PATTERN.search(line) and not _is_allowed_secret_fixture(
                rel,
                line,
            ):
                result.errors.append(
                    f"possible secret-like assignment in {rel}:{line_number}"
                )
            if LOCAL_PATH_PATTERN.search(line) and not _is_allowed_local_path_fixture(
                rel,
                line,
            ):
                result.errors.append(
                    f"local absolute path pattern in {rel}:{line_number}"
                )
            if _should_scan_stale_governance_truth(rel):
                for match in STALE_STATUS_PATTERNS:
                    if match.search(line) and not _is_allowed_stale_fixture(rel, line):
                        result.errors.append(
                            "stale N2/status wording in "
                            f"{rel}:{line_number}: {match.pattern}"
                        )
            if (
                "NOT_RUN" in line
                and "PASS" in line
                and "不能" not in line
                and "not" not in line.lower()
                and re.search(
                    r"\b(status|result)\s*[:=]\s*`?PASS",
                    line,
                    re.IGNORECASE,
                )
            ):
                result.warnings.append(
                    f"line mixes NOT_RUN and PASS in {rel}:{line_number}"
                )


def check_agents_size_and_terms(result: Result, repo: Path) -> None:
    agents = repo / "AGENTS.md"
    if not agents.exists():
        result.errors.append("AGENTS.md missing")
        return
    text = agents.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > 12_000:
        result.errors.append("AGENTS.md exceeds 12KB target")
    for variants in REQUIRED_AGENTS_TERMS:
        if not any(term in text for term in variants):
            result.errors.append(
                "AGENTS.md missing required term variant: "
                + " / ".join(variants)
            )


def check_script_portability(result: Result, repo: Path) -> None:
    script_root = (
        repo
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
    )
    if not script_root.exists():
        result.warnings.append("script directory missing")
        return
    for p in script_root.rglob("*"):
        rel = normalize(str(p.relative_to(repo)))
        if rel.endswith("/governance_contract.py"):
            continue
        if p.is_file() and p.suffix.lower() == ".ps1":
            result.errors.append(f"ps1-only script remains in governance skill: {rel}")
        if p.is_file() and p.suffix.lower() in {".sh", ".bash"}:
            result.errors.append(
                f"shell-only script remains in governance skill: {rel}"
            )
        if p.is_file() and p.suffix.lower() == ".py":
            text = p.read_text(encoding="utf-8")
            if "argparse" not in text:
                result.warnings.append(
                    f"python script has no argparse help surface: {rel}"
                )
            if "pathlib" not in text and "Path" not in text:
                result.warnings.append(f"python script may not use pathlib: {rel}")
            if re.search(r"(?<![A-Za-z])[A-Za-z]:\\", text):
                result.errors.append(f"hardcoded Windows drive path in {rel}")


def check_markdown_paths(result: Result, repo: Path, paths: list[str]) -> None:
    token = re.compile(
        r"`((?:AGENTS\.md|README\.md|PROJECT_STATUS\.md|\.agents/|docs/|src/|tests/|data/|reports/)[^`\s]*)`"
    )
    for p in iter_text_files(repo, paths):
        if p.suffix.lower() != ".md":
            continue
        rel = normalize(str(p.relative_to(repo)))
        text = p.read_text(encoding="utf-8")
        for raw in token.findall(text):
            candidate = raw.rstrip(".,;:)")
            if "<" in candidate or "*" in candidate or candidate.endswith("/"):
                continue
            if any(
                candidate == root.rstrip("/") or candidate.startswith(root)
                for root in PRIVATE_ROOTS
            ):
                continue
            if is_retired_governance_path(candidate):
                continue
            if not (repo / candidate).exists():
                src_candidate = repo / "src" / "quant_research_platform" / candidate
                if src_candidate.exists():
                    continue
                result.warnings.append(
                    f"possible broken repo path in {rel}: {candidate}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run QRP governance checks. By default, only Git changed files are "
            "checked and private data roots are not scanned."
        )
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Optional base ref for committed diff checks.",
    )
    parser.add_argument(
        "--governance-only",
        action="store_true",
        help="Governance-only mode forbids business code and private roots.",
    )
    parser.add_argument(
        "--scan-scope",
        choices=("active-docs", "changed-files"),
        default=None,
        help=(
            "Required with --governance-only. active-docs scans current active "
            "governance docs from repository state; changed-files scans Git diff "
            "and worktree changes."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line summary for final reports.",
    )
    args = parser.parse_args(argv)
    if args.governance_only and args.scan_scope is None:
        parser.error("--governance-only requires --scan-scope")

    repo = Path(args.repo_root).resolve()
    result = Result()
    scan_scope = args.scan_scope or "changed-files"
    check_retired_governance_paths(result, repo)
    changed: list[str] = []
    if scan_scope == "changed-files":
        try:
            changed = changed_files(repo, args.base_ref)
        except RuntimeError as exc:
            result.errors.append(str(exc))
            changed = []
        paths = changed
    else:
        paths = active_doc_files(repo)
    check_scan_non_empty(result, paths, scan_scope=scan_scope)

    if args.governance_only:
        check_governance_only(result, repo, paths)
    check_text_safety(result, repo, paths)
    check_checked_files_non_empty(result, scan_scope=scan_scope)
    check_agents_size_and_terms(result, repo)
    check_script_portability(result, repo)
    check_markdown_paths(result, repo, paths)

    summary = {
        "status": "PASS" if result.ok else "FAIL",
        "scan_scope": scan_scope,
        "changed_files": changed,
        "checked_files": result.checked_files,
        "checked_count": len(result.checked_files),
        "scanned_files": paths,
        "findings": result.errors,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.summary:
        print(
            f"governance_check_status={summary['status']} "
            f"scan_scope={summary['scan_scope']} "
            f"changed_files={len(changed)} "
            f"checked_files={len(result.checked_files)} "
            f"errors={len(result.errors)} "
            f"warnings={len(result.warnings)}"
        )
    else:
        print(f"governance_check_status={summary['status']}")
        print(f"scan_scope={summary['scan_scope']}")
        print(f"changed_files={len(changed)}")
        print(f"checked_files={len(result.checked_files)}")
        print(f"errors={len(result.errors)}")
        print(f"warnings={len(result.warnings)}")
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
