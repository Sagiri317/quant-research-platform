#!/usr/bin/env python3
"""Verify QRP governance contracts and retired surfaces."""

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
    GOVERNANCE_TEST_EXCEPTION_PATHS,
    LOCAL_PATH_TEXT_SAFETY_FIXTURE_SOURCES,
    RETIRED_GOVERNANCE_EXACT_PATHS,
    RETIRED_GOVERNANCE_PREFIXES,
    SOURCE_TREE_ALLOWED_OCCURRENCES,
    STALE_GOVERNANCE_TRUTH_TEST_SOURCES,
    active_doc_forbidden_patterns,
    git_tracked_paths,
    is_allowed_stale_or_retired_fixture,
    is_governance_rule_definition_path,
    is_retired_governance_path,
    is_scanner_definition_line,
    iter_path_name_findings,
    iter_python_symbol_findings,
    iter_structured_key_findings,
    retired_governance_path_findings,
    source_tree_allowlist_definition_lines,
    source_tree_forbidden_patterns,
    source_tree_governance_rule_definition_lines,
    source_tree_python_owner_ranges,
    source_tree_synthesized_string_findings,
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
)

GOVERNANCE_TEST_PREFIXES = (
    "tests/agents/",
    "tests/architecture/",
    "tests/contracts/",
    "tests/interfaces/",
    "tests/reports/",
)

GOVERNANCE_CONFIG_EXCEPTIONS = {
    ".gitignore",
    "pyproject.toml",
}

GOVERNANCE_TOOLING_EXCEPTIONS = {
    "tests/tools/run_profile.py",
    "tests/tools/test_run_profile.py",
}

GOVERNANCE_TEST_EXCEPTIONS = set(GOVERNANCE_TEST_EXCEPTION_PATHS)

REPORTS_EVIDENCE_RENDERER_EXCEPTIONS = {
    "src/quant_research_platform/reports/verification_summary.py",
}

GOVERNANCE_ONLY_SOURCE_TEXT_EXCEPTIONS = {
    "src/quant_research_platform/interfaces/cli/commands.py",
    "src/quant_research_platform/markets/implementations/a_share/verification/"
    "aggregates.py",
    "src/quant_research_platform/sources/csmar/interim.py",
}

RETIRED_DOC_DELETION_PATHS = set(RETIRED_GOVERNANCE_EXACT_PATHS)

RETIRED_DOC_DELETION_PREFIXES = RETIRED_GOVERNANCE_PREFIXES

LOCAL_ONLY_ROOTS = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "data/research/",
    "outputs/",
)

STRUCTURED_KEY_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}

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
    ("private safety", "private data", "绉佹湁"),
    ("current-contract-only",),
)

ACTIVE_DOC_FORBIDDEN_PATTERNS = active_doc_forbidden_patterns()

LOCAL_PATH_PATTERN = re.compile(
    r"((?<![A-Za-z])[A-Za-z]:[\\/]|/Users/[^\s`]+|/home/[^\s`]+|/mnt/(?:data|[a-z])/[^\s`]+)"
)  # noqa: E501
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|cookie)\s*[:=]\s*['\"]?[^\s'\"]+"
)

TEXT_SUFFIXES = {"", ".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml"}

IGNORED_STATE_PATH_PARTS = {
    ".git",
    ".run",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
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
        status_code, status_out, _ = run_git(
            repo,
            ["status", "--short", "--untracked-files=all"],
        )
        if status_code == 0:
            names.extend(_paths_from_status_short(status_out))
    else:
        code, out, err = run_git(repo, ["status", "--short", "--untracked-files=all"])
        if code != 0:
            raise RuntimeError(f"git status failed: {err.strip()}")
        names = _paths_from_status_short(out)
    return sorted({normalize(name) for name in names if name})


def _paths_from_status_short(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        names.append(path)
    return names


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
        if path.startswith(LOCAL_ONLY_ROOTS):
            result.errors.append(f"local-only root changed or included: {path}")
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
                f"business/dependency path changed in governance-only mode: {path}"
            )
        if not allowed:
            result.errors.append(f"path is outside governance-only allow list: {path}")
        if Path(
            path
        ).suffix.lower() in BLOCKED_EXTENSIONS and not is_retired_doc_deletion(
            repo, path
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
    if path in GOVERNANCE_CONFIG_EXCEPTIONS:
        return True
    if path in GOVERNANCE_TOOLING_EXCEPTIONS:
        return True
    if path in GOVERNANCE_TEST_EXCEPTIONS:
        return True
    if path in REPORTS_EVIDENCE_RENDERER_EXCEPTIONS:
        return True
    if path in GOVERNANCE_ONLY_SOURCE_TEXT_EXCEPTIONS:
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


def repo_state_file_paths(
    repo: Path,
    *,
    suffixes: set[str] | None = None,
    result: Result | None = None,
) -> list[str]:
    found: set[str] = set()
    for path in git_tracked_paths(repo):
        if _should_include_state_path(path, suffixes=suffixes):
            found.add(path)
    if repo.exists():
        for path in repo.rglob("*"):
            if not path.is_file():
                continue
            rel = normalize(str(path.relative_to(repo)))
            if _should_include_state_path(rel, suffixes=suffixes):
                found.add(rel)
    return sorted(found)


def _should_include_state_path(
    path: str,
    *,
    suffixes: set[str] | None,
) -> bool:
    normalized = normalize(path)
    parts = Path(normalized).parts
    if any(part in IGNORED_STATE_PATH_PARTS for part in parts):
        return False
    if any(
        normalized == root.rstrip("/") or normalized.startswith(root)
        for root in LOCAL_ONLY_ROOTS
    ):
        return False
    if suffixes is not None and Path(normalized).suffix.lower() not in suffixes:
        return False
    return True


def check_forbidden_path_names(result: Result, repo: Path) -> None:
    for finding in iter_path_name_findings(repo_state_file_paths(repo)):
        result.errors.append(f"retired_path_name: {finding['path']}:{finding['rule']}")


def check_python_symbol_names(
    result: Result,
    repo: Path,
    paths: list[str],
) -> None:
    for rel in sorted({normalize(path) for path in paths}):
        if Path(rel).suffix.lower() != ".py":
            continue
        path = repo / rel
        if not path.is_file():
            continue
        if is_governance_rule_definition_path(rel):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            result.warnings.append(
                f"cannot decode python for symbol scan: {rel}: {exc}"
            )
            continue
        for finding in iter_python_symbol_findings(rel, source):
            result.errors.append(
                "retired_python_symbol: "
                f"{finding['path']}:{finding['line']}:"
                f"{finding['symbol_kind']}:{finding['symbol']}:{finding['rule']}"
            )


def check_synthesized_strings(
    result: Result,
    repo: Path,
    paths: list[str],
) -> None:
    patterns = source_tree_forbidden_patterns()
    for rel in sorted({normalize(path) for path in paths}):
        if Path(rel).suffix.lower() != ".py":
            continue
        path = repo / rel
        if not path.is_file():
            continue
        if is_governance_rule_definition_path(rel):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.warnings.append(f"cannot decode python for string scan: {rel}")
            continue
        lines = source.splitlines()
        relative = Path(rel)
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        owner_ranges = source_tree_python_owner_ranges(path)
        for finding in source_tree_synthesized_string_findings(
            relative,
            path,
            tuple(lines),
            frozenset(skip_lines),
            frozenset(rule_definition_lines),
            source=source,
            owner_ranges=owner_ranges,
            scan_group="synthesized_strings",
            patterns=patterns,
        ):
            finding_detail = finding.get("scan_mode") or finding.get("reason")
            finding_detail = finding_detail or "unknown"
            result.errors.append(
                "retired_synthesized_string: "
                f"{finding['path']}:{finding['line']}:"
                f"{finding['rule']}:{finding_detail}"
            )


def check_structured_key_names(
    result: Result,
    repo: Path,
    paths: list[str],
) -> None:
    for rel in sorted({normalize(path) for path in paths}):
        if Path(rel).suffix.lower() not in STRUCTURED_KEY_SUFFIXES:
            continue
        path = repo / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.warnings.append(f"cannot decode structured file: {rel}")
            continue
        for finding in iter_structured_key_findings(rel, text):
            result.errors.append(
                "retired_structured_key: "
                f"{finding['path']}:{finding['line']}:"
                f"{finding['key_path']}:{finding['key']}:{finding['rule']}"
            )


def iter_text_files(repo: Path, paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for name in paths:
        p = repo / name
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            out.append(p)
    return out


def _skip_regex_definition_line(rel: str, line: str) -> bool:
    return is_scanner_definition_line(rel, line)


def _is_allowed_local_path_fixture(rel: str, line: str) -> bool:
    if (
        rel in LOCAL_PATH_TEXT_SAFETY_FIXTURE_SOURCES
        and LOCAL_PATH_PATTERN.search(line)
    ):
        return True
    return False


def _is_allowed_secret_fixture(rel: str, line: str) -> bool:
    if not SECRET_PATTERN.search(line):
        return False
    return any(
        occurrence.path.as_posix() == rel
        and occurrence.rule_id.endswith(":sensitive_secret")
        and occurrence.exact_line == line.strip()
        for occurrence in SOURCE_TREE_ALLOWED_OCCURRENCES
    )


def _is_allowed_stale_fixture(rel: str, line: str) -> bool:
    return is_allowed_stale_or_retired_fixture(rel, line)


def _should_scan_stale_governance_truth(rel: str) -> bool:
    if rel.startswith(("src/", "tests/")):
        return rel in STALE_GOVERNANCE_TRUTH_TEST_SOURCES
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
                for rule, pattern in ACTIVE_DOC_FORBIDDEN_PATTERNS.items():
                    if pattern.search(line) and not _is_allowed_stale_fixture(
                        rel,
                        line,
                    ):
                        result.errors.append(
                            "governance forbidden wording in "
                            f"{rel}:{line_number}: {rule}: {pattern.pattern}"
                        )
            if (
                "NOT_RUN" in line
                and "PASS" in line
                and "涓嶈兘" not in line
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
                "AGENTS.md missing required term variant: " + " / ".join(variants)
            )


def check_script_portability(result: Result, repo: Path) -> None:
    script_root = (
        repo / ".agents" / "skills" / "quant-research-platform-maintenance" / "scripts"
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
        r"`((?:AGENTS\.md|README\.md|\.agents/|docs/|src/|tests/|data/|reports/)[^`\s]*)`"
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
                for root in LOCAL_ONLY_ROOTS
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
            "checked and local-only roots are not scanned."
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
        help="Governance-only mode forbids business code and local-only roots.",
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
    check_forbidden_path_names(result, repo)
    state_python_files = repo_state_file_paths(repo, suffixes={".py"}, result=result)
    check_python_symbol_names(result, repo, state_python_files)
    check_synthesized_strings(result, repo, state_python_files)
    state_structured_files = repo_state_file_paths(
        repo,
        suffixes=STRUCTURED_KEY_SUFFIXES,
        result=result,
    )
    check_structured_key_names(result, repo, state_structured_files)
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
        "schema_id": "qrp-governance-scan-result",
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
