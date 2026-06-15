#!/usr/bin/env python3
"""Cross-platform governance checks for QRP agent-governance changes.

This script intentionally uses only Python standard library modules. It is safe to
run on Windows, macOS, Linux, and WSL. It does not scan private data directories
by default.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

GOVERNANCE_PREFIXES = (
    "AGENTS.md",
    "README.md",
    "PROJECT_STATUS.md",
    ".agents/",
    "docs/architecture/",
    "docs/api/README.md",
    "docs/data/README.md",
    "docs/code-map/",
)

BUSINESS_PREFIXES = (
    "src/",
    "tests/",
    "pyproject.toml",
    "environment.yml",
    "data/",
    "reports/private_real/",
)

PRIVATE_ROOTS = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "data/research/",
    "reports/private_real/",
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
    ".agents/",
    "PASS",
    "NOT_RUN",
    "Private-data safety",
)

STALE_STATUS_PATTERNS = (
    re.compile(r"当前分支\s*\|\s*`feature/n2-closure`"),
    re.compile(r"current branch\s*[:|].*feature/n2-closure", re.IGNORECASE),
    re.compile(r"Finish the official review-fix handoff package", re.IGNORECASE),
)

LOCAL_PATH_PATTERN = re.compile(
    r"([A-Za-z]:\\|/Users/[^\s`]+|/home/[^\s`]+|/mnt/[a-z]/[^\s`]+)"
)
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|cookie)\s*[:=]\s*['\"]?[^\s'\"]+"
)

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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
        staged_code, staged_out, _ = run_git(repo, ["diff", "--name-only", "--cached"])
        work_code, work_out, _ = run_git(repo, ["diff", "--name-only"])
        names = out.splitlines()
        if staged_code == 0:
            names.extend(staged_out.splitlines())
        if work_code == 0:
            names.extend(work_out.splitlines())
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


def is_governance_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in GOVERNANCE_PREFIXES)


def check_governance_only(result: Result, paths: list[str]) -> None:
    for path in paths:
        if path.startswith(PRIVATE_ROOTS):
            result.errors.append(f"private root changed or included: {path}")
        if path.startswith(".codex/"):
            result.errors.append(f"repo-local .codex change is not allowed by default: {path}")
        if path.startswith(".claude/") or path == "CLAUDE.md":
            result.errors.append(f"Claude-specific project adapter must remain draft-only: {path}")
        if path.startswith(BUSINESS_PREFIXES) and not is_governance_path(path):
            result.errors.append(f"business/dependency path changed in governance-only mode: {path}")
        if Path(path).suffix.lower() in BLOCKED_EXTENSIONS:
            result.errors.append(f"blocked extension in changed path: {path}")


def iter_text_files(repo: Path, paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for name in paths:
        p = repo / name
        if p.is_file() and p.suffix.lower() in {".md", ".py", ".txt", ".yaml", ".yml", ".json"}:
            out.append(p)
    return out


def check_text_safety(result: Result, repo: Path, paths: list[str]) -> None:
    for p in iter_text_files(repo, paths):
        rel = normalize(str(p.relative_to(repo)))
        result.checked_files.append(rel)
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.warnings.append(f"cannot decode as utf-8: {rel}")
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            # Pattern definition lines intentionally contain words like token,
            # password, /Users, or historical stale phrases. Do not treat the
            # checker's own regex constants as repository content findings.
            if any(name in line for name in (
                "SECRET_PATTERN",
                "LOCAL_PATH_PATTERN",
                "STALE_STATUS_PATTERNS",
                "STALE_STATUS_MARKERS",
                "STOCK_SAMPLE_PATTERN",
            )):
                continue
            if rel.endswith(".py") and (
                "re.compile(" in line
                or line.lstrip().startswith(('r"', "r'", "token = re.compile"))
                or "token = re.compile" in line
                or "secret-like assignment" in line
                or "Finish the official review-fix handoff package" in line
            ):
                continue
            if SECRET_PATTERN.search(line):
                result.errors.append(f"possible secret-like assignment in {rel}:{i}")
            if LOCAL_PATH_PATTERN.search(line):
                result.errors.append(f"local absolute path pattern in {rel}:{i}")
            for match in STALE_STATUS_PATTERNS:
                if match.search(line):
                    result.errors.append(f"stale N2/status wording in {rel}:{i}: {match.pattern}")
            upper = line.upper()
            if (
                "NOT_RUN" in upper
                and "PASS" in upper
                and "NOT PASS" not in upper
                and re.search(r"\b(status|result)\s*[:=]\s*`?PASS", line, re.IGNORECASE)
            ):
                result.warnings.append(f"line mixes NOT_RUN and PASS in {rel}:{i}")


def check_agents_size_and_terms(result: Result, repo: Path) -> None:
    agents = repo / "AGENTS.md"
    if not agents.exists():
        result.errors.append("AGENTS.md missing")
        return
    text = agents.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > 12_000:
        result.errors.append("AGENTS.md exceeds 12KB target")
    lower = text.lower()
    if "global" not in lower or ".codex/" not in lower:
        result.errors.append("AGENTS.md must distinguish global .codex")
    if "repo-local" not in lower or ".codex/" not in lower:
        result.errors.append("AGENTS.md must state repo-local .codex is not default")
    for term in REQUIRED_AGENTS_TERMS:
        if term not in text:
            result.errors.append(f"AGENTS.md missing required term: {term}")


def check_script_portability(result: Result, repo: Path) -> None:
    script_root = repo / ".agents" / "skills" / "quant-research-platform-maintenance" / "scripts"
    if not script_root.exists():
        result.warnings.append("script directory missing")
        return
    for p in script_root.rglob("*"):
        rel = normalize(str(p.relative_to(repo)))
        if p.is_file() and p.suffix.lower() == ".ps1":
            result.errors.append(f"ps1-only script remains in governance skill: {rel}")
        if p.is_file() and p.suffix.lower() in {".sh", ".bash"}:
            result.errors.append(f"shell-only script remains in governance skill: {rel}")
        if p.is_file() and p.suffix.lower() == ".py":
            text = p.read_text(encoding="utf-8")
            if "argparse" not in text:
                result.warnings.append(f"python script has no argparse help surface: {rel}")
            if "pathlib" not in text and "Path" not in text:
                result.warnings.append(f"python script may not use pathlib: {rel}")
            if re.search(r"[A-Za-z]:\\", text):
                result.errors.append(f"hardcoded Windows drive path in {rel}")


def check_markdown_paths(result: Result, repo: Path, paths: list[str]) -> None:
    token = re.compile(r"`((?:AGENTS\.md|README\.md|PROJECT_STATUS\.md|\.agents/|docs/|src/|tests/|data/|reports/)[^`\s]*)`")
    for p in iter_text_files(repo, paths):
        if p.suffix.lower() != ".md":
            continue
        rel = normalize(str(p.relative_to(repo)))
        text = p.read_text(encoding="utf-8")
        for raw in token.findall(text):
            candidate = raw.rstrip(".,;:)")
            if "<" in candidate or "*" in candidate or candidate.endswith("/"):
                continue
            if candidate.startswith(("data/raw", "data/interim", "data/processed", "data/research", "reports/private_real")):
                continue
            if not (repo / candidate).exists():
                if candidate.startswith("reports/") and (repo / "src/quant_research_platform" / candidate).exists():
                    continue
                result.warnings.append(f"possible broken repo path in {rel}: {candidate}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QRP governance checks without scanning private data.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--base-ref", default=None, help="Optional base ref for committed diff checks, e.g. origin/feature/qrp-clean-rewrite.")
    parser.add_argument("--governance-only", action="store_true", help="Fail if business code, dependency files, private roots, or repo-local agent adapters changed.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    result = Result()
    try:
        paths = changed_files(repo, args.base_ref)
    except RuntimeError as exc:
        result.errors.append(str(exc))
        paths = []

    if args.governance_only:
        check_governance_only(result, paths)
    check_text_safety(result, repo, paths)
    check_agents_size_and_terms(result, repo)
    check_script_portability(result, repo)
    check_markdown_paths(result, repo, paths)

    summary = {
        "status": "PASS" if result.ok else "FAIL",
        "changed_files": paths,
        "checked_files": result.checked_files,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"governance_check_status={summary['status']}")
        print(f"changed_files={len(paths)}")
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
