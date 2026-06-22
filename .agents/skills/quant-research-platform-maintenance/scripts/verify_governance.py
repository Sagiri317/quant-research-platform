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
from dataclasses import dataclass, field
from pathlib import Path

BASE_GOVERNANCE_PREFIXES = (
    "AGENTS.md",
    "README.md",
    "PROJECT_STATUS.md",
    ".agents/",
    "docs/architecture/",
)

ACTIVE_DOCS_PREFIXES = (
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
    "reports/private_research/",
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
    ("PASS",),
    ("NOT_RUN",),
    ("Private-data safety", "私有数据安全"),
    ("global `.codex/`", "全局 `.codex/`", "global .codex"),
    ("repo-local `.codex/`", "仓库内 `.codex/`", "repo-local .codex"),
)

STALE_STATUS_PATTERNS = (
    re.compile(r"当前分支\s*\|\s*`feature/n2-closure`"),
    re.compile(r"current branch\s*[:|].*feature/n2-closure", re.IGNORECASE),
    re.compile(r"Finish the official review-fix handoff package", re.IGNORECASE),
)

LOCAL_PATH_PATTERN = re.compile(
    r"((?<![A-Za-z])[A-Za-z]:[\\/]|/Users/[^\s`]+|/home/[^\s`]+"
    r"|/mnt/(?:data|[a-z])/[^\s`]+)"
)
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|cookie)\s*[:=]\s*['\"]?[^\s'\"]+"
)

TEXT_SUFFIXES = {".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml"}


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


def allowed_governance_prefixes(allow_active_docs: bool) -> tuple[str, ...]:
    return BASE_GOVERNANCE_PREFIXES + (
        ACTIVE_DOCS_PREFIXES if allow_active_docs else ()
    )


def is_governance_path(path: str, allow_active_docs: bool) -> bool:
    return any(
        path == prefix or path.startswith(prefix)
        for prefix in allowed_governance_prefixes(allow_active_docs)
    )


def check_governance_only(
    result: Result,
    paths: list[str],
    allow_active_docs: bool,
) -> None:
    for path in paths:
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
        if path.startswith(BUSINESS_PREFIXES) and not is_governance_path(
            path,
            allow_active_docs,
        ):
            result.errors.append(
                "business/dependency path changed in governance-only mode: "
                f"{path}"
            )
        if not is_governance_path(path, allow_active_docs):
            result.errors.append(f"path is outside governance-only allow list: {path}")
        if Path(path).suffix.lower() in BLOCKED_EXTENSIONS:
            result.errors.append(f"blocked extension in changed path: {path}")


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
        "E:/quant-workspace",
        "/mnt/data",
        "/home/user",
    )
    if any(marker in line for marker in markers):
        return True
    return rel.endswith(".py") and line.lstrip().startswith(("r\"", "r'"))


def _is_allowed_local_path_fixture(rel: str, line: str) -> bool:
    allowed_fixture_paths = {
        "tests/agents/handoff/test_handoff_tools.py",
        "tests/agents/handoff/test_package_safety_check.py",
    }
    if rel in allowed_fixture_paths and any(
        marker in line
        for marker in ("E:/quant-workspace", "/mnt/data", "/home/user")
    ):
        return True
    public_safe_local_path_markers = (
        '"C:/"',
        r'"C:\\',
        '"/mnt/"',
        '"/home/"',
        '"/Users/"',
    )
    if rel == "tests/architecture/test_public_safe_evidence.py" and any(
        marker in line for marker in public_safe_local_path_markers
    ):
        return True
    if rel in {
        "tests/interfaces/test_api_cli_01_contract.py",
        "tests/storage/test_storage_01.py",
    } and any(
        marker in line for marker in (r"C:\Users", "/mnt/data")
    ):
        return True
    return (
        rel == "docs/architecture/open-issues-register.md"
        and "Negative tests cover" in line
    )


def _is_allowed_secret_fixture(rel: str, line: str) -> bool:
    if rel == "tests/architecture/test_public_safe_evidence.py" and '"token="' in line:
        return True
    if rel in {
        "tests/interfaces/test_api_cli_01_contract.py",
        "tests/storage/test_storage_01.py",
    } and '"token"' in line and "token=" in line:
        return True
    return False


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
            for match in STALE_STATUS_PATTERNS:
                if match.search(line):
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
    if "small-fix" not in text or "standard-implementation" not in text:
        result.errors.append("AGENTS.md must use canonical hyphenated task mode IDs")


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
        "--allow-active-docs",
        action="store_true",
        help="Allow active docs in governance-only mode.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line summary for final reports.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    result = Result()
    try:
        paths = changed_files(repo, args.base_ref)
    except RuntimeError as exc:
        result.errors.append(str(exc))
        paths = []

    if args.governance_only:
        check_governance_only(result, paths, args.allow_active_docs)
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
    elif args.summary:
        print(
            f"governance_check_status={summary['status']} "
            f"changed_files={len(paths)} "
            f"checked_files={len(result.checked_files)} "
            f"errors={len(result.errors)} "
            f"warnings={len(result.warnings)}"
        )
    else:
        print(f"governance_check_status={summary['status']}")
        print(f"changed_files={len(paths)}")
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
