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
    ".agents/",
)

ACTIVE_DOCS_PREFIXES = (
    "docs/architecture/",
    "docs/code-map/",
    "docs/api/",
    "docs/data/",
    "data/README.md",
)

ACTIVE_DOC_SCAN_ROOTS = (
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

RETIRED_DOC_DELETION_PATHS = {
    "PROJECT_STATUS.md",
    ".agents/briefs/current-qrp-status.md",
    "docs/architecture/open-issues-register.md",
}

RETIRED_DOC_DELETION_PREFIXES = (
    "docs/reports/clean-rewrite-0.1-",
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

STALE_STATUS_PATTERNS = (
    re.compile(r"feature/(?:qrp-clean-rewrite|agent-governance-redesign)"),
    re.compile(r"\bclean[- ]rewrite\b", re.IGNORECASE),
    re.compile(r"\b(?:draft PR|pre-merge|merge-ready)\b", re.IGNORECASE),
    re.compile(r"(?:\.run\s+official|official\s+\.run)", re.IGNORECASE),
    re.compile(r"PROJECT_STATUS\.md.*(?:active|status|source)", re.IGNORECASE),
    re.compile(r"current-qrp-status\.md.*(?:active|status|source)", re.IGNORECASE),
)

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
    retired_fragments = (
        "PROJECT_STATUS.md",
        "current-qrp-status.md",
        "docs/reports/clean-rewrite-0.1-",
    )
    return sorted(
        {
            path
            for path in files
            if not any(fragment in path for fragment in retired_fragments)
        }
    )


def check_active_doc_scan_non_empty(result: Result, paths: list[str]) -> None:
    if not paths:
        result.errors.append("active-doc scan found no files")


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
    repo: Path,
    paths: list[str],
    allow_active_docs: bool,
) -> None:
    for path in paths:
        allowed = is_governance_only_allowed_path(repo, path, allow_active_docs)
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
    allow_active_docs: bool,
) -> bool:
    if is_governance_path(path, allow_active_docs):
        return True
    if path.startswith(GOVERNANCE_TEST_PREFIXES):
        return True
    if path in REPORTS_EVIDENCE_RENDERER_EXCEPTIONS:
        return True
    return is_retired_doc_deletion(repo, path)


def is_retired_doc_deletion(repo: Path, path: str) -> bool:
    if not (
        path in RETIRED_DOC_DELETION_PATHS
        or path.startswith(RETIRED_DOC_DELETION_PREFIXES)
    ):
        return False
    return not (repo / path).exists()


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
            "current-qrp-status.md",
            "docs/reports/clean-rewrite-0.1-",
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
    return rel in STALE_NEGATIVE_TEST_PATHS and any(
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
        )
    )


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
    parser.add_argument(
        "--scan-active-docs",
        action="store_true",
        help=(
            "Scan current active governance docs from repository state instead "
            "of only Git changed files."
        ),
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
        changed = changed_files(repo, args.base_ref)
    except RuntimeError as exc:
        result.errors.append(str(exc))
        changed = []

    paths = active_doc_files(repo) if args.scan_active_docs else changed
    if args.scan_active_docs:
        check_active_doc_scan_non_empty(result, paths)

    if args.governance_only:
        check_governance_only(result, repo, paths, args.allow_active_docs)
    check_text_safety(result, repo, paths)
    check_agents_size_and_terms(result, repo)
    check_script_portability(result, repo)
    check_markdown_paths(result, repo, paths)

    summary = {
        "status": "PASS" if result.ok else "FAIL",
        "scan_mode": "active_docs" if args.scan_active_docs else "changed_files",
        "changed_files": changed,
        "checked_files": result.checked_files,
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
            f"scan_mode={summary['scan_mode']} "
            f"changed_files={len(changed)} "
            f"checked_files={len(result.checked_files)} "
            f"errors={len(result.errors)} "
            f"warnings={len(result.warnings)}"
        )
    else:
        print(f"governance_check_status={summary['status']}")
        print(f"scan_mode={summary['scan_mode']}")
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
