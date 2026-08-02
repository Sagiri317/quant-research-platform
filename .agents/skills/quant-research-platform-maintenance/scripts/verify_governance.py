#!/usr/bin/env python3
"""Verify the active QRP governance scope and sensitive-text boundary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[3]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from governance_contract import (  # noqa: E402
    ACTIVE_GOVERNANCE_DOC_ROOTS,
    build_source_tree_scan_input,
    source_tree_sensitive_negative_literal_findings,
)

from quant_research_platform.contracts._json import (  # noqa: E402
    render_qrp_json_object,
)

BASE_GOVERNANCE_PREFIXES = ("AGENTS.md", "README.md", ".agents/")
ACTIVE_DOCS_PREFIXES = (
    "docs/architecture/",
    "docs/code-map/",
    "docs/data/",
    "data/README.md",
)
ACTIVE_DOC_SCAN_ROOTS = ACTIVE_GOVERNANCE_DOC_ROOTS
GOVERNANCE_TEST_PREFIXES = ("tests/agents/", "tests/architecture/")
_GOVERNANCE_PATH_EXCEPTIONS = {
    ".gitignore",
    "pyproject.toml",
    "tests/policy/test_share_safety_runtime_edges.py",
    "tests/tools/run_profile.py",
    "tests/tools/test_run_profile.py",
}
LOCAL_ONLY_ROOTS = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "data/research/",
    "outputs/",
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
TEXT_SUFFIXES = {"", ".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml"}


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
        code, out, err = run_git(
            repo,
            ["status", "--short", "--untracked-files=all"],
        )
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
        elif root.is_dir():
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


def is_governance_only_allowed_path(repo: Path, path: str) -> bool:
    del repo
    normalized = normalize(path)
    return (
        any(
            normalized == prefix or normalized.startswith(prefix)
            for prefix in BASE_GOVERNANCE_PREFIXES + ACTIVE_DOCS_PREFIXES
        )
        or normalized.startswith(GOVERNANCE_TEST_PREFIXES)
        or normalized in _GOVERNANCE_PATH_EXCEPTIONS
    )


def check_governance_only(result: Result, repo: Path, paths: list[str]) -> None:
    for path in paths:
        normalized = normalize(path)
        if normalized.startswith(LOCAL_ONLY_ROOTS):
            result.errors.append(f"local-only root changed or included: {normalized}")
        if not is_governance_only_allowed_path(repo, normalized):
            result.errors.append(
                f"path is outside governance-only allow list: {normalized}"
            )
        if Path(normalized).suffix.lower() in BLOCKED_EXTENSIONS:
            result.errors.append(f"blocked extension in changed path: {normalized}")


def iter_text_files(repo: Path, paths: list[str]) -> list[Path]:
    return [
        repo / name
        for name in paths
        if (repo / name).is_file() and (repo / name).suffix.lower() in TEXT_SUFFIXES
    ]


def check_text_safety(result: Result, repo: Path, paths: list[str]) -> None:
    text_paths = iter_text_files(repo, paths)
    relative_paths = tuple(path.relative_to(repo) for path in text_paths)
    result.checked_files.extend(sorted(normalize(str(path)) for path in relative_paths))
    scan_input = build_source_tree_scan_input(repo, relative_paths)
    for finding in source_tree_sensitive_negative_literal_findings(scan_input):
        reason = finding.get("reason")
        detail = f":{reason}" if reason else ""
        result.errors.append(
            "sensitive_text: "
            f"{finding['path']}:{finding['line']}:{finding['rule']}{detail}"
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
    changed: list[str] = []
    if scan_scope == "changed-files":
        try:
            changed = changed_files(repo, args.base_ref)
        except RuntimeError as exc:
            result.errors.append(str(exc))
        paths = changed
    else:
        paths = active_doc_files(repo)
    check_scan_non_empty(result, paths, scan_scope=scan_scope)
    if args.governance_only:
        check_governance_only(result, repo, paths)
    check_text_safety(result, repo, paths)
    check_checked_files_non_empty(result, scan_scope=scan_scope)

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
        print(render_qrp_json_object(summary))
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
