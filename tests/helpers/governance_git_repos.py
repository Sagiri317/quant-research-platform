from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


def init_governance_git_repo(repo: Path, files: Mapping[str, str]) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _write_files(repo, files)
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "governance@example.invalid")
    run_git(repo, "config", "user.name", "Governance Test")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "base")
    return run_git(repo, "rev-parse", "HEAD").stdout.strip()


def run_governance_script(
    repo: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _write_files(repo: Path, files: Mapping[str, str]) -> None:
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
