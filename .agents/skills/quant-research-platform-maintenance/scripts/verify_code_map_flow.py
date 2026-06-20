#!/usr/bin/env python3
"""Verify that the QRP platform-flow code map has current anchors."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED_TERMS = (
    "QRP Clean Rewrite",
    "0.1.0",
    "/api/0.1",
    "research-datasets",
    "workflows/",
    "reports/private_research",
)
FORBIDDEN_TERMS = (
    "raw" + "_private",
    "interim" + "_private",
    "processed" + "_private",
    "/api/" + "v0.1",
    "/api/" + "pipeline-runs",
    "/api/" + "runs",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify current QRP platform-flow paths and interface anchors."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--markdown-path",
        default="docs/code-map/platform-flow.md",
        help="Markdown file to verify, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--skip-git-diff-check",
        action="store_true",
        help="Skip git diff whitespace check for the markdown file.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    markdown_path = Path(args.markdown_path)
    if not markdown_path.is_absolute():
        markdown_path = repo_root / markdown_path

    failures: list[str] = []
    if not markdown_path.is_file():
        failures.append(f"missing file: {markdown_path}")
    else:
        text = markdown_path.read_text(encoding="utf-8")
        missing = [term for term in REQUIRED_TERMS if term not in text]
        forbidden = [term for term in FORBIDDEN_TERMS if term in text]
        failures.extend(f"missing required term: {term}" for term in missing)
        failures.extend(f"forbidden term found: {term}" for term in forbidden)

    if not args.skip_git_diff_check and markdown_path.exists():
        rel = markdown_path.resolve().relative_to(repo_root).as_posix()
        result = subprocess.run(
            ["git", "diff", "--check", "--", rel],
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            failures.append("git diff --check failed for code map")
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.stderr.strip():
                print(result.stderr.strip())

    for item in failures:
        print(f"FAIL: {item}")
    if failures:
        print("code_map_flow_status=FAIL")
        return 1
    print("code_map_flow_status=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
