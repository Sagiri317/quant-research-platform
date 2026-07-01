#!/usr/bin/env python3
"""Run one command and append a compact review-safe check receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

PYTEST_COUNT_KEYS = (
    "passed",
    "failed",
    "skipped",
    "deselected",
    "errors",
    "xfailed",
    "xpassed",
    "collected",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, dest="command_id")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required after --")
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.perf_counter() - started
    receipt = {
        "schema_id": "qrp-check-receipt",
        "command_id": args.command_id,
        "scope": args.scope,
        "command": " ".join(command),
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "exit_code": proc.returncode,
        "duration_seconds": round(duration, 3),
        "counts": extract_pytest_counts(proc.stdout + "\n" + proc.stderr),
        "stdout_sha256": sha256_text(proc.stdout),
        "stderr_sha256": sha256_text(proc.stderr),
        "output_summary": summarize_output(proc.stdout, proc.stderr),
        "failure_excerpt": (
            "" if proc.returncode == 0 else excerpt(proc.stderr or proc.stdout)
        ),
    }
    out = Path(args.receipt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return proc.returncode


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def extract_pytest_counts(text: str) -> dict[str, int] | None:
    counts = dict.fromkeys(PYTEST_COUNT_KEYS, 0)
    found = False
    for key in PYTEST_COUNT_KEYS:
        label = "error" if key == "errors" else key
        pattern = rf"(\d+)\s+{label}s?\b"
        matches = re.findall(pattern, text)
        if matches:
            counts[key] = sum(int(value) for value in matches)
            found = True
    collected = re.search(r"collected\s+(\d+)\s+items?\b", text)
    if collected:
        counts["collected"] = int(collected.group(1))
        found = True
    return counts if found else None


def summarize_output(stdout: str, stderr: str) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    return excerpt("\n".join(lines[-12:]), limit=1200)


def excerpt(text: str, *, limit: int = 1200) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
