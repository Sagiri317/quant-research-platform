"""Replay public machine checks and produce canonical handoff receipts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_PARENT, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from governance_contract import hard_scan_receipt_from_clone  # noqa: E402
from handoff_contract import (  # noqa: E402
    CHECK_RECEIPTS_ENTRY,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    replay_command_spec,
)
from receipt_contract import (  # noqa: E402
    COUNT_FIELDS,
    normalize_hard_scan_receipt,
    normalize_package_smoke_receipt,
    validate_check_receipts,
)
from run_package_smoke import run_package_smoke  # noqa: E402


def build_replayed_check_receipts(repo_root: Path) -> dict[str, object]:
    rows = [
        _run_replay_command(repo_root, "profile-smoke"),
        _run_replay_command(repo_root, "code-map"),
    ]
    status = _aggregate_status([str(row["status"]) for row in rows])
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": status,
        "checks": rows,
    }
    failures = validate_check_receipts(payload)
    if failures:
        raise SystemExit(";".join(failures))
    return payload


def build_replayed_package_smoke_receipt(repo_root: Path) -> dict[str, object]:
    payload = normalize_package_smoke_receipt(run_package_smoke(repo_root))
    return payload


def build_replayed_hard_scan_receipt(
    repo_root: Path,
    final_head: str,
) -> dict[str, object]:
    return normalize_hard_scan_receipt(
        hard_scan_receipt_from_clone(repo_root, final_head=final_head)
    )


def build_machine_receipts(
    repo_root: Path,
    final_head: str,
) -> dict[str, dict[str, object]]:
    return {
        CHECK_RECEIPTS_ENTRY: build_replayed_check_receipts(repo_root),
        PACKAGE_SMOKE_RECEIPT_ENTRY: build_replayed_package_smoke_receipt(repo_root),
        HARD_SCAN_RECEIPT_ENTRY: build_replayed_hard_scan_receipt(repo_root, final_head),
    }


def _run_replay_command(repo_root: Path, command_id: str) -> dict[str, object]:
    spec = replay_command_spec(command_id)
    result = subprocess.run(
        list(spec.command_argv),
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    status = "PASS" if result.returncode == 0 else "FAIL"
    counts = parse_pytest_counts(
        "\n".join((result.stdout or "", result.stderr or ""))
    ) if command_id == "profile-smoke" else None
    return {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": spec.scope,
        "command_argv": list(spec.command_argv),
        "command": spec.command,
        "status": status,
        "exit_code": result.returncode,
        "counts": counts,
    }


def parse_pytest_counts(text: str) -> dict[str, int]:
    counts = {field: 0 for field in COUNT_FIELDS}
    lowered = text.lower()
    for label, field in (
        ("passed", "passed"),
        ("failed", "failed"),
        ("skipped", "skipped"),
        ("deselected", "deselected"),
        ("error", "errors"),
        ("errors", "errors"),
        ("xfailed", "xfailed"),
        ("xpassed", "xpassed"),
    ):
        for match in re.finditer(rf"(\d+)\s+{label}\b", lowered):
            counts[field] += int(match.group(1))
    counts["collected"] = (
        counts["passed"]
        + counts["failed"]
        + counts["skipped"]
        + counts["errors"]
        + counts["xfailed"]
        + counts["xpassed"]
    )
    return counts


def _aggregate_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUN" in statuses:
        return "NOT_RUN"
    return "PASS"
