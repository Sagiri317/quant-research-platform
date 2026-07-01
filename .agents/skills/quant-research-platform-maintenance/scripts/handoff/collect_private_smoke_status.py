#!/usr/bin/env python3
"""Extract review-safe private smoke status from a local run.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VALID = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", default=None)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--status", choices=sorted(VALID), default="NOT_RUN")
    parser.add_argument("--not-run-reason", default="")
    parser.add_argument("--blocked-reason", default="")
    args = parser.parse_args(argv)
    receipt = build_receipt(
        run_json=Path(args.run_json) if args.run_json else None,
        status=args.status,
        not_run_reason=args.not_run_reason,
        blocked_reason=args.blocked_reason,
    )
    out = Path(args.receipt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"PASS", "NOT_RUN"} else 1


def build_receipt(
    *,
    run_json: Path | None,
    status: str,
    not_run_reason: str,
    blocked_reason: str,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    run_sha = ""
    exit_code = None
    if run_json is not None and run_json.is_file():
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        run_sha = sha256_file(run_json)
        status = str(payload.get("status") or payload.get("overall_status") or status)
        if status not in VALID:
            status = "FAIL"
        raw_exit = payload.get("exit_code")
        exit_code = int(raw_exit) if isinstance(raw_exit, int) else None
    receipt: dict[str, object] = {
        "schema_id": "qrp-private-smoke-status",
        "status": status,
        "command_id": "private-smoke",
        "exit_code": exit_code,
        "run_level_overall": status,
        "private_smoke_run_level": status,
        "private_smoke_child_checks": child_status(payload, status),
        "compare": "NOT_RUN",
        "full_horizon": "NOT_RUN",
        "child_statuses": child_statuses(payload),
        "run_json_sha256": run_sha,
        "row_level_data_included": False,
    }
    if status == "NOT_RUN":
        receipt["not_run_reason"] = not_run_reason or "private smoke not run"
    if status == "BLOCKED":
        receipt["blocked_reason"] = blocked_reason or "private smoke blocked"
    return receipt


def child_status(payload: dict[str, object], fallback: str) -> str:
    statuses = child_statuses(payload)
    if not statuses:
        return "NOT_RUN" if fallback == "NOT_RUN" else fallback
    values = list(statuses.values())
    if "FAIL" in values:
        return "FAIL"
    if "BLOCKED" in values:
        return "BLOCKED"
    if "NOT_RUN" in values:
        return "NOT_RUN"
    return "PASS"


def child_statuses(payload: dict[str, object]) -> dict[str, str]:
    rows = payload.get("checks") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        status = str(row.get("status") or "")
        if name and status in VALID:
            result[name] = status
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
