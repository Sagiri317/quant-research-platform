#!/usr/bin/env python3
"""Collect compact handoff receipts into the package-ready receipts directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

VALID = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
REQUIRED = {
    "package-smoke-receipt.json",
    "hard-scan-receipt.json",
    "check-receipts.json",
    "private-smoke-status.json",
}
OLD_NAMES = {
    "package-smoke.json",
    "hard-scan-classification.json",
    "test-checks.json",
    "subagent-review-findings.json",
    "subagent-review-reconciliation.json",
}
FORBIDDEN_KEYS = {"stdout", "stderr", "raw_output", "raw_grep_output", "grep_hits"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--check-jsonl", default=None)
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.check_jsonl:
        write_check_receipts(Path(args.check_jsonl), out_dir / "check-receipts.json")
    for path in input_dir.glob("*.json"):
        if path.name == "check-receipts.json":
            target = out_dir / path.name
        elif path.name in REQUIRED or path.name == "advisory-review-receipt.json":
            target = out_dir / path.name
        else:
            continue
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
    failures = validate_receipts(out_dir)
    index = {
        "schema_id": "qrp-handoff-receipt-index",
        "status": "PASS" if not failures else "FAIL",
        "receipts": sorted(path.name for path in out_dir.glob("*.json")),
        "failures": failures,
    }
    (out_dir / "receipt-index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


def write_check_receipts(source: Path, target: Path) -> None:
    checks = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    statuses = [str(row.get("status")) for row in checks if isinstance(row, dict)]
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": aggregate(statuses),
        "checks": checks,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_receipts(root: Path) -> list[str]:
    failures: list[str] = []
    for name in OLD_NAMES:
        if (root / name).exists():
            failures.append(f"old_receipt_name_present: {name}")
    for name in REQUIRED:
        if not (root / name).is_file():
            failures.append(f"required_receipt_missing: {name}")
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"receipt_json_invalid: {path.name}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"receipt_not_object: {path.name}")
            continue
        schema_id = str(payload.get("schema_id") or "")
        if not schema_id.startswith("qrp-"):
            failures.append(f"receipt_schema_id_invalid: {path.name}")
        if payload.get("status") not in VALID:
            failures.append(f"receipt_status_invalid: {path.name}")
        failures.extend(forbidden_key_failures(path.name, payload))
    return failures


def forbidden_key_failures(name: str, payload: object, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_KEYS:
                failures.append(f"receipt_forbidden_raw_field: {name}:{dotted}")
            failures.extend(forbidden_key_failures(name, value, dotted))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            failures.extend(forbidden_key_failures(name, value, f"{prefix}.{index}"))
    return failures


def aggregate(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUN" in statuses:
        return "NOT_RUN"
    return "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
