#!/usr/bin/env python3
"""Collect compact handoff receipts into the package-ready receipts directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from receipt_contract import (  # noqa: E402
    ADVISORY_REVIEW_RECEIPT_NAME,
    CHECK_RECEIPTS_NAME,
    OPTIONAL_RECEIPT_FILES,
    REQUIRED_RECEIPT_FILES,
    validate_receipt_file_set,
)
from quant_research_platform.policy.check_status import (  # noqa: E402
    aggregate_validation_matrix_status,
)


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
        write_check_receipts(Path(args.check_jsonl), out_dir / CHECK_RECEIPTS_NAME)
    for name in sorted(REQUIRED_RECEIPT_FILES | OPTIONAL_RECEIPT_FILES):
        source = input_dir / name
        if not source.is_file():
            continue
        target = out_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    failures = validate_receipt_file_set(out_dir)
    result = {
        "schema_id": "qrp-handoff-receipt-collection",
        "status": "PASS" if not failures else "FAIL",
        "receipts": sorted(
            path.name
            for path in out_dir.glob("*.json")
            if path.name in REQUIRED_RECEIPT_FILES
            or path.name == ADVISORY_REVIEW_RECEIPT_NAME
        ),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


def write_check_receipts(source: Path, target: Path) -> None:
    checks = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    statuses = [
        str(row.get("status"))
        for row in checks
        if isinstance(row, dict)
        and str(row.get("status")) in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
    ]
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": aggregate_validation_matrix_status(statuses),
        "checks": checks,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
