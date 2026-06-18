#!/usr/bin/env python3
"""Render and verify PR body validation rows from evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID_STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


def render_pr_body(
    task_metadata: dict[str, Any],
    validation_evidence: dict[str, Any],
) -> str:
    rows = validation_rows(validation_evidence)
    lines = [
        "# QRP Research Foundation Semantic Closure",
        "",
        f"- branch: `{task_metadata.get('branch', '')}`",
        f"- start head: `{task_metadata.get('base_commit_before_task', '')}`",
        f"- final head: `{task_metadata.get('final_sha', '')}`",
        f"- PR: `{task_metadata.get('pr_url', '')}`",
        "- PR status: `draft`",
        "",
        "## Validation",
        "",
        "| Check | Status | Details |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['check']}` | `{row['status']}` | {row['details']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- review zip: `{validation_evidence.get('review_zip_path', '')}`",
            "- external evidence JSON: "
            f"`{validation_evidence.get('external_evidence_json_path', '')}`",
            "",
            "## Non-goals",
            "",
            (
                "No public API/CLI expansion, scheduler, worker, web surface, "
                "external execution integration, or investment advice."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def validate_pr_body_against_evidence(
    body: str,
    validation_evidence: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for row in validation_rows(validation_evidence):
        expected = f"| `{row['check']}` | `{row['status']}` | {row['details']} |"
        if expected not in body:
            failures.append(f"validation_row_mismatch: {row['check']}")
    return failures


def validation_rows(validation_evidence: dict[str, Any]) -> list[dict[str, str]]:
    if "validation_table" in validation_evidence:
        raw_rows = validation_evidence["validation_table"]
        row_kind = "validation_table"
    elif "checks" in validation_evidence:
        raw_rows = validation_evidence["checks"]
        row_kind = "checks"
    else:
        raise ValueError("validation_rows_required")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{row_kind}_must_be_list")
    if not raw_rows:
        raise ValueError("validation_rows_required")
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("validation_row_must_be_object")
        check_key = "check" if row_kind == "validation_table" else "name"
        details_key = "details" if row_kind == "validation_table" else "evidence"
        check = str(raw.get(check_key) or "").strip()
        status = str(raw.get("status") or "").strip().upper()
        details = _clean_cell(str(raw.get(details_key) or ""))
        if not check:
            raise ValueError("validation_check_required")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid_validation_status: {check}")
        rows.append({"check": check, "status": status, "details": details})
    return rows


def _clean_cell(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").replace("|", "/").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-metadata", required=True)
    parser.add_argument("--validation-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    task_metadata = _load_json_object(Path(args.task_metadata), "task_metadata")
    validation_evidence = _load_json_object(
        Path(args.validation_evidence),
        "validation_evidence",
    )
    body = render_pr_body(task_metadata, validation_evidence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    if args.verify:
        failures = validate_pr_body_against_evidence(body, validation_evidence)
        if failures:
            print("pr_body_validation_status=FAIL")
            for failure in failures:
                print(failure)
            return 1
    print("pr_body_validation_status=PASS")
    print(f"pr_body={output.as_posix()}")
    return 0


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{label}_must_be_json_object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
