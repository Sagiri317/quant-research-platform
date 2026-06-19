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
    ]
    lines.extend(
        _paragraph_section("Scope Summary", task_metadata.get("scope_summary"))
    )
    lines.extend(_list_section("Commit List", task_metadata.get("commit_list")))
    lines.extend(_list_section("Fixed Blockers", task_metadata.get("fixed_blockers")))
    lines.extend(_mapping_section("Layer Summary", task_metadata.get("layer_summary")))
    lines.extend(
        _paragraph_section(
            "Old Naming Cleanup",
            task_metadata.get("old_naming_cleanup_summary"),
        )
    )
    lines.extend(
        _paragraph_section(
            "Semantic Scan",
            task_metadata.get("semantic_scan_summary"),
        )
    )
    lines.extend(
        _mapping_section("Smoke And Compare", task_metadata.get("smoke_compare"))
    )
    non_pass_rows = [row for row in rows if row["status"] != "PASS"]
    lines.extend(
        [
            "",
            "## NOT_RUN / BLOCKED / FAIL",
            "",
        ]
    )
    if non_pass_rows:
        for row in non_pass_rows:
            lines.append(
                f"- `{row['check']}`: `{row['status']}` - {row['details']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
        "",
        "## Validation",
        "",
        "| Check | Status | Details |",
        "| --- | --- | --- |",
        ]
    )
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


def _paragraph_section(title: str, value: object) -> list[str]:
    text = _clean_cell(str(value or ""))
    if not text:
        return []
    return ["", f"## {title}", "", text]


def _list_section(title: str, values: object) -> list[str]:
    if not isinstance(values, list | tuple) or not values:
        return []
    lines = ["", f"## {title}", ""]
    for value in values:
        lines.append(f"- {_clean_cell(str(value))}")
    return lines


def _mapping_section(title: str, values: object) -> list[str]:
    if not isinstance(values, dict) or not values:
        return []
    lines = ["", f"## {title}", ""]
    for key in sorted(values):
        lines.append(f"- `{_clean_cell(str(key))}`: {_clean_cell(str(values[key]))}")
    return lines


def validate_pr_body_against_evidence(
    body: str,
    validation_evidence: dict[str, Any],
    *,
    external_evidence: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    rows = validation_rows(validation_evidence)
    for row in rows:
        expected = f"| `{row['check']}` | `{row['status']}` | {row['details']} |"
        if expected not in body:
            failures.append(f"validation_row_mismatch: {row['check']}")
    if external_evidence is not None:
        failures.extend(_external_status_failures(rows, external_evidence))
    return failures


def _external_status_failures(
    rows: list[dict[str, str]],
    external_evidence: dict[str, Any],
) -> list[str]:
    row_status = {row["check"]: row["status"] for row in rows}
    failures: list[str] = []
    status_pairs = (
        ("handoff_self_test_status", "handoff_self_test"),
        ("handoff_verify_status", "handoff_verify"),
        ("package_verification_status", "handoff_verify"),
        ("handoff_safety_scan_status", "handoff_safety_scan"),
        ("package_safety_status", "package_safety_check"),
        ("pr_body_evidence_match_status", "pr_body_evidence_match"),
        ("github_review_threads_status", "github_review_threads"),
    )
    for external_key, check_name in status_pairs:
        if external_key not in external_evidence or check_name not in row_status:
            continue
        external_status = str(external_evidence[external_key]).strip().upper()
        validation_status = row_status[check_name]
        if external_status != validation_status:
            failures.append(
                "external_status_mismatch: "
                f"{external_key}={external_status} != "
                f"{check_name}={validation_status}"
            )
    summary = external_evidence.get("validation_summary")
    if isinstance(summary, dict):
        top_status = str(external_evidence.get("status") or "").strip().upper()
        summary_status = str(summary.get("status") or "").strip().upper()
        if top_status and summary_status and top_status != summary_status:
            failures.append(
                "external_status_mismatch: "
                f"status={top_status} != validation_summary.status={summary_status}"
            )
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
    parser.add_argument("--external-evidence")
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
        external_evidence = (
            _load_json_object(Path(args.external_evidence), "external_evidence")
            if args.external_evidence
            else None
        )
        failures = validate_pr_body_against_evidence(
            body,
            validation_evidence,
            external_evidence=external_evidence,
        )
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
