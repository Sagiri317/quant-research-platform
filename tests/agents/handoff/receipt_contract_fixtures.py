from __future__ import annotations

import subprocess
from pathlib import Path

from tests.helpers.maintenance_script_loader import load_handoff_script_module


def make_successful_subprocess_result(
    argv: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout, stderr)


def make_timeout_subprocess_run():
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    return run


def make_package_build_subprocess_run(
    artifacts: tuple[str, ...],
    calls: list[tuple[list[str], object]] | None = None,
):
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append((argv, kwargs.get("timeout")))
        dist = Path(argv[-1])
        dist.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            (dist / artifact).write_text("artifact", encoding="utf-8")
        return make_successful_subprocess_result(argv)

    return run


def make_valid_check_counts(**overrides: int) -> dict[str, int]:
    counts = {
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "collected": 1,
    }
    counts.update(overrides)
    return counts

def make_valid_check_row(
    command_id: str = "profile-smoke", **overrides: object
) -> dict[str, object]:
    handoff_contract = load_handoff_script_module("handoff_contract")
    spec = handoff_contract.replay_command_spec(command_id)
    row: dict[str, object] = {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": spec.scope,
        "command_argv": list(spec.command_argv),
        "command": spec.command,
        "status": "PASS",
        "exit_code": 0,
        "duration_seconds": 0.1,
        "counts": make_valid_check_counts() if command_id == "profile-smoke" else None,
        "failure_class": None,
        "blocked_reason": None,
        "not_run_reason": None,
        "timeout_seconds": None,
    }
    row.update(overrides)
    return row

def make_check_receipts_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [make_valid_check_row("profile-smoke")],
    }
    payload.update(overrides)
    return payload

def make_package_smoke_receipt_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-package-smoke-receipt",
        "status": "PASS",
        "build": {"status": "PASS", "artifact_kinds": ["wheel", "sdist"]},
        "wheel_extract": {
            "status": "PASS",
            "method": "zipfile_extract",
        },
        "import": {"status": "PASS", "module": "quant_research_platform"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
            "scanned_artifact_kinds": ["wheel", "sdist", "wheel_extract"],
        },
    }
    payload.update(overrides)
    return payload

def make_hard_scan_receipt_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-hard-scan-receipt",
        "status": "PASS",
        "scanned_head": "1" * 40,
        "scanner_source_sha256": "2" * 64,
        "scan_runs": [
            {
                "scan_id": "sensitive_negative_literals",
                "rule_set_id": "source-tree:sensitive_negative_literals",
                "match_count": 0,
                "actual_violation_count": 0,
                "result_sha256": "3" * 64,
                "replayable_from_bundle": True,
            }
        ],
    }
    payload.update(overrides)
    return payload

def make_valid_validation_summary_rows() -> list[dict[str, str]]:
    handoff_contract = load_handoff_script_module("handoff_contract")
    return [
        {"name": name, "status": "PASS"}
        for name in handoff_contract.MANDATORY_VALIDATION_ROWS
    ]

def receipt_row_failures(row: dict[str, object]) -> list[str]:
    contract = load_handoff_script_module("receipt_contract")
    return contract.validate_check_receipt_row(row)
