from __future__ import annotations

from pathlib import Path

try:
    from tests.agents.handoff.conftest import _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _load_handoff_module  # type: ignore[no-redef]


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_id": "qrp-check-receipt",
        "command_id": "unit",
        "scope": "unit scope",
        "command_argv": ["python", "-m", "pytest"],
        "command": "python -m pytest",
        "status": "PASS",
        "exit_code": 0,
        "duration_seconds": 1.0,
        "counts": {
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "collected": 1,
        },
        "stdout_sha256": "0" * 64,
        "stderr_sha256": "1" * 64,
        "output_summary": "1 passed",
        "failure_excerpt": "",
    }
    row.update(overrides)
    return row


def _failures(row: dict[str, object]) -> list[str]:
    contract = _load_handoff_module("receipt_contract")
    return contract.validate_check_receipt_row(row)


def test_check_receipt_rejects_missing_schema_id() -> None:
    row = _valid_row()
    row.pop("schema_id")

    assert "check_receipt_missing: 0:schema_id" in _failures(row)


def test_check_receipt_rejects_pass_with_nonzero_exit_code() -> None:
    assert "check_receipt_pass_exit_code_invalid: 0" in _failures(
        _valid_row(exit_code=1)
    )


def test_check_receipt_rejects_fail_with_zero_exit_code() -> None:
    assert "check_receipt_fail_exit_code_invalid: 0" in _failures(
        _valid_row(status="FAIL", exit_code=0, failure_excerpt="failed")
    )


def test_check_receipt_rejects_pass_with_failed_count() -> None:
    row = _valid_row()
    counts = dict(row["counts"])
    counts["failed"] = 1
    row["counts"] = counts

    assert "check_receipt_pass_counts_invalid: 0" in _failures(row)


def test_check_receipt_rejects_pass_with_error_count() -> None:
    row = _valid_row()
    counts = dict(row["counts"])
    counts["errors"] = 1
    row["counts"] = counts

    assert "check_receipt_pass_counts_invalid: 0" in _failures(row)


def test_check_receipt_rejects_zero_collected_with_outcome_counts() -> None:
    row = _valid_row()
    counts = dict(row["counts"])
    counts["collected"] = 0
    row["counts"] = counts

    assert "check_receipt_counts_collected_mismatch: 0" in _failures(row)


def test_check_receipt_rejects_not_run_without_reason() -> None:
    assert "check_receipt_not_run_reason_missing: 0" in _failures(
        _valid_row(status="NOT_RUN", exit_code=None, counts=None)
    )


def test_check_receipt_rejects_blocked_without_reason() -> None:
    assert "check_receipt_blocked_reason_missing: 0" in _failures(
        _valid_row(status="BLOCKED", exit_code=None, counts=None)
    )


def test_check_receipt_rejects_raw_output_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [
            _valid_row(stdout="raw", stderr="raw", raw_output="raw"),
            _valid_row(command_id="grep", raw_grep_output="raw", grep_hits=[]),
        ],
    }

    failures = contract.validate_receipt_payload("check-receipts.json", payload)

    assert "receipt_forbidden_raw_field: check-receipts.json:checks.0.stdout" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.0.stderr" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.0.raw_output" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.1.raw_grep_output" in failures
    assert "receipt_forbidden_raw_field: check-receipts.json:checks.1.grep_hits" in failures


def test_run_check_writes_command_argv_and_collected_sum(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    receipt_out = tmp_path / "check-receipts.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/quant-research-platform-maintenance/scripts/run_check.py",
            "--id",
            "synthetic-pytest",
            "--scope",
            "unit",
            "--receipt-out",
            str(receipt_out),
            "--",
            sys.executable,
            "-c",
            "print('2 passed, 1 skipped, 3 deselected in 0.01s')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = json.loads(receipt_out.read_text(encoding="utf-8").splitlines()[0])
    assert result.returncode == 0
    assert receipt["command_argv"][0] == sys.executable
    assert receipt["command"]
    assert receipt["counts"]["collected"] == 6
