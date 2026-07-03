from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    _check_receipts,
    _failures,
    _valid_counts,
    _valid_row,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


def test_compare_functions_detect_replay_drift() -> None:
    contract = load_handoff_script_module("receipt_contract")
    packaged = _check_receipts()
    replayed = _check_receipts()
    replayed["checks"][0]["exit_code"] = 1  # type: ignore[index]
    replayed["checks"][0]["status"] = "FAIL"  # type: ignore[index]
    replayed["checks"][0]["failure_class"] = "pytest_failed"  # type: ignore[index]
    replayed["status"] = "FAIL"

    assert contract.compare_check_receipts(packaged, replayed) == [
        "check_receipts_replay_mismatch"
    ]

@pytest.mark.parametrize(
    ("collected", "expected"),
    (
        (0, "check_receipt_counts_collected_mismatch: 0"),
        (2, "check_receipt_counts_collected_mismatch: 0"),
    ),
)
def test_profile_smoke_counts_collected_must_equal_outcome_total(
    collected: int,
    expected: str,
) -> None:
    failures = _failures(_valid_row(counts=_valid_counts(collected=collected)))

    assert expected in failures

def test_compare_functions_detect_pytest_count_drift() -> None:
    contract = load_handoff_script_module("receipt_contract")
    packaged = _check_receipts()
    replayed = _check_receipts()
    replayed["checks"][0]["counts"]["passed"] += 1  # type: ignore[index]
    replayed["checks"][0]["counts"]["collected"] += 1  # type: ignore[index]

    assert contract.compare_check_receipts(packaged, replayed) == [
        "check_receipts_replay_mismatch"
    ]

def test_parse_pytest_counts_counts_deselected_as_collected() -> None:
    replay = load_handoff_script_module("replay_evidence")

    counts = replay.parse_pytest_counts("35 passed, 33 deselected in 0.86s")

    assert counts["passed"] == 35
    assert counts["deselected"] == 33
    assert counts["collected"] == 68

def test_replay_command_executes_current_interpreter_but_records_canonical_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "1 passed in 0.01s", "")

    monkeypatch.setattr(replay.subprocess, "run", fake_run)

    row = replay._run_replay_command(tmp_path, "profile-smoke")

    assert calls
    assert calls[0][0] == sys.executable
    assert row["command_argv"][0] == "python"
    assert str(row["command"]).startswith("python ")

def test_replay_command_timeout_records_blocked_without_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replay = load_handoff_script_module("replay_evidence")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(replay.subprocess, "run", fake_run)

    row = replay._run_replay_command(tmp_path, "profile-smoke")

    assert row["status"] == "BLOCKED"
    assert row["exit_code"] is None
    assert row["blocked_reason"] == "timeout"
    assert row["failure_class"] == "command_timeout"
    assert row["timeout_seconds"] == replay.replay_command_spec(
        "profile-smoke"
    ).timeout_seconds
