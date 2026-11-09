"""Actual append-only learning attempts are facts, not run permissions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.experiments import read_trial_ledger


@pytest.mark.parametrize(
    "error,status",
    [
        (None, "completed"),
        (ValueError("fit"), "failed"),
        (KeyboardInterrupt(), "abandoned"),
    ],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_trial_retains_attempt_and_actual_terminal(tmp_path, error, status, legacy):
    recorder = TrialRecorder(
        tmp_path / "trial",
        {"model": {"model_id": "candidate"}, "fit_cutoff": "2020-01-01"},
    )
    attempted = recorder.begin()
    before = (recorder.root / "attempted.json").read_bytes()
    terminal = recorder.finish(result={"rows": 3}, error=error)
    terminal_path = recorder.root / "terminal.json"
    compact = json.loads(terminal_path.read_text())
    assert compact == {
        "trial_id": attempted["trial_id"],
        "status": status,
        "result": {"rows": 3},
        **({"error": type(error).__name__} if error is not None else {}),
    }
    if legacy:
        terminal_path.write_text(json.dumps(terminal))
    records = read_trial_ledger(recorder.root)
    assert records == (attempted, terminal)
    assert (recorder.root / "attempted.json").read_bytes() == before
    with pytest.raises(OutputAlreadyExistsError):
        recorder.finish(error=error)


def test_hard_exit_leaves_attempted_fact(tmp_path):
    root = tmp_path / "trial"
    code = (
        "from pathlib import Path; "
        "from quant_research_platform.research.experiments import TrialRecorder; "
        "import os; "
        f"TrialRecorder(Path({str(root)!r}),"
        "{'model':{'model_id':'candidate'}}).begin(); os._exit(3)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    assert result.returncode == 3
    assert len(read_trial_ledger(root)) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_trial_rejects_rebound_terminal(tmp_path, legacy):
    recorder = TrialRecorder(tmp_path, {"model": {"model_id": "a"}})
    recorder.begin()
    terminal = recorder.finish()
    path = tmp_path / "terminal.json"
    payload = terminal if legacy else json.loads(path.read_text())
    if legacy:
        payload["model"]["model_id"] = "b"
    else:
        payload["trial_id"] = "another-trial"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="completion_mismatch"):
        read_trial_ledger(tmp_path)
