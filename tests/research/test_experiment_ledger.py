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
def test_trial_retains_attempt_and_actual_terminal(tmp_path, error, status):
    recorder = TrialRecorder(
        tmp_path / "trial",
        {"model": {"model_id": "candidate"}, "fit_cutoff": "2020-01-01"},
    )
    attempted = recorder.begin()
    before = (recorder.root / "attempted.json").read_bytes()
    recorder.finish(error=error)
    records = read_trial_ledger(recorder.root)
    assert records[0] == attempted and records[-1]["status"] == status
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


def test_trial_rejects_rebound_terminal(tmp_path):
    recorder = TrialRecorder(tmp_path, {"model": {"model_id": "a"}})
    recorder.begin()
    recorder.finish()
    path = tmp_path / "terminal.json"
    payload = json.loads(path.read_text())
    payload["model"]["model_id"] = "b"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="completion_mismatch"):
        read_trial_ledger(tmp_path)
