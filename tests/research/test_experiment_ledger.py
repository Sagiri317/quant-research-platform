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
    terminal = recorder.finish(result={"rows": 3}, error=error)
    terminal_path = recorder.root / "terminal.json"
    compact = json.loads(terminal_path.read_text())
    assert compact == {
        "status": status,
        "result": {"rows": 3},
        **({"error": type(error).__name__} if error is not None else {}),
    }
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


@pytest.mark.parametrize("primary", [RuntimeError("compute"), KeyboardInterrupt()])
def test_failure_record_preserves_primary_when_storage_fails(
    tmp_path, monkeypatch, primary
):
    import quant_research_platform.research.experiments as experiments

    recorder = TrialRecorder(tmp_path / "trial", {})
    recorder.begin()

    def unavailable(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(experiments, "append_trial_record", unavailable)
    with pytest.raises(type(primary)) as caught:
        try:
            raise primary
        except BaseException as error:
            recorder.fail(error)
            raise
    assert caught.value is primary
    assert primary.__notes__ == ["trial failure record also failed: OSError"]
    assert (recorder.root / "attempted.json").exists()
    assert not (recorder.root / "terminal.json").exists()
