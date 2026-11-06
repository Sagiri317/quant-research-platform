import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.agents import execution


def test_sandbox_command_isolated_and_missing_runtime_preserves_primary_error(
    tmp_path, monkeypatch
):
    config = {
        "image": "qrp-research:test",
        "inputs": {},
        "user": "1000:1000",
        "cpus": 1,
        "memory": "1g",
        "pids_limit": 64,
    }
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        Path(args[args.index("--cidfile") + 1]).write_text("synthetic-owned-container")
        return 0

    monkeypatch.setattr(execution, "_run_process", execute)
    cleanup = []

    def remove(args, **kwargs):
        cleanup.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(execution.subprocess, "run", remove)
    assert (
        execution.run_isolated_python("print(1)", config, tmp_path / "code", timeout=1)[
            "status"
        ]
        == "completed"
    )
    args = calls[0]
    assert {
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    } <= set(args)
    assert args[-3:] == ["python", "-I", "/experiment.py"]
    assert cleanup[0][:3] == ["docker", "rm", "--force"]

    def missing(*args, **kwargs):
        raise FileNotFoundError("runtime absent")

    monkeypatch.setattr(execution, "_run_process", missing)
    with pytest.raises(FileNotFoundError, match="runtime absent"):
        execution.run_isolated_python(
            "print(1)", config, tmp_path / "missing", timeout=1
        )
    assert len(cleanup) == 1


def test_generated_artifacts_cannot_be_loaded_as_host_models(tmp_path):
    artifacts = {"e0001": {"path": tmp_path, "trusted": False}}
    with pytest.raises(ValueError, match="dataset_reader"):
        execution._resolve_artifacts("artifact:e0001/model.pkl", artifacts)
    with pytest.raises(ValueError, match="path_escape"):
        execution._resolve_artifacts(
            "artifact:e0001/../private", artifacts, allow_generated=True
        )


def test_owned_process_timeout_and_numeric_feedback(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        execution._run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.1,
            cwd=tmp_path,
            log=tmp_path / "private.log",
        )
    (tmp_path / "metrics.json").write_text('{"value": 1e309}')
    with pytest.raises(ValueError):
        execution._metrics(
            tmp_path,
            {
                "score": {
                    "member": "metrics.json",
                    "column": "value",
                    "reduction": "mean",
                }
            },
        )


def test_parquet_metrics_require_a_regular_file_not_a_dataset_directory(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "private.parquet"
    pd.DataFrame({"value": [99.0]}).to_parquet(outside)
    directory = output / "metrics.parquet"
    directory.mkdir()
    (directory / "part.parquet").symlink_to(outside)
    definitions = {
        "score": {
            "member": "metrics.parquet",
            "column": "value",
            "reduction": "mean",
        }
    }
    with pytest.raises(ValueError, match="agent_metric_file_required"):
        execution._metrics(output, definitions)
    regular = output / "regular.parquet"
    pd.DataFrame({"value": [1.0, None, 3.0]}).to_parquet(regular)
    definitions["score"]["member"] = regular.name
    assert execution._metrics(output, definitions) == {"score": 2.0}
