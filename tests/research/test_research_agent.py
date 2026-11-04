"""Agent control decisions use real CLI operations and keep private inputs local."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

from quant_research_platform.agents import execution
from quant_research_platform.agents.client import AgentAction, OpenAICompatibleClient
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.research.test_research_learners import learning_panel


class ScriptedClient:
    """Local action driver for controller tests; it is not an external model service."""

    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages, *, timeout):
        assert timeout > 0
        self.messages.append(json.dumps(messages))
        return AgentAction.model_validate(next(self.actions))


def agent_config(tmp_path):
    learning_panel(tmp_path / "input")
    protocol = {
        "label": "label",
        "train_window": {"start": "2020-01-01", "end": "2020-01-03"},
        "validation_window": {"start": "2020-01-05", "end": "2020-01-07"},
        "test_window": {"start": "2020-01-20", "end": "2020-01-30"},
        "costs": {},
        "metric": "validation_rank_ic",
    }
    search = {
        "dataset": str(tmp_path / "input"),
        "columns": ["x"],
        "baseline_columns": [],
        "label": "protocol:label",
        "train_window": "protocol:train_window",
        "validation_window": "protocol:validation_window",
        "train_cutoff": "2020-01-04",
        "selection_cutoff": "2020-01-08",
        "method": "random",
        "budget": 3,
        "max_depth": 1,
        "windows": [2],
        "seed": 1,
        "output": "assigned_by_runner",
    }
    tools = {
        "search": {
            "operation": "search-factors",
            "config": search,
            "vary": {"expressions": [["x", "-x"]]},
            "metrics": {
                "score": {
                    "member": "",
                    "column": "validation_score",
                    "reduction": "mean",
                }
            },
        },
        "apply": {
            "operation": "apply-formula",
            "config": {
                "search": "bound_by_runner",
                "dataset": str(tmp_path / "input"),
                "input_window": {"start": "2020-01-08", "end": "2020-01-12"},
                "output_window": {"start": "2020-01-10", "end": "2020-01-12"},
                "output": "assigned_by_runner",
            },
            "vary": {},
            "metrics": {},
            "bindings": {"search": ""},
        },
    }
    return {
        "question": "Compare declared causal formulas.",
        "protocol": protocol,
        "tools": tools,
        "budget": {"max_requests": 4, "max_experiments": 3, "wall_seconds": 60},
        "output": str(tmp_path / "agent"),
        "llm": {},
    }


def test_agent_real_cli_dynamic_binding_and_numerical_feedback(tmp_path):
    config = agent_config(tmp_path)
    client = ScriptedClient(
        [
            {
                "kind": "run_tool",
                "tool": "search",
                "changes": {"expressions": ["x", "-x"]},
            },
            {"kind": "run_tool", "tool": "apply", "inputs": {"search": "e0001"}},
            {
                "kind": "finish",
                "conclusion": "Applied the selected formula to the later window.",
                "evidence": ["e0001", "e0002"],
            },
        ]
    )
    root = run_agent(config, client=client)
    report = parse_qrp_json_object((root / "report.json").read_bytes())
    assert report["status"] == "completed"
    assert [item["status"] for item in report["experiments"]] == [
        "completed",
        "completed",
    ]
    # Increasing x and labels have exactly matching ranks in every cross section.
    assert report["experiments"][0]["metrics"] == {"score": 1.0}
    assert (
        read_dataset(root / "e0002" / "result").discovered_factor.tolist()
        == [1.0, 2.0, 3.0] * 3
    )
    assert str(tmp_path) not in "\n".join(client.messages)
    assert "PRIVATE_SENTINEL" not in "\n".join(client.messages)


def test_agent_retains_failed_attempt_and_rejects_holdout_and_protocol_changes(
    tmp_path,
):
    config = agent_config(tmp_path)
    client = ScriptedClient(
        [
            {"kind": "run_tool", "tool": "search", "changes": {"label": "different"}},
            {"kind": "finish", "conclusion": "The attempt failed.", "evidence": []},
        ]
    )
    root = run_agent(config, client=client)
    report = parse_qrp_json_object((root / "report.json").read_bytes())
    assert report["experiments"][0]["status"] == "failed"
    assert report["experiments"][0]["error"] == "ValueError"
    config["tools"]["search"]["config"]["selection_cutoff"] = "2020-01-20"
    with pytest.raises(ValueError, match="reaches_holdout"):
        run_agent(config, client=client)
    config["tools"]["search"]["config"]["selection_cutoff"] = "2020-01-08"
    config["tools"]["search"]["config"]["label"] = "another_label"
    with pytest.raises(ValueError, match="label_conflict"):
        run_agent(config, client=client)


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


def test_http_transport_and_response_schema(monkeypatch):
    received = []

    def respond(request):
        received.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"kind":"finish","conclusion":"Done"}'}}
                ]
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setenv("QRP_TEST_API_KEY", "synthetic-key")
    client = OpenAICompatibleClient(
        {
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
            "credential_env": "QRP_TEST_API_KEY",
            "max_output_tokens": 100,
        }
    )
    assert client.complete([], timeout=1).kind == "finish"
    assert received[0].url.path == "/v1/chat/completions"
    assert received[0].headers["authorization"] == "Bearer synthetic-key"
    assert json.loads(received[0].content)["max_tokens"] == 100
    with pytest.raises(ValueError, match="agent_action_fields_invalid"):
        AgentAction(kind="finish", conclusion="Done", code="print(1)")


def test_agent_budget_and_model_failure_are_recorded(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_requests"] = 1
    client = ScriptedClient(
        [{"kind": "run_tool", "tool": "search", "changes": {"label": "forbidden"}}]
    )
    root = run_agent(config, client=client)
    report = parse_qrp_json_object((root / "report.json").read_bytes())
    assert report["status"] == "budget_exhausted"
    assert len(report["experiments"]) == 1

    class Unavailable:
        def complete(self, messages, *, timeout):
            raise RuntimeError("synthetic provider failure")

    config["output"] = str(tmp_path / "unavailable")
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        run_agent(config, client=Unavailable())
    report = parse_qrp_json_object(
        (tmp_path / "unavailable" / "report.json").read_bytes()
    )
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError"


def test_sandbox_rejects_labels_that_mature_in_holdout(tmp_path):
    config = agent_config(tmp_path)
    frame, spec = learning_panel(tmp_path / "earlier")
    frame.loc[frame.timestamp.eq(frame.timestamp.max()), "label_available_at"] = (
        pd.Timestamp("2020-01-20")
    )
    write_dataset(tmp_path / "late-labels", frame, spec=spec)
    config["sandbox"] = {"inputs": {"training": str(tmp_path / "late-labels")}}
    with pytest.raises(ValueError, match="labels_reach_holdout"):
        run_agent(config, client=ScriptedClient([]))


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
