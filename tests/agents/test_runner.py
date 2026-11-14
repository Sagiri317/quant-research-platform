from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.agents.config import dataset_context
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import read_session
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.agents.support import ScriptedClient, agent_config
from tests.helpers.learning import learning_panel


def test_agent_real_cli_dynamic_binding_and_numerical_feedback(tmp_path):
    config = agent_config(tmp_path)
    client = ScriptedClient(
        [
            {
                "kind": "tool",
                "tool": "search",
                "arguments_json": '{"parameters":{"expressions":["x","-x"]}}',
                "summary": "Positive x should rank returns better than negative x.",
            },
            {
                "kind": "tool",
                "tool": "apply",
                "arguments_json": '{"inputs":{"search":"e0001"}}',
                "summary": "Apply the selection to later rows.",
            },
            {
                "kind": "finish",
                "summary": "Applied the selected formula to the later window.",
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


def test_agent_retains_undeclared_parameter_failure(
    tmp_path,
):
    config = agent_config(tmp_path)
    client = ScriptedClient(
        [
            {
                "kind": "tool",
                "tool": "search",
                "arguments_json": '{"parameters":{"label":"other"}}',
                "summary": "Try changing the target.",
            },
            {"kind": "finish", "summary": "The attempt failed.", "evidence": ["e0001"]},
        ]
    )
    root = run_agent(config, client=client)
    report = parse_qrp_json_object((root / "report.json").read_bytes())
    assert report["experiments"][0]["status"] == "failed"
    assert report["experiments"][0]["error"] == "ValueError"


def test_agent_budget_and_model_failure_are_recorded(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_requests"] = 1
    client = ScriptedClient(
        [
            {
                "kind": "tool",
                "tool": "search",
                "arguments_json": '{"parameters":{"label":"other"}}',
                "summary": "An invalid change must remain a failed attempt.",
            }
        ]
    )
    root = run_agent(config, client=client)
    report = parse_qrp_json_object((root / "report.json").read_bytes())
    assert report["status"] == "budget_exhausted"
    assert len(report["experiments"]) == 1
    assert report["experiments"][0]["status"] == "failed"
    assert report["experiments"][0]["error"] == "ValueError"

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
    assert read_session(tmp_path / "unavailable")["error"]["error"] == "RuntimeError"


def test_sandbox_full_table_is_context_while_native_search_filters_maturity(tmp_path):
    config = agent_config(tmp_path)
    frame, spec = learning_panel(tmp_path / "earlier")
    late = frame.timestamp.between("2020-01-06", "2020-01-07")
    frame.loc[late, "label_available_at"] = pd.Timestamp("2020-01-20")
    frame.loc[late, "label"] *= -1
    write_dataset(tmp_path / "late-labels", frame, spec=spec)
    config["sandbox"] = {"inputs": {"panel": str(tmp_path / "late-labels")}}
    config["tools"].pop("apply")
    config["tools"]["search"]["config"]["dataset"] = str(tmp_path / "late-labels")
    client = ScriptedClient(
        [
            {"kind": "tool", "tool": "search", "summary": "Use only mature labels."},
            {
                "kind": "finish",
                "summary": "Only the positive validation day is mature.",
            },
        ]
    )
    root = run_agent(config, client=client)
    state = read_session(root)
    metadata = state["public"]["sandbox_inputs"]["panel"]
    assert metadata["rows"] == 36
    assert metadata["date_min"] == "2020-01-01"
    assert metadata["date_max"] == "2020-01-12"
    assert metadata["label_availability_columns"] == {"label": "label_available_at"}
    assert state["events"][0]["result"]["status"] == "completed"
    # The two reversed-label dates are immature; the retained date has equal ranks.
    assert state["events"][0]["result"]["metrics"] == {"score": 1.0}
    assert "label_available_at" in client.messages[0]


@pytest.mark.parametrize("empty", [False, True])
def test_sandbox_metadata_does_not_require_label_maturity_declaration(tmp_path, empty):
    frame, spec = learning_panel(tmp_path / "earlier")
    if empty:
        frame = frame.iloc[:0]
    root = tmp_path / "input"
    write_dataset(root, frame, spec=replace(spec, label_availability_columns={}))
    metadata = dataset_context({"panel": root})["panel"]
    assert metadata["rows"] == len(frame)
    assert metadata["date_max"] == (None if empty else "2020-01-12")
    assert metadata["label_availability_columns"] == {}


def test_sandbox_dynamic_input_uses_successful_artifact_alongside_fixed_input(
    tmp_path, monkeypatch
):
    config = agent_config(tmp_path)
    config["sandbox"] = {"inputs": {"fixed": str(tmp_path / "input")}}
    seen = []

    def isolated(code, sandbox, output, *, timeout):
        seen.append(read_dataset(sandbox["inputs"]["dynamic"]))
        assert read_dataset(sandbox["inputs"]["fixed"]).shape[0] == 36
        (output / "results").mkdir(parents=True)
        return {"status": "completed", "exit_code": 0}

    monkeypatch.setattr(
        "quant_research_platform.agents.runner.run_isolated_python", isolated
    )
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                {"kind": "tool", "tool": "search", "summary": "Select a formula."},
                {
                    "kind": "tool",
                    "tool": "apply",
                    "arguments_json": '{"inputs":{"search":"e0001"}}',
                    "summary": "Create an input artifact.",
                },
                {
                    "kind": "tool",
                    "tool": "python",
                    "arguments_json": '{"code":"pass","inputs":{"dynamic":"e0002"}}',
                    "summary": "Read the bound result.",
                },
                {"kind": "finish", "summary": "Both inputs remain accessible."},
            ]
        ),
    )
    assert read_session(root)["events"][2]["result"]["status"] == "completed"
    assert seen[0].discovered_factor.tolist() == [1.0, 2.0, 3.0] * 3
