import pandas as pd
import pytest

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


def test_agent_retains_failed_attempt_and_rejects_holdout_and_protocol_changes(
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
    config["tools"]["search"]["config"]["selection_cutoff"] = "2020-01-20"
    with pytest.raises(ValueError, match="reaches_holdout"):
        run_agent(config, client=client)
    config["tools"]["search"]["config"]["selection_cutoff"] = "2020-01-08"
    config["tools"]["search"]["config"]["label"] = "another_label"
    with pytest.raises(ValueError, match="label_conflict"):
        run_agent(config, client=client)


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
