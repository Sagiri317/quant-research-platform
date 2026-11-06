"""Persistent conversations retain evidence, steering and scientific boundaries."""

import json

import pytest

from quant_research_platform.agents.runner import resume_agent, run_agent
from quant_research_platform.agents.session import context_messages, read_session
from quant_research_platform.storage.datasets import read_dataset
from tests.agents.support import ScriptedClient, agent_config


def call(tool, **arguments):
    return {
        "kind": "tool",
        "tool": tool,
        "arguments_json": json.dumps(arguments),
        "summary": "Test the declared research hypothesis.",
    }


def finish(**arguments):
    return {
        "kind": "finish",
        "summary": "The evidence and limitations are retained.",
        "arguments_json": json.dumps(arguments),
        "evidence": ["e0001"],
    }


def test_pause_resume_reuses_successful_artifact_without_rerunning(tmp_path):
    config = agent_config(tmp_path)
    messages = []

    def control():
        pending = messages.copy()
        messages.clear()
        return pending

    def completed(event):
        messages.append("/pause")

    root = run_agent(
        config,
        client=ScriptedClient([call("search")]),
        control=control,
        on_event=completed,
    )
    original = (root / "e0001/result.json").read_bytes()
    assert read_session(root)["status"] == "paused"
    client = ScriptedClient([call("apply", inputs={"search": "e0001"}), finish()])
    resume_agent(
        root,
        message="Keep the original formula and compare the later sample.",
        client=client,
    )
    state = read_session(root)
    assert state["status"] == "completed"
    assert len(state["requests"]) == 3
    assert (root / "e0001/result.json").read_bytes() == original
    assert (
        read_dataset(root / "e0002/result").discovered_factor.tolist()
        == [1.0, 2.0, 3.0] * 3
    )
    assert "Keep the original formula" in client.messages[0]


def test_question_waits_for_answer_and_keeps_protocol(tmp_path):
    config = agent_config(tmp_path)
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                {
                    "kind": "ask_user",
                    "summary": "What economic rationale should the baseline test?",
                }
            ]
        ),
    )
    resume_agent(root, budget={"max_requests": 7}, client=ScriptedClient([]))
    assert read_session(root)["status"] == "waiting_for_user"
    assert read_session(root)["config"]["budget"]["max_requests"] == 7
    client = ScriptedClient([finish()])
    resume_agent(
        root, message="Test whether positive momentum persists.", client=client
    )
    state = read_session(root)
    assert state["public"]["protocol"] == config["protocol"]
    assert "positive momentum" in client.messages[0]
    assert state["pending_question"] is None


def test_budget_resume_is_explicit_and_cumulative(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_requests"] = 1
    root = run_agent(config, client=ScriptedClient([call("search")]))
    assert read_session(root)["status"] == "budget_exhausted"
    resume_agent(root, client=ScriptedClient([]))
    assert len(read_session(root)["requests"]) == 1
    resume_agent(root, budget={"max_requests": 2}, client=ScriptedClient([finish()]))
    assert read_session(root)["status"] == "completed"


def test_new_steering_supersedes_an_unexecuted_model_action(tmp_path):
    config = agent_config(tmp_path)
    queued = []

    class SteeringClient(ScriptedClient):
        def complete(self, messages, *, timeout):
            result = super().complete(messages, timeout=timeout)
            if len(self.messages) == 1:
                queued.append("Stop exploration and report its absence.")
            return result

    def control():
        items = queued.copy()
        queued.clear()
        return items

    client = SteeringClient([call("search"), finish()])
    root = run_agent(config, client=client, control=control)
    state = read_session(root)
    assert state["events"][0]["result"]["status"] == "superseded"
    assert not (root / "e0001/result.json").exists()
    assert "Stop exploration" in client.messages[1]


def final_config(tmp_path):
    config = agent_config(tmp_path)
    apply = config["tools"].pop("apply")
    apply.pop("bindings")
    apply["name"] = "final_formula"
    apply["config"]["search"] = "artifact:candidate"
    apply["config"]["output_window"] = "protocol:test_window"
    config["protocol"]["test_window"] = {"start": "2020-01-10", "end": "2020-01-12"}
    config["final_evaluation"] = [apply]
    config["budget"]["max_requests"] = 5
    return config


def test_freeze_runs_native_final_step_and_forbids_further_tuning(tmp_path):
    config = final_config(tmp_path)
    client = ScriptedClient(
        [
            call("search", parameters={"expressions": ["x", "-x"]}),
            {
                "kind": "freeze",
                "summary": "Freeze positive x after development comparison.",
                "arguments_json": '{"candidates":{"candidate":"e0001"}}',
                "evidence": ["e0001"],
            },
            call("search", parameters={"expressions": ["-x"]}),
            {
                "kind": "finish",
                "summary": (
                    "The fixed formula was applied; association is not a return "
                    "guarantee."
                ),
                "evidence": ["e0001", "e0003"],
            },
        ]
    )
    root = run_agent(config, client=client)
    state = read_session(root)
    assert state["phase"] == "reporting"
    assert state["status"] == "completed"
    assert state["holdout_viewed"] is True
    assert state["events"][2]["action"]["tool"] == "final_formula"
    assert (
        state["events"][3]["result"]["reason"]
        == "agent_only_report_allowed_after_final"
    )
    assert (
        read_dataset(root / "e0003/result").discovered_factor.tolist()
        == [1.0, 2.0, 3.0] * 3
    )
    with pytest.raises(ValueError, match="completed_session"):
        resume_agent(
            root, message="Try the opposite formula.", client=ScriptedClient([])
        )


def test_negative_research_can_finish_without_viewing_holdout(tmp_path):
    config = final_config(tmp_path)
    client = ScriptedClient(
        [
            call("search", parameters={"expressions": ["x/(x-x)"]}),
            {
                "kind": "finish",
                "summary": "The proposed formula was undefined.",
                "arguments_json": '{"outcome":"no_candidate"}',
                "evidence": ["e0001"],
            },
        ]
    )
    # A one-candidate budget prevents automatic random proposals replacing it.
    config["tools"]["search"]["config"]["budget"] = 1
    root = run_agent(config, client=client)
    state = read_session(root)
    assert state["events"][0]["result"]["metrics"] == {"score": None}
    assert state["holdout_viewed"] is False
    assert state["status"] == "completed"


def test_context_compaction_keeps_constraints_and_retrievable_history(tmp_path):
    config = agent_config(tmp_path)
    document = tmp_path / "methods.md"
    document.write_text(
        "Causal hypothesis with a falsifiable baseline.\n" * 200, encoding="utf-8"
    )
    config["documents"] = {"methods": str(document)}
    config["budget"].update(max_requests=6, max_context_chars=12000)
    actions = [
        call("lookup", scope="documents", query="", key="methods") for _ in range(4)
    ]
    root = run_agent(config, client=ScriptedClient([*actions, finish()]))
    state = read_session(root)
    messages, omitted = context_messages(state)
    assert state["compactions"] > 0
    assert omitted > 0
    assert len(state["events"]) == 5
    assert "train_window" in messages[1]["content"]
    assert (root / "e0001/action.json").is_file()
    state["events"] = state["events"][:-1]
    state["config"]["budget"]["max_context_chars"] = 6000
    messages, _ = context_messages(state)
    assert json.loads(messages[-1]["content"])["id"] == "e0004"
    assert sum(len(item["content"]) for item in messages) <= 6000


def test_completed_research_memory_links_evidence_and_can_be_retrieved(tmp_path):
    config = agent_config(tmp_path)
    config["memory"] = str(tmp_path / "memory")
    root = run_agent(config, client=ScriptedClient([call("search"), finish()]))
    memory = json.loads(next((tmp_path / "memory").glob("*.json")).read_text())
    assert memory["evidence"][0]["metrics"] == {"score": 1.0}
    assert memory["protocol"] == config["protocol"]
    assert memory["session"] == read_session(root)["id"]
    assert memory["local_source"]["root"] == str(root)
    assert (root / memory["local_source"]["artifacts"]["e0001"]).is_file()
    resume_agent(root, client=ScriptedClient([]))
    assert len(list((tmp_path / "memory").glob("*.json"))) == 1
    config["output"] = str(tmp_path / "second")
    second = run_agent(
        config,
        client=ScriptedClient(
            [call("lookup", scope="memory", query="evidence"), finish()]
        ),
    )
    assert read_session(second)["events"][0]["result"]["total"] == 1
    assert "local_source" not in json.dumps(read_session(second)["events"][0]["result"])


def test_interrupted_final_step_resumes_without_duplicate_computation(tmp_path):
    def interrupt_after_saved_result(event):
        if event["tool"] == "final_formula":
            raise KeyboardInterrupt

    root = run_agent(
        final_config(tmp_path),
        client=ScriptedClient(
            [
                call("search"),
                {
                    "kind": "freeze",
                    "summary": "Freeze the development formula.",
                    "arguments_json": '{"candidates":{"candidate":"e0001"}}',
                },
            ]
        ),
        on_event=interrupt_after_saved_result,
    )
    assert read_session(root)["status"] == "paused"
    before = (root / "e0003/result/spec.json").read_bytes()
    resume_agent(root, client=ScriptedClient([finish()]))
    state = read_session(root)
    assert state["status"] == "completed"
    assert (
        len([e for e in state["events"] if e["action"]["tool"] == "final_formula"]) == 1
    )
    assert (root / "e0003/result/spec.json").read_bytes() == before


def test_completed_session_can_finish_memory_publication_after_io_failure(tmp_path):
    config = agent_config(tmp_path)
    memory_path = tmp_path / "memory"
    memory_path.write_text("A file blocks the configured memory directory.")
    config["memory"] = str(memory_path)
    with pytest.raises(FileExistsError):
        run_agent(config, client=ScriptedClient([call("search"), finish()]))
    memory_path.unlink()
    root = resume_agent(config["output"], client=ScriptedClient([]))
    assert read_session(root)["status"] == "completed"
    assert len(list(memory_path.glob("*.json"))) == 1


def test_token_budget_rejects_missing_supplier_usage(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_tokens"] = 10000
    client = ScriptedClient([call("lookup", scope="catalog", query="momentum")])
    client.usage = None
    with pytest.raises(ValueError, match="usage_unavailable"):
        run_agent(config, client=client)
    state = read_session(config["output"])
    assert len(state["requests"]) == 1


def test_pause_keeps_queued_steering_and_orphan_output_is_not_overwritten(tmp_path):
    config = agent_config(tmp_path)
    root = run_agent(
        config,
        client=ScriptedClient([]),
        control=lambda: ["/pause", "Retain the original baseline."],
    )
    assert read_session(root)["steering"] == ["Retain the original baseline."]
    orphan = root / "e0001"
    orphan.mkdir()
    (orphan / "partial.txt").write_text("Interrupted output")
    resume_agent(
        root,
        client=ScriptedClient([{"kind": "finish", "summary": "No experiments yet."}]),
    )
    assert read_session(root)["events"][0]["id"] == "e0002"
    assert (orphan / "partial.txt").read_text() == "Interrupted output"
