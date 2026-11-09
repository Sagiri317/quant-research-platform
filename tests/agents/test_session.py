"""Persistent conversations retain evidence, steering and scientific boundaries."""

import json

import pytest

from quant_research_platform.agents.runner import resume_agent, run_agent
from quant_research_platform.agents.session import (
    context_messages,
    final_evaluation_status,
    read_session,
)
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


def test_checkpoint_preserves_primary_and_original_when_cleanup_fails(
    tmp_path, monkeypatch
):
    from pathlib import Path

    import quant_research_platform.agents.session as session

    target = tmp_path / "checkpoint.json"
    target.write_text("original")
    primary = RuntimeError("synthetic fsync failure")

    def fail_sync(*args):
        raise primary

    original_unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.parent == tmp_path and path != target:
            raise OSError("synthetic cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(session.os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(RuntimeError) as caught:
        session.replace_text(target, "new")
    assert caught.value is primary
    assert primary.__notes__ == ["session checkpoint cleanup also failed: OSError"]
    assert target.read_text() == "original"


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
    assert final_evaluation_status(state) == "completed"
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


def test_reporting_can_page_omitted_evidence_without_reopening_development(tmp_path):
    config = final_config(tmp_path)
    config["budget"]["max_context_chars"] = 10000
    search = call("search")
    search["summary"] = "Compare the candidate to the baseline. " * 400
    actions = [
        search,
        {
            "kind": "freeze",
            "summary": "Freeze the development formula.",
            "arguments_json": '{"candidates":{"candidate":"e0001"}}',
        },
        call("lookup", scope="evidence", query="", key="e0001", page_size=512),
        call(
            "lookup", scope="evidence", query="", key="e0001", offset=512, page_size=512
        ),
        finish(),
    ]
    client = ScriptedClient(actions)
    root = run_agent(config, client=client)
    state = read_session(root)
    original = state["events"][0]
    text = json.dumps(
        {"action": original["action"], "result": original["result"]}, ensure_ascii=False
    )
    assert state["status"] == "completed"
    _, omitted = context_messages(state)
    assert omitted > 0
    for offset, event, message in zip(
        (0, 512), state["events"][3:5], client.messages[3:5], strict=True
    ):
        assert event["phase"] == "reporting"
        assert event["result"]["status"] == "completed"
        assert event["result"]["text"] == text[offset : offset + 512]
        assert event["result"]["next_offset"] == offset + 512
        received = json.loads(json.loads(message)[-1]["content"])
        assert received["result"]["text"] == event["result"]["text"]


def test_negative_research_can_finish_without_final_evaluation(tmp_path):
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
    assert final_evaluation_status(state) == "not_started"
    assert "holdout_viewed" not in state
    assert state["status"] == "completed"
    assert (root / "report.md").read_text(encoding="utf-8").count(
        "The proposed formula was undefined."
    ) == 1


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
    assert omitted > 0
    assert len(state["events"]) == 5
    assert "train_window" in messages[1]["content"]
    assert (root / "e0001/action.json").is_file()
    state["events"] = state["events"][:-1]
    state["config"]["budget"]["max_context_chars"] = 10000
    messages, _ = context_messages(state)
    assert json.loads(messages[-1]["content"])["id"] == "e0004"
    assert sum(len(item["content"]) for item in messages) <= 10000


def test_context_keeps_cited_baseline_and_skips_oversized_recent_records(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_requests"] = 8
    baseline, large = tmp_path / "baseline.md", tmp_path / "large.md"
    baseline.write_text("The baseline net return is negative.", encoding="utf-8")
    large.write_text("A long intermediate observation.\n" * 300, encoding="utf-8")
    config["documents"] = {"baseline": str(baseline), "large": str(large)}
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                call("lookup", scope="documents", query="", key="baseline"),
                *[
                    call("lookup", scope="documents", query="", key="large")
                    for _ in range(4)
                ],
                finish(),
            ]
        ),
    )
    state = read_session(root)
    state["config"]["budget"]["max_context_chars"] = 10000
    messages, omitted = context_messages(state)
    records = [json.loads(item["content"]) for item in messages[2:]]
    assert records[0]["id"] == "e0001"
    assert records[0]["result"]["text"] == baseline.read_text(encoding="utf-8")
    assert records[-1]["id"] == "e0006"
    assert omitted > 0
    assert sum(len(item["content"]) for item in messages) <= 10000
    state["events"].pop()
    messages, _ = context_messages(state)
    records = [json.loads(item["content"]) for item in messages[2:]]
    assert records[0]["id"] == "e0001"
    assert records[-1]["id"] == "e0005"
    assert records[-1]["details_omitted"] is True
    assert len(read_session(root)["events"]) == 6


def test_finish_checks_raw_metric_claims_and_can_correct_a_rejected_claim(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_requests"] = 8
    invalid = [
        (
            {"evidence": "e0001", "metric": "score", "value": 0.5},
            "agent_claim_value_mismatch",
        ),
        (
            {"evidence": "e0001", "metric": "return", "value": 1},
            "agent_claim_metric_not_found",
        ),
        (
            {"evidence": "e0002", "metric": "score", "value": 1},
            "agent_claim_evidence_not_cited",
        ),
        (
            {"evidence": "e0001", "metric": "score", "value": True},
            "agent_claim_value_must_be_numeric_or_null",
        ),
    ]
    client = ScriptedClient(
        [
            call("search"),
            finish(claims={}),
            *[finish(claims=[claim]) for claim, _ in invalid],
            finish(claims=[{"evidence": "e0001", "metric": "score", "value": 1.0}]),
        ]
    )
    root = run_agent(config, client=client)
    state = read_session(root)
    assert state["status"] == "completed"
    assert state["events"][0]["result"]["metrics"] == {"score": 1.0}
    assert [item["result"]["reason"] for item in state["events"][1:-1]] == [
        "agent_claims_must_be_list",
        *[reason for _, reason in invalid],
    ]
    assert state["events"][-1]["result"]["status"] == "completed"
    assert "agent_claim_value_mismatch" in client.messages[3]


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


def test_interrupted_unrecorded_final_output_retries_same_frozen_config(
    tmp_path, monkeypatch
):
    from quant_research_platform.agents import runner

    native = runner.run_native
    interrupted = False

    def interrupt_after_output(template, *args, **kwargs):
        nonlocal interrupted
        result = native(template, *args, **kwargs)
        if template.get("name") == "final_formula" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(runner, "run_native", interrupt_after_output)
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
    )
    state = read_session(root)
    assert state["status"] == "paused"
    assert final_evaluation_status(state) == "interrupted"
    original = (root / "e0003/result/spec.json").read_bytes()
    resume_agent(root, client=ScriptedClient([finish()]))
    state = read_session(root)
    assert state["status"] == "completed"
    assert final_evaluation_status(state) == "completed"
    before = json.loads((root / "e0003/config.json").read_bytes())
    after = json.loads((root / "e0004/config.json").read_bytes())
    assert before.pop("output") != after.pop("output")
    assert before == after
    assert (root / "e0003/result/spec.json").read_bytes() == original
    assert (root / "e0004/result/spec.json").read_bytes() == original
    assert [item["result"]["status"] for item in state["events"][2:4]] == [
        "interrupted",
        "completed",
    ]


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


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": 1},
        {"input_tokens": 1, "output_tokens": -1},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": "2"},
    ],
)
def test_token_budget_rejects_unusable_supplier_usage_before_accepting_finish(
    tmp_path, usage
):
    config = agent_config(tmp_path)
    config["budget"]["max_tokens"] = 10000
    client = ScriptedClient([finish()])
    client.usage = usage
    with pytest.raises(ValueError, match="usage_unavailable"):
        run_agent(config, client=client)
    state = read_session(config["output"])
    assert len(state["requests"]) == 1
    assert state["requests"][0]["usage"] is None
    assert state["events"] == []
    assert state["status"] == "failed"


def test_token_budget_counts_valid_usage_and_stops_following_requests(tmp_path):
    config = agent_config(tmp_path)
    config["budget"]["max_tokens"] = 10
    client = ScriptedClient([call("lookup", scope="catalog", query="momentum")])
    client.usage = {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}
    root = run_agent(config, client=client)
    state = read_session(root)
    assert state["status"] == "budget_exhausted"
    assert len(state["requests"]) == 1
    assert state["requests"][0]["usage"] == {"input_tokens": 4, "output_tokens": 6}
    assert state["events"] == []


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
