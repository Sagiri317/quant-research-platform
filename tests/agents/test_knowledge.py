"""Knowledge and patch tools expose declared evidence without applying changes."""

import json

from quant_research_platform.agents.execution import error_feedback
from quant_research_platform.agents.knowledge import lookup
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import read_session
from tests.agents.support import ScriptedClient, agent_config


def test_report_csv_lookup_preserves_values_and_pagination(tmp_path):
    source = tmp_path / "summary.csv"
    content = "组合,净收益_pct,费用_pct\n底仓增强,-0.25,0.10\n"
    source.write_text(content, encoding="utf-8")
    state = {"config": {"documents": {"summary": str(source)}}}
    matches = lookup({"scope": "documents", "query": "底仓增强"}, state)
    assert matches["matches"][0]["excerpt"] == "底仓增强,-0.25,0.10"
    first = lookup(
        {"scope": "documents", "query": "", "key": "summary", "page_size": 12},
        state,
    )
    rest = lookup(
        {
            "scope": "documents",
            "query": "",
            "key": "summary",
            "offset": first["next_offset"],
        },
        state,
    )
    assert first["text"] + rest["text"] == content
    assert rest["next_offset"] is None


def test_patch_is_reviewable_and_original_source_is_unchanged(tmp_path, monkeypatch):
    config = agent_config(tmp_path)
    source = tmp_path / "experiment.py"
    source.write_text("value = 1\n", encoding="utf-8")
    config["documents"] = {"experiment": str(source)}
    monkeypatch.chdir(tmp_path)
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                {
                    "kind": "tool",
                    "tool": "propose_patch",
                    "summary": "Propose the corrected constant.",
                    "arguments_json": json.dumps(
                        {"document": "experiment", "content": "value = 2\n"}
                    ),
                },
                {
                    "kind": "finish",
                    "summary": "A patch is available for review, not applied.",
                    "evidence": ["e0001"],
                },
            ]
        ),
    )
    assert source.read_text() == "value = 1\n"
    patch = (root / "e0001/proposal.patch").read_text()
    assert "--- a/experiment.py" in patch
    assert "-value = 1\n+value = 2\n" in patch
    assert read_session(root)["events"][0]["result"]["applied"] is False


def test_generated_error_feedback_keeps_location_without_values(tmp_path):
    log = tmp_path / "private.log"
    log.write_text(
        'PRIVATE_SENTINEL\n  File "/experiment.py", line 17, in <module>\n'
        "NameError: PRIVATE_SENTINEL\n"
    )
    assert error_feedback(RuntimeError("agent_python_failed"), log=log) == {
        "error": "NameError",
        "reason": "agent_python_failed",
        "code_line": 17,
    }


def test_native_failure_can_be_corrected_using_safe_feedback(tmp_path):
    config = agent_config(tmp_path)
    client = ScriptedClient(
        [
            {
                "kind": "tool",
                "tool": "search",
                "summary": "Try an invalid forward reference.",
                "arguments_json": '{"parameters":{"expressions":["delay(x,-1)"]}}',
            },
            {
                "kind": "tool",
                "tool": "search",
                "summary": "Use a valid causal formula after the rejection.",
                "arguments_json": '{"parameters":{"expressions":["x"]}}',
            },
            {
                "kind": "finish",
                "summary": (
                    "The first attempt failed; the corrected formula ranks "
                    "the synthetic labels perfectly."
                ),
                "evidence": ["e0001", "e0002"],
            },
        ]
    )
    root = run_agent(config, client=client)
    events = read_session(root)["events"]
    assert events[0]["result"]["reason"] == "factor_window_positive_integer_required"
    assert events[1]["result"]["metrics"] == {"score": 1.0}
    assert "factor_window_positive_integer_required" in client.messages[1]
    assert "Traceback" not in client.messages[1]
    result = lookup({"scope": "evidence", "query": "e0001"}, read_session(root))
    first = next(item for item in result["matches"] if item["key"] == "e0001")
    assert "factor_window_positive_integer_required" in first["excerpt"]
