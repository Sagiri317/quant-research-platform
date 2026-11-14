"""Conversation research calls use native contracts and retain scientific scope."""

import json

import pytest

from quant_research_platform.agents.config import (
    OPERATIONS,
    validate_budget,
)
from quant_research_platform.agents.knowledge import lookup
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import context_messages, read_session
from tests.agents.support import ScriptedClient, agent_config


def tool(name, **arguments):
    return {
        "kind": "tool",
        "tool": name,
        "summary": "Run the declared research step.",
        "arguments_json": json.dumps(arguments),
    }


def test_direct_native_search_and_compare_use_actual_data_and_windows(tmp_path):
    config = agent_config(tmp_path)
    search = config["tools"]["search"]
    config.update(tools={}, inputs={"source": str(tmp_path / "input")})
    config["budget"]["max_requests"] = 8
    search["config"]["dataset"] = "input:source"
    later_cutoff = {**search["config"], "selection_cutoff": "2020-01-20"}
    client = ScriptedClient(
        [
            tool("research", operation="search-factors", config={}),
            tool(
                "research",
                operation="search-factors",
                config=search["config"],
                metrics=search["metrics"],
                facts={"expression": {"member": "", "path": "selected_expression"}},
            ),
            tool(
                "research",
                operation="search-factors",
                config=later_cutoff,
                metrics=search["metrics"],
            ),
            tool("compare", evidence=["e0001", "e0002", "e0003"]),
            {
                "kind": "finish",
                "summary": "The native rank correlation is one.",
                "evidence": ["e0002"],
                "arguments_json": json.dumps(
                    {"claims": [{"evidence": "e0002", "metric": "score", "value": 1.0}]}
                ),
            },
        ]
    )
    root = run_agent(config, client=client)
    state = read_session(root)
    events = state["events"]
    assert state["status"] == "completed"
    assert events[0]["result"]["status"] == "failed"
    assert (
        events[1]["result"]["metrics"]
        == events[2]["result"]["metrics"]
        == {"score": 1.0}
    )
    assert events[1]["result"]["facts"]["expression"] == "x"
    rows = events[3]["result"]["rows"]
    assert [row["status"] for row in rows] == ["failed", "completed", "completed"]
    assert all("rows" not in row for row in rows)
    assert state["public"]["inputs"]["source"]["columns"]["x"] == "factor"
    assert str(tmp_path) not in "\n".join(client.messages)


def test_every_allowed_operation_exposes_actual_native_api():
    for name in OPERATIONS:
        result = lookup(
            {"scope": "operations", "query": "", "key": name}, {"config": {}}
        )
        assert "Native entry point" in result["text"]
        assert "def " in result["text"]
    fit = lookup({"scope": "operations", "query": "", "key": "fit"}, {"config": {}})
    assert '"input_factor_columns"' in fit["text"]
    assert '"model_id"' in fit["text"]


@pytest.mark.parametrize("limit", [300000, None])
def test_large_and_unlimited_context_keep_full_evidence(tmp_path, limit):
    config = agent_config(tmp_path)
    document = tmp_path / "long.md"
    document.write_text("研究证据" * 7500, encoding="utf-8")
    config.update(tools={}, documents={"long": str(document)})
    config["budget"]["max_context_chars"] = limit
    client = ScriptedClient(
        [
            tool("lookup", scope="documents", query="", key="long"),
            tool("lookup", scope="documents", query="", key="long"),
            {"kind": "finish", "summary": "The full evidence is retained."},
        ]
    )
    state = read_session(run_agent(config, client=client))
    messages, omitted = context_messages(state)
    assert omitted == 0
    assert sum(len(item["content"]) for item in messages) > 60000
    assert state["events"][0]["result"]["next_offset"] is None
    assert state["config"]["budget"]["max_context_chars"] == limit
    with pytest.raises(ValueError, match="agent_budget_invalid"):
        validate_budget({**config["budget"], "max_requests": None})


@pytest.mark.parametrize(
    "operation,key",
    [
        ("predict", "model"),
        ("explain", "model"),
        ("backtest-cost-model", "model"),
        ("backtest-policy", "policy"),
    ],
)
def test_dynamic_model_loaders_require_native_evidence(tmp_path, operation, key):
    config = agent_config(tmp_path)
    config.update(tools={}, inputs={})
    client = ScriptedClient(
        [
            tool(
                "research",
                operation=operation,
                config={key: "untrusted/model", "strategy": "ppo"},
            ),
            {"kind": "finish", "summary": "No untrusted model was loaded."},
        ]
    )
    state = read_session(run_agent(config, client=client))
    assert (
        state["events"][0]["result"]["reason"] == "agent_model_requires_native_evidence"
    )
    assert "artifact" not in state["events"][0]
