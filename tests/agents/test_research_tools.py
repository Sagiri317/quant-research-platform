"""Conversation research calls use native contracts and retain scientific scope."""

import json
from copy import deepcopy

import pytest

from quant_research_platform.agents.knowledge import lookup
from quant_research_platform.agents.protocol import (
    OPERATIONS,
    validate_budget,
    validate_config,
)
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


def test_configure_search_and_compare_use_actual_data_and_lock_protocol(tmp_path):
    config = agent_config(tmp_path)
    protocol, search = config["protocol"], config["tools"]["search"]
    config.update(protocol=None, tools={}, inputs={"source": str(tmp_path / "input")})
    config["budget"]["max_requests"] = 10
    search["config"]["dataset"] = "input:source"
    changed = deepcopy(protocol)
    changed["label"] = "another_label"
    client = ScriptedClient(
        [
            tool("research", operation="search-factors", config=search["config"]),
            tool("configure", protocol=protocol),
            tool(
                "research",
                operation="search-factors",
                config=search["config"],
                metrics=search["metrics"],
                facts={"expression": {"member": "", "path": "selected_expression"}},
            ),
            tool("configure", protocol=changed),
            tool("compare", evidence=["e0003"]),
            tool("compare", evidence=["e0001", "e0003", "e0005"]),
            {
                "kind": "finish",
                "summary": "The native rank correlation is one.",
                "evidence": ["e0003"],
                "arguments_json": json.dumps(
                    {"claims": [{"evidence": "e0003", "metric": "score", "value": 1.0}]}
                ),
            },
        ]
    )
    root = run_agent(config, client=client)
    state = read_session(root)
    events = state["events"]
    assert state["status"] == "completed"
    assert events[0]["result"]["reason"] == "agent_research_protocol_required"
    assert events[2]["result"]["metrics"] == {"score": 1.0}
    assert events[2]["result"]["facts"]["expression"] == "x"
    assert events[3]["result"]["reason"] == "agent_protocol_already_in_use"
    assert events[4]["result"]["rows"][0]["metrics"] == {"score": 1.0}
    rows = events[5]["result"]["rows"]
    assert rows[0]["status"] == "failed"
    assert rows[0]["reason"] == "agent_research_protocol_required"
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


def test_no_protocol_is_valid_only_for_nonexperimental_conversation(tmp_path):
    config = agent_config(tmp_path)
    config["protocol"] = None
    with pytest.raises(ValueError, match="agent_research_protocol_required"):
        validate_config(config)
    config["tools"] = {}
    assert validate_config(config)[1]["protocol"] is None


def test_protocol_can_be_corrected_before_any_scientific_experiment(tmp_path):
    config = agent_config(tmp_path)
    protocol = config["protocol"]
    config.update(protocol=None, tools={}, inputs={})
    config["budget"]["max_requests"] = 5
    corrected = {**protocol, "metric": "corrected_metric"}
    state = read_session(
        run_agent(
            config,
            client=ScriptedClient(
                [
                    tool("research", operation="fit", config={}),
                    tool("configure", protocol=protocol),
                    tool("configure", protocol=corrected),
                    {
                        "kind": "finish",
                        "summary": "Corrected the protocol before computation.",
                    },
                ]
            ),
        )
    )
    assert state["events"][0]["phase"] == "discussion"
    assert state["events"][2]["result"]["status"] == "completed"
    assert state["public"]["protocol"] == corrected


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
