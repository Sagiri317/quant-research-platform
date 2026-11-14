"""Agent calls compose native calculations, including a negative later result."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from quant_research_platform.agents.config import set_argument
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import read_session
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.storage.datasets import read_dataset
from tests.agents.research_support import daily_research_config
from tests.agents.support import ScriptedClient


@pytest.mark.parametrize("dynamic", [False, True])
def test_factor_to_later_account_reuses_native_science_and_reports_negative_result(
    tmp_path,
    dynamic,
):
    config = daily_research_config(tmp_path)

    def call(tool, **arguments):
        return {
            "kind": "tool",
            "tool": tool,
            "summary": "Compare the causal rank formula against the x baseline.",
            "arguments_json": json.dumps(arguments),
        }

    client = ScriptedClient(
        [
            call("search", parameters={"expressions": ["rank(x)", "-x"]}),
            call("apply", inputs={"search": "e0001"}),
            call("assemble", inputs={"inputs.formula.path": "e0002"}),
            call("evaluate", inputs={"inputs": "e0003"}),
            call("statistics", inputs={"dataset": "e0003"}),
            call("fit", inputs={"inputs": "e0003"}),
            call("predict", inputs={"model": "e0006", "inputs": "e0003"}),
            call("backtest", inputs={"predictions.0.path": "e0007"}),
            call(
                "evaluate",
                parameters={
                    "window": {"start": "2023-01-04", "end": "2023-01-06"},
                    "cutoff": "2023-01-09",
                },
                inputs={"inputs": "e0003"},
            ),
            call("curve", inputs={"inputs": "e0008"}),
            {
                "kind": "finish",
                "summary": (
                    "Historical ranks match the baseline. Later ranks reverse, and "
                    "the constant-price account loses its fees; this is a "
                    "negative synthetic result."
                ),
                "arguments_json": json.dumps(
                    {
                        "claims": [
                            {
                                "evidence": "e0005",
                                "metric": "candidate_hac_p_value",
                                "value": None,
                            },
                            {
                                "evidence": "e0009",
                                "metric": "candidate_ic",
                                "value": -1.0,
                            },
                        ]
                    }
                ),
                "evidence": ["e0004", "e0005", "e0009", "e0008"],
            },
        ]
    )
    if dynamic:
        tools = config["tools"]
        config["tools"], config["inputs"] = {}, {}

        def attach(value):
            if isinstance(value, dict):
                return {key: attach(item) for key, item in value.items()}
            if isinstance(value, list):
                return [attach(item) for item in value]
            if (
                isinstance(value, str)
                and Path(value).is_absolute()
                and Path(value).exists()
            ):
                alias = f"source_{len(config['inputs'])}"
                config["inputs"][alias] = value
                return "input:" + alias
            return value

        actions = []
        for action in client.actions:
            if action.get("tool") in tools:
                template = deepcopy(tools[action["tool"]])
                arguments = json.loads(action["arguments_json"])
                for path, value in arguments.get("parameters", {}).items():
                    set_argument(template["config"], path, value)
                for path, identifier in arguments.get("inputs", {}).items():
                    set_argument(
                        template["config"],
                        path,
                        f"artifact:{identifier}/{template['bindings'][path]}",
                    )
                action = {
                    **action,
                    "tool": "research",
                    "arguments_json": json.dumps(
                        {
                            "operation": template["operation"],
                            "config": attach(template["config"]),
                            "metrics": template["metrics"],
                            "facts": template.get("facts", {}),
                        }
                    ),
                }
            actions.append(action)
        client = ScriptedClient(actions)
    root = run_agent(config, client=client)
    state = read_session(root)
    assert read_dataset(root / "e0002/result").market_id.eq("a_share").all()
    assert state["status"] == "completed"
    assert all(event["result"]["status"] == "completed" for event in state["events"])
    assert state["events"][3]["result"]["metrics"] == {
        "candidate_ic": pytest.approx(1.0),
        "baseline_ic": pytest.approx(1.0),
        "paired_count": 12.0,
        "eligible_count": 12.0,
        "mature_count": 12.0,
        "candidate_rankic_std": pytest.approx(0.0),
        "candidate_positive_rankic_rate": 1.0,
    }
    assert state["events"][4]["result"]["metrics"] == {
        "candidate_hac_standard_error": None,
        "candidate_hac_p_value": None,
        "candidate_fdr_p_value": None,
        "inference_date_count": 4.0,
    }
    # Later labels are the negative of x, so both Spearman correlations are -1.
    later_evaluation = state["events"][8]["result"]["metrics"]
    assert later_evaluation["candidate_ic"] == pytest.approx(-1.0)
    assert later_evaluation["baseline_ic"] == pytest.approx(-1.0)
    assert later_evaluation["candidate_positive_rankic_rate"] == 0.0
    ledger = read_portfolio_ledger_artifact(root / "e0008/result")
    economics = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
    assert economics["fees"] > 0
    # Entry and endpoint prices are constant on Jan 5-6; only transaction fees remain.
    assert economics["net_pnl"] == pytest.approx(-economics["fees"])
    assert state["events"][7]["result"]["metrics"]["net_pnl"] == pytest.approx(
        economics["net_pnl"]
    )
    assert "![e0010 图表](e0010/result.png)" in (root / "report.md").read_text(
        encoding="utf-8"
    )
