"""Agent decisions compose actual native calculations, including a negative holdout."""

import json

import pytest

from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import final_evaluation_status, read_session
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.storage.datasets import read_dataset
from tests.agents.research_support import daily_research_config
from tests.agents.support import ScriptedClient


def test_factor_to_final_account_reuses_native_science_and_reports_negative_result(
    tmp_path,
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
            {
                "kind": "freeze",
                "summary": (
                    "Freeze the redundant rank formula and model for the final "
                    "diagnostic."
                ),
                "arguments_json": '{"candidates":{"formula":"e0001","model":"e0006"}}',
                "evidence": ["e0004", "e0005", "e0008"],
            },
            {
                "kind": "finish",
                "summary": (
                    "Development ranks match the baseline. Final ranks reverse, and "
                    "the constant-price final account loses its fees; this is a "
                    "negative synthetic result."
                ),
                "evidence": ["e0004", "e0012", "e0015"],
            },
        ]
    )
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
    # The holdout label is the negative of x, so both Spearman correlations are -1.
    final_evaluation = state["events"][11]["result"]["metrics"]
    assert final_evaluation["candidate_ic"] == pytest.approx(-1.0)
    assert final_evaluation["baseline_ic"] == pytest.approx(-1.0)
    assert final_evaluation["candidate_positive_rankic_rate"] == 0.0
    ledger = read_portfolio_ledger_artifact(root / "e0015/result")
    economics = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
    assert economics["fees"] > 0
    # Entry and endpoint prices are constant on Jan 5-6; only transaction fees remain.
    assert economics["net_pnl"] == pytest.approx(-economics["fees"])
    assert state["events"][14]["result"]["metrics"]["net_pnl"] == pytest.approx(
        economics["net_pnl"]
    )
    assert final_evaluation_status(state) == "completed"
    assert "![e0016 图表](e0016/result.png)" in (root / "report.md").read_text(
        encoding="utf-8"
    )
