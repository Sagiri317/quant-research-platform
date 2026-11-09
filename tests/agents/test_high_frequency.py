"""Native time-series metrics and continuous accounts through Agent research tools."""

import json

import pandas as pd
import pytest

from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import read_session
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.storage.datasets import write_dataset
from tests.agents.support import ScriptedClient
from tests.agents.test_research_tools import tool
from tests.helpers.high_frequency import parameters, quotes
from tests.research.test_policy_learning import hft_episode


def test_hft_fit_evaluate_backtest_and_inventory_use_native_artifacts(tmp_path):
    episode = hft_episode(tmp_path / "market")
    source = tmp_path / "research"
    frame = pd.concat(
        [quotes(day, assets=("a",)) for day in ("2024-01-02", "2024-01-03")],
        ignore_index=True,
    )
    frame = frame[["timestamp", "asset_id", "eligible", "segment_id"]].copy()
    frame["x"] = [-1.0, 0.0, 1.0] * 2
    frame["label"] = frame.x * 0.01
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(seconds=1)
    write_dataset(
        source,
        frame,
        spec=DataSpec(
            data_id="agent_hft",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=(),
            column_roles={
                "timestamp": "timestamp",
                "asset_id": "key",
                "eligible": "mask",
                "segment_id": "context",
                "x": "factor",
                "label": "label",
                "label_available_at": "context",
            },
            column_units={"label": "decimal_return"},
            label_availability_columns={"label": "label_available_at"},
        ),
    )
    account = episode["account"]
    attached = {
        "source": str(source),
        "quotes": account["panel"],
        "fees": account["fee_schedule"],
        **{"daily_" + name: path for name, path in account["daily_inputs"].items()},
    }
    window = {"start": "2024-01-03", "end": "2024-01-03"}
    protocol = {
        "label": "label",
        "train_window": {"start": "2024-01-02", "end": "2024-01-02"},
        "validation_window": window,
        "test_window": {"start": "2024-01-04", "end": "2024-01-04"},
        "costs": {
            "initial_capital": 5000.0,
            "minimum_commission": 0.0,
            "fee_schedule": "input:fees",
        },
        "metric": "rank_ic",
    }
    fit = {
        "market_id": "a_share",
        "inputs": "input:source",
        "output": "assigned",
        "model": {
            "model_id": "hft_linear",
            "model_algorithm": "linear_regression",
            "input_factor_columns": ["x"],
            "target_column": "protocol:label",
            "random_seed": 1,
        },
        "fit_windows": [{"start": "2024-01-02T01:30:00", "end": "2024-01-02T01:30:06"}],
        "fit_cutoff": "2024-01-02T23:59:59",
        "label_unit": "decimal_return",
    }
    evaluate = {
        "inputs": "input:source",
        "predictions": {"linear": ["artifact:e0002/"]},
        "factor_columns": [],
        "label": "protocol:label",
        "window": window,
        "cutoff": "2024-01-03T23:59:59",
        "sessions": ["2024-01-03"],
        "asset_ids": ["a"],
        "comparison_groups": {},
        "inference_groups": {"linear": ["linear"]},
        "statistics": {
            "hac_lag": 0,
            "block_size": 1,
            "repetitions": 20,
            "seed": 1,
            "confidence": 0.95,
            "fdr_method": "BY",
            "alpha": 0.05,
        },
        "output": "assigned",
    }
    backtest = {
        "panel": "input:quotes",
        "predictions": ["artifact:e0002/"],
        "fee_schedule": "protocol:costs.fee_schedule",
        "daily_inputs": {
            name: "input:daily_" + name for name in account["daily_inputs"]
        },
        "window": window,
        "parameters": {
            **parameters(("a",)),
            "buy_threshold": 0.005,
            "sell_threshold": -0.005,
        },
        "output": "assigned",
    }
    metrics = {
        name: {
            "member": "daily",
            "column": "value",
            "reduction": "mean",
            "filters": [
                ["candidate_id", "==", "linear"],
                ["sample_scope", "==", "own"],
                ["metric", "==", name],
            ],
        }
        for name in ("ic", "rank_ic")
    }
    ledger_metrics = {
        name: {"member": "", "column": name, "reduction": "mean"}
        for name in ("net_pnl", "nav", "fees")
    }
    client = ScriptedClient(
        [
            tool("research", operation="fit", config=fit),
            tool(
                "research",
                operation="predict",
                config={
                    "inputs": "input:source",
                    "model": "artifact:e0001/",
                    "prediction_window": window,
                    "output": "assigned",
                },
            ),
            tool(
                "research",
                operation="evaluate-time-series",
                config=evaluate,
                metrics=metrics,
            ),
            tool(
                "research",
                operation="backtest-hft",
                config=backtest,
                metrics=ledger_metrics,
            ),
            tool(
                "research",
                operation="backtest-hft",
                config={**backtest, "predictions": []},
                metrics=ledger_metrics,
            ),
            tool(
                "research",
                operation="analyze-inventory",
                config={
                    "ledger": "artifact:e0004/",
                    "benchmark": "artifact:e0005/",
                    "output": "assigned",
                },
            ),
            tool("compare", evidence=["e0003", "e0004", "e0005", "e0006"]),
            {
                "kind": "finish",
                "summary": "Perfect synthetic IC does not offset the spread loss.",
                "evidence": ["e0003", "e0004", "e0005"],
            },
        ]
    )
    root = run_agent(
        {
            "question": "Compare the high-frequency model with holding.",
            "protocol": protocol,
            "tools": {},
            "inputs": attached,
            "budget": {"max_requests": 10, "max_experiments": 8, "wall_seconds": 300},
            "llm": {},
            "output": str(tmp_path / "agent"),
        },
        client=client,
    )
    events = read_session(root)["events"]
    assert all(event["result"]["status"] == "completed" for event in events)
    assert events[2]["result"]["metrics"] == {
        "ic": pytest.approx(1.0),
        "rank_ic": pytest.approx(1.0),
    }
    # Flat prices and one sale of 100 shares at a 0.01 CNY half-spread.
    assert events[3]["result"]["metrics"]["net_pnl"] == pytest.approx(-1.0)
    assert events[3]["result"]["metrics"]["nav"] == pytest.approx(4999.0)
    assert events[4]["result"]["metrics"]["net_pnl"] == 0.0
    assert len(events[6]["result"]["rows"]) == 4
    assert "2024-01-03" in json.dumps(events[3]["result"]["scope"])
