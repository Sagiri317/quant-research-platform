"""Asset-scoped market inputs stay selected before materialization and key checks."""

from dataclasses import replace

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.high_frequency import inventory_inputs
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import materialize_dataset_session
from tests.helpers.high_frequency import trade_input
from tests.research.test_policy_learning import hft_episode


def test_hft_quotes_and_trades_push_assets_and_preserve_parent_selection(
    tmp_path, monkeypatch
):
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.storage.datasets import reading

    episode = hft_episode(tmp_path / "inputs")
    config = episode["account"]
    panel = read_dataset(config["panel"])
    panel = pd.concat(
        [panel, panel.assign(asset_id="b"), panel.assign(asset_id="c")],
        ignore_index=True,
    )
    declared = reading.open_dataset_session(config["panel"]).spec
    spec = ProcessedDataSpec(
        data_id=declared["data_id"],
        market_id="a_share",
        kind="panel",
        frequency="tick",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles=declared["column_roles"],
    )
    root = tmp_path / "quotes"
    write_dataset(root, panel, spec=replace(spec, partition_columns=("year",)))
    trades = trade_input(
        tmp_path / "trades",
        [(second, asset, "sell", 1.0, 10.0) for second in (1, 4) for asset in "abc"],
    )
    config["parameters"]["passive"] = {
        "queue_model": "conservative",
        "cancel_latency_ms": 0,
    }
    parent = {
        "years": [2024],
        "time_window": {
            "start": "2024-01-02T01:30:03",
            "end": "2024-01-02T01:30:06",
        },
        "row_filters": [["asset_id", "in", ["a", "b"]]],
    }
    config["panel"] = {
        **parent,
        "path": str(root),
        "files": [str(next(root.rglob("*.parquet")).relative_to(root))],
    }
    config["trades"] = {**parent, "path": str(trades.root)}
    consumed = {}
    original = reading.iter_dataset_file_batches

    def observe(session, **kwargs):
        for path, frame in original(session, **kwargs):
            if session.spec["data_id"] in {"hft_panel", "hft_trades"}:
                consumed.setdefault(session.spec["data_id"], []).extend(
                    zip(frame.asset_id, frame.timestamp, strict=True)
                )
            yield path, frame

    monkeypatch.setattr(reading, "iter_dataset_file_batches", observe)
    sessions, _, selected_trades = inventory_inputs(
        config, TimeWindow(**episode["window"]), decision_columns=("mid",)
    )
    rows = list(sessions)
    assert len(rows) == 1
    assert rows[0][1].asset_id.tolist() == ["a", "a"]
    actual_trades = materialize_dataset_session(selected_trades)
    assert actual_trades.asset_id.tolist() == ["a"]
    assert consumed == {
        "hft_panel": [
            ("a", pd.Timestamp("2024-01-02T01:30:03")),
            ("a", pd.Timestamp("2024-01-02T01:30:06")),
        ],
        "hft_trades": [("a", pd.Timestamp("2024-01-02T01:30:04"))],
    }
