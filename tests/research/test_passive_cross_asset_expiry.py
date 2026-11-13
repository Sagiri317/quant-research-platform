"""A missing quote releases only the unfilled reservation, never prior fills."""

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from tests.helpers.high_frequency import quotes
from tests.research.test_passive_execution import replay


def test_expired_partial_order_releases_cash_before_another_asset_request(tmp_path):
    panel = pd.concat(
        [
            quotes("2024-01-02", assets=("a",)),
            quotes(
                "2024-01-02",
                assets=("b",),
                clocks=("01:30:00", "01:30:13", "01:30:14"),
            ),
        ],
        ignore_index=True,
    ).sort_values(["timestamp", "asset_id"], kind="stable")
    panel["bid_vol1"] = 100.0
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02T01:30:00", "2024-01-02T01:30:13"]),
            "asset_id": ["a", "b"],
            "prediction_score": [1.0, 1.0],
        }
    )
    batches, summary = replay(
        tmp_path,
        panel,
        signals,
        [(4, "a", "sell", 130, 9.99), (15, "b", "sell", 300, 9.99)],
        capital=4005,
        minimum=5,
        window=TimeWindow("2024-01-02T01:30:00", "2024-01-02T01:30:16"),
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [(t.asset_id, t.quantity) for t in trades] == [("a", 30), ("b", 100)]
    assert [pd.Timestamp(t.timestamp_ns).second for t in trades] == [4, 15]
    expiry = [
        (batch.valuations[-1].timestamp_ns, diagnostic.number)
        for batch in batches
        for diagnostic in batch.diagnostics
        if diagnostic.asset_id == "a"
        and diagnostic.name == "passive_observation_gap_unfilled_quantity"
    ]
    assert expiry == [(pd.Timestamp("2024-01-02T01:30:13").value, 170)]
    fees = sum(
        -flow.amount
        for batch in batches
        for flow in batch.cashflows
        if flow.kind == "commission_cost"
    )
    assert fees == 10
    final = {v.asset_id: v for v in batches[-1].valuations if v.phase == "end"}
    assert final["a"].quantity == 130
    assert final["b"].quantity == 200
    assert final["__cash__"].value == pytest.approx(696.3)
    assert summary["nav"] == pytest.approx(3996.3)
