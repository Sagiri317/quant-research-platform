"""Directional executions require this asset's current quote and known mark."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from tests.helpers.high_frequency import daily_view, parameters, quotes
from tests.research.test_hft_portfolio import native_account


def test_taker_waits_for_its_own_one_sided_quote(tmp_path):
    panel = quotes("2024-01-02")
    view = daily_view(panel)
    first, middle, last = sorted(panel.timestamp.unique())
    signal = panel.loc[
        panel.asset_id.eq("a") & panel.timestamp.eq(first), ["timestamp", "asset_id"]
    ].assign(prediction_score=-1.0)
    panel = panel.loc[~(panel.asset_id.eq("a") & panel.timestamp.eq(middle))].copy()
    panel.loc[panel.asset_id.eq("b"), ["mid", "bid_px1", "ask_px1"]] = [
        20.0,
        19.99,
        20.01,
    ]
    panel.loc[
        panel.asset_id.eq("a") & panel.timestamp.eq(last),
        ["mid", "ask_px1", "ask_vol1"],
    ] = np.nan
    params = parameters()
    params["initial_inventory"] = {"a": 100, "b": 0}
    batches, summary = native_account(
        tmp_path,
        [(view, panel, signal)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [(t.asset_id, t.timestamp_ns, t.quantity) for t in trades] == [
        ("a", pd.Timestamp(last).value, -100)
    ]
    assert trades[0].notional == pytest.approx(-999.0)
    assert summary["nav"] == pytest.approx(4999.0)


def test_taker_never_borrows_another_assets_mark_or_its_own_future_close(tmp_path):
    panel = quotes("2024-01-02")
    view = daily_view(panel)
    first, middle, last = sorted(panel.timestamp.unique())
    view = replace(
        view,
        panel=view.panel.assign(
            open=lambda frame: frame.open.where(frame.asset_id.ne("b"))
        ),
        valuation=view.valuation.assign(
            open=lambda frame: frame.open.where(frame.asset_id.ne("b")),
            close=lambda frame: frame.close.where(frame.asset_id.ne("b"), 30.0),
        ),
    )
    panel.loc[panel.asset_id.eq("a"), ["mid", "bid_px1", "ask_px1"]] = [
        20.0,
        19.99,
        20.01,
    ]
    panel.loc[
        panel.asset_id.eq("b") & panel.timestamp.le(middle),
        ["mid", "bid_px1", "bid_vol1"],
    ] = np.nan
    panel.loc[
        panel.asset_id.eq("b") & panel.timestamp.eq(last),
        ["mid", "bid_px1", "ask_px1"],
    ] = [30.0, 29.99, 30.01]
    signal = panel.loc[
        panel.asset_id.eq("b") & panel.timestamp.eq(first), ["timestamp", "asset_id"]
    ].assign(prediction_score=1.0)
    params = parameters()
    params["initial_inventory"] = {"a": 0, "b": 0}
    batches, summary = native_account(
        tmp_path,
        [(view, panel, signal)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    assert not any(batch.trades for batch in batches)
    assert summary["nav"] == 5000.0
