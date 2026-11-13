"""Published trading status acts independently of forecasts and actual fill capacity."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.high_frequency import (
    daily_view,
    parameters,
    quotes,
    schedule,
    trade_input,
    write_daily_inputs,
)

START = pd.Timestamp("2024-01-02T01:30:00")


def status_source(root, panel, *, flag="is_st", value=True, known_second=1.5):
    daily = write_daily_inputs(root / "daily", panel)
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "a"],
            "timestamp": [START, START + pd.Timedelta(seconds=1)],
            "published_at": [START, START + pd.Timedelta(seconds=known_second)],
            "listing_date": [START.normalize()] * 2,
            "market_type": [4, 4],
            "is_st": [False, False],
            "is_pt": [False, False],
            "is_delisting": [False, False],
        }
    )
    frame[flag] = frame[flag].astype("boolean")
    frame.loc[1, flag] = value
    roles = dict.fromkeys(frame, "context")
    roles.update(timestamp="timestamp", asset_id="key", published_at="key")
    write_dataset(
        root / "status",
        frame,
        spec=InterimDataSpec(
            data_id="synthetic_stock_status",
            source_id="synthetic",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id", "published_at"),
            column_roles=roles,
            column_units={},
            partition_columns=(),
        ),
    )
    return {
        "stock_master": daily["stock_master"],
        "status": str(root / "status"),
        "available_at_column": "published_at",
    }


def replay(panel, signals, params, source, *, events=None):
    sessions = [
        (
            daily_view(frame, listing=panel.session_date.min()),
            frame,
            signals.loc[signals.timestamp.dt.normalize().eq(day)],
        )
        for day, frame in panel.groupby("session_date", sort=True)
    ]
    return list(
        inventory_batches(
            sessions,
            schedule(),
            params,
            TimeWindow(
                panel.session_date.min().date(), panel.session_date.max().date()
            ),
            stock_status=source,
            trade_session=events,
            max_quote_gap_seconds=6,
        )
    )


def no_predictions(panel):
    return panel.iloc[:0][["timestamp", "asset_id"]].assign(prediction_score=np.nan)


def held(batches):
    return next(
        row.quantity
        for row in batches[-1].valuations
        if row.asset_id == "a" and row.phase == "end"
    )


@pytest.mark.parametrize("flag", ["is_st", "is_delisting", "is_pt"])
def test_published_exit_is_not_forecast_or_pt_alias(tmp_path, flag):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:01", "01:30:02", "01:30:03"),
    )
    params = parameters(("a",))
    params["latency_ms"] = 10
    source = status_source(tmp_path, panel, flag=flag)
    batches = replay(panel, no_predictions(panel), params, source)
    trades = [trade for batch in batches for trade in batch.trades]
    if flag == "is_pt":
        assert trades == []
        assert held(batches) == 100
    else:
        assert [trade.quantity for trade in trades] == [-100]
        assert trades[0].timestamp_ns == (START + pd.Timedelta(seconds=2)).value
        assert held(batches) == 0
    # Prove the source publication, not its earlier effective date, drives action.
    diagnostics = [
        (next(v.timestamp_ns for v in batch.valuations if v.phase == "end"), row.text)
        for batch in batches
        for row in batch.diagnostics
        if row.name == "stock_status_control"
    ]
    assert any(at == (START + pd.Timedelta(seconds=1.5)).value for at, _ in diagnostics)


def test_exit_respects_missing_depth_t1_and_next_session(tmp_path):
    first = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:01", "01:30:02", "01:30:03"),
    )
    first.loc[first.timestamp.eq(START + pd.Timedelta(seconds=2)), "bid_vol1"] = np.nan
    second = quotes("2024-01-03", assets=("a",), clocks=("01:30:00", "01:30:01"))
    panel = pd.concat((first, second), ignore_index=True)
    params = parameters(("a",))
    params["latency_ms"] = 10
    signals = first.iloc[[0]][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    source = status_source(tmp_path, panel)
    batches = replay(panel, signals, params, source)
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [200, -100, -200]
    assert [pd.Timestamp(trade.timestamp_ns) for trade in trades] == [
        START + pd.Timedelta(seconds=1),
        START + pd.Timedelta(seconds=3),
        START + pd.Timedelta(days=1, seconds=1),
    ]
    assert held(batches) == 0
    capital = PortfolioCapital(
        "hft_inventory", params["initial_capital"], "CNY", "cash_equity"
    )
    path = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "kind": "a_share_hft",
            "window": {"start": "2024-01-02", "end": "2024-01-03"},
        },
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    assert held(list(read_portfolio_ledger_artifact(path).iter_chunks())) == 0


def test_maker_cancel_keeps_pre_ack_fills_and_real_locked_inventory(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:01", "01:30:02", "01:30:03", "01:30:04"),
    ).assign(bid_vol1=0.0, ask_vol1=0.0)
    params = parameters(("a",))
    params.update(
        latency_ms=10,
        passive={"queue_model": "conservative", "cancel_latency_ms": 1000},
    )
    signals = panel.iloc[[0]][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    events = trade_input(
        tmp_path / "trades",
        [
            (1.5, "a", "sell", 50, 9.99),
            (2.5, "a", "sell", 100, 9.99),
            (3.5, "a", "buy", 100, 10.01),
        ],
    )
    source = status_source(tmp_path, panel, known_second=1.25)
    batches = replay(panel, signals, params, source, events=events)
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [50, -100]
    assert held(batches) == 50
    assert any(
        row.name == "passive_cancel_requested"
        for batch in batches
        for row in batch.diagnostics
    )


def test_status_after_last_quote_reports_unfinished_without_fabricated_fill(tmp_path):
    panel = quotes("2024-01-02", assets=("a",), clocks=("01:30:00", "01:30:01"))
    source = status_source(tmp_path, panel, known_second=2)
    batches = replay(panel, no_predictions(panel), parameters(("a",)), source)
    assert not any(batch.trades for batch in batches)
    assert held(batches) == 100
    assert any(
        row.name == "stock_status_control" and row.text == "special_treatment_exit"
        for batch in batches
        for row in batch.diagnostics
    )


def test_unknown_status_blocks_new_buy_without_inventing_exit(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:01", "01:30:02", "01:30:03"),
    )
    source = status_source(tmp_path, panel, value=pd.NA)
    signals = panel.iloc[[2]][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches = replay(panel, signals, parameters(("a",)), source)
    assert not any(batch.trades for batch in batches)
    assert held(batches) == 100
    assert any(
        row.name == "stock_status_control" and row.text == "status_unknown"
        for batch in batches
        for row in batch.diagnostics
    )
