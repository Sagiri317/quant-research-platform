"""Observed order deadlines never depend on a future snapshot."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    continuous_session_end,
)
from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
    build_panel,
)
from quant_research_platform.research.policies.reinforcement import backtest_policy
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.sources.venvoo import normalize
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.helpers.high_frequency import parameters, quotes, schedule, trade_input
from tests.markets.a_share.test_high_frequency import (
    INSTRUMENTS,
    raw_events,
    raw_quotes,
)
from tests.research.test_passive_execution import replay
from tests.research.test_policy_learning import hft_episode

START = pd.Timestamp("2024-01-02T01:30:00")


@pytest.mark.parametrize("future_quote", [9, 30, None])
def test_future_snapshot_and_missing_tail_snapshot_cannot_erase_earlier_fill(
    tmp_path, future_quote
):
    seconds = [0, 3, 6] + ([] if future_quote is None else [future_quote])
    raw = raw_quotes([93000000 + second * 1000 for second in seconds]).assign(
        date=20240102
    )
    for level in range(1, 11):
        raw[f"bid_px{level}"] = 100000 - level * 100
        raw[f"ask_px{level}"] = 100000 + level * 100
        raw[f"bid_vol{level}"] = 100.0
    panel = build_panel(
        normalize(raw, "hft_quotes", INSTRUMENTS),
        normalize(raw_events("hft_orders").iloc[:0], "hft_orders", INSTRUMENTS),
        normalize(raw_events("hft_trades").iloc[:0], "hft_trades", INSTRUMENTS),
        max_gap_seconds=6,
    )
    asset = INSTRUMENTS[0]["asset_id"]
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(7, asset, "sell", 300, 9.99)],
        window=TimeWindow(START, START + pd.Timedelta(seconds=33)),
        account_parameters={"markout_ms": 1000},
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [200]
    assert trades[0].timestamp_ns == (START + pd.Timedelta(seconds=7)).value
    times = [batch.valuations[-1].timestamp_ns for batch in batches]
    assert times == sorted(times)
    if future_quote is None:
        reasons = [
            d for b in batches for d in b.diagnostics if d.name == "markout_reason"
        ]
        assert [d.text for d in reasons] == ["no_observed_markout"]


@pytest.mark.parametrize("second,filled", [(12, 200), (12.000000001, 0), (13, 0)])
def test_quote_age_boundary_and_tail_expiry_release_reservations(
    tmp_path, second, filled
):
    panel = quotes("2024-01-02", assets=("a",)).assign(bid_vol1=100.0)
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(second, "a", "sell", 300, 9.99)],
        window=TimeWindow(START, START + pd.Timedelta(seconds=13)),
    )
    assert sum(t.quantity for b in batches for t in b.trades) == filled
    frozen = [
        d.number for b in batches for d in b.diagnostics if d.name == "frozen_cash"
    ]
    assert frozen[-1] == 0.0
    assert any(
        d.name == "passive_order_observation_gap"
        for b in batches
        for d in b.diagnostics
    ) == (filled == 0)


@pytest.mark.parametrize("boundary", ["03:30:00", "06:57:00"])
def test_continuous_session_end_is_exclusive_even_with_long_quote_gap(
    tmp_path, boundary
):
    end = pd.Timestamp("2024-01-02T" + boundary)
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=tuple(
            (end - pd.Timedelta(seconds=s)).time().isoformat() for s in (6, 3)
        ),
    ).assign(bid_vol1=100.0)
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    seconds = (end - START).total_seconds()
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(seconds - 1, "a", "sell", 130, 9.99), (seconds, "a", "sell", 1000, 9.99)],
        window=TimeWindow(panel.timestamp.min(), end + pd.Timedelta(seconds=1)),
        max_quote_gap_seconds=10_000,
    )
    assert [t.quantity for b in batches for t in b.trades] == [30]
    frozen = [
        d.number for b in batches for d in b.diagnostics if d.name == "frozen_cash"
    ]
    assert frozen[-1] == 0.0


def test_window_excludes_later_tail_print_without_forced_expiry(tmp_path):
    panel = quotes("2024-01-02", assets=("a",)).assign(bid_vol1=100.0)
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(7, "a", "sell", 300, 9.99)],
        window=TimeWindow(START, START + pd.Timedelta(seconds=6.5)),
    )
    assert not any(b.trades for b in batches)
    assert [
        d.number for b in batches for d in b.diagnostics if d.name == "frozen_cash"
    ][-1] == 1998.0


@pytest.mark.parametrize("gap,fill", [(6, 100), (0.5, 0), (None, None)])
def test_native_policy_reads_only_declared_quote_gap(tmp_path, gap, fill):
    episode = hft_episode(tmp_path / "inputs")
    episode["window"]["end"] = "2024-01-02T01:30:09"
    episode["account"]["parameters"]["passive"] = {
        "queue_model": "conservative",
        "cancel_latency_ms": 0,
    }
    original = open_dataset_session(episode["account"]["panel"])
    spec = ProcessedDataSpec(
        data_id="hft_panel",
        market_id="a_share",
        kind="panel",
        frequency="tick",
        grain="asset_timestamp",
        column_roles=original.spec["column_roles"],
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        parameters_json="{}"
        if gap is None
        else '{"max_gap_seconds": ' + str(gap) + "}",
    )
    write_dataset(tmp_path / "panel", read_dataset(original.root), spec=spec)
    episode["account"]["panel"] = str(tmp_path / "panel")
    events = trade_input(tmp_path / "trades", [(7, "a", "buy", 600, 10.01)])
    episode["account"]["trades"] = str(events.root)
    config = {
        "episode": episode,
        "strategy": "immediate",
        "output": str(tmp_path / "policy"),
    }
    if gap is None:
        with pytest.raises(ValueError, match="passive_quote_gap_required"):
            backtest_policy(config)
    else:
        output = backtest_policy(config)
        batches = list(read_portfolio_ledger_artifact(output / "ledger").iter_chunks())
        assert -sum(t.quantity for b in batches for t in b.trades) == fill
        summary = parse_qrp_json_object((output / "summary.json").read_bytes())
        assert summary["filled_quantity"] == fill


def test_direct_account_requires_an_explicit_gap_and_calendar_keeps_historical_hours():
    params = parameters(("a",))
    params["passive"] = {"queue_model": "conservative", "cancel_latency_ms": 0}
    with pytest.raises(ValueError, match="passive_quote_gap_required"):
        next(
            inventory_batches(
                [], schedule(), params, TimeWindow("2024-01-02", "2024-01-02")
            )
        )
    times = pd.Series(
        pd.to_datetime(
            ["2018-08-17T02:00:00", "2018-08-17T06:00:00", "2018-08-20T06:00:00"]
        )
    )
    exchange = pd.Series(["SSE"] * 3)
    kinds = pd.Series(["stock"] * 3)
    ends = continuous_session_end(times, exchange, kinds, break_at_lunch=True)
    assert list(ends.dt.strftime("%H:%M:%S")) == ["03:30:00", "07:00:00", "06:57:00"]
    # Existing factor consumers still ask for the daily continuous close.
    assert continuous_session_end(times, exchange, kinds).iloc[0].hour == 7


def test_tied_cancel_and_quote_age_keep_partial_fills_and_paid_commission(tmp_path):
    panel = quotes("2024-01-02", assets=("a",)).assign(bid_vol1=100.0)
    signals = pd.DataFrame(
        {
            "timestamp": [START, START + pd.Timedelta(seconds=4)],
            "asset_id": ["a", "a"],
            "prediction_score": [1.0, -1.0],
        }
    )
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [
            (5, "a", "sell", 130, 9.99),
            (12, "a", "sell", 20, 9.99),
            (12.000000001, "a", "sell", 1000, 9.99),
        ],
        minimum=5,
        cancel_ms=8000,
        window=TimeWindow(START, START + pd.Timedelta(seconds=13)),
        account_parameters={"markout_ms": 1000},
    )
    trades = [t for b in batches for t in b.trades]
    assert [t.quantity for t in trades] == [30, 20]
    assert [pd.Timestamp(t.timestamp_ns).second for t in trades] == [5, 12]
    assert (
        -sum(
            f.amount
            for b in batches
            for f in b.cashflows
            if f.kind == "commission_cost"
        )
        == 5
    )
    final = {v.asset_id: v.quantity for v in batches[-1].valuations if v.phase == "end"}
    assert final == pytest.approx({"a": 150, "__cash__": 3495.5})
    assert [
        d.number for b in batches for d in b.diagnostics if d.name == "frozen_cash"
    ][-1] == 0
    times = [b.valuations[-1].timestamp_ns for b in batches]
    assert times == sorted(times)


def test_full_day_does_not_process_after_close_prints_before_daily_close(tmp_path):
    panel = quotes("2024-01-02", assets=("a",)).assign(bid_vol1=100.0)
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(7, "a", "sell", 130, 9.99), (19801, "a", "sell", 1000, 9.99)],
        window=TimeWindow("2024-01-02", "2024-01-02"),
        minimum=5,
    )
    assert [t.quantity for b in batches for t in b.trades] == [30]
    times = [b.valuations[-1].timestamp_ns for b in batches]
    assert times == sorted(times)
    assert times[-1] == pd.Timestamp("2024-01-02T07:00:00").value


def test_tail_status_request_and_markout_follow_the_last_trade_clock(tmp_path):
    from quant_research_platform.artifacts.portfolio_ledger import (
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioCapital
    from tests.research.test_hft_status_controls import replay as status_replay
    from tests.research.test_hft_status_controls import status_source

    panel = quotes("2024-01-02", assets=("a",)).assign(bid_vol1=100.0)
    params = parameters(("a",))
    params.update(
        passive={"queue_model": "conservative", "cancel_latency_ms": 0},
        markout_ms=1000,
        minimum_commission=5,
    )
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    source = status_source(tmp_path / "status-source", panel, known_second=8)
    events = trade_input(tmp_path / "trades", [(7, "a", "sell", 130, 9.99)])
    batches = status_replay(panel, signals, params, source, events=events)
    assert [t.quantity for b in batches for t in b.trades] == [30]
    times = [b.valuations[-1].timestamp_ns for b in batches]
    assert times == sorted(times)
    assert any(
        b.valuations[-1].timestamp_ns == (START + pd.Timedelta(seconds=8)).value
        and any(d.name == "stock_status_control" for d in b.diagnostics)
        for b in batches
    )
    output = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("hft_inventory", 5000, "CNY", "cash_equity"),
        metadata={
            "kind": "a_share_hft",
            "window": {"start": "2024-01-02", "end": "2024-01-02"},
        },
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    persisted = list(read_portfolio_ledger_artifact(output).iter_chunks())
    assert len(persisted) == len(batches)
