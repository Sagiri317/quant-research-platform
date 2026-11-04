"""Independent queue arithmetic and native shared-account partial-fill tests."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.policies.reinforcement import backtest_policy
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.helpers.high_frequency import daily_view
from tests.research.test_hft_portfolio import parameters, quotes, schedule
from tests.research.test_policy_learning import hft_episode


def trade_input(root, records):
    frame = pd.DataFrame(
        records, columns=["second", "asset_id", "side", "volume", "price"]
    )
    frame["timestamp"] = pd.Timestamp("2024-01-02T01:30:00") + pd.to_timedelta(
        frame.pop("second"), unit="s"
    )
    frame["source_row"] = range(len(frame))
    frame["event_kind"] = "trade"
    roles = dict.fromkeys(frame, "context")
    roles.update(timestamp="timestamp", asset_id="key", source_row="key")
    write_dataset(
        root,
        frame,
        spec=InterimDataSpec(
            data_id="hft_trades",
            source_id="synthetic",
            frequency="tick",
            grain="trade_event",
            primary_key=("timestamp", "asset_id", "source_row"),
            column_roles=roles,
            column_units={"price": "CNY", "volume": "shares"},
            partition_columns=(),
        ),
    )
    return open_dataset_session(root)


def replay(
    root,
    panel,
    signals,
    records,
    *,
    model="conservative",
    cancel_ms=0,
    minimum=0,
    capital=5000,
):
    params = parameters(tuple(panel.asset_id.unique()))
    params.update(
        passive={"queue_model": model, "cancel_latency_ms": cancel_ms},
        minimum_commission=minimum,
        initial_capital=capital,
    )
    events = trade_input(root / "trades", records)
    window = TimeWindow(
        panel.timestamp.min().isoformat(), panel.timestamp.max().isoformat()
    )
    cap = PortfolioCapital("hft_inventory", capital, "CNY", "cash_equity")

    def produce(emit):
        for batch in inventory_batches(
            [(daily_view(panel), panel, signals)],
            schedule(),
            params,
            window,
            trade_session=events,
        ):
            emit(batch)

    output = write_portfolio_ledger_artifact(
        root / "ledger",
        capital=cap,
        metadata={
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "kind": "a_share_hft",
        },
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(output)
    batches = list(artifact.iter_chunks())
    return batches, summarize_portfolio_ledger(cap, iter(batches))


def test_partial_fills_consume_each_print_once_and_charge_one_commission(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a", "b"),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:09"),
    )
    panel["bid_vol1"] = 100.0
    signals = panel.loc[
        panel.timestamp.eq(panel.timestamp.min()), ["timestamp", "asset_id"]
    ].assign(prediction_score=1.0)
    batches, summary = replay(
        tmp_path,
        panel,
        signals,
        [
            (3, "a", "sell", 10000, 9.99),
            (4, "a", "sell", 130, 9.99),
            (5, "a", "sell", 20, 9.99),
            (7, "a", "sell", 150, 9.99),
            (8, "b", "sell", 10000, 9.99),
        ],
        minimum=5,
        capital=4005,
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [(t.asset_id, t.quantity) for t in trades] == [
        ("a", 30),
        ("a", 20),
        ("a", 150),
    ]
    assert len({t.order_id for t in trades}) == 1
    assert all(
        t.timestamp_ns > pd.Timestamp("2024-01-02T01:30:03").value for t in trades
    )
    assert (
        sum(
            -f.amount
            for b in batches
            for f in b.cashflows
            if f.kind == "commission_cost"
        )
        == 5
    )
    assert summary["nav"] == pytest.approx(4002)
    frozen = [d.number for d in batches[2].diagnostics if d.name == "frozen_cash"]
    assert frozen == [2003]
    assert (
        min(v.value for b in batches for v in b.valuations if v.asset_id == "__cash__")
        >= 0
    )


@pytest.mark.parametrize(
    "model,expected", [("conservative", []), ("probability", [10])]
)
def test_probability_cancellation_excludes_trade_depletion(tmp_path, model, expected):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:09", "01:30:12"),
    )
    panel["bid_vol1"] = [100, 100, 80, 40, 40]
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(4, "a", "sell", 40, 9.99), (10, "a", "sell", 40, 9.99)],
        model=model,
    )
    # After 40 traded, 60 remain ahead. Depth rises to 80, then 40 cancel:
    # uniform cancellation removes 40 * 60/80 = 30 ahead. The next 40 fill 10.
    assert [t.quantity for b in batches for t in b.trades] == expected


def test_cancel_latency_includes_tied_trade_and_preserves_t1(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:09", "01:30:12"),
    )
    panel["bid_vol1"] = 100.0
    signals = panel.iloc[:2][["timestamp", "asset_id"]].assign(
        prediction_score=[1.0, -1.0]
    )
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [
            (7, "a", "sell", 130, 9.99),
            (8, "a", "sell", 20, 9.99),
            (8.5, "a", "sell", 1000, 9.99),
            (10, "a", "buy", 1000, 10.01),
        ],
        cancel_ms=2000,
        minimum=5,
    )
    assert [t.quantity for b in batches for t in b.trades] == [30, 20, -100]
    last = {v.asset_id: v.quantity for v in batches[-1].valuations if v.phase == "end"}
    assert last["a"] == 50
    assert (
        sum(
            -f.amount
            for b in batches
            for f in b.cashflows
            if f.kind == "commission_cost"
        )
        == 10
    )
    assert any(
        d.name == "passive_cancelled_unfilled_quantity" and d.number == 150
        for b in batches
        for d in b.diagnostics
    )


def test_gap_cannot_fill_and_final_open_order_remains_unfinished(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:30", "01:30:33"),
    )
    panel["segment_id"] = [0, 0, 0, 1, 1]
    signals = panel.iloc[[0, 3]][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(20, "a", "sell", 10000, 9.99), (33, "a", "sell", 10000, 9.99)],
    )
    assert not any(b.trades for b in batches)
    assert any(
        d.name == "passive_order_observation_gap"
        for b in batches
        for d in b.diagnostics
    )
    assert any(
        d.name == "passive_unfilled_quantity" and d.number == 200
        for d in batches[-1].diagnostics
    )


def test_native_policy_uses_passive_shared_account_and_keeps_unfinished_order(tmp_path):
    episode = hft_episode(tmp_path / "inputs")
    episode["account"]["parameters"]["passive"] = {
        "queue_model": "conservative",
        "cancel_latency_ms": 0,
    }
    trade_input(tmp_path / "events", [(4, "a", "buy", 530, 10.01)])
    episode["account"]["trades"] = str(tmp_path / "events")
    root = backtest_policy(
        {
            "episode": episode,
            "strategy": "immediate",
            "output": str(tmp_path / "policy"),
        }
    )
    summary = parse_qrp_json_object((root / "summary.json").read_bytes())
    assert summary["completion_rate"] == pytest.approx(0.3)


def test_native_book_segments_and_unquoted_unheld_asset_allow_passive_fill(tmp_path):
    from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
        build_panel,
    )
    from quant_research_platform.sources.venvoo import normalize
    from tests.markets.a_share.test_high_frequency import (
        INSTRUMENTS,
        raw_events,
        raw_quotes,
    )

    panel = build_panel(
        normalize(raw_quotes().assign(date=20240102), "hft_quotes", INSTRUMENTS),
        normalize(
            raw_events("hft_orders").assign(date=20240102), "hft_orders", INSTRUMENTS
        ),
        normalize(
            raw_events("hft_trades").assign(date=20240102), "hft_trades", INSTRUMENTS
        ),
        max_gap_seconds=6,
    )
    asset = INSTRUMENTS[0]["asset_id"]
    params = parameters((asset, "unlisted"))
    params["initial_inventory"]["unlisted"] = 0
    params["passive"] = {"queue_model": "conservative", "cancel_latency_ms": 0}
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    events = trade_input(tmp_path / "events", [(4, asset, "sell", 1000, 9.99)])
    batches = list(
        inventory_batches(
            [(daily_view(panel), panel, signals)],
            schedule(),
            params,
            TimeWindow(
                panel.timestamp.min().isoformat(), panel.timestamp.max().isoformat()
            ),
            trade_session=events,
        )
    )
    assert panel.segment_id.map(type).eq(str).all()
    assert [trade.quantity for batch in batches for trade in batch.trades] == [200]
