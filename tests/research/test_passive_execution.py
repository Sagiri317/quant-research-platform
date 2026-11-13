"""Independent queue arithmetic and native shared-account partial-fill tests."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.policies.reinforcement import backtest_policy
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from tests.helpers.high_frequency import (
    daily_view,
    parameters,
    quotes,
    schedule,
    trade_input,
)
from tests.research.test_policy_learning import hft_episode


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
    account_parameters=None,
    policy=None,
    execution_order=None,
    window=None,
    max_quote_gap_seconds=6,
):
    params = parameters(tuple(panel.asset_id.unique()))
    params.update(
        passive={"queue_model": model, "cancel_latency_ms": cancel_ms},
        minimum_commission=minimum,
        initial_capital=capital,
    )
    params.update(account_parameters or {})
    events = trade_input(root / "trades", records)
    window = window or TimeWindow(
        panel.timestamp.min().isoformat(), panel.timestamp.max().isoformat()
    )
    cap = PortfolioCapital("hft_inventory", capital, "CNY", "cash_equity")

    def produce(emit):
        stream = inventory_batches(
            [(daily_view(panel), panel, signals)],
            schedule(),
            params,
            window,
            trade_session=events,
            decision_columns=("mid",) if policy is not None else None,
            execution_order=execution_order,
            max_quote_gap_seconds=max_quote_gap_seconds,
        )
        try:
            event = next(stream)
            while True:
                if isinstance(event, PolicyDecision):
                    event = stream.send(policy(event))
                else:
                    emit(event)
                    event = next(stream)
        except StopIteration:
            pass
        finally:
            stream.close()

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


def test_cash_limited_passive_order_keeps_native_ledger_quantities(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:09"),
    )
    panel["bid_vol1"] = 100.0
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, summary = replay(
        tmp_path,
        panel,
        signals,
        [(4, "a", "sell", 130, 9.99), (5, "a", "sell", 100, 9.99)],
        minimum=5,
        capital=2504,
    )
    # Cash of 1504 covers 100 shares at 9.99 plus 5 commission, but not 200.
    assert [trade.quantity for batch in batches for trade in batch.trades] == [30, 70]
    assert summary["nav"] == pytest.approx(2500)
    assert min(
        value.value
        for batch in batches
        for value in batch.valuations
        if value.asset_id == "__cash__"
    ) == pytest.approx(500)


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


@pytest.mark.parametrize("interactive", [False, True])
def test_cancel_latency_includes_tied_trade_and_preserves_t1(tmp_path, interactive):
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
            (4, "a", "sell", 130, 9.99),
            (5, "a", "sell", 20, 9.99),
            (5.5, "a", "sell", 1000, 9.99),
            (7, "a", "buy", 1000, 10.01),
        ],
        cancel_ms=2000,
        minimum=5,
        account_parameters={"latency_ms": 10},
        policy=(
            lambda event: pd.Series(
                {"a": 300.0 if event.timestamp == panel.timestamp.iloc[0] else 0.0}
            )
        )
        if interactive
        else None,
    )
    # Cancel requested at 3 s acknowledges at 5 s, not at the 6 s quote + 2 s.
    # A 130-share print consumes 100 ahead and fills 30; tied 5 s fills 20 more.
    trades = [t for b in batches for t in b.trades]
    assert [t.quantity for t in trades] == [30, 20, -100]
    assert [pd.Timestamp(t.timestamp_ns).second for t in trades] == [4, 5, 7]
    requests = [
        b
        for b in batches
        if any(d.name == "passive_cancel_requested" for d in b.diagnostics)
    ]
    assert [b.valuations[0].timestamp_ns for b in requests] == [
        panel.timestamp.iloc[1].value
    ]
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
            max_quote_gap_seconds=6,
        )
    )
    assert panel.segment_id.map(type).eq(str).all()
    assert [trade.quantity for batch in batches for trade in batch.trades] == [200]


def test_passive_depth_and_markouts_reuse_native_partial_fill_ledger(tmp_path):
    panel = quotes("2024-01-02", assets=("a",)).assign(bid_px2=9.98, bid_vol2=100.0)
    panel.loc[2, ["mid", "bid_px1", "ask_px1"]] = [10.5, 10.49, 10.51]
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(4, "a", "sell", 130, 9.98), (5, "a", "sell", 20, 9.98)],
        minimum=5,
        account_parameters={
            "passive": {
                "queue_model": "conservative",
                "cancel_latency_ms": 0,
                "quote_level": 2,
            },
            "markout_ms": 1000.0,
        },
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [30, 20]
    assert [trade.notional for trade in trades] == pytest.approx([30 * 9.98, 20 * 9.98])
    results = [
        row
        for batch in batches
        for row in batch.diagnostics
        if row.name == "markout_bps"
    ]
    assert [row.fill_id for row in results] == [trade.fill_id for trade in trades]
    assert [row.number for row in results] == pytest.approx(
        [(10.5 / 9.98 - 1) * 10000] * 2
    )
    assert (
        sum(
            -flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "commission_cost"
        )
        == 5.0
    )


def test_passive_owner_rejects_replayed_trade_but_accepts_distinct_same_time_print():
    from datetime import date
    from types import SimpleNamespace

    from quant_research_platform.contracts.portfolios import ExecutionAssumptions
    from quant_research_platform.research.portfolios.passive import PassiveOrders

    panel = quotes("2024-01-02", assets=("a",)).assign(bid_px2=9.98, bid_vol2=50.0)
    at = panel.timestamp.iloc[1]
    orders = PassiveOrders(
        {"queue_model": "conservative", "cancel_latency_ms": 0, "quote_level": 2},
        pd.Series({"a": 4}),
        date(2024, 1, 2),
        schedule(),
        ExecutionAssumptions(5000.0, 0.0, 0.0, 1.0, (1.0,), 0.0),
        1.0,
        max_quote_gap_seconds=6,
    )
    orders.request(panel.timestamp.iloc[0], {"a": 300.0}, {"a": at.value})
    orders.quote(
        at,
        panel.iloc[[1]].set_index("asset_id"),
        pd.Series({"a": 100.0}),
        pd.Series({"a": 100.0}),
        4000.0,
    )
    event = SimpleNamespace(
        timestamp=at + pd.Timedelta(seconds=1),
        source_row=0,
        asset_id="a",
        side="sell",
        price=9.98,
        volume=70.0,
    )
    assert orders.trade(event)[2] == 20
    with pytest.raises(ValueError, match="passive_trade_event_reused_or_out_of_order"):
        orders.trade(event)
    event.source_row = 1
    assert orders.trade(event)[2] == 70
    assert orders.orders["a"].remaining == 110


def test_passive_unobserved_quote_level_blocks_instead_of_falling_back(tmp_path):
    panel = quotes("2024-01-02", assets=("a",))
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(4, "a", "sell", 10000, 9.99)],
        account_parameters={
            "passive": {
                "queue_model": "conservative",
                "cancel_latency_ms": 0,
                "quote_level": 2,
            }
        },
    )
    assert not any(batch.trades for batch in batches)
    assert any(
        row.text == "passive_depth_unavailable"
        for batch in batches
        for row in batch.diagnostics
    )


@pytest.mark.parametrize("level", [0, 11, True, 1.5, "2"])
def test_passive_quote_level_validation(level):
    from quant_research_platform.research.portfolios.passive import passive_parameters

    with pytest.raises(ValueError, match="passive_quote_level_invalid"):
        passive_parameters(
            {
                "queue_model": "conservative",
                "cancel_latency_ms": 0,
                "quote_level": level,
            }
        )


def test_maker_taker_comparison_keeps_same_mother_order_and_separate_cash_ledgers(
    tmp_path,
):
    from copy import deepcopy

    episode = hft_episode(tmp_path / "inputs")
    maker = deepcopy(episode)
    maker["account"]["parameters"]["passive"] = {
        "queue_model": "conservative",
        "cancel_latency_ms": 0,
    }
    trade_input(tmp_path / "events", [(4, "a", "buy", 530, 10.01)])
    maker["account"]["trades"] = str(tmp_path / "events")
    summaries = []
    for name, branch in (("taker", episode), ("maker", maker)):
        branch["account"]["candidate_id"] = name
        output = backtest_policy(
            {
                "episode": branch,
                "strategy": "immediate",
                "output": str(tmp_path / name),
            }
        )
        summaries.append(parse_qrp_json_object((output / "summary.json").read_bytes()))
        ledger = read_portfolio_ledger_artifact(output / "ledger")
        batches = list(ledger.iter_chunks())
        initial = {
            row.asset_id: row.quantity
            for row in batches[0].valuations
            if row.phase == "before"
        }
        assert initial == {"__cash__": 4000.0, "a": 100.0}
        assert ledger.capital.initial_capital == 5000.0
    artifact = read_statistics_artifact(
        portfolio_statistics(
            {
                "ledgers": [
                    str(tmp_path / name / "ledger") for name in ("taker", "maker")
                ],
                "window": episode["window"],
                "output": str(tmp_path / "comparison"),
                "analyses": ["performance"],
            }
        )
    )
    with artifact.table("portfolio_performance.parquet") as performance:
        values = (
            performance.loc[performance.metric_id.eq("net_pnl")]
            .set_index("candidate_id")
            .value
        )
        # The taker pays one yuan of spread; the maker earns 0.30 yuan on 30 shares.
        assert values.to_dict() == pytest.approx({"taker": -1.0, "maker": 0.3})
    assert episode["order"] == maker["order"] == {"asset_id": "a", "quantity": -100}
    assert [row["filled_quantity"] for row in summaries] == [100.0, 30.0]
    assert [row["unfilled_quantity"] for row in summaries] == [0.0, 70.0]
    assert [row["opportunity_cost_bps"] for row in summaries] == [0.0, 0.0]
    assert summaries[1]["total_implementation_shortfall_bps"] == pytest.approx(-3.0)


@pytest.mark.parametrize("missing,expected", [("bid_vol1", 0), ("ask_vol1", 200)])
def test_passive_submission_requires_own_depth_not_opposite_side_volume(
    tmp_path, missing, expected
):
    panel = quotes("2024-01-02", assets=("a",))
    panel[missing] = float("nan")
    panel["eligible"] = False
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(tmp_path, panel, signals, [(4, "a", "sell", 1000, 9.99)])
    assert (
        sum(trade.quantity for batch in batches for trade in batch.trades) == expected
    )


@pytest.mark.parametrize("price", [9.99, 10.02])
def test_passive_deeper_level_cannot_duplicate_best_or_cross_book(tmp_path, price):
    panel = quotes("2024-01-02", assets=("a",)).assign(bid_px2=price, bid_vol2=100.0)
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [(4, "a", "sell", 1000, price)],
        account_parameters={
            "passive": {
                "queue_model": "conservative",
                "cancel_latency_ms": 0,
                "quote_level": 2,
            }
        },
    )
    assert not any(batch.trades for batch in batches)
    assert any(
        row.text == "passive_depth_unavailable"
        for batch in batches
        for row in batch.diagnostics
    )


@pytest.mark.parametrize(
    "cancel_ms,fill_times", [(0, (0.011, 0.011)), (2, (0.012, 0.013))]
)
def test_off_quote_cancel_precedes_later_print_without_early_replacement(
    tmp_path, cancel_ms, fill_times
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:00.010", "01:30:00.020", "01:30:00.030"),
    )
    panel["bid_vol1"] = 100.0
    signals = pd.DataFrame(
        {
            "timestamp": [
                panel.timestamp.iloc[0],
                pd.Timestamp("2024-01-02T01:30:00.011"),
            ],
            "asset_id": ["a", "a"],
            "prediction_score": [1.0, -1.0],
        }
    )
    batches, summary = replay(
        tmp_path,
        panel,
        signals,
        [
            (0.010, "a", "sell", 10000, 9.99),
            (fill_times[0], "a", "sell", 130, 9.99),
            (fill_times[1], "a", "sell", 20, 9.99),
            (0.014, "a", "sell", 1000, 9.99),
            (0.020, "a", "buy", 1000, 10.01),
        ],
        cancel_ms=cancel_ms,
        account_parameters={"latency_ms": 10},
    )
    # Cancellation acknowledges at 11 or 13 ms; tied prints still precede it.
    # The replacement is ready at 21 ms and waits for the 30 ms quote.
    # Neither the late 14 ms print nor the premature 20 ms print can fill.
    trades = [t for b in batches for t in b.trades]
    assert [t.quantity for t in trades] == [30, 20]
    assert summary["nav"] == pytest.approx(5000.5)
    submitted = [
        next(v.timestamp_ns for v in b.valuations if v.phase == "end")
        for b in batches
        if any(d.name == "passive_order_submitted" for d in b.diagnostics)
    ]
    assert submitted == [t.value for t in panel.timestamp.iloc[[1, 3]]]


def test_repeated_pending_target_keeps_first_activation_and_mother_order(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:00.005",
            "01:30:00.010",
            "01:30:00.015",
            "01:30:00.020",
        ),
    )
    panel["ask_vol1"] = 0.0
    batches, _ = replay(
        tmp_path,
        panel,
        pd.DataFrame(),
        [
            (0.011, "a", "buy", 50, 10.01),
            (0.016, "a", "buy", 1000, 10.01),
            (0.019, "a", "sell", 1000, 9.99),
        ],
        account_parameters={"latency_ms": 10, "initial_inventory": {"a": 300}},
        policy=lambda event: pd.Series({"a": 100.0}),
        execution_order={"asset_id": "a", "quantity": -200},
    )
    assert [t.quantity for b in batches for t in b.trades] == [-50, -150]
    assert (
        next(
            v.quantity
            for v in batches[-1].valuations
            if v.asset_id == "a" and v.phase == "end"
        )
        == 100
    )


@pytest.mark.parametrize(
    "restore,expected", [(None, [30, 170]), ("14:50:00", [30, 170, -100])]
)
def test_missing_predictions_preserve_orders_but_do_not_disable_restore(
    tmp_path, restore, expected
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:03",
            "01:30:06",
            "01:30:09",
            "06:50:00",
            "06:50:00.010",
            "06:50:03",
        ),
    )
    panel["bid_vol1"] = 100.0
    panel.loc[5:, "segment_id"] = 1
    signals = panel[["timestamp", "asset_id"]].assign(prediction_score=float("nan"))
    signals.loc[0, "prediction_score"] = 1.0
    batches, _ = replay(
        tmp_path,
        panel,
        signals,
        [
            (4, "a", "sell", 130, 9.99),
            (7, "a", "sell", 170, 9.99),
            (19201, "a", "buy", 1000, 10.01),
        ],
        account_parameters={"latency_ms": 10, "restore_at": restore},
    )
    assert [t.quantity for b in batches for t in b.trades] == expected
    assert not any(
        d.name == "passive_cancel_requested" for b in batches for d in b.diagnostics
    )
