"""PPO execution uses causal predictions, one order book, and native cash economics."""

import pickle
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    write_prediction_outputs,
)
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.policies import PolicyDecision, PolicyExecution
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.policies.environments import (
    TradingEnv,
    _trading_seconds,
)
from quant_research_platform.research.policies.inventory_policy import inventory_targets
from quant_research_platform.research.policies.reinforcement import (
    backtest_policy,
    train_policy,
)
from quant_research_platform.research.portfolios import high_frequency
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.high_frequency import (
    daily_view,
    memory_account,
    parameters,
    quotes,
    trade_input,
    write_daily_inputs,
)
from tests.helpers.portfolio_inputs import _interim_specs
from tests.research.test_etf_inventory_policy import etf_schedule, forecast_metadata


def quote_spec(panel):
    roles = dict.fromkeys(panel, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    return ProcessedDataSpec(
        data_id="execution_quotes",
        market_id="a_share",
        kind="panel",
        frequency="tick",
        grain="asset_timestamp",
        column_roles=roles,
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        parameters_json='{"max_gap_seconds": 6}',
    )


@pytest.mark.parametrize(
    "initial,limit,capital,first_target,replacement_target,capacity,filled",
    [
        (300, 300, 10_000, 150, 0, 500, -100),
        (0, 400, 10_000, 200, 400, 500, 200),
        (0, 400, 10_000, 200, 400, 100, 100),
        (0, 200, 1006, 100.6, 100.6, 500, 100),
    ],
)
def test_inflight_replacement_cannot_block_or_change_prior_ioc_economics(
    monkeypatch,
    initial,
    limit,
    capital,
    first_target,
    replacement_target,
    capacity,
    filled,
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:00.020",
            "01:30:03",
            "01:30:04.001",
            "01:30:04.501",
            "01:30:06",
            "01:30:09",
            "01:30:12",
        ),
    ).assign(instrument_type="gold_etf")
    panel.loc[
        panel.timestamp.eq(pd.Timestamp("2024-01-02T01:30:04.001")),
        ["bid_vol1", "ask_vol1"],
    ] = capacity
    predictions = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02T01:30:00.001")
            + pd.to_timedelta([0, 3, 6, 9], unit="s"),
            "asset_id": "a",
            "prediction_score": 0.1 if filled > 0 else -0.1,
        }
    )
    empty_trades = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "asset_id": pd.Series(dtype="str"),
            "source_row": pd.Series(dtype="int64"),
            "event_kind": pd.Series(dtype="str"),
            "side": pd.Series(dtype="str"),
            "price": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    # Only the explicit I/O boundary is replaced; matching and accounting stay native.
    monkeypatch.setattr(
        high_frequency, "materialize_dataset_session", lambda _: empty_trades
    )
    params = parameters(("a",))
    params.update(
        initial_capital=capital,
        initial_inventory={"a": initial},
        maximum_weights={"a": min(1.0, limit * 10 / capital)},
        latency_ms=4000,
        minimum_commission=5,
        passive={"queue_model": "conservative", "cancel_latency_ms": 1500},
    )
    for replacement in (False, True):
        stream = high_frequency.inventory_batches(
            [(daily_view(panel), panel, predictions)],
            etf_schedule(),
            params,
            TimeWindow("2024-01-02T01:30:00", "2024-01-02T01:30:12"),
            decision_columns=("prediction_score",),
            prediction_decisions=True,
            prediction_task="dynamic_execution",
            quote_visibility_delay_ms=1,
            trade_session=SimpleNamespace(filtered_view=lambda _: None),
            max_quote_gap_seconds=6,
        )
        batches = []
        decision_count = 0
        try:
            event = next(stream)
            while True:
                if isinstance(event, PolicyDecision):
                    target = first_target if decision_count == 0 else replacement_target
                    mode = (
                        "taker"
                        if decision_count == 0
                        else "maker"
                        if decision_count == 1 and replacement
                        else "wait"
                    )
                    event = stream.send(
                        PolicyExecution(
                            pd.Series(
                                {
                                    "a": min(
                                        target,
                                        max(
                                            event.quantities.a,
                                            event.maximum_quantities.a,
                                        ),
                                    )
                                }
                            ),
                            pd.Series({"a": mode}),
                            pd.Series(
                                {
                                    "a": min(
                                        replacement_target,
                                        max(
                                            event.quantities.a,
                                            event.maximum_quantities.a,
                                        ),
                                    )
                                }
                            ),
                        )
                    )
                    decision_count += 1
                else:
                    batches.append(event)
                    event = next(stream)
        except StopIteration:
            pass
        finally:
            stream.close()
        trades = [trade for batch in batches for trade in batch.trades]
        assert [trade.quantity for trade in trades] == [filled]
        assert trades[0].timestamp_ns == pd.Timestamp("2024-01-02T01:30:04.001").value
        terminal = {
            row.asset_id: row for row in batches[-1].valuations if row.phase == "end"
        }
        price = 10.01 if filled > 0 else 9.99
        expected_cash = capital - initial * 10 - filled * price - 5
        assert terminal["__cash__"].value == pytest.approx(expected_cash)
        assert sum(row.value for row in terminal.values()) == pytest.approx(
            expected_cash + (initial + filled) * 10
        )


def execution_episode(
    root,
    *,
    kinds=("gold_etf",),
    records=(),
    scores=(0.1, 0.1, 0.1, 0.1),
    capital=None,
    cancel_ms=1000,
    bid_depth=10.0,
):
    assets = tuple("ab"[: len(kinds)])
    clocks = (
        "01:30:00",
        "01:30:00.0005",
        "01:30:00.020",
        "01:30:01",
        "01:30:03",
        "01:30:03.020",
        "01:30:04.001",
        "01:30:06",
        "01:30:06.020",
        "01:30:07.001",
        "01:30:09",
        "01:30:09.020",
        "01:30:12",
    )
    panel = pd.concat(
        [
            quotes(day, assets=assets, clocks=clocks)
            for day in ("2024-01-02", "2024-01-03", "2024-01-04")
        ],
        ignore_index=True,
    )
    panel["instrument_type"] = panel.asset_id.map(dict(zip(assets, kinds, strict=True)))
    panel["bid_vol1"] = bid_depth
    daily = write_daily_inputs(root / "daily", panel)
    write_dataset(root / "quotes", panel, spec=quote_spec(panel))
    fees = pd.DataFrame([asdict(rule) for rule in etf_schedule().rules])
    fees["effective_date"] = pd.to_datetime(fees.effective_date)
    fees["end_date"] = pd.to_datetime(fees.end_date)
    fees["included_in"] = fees.included_in.astype("string")
    write_dataset(
        root / "fees", fees, spec=_interim_specs().get("a_share_fee_schedule")
    )
    trade_input(root / "trades", records or [(100, assets[0], "sell", 1, 9.99)])
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(day)
                + pd.Timedelta(hours=1, minutes=30, seconds=second, milliseconds=1),
                "asset_id": asset,
                "prediction_score": score,
                "prediction_rank_score": 0.0,
            }
            for day in ("2024-01-02", "2024-01-03", "2024-01-04")
            for second, score in zip((0, 3, 6, 9), scores, strict=True)
            for asset in assets
        ]
    )
    metadata = forecast_metadata()
    metadata["prediction_window"] = {"start": "2024-01-02", "end": "2024-01-04"}
    prediction = write_prediction_outputs(
        output_path=root / "predictions", output_chunks=[frame], metadata=metadata
    )
    parameters = {
        "initial_capital": 1_000_000 * len(assets) if capital is None else capital,
        "restore_at": "14:50:00",
        "initial_inventory": dict.fromkeys(assets, 100),
        "maximum_weights": dict.fromkeys(
            assets, min(1.0, 3000 / (capital or 1_000_000 * len(assets)))
        ),
        "latency_ms": 10,
        "participation_rate": 1,
        "minimum_commission": 5,
        "passive": {"queue_model": "conservative", "cancel_latency_ms": cancel_ms},
    }
    return {
        "kind": "hft",
        "task": "dynamic_execution",
        "asset_ids": list(assets),
        "feature_columns": ["prediction_score", "mid", "bid_vol1", "ask_vol1"],
        "window": {"start": "2024-01-02T01:30:00", "end": "2024-01-02T01:30:12"},
        "prediction": {"paths": [str(prediction)]},
        "account": {
            "panel": str(root / "quotes"),
            "trades": str(root / "trades"),
            "fee_schedule": str(root / "fees"),
            "daily_inputs": daily,
            "parameters": parameters,
        },
        "inventory_parameters": {"trade_cost_bps": 0, "exit_cost_bps": 0},
        "risk_aversion": 0,
        "target_deviation_bps": 0,
        "unfinished_penalty_bps": 0,
    }


def replay(episode, actions):
    batches = []
    env = TradingEnv([episode], on_batch=batches.append)
    decisions, rewards = [], []
    try:
        env.reset(seed=0)
        initial = env.nav
        for action in actions:
            decisions.append(env.decision)
            _, reward, done, _, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        assert done
        return decisions, batches, rewards, info, initial
    finally:
        env.close()


def test_half_capital_and_fixed_reference_survive_unfinished_restoration(monkeypatch):
    panels = []
    signals = []
    for day, price in (("2024-01-02", 10.0), ("2024-01-03", 20.0)):
        panel = quotes(
            day,
            assets=("a",),
            mid=price,
            clocks=("01:30:00", "01:30:00.020", "06:49:59", "06:50:00.020"),
        ).assign(instrument_type="gold_etf")
        if not panels:
            panel.loc[panel.timestamp.dt.hour.eq(6), "bid_vol1"] = 0.0
        panels.append(panel)
        signals.append(
            pd.DataFrame(
                {
                    "timestamp": [
                        pd.Timestamp(day)
                        + pd.Timedelta(hours=1, minutes=30, milliseconds=1)
                    ],
                    "asset_id": ["a"],
                    "prediction_score": [0.1 if day.endswith("02") else np.nan],
                }
            )
        )
    params = parameters(("a",), restore_at="14:50:00")
    del params["initial_inventory"], params["buy_threshold"], params["sell_threshold"]
    params.update(
        initial_weights={"a": 0.5},
        initial_capital=4500,
        maximum_weights={"a": 2 / 3},
        latency_ms=10,
        passive={"queue_model": "conservative", "cancel_latency_ms": 10},
    )
    episode = {
        "kind": "hft",
        "task": "dynamic_execution",
        "asset_ids": ["a"],
        "feature_columns": ["prediction_score"],
        "window": {"start": "2024-01-02", "end": "2024-01-03"},
        "prediction": {"paths": ["memory"]},
        "account": memory_account(
            monkeypatch,
            [
                (daily_view(panel, listing=pd.Timestamp("2024-01-02")), panel, signal)
                for panel, signal in zip(panels, signals, strict=True)
            ],
            etf_schedule(),
            params,
        ),
        "inventory_parameters": {"trade_cost_bps": 0, "exit_cost_bps": 0},
        "risk_aversion": 0,
        "target_deviation_bps": 0,
        "unfinished_penalty_bps": 0,
    }

    decisions, batches, _, _, _ = replay(episode, [[7], [0]])
    # 4500 * 50% / 10 = 225 shares, rounded down to two lots.
    first = {v.asset_id: v for v in batches[0].valuations if v.phase == "before"}
    assert first["a"].quantity == 200
    assert first["__cash__"].value == 2500
    assert all(d.reference_quantities.a == 200 for d in decisions)
    second_day = next(d for d in decisions if d.timestamp.day == 3)
    assert second_day.quantities.a == 300
    assert second_day.marks.a == 20
    assert [t.quantity for b in batches for t in b.trades] == [100, -100]
    assert [
        d.text
        for b in batches
        for d in b.diagnostics
        if d.name == "inventory_restoration_status"
    ] == ["INCOMPLETE", "COMPLETE"]


@pytest.mark.parametrize("kind", ["gold_etf", "equity_etf"])
def test_native_restoration_overrides_wait_and_respects_cancellation_and_t1(
    monkeypatch, kind
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "06:49:53",
            "06:49:53.020",
            "06:49:56",
            "06:49:56.020",
            "06:49:59",
            "06:50:00.020",
            "06:50:00.500",
            "06:50:01.001",
            "06:50:01.020",
        ),
    ).assign(instrument_type=kind, bid_vol1=1000.0, ask_vol1=1000.0)
    panel.loc[panel.timestamp.ge(pd.Timestamp("2024-01-02T06:49:56")), "bid_vol1"] = (
        100.0
    )
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T06:49:53.001",
                    "2024-01-02T06:49:56.001",
                    "2024-01-02T06:49:59.001",
                    "2024-01-02T06:50:01.002",
                ]
            ),
            "asset_id": "a",
            "prediction_score": [-0.1, 0.1, 0.1, 0.1],
        }
    )
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T06:49:57", "2024-01-02T06:50:00.500"], format="mixed"
            ),
            "asset_id": "a",
            "source_row": [0, 1],
            "event_kind": "trade",
            "side": "sell",
            "price": 9.99,
            "volume": [700.0, 100.0],
        }
    )
    params = parameters(("a",), restore_at="14:50:00")
    del params["buy_threshold"], params["sell_threshold"]
    params.update(
        initial_capital=10_000,
        initial_inventory={"a": 500},
        maximum_weights={"a": 1.0},
        latency_ms=10,
        passive={"queue_model": "conservative", "cancel_latency_ms": 1000},
    )
    episode = {
        "kind": "hft",
        "task": "dynamic_execution",
        "asset_ids": ["a"],
        "feature_columns": ["prediction_score"],
        "window": {"start": "2024-01-02", "end": "2024-01-02"},
        "prediction": {"paths": ["memory"]},
        "account": memory_account(
            monkeypatch,
            [(daily_view(panel), panel, signals)],
            etf_schedule(),
            params,
            trades=events,
        ),
        "inventory_parameters": {"trade_cost_bps": 0, "exit_cost_bps": 0},
        "risk_aversion": 0,
        "target_deviation_bps": 10,
        "unfinished_penalty_bps": 0,
    }

    decisions, batches, rewards, _, _ = replay(episode, [[7], [4], [0]])
    assert len(decisions) == 3
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == (
        [-500, 600, 100, -100, -100] if kind == "gold_etf" else [-500, 600, 100]
    )
    if kind == "gold_etf":
        assert trades[-2].timestamp_ns == pd.Timestamp("2024-01-02T06:50:01.001").value
    assert any(
        d.name == "passive_cancel_requested" for b in batches for d in b.diagnostics
    )
    assert [
        d.text
        for b in batches
        for d in b.diagnostics
        if d.name == "inventory_restoration_status"
    ] == ["COMPLETE" if kind == "gold_etf" else "INCOMPLETE"]
    last = next(
        v for v in batches[-1].valuations if v.phase == "end" and v.asset_id == "a"
    )
    assert last.quantity == (500 if kind == "gold_etf" else 700)

    # Integrate the last action from 14:49:59.001: the account changes its
    # target to 500 at 14:50, before cancellation and the subsequent fills.
    squared_share_seconds = 400.1**2 * 0.999 + 100**2 * 0.5
    squared_share_seconds += (
        200**2 * 0.501 + 100**2 * 0.019 if kind == "gold_etf" else 200**2 * 599.5
    )
    penalty = 10 * (10 / 10_000) ** 2 * squared_share_seconds / 14400
    assert rewards[-1] == pytest.approx((-1 if kind == "gold_etf" else 1) - penalty)


def test_half_capital_is_saved_as_actual_shares_in_native_ledger(tmp_path):
    episode = execution_episode(tmp_path / "inputs", capital=4500)
    params = episode["account"]["parameters"]
    del params["initial_inventory"]
    params["initial_weights"] = {"a": 0.5}
    output = backtest_policy(
        {
            "strategy": "immediate",
            "episode": episode,
            "output": str(tmp_path / "result"),
        }
    )
    ledger = read_portfolio_ledger_artifact(output / "ledger")
    with closing_iterator(ledger.iter_chunks()) as batches:
        before = {
            v.asset_id: v for v in next(batches).valuations if v.phase == "before"
        }
    assert before["a"].quantity == 200
    assert before["__cash__"].value == 2500


@pytest.mark.parametrize("task", ["inventory", "execution"])
def test_policy_can_hold_or_reduce_an_existing_position_above_money_cap(
    monkeypatch, task
):
    panel = quotes(
        "2024-01-02", assets=("a",), clocks=("01:29:59", "01:30:03", "01:30:06")
    ).assign(instrument_type="gold_etf")
    panel.loc[panel.timestamp.dt.minute.eq(30), ["mid", "bid_px1", "ask_px1"]] = [
        20,
        19.99,
        20.01,
    ]
    params = parameters(("a",))
    del params["buy_threshold"], params["sell_threshold"]
    params.update(
        initial_capital=10_000, initial_inventory={"a": 500}, maximum_weights={"a": 0.5}
    )
    episode = {
        "kind": "hft",
        "task": task,
        "asset_ids": ["a"],
        "feature_columns": ["mid"],
        "window": {"start": "2024-01-02T01:30:00", "end": "2024-01-02T01:30:06"},
        "account": memory_account(
            monkeypatch,
            [(daily_view(panel), panel, pd.DataFrame())],
            etf_schedule(),
            params,
        ),
        "risk_aversion": 0,
        "unfinished_penalty_bps": 0,
    }
    if task == "execution":
        episode["order"] = {"asset_id": "a", "quantity": -100}

    decisions, batches, _, _, _ = replay(episode, [[1.0], [1.0]])
    assert decisions[0].nav == 15_000
    assert decisions[0].quantities.a == 500
    assert decisions[0].maximum_quantities.a == 375
    assert [trade.quantity for batch in batches for trade in batch.trades] == (
        [] if task == "inventory" else [-100]
    )


def test_inventory_action_waits_for_a_known_money_cap(monkeypatch):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:03.020", "01:30:06", "01:30:06.020"),
    ).assign(instrument_type="gold_etf", ask_vol1=5000.0)
    panel.loc[panel.timestamp.eq(pd.Timestamp("2024-01-02T01:30:03")), "mid"] = np.nan
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T01:30:03.001", "2024-01-02T01:30:06.001"]
            ),
            "asset_id": "a",
            "prediction_score": 0.1,
        }
    )
    params = parameters(("a",), restore_at="14:50:00")
    del params["buy_threshold"], params["sell_threshold"]
    params.update(
        initial_capital=10_000, initial_inventory={"a": 0}, maximum_weights={"a": 0.1}
    )
    episode = {
        "kind": "hft",
        "task": "inventory",
        "asset_ids": ["a"],
        "feature_columns": ["prediction_score"],
        "prediction": {"paths": ["memory"]},
        "window": {"start": "2024-01-02T01:30:00", "end": "2024-01-02T01:30:07"},
        "account": memory_account(
            monkeypatch, [(daily_view(panel), panel, signals)], etf_schedule(), params
        ),
        "risk_aversion": 0,
        "unfinished_penalty_bps": 0,
    }

    decisions, batches, _, _, _ = replay(episode, [[1.0], [1.0]])
    assert pd.isna(decisions[0].maximum_quantities.a)
    assert decisions[1].maximum_quantities.a == 100
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [100]
    assert trades[0].timestamp_ns == pd.Timestamp("2024-01-02T01:30:06.020").value


@pytest.mark.parametrize("weights", [{"a": -0.1}, {"a": 1.1}, {"b": 0.5}])
def test_initial_weight_input_rejects_invalid_allocation(weights):
    params = parameters(("a",))
    del params["initial_inventory"]
    with pytest.raises(ValueError, match="hft_initial_weights_invalid"):
        high_frequency._parameters({**params, "initial_weights": weights})


def test_maker_to_taker_waits_for_cancel_and_settles_partial_fills_once(tmp_path):
    episode = execution_episode(
        tmp_path,
        records=[
            (1, "a", "sell", 60, 9.99),
            (3.5, "a", "sell", 25, 9.99),
            (4.001, "a", "sell", 25, 9.99),
        ],
    )
    episode["account"]["parameters"]["maximum_weights"] = {"a": 0.0031}
    decisions, batches, rewards, info, initial = replay(episode, [[4], [7], [0], [0]])
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [50, 25, 25, 100]
    assert [trade.order_id.startswith("passive:") for trade in trades] == [
        True,
        True,
        True,
        False,
    ]
    assert trades[-1].timestamp_ns == pd.Timestamp("2024-01-02T01:30:04.001").value
    assert decisions[1].quantities.a == 150
    assert decisions[1].pending_quantities.a == 150
    assert decisions[1].reserved_cash == pytest.approx(150 * 9.99)
    assert decisions[2].quantities.a == 300
    assert decisions[2].reserved_cash == 0
    # Maker earns 1 CNY spread, taker pays 1 CNY; two orders each cost 5 CNY.
    assert info["fees"] == 10
    assert info["nav"] == pytest.approx(initial - 10)
    assert sum(rewards) == pytest.approx(-0.1)
    assert info["economic_reward_bps"] == pytest.approx(sum(rewards))


def test_wait_keeps_order_cancel_allows_tied_print_until_ack(tmp_path):
    episode = execution_episode(
        tmp_path,
        records=[
            (4, "a", "sell", 30, 9.99),
            (6.5, "a", "sell", 20, 9.99),
            (7.001, "a", "sell", 20, 9.99),
            (7.5, "a", "sell", 20, 9.99),
        ],
    )
    decisions, batches, _, _, _ = replay(episode, [[4], [0], [1], [0]])
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [20, 20, 20]
    assert decisions[2].reserved_cash > 0
    assert decisions[3].reserved_cash == 0
    requested = [
        batch
        for batch in batches
        if any(row.name == "passive_cancel_requested" for row in batch.diagnostics)
    ]
    assert len(requested) == 1
    assert (
        requested[0].valuations[-1].timestamp_ns
        == pd.Timestamp("2024-01-02T01:30:06.001").value
    )


def test_quantity_action_preserves_full_strategy_target_in_native_ledger(tmp_path):
    episode = execution_episode(tmp_path, records=[(1, "a", "sell", 110, 9.99)])
    _, batches, _, _, _ = replay(episode, [[3], [0], [0], [0]])
    first = next(
        batch
        for batch in batches
        if any(row.name == "execution_mode" for row in batch.diagnostics)
    )
    facts = {row.name: row.number for row in first.diagnostics if row.asset_id == "a"}
    assert facts["inventory_target"] == 300
    assert facts["execution_destination"] == 200
    assert [trade.quantity for batch in batches for trade in batch.trades] == [100]


@pytest.mark.parametrize("kind,remaining", [("equity_etf", 200), ("gold_etf", 0)])
def test_dynamic_execution_obeys_native_t1_and_t0(kind, remaining, tmp_path):
    episode = execution_episode(
        tmp_path, kinds=(kind,), scores=(0.1, -0.1, np.nan, np.nan)
    )
    # The observed bid must have enough size for the requested whole-lot sale.
    panel = read_dataset(episode["account"]["panel"])
    panel["bid_vol1"] = 500.0
    write_dataset(tmp_path / "liquid", panel, spec=quote_spec(panel))
    episode["account"]["panel"] = str(tmp_path / "liquid")
    decisions, batches, _, _, _ = replay(episode, [[7], [7], [7], [7]])
    assert decisions[2].quantities.a == remaining
    assert decisions[2].sellable.a == (0 if kind == "equity_etf" else remaining)
    assert [trade.quantity for batch in batches for trade in batch.trades] == (
        [200, -100] if kind == "equity_etf" else [200, -300]
    )


@pytest.mark.parametrize(
    "action,expected", [(2, 100), (3, 200), (4, 200), (5, 100), (6, 200), (7, 200)]
)
def test_ppo_quantity_actions_share_opposite_capacity(tmp_path, action, expected):
    episode = execution_episode(tmp_path)
    episode["account"]["parameters"].update(
        maximum_weights={"a": 0.005}, participation_rate=0.4
    )
    decisions, batches, _, _, _ = replay(episode, [[action], [0], [0], [0]])
    # Gap 400; fractions request 100/200/400, capped at ask depth 500 * .4.
    if action < 5:
        assert decisions[1].pending_quantities.a == expected
        assert decisions[1].reserved_cash == pytest.approx(expected * 9.99 + 5)
    else:
        assert [trade.quantity for batch in batches for trade in batch.trades] == [
            expected
        ]


def test_shared_cash_inflight_reservations_and_taker_capacity(tmp_path):
    episode = execution_episode(
        tmp_path,
        kinds=("gold_etf", "gold_etf"),
        capital=3500,
        records=[(1, "a", "sell", 500, 9.99)],
    )
    decisions, batches, _, info, _ = replay(episode, [[4, 7], [0, 0], [0, 0], [0, 0]])
    assert [
        (trade.asset_id, trade.quantity) for batch in batches for trade in batch.trades
    ] == [("a", 100)]
    assert decisions[1].quantities.to_dict() == {"a": 200, "b": 100}
    assert info["nav"] == pytest.approx(3496)
    assert all(
        row.value >= 0
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "__cash__"
    )
    # The cost is charged by the account, not once again by the reward.
    assert info["economic_reward_bps"] == pytest.approx(-4 / 3500 * 10000)


def test_multiple_arriving_iocs_share_cash_before_their_replacements(tmp_path):
    episode = execution_episode(
        tmp_path, kinds=("gold_etf", "gold_etf"), capital=6000, cancel_ms=1500
    )
    episode["account"]["parameters"].update(
        initial_inventory={"a": 0, "b": 0}, latency_ms=4000
    )
    _, batches, _, info, _ = replay(episode, [[6, 6], [4, 4], [0, 0], [0, 0]])
    assert [
        (trade.asset_id, trade.quantity) for batch in batches for trade in batch.trades
    ] == [("a", 100), ("b", 100)]
    cash = next(
        row.value
        for row in batches[-1].valuations
        if row.phase == "end" and row.asset_id == "__cash__"
    )
    # Two IOC orders spend 100 * 10.01 + 5 each; successor orders have no fills.
    assert cash == pytest.approx(3988)
    assert info["nav"] == pytest.approx(5988)
    assert info["fees"] == 10


def test_observation_and_action_follow_caller_asset_order(tmp_path):
    episode = execution_episode(tmp_path, kinds=("gold_etf", "gold_etf"))
    panel = read_dataset(episode["account"]["panel"])
    panel.loc[panel.asset_id.eq("b"), ["mid", "bid_px1", "ask_px1"]] = [
        20,
        19.99,
        20.01,
    ]
    write_dataset(tmp_path / "different_prices", panel, spec=quote_spec(panel))
    episode["account"]["panel"] = str(tmp_path / "different_prices")
    episode["account"]["daily_inputs"] = write_daily_inputs(
        tmp_path / "different_daily", panel
    )
    episode["asset_ids"] = ["b", "a"]
    episode["account"]["parameters"]["maximum_weights"] = {"a": 0.0015, "b": 0.003}
    env = TradingEnv([episode])
    try:
        observation, _ = env.reset(seed=0)
        assert observation[[1, 5]].tolist() == [20, 10]
        # Four features and two balance fields per asset, then three global fields.
        assert observation[15:17].tolist() == pytest.approx(
            [300 * 20 / 2_000_000, 300 * 10 / 2_000_000]
        )
        env.step([7, 0])
        assert env.decision.quantities.to_dict() == {"a": 100, "b": 300}
    finally:
        env.close()


def test_observation_uses_visible_quote_and_rejects_future_fitted_prediction(tmp_path):
    episode = execution_episode(tmp_path)
    panel = read_dataset(episode["account"]["panel"])
    panel.loc[
        panel.timestamp.dt.microsecond.eq(500),
        ["mid", "bid_px1", "ask_px1", "bid_vol1"],
    ] = [20, 19.99, 20.01, 999]
    write_dataset(tmp_path / "delayed", panel, spec=quote_spec(panel))
    episode["account"]["panel"] = str(tmp_path / "delayed")
    with pytest.raises(ValueError, match="policy_action_invalid"):
        with_env = TradingEnv([episode])
        try:
            obs, _ = with_env.reset(seed=0)
            assert obs[:4].tolist() == pytest.approx([0.1, 10, 10, 500])
            assert with_env.decision.marks.a == 10
            with_env.step([8])
        finally:
            with_env.close()
    metadata = forecast_metadata()
    metadata["model_facts"]["fit_cutoff"] = "2024-01-02T01:30:03"
    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02T01:30:00.001")],
            "asset_id": ["a"],
            "prediction_score": [0.1],
            "prediction_rank_score": [0.0],
        }
    )
    path = write_prediction_outputs(
        output_path=tmp_path / "future_fit", output_chunks=[rows], metadata=metadata
    )
    episode["prediction"]["paths"] = [str(path)]
    with pytest.raises(ValueError, match="policy_prediction_not_out_of_sample"):
        backtest_policy(
            {
                "episode": episode,
                "strategy": "immediate",
                "output": str(tmp_path / "invalid"),
            }
        )
    assert not (tmp_path / "invalid").exists()


def test_future_quote_absence_cannot_change_current_pending_state(tmp_path):
    episode = execution_episode(tmp_path)
    episode["account"]["parameters"]["latency_ms"] = 4000
    truncated = deepcopy(episode)
    panel = read_dataset(episode["account"]["panel"])
    panel = panel.loc[panel.timestamp.dt.time <= pd.Timestamp("01:30:03").time()]
    write_dataset(tmp_path / "unknown_future", panel, spec=quote_spec(panel))
    truncated["account"]["panel"] = str(tmp_path / "unknown_future")
    states = []
    for config in (episode, truncated):
        env = TradingEnv([config])
        try:
            observation, _ = env.reset(seed=0)
            next_observation, _, done, _, _ = env.step([7])
            assert not done
            assert env.decision.timestamp == pd.Timestamp("2024-01-02T01:30:03.001")
            states.append(
                (
                    observation,
                    next_observation,
                    env.decision.cash,
                    env.decision.reserved_cash,
                )
            )
        finally:
            env.close()
    np.testing.assert_allclose(states[0][0], states[1][0])
    np.testing.assert_allclose(states[0][1], states[1][1])
    assert states[0][2:] == pytest.approx(states[1][2:])


def test_risk_time_excludes_lunch_overnight_and_weekend():
    assert (
        _trading_seconds(
            pd.Timestamp("2024-01-05T03:00"), pd.Timestamp("2024-01-05T05:30")
        )
        == 3600
    )

    assert (
        _trading_seconds(
            pd.Timestamp("2024-01-05T07:00"), pd.Timestamp("2024-01-08T01:30")
        )
        == 0
    )


@pytest.mark.parametrize(
    "end,trading_seconds",
    [
        ("2024-01-05T07:00:00", 14399.999),
        ("2024-01-05", 14399.999),
        ("2024-01-07T06:00:00", 14399.999),
        ("2024-01-05T01:30:13", 12.999),
    ],
)
def test_terminal_exposure_uses_only_observed_native_sessions(
    monkeypatch, end, trading_seconds
):
    panel = quotes(
        "2024-01-05",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:00.020",
            "01:30:03",
            "01:30:04.001",
            "01:30:06",
            "01:30:09",
            "01:30:12",
        ),
    ).assign(instrument_type="gold_etf")
    predictions = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-05T01:30:00.001")
            + pd.to_timedelta([0, 3, 6, 9], unit="s"),
            "asset_id": "a",
            "prediction_score": 0.1,
        }
    )
    params = parameters(("a",), restore_at="14:50:00")
    del params["buy_threshold"], params["sell_threshold"]
    params.update(
        initial_capital=1_000_000,
        passive={"queue_model": "conservative", "cancel_latency_ms": 10},
    )
    episode = {
        "kind": "hft",
        "task": "dynamic_execution",
        "asset_ids": ["a"],
        "feature_columns": ["prediction_score"],
        "window": {
            "start": "2024-01-05" if len(end) == 10 else "2024-01-05T01:30:00",
            "end": end,
        },
        "prediction": {"paths": ["memory"]},
        "account": memory_account(
            monkeypatch,
            [(daily_view(panel), panel, predictions)],
            etf_schedule(),
            params,
        ),
        "inventory_parameters": {"trade_cost_bps": 0, "exit_cost_bps": 0},
        "risk_aversion": 1,
        "target_deviation_bps": 0,
        "unfinished_penalty_bps": 0,
    }

    _, batches, rewards, info, _ = replay(episode, [[0], [0], [0], [0]])
    assert info["economic_reward_bps"] == 0
    assert not any(batch.trades or batch.cashflows for batch in batches)
    assert sum(rewards) == pytest.approx(
        -((100 * 10 / 1_000_000) ** 2) * trading_seconds / 14400, rel=1e-12, abs=1e-15
    )


@pytest.mark.parametrize("cancel_ms,filled", [(500, 0), (1500, 200)])
def test_inflight_cancel_reserves_until_ack_and_cannot_undo_arriving_ioc(
    tmp_path, cancel_ms, filled
):
    episode = execution_episode(tmp_path, cancel_ms=cancel_ms)
    episode["account"]["parameters"]["latency_ms"] = 4000
    decisions, batches, _, _, _ = replay(episode, [[7], [1], [0], [0]])
    assert decisions[1].order_state.loc["a", "quantity"] == 0
    assert decisions[1].pending_quantities.a == 200
    assert decisions[1].reserved_cash == pytest.approx(200 * 10.01 + 5)
    assert sum(trade.quantity for batch in batches for trade in batch.trades) == filled
    assert decisions[2].reserved_cash == 0


def test_replace_inflight_waits_for_old_ack_and_uses_post_fill_inventory(tmp_path):
    episode = execution_episode(
        tmp_path,
        cancel_ms=1500,
        scores=(0.1, -0.1, np.nan, np.nan),
        records=[(8, "a", "buy", 810, 10.01)],
        bid_depth=500.0,
    )
    episode["account"]["parameters"]["latency_ms"] = 4000
    decisions, batches, _, _, _ = replay(episode, [[7], [4], [0], [0]])
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [200, -300]
    assert trades[0].timestamp_ns == pd.Timestamp("2024-01-02T01:30:04.001").value
    assert trades[1].timestamp_ns == pd.Timestamp("2024-01-02T01:30:08").value
    assert decisions[2].pending_quantities.a == -300
    assert decisions[2].sellable.a == 0
    assert decisions[3].quantities.a == 0


def test_target_penalty_is_integrated_without_a_second_fee_or_cashflow(tmp_path):
    episode = execution_episode(tmp_path)
    episode["target_deviation_bps"] = 10
    _, batches, rewards, info, _ = replay(episode, [[0], [0], [0], [0]])
    expected = -10 * ((100 - 300) * 10 / 1_000_000) ** 2 * 11.999 / 14400
    assert sum(rewards) == pytest.approx(expected, rel=1e-9, abs=1e-15)
    assert info["economic_reward_bps"] == 0
    assert not any(batch.trades or batch.cashflows for batch in batches)


@pytest.mark.parametrize(
    "kind,side,partial_fill",
    [("gold_etf", side, fill) for side in (1, -1) for fill in (0, 100)]
    + [("equity_etf", 1, 100)],
)
def test_execution_reservations_cannot_change_strategy_target_or_earn_reward(
    monkeypatch, kind, side, partial_fill
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:00.020",
            "01:30:03",
            "01:30:04.001",
            "01:30:06",
            "01:30:09",
            "01:30:12",
        ),
    ).assign(instrument_type=kind)
    predictions = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02T01:30:00.001")
            + pd.to_timedelta([0, 3, 6, 9], unit="s"),
            "asset_id": "a",
            "prediction_score": side * 0.1,
        }
    )
    events = pd.DataFrame(
        [
            (
                pd.Timestamp("2024-01-02T01:30:04.5"),
                "a",
                0,
                "trade",
                "sell" if side == 1 else "buy",
                10 - side * 0.01,
                500 + partial_fill,
            )
        ]
        if partial_fill
        else [],
        columns=(
            "timestamp",
            "asset_id",
            "source_row",
            "event_kind",
            "side",
            "price",
            "volume",
        ),
    ).astype({"timestamp": "datetime64[ns]", "price": "float64", "volume": "float64"})
    params = parameters(("a",), restore_at="14:50:00")
    del params["buy_threshold"], params["sell_threshold"]
    params.update(
        initial_capital=20_000,
        initial_inventory={"a": 1000},
        maximum_weights={"a": 1.0},
        minimum_commission=0,
        passive={"queue_model": "conservative", "cancel_latency_ms": 10_000},
    )
    episode = {
        "kind": "hft",
        "task": "dynamic_execution",
        "asset_ids": ["a"],
        "feature_columns": ["prediction_score"],
        "window": {"start": "2024-01-02T01:30:00", "end": "2024-01-02T01:30:12"},
        "prediction": {"paths": ["memory"]},
        "account": memory_account(
            monkeypatch,
            [(daily_view(panel), panel, predictions)],
            etf_schedule(),
            params,
            trades=events,
        ),
        "inventory_parameters": {"trade_cost_bps": 0, "exit_cost_bps": 0},
        "risk_aversion": 0,
        "target_deviation_bps": 10,
        "unfinished_penalty_bps": 0,
    }

    batches, decisions, targets, rewards = [], [], [], []
    env = TradingEnv([episode], on_batch=batches.append)
    try:
        env.reset(seed=0)
        for action in (2, 4, 0, 0):
            decisions.append(env.decision)
            targets.append(env.target.a)
            _, reward, done, _, info = env.step([action])
            rewards.append(reward)
        assert done
    finally:
        env.close()
    # The live order is 200 shares; the larger replacement reserves at most 500.
    maximum_after_fill = (20_000 + partial_fill * 0.01) / 10
    assert targets == (
        [2000.0, 2000.0, maximum_after_fill, maximum_after_fill]
        if side == 1
        else [0.0] * 4
    )
    assert decisions[1].pending_quantities.a == side * 200
    assert decisions[2].pending_quantities.a == side * (200 - partial_fill)
    assert decisions[2].quantities.a == 1000 + side * partial_fill
    gross_cash = 10_000 - side * partial_fill * (10 - side * 0.01)
    assert decisions[2].cash + decisions[2].reserved_cash == pytest.approx(gross_cash)
    physical = 1000 + side * partial_fill if side == -1 or kind == "gold_etf" else 1000
    assert decisions[2].physical_sellable.a == physical
    if side == 1:
        assert decisions[1].reserved_cash == pytest.approx(1998)
        assert decisions[2].reserved_cash == pytest.approx(500 * 9.99)
        assert decisions[2].sellable.a == physical
    else:
        assert decisions[1].sellable.a == 800
        assert decisions[2].sellable.a == physical - 500
        assert decisions[2].reserved_cash == 0
    # The later three-second interval rewards only actual inventory reduction.
    distance = (maximum_after_fill if side == 1 else 0) - (1000 + side * partial_fill)
    expected_penalty = -10 * (distance * 10 / 20_000) ** 2 * 3 / 14400
    assert rewards[2] == pytest.approx(expected_penalty, rel=1e-10, abs=1e-15)
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == (
        [side * partial_fill] if partial_fill else []
    )
    assert info["economic_reward_bps"] == pytest.approx(
        partial_fill * 0.01 / 20_000 * 10000
    )
    opposite = decisions[2].features.copy()
    opposite["prediction_score"] = -side * 0.1
    opposite_target = inventory_targets(
        replace(decisions[2], features=opposite),
        decisions[2].maximum_quantities,
        env.episode["inventory_parameters"],
    )
    assert opposite_target.a == (
        maximum_after_fill
        if side == -1
        else partial_fill
        if kind == "equity_etf"
        else 0
    )


@pytest.mark.parametrize(
    "n_envs,timesteps,eval_freq,evaluations,updates,stopped_early",
    [
        (1, 64, 4, [4, 8, 12], [0, 1, 2], True),
        (2, 64, 4, [4, 8, 12], [0, 0, 1], True),
        (1, 4, 4, [4, 4], [0, 1], False),
        (1, 4, 8, [4], [1], False),
    ],
)
def test_real_ppo_historical_early_stop_best_pair_and_frozen_backtest(
    tmp_path,
    monkeypatch,
    n_envs,
    timesteps,
    eval_freq,
    evaluations,
    updates,
    stopped_early,
):
    from stable_baselines3.common import callbacks

    evaluate = callbacks.evaluate_policy
    evaluated_updates = []

    def observed_evaluation(model, *args, **kwargs):
        assert model.get_env().norm_reward
        assert not args[0].training
        assert not args[0].norm_reward
        evaluated_updates.append(model._n_updates)
        return evaluate(model, *args, **kwargs)

    monkeypatch.setattr(callbacks, "evaluate_policy", observed_evaluation)
    episode = execution_episode(tmp_path / "inputs", scores=(0, 0, 0, 0))
    validation = deepcopy(episode)
    validation["window"] = {
        "start": "2024-01-03T01:30:00",
        "end": "2024-01-03T01:30:12",
    }
    config = {
        "episodes": [episode],
        "validation_episodes": [validation],
        "early_stopping": {"eval_freq": eval_freq, "patience": 1, "min_evals": 1},
        "fit_cutoff": "2024-01-03T01:30:12",
        "output": str(tmp_path / "policy"),
        "timesteps": timesteps,
        "seed": 1,
        "device": "cpu",
        "n_envs": n_envs,
        "threads": 1,
        "ppo": {
            "n_steps": 4,
            "gamma": 0.999,
            "batch_size": 4,
            "n_epochs": 1,
            "policy_kwargs": {"net_arch": [8]},
            "target_kl": 0.02,
        },
    }
    model = train_policy(config)
    facts = parse_qrp_json_object((model / "policy.json").read_bytes())
    assert facts["actual_timesteps"] == evaluations[-1]
    assert facts["validation"]["stopped_early"] == stopped_early
    assert [
        item["timesteps"] for item in facts["validation"]["evaluations"]
    ] == evaluations
    assert evaluated_updates == updates
    assert [item["updates"] for item in facts["validation"]["evaluations"]] == updates
    for item in facts["validation"]["evaluations"]:
        assert item["episode_lengths"] == [4]
        assert item["episode_metrics"][0]["episode_index"] == 0
        assert item["episode_metrics"][0]["economic_reward"] == 0
        assert item["episode_metrics"][0]["reward_penalty"] == 0
    progress = pd.read_csv(model.with_name(model.name + ".trial") / "progress.csv")
    assert progress["train/n_updates"].max() == facts["actual_updates"]
    assert facts["validation"]["selected_timesteps"] == 4
    assert facts["validation"]["best_validation_reward"] == 0
    assert facts["training_decisions"]["start"] == "2024-01-02T01:30:00.001000"
    feature_scaler = (model / "feature_scaler.pkl").read_bytes()
    normalization = pickle.loads((model / "normalization.pkl").read_bytes())
    assert normalization.norm_reward
    assert normalization.gamma == 0.999
    assert pickle.loads(feature_scaler).medians == (0, 10, 10, 500)
    later = deepcopy(episode)
    later["window"] = {"start": "2024-01-04T01:30:00", "end": "2024-01-04T01:30:12"}
    output = backtest_policy(
        {
            "episode": later,
            "strategy": "ppo",
            "policy": str(model),
            "output": str(tmp_path / "replay"),
        }
    )
    ledger = read_portfolio_ledger_artifact(output / "ledger")
    assert list(ledger.iter_chunks())
    assert ledger.metadata["dynamic_execution"]["reward_unit"] == "bps_of_initial_nav"
    final = backtest_policy(
        {
            "episode": later,
            "strategy": "ppo",
            "policy": str(model),
            "checkpoint": "final",
            "output": str(tmp_path / "final-replay"),
        }
    )
    assert list(read_portfolio_ledger_artifact(final / "ledger").iter_chunks())
    with pytest.raises(ValueError, match="policy_validation_not_after_training"):
        train_policy(
            {
                **config,
                "validation_episodes": [episode],
                "output": str(tmp_path / "overlap"),
            }
        )
    with pytest.raises(ValueError, match="policy_validation_not_after_training"):
        train_policy(
            {
                **config,
                "validation_episodes": [
                    {
                        **validation,
                        "window": {
                            "start": episode["window"]["end"],
                            "end": validation["window"]["end"],
                        },
                    }
                ],
                "output": str(tmp_path / "shared-endpoint"),
            }
        )


def test_hold_baseline_keeps_the_initial_shares_on_the_same_signal_clock(tmp_path):
    episode = execution_episode(tmp_path / "inputs", capital=10_000)
    episode["account"]["parameters"]["restore_at"] = "14:45:00"
    fills = {}
    for strategy in ("hold", "immediate"):
        root = backtest_policy(
            dict(episode=episode, strategy=strategy, output=str(tmp_path / strategy))
        )
        ledger = read_portfolio_ledger_artifact(root / "ledger")
        fills[strategy] = sum(len(batch.trades) for batch in ledger.iter_chunks())
    assert fills["hold"] == 0
    assert fills["immediate"] > 0
