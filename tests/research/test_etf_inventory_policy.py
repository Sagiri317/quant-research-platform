"""Dynamic ETF quantities use causal forecasts and the existing native account."""

from dataclasses import asdict, replace

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
from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.policies.inventory_policy import (
    inventory_policy_parameters,
    inventory_targets,
    prediction_parameters,
    validate_inventory_prediction,
)
from quant_research_platform.research.policies.reinforcement import backtest_policy
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.research.protocols.a_share_hft import label_catalog
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.high_frequency import (
    daily_view,
    parameters,
    quotes,
    schedule,
    trade_input,
    write_daily_inputs,
)
from tests.helpers.output_frames import make_price_volume_prediction_output_frame
from tests.helpers.portfolio_inputs import _interim_specs

START = pd.Timestamp("2024-01-02T01:30:00")


def etf_schedule():
    return replace(
        schedule(),
        rules=tuple(replace(rule, market="etf") for rule in schedule().rules),
    )


def forecast_metadata():
    metadata = make_price_volume_prediction_output_frame().attrs["prediction_metadata"]
    facts = metadata["model_facts"]
    facts["model"]["target_column"] = "future_mid_return_30s"
    facts["target_definition"] = {
        "source_label": "future_mid_return_30s",
        "target_transform": "raw",
        "target_clip": "none",
        "label_semantics": {
            **label_catalog().get("future_mid_return_30s").to_manifest(),
            "horizon_origin": "common_decision_time",
            "decision_interval_seconds": 3,
            "quote_visibility_delay_ms": 1,
        },
    }
    metadata.update(prediction_window={"start": "2024-01-02", "end": "2024-01-02"})
    return metadata


def run_stream(panel, signals, params, *, events=None, policy=None):
    stream = inventory_batches(
        [(daily_view(panel), panel, signals)],
        etf_schedule(),
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
        decision_columns=("prediction_score",),
        prediction_decisions=True,
        quote_visibility_delay_ms=1,
        trade_session=events,
        max_quote_gap_seconds=6,
    )
    decisions, batches = [], []
    try:
        event = next(stream)
        while True:
            if isinstance(event, PolicyDecision):
                decisions.append(event)
                target = (
                    inventory_targets(
                        event,
                        event.maximum_quantities,
                        inventory_policy_parameters({}),
                    )
                    if policy is None
                    else policy(event)
                )
                event = stream.send(target)
            else:
                batches.append(event)
                event = next(stream)
    except StopIteration:
        pass
    finally:
        stream.close()
    return decisions, batches


@pytest.mark.parametrize("price", [5.0, 10.0])
def test_inventory_cap_tracks_current_nav_instead_of_fixed_shares(price):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        mid=price,
        clocks=("01:30:00", "01:30:00.020", "01:30:03", "01:30:03.020"),
    ).assign(instrument_type="gold_etf", ask_vol1=10_000.0)
    signals = pd.DataFrame(
        {
            "timestamp": [
                START + pd.Timedelta(milliseconds=1),
                START + pd.Timedelta(seconds=3, milliseconds=1),
            ],
            "asset_id": "a",
            "prediction_score": 0.1,
        }
    )
    params = parameters(("a",))
    del params["initial_inventory"]
    params.update(
        initial_capital=10_000, initial_weights={"a": 0.5}, maximum_weights={"a": 1.0}
    )
    decisions, batches = run_stream(panel, signals, params)
    filled = 900 if price == 5 else 400
    assert decisions[0].quantities.a == 5000 / price
    assert decisions[0].maximum_quantities.a == 10_000 / price
    assert [t.quantity for b in batches for t in b.trades] == [filled]
    assert decisions[1].nav == pytest.approx(10_000 - filled * 0.01)
    assert decisions[1].maximum_quantities.a == pytest.approx(
        (10_000 - filled * 0.01) / price
    )
    cash = next(
        v.value
        for v in batches[-1].valuations
        if v.phase == "end" and v.asset_id == "__cash__"
    )
    assert cash == pytest.approx(5000 - filled * (price + 0.01))


@pytest.mark.parametrize("change", [0, -100, 1])
def test_price_gain_above_weight_cap_allows_holding_or_reducing_only(change):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:00.020", "01:30:03", "01:30:03.020"),
    ).assign(instrument_type="gold_etf")
    panel.loc[
        panel.timestamp.ge(START + pd.Timedelta(seconds=3)),
        ["mid", "bid_px1", "ask_px1"],
    ] = [20, 19.99, 20.01]
    signals = pd.DataFrame(
        {
            "timestamp": [
                START + pd.Timedelta(milliseconds=1),
                START + pd.Timedelta(seconds=3, milliseconds=1),
            ],
            "asset_id": "a",
            "prediction_score": 0.1,
        }
    )
    params = parameters(("a",))
    params.update(
        initial_capital=10_000, initial_inventory={"a": 500}, maximum_weights={"a": 0.5}
    )

    def target(decision):
        return decision.quantities + (change if decision.marks.a == 20 else 0)

    if change > 0:
        with pytest.raises(ValueError, match="policy_target_inventory_exceeded"):
            run_stream(panel, signals, params, policy=target)
        return
    decisions, batches = run_stream(panel, signals, params, policy=target)
    assert decisions[1].nav == 15_000
    assert decisions[1].maximum_quantities.a == 375
    assert decisions[1].quantities.a == 500
    assert [t.quantity for b in batches for t in b.trades] == (
        [] if change == 0 else [-100]
    )


@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("sold", [0, 400])
def test_restore_preserves_fixed_reference_but_caps_new_purchases(
    tmp_path, passive, sold
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:00.020",
            "01:30:00.040",
            "01:30:03",
            "06:49:59",
            "06:50:00.020",
            "06:50:00.040",
        ),
    ).assign(instrument_type="gold_etf")
    panel.loc[
        panel.timestamp.ge(START + pd.Timedelta(seconds=3)),
        ["mid", "bid_px1", "ask_px1"],
    ] = [20, 19.99, 20.01]
    signals = pd.DataFrame(
        {
            "timestamp": [START + pd.Timedelta(milliseconds=1)],
            "asset_id": "a",
            "prediction_score": -0.1,
        }
    )
    params = parameters(("a",), restore_at="14:50:00")
    params.update(
        initial_capital=10_000,
        initial_inventory={"a": 500},
        maximum_weights={"a": 0.5},
        latency_ms=10,
    )
    events = None
    if passive:
        params["passive"] = {"queue_model": "conservative", "cancel_latency_ms": 0}
        events = trade_input(tmp_path / "trades", [(0.030, "a", "buy", 900, 10.01)])
    _, batches = run_stream(
        panel,
        signals,
        params,
        events=events,
        policy=lambda decision: decision.quantities - sold,
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == ([-400, 100] if sold else [])
    last = {v.asset_id: v for v in batches[-1].valuations if v.phase == "end"}
    assert last["a"].quantity == (200 if sold else 500)
    sale_price = 10.01 if passive else 9.99
    assert last["__cash__"].value == pytest.approx(
        5000 + 400 * sale_price - 100 * 20.01 if sold else 5000
    )
    # After the gain, unchanged 500 shares exceed 50% * 15000 / 20 = 375;
    # after selling 400, the cap is 274.9/275.1 and only one buy lot fits.
    assert [
        d.text
        for b in batches
        for d in b.diagnostics
        if d.name == "inventory_restoration_status"
    ] == ["INCOMPLETE" if sold else "COMPLETE"]
    assert {
        d.number
        for b in batches
        if b.valuations[0].timestamp_ns >= pd.Timestamp("2024-01-02T06:50:00").value
        for d in b.diagnostics
        if d.name == "inventory_target"
    } == {500.0}


def test_unknown_decision_price_cannot_bypass_money_cap_with_explicit_target():
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:03.020"),
    ).assign(instrument_type="gold_etf", ask_vol1=5000.0)
    panel.loc[panel.timestamp.eq(START + pd.Timedelta(seconds=3)), "mid"] = np.nan
    signals = pd.DataFrame(
        {
            "timestamp": [START + pd.Timedelta(seconds=3, milliseconds=1)],
            "asset_id": "a",
            "prediction_score": 0.1,
        }
    )
    params = parameters(("a",))
    params.update(
        initial_capital=10_000, initial_inventory={"a": 0}, maximum_weights={"a": 0.1}
    )

    def target(decision):
        assert pd.isna(decision.marks.a)
        assert pd.isna(decision.maximum_quantities.a)
        return pd.Series({"a": 500.0})

    with pytest.raises(ValueError, match="policy_target_inventory_exceeded"):
        run_stream(panel, signals, params, policy=target)


def test_quadratic_band_sizes_weak_missing_and_pending():
    decision = PolicyDecision(
        START,
        pd.DataFrame({"prediction_score": [0.0007]}, index=["a"]),
        pd.Series({"a": 100.0}),
        pd.Series({"a": 100.0}),
        pd.Series({"a": 10.0}),
        10000.0,
        11000.0,
        pd.Series({"a": 100.0}),
        pd.Series({"a": 0.0}),
        pd.Series({"a": np.nan}),
        pd.Series({"a": 0.0}),
        lambda asset, delta: 0.0,
        physical_sellable=pd.Series({"a": 100.0}),
    )
    p = inventory_policy_parameters(
        {"trade_cost_bps": 1, "exit_cost_bps": 1, "inventory_risk_bps": 10}
    )
    # Marginal reward is 7 - 1 - 1 bps, so displacement is 300 * 5/10.
    assert inventory_targets(decision, {"a": 300}, p)["a"] == pytest.approx(250)
    no_risk = inventory_policy_parameters({**p, "inventory_risk_bps": 0})
    assert inventory_targets(decision, {"a": 300}, no_risk)["a"] == 300
    weak = replace(
        decision, features=pd.DataFrame({"prediction_score": [0.0001]}, index=["a"])
    )
    assert inventory_targets(weak, {"a": 300}, p)["a"] == 100
    away = replace(
        weak,
        quantities=pd.Series({"a": 200.0}),
        sellable=pd.Series({"a": 200.0}),
        physical_sellable=pd.Series({"a": 200.0}),
    )
    assert inventory_targets(away, {"a": 300}, p)["a"] < 200
    pending = replace(
        weak,
        pending_quantities=pd.Series({"a": 100.0}),
        pending_targets=pd.Series({"a": 200.0}),
    )
    expensive = {**p, "trade_cost_bps": 10}
    # Cancelling a pending buy back to actual inventory is not a sale.
    assert inventory_targets(pending, {"a": 300}, expensive)["a"] == 100
    missing = replace(
        away, features=pd.DataFrame({"prediction_score": [np.nan]}, index=["a"])
    )
    assert np.isnan(inventory_targets(missing, {"a": 300}, p)["a"])


def test_off_quote_decision_uses_visible_book_and_native_candidate_fees():
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:00.0005", "01:30:00.020", "01:30:03.020"),
    ).assign(instrument_type="gold_etf")
    panel.loc[1:, ["mid", "bid_px1", "ask_px1"]] = [20, 19.99, 20.01]
    signals = pd.DataFrame(
        {
            "timestamp": [START + pd.Timedelta(milliseconds=1)],
            "asset_id": ["a"],
            "prediction_score": [0.1],
        }
    )
    params = parameters(("a",))
    params.update(latency_ms=10.0, minimum_commission=5.0)
    decisions, batches = run_stream(panel, signals, params)
    assert len(decisions) == 1 and decisions[0].timestamp == signals.timestamp[0]
    assert (
        decisions[0].marks["a"] == 10
    )  # The 0.5 ms quote is not yet visible at T=1 ms.
    assert decisions[0].estimate_trade_cost("a", 100) == pytest.approx(6)
    assert decisions[0].estimate_trade_cost("a", 10000) == pytest.approx(105)
    assert decisions[0].estimate_trade_cost("a", 0) == 0
    trades = [trade for batch in batches for trade in batch.trades]
    assert trades[0].timestamp_ns == panel.timestamp.iloc[2].value
    assert int(trades[0].order_id) == signals.timestamp[0].value


@pytest.mark.parametrize(
    "kind,expected,status",
    [("equity_etf", 200, "INCOMPLETE"), ("gold_etf", 100, "COMPLETE")],
)
def test_missing_forecast_does_not_exit_at_horizon_and_restore_obeys_t1(
    kind, expected, status
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=(
            "01:30:00",
            "01:30:00.020",
            "01:30:03.020",
            "01:30:33",
            "06:50:00",
            "06:50:00.020",
        ),
    ).assign(instrument_type=kind)
    signals = pd.DataFrame(
        {
            "timestamp": [
                START + pd.Timedelta(milliseconds=1),
                START + pd.Timedelta(seconds=3, milliseconds=1),
            ],
            "asset_id": ["a", "a"],
            "prediction_score": [0.01, np.nan],
        }
    )
    params = parameters(("a",), restore_at="14:50")
    params["latency_ms"] = 10.0
    decisions, batches = run_stream(panel, signals, params)
    assert len(decisions) == 2 and np.isnan(decisions[1].features.prediction_score["a"])
    trades = [trade for batch in batches for trade in batch.trades]
    assert trades[0].quantity == 200
    assert all(
        trade.timestamp_ns < (START + pd.Timedelta(seconds=30)).value
        or trade.timestamp_ns > pd.Timestamp("2024-01-02T06:50:00").value
        for trade in trades
    )
    end = {v.asset_id: v.quantity for v in batches[-1].valuations if v.phase == "end"}
    assert end["a"] == expected
    assert any(
        d.name == "inventory_restoration_status" and d.text == status
        for d in batches[-1].diagnostics
    )


def test_missing_signal_keeps_live_maker_and_pending_is_not_actual(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:00.020", "01:30:04", "01:30:07"),
    ).assign(instrument_type="gold_etf", bid_vol1=10.0, ask_vol1=500.0)
    signals = pd.DataFrame(
        {
            "timestamp": [
                START + pd.Timedelta(milliseconds=1),
                START + pd.Timedelta(seconds=3, milliseconds=1),
            ],
            "asset_id": ["a", "a"],
            "prediction_score": [0.01, np.nan],
        }
    )
    params = parameters(("a",))
    params.update(
        latency_ms=10.0,
        passive={"queue_model": "conservative", "cancel_latency_ms": 10.0},
    )
    events = trade_input(
        tmp_path / "trades", [(2, "a", "sell", 60, 9.99), (5, "a", "sell", 500, 9.99)]
    )
    decisions, batches = run_stream(panel, signals, params, events=events)
    assert decisions[1].quantities["a"] == 150
    assert decisions[1].pending_quantities["a"] == 150
    assert decisions[1].pending_targets["a"] == 300
    assert [t.quantity for b in batches for t in b.trades] == [50, 150]
    assert not any(
        d.name == "passive_cancel_requested" for b in batches for d in b.diagnostics
    )


@pytest.mark.parametrize("mutation", ["unit", "classifier", "horizon", "interval"])
def test_prediction_contract_rejects_nonreturn_or_wrong_clocks(mutation):
    metadata = forecast_metadata()
    facts = metadata["model_facts"]
    if mutation == "unit":
        facts["label_unit"] = "percent"
    elif mutation == "classifier":
        facts["model"]["model_algorithm"] = "deeplob_classifier"
    elif mutation == "horizon":
        facts["target_definition"]["label_semantics"]["horizon"] = 3000
    else:
        facts["target_definition"]["label_semantics"]["decision_interval_seconds"] = 30
    with pytest.raises(
        ValueError, match="inventory_prediction_return_or_clock_mismatch"
    ):
        validate_inventory_prediction(
            metadata, parameters=prediction_parameters({"paths": ["pred"]})
        )


@pytest.mark.parametrize("empty_prediction", [False, True])
def test_native_pred_to_policy_to_persisted_ledger(tmp_path, empty_prediction):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03.020", "01:30:06.020", "06:50:00.020"),
    ).assign(instrument_type="gold_etf")
    daily = write_daily_inputs(tmp_path / "daily", panel)
    roles = dict.fromkeys(panel.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    write_dataset(
        tmp_path / "panel",
        panel,
        spec=ProcessedDataSpec(
            data_id="hft_panel",
            market_id="a_share",
            kind="panel",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=(),
        ),
    )
    fees = pd.DataFrame([asdict(rule) for rule in etf_schedule().rules])
    fees["effective_date"] = pd.to_datetime(fees.effective_date)
    fees["end_date"] = pd.to_datetime(fees.end_date)
    fees["included_in"] = fees.included_in.astype("string")
    write_dataset(
        tmp_path / "fees", fees, spec=_interim_specs().get("a_share_fee_schedule")
    )
    frame = pd.DataFrame(
        {
            "timestamp": [
                START + pd.Timedelta(seconds=3),
                START + pd.Timedelta(seconds=3, milliseconds=1),
            ],
            "asset_id": ["a", "a"],
            "prediction_score": [0.01, np.nan],
            "prediction_rank_score": [0.0, np.nan],
        }
    )
    frame = frame.iloc[: 0 if empty_prediction else 1]
    metadata = forecast_metadata()
    metadata.update(
        prediction_rows=len(frame),
        effective_prediction_end=None
        if empty_prediction
        else frame.timestamp.max().isoformat(),
    )
    prediction = write_prediction_outputs(
        output_path=tmp_path / "pred", output_chunks=[frame], metadata=metadata
    )
    params = parameters(("a",))
    del (
        params["buy_threshold"],
        params["sell_threshold"],
        params["initial_capital"],
        params["restore_at"],
    )
    params["maximum_weights"] = {"a": 0.003}
    config = {
        "strategy": "etf_inventory",
        "output": str(tmp_path / "result"),
        "episode": {
            "kind": "hft",
            "task": "inventory",
            "asset_ids": ["a"],
            "feature_columns": ["prediction_score"],
            "risk_aversion": 0.0,
            "unfinished_penalty_bps": 0.0,
            "window": {"start": "2024-01-02", "end": "2024-01-02"},
            "prediction": {"paths": [str(prediction)]},
            "account": {
                "panel": str(tmp_path / "panel"),
                "fee_schedule": str(tmp_path / "fees"),
                "daily_inputs": daily,
                "parameters": params,
            },
        },
    }
    root = backtest_policy(config)
    ledger = read_portfolio_ledger_artifact(root / "ledger")
    batches = list(ledger.iter_chunks())
    assert ledger.capital.initial_capital == 1_000_000
    assert [t.quantity for b in batches for t in b.trades] == (
        [] if empty_prediction else [200, -200]
    )
    assert ledger.metadata["inventory_policy"]["forecast_horizon_seconds"] == 30
    assert ledger.metadata["inventory_policy"]["decision_interval_seconds"] == 3
    assert ledger.metadata["inventory_policy"]["capital_scale"] == 1


@pytest.mark.parametrize("capital,filled", [(3500.0, 0), (1_000_000.0, 200)])
def test_targets_cannot_override_shared_cash_or_book_capacity(capital, filled):
    panel = quotes("2024-01-02", clocks=("01:30:00", "01:30:00.020")).assign(
        instrument_type="gold_etf"
    )
    signals = pd.DataFrame(
        {
            "timestamp": [START + pd.Timedelta(milliseconds=1)] * 2,
            "asset_id": ["a", "b"],
            "prediction_score": [0.1, 0.1],
        }
    )
    params = parameters()
    params.update(initial_capital=capital, participation_rate=0.2, latency_ms=10.0)
    _, batches = run_stream(panel, signals, params)
    assert sum(t.quantity for b in batches for t in b.trades) == filled
    assert all(
        v.value >= 0 for b in batches for v in b.valuations if v.asset_id == "__cash__"
    )


def test_restore_without_opposite_depth_is_incomplete():
    panel = quotes(
        "2024-01-02", assets=("a",), clocks=("01:30:00", "01:30:00.020", "06:50:00.020")
    ).assign(instrument_type="gold_etf")
    panel.loc[2, ["bid_vol1", "ask_vol1"]] = 0
    signals = pd.DataFrame(
        {
            "timestamp": [START + pd.Timedelta(milliseconds=1)],
            "asset_id": ["a"],
            "prediction_score": [0.1],
        }
    )
    params = parameters(("a",), restore_at="14:50")
    params["latency_ms"] = 10.0
    _, batches = run_stream(panel, signals, params)
    assert [t.quantity for b in batches for t in b.trades] == [200]
    assert any(
        d.name == "inventory_restoration_status" and d.text == "INCOMPLETE"
        for d in batches[-1].diagnostics
    )
