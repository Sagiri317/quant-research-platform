"""Historical transfer fees use supplied share face value, never trading price."""

from dataclasses import replace
from datetime import date

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
from quant_research_platform.markets.implementations.a_share.execution import (
    prepare_trade_cost,
)
from quant_research_platform.markets.implementations.a_share.portfolio_sessions import (
    iter_portfolio_sessions,
)
from quant_research_platform.research.portfolios.accounting import execute_stock_target
from quant_research_platform.research.portfolios.daily import _PANEL_COLUMNS
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.helpers.high_frequency import daily_view, parameters, quotes, trade_input
from tests.helpers.portfolio_inputs import _processed_specs, closed_market_inputs
from tests.helpers.portfolio_ledger import stock_orders
from tests.research.test_portfolio_ledger import _real_fee_schedule
from tests.research.test_stock_execution import SETTINGS


def test_historical_face_fee_execution_and_optimizer_share_independent_units(tmp_path):
    schedule = _real_fee_schedule(tmp_path)
    old = date(2015, 7, 31)
    cost = prepare_trade_cost(old, "shanghai", "buy", schedule, SETTINGS)
    prices = np.array([10.0, 20.0])
    face = np.array([0.1, 0.2])
    amounts = np.array([1000.0, 4000.0])
    orders = stock_orders({"a": 0, "b": 0}, {"a": 100, "b": 200}, prices=prices)
    orders["face_value_per_share"] = face
    account = execute_stock_target(
        orders,
        cash=10000.0,
        execution_date=old,
        fee_schedule=schedule,
        assumptions=SETTINGS,
    )
    np.testing.assert_allclose(account.asset_ledger.transfer_fee_cost, [0.006, 0.024])
    assert account.post_trade_cash == pytest.approx(4990 - 0.03)
    totals, gradient = cost.totals_and_gradients(
        amounts,
        np.full(2, 1e6),
        prices=prices,
        face_value_per_share=face,
    )
    # Continuous optimization has no discrete minimum commission; d(cost)/d(amount).
    np.testing.assert_allclose(totals, [0.306, 1.224])
    np.testing.assert_allclose(gradient, [0.000306, 0.000306])
    for shift in [-1.0, 1.0]:
        zero, tangent = cost.totals_and_gradients(
            np.full(2, shift),
            np.full(2, 1e6),
            prices=prices,
            face_value_per_share=face,
        )
        np.testing.assert_allclose(zero, shift * gradient)
        np.testing.assert_allclose(tangent, gradient)
    with pytest.raises(ValueError, match="face_value_per_share_required"):
        execute_stock_target(
            orders.drop(columns="face_value_per_share"),
            cash=10000.0,
            execution_date=old,
            fee_schedule=schedule,
            assumptions=SETTINGS,
        )
    # The new amount-based rule needs no face-value data.
    result = execute_stock_target(
        orders.drop(columns="face_value_per_share"),
        cash=10000.0,
        execution_date=date(2015, 8, 3),
        fee_schedule=schedule,
        assumptions=SETTINGS,
    )
    np.testing.assert_allclose(result.asset_ledger.transfer_fee_cost, [0.02, 0.08])


@pytest.mark.parametrize("passive", [False, True])
def test_hft_face_fees_survive_native_ledger_and_partial_fills(tmp_path, passive):
    panel = quotes(
        "2015-07-31",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:09"),
    )
    panel["exchange"] = "SSE"
    panel["bid_vol1"] = 0.0
    view = daily_view(panel)
    view.panel["face_value_per_share"] = 0.1
    view.valuation["face_value_per_share"] = 0.1
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    params = parameters(("a",))
    params["minimum_commission"] = 5.0
    trade_session = None
    if passive:
        params["passive"] = {"queue_model": "conservative", "cancel_latency_ms": 0.0}
        # Cumulative per-order commission is paid once across the two partial fills.
        trade_session = trade_input(
            tmp_path / "trades",
            [(4, "a", "sell", 30, 9.99), (7, "a", "sell", 170, 9.99)],
        )
        frame = read_dataset(tmp_path / "trades")
        frame["timestamp"] -= pd.Timestamp("2024-01-02") - pd.Timestamp("2015-07-31")
        spec = open_dataset_session(tmp_path / "trades").spec
        typed_spec = InterimDataSpec(
            **{
                key: (
                    tuple(value)
                    if key in {"primary_key", "partition_columns"}
                    else value
                )
                for key, value in spec.items()
                if key not in {"schema_id", "required_columns"}
            }
        )
        write_dataset(tmp_path / "historical-trades", frame, spec=typed_spec)
        trade_session = open_dataset_session(tmp_path / "historical-trades")
    window = TimeWindow(panel.timestamp.min(), panel.timestamp.max())
    capital = PortfolioCapital("hft_inventory", 5000.0, "CNY", "cash_equity")
    schedule = _real_fee_schedule(tmp_path)

    def produce(emit):
        for batch in inventory_batches(
            [(view, panel, signals)],
            schedule,
            params,
            window,
            trade_session=trade_session,
        ):
            emit(batch)

    output = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "kind": "a_share_hft",
        },
        produce=produce,
    )
    batches = list(read_portfolio_ledger_artifact(output).iter_chunks())
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == ([30, 170] if passive else [200])
    flows = [flow for batch in batches for flow in batch.cashflows]
    assert -sum(
        flow.amount for flow in flows if flow.kind == "transfer_fee_cost"
    ) == pytest.approx(0.012)
    assert -sum(
        flow.amount for flow in flows if flow.kind == "commission_cost"
    ) == pytest.approx(5.0)


def test_portfolio_session_preserves_explicit_optional_face_value(tmp_path):
    inputs, _, calendar, _ = closed_market_inputs(tmp_path, fault="normal_open")
    panel = read_dataset(inputs["panel"])
    panel["face_value_per_share"] = 0.2
    spec = _processed_specs().get("market_panel_1d")
    spec = replace(
        spec, column_roles={**spec.column_roles, "face_value_per_share": "context"}
    )
    root = tmp_path / "panel-with-face"
    write_dataset(root, panel, spec=spec)
    views = list(
        iter_portfolio_sessions(
            market_bars_root=inputs["market_bars"],
            stock_master_root=inputs["stock_master"],
            panel_root=root,
            panel_columns=_PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2023-01-02", "2023-01-05"),
        )
    )
    assert views[0].panel.face_value_per_share.eq(0.2).all()
