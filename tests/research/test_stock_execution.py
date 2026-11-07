"""Share execution checked against exchange quantity rules and hand cash budgets."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeSchedule,
    round_stock_orders,
    stock_order_limits,
)
from quant_research_platform.research.portfolios.accounting import execute_stock_target
from tests.helpers.portfolio_ledger import stock_orders
from tests.research.test_portfolio_ledger import _schedule

SETTINGS = ExecutionAssumptions(10000.0, 0.0, 0.0, 1.0, (1.0,), 5.0)
DAY = date(2024, 1, 3)


@pytest.mark.parametrize(
    "market,requested,available,expected",
    [
        (1, 350, None, 300),
        (1, 9399.999999999998, None, 9400),
        (1, 9399.9, None, 9300),
        (32, 350, None, 350),
        (32, 199, None, 0),
        (1, 150, 250, 150),
        (1, 120, 250, 100),
        (1, 50, 250, 50),
        (1, 49, 250, 0),
        (32, 199, 250, 0),
        (32, 150, 150, 150),
        (1, 0, 0, 0),
        (32, 100, 0, 0),
    ],
)
def test_limit_order_quantity_rules(market, requested, available, expected):
    _, minimum, increment, _ = stock_order_limits(pd.Series([market]), DAY)
    result = round_stock_orders(
        np.array([requested]),
        minimum,
        increment,
        sellable=None if available is None else np.array([available]),
    )
    assert result.tolist() == [expected]


@pytest.mark.parametrize(
    "day,maximum", [(date(2020, 8, 21), 1000000), (date(2020, 8, 24), 300000)]
)
def test_growth_board_maximum_has_effective_date(day, maximum):
    assert stock_order_limits(pd.Series([16]), day)[3].item() == maximum


def test_etfs_use_fund_fees_and_hundred_unit_orders_in_mixed_account():
    orders = stock_orders(
        {"stock": 0, "equity": 0, "gold": 0},
        {
            "stock": 350,
            "equity": 350,
            "gold": 350,
        },
    )
    orders["instrument_type"] = pd.Series(
        {
            "stock": "stock",
            "equity": "equity_etf",
            "gold": "gold_etf",
        }
    )
    orders["market_type"] = pd.Series({"stock": 1, "equity": 4, "gold": 1})
    stock_fees = _schedule()
    fund_rules = tuple(
        replace(
            rule,
            market="etf",
            rate=0.0001 if rule.fee_code == "BROKER_COMMISSION" else 0.0,
        )
        for rule in stock_fees.rules
    )
    result = execute_stock_target(
        orders,
        cash=10000.0,
        execution_date=DAY,
        fee_schedule=FeeSchedule(stock_fees.rules + fund_rules, "mixed"),
        assumptions=SETTINGS,
    )
    rows = result.asset_ledger.set_index("asset_id")
    assert rows.executed_quantity.to_dict() == {
        "stock": 300,
        "equity": 300,
        "gold": 300,
    }
    assert rows.loc[["equity", "gold"], "transfer_fee_cost"].tolist() == [0.0, 0.0]
    assert rows.loc["stock", "transfer_fee_cost"] > 0.0
    assert rows.commission_cost.tolist() == [5.0, 5.0, 5.0]


def test_sells_fund_shared_cash_buys_without_selling_todays_purchase():
    orders = stock_orders({"a": 250, "b": 0}, {"a": 0, "b": 500})
    orders.loc["a", "sellable_quantity"] = 150
    account = execute_stock_target(
        orders,
        cash=100.0,
        execution_date=DAY,
        fee_schedule=_schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0),
        assumptions=SETTINGS,
    )
    rows = account.asset_ledger.set_index("asset_id")
    assert rows.executed_quantity.to_dict() == {"a": -150, "b": 200}
    assert rows.reason.to_dict() == {"a": "locked_shares", "b": "cash_pro_rata"}
    assert account.post_trade_cash == 40.0
    assert account.nav_after == 340.0  # 350 capital less two five-yuan orders.
    assert rows.commission_cost.tolist() == [5.0, 5.0]


def test_cash_sizing_is_proportional_order_invariant_and_includes_minimum_fees():
    orders = stock_orders({"a": 0, "b": 0, "c": 0}, {"a": 1000, "b": 1000, "c": 1000})
    results = [
        execute_stock_target(
            frame,
            cash=2000.0,
            execution_date=DAY,
            fee_schedule=_schedule(),
            assumptions=SETTINGS,
        )
        for frame in (orders, orders.iloc[::-1])
    ]
    pd.testing.assert_frame_equal(results[0].asset_ledger, results[1].asset_ledger)
    assert results[0].asset_ledger.executed_quantity.tolist() == [600, 600, 600]
    assert results[0].post_trade_cash == pytest.approx(2000 - 1800 - 15 - 0.18)
    assert results[0].nav_before - results[0].nav_after == pytest.approx(15.18)


def test_missing_facts_only_block_requested_orders_and_never_fund_buys():
    orders = stock_orders({"a": 1000, "b": 0, "c": 50}, {"a": 0, "b": 1000, "c": 50})
    orders["can_exit"] = orders.can_exit.astype("boolean")
    orders.loc["a", "can_exit"] = pd.NA
    orders.loc["c", ["buy_capacity", "sell_capacity", "adv"]] = np.nan
    result = execute_stock_target(
        orders,
        cash=0.0,
        execution_date=DAY,
        fee_schedule=_schedule(),
        assumptions=SETTINGS,
    )
    rows = result.asset_ledger.set_index("asset_id")
    assert rows.executed_quantity.tolist() == [0, 0, 0]
    assert rows.reason.iloc[:2].tolist() == ["execution_fact_unknown", "cash_pro_rata"]
    assert pd.isna(rows.reason.iloc[2])
    assert result.post_trade_cash == 0
    assert result.nav_after == 1050


def test_blocked_etf_order_preserves_inventory_and_cash_without_fees():
    orders = stock_orders({"a": 100}, {"a": 200}, prices=10.0)
    orders["instrument_type"] = "equity_etf"
    orders["can_enter"] = False
    result = execute_stock_target(
        orders,
        cash=1000.0,
        execution_date=DAY,
        fee_schedule=_schedule(),
        assumptions=SETTINGS,
    )
    row = result.asset_ledger.iloc[0]
    assert row.executed_quantity == 0
    assert row.reason == "market_blocked"
    assert result.post_trade_cash == 1000.0
    assert result.nav_before == result.nav_after == 2000.0


@pytest.mark.parametrize(
    "market,quantity,commission", [(1, 1000100, 305.0), (32, 100001, 34.9403)]
)
def test_maximum_order_splits_charge_actual_minimum_commissions(
    market, quantity, commission
):
    # STAR 100001 shares split into 99801 and 200, not an illegal one-share tail.
    orders = stock_orders({"a": 0}, {"a": quantity}, market_types=market)
    result = execute_stock_target(
        orders,
        cash=2000000.0,
        execution_date=DAY,
        fee_schedule=_schedule(transfer_rate=0.0),
        assumptions=SETTINGS,
    )
    row = result.asset_ledger.iloc[0]
    assert row.executed_quantity == quantity
    assert row.commission_cost == pytest.approx(commission)
    assert result.post_trade_cash == pytest.approx(2000000.0 - quantity - commission)


def test_bid_ask_cash_and_mark_valuation_remain_distinct():
    orders = stock_orders({"a": 100}, {"a": 0}, prices=10.0)
    orders.loc["a", "sell_price"] = 9.9
    result = execute_stock_target(
        orders,
        cash=0.0,
        execution_date=DAY,
        fee_schedule=_schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0),
        assumptions=SETTINGS,
    )
    assert result.asset_ledger.executed_quantity.item() == -100
    assert result.asset_ledger.executed_delta_notional.item() == -990.0
    assert result.nav_before == 1000
    assert result.nav_after == 985.0


def test_cash_receivable_cannot_fund_buys_and_locked_shares_cannot_fund_them_either():
    orders = stock_orders({"a": 900, "b": 0}, {"a": 0, "b": 1000})
    orders.loc["a", "sellable_quantity"] = 600
    result = execute_stock_target(
        orders,
        cash=0.0,
        receivable=100.0,
        execution_date=DAY,
        fee_schedule=_schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0),
        assumptions=replace(SETTINGS, minimum_commission=0.0),
    )
    assert result.asset_ledger.executed_quantity.tolist() == [-600, 600]
    assert result.nav_after == result.nav_before == 1000
    assert result.post_trade_cash == 0


@pytest.mark.parametrize(
    "column,value",
    [
        ("quantity", -1),
        ("sellable_quantity", 101),
        ("target_quantity", np.nan),
        ("buy_price", np.inf),
        ("can_enter", "yes"),
    ],
)
def test_stock_execution_validates_external_order_facts(column, value):
    orders = stock_orders({"a": 100}, {"a": 200})
    orders[column] = orders[column].astype(object)
    orders.loc["a", column] = value
    with pytest.raises(ValueError):
        execute_stock_target(
            orders,
            cash=1000.0,
            execution_date=DAY,
            fee_schedule=_schedule(),
            assumptions=SETTINGS,
        )
