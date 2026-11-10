"""Cash sizing against exact lot breakpoints and independent depth/fee arithmetic."""

from dataclasses import replace
from fractions import Fraction
from math import floor, fsum, sqrt

import numpy as np
import pytest

from quant_research_platform.research.portfolios.accounting import execute_stock_target
from tests.helpers.portfolio_ledger import stock_orders
from tests.research.test_stock_execution import DAY, SETTINGS, _schedule


@pytest.mark.parametrize("impact_bps", [0.0, 25.0])
def test_shared_cash_preserves_lot_breakpoints_depth_and_each_fee(impact_bps):
    targets = (700, 450)
    minimum, increments = (100, 200), (100, 1)
    books = (((10.0, 200), (10.05, 500)), ((20.0, 250), (20.10, 200)))
    cash, adv = 9500.0, 50000.0
    scales = sorted(
        {Fraction(0)}
        | {
            Fraction(quantity, target)
            for target, least, step in zip(targets, minimum, increments, strict=True)
            for quantity in range(least, target + 1, step)
        }
    )

    def price_and_fees(quantities):
        amounts, fees = [], []
        for quantity, levels in zip(quantities, books, strict=True):
            remaining = quantity
            amount = 0.0
            for price, capacity in levels:
                filled = min(remaining, capacity)
                amount += filled * price
                remaining -= filled
            amounts.append(amount)
            fees.append(
                (
                    max(amount * 0.0003, 5.0) if quantity else 0.0,
                    0.0,
                    amount * 0.0001,
                    amount * 2.0 / 10000,
                    amount * impact_bps / 10000 * sqrt(amount / adv),
                )
            )
        return amounts, fees

    expected = None
    for scale in scales:
        quantities = [
            floor(target * scale) // step * step if target * scale >= least else 0
            for target, least, step in zip(targets, minimum, increments, strict=True)
        ]
        amounts, fees = price_and_fees(quantities)
        spend = fsum([*amounts, *(fee for row in fees for fee in row)])
        if spend <= cash:
            expected = quantities, amounts, fees, spend
    quantities, amounts, fees, spend = expected
    orders = stock_orders({"a": 0, "b": 0}, dict(zip(("a", "b"), targets, strict=True)))
    orders["market_type"] = [1, 32]
    orders["mark_price"] = [10.0, 20.0]
    orders["adv"] = adv
    account = execute_stock_target(
        orders,
        cash=cash,
        execution_date=DAY,
        fee_schedule=_schedule(),
        assumptions=replace(SETTINGS, slippage_bps=2.0, impact_bps=impact_bps),
        depth=dict(zip(("a", "b"), map(np.asarray, books), strict=True)),
    )
    rows = account.asset_ledger
    assert rows.executed_quantity.tolist() == quantities
    np.testing.assert_allclose(rows.executed_delta_notional, amounts, rtol=1e-14)
    np.testing.assert_allclose(
        rows[
            [
                "commission_cost",
                "stamp_duty_cost",
                "transfer_fee_cost",
                "slippage_cost",
                "impact_cost",
            ]
        ],
        fees,
        rtol=1e-14,
        atol=1e-14,
    )
    assert rows.reason.tolist() == ["cash_pro_rata", "cash_pro_rata"]
    assert account.post_trade_cash == pytest.approx(cash - spend, rel=1e-13)
    assert account.nav_after == pytest.approx(
        cash - spend + quantities[0] * 10 + quantities[1] * 20
    )


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_execution_requires_fees_only_for_nonzero_candidate_side(side):
    initial, target = (0, 100) if side == "buy" else (100, 0)
    full = _schedule()
    selected = replace(
        full, rules=tuple(rule for rule in full.rules if rule.side == side)
    )
    opposite = replace(
        full, rules=tuple(rule for rule in full.rules if rule.side != side)
    )
    orders = stock_orders({"a": initial}, {"a": target})
    kwargs = dict(cash=1000.0, execution_date=DAY, assumptions=SETTINGS)
    account = execute_stock_target(orders, fee_schedule=selected, **kwargs)
    row = account.asset_ledger.iloc[0]
    assert row.executed_quantity == (100 if side == "buy" else -100)
    assert row.commission_cost == 5.0
    assert row.transfer_fee_cost == pytest.approx(0.01)
    assert row.stamp_duty_cost == pytest.approx(0.0 if side == "buy" else 0.05)
    assert account.post_trade_cash == pytest.approx(
        894.99 if side == "buy" else 1094.94
    )
    with pytest.raises(ValueError, match="a_share_fee_schedule_missing"):
        execute_stock_target(orders, fee_schedule=opposite, **kwargs)
    if side == "buy":
        orders["target_quantity"] = 99
    else:
        orders["sellable_quantity"] = 0
    blocked = execute_stock_target(orders, fee_schedule=opposite, **kwargs)
    assert blocked.asset_ledger.executed_quantity.item() == 0
    assert blocked.post_trade_cash == 1000.0


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_passive_fee_obligation_is_known_before_nonzero_order_reservation(side):
    from types import SimpleNamespace

    import pandas as pd

    from quant_research_platform.research.portfolios.passive import PassiveOrders
    from tests.helpers.high_frequency import quotes

    full = _schedule()
    selected = replace(
        full, rules=tuple(rule for rule in full.rules if rule.side == side)
    )
    opposite = replace(
        full, rules=tuple(rule for rule in full.rules if rule.side != side)
    )
    panel = quotes("2024-01-03", assets=("a",))
    start, at = panel.timestamp.iloc[:2]
    current = panel.iloc[[1]].set_index("asset_id")
    units = pd.Series({"a": 100.0})

    def owner(schedule, target):
        orders = PassiveOrders(
            {"queue_model": "conservative", "cancel_latency_ms": 0},
            pd.Series({"a": 4}),
            DAY,
            schedule,
            SETTINGS,
            1.0,
        )
        orders.request(start, {"a": target}, {"a": at.value})
        return orders

    target = 200 if side == "buy" else 0
    manager = owner(selected, target)
    manager.quote(at, current, units, units, 4000.0)
    assert manager.orders["a"].remaining == 100
    assert manager.frozen_cash() == pytest.approx(1004.0999 if side == "buy" else 0.0)
    event = SimpleNamespace(
        timestamp=at + pd.Timedelta(seconds=1),
        source_row=0,
        asset_id="a",
        side="sell" if side == "buy" else "buy",
        price=9.99 if side == "buy" else 10.01,
        volume=1000.0,
    )
    fill = manager.trade(event)
    assert fill[2] == (100 if side == "buy" else -100)
    assert fill[3] == pytest.approx(999.0 if side == "buy" else -1001.0)
    np.testing.assert_allclose(
        fill[4],
        [
            5.0,
            0.0 if side == "buy" else 0.5005,
            0.0999 if side == "buy" else 0.1001,
            0.0,
            0.0,
        ],
    )
    missing = owner(opposite, target)
    with pytest.raises(ValueError, match="a_share_fee_schedule_missing"):
        missing.quote(at, current, units, units, 4000.0)
    assert not missing.orders and missing.frozen_cash() == 0.0
    blocked = owner(opposite, 199 if side == "buy" else 0)
    blocked.quote(at, current, units, units if side == "buy" else units * 0, 4000.0)
    assert not blocked.orders and blocked.frozen_cash() == 0.0


@pytest.mark.parametrize("shortfall", [0.0, 0.01])
def test_exact_cash_budget_uses_currency_tolerance_not_share_loss(shortfall):
    schedule = _schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0)
    bought = execute_stock_target(
        stock_orders({"a": 0}, {"a": 300}, prices=10.0),
        cash=2006.0 - shortfall,
        execution_date=DAY,
        fee_schedule=schedule,
        assumptions=SETTINGS,
        depth={"a": np.array([[10.0, 100], [10.01, 200]])},
    )
    row = bought.asset_ledger.iloc[0]
    # 100*10 + 100*10.01 + 5 = 2006. VWAP division/multiplication
    # must not discard the second lot, but a genuine cent shortage does.
    expected = 200 if shortfall == 0 else 100
    assert row.executed_quantity == expected
    assert row.executed_delta_notional == pytest.approx(
        2001.0 if expected == 200 else 1000.0
    )
    assert row.commission_cost == 5.0
    assert bought.post_trade_cash == pytest.approx(0.0 if expected == 200 else 1000.99)
    sold = execute_stock_target(
        stock_orders({"a": 1}, {"a": 0}, prices=0.69),
        cash=4.31 - shortfall,
        execution_date=DAY,
        fee_schedule=schedule,
        assumptions=SETTINGS,
    )
    row = sold.asset_ledger.iloc[0]
    # 4.31 cash + 0.69 proceeds pays exactly the 5 CNY minimum commission.
    assert row.executed_quantity == (-1 if shortfall == 0 else 0)
    assert row.commission_cost == (5.0 if shortfall == 0 else 0.0)
    assert sold.post_trade_cash == pytest.approx(0.0 if shortfall == 0 else 4.30)
    assert row.reason is None if shortfall == 0 else row.reason == "cash_limit"


@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize("budget", ["exact", "cent_short", "zero"])
def test_passive_reservation_uses_the_same_currency_budget(side, budget):
    from types import SimpleNamespace

    import pandas as pd

    from quant_research_platform.research.portfolios.passive import PassiveOrders
    from tests.helpers.high_frequency import quotes

    schedule = _schedule(broker_rate=0.0, transfer_rate=0.0, stamp_rate=0.0)
    panel = quotes("2024-01-03", assets=("a",))
    panel["mid"], panel["bid_px1"], panel["ask_px1"] = (
        (1.10, 1.09, 1.11) if side == "buy" else (0.68, 0.67, 0.69)
    )
    start, at = panel.timestamp.iloc[:2]
    manager = PassiveOrders(
        {"queue_model": "conservative", "cancel_latency_ms": 0},
        pd.Series({"a": 4}),
        DAY,
        schedule,
        SETTINGS,
        1.0,
    )
    units = pd.Series({"a": 0.0 if side == "buy" else 1.0})
    manager.request(start, {"a": 300 if side == "buy" else 0}, {"a": at.value})
    cash = 223.0 if side == "buy" else 4.31
    cash = 0.0 if budget == "zero" else cash - (0.01 if budget == "cent_short" else 0.0)
    manager.quote(at, panel.iloc[[1]].set_index("asset_id"), units, units, cash)
    expected = (
        (0 if budget == "zero" else 200 if budget == "exact" else 100)
        if side == "buy"
        else (1 if budget == "exact" else 0)
    )
    assert (manager.orders["a"].remaining if manager.orders else 0) == expected
    if expected:
        assert manager.frozen_cash() == pytest.approx(
            expected * 1.09 + 5 if side == "buy" else 4.31
        )

    event = SimpleNamespace(
        timestamp=at + pd.Timedelta(seconds=1),
        source_row=0,
        asset_id="a",
        side="sell" if side == "buy" else "buy",
        price=1.08 if side == "buy" else 0.70,
        volume=1000.0,
    )
    fill = manager.trade(event)
    if expected:
        assert fill[2] == (expected if side == "buy" else -expected)
        assert fill[3] == pytest.approx(expected * (1.09 if side == "buy" else -0.69))
        np.testing.assert_array_equal(fill[4], [5.0, 0.0, 0.0, 0.0, 0.0])
        assert fsum((cash, -fill[3], *(-fill[4]))) == pytest.approx(
            108.99 if side == "buy" and budget == "cent_short" else 0.0,
            abs=1e-12,
        )
    else:
        assert fill is None
    assert not manager.orders and manager.frozen_cash() == 0.0
