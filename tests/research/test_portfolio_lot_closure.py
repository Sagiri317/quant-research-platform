"""Native lot closure must preserve true costs and strict scientific limits."""

from collections import deque
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_LEDGER_COST_COLUMNS,
    RobustPortfolioConfig,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    round_stock_orders,
)
from quant_research_platform.research.portfolios import daily, optimizer
from tests.helpers.portfolio_inputs import decision_inputs, portfolio_inputs


def _inputs(*, nav, price=1.0, rate=0.0004, config=None, market_type=1):
    inputs = decision_inputs(benchmark=(1.0,), holdings=(0.0,))
    decision = inputs["enhanced_decision"]
    _, fees, _ = portfolio_inputs()
    fees["rate"] = np.where(fees.fee_code.eq("BROKER_COMMISSION"), rate, 0.0)
    inputs.update(
        candidate=replace(
            inputs["candidate"], construction="robust_long_only", benchmark_id=None
        ),
        cash=float(nav),
        nav=float(nav),
        assumptions=replace(inputs["assumptions"], initial_nav=float(nav)),
        fee_schedule=fee_schedule_from_frame(fees, content_identity="synthetic"),
        alpha=pd.Series([0.25], index=["a"]),
        enhanced_decision=replace(
            decision,
            benchmark_weights=pd.Series(dtype=float),
            expected_alpha=pd.Series([0.02], index=["a"]),
            risk=None,
            holding_sessions=21,
            calibration=replace(
                decision.calibration, slope=0.08, slope_standard_error=0.0
            ),
        ),
        optimizer_config=config or optimizer.AbsoluteRiskConfig(),
        robust_config=RobustPortfolioConfig(risk_aversion=0.0),
        decision_reason="robust_optimizer",
    )
    for key in ("panel", "next_panel", "endpoint_panel"):
        inputs[key][["open", "close"]] = price
        inputs[key]["market_type"] = market_type
    return inputs


@pytest.mark.parametrize(
    ("nav", "price", "rate", "name_limit", "market_type", "expected"),
    [
        (100.04, 1.0, 0.0004, None, 1, 100),
        (200.04, 1.0, 0.0004, 0.5, 1, 100),
        (198.0396, 1.0, 0.0004, 0.5, 1, 0),
        (200.039999, 1.0, 0.0004, 0.5, 1, 0),
        (200.04 - 1e-9, 1.0, 0.0004, 0.5, 1, 0),
        (100.04 - 1e-9, 1.0, 0.0004, None, 1, 0),
        (1000.4, 10.0, 0.0004, None, 1, 100),
        (1461.241, 7.3, 0.0017, 0.5, 1, 100),
        (200.08, 1.0, 0.0004, None, 32, 200),
        (199.0796, 1.0, 0.0004, None, 32, 0),
    ],
)
def test_native_lot_closure_accepts_true_boundaries_and_rejects_infeasible_neighbors(
    nav, price, rate, name_limit, market_type, expected
):
    inputs = _inputs(
        nav=nav,
        price=price,
        rate=rate,
        market_type=market_type,
        config=optimizer.AbsoluteRiskConfig(name_weight_limit=name_limit),
    )
    account, _, solved, _, ledger = daily._build_session(**inputs)
    assert solved is not None
    assert ledger.executed_quantity.tolist() == [expected]
    fee = float(ledger.loc[:, PORTFOLIO_LEDGER_COST_COLUMNS].to_numpy().sum())
    assert fee == pytest.approx(expected * price * rate, abs=1e-12)
    assert account.post_trade_cash == pytest.approx(
        nav - expected * price - fee, abs=1e-12
    )
    assert ledger.attrs["realized_constraints"]["execution_constraints_status"] == (
        "SATISFIED"
    )
    if name_limit is not None and expected:
        assert expected * price / account.nav_after <= name_limit + 4e-16


@pytest.mark.parametrize("constraint", ["volatility", "style"])
@pytest.mark.parametrize("shortfall", [0.0, 1e-6])
def test_native_lot_closure_uses_strict_post_fee_risk_boundaries(constraint, shortfall):
    config = (
        optimizer.AbsoluteRiskConfig(total_volatility_limit=float(np.sqrt(0.02) / 2))
        if constraint == "volatility"
        else optimizer.AbsoluteRiskConfig(style_exposure_limit=0.5)
    )
    inputs = _inputs(nav=200.04 - shortfall, config=config)
    inputs["enhanced_decision"] = replace(
        inputs["enhanced_decision"], risk=decision_inputs()["enhanced_decision"].risk
    )
    _, _, _, _, ledger = daily._build_session(**inputs)
    assert ledger.executed_quantity.tolist() == [100 if shortfall == 0.0 else 0]


def test_actual_minimum_commission_and_capacity_do_not_become_continuous_costs():
    inputs = _inputs(nav=100.04)
    inputs["assumptions"] = replace(inputs["assumptions"], minimum_commission=5.0)
    account, _, _, _, ledger = daily._build_session(**inputs)
    assert ledger.executed_quantity.tolist() == [0]
    assert account.post_trade_cash == 100.04

    inputs = _inputs(nav=200.04)
    inputs["amount_history"] = {
        key: deque([99.0] * 20) for key in inputs["amount_history"]
    }
    account, _, _, _, ledger = daily._build_session(**inputs)
    assert ledger.executed_quantity.tolist() == [0]
    assert account.post_trade_cash == 200.04


def test_actual_utility_rejects_a_nearest_buy_with_a_loss_from_minimum_commission():
    inputs = _inputs(nav=1500.0)
    inputs["assumptions"] = replace(inputs["assumptions"], minimum_commission=5.0)
    decision = inputs["enhanced_decision"]
    risk = decision_inputs()["enhanced_decision"].risk
    risk = replace(
        risk,
        factor_covariance=risk.factor_covariance * 0.0,
        specific_variance=pd.Series([0.25], index=["a"]),
    )
    inputs["enhanced_decision"] = replace(decision, risk=risk, holding_sessions=252)
    inputs["robust_config"] = RobustPortfolioConfig(risk_aversion=2.0)
    account, _, solved, _, ledger = daily._build_session(**inputs)
    assert 50.0 < solved.weights.iloc[0] * 1500.0 < 100.0
    # U(100) = .02*(100/1500)-.25*(100/1500)^2-5/1500 < U(0).
    assert 0.02 / 15 - 0.25 / 225 - 5 / 1500 < 0.0
    assert ledger.executed_quantity.tolist() == [0]
    assert account.post_trade_cash == 1500.0


def test_fully_invested_buys_the_affordable_lot_despite_negative_net_utility():
    inputs = _inputs(
        nav=100.04, config=optimizer.AbsoluteRiskConfig(fully_invested=True)
    )
    decision = inputs["enhanced_decision"]
    inputs["enhanced_decision"] = replace(
        decision,
        calibration=replace(decision.calibration, slope=0.0),
        expected_alpha=decision.expected_alpha * 0.0,
    )
    account, _, solved, _, ledger = daily._build_session(**inputs)
    assert solved.objective_value < 0.0
    assert ledger.executed_quantity.tolist() == [100]
    assert account.post_trade_cash == pytest.approx(0.0, abs=1e-12)


def test_nearest_market_quantity_handles_both_regular_and_odd_sale_grids():
    requested = np.array([99.999997, 100.0, 149.999997, 350.0, 1.0, 0.0, 125.0])
    minimum = np.full(len(requested), 100)
    increment = np.full(len(requested), 100)
    sellable = np.array([350.0, 350.0, 350.0, 350.0, 1.0, 350.0, 350.0])
    assert round_stock_orders(
        requested, minimum, increment, sellable=sellable, nearest=True
    ).tolist() == [100, 100, 150, 350, 1, 0, 100]
    assert round_stock_orders(
        requested, minimum, increment, sellable=sellable
    ).tolist() == ([50, 100, 100, 350, 1, 0, 100])
    assert round_stock_orders(
        np.array([49.99, 50.0, 50.01, 100.0, 100.01, 199.99999]),
        np.array([100, 100, 100, 200, 200, 200]),
        np.array([100, 100, 100, 1, 1, 1]),
        nearest=True,
    ).tolist() == [0, 0, 100, 0, 200, 200]


def test_native_sale_neighbor_preserves_t1_locks_and_freezes():
    inputs = _inputs(nav=500.0)
    inputs["holdings"] = pd.Series([350.0], index=["a"])
    inputs["held_units"] = pd.Series([350.0], index=["a"])
    inputs["cash"] = 150.0
    inputs["locked_quantities"] = {"a": 251.0}
    decision = inputs["enhanced_decision"]
    inputs["enhanced_decision"] = replace(
        decision, expected_alpha=pd.Series([-0.02], index=["a"])
    )
    _, _, _, _, ledger = daily._build_session(**inputs)
    assert ledger.executed_quantity.tolist() == [-99]
    inputs["execution"]["can_exit"] = False
    _, _, _, _, ledger = daily._build_session(**inputs)
    assert ledger.executed_quantity.tolist() == [0]


def test_native_sale_neighbor_recovers_the_regular_grid_beside_odd_lots():
    inputs = decision_inputs(benchmark=(1.0,), holdings=(0.7,))
    for key in ("panel", "next_panel", "endpoint_panel"):
        inputs[key][["open", "close"]] = 1.0
    inputs.update(
        holdings=pd.Series([350.0], index=["a"]),
        held_units=pd.Series([350.0], index=["a"]),
        cash=150.0,
        nav=500.0,
        assumptions=replace(inputs["assumptions"], initial_nav=500.0),
        enhanced_decision=replace(
            inputs["enhanced_decision"],
            expected_alpha=pd.Series([-0.02], index=["a"]),
        ),
        optimizer_config=optimizer.OptimizerConfig(
            active_name_limit=0.5,
            tracking_error_limit=1.0,
            target_turnover_limit=1.0,
        ),
    )
    account, _, solved, _, ledger = daily._build_session(**inputs)
    assert 250.0 <= solved.weights.iloc[0] * 500.0 < 300.0
    # Sellable=350 admits both 50+100*k and 100*k; the 100-share sale is closest.
    assert ledger.executed_quantity.tolist() == [-100]
    assert account.post_trade_cash == 250.0


@pytest.mark.parametrize("exposure_error", [0.0, 1e-8])
def test_native_zero_exposure_counts_cancellation_roundoff_in_product_units(
    exposure_error,
):
    inputs = _inputs(
        nav=400.16, config=optimizer.AbsoluteRiskConfig(style_exposure_limit=0.0)
    )
    two = decision_inputs(benchmark=(0.5, 0.5), holdings=(0.0, 0.0))
    for key in (
        "panel",
        "next_panel",
        "endpoint_panel",
        "dataset",
        "execution",
        "holdings",
        "held_units",
        "amount_history",
    ):
        inputs[key] = two[key]
    for key in ("panel", "next_panel", "endpoint_panel"):
        inputs[key][["open", "close"]] = 1.0
    inputs["alpha"] = pd.Series([0.25, 0.25], index=["a", "b"])
    inputs["enhanced_decision"] = replace(
        inputs["enhanced_decision"],
        expected_alpha=pd.Series([0.02, 0.02], index=["a", "b"]),
        exposures=pd.DataFrame(
            {"asset_id": ["a", "b"], "style": [-0.1, 0.3 + exposure_error]}
        ),
    )
    account, _, _, _, ledger = daily._build_session(**inputs)
    assert ledger.executed_quantity.tolist() == (
        [300, 100] if exposure_error == 0.0 else [0, 0]
    )
    if exposure_error == 0.0:
        assert account.nav_after == pytest.approx(400.0, abs=1e-12)
        # 3/4*(-1/10)+1/4*(3/10)=0; each product has nonzero roundoff scale.
        assert abs(0.75 * -0.1 + 0.25 * 0.3) < 2e-17


@pytest.mark.parametrize("risk_shortfall", [0.0, 1e-8])
def test_strict_tracking_error_propagates_nav_normalization_without_zeroing_risk(
    risk_shortfall,
):
    weights = np.array([100001 / 200000, 99999 / 200000, 0.0])
    benchmark = np.array([0.5, 0.5, 0.0])
    exposure = np.array([[-1.0], [1.0], [0.0]])
    specific = np.array([0.01, 0.01, 0.0])
    # Exact arithmetic: .04*(2/200000)^2 + .01*2*(1/200000)^2 = 4.5e-12.
    limit = float(np.sqrt(4.5e-12) * (1.0 - risk_shortfall))
    facts, violation = optimizer._portfolio_risk_limits(
        weights,
        benchmark,
        exposure,
        specific,
        np.array([[0.04]]),
        np.array([False, False, True]),
        ("style",),
        optimizer.OptimizerConfig(
            tracking_error_limit=limit,
            active_name_limit=0.1,
            style_active_limit=1.0,
        ),
        strict=True,
    )
    actual_variance = 0.04 * ((weights[1] - 0.5) - (weights[0] - 0.5)) ** 2 + 0.01 * (
        (weights[0] - 0.5) ** 2 + (weights[1] - 0.5) ** 2
    )
    assert facts["realized_tracking_error"] ** 2 == pytest.approx(
        actual_variance, rel=1e-14
    )
    assert actual_variance > 4.5e-12
    assert facts["realized_constraints_status"] == (
        "SATISFIED" if risk_shortfall == 0.0 else "BREACHED"
    )
    assert (violation[0] == 0.0) == (risk_shortfall == 0.0)


@pytest.mark.parametrize("next_quote", [True, False])
def test_full_investment_rebalances_despite_lot_cash_and_respects_missing_open(
    next_quote,
):
    inputs = decision_inputs(benchmark=(0.5, 0.5), holdings=(1.0, 0.0))
    for key in ("panel", "next_panel", "endpoint_panel"):
        inputs[key]["open"] = inputs[key]["close"] = [1.0, 1.01]
    if not next_quote:
        inputs["next_panel"] = inputs["next_panel"].loc[lambda f: f.asset_id.eq("a")]
    inputs.update(
        candidate=replace(
            inputs["candidate"], construction="robust_long_only", benchmark_id=None
        ),
        holdings=pd.Series({"a": 10000.0}),
        held_units=pd.Series({"a": 10000.0}),
        nav=10000.0,
        alpha=pd.Series({"a": -0.25, "b": 0.25}),
        amount_history={k: deque([10000.0] * 20) for k in inputs["amount_history"]},
        enhanced_decision=replace(
            inputs["enhanced_decision"],
            benchmark_weights=pd.Series(dtype=float),
            calibration=None,
            expected_alpha=None,
        ),
        optimizer_config=optimizer.AbsoluteRiskConfig(
            fully_invested=True, target_turnover_limit=0.2
        ),
        robust_config=RobustPortfolioConfig(objective="rank"),
        decision_reason="robust_optimizer",
    )
    account, _, solved, _, ledger = daily._build_session(**inputs)
    assert solved.weights["b"] == pytest.approx(0.2, abs=1e-7)
    assert ledger.set_index("asset_id").executed_quantity["b"] == (
        1900 if next_quote else 0
    )
    if next_quote:
        assert account.post_trade_cash == pytest.approx(81.0)
