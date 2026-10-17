"""Hand-derived economic and differential oracles for fee-funded targets."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios import optimizer


def _assets(current, *, alpha=None, benchmark=None):
    count = len(current) - 1
    return pd.DataFrame(
        dict(
            asset_id=[f"a{i}" for i in range(count)] + ["cash"],
            current_weight=current,
            benchmark_weight=benchmark or [1.0 / count] * count + [0.0],
            expected_alpha=alpha or [0.2] * count + [0.0],
            specific_variance=[0.01] * count + [0.0],
            style=[1.0] * count + [0.0],
            adv=[1e308] * count + [np.inf],
            is_cash=[False] * count + [True],
        )
    )


def _linear_cost(buy_rate, sell_rate):
    def cost(buy, sell):
        return (
            float(buy_rate * buy[:-1].sum() + sell_rate * sell[:-1].sum()),
            np.r_[np.full(len(buy) - 1, buy_rate), 0.0],
            np.r_[np.full(len(sell) - 1, sell_rate), 0.0],
        )

    return cost


def _solve(assets, *, nav=100.0, cost=None, config=None, **kwargs):
    return optimizer.optimize_portfolio(
        assets,
        factor_covariance=pd.DataFrame([[0.04]], index=["style"], columns=["style"]),
        style_columns=("style",),
        cash_column="is_cash",
        expected_transaction_cost=cost or _linear_cost(0.1, 0.1),
        nav=nav,
        assumptions=ExecutionAssumptions(nav, 0.0, 0.0, 1.0, (1.0,)),
        config=config
        or optimizer.OptimizerConfig(
            active_name_limit=1.0,
            tracking_error_limit=1.0,
            target_turnover_limit=1.0,
            style_active_limit=1.0,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("nav", [1e-300, 1e-100, 100.0, 1e100, 1e300])
@pytest.mark.parametrize("receivable", [0.0, 0.2])
def test_purchase_reserves_exact_fee_and_keeps_receivable_in_full_nav(nav, receivable):
    # A buy of x costs x/10, so x + x/10 = the spendable budget.
    budget = 1.0 - receivable
    result = _solve(_assets([0.0, budget]), nav=nav, receivable_weight=receivable)
    stock, reserve = budget / 1.1, budget / 11.0
    assert result.weights.to_numpy() == pytest.approx([stock, reserve], abs=2e-10)
    fee = result.expected_transaction_cost / nav
    assert fee == pytest.approx(reserve, rel=1e-10)
    assert result.weights.sum() == pytest.approx(budget, abs=1e-12)
    # Check conservation in pre-fee-NAV units, including cost and receivable.
    cash_after = result.weights.iloc[-1] - fee
    assert cash_after == pytest.approx(0.0, abs=2e-10)
    assert stock + cash_after + receivable + fee == pytest.approx(1.0, abs=2e-10)
    net_stock = stock / (1.0 - fee)
    assert result.factor_variance == pytest.approx(
        0.04 * (net_stock - 1.0) ** 2, abs=1e-14
    )
    assert result.specific_variance == pytest.approx(
        0.01 * (net_stock - 1.0) ** 2, abs=1e-14
    )
    assert result.target_turnover == pytest.approx(stock, abs=2e-10)


@pytest.mark.parametrize("bound", ["name", "tracking_error"])
def test_fee_funded_allocation_reaches_actual_postcost_hard_boundary(bound):
    assets = _assets([0.0, 0.0, 1.0], alpha=[0.2, 0.15, 0.0])
    assets["style"] = [-1.0, 1.0, 0.0]
    deviation = 0.1 if bound == "name" else 0.05
    config = optimizer.OptimizerConfig(
        active_name_limit=0.1,
        tracking_error_limit=1.0
        if bound == "name"
        else float(np.sqrt(0.18) * deviation),
        target_turnover_limit=1.0,
        style_active_limit=1.0,
    )
    result = _solve(assets, config=config)
    fee = 1.0 / 11.0
    expected = np.array([0.5 + deviation, 0.5 - deviation, 0.0])
    assert result.weights.to_numpy() == pytest.approx(
        [expected[0] / 1.1, expected[1] / 1.1, fee], abs=1e-8
    )
    net = (result.weights.to_numpy() - [0.0, 0.0, fee]) / (1.0 - fee)
    assert net == pytest.approx(expected, abs=1e-8)
    assert result.predicted_tracking_error == pytest.approx(
        np.sqrt(0.18) * deviation, abs=1e-8
    )


def test_frozen_quantity_does_not_shrink_with_nav_after_an_independent_purchase():
    assets = _assets([0.4, 0.0, 0.6], alpha=[0.0, 0.2, 0.0])
    assets.loc[0, "adv"] = np.nan
    result = _solve(assets)
    bought = 0.6 / 1.1
    fee = bought / 10.0
    assert result.weights.to_numpy() == pytest.approx([0.4, bought, fee], abs=1e-9)
    assert result.weights.iloc[0] * 100.0 == pytest.approx(40.0, abs=1e-10)
    assert result.weights.iloc[0] / (1.0 - fee) > 0.4
    assert result.target_turnover == pytest.approx(bought, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.09, 0.1, 0.11])
def test_exact_zero_trade_fee_threshold(alpha):
    result = _solve(_assets([0.0, 1.0], alpha=[alpha, 0.0]))
    expected_stock = 0.0 if alpha <= 0.1 else 1.0 / 1.1
    assert result.weights.iloc[0] == pytest.approx(expected_stock, abs=1e-9)
    assert result.expected_transaction_cost == pytest.approx(
        10.0 * expected_stock, abs=1e-8
    )


def _capture_fee_constraints(monkeypatch):
    assets = _assets([0.3, 0.3, 0.4], alpha=[0.03, -0.01, 0.0])
    assets["style"] = [-1.0, 2.0, 0.0]
    captured = {}

    def solve(objective, x0, **kwargs):
        captured.update(kwargs, objective=objective)
        return SimpleNamespace(success=True, x=x0)

    def cost(buy, sell):
        buy_weight, sell_weight = buy / 100.0, sell / 100.0
        return (
            float(
                100.0
                * (
                    0.01 * buy_weight[:-1].sum()
                    + 0.03 * sell_weight[:-1].sum()
                    + 0.02 * (buy_weight[:-1] ** 2 + sell_weight[:-1] ** 2).sum()
                )
            ),
            np.r_[0.01 + 0.04 * buy_weight[:-1], 0.0],
            np.r_[0.03 + 0.04 * sell_weight[:-1], 0.0],
        )

    monkeypatch.setattr(optimizer, "minimize", solve)
    _solve(assets, cost=cost)
    return captured, assets


@pytest.mark.parametrize("unchanged", [False, True])
def test_actual_fee_constraint_jacobians_include_central_zero_trade_subgradient(
    monkeypatch, unchanged
):
    captured, assets = _capture_fee_constraints(monkeypatch)
    current = assets.current_weight.to_numpy()
    weights = current if unchanged else np.array([0.34, 0.27, 0.39])
    point = np.r_[weights, np.abs(weights - current) + 0.05]
    step = 1e-6
    for constraint in captured["constraints"]:
        numeric = np.column_stack(
            [
                (
                    np.atleast_1d(constraint["fun"](point + step * direction))
                    - np.atleast_1d(constraint["fun"](point - step * direction))
                )
                / (2.0 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert np.atleast_2d(constraint["jac"](point)) == pytest.approx(
            numeric, abs=1e-8, rel=1e-6
        )


def test_fee_feasibility_ignores_inflated_objective_epigraph(monkeypatch):
    captured, assets = _capture_fee_constraints(monkeypatch)
    weights = np.array([0.34, 0.27, 0.39])
    delta = weights - assets.current_weight.to_numpy()
    exact = np.r_[weights, np.abs(delta)]
    inflated = exact + np.r_[np.zeros(3), np.full(3, 0.1)]
    fee = 0.01 * 0.04 + 0.03 * 0.03 + 0.02 * (0.04**2 + 0.03**2)
    # Objective auxiliaries can be slack, but risk/name/cash feasibility cannot
    # claim their fictitious round-trip fees as an economic NAV reduction.
    for position in (1, 6, 7):
        constraint = captured["constraints"][position]
        assert constraint["fun"](exact) == pytest.approx(
            constraint["fun"](inflated), abs=1e-15
        )
        assert constraint["jac"](exact) == pytest.approx(
            constraint["jac"](inflated), abs=1e-15
        )
    assert captured["constraints"][7]["fun"](exact) == pytest.approx(weights[-1] - fee)
    value, gradient = captured["objective"](exact)
    benchmark = assets.benchmark_weight.to_numpy()
    assert value == pytest.approx(
        -assets.expected_alpha.to_numpy() @ (weights - benchmark) + fee
    )
    assert captured["objective"](inflated)[0] > value
    assert gradient.shape == exact.shape
    active_numerator = weights - (1.0 - fee) * benchmark - [0.0, 0.0, fee]
    variance_numerator = 0.04 * (
        -active_numerator[0] + 2.0 * active_numerator[1]
    ) ** 2 + 0.01 * (active_numerator[0] ** 2 + active_numerator[1] ** 2)
    assert captured["constraints"][1]["fun"](exact) == pytest.approx(
        (1.0 - fee) ** 2 - variance_numerator
    )
    # With benchmark .5 and name allowance 1, the admitted net stock band is [0,1].
    assert captured["constraints"][6]["fun"](exact) == pytest.approx(
        [weights[0], weights[1], 1.0 - fee - weights[0], 1.0 - fee - weights[1]]
    )


@pytest.mark.parametrize("kind", ["absent", "none", "two", "nonzero_benchmark"])
def test_explicit_cash_contract_rejects_invalid_cash_universe(kind):
    assets = _assets([0.5, 0.5])
    if kind == "absent":
        assets = assets.drop(columns="is_cash")
        error = "optimizer_cash_column_missing"
    else:
        error = "optimizer_cash_asset_invalid"
        if kind == "none":
            assets["is_cash"] = False
        elif kind == "two":
            assets["is_cash"] = True
        else:
            assets["benchmark_weight"] = [0.9, 0.1]
    with pytest.raises(ValueError, match=f"^{error}$"):
        _solve(assets)


def test_smooth_objective_gradient_matches_its_asymmetric_trade_cost(monkeypatch):
    captured, assets = _capture_fee_constraints(monkeypatch)
    for point in (
        np.r_[assets.current_weight.to_numpy(), np.zeros(3)],
        np.array([0.34, 0.27, 0.39, 0.06, 0.05, 0.02]),
    ):
        step = 1e-6
        numerical = np.array(
            [
                (
                    captured["objective"](point + step * direction)[0]
                    - captured["objective"](point - step * direction)[0]
                )
                / (2 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert captured["objective"](point)[1] == pytest.approx(numerical, abs=1e-10)


@pytest.mark.parametrize("missing_risk", [False, True])
def test_frozen_pre_fee_underweight_can_match_benchmark_after_real_fees(missing_risk):
    assets = _assets([0.49, 0.01, 0.5])
    assets.loc[0, "adv"] = np.nan
    assets["financial"] = ~assets.is_cash
    if missing_risk:
        assets.loc[0, "specific_variance"] = np.nan
    # Frozen a=.49 becomes exactly .5 of post-fee NAV when c=.02.
    # Buying b from .01 to .49 costs .48/24=.02 and consumes all cash.
    result = _solve(
        assets,
        cost=_linear_cost(1.0 / 24.0, 1.0 / 24.0),
        financial_column="financial",
        config=optimizer.OptimizerConfig(
            active_name_limit=0.005,
            tracking_error_limit=1.0,
            target_turnover_limit=1.0,
        ),
    )
    assert result.weights.to_numpy() == pytest.approx([0.49, 0.49, 0.02], abs=1e-10)
    assert result.expected_transaction_cost == pytest.approx(2.0, abs=1e-9)
    assert result.predicted_tracking_error == pytest.approx(0.0, abs=1e-10)


def test_fees_cannot_make_reserved_receivable_available_to_fixed_benchmark():
    assets = _assets([0.4, 0.4, 0.1])
    assets["financial"] = ~assets.is_cash
    config = optimizer.OptimizerConfig()
    # All stock weights fixed to a 100% post-fee benchmark cannot coexist
    # with an independent positive receivable, whose net-NAV weight only grows.
    with pytest.raises(ValueError, match="^optimizer_budget_bounds_infeasible$"):
        _solve(
            assets, config=config, receivable_weight=0.1, financial_column="financial"
        )


@pytest.mark.parametrize("missing", ["specific_variance", "style"])
@pytest.mark.parametrize("rate", [0.0004, 0.01])
def test_fee_normalized_fixed_missing_risk_has_zero_active_coefficient(missing, rate):
    assets = _assets([0.4, 0.4, 0.2], alpha=[0.0, 0.2, 0.0])
    assets.loc[0, missing] = np.nan
    # The fixed unknown-risk stock is exactly half of post-fee NAV. Buying
    # both stocks spends .2-c and costs c=rate*(.2-c), independently of alpha.
    fee = 0.2 * rate / (1.0 + rate)
    result = _solve(assets, cost=_linear_cost(rate, rate))
    assert result.weights.to_numpy() == pytest.approx(
        [0.5 * (1.0 - fee), 0.5 * (1.0 - fee), fee], abs=1e-12
    )
    assert result.predicted_tracking_error == pytest.approx(0.0, abs=1e-14)
    assert result.expected_transaction_cost == pytest.approx(100.0 * fee, abs=1e-12)


@pytest.mark.parametrize("missing", ["specific", "exposure"])
@pytest.mark.parametrize("ulps", [8, 9])
def test_missing_risk_roundoff_boundary_does_not_erase_known_coefficients(
    missing, ulps
):
    benchmark = np.array([0.5, 0.5])
    shift = ulps * np.spacing(0.5)
    weights = benchmark + [shift, -shift]
    exposure = np.array([[np.nan if missing == "exposure" else 1.0], [1.0]])
    specific = np.array([np.nan if missing == "specific" else 0.01, 0.01])
    active = optimizer._risk_active_weights(weights, benchmark, exposure, specific)
    raw = weights - benchmark
    assert active[1] == raw[1] != 0.0
    if ulps == 8:
        assert active[0] == 0.0
        factor, residual = optimizer._risk_variances(
            active, exposure, np.array([[0.04]]), specific
        )
        assert factor == pytest.approx(0.04 * raw[1] ** 2, rel=1e-14, abs=0.0)
        assert residual == pytest.approx(0.01 * raw[1] ** 2, rel=1e-14, abs=0.0)
    else:
        assert active[0] == raw[0] != 0.0
        with pytest.raises(ValueError, match="optimizer_risk_nonfinite"):
            optimizer._risk_variances(active, exposure, np.array([[0.04]]), specific)


def test_unknown_risk_at_zero_benchmark_has_no_absolute_weight_floor():
    weights = np.array([1e-300, 1.0])
    benchmark = np.array([0.0, 1.0])
    exposure = np.array([[np.nan], [1.0]])
    specific = np.array([0.01, 0.01])
    active = optimizer._risk_active_weights(weights, benchmark, exposure, specific)
    assert active[0] == 1e-300
    with pytest.raises(ValueError, match="optimizer_risk_nonfinite"):
        optimizer._risk_variances(active, exposure, np.array([[0.04]]), specific)


def test_extreme_nav_no_trade_does_not_price_unaffordable_hypothetical_trades():
    from quant_research_platform.markets.implementations.a_share.execution import (
        PreparedTradeCost,
    )

    assets = _assets([0.1, 0.9, 0.0], alpha=[0.0, 0.0, 0.0], benchmark=[0.1, 0.9, 0.0])
    owner = PreparedTradeCost(
        (0.0, 0.0, 0.0),
        (False, False, False),
        np.frexp(30000.0),
        np.frexp(0.0),
        "synthetic",
    )

    def cost(buy, sell):
        adv = np.full(2, 1e308)
        buy_cost, buy_gradient = owner.totals_and_gradients(buy[:-1], adv)
        sell_cost, sell_gradient = owner.totals_and_gradients(sell[:-1], adv)
        return (
            float(buy_cost.sum() + sell_cost.sum()),
            np.r_[buy_gradient, 0.0],
            np.r_[sell_gradient, 0.0],
        )

    result = _solve(assets, nav=1e308, cost=cost)
    assert result.weights.to_numpy() == pytest.approx([0.1, 0.9, 0.0], abs=1e-14)
    assert result.expected_transaction_cost == 0.0
    assert result.target_turnover == 0.0
    assert result.predicted_tracking_error == 0.0


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("ulps", [8, 9])
def test_frozen_missing_risk_admission_uses_the_same_relative_roundoff(direction, ulps):
    shift = direction * ulps * np.spacing(0.5)
    assets = _assets([0.5 + shift, 0.5 - shift, 0.0], alpha=[0.0, 0.0, 0.0])
    assets.loc[0, "adv"] = np.nan
    assets.loc[0, "specific_variance"] = np.nan
    if ulps == 9:
        reason = (
            "optimizer_held_risk_unavailable"
            if direction > 0
            else "optimizer_frozen_constraints_infeasible"
        )
        with pytest.raises(ValueError, match=f"^{reason}$"):
            _solve(assets, cost=_linear_cost(0.0, 0.0))
    else:
        result = _solve(assets, cost=_linear_cost(0.0, 0.0))
        assert result.weights.iloc[0] == assets.current_weight.iloc[0]
        assert result.expected_transaction_cost == 0.0
        assert result.predicted_tracking_error == pytest.approx(
            np.sqrt(0.05) * abs(shift), rel=1e-12, abs=0.0
        )


@pytest.mark.parametrize("error_type", [ValueError, OverflowError, FloatingPointError])
def test_actual_signed_trade_cost_failures_propagate_without_fallback(
    monkeypatch, error_type
):
    assets = _assets([0.5, 0.5, 0.0])
    calls = []

    def cost(buy, sell):
        calls.append((buy.copy(), sell.copy()))
        if len(calls) == 1:
            # Only the preliminary hypothetical bound is representable.
            return 0.0, np.zeros_like(buy), np.zeros_like(sell)
        raise error_type("actual_cost_failure")

    weights = np.array([0.51, 0.49, 0.0])
    monkeypatch.setattr(
        optimizer,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.r_[weights, np.abs(weights - assets.current_weight)]
        ),
    )
    with pytest.raises(error_type, match="^actual_cost_failure$"):
        _solve(assets, cost=cost)
    assert len(calls) == 2
    assert calls[-1][0] == pytest.approx([1.0, 0.0, 0.0])
    assert calls[-1][1] == pytest.approx([0.0, 1.0, 0.0])


def test_hypothetical_fee_domain_error_is_not_replaced_with_a_cash_bound():
    def cost(buy, sell):
        raise ValueError("cost_owner_domain_error")

    with pytest.raises(ValueError, match="^cost_owner_domain_error$"):
        _solve(_assets([0.5, 0.5, 0.0]), cost=cost)
