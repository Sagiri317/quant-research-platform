"""Hand-derived economic and differential oracles for fee-funded targets."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios import optimizer
from tests.helpers.optimizer import (
    ECONOMIC_UTILITY_ATOL,
    capture_solver_problem,
    constraint_jacobian,
    constraint_value,
)


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
            adv=[1e12] * count + [np.inf],
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


@pytest.mark.parametrize("nav", [100.0, 1e6, 1e9])
@pytest.mark.parametrize("receivable", [0.0, 0.2])
def test_purchase_reserves_exact_fee_and_keeps_receivable_in_full_nav(nav, receivable):
    # A buy of x costs x/10, so x + x/10 = the spendable budget.
    budget = 1.0 - receivable
    result = _solve(_assets([0.0, budget]), nav=nav, receivable_weight=receivable)
    optimal_stock = budget / 1.1
    stock, cash = result.weights.to_numpy()
    fee = result.expected_transaction_cost / nav
    assert fee == pytest.approx(0.1 * stock, rel=1e-12, abs=1e-15)
    assert -1e-14 <= 0.1 * (optimal_stock - stock) <= ECONOMIC_UTILITY_ATOL
    assert result.weights.sum() == pytest.approx(budget, abs=1e-12)
    # Check conservation in pre-fee-NAV units, including cost and receivable.
    cash_after = cash - fee
    assert -1e-12 <= cash_after <= 11 * ECONOMIC_UTILITY_ATOL + 1e-12
    assert stock + cash_after + receivable + fee == pytest.approx(1.0, abs=1e-12)
    net_stock = stock / (1.0 - fee)
    assert result.factor_variance == pytest.approx(
        0.04 * (net_stock - 1.0) ** 2, abs=1e-14
    )
    assert result.specific_variance == pytest.approx(
        0.01 * (net_stock - 1.0) ** 2, abs=1e-14
    )
    assert result.target_turnover == pytest.approx(stock, abs=1e-12)


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
    fee = result.expected_transaction_cost / 100.0
    weights = result.weights.to_numpy()
    assert fee == pytest.approx(0.1 * weights[:2].sum(), abs=1e-14)
    net = (result.weights.to_numpy() - [0.0, 0.0, fee]) / (1.0 - fee)
    assert net.sum() == pytest.approx(1.0, abs=1e-12)
    assert net[-1] >= -1e-12
    assert np.abs(net[:2] - 0.5).max() <= config.active_name_limit + 1e-12
    half_difference = (net[0] - net[1]) / 2
    tracking_error = np.sqrt(0.18 * half_difference**2 + 0.005 * net[-1] ** 2)
    assert result.predicted_tracking_error == pytest.approx(
        tracking_error, rel=1e-12, abs=1e-15
    )
    assert tracking_error <= config.tracking_error_limit + 1e-12
    optimal_utility = (0.075 + 0.05 * deviation) / 1.1
    actual_utility = 0.1 * weights[0] + 0.05 * weights[1]
    assert -1e-14 <= optimal_utility - actual_utility <= ECONOMIC_UTILITY_ATOL


def test_frozen_quantity_does_not_shrink_with_nav_after_an_independent_purchase():
    assets = _assets([0.4, 0.0, 0.6], alpha=[0.0, 0.2, 0.0])
    assets.loc[0, "adv"] = np.nan
    result = _solve(assets)
    bought = result.weights.iloc[1]
    fee = result.expected_transaction_cost / 100.0
    assert fee == pytest.approx(bought / 10, abs=1e-14)
    assert -1e-14 <= 0.1 * (0.6 / 1.1 - bought) <= ECONOMIC_UTILITY_ATOL
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert result.weights.iloc[-1] >= fee - 1e-12
    assert result.weights.iloc[0] * 100.0 == pytest.approx(40.0, abs=1e-10)
    assert result.weights.iloc[0] / (1.0 - fee) > 0.4
    assert result.target_turnover == pytest.approx(bought, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.09, 0.1, 0.11])
def test_exact_zero_trade_fee_threshold(alpha):
    result = _solve(_assets([0.0, 1.0], alpha=[alpha, 0.0]))
    stock = result.weights.iloc[0]
    if alpha <= 0.1:
        assert (
            stock == result.expected_transaction_cost == result.target_turnover == 0.0
        )
        return
    assert -1e-14 <= (alpha - 0.1) * (1.0 / 1.1 - stock) <= ECONOMIC_UTILITY_ATOL
    assert result.expected_transaction_cost == pytest.approx(
        10.0 * stock, rel=1e-12, abs=1e-14
    )
    assert result.target_turnover == pytest.approx(stock, abs=1e-12)


def _capture_fee_constraints(monkeypatch):
    assets = _assets([0.3, 0.3, 0.4], alpha=[0.03, -0.01, 0.0])
    assets["style"] = [-1.0, 2.0, 0.0]

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

    captured = capture_solver_problem(monkeypatch, lambda: _solve(assets, cost=cost))
    return captured, assets


def _fee_constraints(captured, point):
    """Identify the actual-fee equality, risk inequality and cash reserve."""
    nonlinear = [
        c for c in captured["constraints"] if not isinstance(c, LinearConstraint)
    ]
    fee = next(c for c in nonlinear if np.array_equal(c.lb, c.ub))
    risk = next(c for c in nonlinear if np.isposinf(c.ub).all())
    cash = next(
        c
        for c in captured["constraints"]
        if isinstance(c, LinearConstraint)
        and c.A.shape[0] == 1
        and np.array_equal(constraint_jacobian(c, point)[0], [0, 0, 1, 0, 0, 0, -1])
    )
    return {"fee": fee, "cash": cash, "risk": risk}


@pytest.mark.parametrize("unchanged", [False, True])
def test_actual_fee_constraint_jacobians_include_central_zero_trade_subgradient(
    monkeypatch, unchanged
):
    captured, assets = _capture_fee_constraints(monkeypatch)
    current = assets.current_weight.to_numpy()
    weights = current if unchanged else np.array([0.34, 0.27, 0.39])
    point = np.r_[weights, np.abs(weights - current) + 0.05, 0.001]
    assert point.shape == captured["initial"].shape
    step = 1e-6
    for constraint in captured["constraints"]:
        numeric = np.column_stack(
            [
                (
                    constraint_value(constraint, point + step * direction)
                    - constraint_value(constraint, point - step * direction)
                )
                / (2.0 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert constraint_jacobian(constraint, point) == pytest.approx(
            numeric, abs=1e-8, rel=1e-6
        )


def test_fee_feasibility_ignores_inflated_objective_epigraph(monkeypatch):
    captured, assets = _capture_fee_constraints(monkeypatch)
    weights = np.array([0.34, 0.27, 0.39])
    delta = weights - assets.current_weight.to_numpy()
    fee = 0.01 * 0.04 + 0.03 * 0.03 + 0.02 * (0.04**2 + 0.03**2)
    exact = np.r_[weights, np.abs(delta), fee]
    inflated = exact + np.r_[np.zeros(3), np.full(3, 0.1), 0.0]
    # Objective auxiliaries can be slack, but fee/risk/cash feasibility cannot
    # claim their fictitious round-trip fees as an economic NAV reduction.
    constraints = _fee_constraints(captured, exact)
    for constraint in constraints.values():
        assert constraint_value(constraint, exact) == pytest.approx(
            constraint_value(constraint, inflated), abs=1e-15
        )
        assert constraint_jacobian(constraint, exact) == pytest.approx(
            constraint_jacobian(constraint, inflated), abs=1e-15
        )
    assert constraint_value(constraints["cash"], exact) == pytest.approx(
        weights[-1] - fee
    )
    assert constraint_value(constraints["fee"], exact) == pytest.approx(0.0, abs=1e-15)
    fictitious_fee = exact.copy()
    fictitious_fee[-1] += 0.001
    assert constraint_value(constraints["fee"], fictitious_fee) == pytest.approx(0.001)
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
    assert constraint_value(constraints["risk"], exact) == pytest.approx(
        (1.0 - fee) ** 2 - variance_numerator
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
        np.r_[assets.current_weight.to_numpy(), np.zeros(3), 0.0],
        np.array([0.34, 0.27, 0.39, 0.06, 0.05, 0.02, 0.001]),
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
    # The unknown-risk stock stays at half of actual post-fee NAV; the
    # independent optimum invests the remaining cash in the positive-alpha stock.
    fee = 0.2 * rate / (1.0 + rate)
    result = _solve(assets, cost=_linear_cost(rate, rate))
    if missing == "style":
        # The additional stock-leg exposure equality fixes the entire allocation.
        assert result.weights.to_numpy() == pytest.approx(
            [0.5 * (1.0 - fee), 0.5 * (1.0 - fee), fee], abs=1e-12
        )
        assert result.predicted_tracking_error == pytest.approx(0.0, abs=1e-14)
        assert result.expected_transaction_cost == pytest.approx(100.0 * fee, abs=1e-12)
        return
    actual = result.weights.to_numpy()
    actual_fee = result.expected_transaction_cost / 100.0
    assert actual_fee == pytest.approx(rate * (actual[:2].sum() - 0.8), abs=1e-14)
    assert actual.sum() == pytest.approx(1.0, abs=1e-12)
    assert actual[-1] >= actual_fee - 1e-12
    net = actual[:2] / (1.0 - actual_fee)
    assert abs(net[0] - 0.5) <= 8 * np.spacing(0.5)
    known_coefficient = net[1] - 0.5
    assert result.factor_variance == pytest.approx(
        0.04 * known_coefficient**2, rel=1e-12, abs=0.0
    )
    assert result.specific_variance == pytest.approx(
        0.01 * known_coefficient**2, rel=1e-12, abs=0.0
    )
    assert result.predicted_tracking_error == pytest.approx(
        np.sqrt(0.05) * abs(known_coefficient), rel=1e-12, abs=0.0
    )
    utility_gap = 0.2 * (0.5 * (1.0 - fee) - actual[1]) - (fee - actual_fee)
    assert -1e-14 <= utility_gap <= ECONOMIC_UTILITY_ATOL


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
    weights = np.array([1e-8, 1.0])
    benchmark = np.array([0.0, 1.0])
    exposure = np.array([[np.nan], [1.0]])
    specific = np.array([0.01, 0.01])
    active = optimizer._risk_active_weights(weights, benchmark, exposure, specific)
    assert active[0] == 1e-8
    with pytest.raises(ValueError, match="optimizer_risk_nonfinite"):
        optimizer._risk_variances(active, exposure, np.array([[0.04]]), specific)


def test_high_cost_no_trade_keeps_self_financing_weights():
    from quant_research_platform.markets.implementations.a_share.execution import (
        PreparedTradeCost,
    )

    assets = _assets([0.1, 0.9, 0.0], alpha=[0.0, 0.0, 0.0], benchmark=[0.1, 0.9, 0.0])
    owner = PreparedTradeCost(
        (0.0, 0.0, 0.0),
        (False, False, False),
        30000.0,
        0.0,
        "synthetic",
    )

    def cost(buy, sell):
        adv = np.full(2, 1e8)
        buy_cost, buy_gradient = owner.totals_and_gradients(buy[:-1], adv)
        sell_cost, sell_gradient = owner.totals_and_gradients(sell[:-1], adv)
        return (
            float(buy_cost.sum() + sell_cost.sum()),
            np.r_[buy_gradient, 0.0],
            np.r_[sell_gradient, 0.0],
        )

    result = _solve(assets, nav=1e6, cost=cost)
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


def test_trade_cost_provider_error_propagates_without_replacement():
    def cost(buy, sell):
        raise ValueError("cost_owner_domain_error")

    with pytest.raises(ValueError, match="^cost_owner_domain_error$"):
        _solve(_assets([0.5, 0.5, 0.0]), cost=cost)


@pytest.mark.parametrize("weights", [[0.34, 0.27, 0.39], [0.2, 0.4, 0.4]])
def test_net_risk_gradient_matches_dense_chain_rule(monkeypatch, weights):
    captured, assets = _capture_fee_constraints(monkeypatch)
    weights = np.asarray(weights)
    delta = weights - assets.current_weight.to_numpy()
    buy, sell = np.maximum(delta[:2], 0.0), np.maximum(-delta[:2], 0.0)
    fee = 0.01 * buy.sum() + 0.03 * sell.sum() + 0.02 * (buy**2 + sell**2).sum()
    point = np.r_[weights, abs(delta), fee]
    constraint = _fee_constraints(captured, point)["risk"]
    fee_gradient = np.r_[
        np.where(delta[:2] > 0, 0.01 + 0.04 * buy, -0.03 - 0.04 * sell), 0.0
    ]
    benchmark = assets.benchmark_weight.to_numpy()
    cash_direction = np.array([0.0, 0.0, 1.0])
    active = weights - benchmark + (benchmark - cash_direction) * fee
    exposure = assets["style"].to_numpy()
    variance_gradient = (
        0.08 * exposure * (active @ exposure)
        + 2 * assets.specific_variance.to_numpy() * active
    )
    dense = np.eye(3) + np.outer(benchmark - cash_direction, fee_gradient)
    expected = -2 * (1 - fee) * fee_gradient - variance_gradient @ dense
    partial = constraint_jacobian(constraint, point)[0]
    np.testing.assert_array_equal(partial[3:6], np.zeros(3))
    # Eliminate the explicit fee variable with its independently derived gradient.
    np.testing.assert_allclose(
        partial[:3] + partial[-1] * fee_gradient, expected, rtol=2e-14, atol=1e-16
    )


@pytest.mark.parametrize("scale", [1e-6, 1e-4, 1.0])
def test_optimizer_rejects_material_asymmetry_in_covariance_units(scale):
    assets = _assets([0.5, 0.5, 0.0], alpha=[0.0, 0.0, 0.0])
    assets["specific_variance"] = 0.0
    assets["x"] = [1.0, 0.0, 0.0]
    assets["y"] = [0.0, 1.0, 0.0]
    covariance = scale * np.array([[1.0, 1.009], [1.0, 1.0]])
    # The old absolute allclose tolerance admitted negative quadratic risk.
    assert np.array([1.0, -1.0]) @ covariance @ np.array([1.0, -1.0]) < 0
    with pytest.raises(ValueError, match="optimizer_covariance_invalid"):
        optimizer.optimize_portfolio(
            assets,
            factor_covariance=pd.DataFrame(
                covariance, index=["x", "y"], columns=["x", "y"]
            ),
            style_columns=("x", "y"),
            cash_column="is_cash",
            expected_transaction_cost=_linear_cost(0.0, 0.0),
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        )
