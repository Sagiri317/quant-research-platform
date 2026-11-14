"""Independent checks for exact optimizer constraint derivatives."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint
from scipy.sparse import issparse

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios import optimizer
from tests.helpers.optimizer import (
    ECONOMIC_UTILITY_ATOL,
    capture_solver_problem,
    constraint_jacobian,
    constraint_value,
    with_cash_asset,
)


def _capture_constraints(monkeypatch, asymmetry=0.0, nonzero_cost=False):
    benchmark = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
    current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d", "cash"],
            "expected_alpha": [0.01, -0.02, 0.03, -0.01, 0.0],
            "current_weight": current,
            "benchmark_weight": benchmark,
            "specific_variance": [0.01, 0.02, 0.03, 0.04, 0.0],
            "adv": [1000.0, 2000.0, 3000.0, 4000.0, np.inf],
            "is_cash": [False, False, False, False, True],
            "industry": [1.0, 1.0, 0.0, 0.0, 0.0],
            "style": [-1.0, 0.0, 0.5, 2.0, 0.0],
        }
    )
    buy_rates = np.array([0.001, 0.002, 0.003, 0.004, 0.0])
    sell_rates = np.array([0.005, 0.006, 0.007, 0.008, 0.0])

    def expected_cost(buy, sell):
        assert buy.shape == sell.shape == current.shape
        if nonzero_cost:
            return float(buy @ buy_rates + sell @ sell_rates), buy_rates, sell_rates
        return 0.0, np.zeros_like(buy), np.zeros_like(sell)

    captured = capture_solver_problem(
        monkeypatch,
        lambda: optimizer.optimize_portfolio(
            assets,
            factor_covariance=pd.DataFrame(
                [[0.02, 0.003 + asymmetry], [0.003, 0.01]],
                index=["industry", "style"],
                columns=["industry", "style"],
            ),
            industry_columns=("industry",),
            style_columns=("style",),
            expected_transaction_cost=expected_cost,
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
            cash_column="is_cash",
            config=optimizer.OptimizerConfig(active_name_limit=0.2),
        ),
    )
    return captured, current


@pytest.mark.parametrize(
    "asymmetry,nonzero_cost", [(0.0, False), (2e-18, False), (0.0, True)]
)
def test_constraint_jacobians_match_independent_central_differences(
    monkeypatch, asymmetry, nonzero_cost
):
    captured, _ = _capture_constraints(monkeypatch, asymmetry, nonzero_cost)
    point = np.r_[
        [0.19, 0.21, 0.23, 0.25, 0.12, 0.01, 0.01, 0.03, 0.05, 0.08],
        [0.001] if nonzero_cost else [],
    ]
    assert point.shape == captured["initial"].shape
    step = 1e-6
    for constraint in captured["constraints"]:
        derivative = np.column_stack(
            [
                (
                    constraint_value(constraint, point + direction * step)
                    - constraint_value(constraint, point - direction * step)
                )
                / (2 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert constraint_jacobian(constraint, point) == pytest.approx(
            derivative, abs=1e-10, rel=1e-8
        )


def test_linear_operators_and_zero_cost_name_cash_bounds(monkeypatch):
    captured, current = _capture_constraints(monkeypatch)
    bounds = captured["bounds"]
    np.testing.assert_allclose(bounds.lb[:5], [0.05] * 4 + [0.0])
    np.testing.assert_allclose(bounds.ub[:5], [0.45] * 4 + [1.0])
    for constraint in captured["constraints"]:
        if isinstance(constraint, LinearConstraint):
            assert issparse(constraint.A)
            assert constraint.A.nnz <= 4 * len(current)


def test_trade_epigraph_is_exact_at_and_away_from_no_trade(monkeypatch):
    captured, current = _capture_constraints(monkeypatch)
    linear = [c for c in captured["constraints"] if isinstance(c, LinearConstraint)]
    trades = next(c for c in linear if c.A.shape[0] == 2 * len(current))
    turnover = next(
        c for c in linear if np.all(c.ub == 0.2) and np.isneginf(c.lb).all()
    )
    zero = np.r_[current, np.zeros(5)]
    assert constraint_jacobian(turnover, zero) == pytest.approx(
        np.r_[np.zeros(5), np.full(5, 0.5)][None, :]
    )
    expected_jac = np.block([[-np.eye(5), np.eye(5)], [np.eye(5), np.eye(5)]])
    assert constraint_jacobian(trades, zero) == pytest.approx(expected_jac)
    assert np.isneginf(captured["bounds"].lb[5:]).all()
    assert np.isposinf(captured["bounds"].ub[5:]).all()
    for delta in (np.zeros(5), np.array([0.01, -0.02, 0.03, -0.04, 0.02])):
        point = np.r_[current + delta, np.abs(delta)]
        assert np.minimum(
            (constraint_value(trades, point) - trades.lb)[:5],
            (constraint_value(trades, point) - trades.lb)[5:],
        ) == pytest.approx(np.zeros(5))
        assert turnover.ub - constraint_value(turnover, point) == pytest.approx(
            0.2 - 0.5 * np.abs(delta).sum()
        )
        assert constraint_jacobian(trades, point) == pytest.approx(expected_jac)
        # Every admitted auxiliary vector bounds absolute turnover from above;
        # understating a trade is rejected even at the no-trade kink.
        understated = point.copy()
        understated[5:] -= 0.001
        assert np.min(constraint_value(trades, understated) - trades.lb) < 0.0


def test_exposure_bands_are_affine_even_at_exact_neutrality(monkeypatch):
    captured, current = _capture_constraints(monkeypatch)
    cash_only = np.r_[np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.zeros(5)]
    point = np.r_[current, np.zeros(5)]
    bands = [
        c
        for c in captured["constraints"]
        if isinstance(c, LinearConstraint) and c.A.shape[0] == 2
    ]
    assert len(bands) == 2
    for constraint, limit in zip(bands, [0.02, 0.2], strict=True):
        jac = constraint_jacobian(constraint, cash_only)
        assert jac.shape == (2, 10)
        assert jac == pytest.approx(constraint_jacobian(constraint, point))
        assert jac.sum(axis=0) == pytest.approx([2 * limit] * 4 + [0.0] * 6)
        assert constraint_value(constraint, cash_only) == pytest.approx([0.0, 0.0])
        assert constraint_value(constraint, point) == pytest.approx(jac @ point)


def test_exact_exposure_bands_reach_independent_linear_optimum():
    from scipy.optimize import linprog

    alpha = np.array(
        [
            0.002496853370236516,
            0.024272558398402866,
            0.025005339808430904,
            -0.01140459343498049,
            0.006091753418979527,
            0.0047327477048703945,
        ]
    )
    exposure = np.column_stack(
        (
            np.repeat([1.0, 0.0], 3),
            np.repeat([0.0, 1.0], 3),
            [
                -0.2557045017824022,
                -0.5894028246931097,
                0.9329753853715171,
                -0.7413983324170841,
                0.8370285901199255,
                -1.0313701693208357,
            ],
        )
    )
    benchmark = np.full(6, 1.0 / 6)
    centered = exposure - exposure.T @ benchmark
    oracle = linprog(
        -alpha,
        A_eq=np.vstack((np.ones(6), centered.T)),
        b_eq=[1.0, 0.0, 0.0, 0.0],
        bounds=[(1 / 6 - 0.1, 1 / 6 + 0.1)] * 6,
        method="highs",
    )
    assert oracle.success
    frame = pd.DataFrame(
        {
            "asset_id": list("abcdef"),
            "expected_alpha": alpha,
            "current_weight": benchmark,
            "benchmark_weight": benchmark,
            "specific_variance": 0.0,
            "adv": 1e6,
            "industry_a": exposure[:, 0],
            "industry_b": exposure[:, 1],
            "style": exposure[:, 2],
        }
    )
    columns = ["industry_a", "industry_b", "style"]
    result = optimizer.optimize_portfolio(
        with_cash_asset(frame),
        cash_column="is_cash",
        factor_covariance=pd.DataFrame(0.0, index=columns, columns=columns),
        industry_columns=tuple(columns[:2]),
        style_columns=("style",),
        expected_transaction_cost=lambda buy, sell: (
            0.0,
            np.zeros_like(buy),
            np.zeros_like(sell),
        ),
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        config=optimizer.OptimizerConfig(
            active_name_limit=0.1,
            tracking_error_limit=1.0,
            target_turnover_limit=1.0,
            industry_active_limit=0.0,
            style_active_limit=0.0,
        ),
    )
    utility_gap = -oracle.fun - alpha @ result.weights.to_numpy()[:-1]
    assert -1e-12 <= utility_gap <= ECONOMIC_UTILITY_ATOL
    assert centered.T @ result.weights.to_numpy()[:-1] == pytest.approx(
        np.zeros(3), abs=1e-12
    )


@pytest.mark.parametrize(
    "perturbation",
    ["original", "native_trajectory"],
)
def test_native_turnover_boundary_converges_with_actual_cost_owner(perturbation):
    import json
    from pathlib import Path

    from quant_research_platform.markets.implementations.a_share.execution import (
        PreparedTradeCost,
    )

    # Preserve native trajectories that stalled with nonsmooth absolute turnover.
    case = json.loads(
        (
            Path(__file__).parent / "fixtures" / "optimizer_turnover_boundary.json"
        ).read_text()
    )
    assets = pd.DataFrame(case["assets"]["rows"], columns=case["assets"]["columns"])
    assert assets.loc[~assets.is_cash, "adv"].notna().all()
    assets.loc[assets.is_cash, "adv"] = np.inf
    nav = case["nav"]
    if perturbation == "native_trajectory":
        assets["current_weight"] = case["nearby_native_trajectory"]["current_weight"]
        nav = case["nearby_native_trajectory"]["nav"]
    current = assets.current_weight.to_numpy()
    adv = assets.adv.to_numpy()
    prepared = {}
    for record in case["prepared_costs"]:
        values = {field: tuple(record[field]) for field in ("rates", "face_value")}
        prepared[record["market"], record["side"]] = PreparedTradeCost(
            **values,
            slippage_bps=record["slippage_bps"],
            impact_bps=record["impact_bps"],
            source_identity=record["source_identity"],
        )
    markets = {
        name: np.asarray(mask, dtype=bool)
        for name, mask in case["market_masks"].items()
    }

    def expected_cost(buys, sells):
        total, gradients = 0.0, []
        for side, notionals in (("buy", buys), ("sell", sells)):
            gradient = np.zeros(len(notionals))
            for market, mask in markets.items():
                costs, gradient[mask] = prepared[market, side].totals_and_gradients(
                    notionals[mask], adv[mask]
                )
                total += float(costs.sum())
            gradients.append(gradient)
        return total, *gradients

    def actual_cost(weights):
        delta = weights - current
        return expected_cost(np.maximum(delta, 0) * nav, np.maximum(-delta, 0) * nav)[0]

    assumptions = ExecutionAssumptions(
        **(
            case["assumptions"]
            | {"adv_sensitivities": tuple(case["assumptions"]["adv_sensitivities"])}
        )
    )
    config = optimizer.OptimizerConfig(**case["config"])
    result = optimizer.optimize_portfolio(
        assets,
        factor_covariance=pd.DataFrame(
            case["factor_covariance"],
            index=case["factor_columns"],
            columns=case["factor_columns"],
        ),
        industry_columns=tuple(case["industry_columns"]),
        style_columns=tuple(case["style_columns"]),
        expected_transaction_cost=expected_cost,
        nav=nav,
        assumptions=assumptions,
        financial_column="is_financial",
        cash_column="is_cash",
        eligibility_column="eligible",
        config=config,
        receivable_weight=case["receivable_weight"],
    )
    weights = result.weights.to_numpy()
    delta = weights - current
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    actual_turnover = 0.5 * np.abs(delta).sum()
    assert actual_turnover <= config.target_turnover_limit + 1e-8
    assert result.target_turnover == pytest.approx(actual_turnover)
    assert np.max(np.abs(delta) * nav / adv) <= assumptions.adv_participation + 1e-8
    assert result.predicted_tracking_error <= config.tracking_error_limit + 1e-8
    assert (
        assets.expected_alpha.to_numpy() @ delta - actual_cost(weights) / nav > 0.00036
    )
    assert result.expected_transaction_cost == pytest.approx(actual_cost(weights))


@pytest.mark.parametrize("adv", [100.0, 10000.0])
def test_positive_capacity_preserves_feasible_no_trade(adv):
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "expected_alpha": [0.0, 0.0],
            "current_weight": [0.5, 0.5],
            "benchmark_weight": [0.5, 0.5],
            "specific_variance": [0.0, 0.0],
            "style": [-1.0, 1.0],
            "adv": [adv, adv],
        }
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = optimizer.optimize_portfolio(
            with_cash_asset(assets),
            cash_column="is_cash",
            factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
            style_columns=("style",),
            expected_transaction_cost=lambda buy, sell: (
                0.0,
                np.zeros_like(buy),
                np.zeros_like(sell),
            ),
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
            config=optimizer.OptimizerConfig(active_name_limit=0.1),
        )
    assert result.weights.tolist() == [0.5, 0.5, 0.0]
    assert result.maximum_adv_participation == 0.0


@pytest.mark.parametrize("nav, adv", [(1e6, 1e6), (1e8, 1e8), (1e8, 1e6)])
def test_capacity_matches_independent_fraction_oracle(nav, adv):
    from fractions import Fraction

    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "expected_alpha": [0.1, 0.0],
            "current_weight": [0.5, 0.5],
            "benchmark_weight": [0.5, 0.5],
            "specific_variance": [0.0, 0.0],
            "style": [0.0, 0.0],
            "adv": [adv, adv],
        }
    )
    result = optimizer.optimize_portfolio(
        with_cash_asset(assets),
        cash_column="is_cash",
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=lambda buy, sell: (
            0.0,
            np.zeros_like(buy),
            np.zeros_like(sell),
        ),
        nav=nav,
        assumptions=ExecutionAssumptions(nav, 0.0, 0.0, 0.05, (0.05,)),
        config=optimizer.OptimizerConfig(active_name_limit=0.1),
    )
    capacity = float(Fraction(0.05) * Fraction(adv) / Fraction(nav))
    assert result.weights.to_numpy() == pytest.approx(
        [0.5 + capacity, 0.5 - capacity, 0.0], abs=1e-10
    )


@pytest.mark.parametrize("fee_rate", [0.0, 0.01])
def test_fixed_benchmark_weights_reserve_actual_fee_cash(fee_rate):
    current = np.array([0.6, 0.4])
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "expected_alpha": [0.1, 0.0],
            "current_weight": current,
            "benchmark_weight": [0.5, 0.5],
            "specific_variance": [0.0, 0.0],
            "style": [0.0, 0.0],
            "adv": [1000.0, 1000.0],
            "financial": [True, True],
        }
    )
    result = optimizer.optimize_portfolio(
        with_cash_asset(assets),
        cash_column="is_cash",
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=lambda buy, sell: (
            float((buy[:-1] + sell[:-1]).sum()) * fee_rate,
            np.r_[np.full(len(buy) - 1, fee_rate), 0.0],
            np.r_[np.full(len(sell) - 1, fee_rate), 0.0],
        ),
        financial_column="financial",
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
    )
    # The stock rotation is 0.1 in each direction before common fee shrinkage.
    fee = fee_rate * 0.2
    assert result.weights.to_numpy() == pytest.approx(
        [0.5 * (1.0 - fee), 0.5 * (1.0 - fee), fee]
    )
    assert result.target_turnover == pytest.approx(0.1 + fee / 2.0)
    assert result.expected_transaction_cost == pytest.approx(fee * 100.0)
