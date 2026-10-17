"""Independent checks for exact optimizer constraint derivatives."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios import optimizer


def _capture_constraints(monkeypatch, asymmetry=0.0):
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
    captured = {}

    def solve(fun, x0, **kwargs):
        captured.update(kwargs)
        assert fun(np.r_[benchmark, np.abs(benchmark - current)]) == pytest.approx(0.0)
        return SimpleNamespace(
            success=True, x=np.r_[benchmark, np.abs(benchmark - current)]
        )

    def expected_cost(weights):
        assert weights.shape == current.shape
        return 0.0

    monkeypatch.setattr(optimizer, "minimize", solve)
    optimizer.optimize_portfolio(
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
    )
    return captured["constraints"], current, captured["bounds"]


@pytest.mark.parametrize("asymmetry", [0.0, 2e-8])
def test_constraint_jacobians_match_independent_central_differences(
    monkeypatch, asymmetry
):
    constraints, _, _ = _capture_constraints(monkeypatch, asymmetry)
    point = np.array([0.19, 0.21, 0.23, 0.25, 0.12, 0.01, 0.01, 0.03, 0.05, 0.08])
    step = 1e-6
    for constraint in constraints:
        derivative = np.column_stack(
            [
                (
                    np.atleast_1d(constraint["fun"](point + direction * step))
                    - np.atleast_1d(constraint["fun"](point - direction * step))
                )
                / (2 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert np.atleast_2d(constraint["jac"](point)) == pytest.approx(
            derivative, abs=1e-10, rel=1e-8
        )


def test_trade_epigraph_is_exact_at_and_away_from_no_trade(monkeypatch):
    constraints, current, bounds = _capture_constraints(monkeypatch)
    turnover, trades = constraints[2], constraints[5]
    zero = np.r_[current, np.zeros(5)]
    assert turnover["jac"](zero) == pytest.approx(np.r_[np.zeros(5), np.full(5, -0.5)])
    expected_jac = np.block([[-np.eye(5), np.eye(5)], [np.eye(5), np.eye(5)]])
    assert trades["jac"](zero) == pytest.approx(expected_jac)
    assert bounds[5:] == ((0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.0, 2.0), (0.0, np.inf))
    for delta in (np.zeros(5), np.array([0.01, -0.02, 0.03, -0.04, 0.02])):
        point = np.r_[current + delta, np.abs(delta)]
        assert np.minimum(
            trades["fun"](point)[:5], trades["fun"](point)[5:]
        ) == pytest.approx(np.zeros(5))
        assert turnover["fun"](point) == pytest.approx(0.2 - 0.5 * np.abs(delta).sum())
        assert trades["jac"](point) == pytest.approx(expected_jac)
        # Every admitted auxiliary vector bounds absolute turnover from above;
        # understating a trade is rejected even at the no-trade kink.
        understated = point.copy()
        understated[5:] -= 0.001
        assert np.min(trades["fun"](understated)) < 0.0


def test_exposure_bands_are_affine_even_at_exact_neutrality(monkeypatch):
    constraints, current, _ = _capture_constraints(monkeypatch)
    cash_only = np.r_[np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.zeros(5)]
    point = np.r_[current, np.zeros(5)]
    for constraint, limit in zip(constraints[3:5], [0.02, 0.2], strict=True):
        jac = constraint["jac"](cash_only)
        assert jac.shape == (2, 10)
        assert jac == pytest.approx(constraint["jac"](point))
        assert jac.sum(axis=0) == pytest.approx([2 * limit] * 4 + [0.0] * 6)
        assert constraint["fun"](cash_only) == pytest.approx([0.0, 0.0])
        assert constraint["fun"](point) == pytest.approx(jac @ point)


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
        frame,
        factor_covariance=pd.DataFrame(0.0, index=columns, columns=columns),
        industry_columns=tuple(columns[:2]),
        style_columns=("style",),
        expected_transaction_cost=lambda weights: 0.0,
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
    assert alpha @ result.weights.to_numpy() == pytest.approx(-oracle.fun, abs=1e-12)
    assert centered.T @ result.weights.to_numpy() == pytest.approx(
        np.zeros(3), abs=1e-12
    )


@pytest.mark.parametrize(
    "perturbation",
    ["original", "native_trajectory", "nav_up", "nav_down", "weight_up", "weight_down"],
)
def test_native_turnover_boundary_converges_with_actual_cost_owner(perturbation):
    import json
    from pathlib import Path

    from quant_research_platform.markets.implementations.a_share.execution import (
        PreparedTradeCost,
    )

    # Keep exact captured floats from the native synthetic workflow, including
    # the next failed trajectory. Both reach 1000 iterations with abs turnover.
    # Nearby input perturbations exercise robustness beyond one solver path.
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
    elif perturbation.startswith("nav_"):
        nav = float(np.nextafter(nav, np.inf if perturbation == "nav_up" else -np.inf))
    elif perturbation.startswith("weight_"):
        shift = 1e-12 if perturbation == "weight_up" else -1e-12
        assets.loc[0, "current_weight"] += shift
        assets.loc[1, "current_weight"] -= shift
    current = assets.current_weight.to_numpy()
    adv = assets.adv.to_numpy()
    prepared = {}
    for record in case["prepared_costs"]:
        values = {
            field: tuple(record[field])
            for field in ("rates", "face_value", "slippage", "impact")
        }
        prepared[record["market"], record["side"]] = PreparedTradeCost(
            **values, source_identity=record["source_identity"]
        )
    markets = {
        name: np.asarray(mask, dtype=bool)
        for name, mask in case["market_masks"].items()
    }

    def expected_cost(weights):
        delta = weights - current
        notional = np.abs(delta) * nav
        totals = np.zeros(len(delta))
        for (market, side), evaluator in prepared.items():
            active = markets[market] & (delta > 0 if side == "buy" else delta < 0)
            if active.any():
                totals[active] = evaluator.totals(notional[active], adv[active])
        return sum(map(float, totals), 0.0)

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
    assert 0.5 * np.abs(delta).sum() == pytest.approx(0.2, abs=1e-8)
    assert np.max(np.abs(delta) * nav / adv) <= assumptions.adv_participation + 1e-8
    assert result.predicted_tracking_error <= config.tracking_error_limit + 1e-8
    assert (
        assets.expected_alpha.to_numpy() @ delta - expected_cost(weights) / nav
        > 0.00036
    )
    assert result.expected_transaction_cost == pytest.approx(expected_cost(weights))


@pytest.mark.parametrize("adv", [1e-320, np.nextafter(0.0, 1.0)])
def test_tiny_positive_capacity_preserves_feasible_no_trade(adv):
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
            assets,
            factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
            style_columns=("style",),
            expected_transaction_cost=lambda weights: 0.0,
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
            config=optimizer.OptimizerConfig(active_name_limit=0.1),
        )
    assert result.weights.tolist() == [0.5, 0.5]
    assert result.maximum_adv_participation == 0.0


@pytest.mark.parametrize(
    "nav, adv", [(1e-320, 1e-320), (5e-324, 5e-324), (1e-320, 5e-324)]
)
def test_small_money_capacity_does_not_round_before_dividing(nav, adv):
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
        assets,
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=lambda weights: 0.0,
        nav=nav,
        assumptions=ExecutionAssumptions(nav, 0.0, 0.0, 0.05, (0.05,)),
        config=optimizer.OptimizerConfig(active_name_limit=0.1),
    )
    capacity = float(Fraction(0.05) * Fraction(adv) / Fraction(nav))
    assert result.weights.to_numpy() == pytest.approx(
        [0.5 + capacity, 0.5 - capacity], abs=1e-10
    )


def test_fixed_weights_fix_their_actual_trade_auxiliaries():
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
        assets,
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=lambda weights: float(
            np.abs(weights - current).sum()
        ),
        financial_column="financial",
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
    )
    assert result.weights.to_numpy() == pytest.approx([0.5, 0.5])
    assert result.target_turnover == pytest.approx(0.1)
    assert result.expected_transaction_cost == pytest.approx(0.2)
