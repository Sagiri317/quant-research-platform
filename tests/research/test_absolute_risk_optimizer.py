"""Hand-derived targets for absolute risk, fees, and partial sellable holdings."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios import optimizer


def _assets(current, alpha):
    return pd.DataFrame(
        {
            "asset_id": [f"stock_{i}" for i in range(len(current) - 1)] + ["cash"],
            "current_weight": current,
            "expected_alpha": alpha,
            "is_cash": [False] * (len(current) - 1) + [True],
        }
    )


def _cost(rate=0.0):
    def cost(buy, sell):
        rates = np.r_[np.full(len(buy) - 1, rate), 0.0]
        return float((buy + sell) @ rates), rates, rates

    return cost


def _solve(assets, *, config=None, cost=None, **kwargs):
    return optimizer.optimize_portfolio(
        assets,
        nav=100.0,
        cash_column="is_cash",
        expected_transaction_cost=cost or _cost(),
        config=config or optimizer.AbsoluteRiskConfig(),
        **kwargs,
    )


def test_absolute_defaults_are_unset_and_require_neither_benchmark_nor_daily_adv():
    config = optimizer.AbsoluteRiskConfig()
    assert all(value is None for key, value in asdict(config).items() if "limit" in key)
    assets = _assets([0.0, 1.0], [0.02, 0.0])
    result = _solve(assets)
    assert result.weights.tolist() == pytest.approx([1.0, 0.0], abs=1e-9)
    assert result.risk_mode == "absolute"
    assert result.predicted_tracking_error is None
    assert result.predicted_total_volatility is None
    assert result.factor_variance is None
    assert result.specific_variance is None
    assert result.maximum_adv_participation is None
    assert set(assets) == {"asset_id", "current_weight", "expected_alpha", "is_cash"}


@pytest.mark.parametrize("alpha, expected", [(0.09, 0.0), (0.1, 0.0), (0.11, 1 / 1.1)])
def test_absolute_alpha_uses_decision_cost_threshold_without_annual_risk_penalty(
    alpha, expected
):
    result = _solve(_assets([0.0, 1.0], [alpha, 0.0]), cost=_cost(0.1))
    assert result.weights.iloc[0] == pytest.approx(expected, abs=1e-9)
    assert result.expected_transaction_cost == pytest.approx(10 * expected, abs=1e-8)
    assert result.weights.iloc[1] - result.expected_transaction_cost / 100 >= -1e-9


@pytest.mark.parametrize("kind", ["name", "industry", "style", "volatility"])
def test_absolute_constraints_use_postfee_total_nav_with_hand_derived_boundary(kind):
    assets = _assets([0.0, 1.0], [0.2, 0.0])
    assets["industry"] = [1.0, 0.0]
    assets["style"] = [2.0, 0.0]
    assets["specific_variance"] = [0.01, 0.0]
    covariance = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.01]],
        index=["industry", "style"],
        columns=["industry", "style"],
    )
    limits = {
        "name": {"name_weight_limit": 0.3},
        "industry": {"industry_weight_limit": 0.3},
        "style": {"style_exposure_limit": 0.3},
        "volatility": {"total_volatility_limit": 0.1},
    }
    net_stock = {
        "name": 0.3,
        "industry": 0.3,
        "style": 0.15,
        "volatility": 0.1 / np.sqrt(0.05),
    }[kind]
    # Post-fee stock = x/(1 - .1*x), so x = target/(1 + .1*target).
    bought = net_stock / (1.0 + 0.1 * net_stock)
    result = _solve(
        assets,
        config=optimizer.AbsoluteRiskConfig(**limits[kind]),
        cost=_cost(0.1),
        factor_covariance=covariance,
        industry_columns=("industry",),
        style_columns=("style",),
    )
    assert result.weights.tolist() == pytest.approx([bought, 1 - bought], abs=2e-8)
    assert result.expected_transaction_cost == pytest.approx(10 * bought, abs=2e-7)
    assert result.factor_variance == pytest.approx(0.04 * net_stock**2, abs=1e-9)
    assert result.specific_variance == pytest.approx(0.01 * net_stock**2, abs=1e-9)
    assert result.predicted_total_volatility == pytest.approx(
        np.sqrt(0.05) * net_stock, abs=1e-8
    )
    assert result.predicted_tracking_error is None


@pytest.mark.parametrize(
    "case, expected",
    [
        ("partial", [0.4, 0.6, 0.0]),
        ("capacity", [0.5, 0.5, 0.0]),
        ("no_exit", [0.6, 0.4, 0.0]),
        ("missing_alpha", [0.6, 0.4, 0.0]),
    ],
)
def test_absolute_target_intersects_partial_sellability_capacity_and_signal_freeze(
    case, expected
):
    assets = _assets([0.6, 0.4, 0.0], [-0.01, 0.02, 0.0])
    assets["sellable_weight"] = [0.2, 0.4, 0.0]
    if case == "capacity":
        assets["trade_capacity_weight"] = [0.1, 0.1, 0.0]
    elif case == "no_exit":
        assets["can_exit"] = [False, True, True]
    elif case == "missing_alpha":
        assets.loc[0, "expected_alpha"] = np.nan
    result = _solve(assets)
    assert result.weights.tolist() == pytest.approx(expected, abs=1e-9)


def test_missing_alpha_freezes_physical_holding_while_other_purchase_pays_real_fee():
    assets = _assets([0.4, 0.0, 0.6], [np.nan, 0.2, 0.0])
    assets["sellable_weight"] = [0.4, 0.0, 0.0]
    result = _solve(assets, cost=_cost(0.1))
    assert result.weights.tolist() == pytest.approx(
        [0.4, 0.6 / 1.1, 0.6 / 11], abs=1e-9
    )
    assert np.isnan(assets.loc[0, "expected_alpha"])
    # Actual stock shares stayed fixed even though fee-reduced NAV raises weight.
    assert result.weights.iloc[0] / (1 - result.expected_transaction_cost / 100) > 0.4


def test_missing_risk_blocks_only_dependent_targets_and_does_not_erase_raw_alpha():
    assets = _assets([0.0, 0.0, 1.0], [0.2, 0.1, 0.0])
    assets["style"] = [1.0, 0.0, 0.0]
    assets["specific_variance"] = [np.nan, 0.01, 0.0]
    kwargs = dict(
        factor_covariance=pd.DataFrame([[0.04]], index=["style"], columns=["style"]),
        style_columns=("style",),
    )
    config = optimizer.AbsoluteRiskConfig(total_volatility_limit=0.2)
    result = _solve(assets, config=config, **kwargs)
    assert result.weights.tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert assets.expected_alpha.tolist() == [0.2, 0.1, 0.0]
    assets["current_weight"] = [0.4, 0.0, 0.6]
    with pytest.raises(ValueError, match="optimizer_held_risk_unavailable"):
        _solve(assets, config=config, **kwargs)
    unconstrained = _solve(assets, **kwargs)
    assert unconstrained.weights.iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert unconstrained.predicted_total_volatility is None
    with pytest.raises(ValueError, match="optimizer_covariance_unavailable"):
        _solve(assets, config=config)


def test_absolute_eligibility_prevents_new_buys_without_inventing_liquidation():
    assets = _assets([0.2, 0.0, 0.8], [0.2, 0.1, 0.0])
    assets["eligible"] = [False, True, False]
    result = _solve(assets, eligibility_column="eligible")
    assert result.weights.tolist() == pytest.approx([0.2, 0.8, 0.0], abs=1e-9)


def test_relative_mode_keeps_benchmark_bands_and_relative_risk():
    assets = _assets([0.0, 0.0, 1.0], [0.2, 0.15, 0.0])
    assets["benchmark_weight"] = [0.5, 0.5, 0.0]
    assets["specific_variance"] = [0.01, 0.01, 0.0]
    assets["style"] = [-1.0, 1.0, 0.0]
    assets["adv"] = [1e12, 1e12, np.inf]
    result = _solve(
        assets,
        config=optimizer.OptimizerConfig(
            active_name_limit=0.1,
            tracking_error_limit=1.0,
            target_turnover_limit=1.0,
            style_active_limit=1.0,
        ),
        cost=_cost(0.1),
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        factor_covariance=pd.DataFrame([[0.04]], index=["style"], columns=["style"]),
        style_columns=("style",),
    )
    assert result.weights.tolist() == pytest.approx(
        [0.6 / 1.1, 0.4 / 1.1, 1 / 11], abs=1e-8
    )
    assert result.predicted_tracking_error == pytest.approx(np.sqrt(0.0018), abs=1e-8)
    assert result.predicted_total_volatility is None
    assert result.risk_mode == "benchmark_relative"


def test_absolute_fee_and_risk_constraint_gradients_match_central_differences(
    monkeypatch,
):
    assets = _assets([0.2, 0.2, 0.6], [0.03, 0.02, 0.0])
    assets["industry"] = [1.0, 0.0, 0.0]
    assets["style"] = [-1.0, 2.0, 0.0]
    assets["specific_variance"] = [0.01, 0.02, 0.0]
    captured = {}

    def solve(fun, x0, **kwargs):
        captured.update(kwargs, objective=fun)
        return SimpleNamespace(success=True, x=x0)

    def cost(buy, sell):
        buy, sell = buy / 100, sell / 100
        return (
            float(
                100
                * (
                    0.01 * buy[:-1].sum()
                    + 0.03 * sell[:-1].sum()
                    + 0.02 * (buy[:-1] ** 2 + sell[:-1] ** 2).sum()
                )
            ),
            np.r_[0.01 + 0.04 * buy[:-1], 0.0],
            np.r_[0.03 + 0.04 * sell[:-1], 0.0],
        )

    monkeypatch.setattr(optimizer, "minimize", solve)
    _solve(
        assets,
        cost=cost,
        config=optimizer.AbsoluteRiskConfig(
            name_weight_limit=0.8,
            industry_weight_limit=0.8,
            style_exposure_limit=1.0,
            total_volatility_limit=1.0,
            target_turnover_limit=1.0,
        ),
        factor_covariance=pd.DataFrame(
            [[0.02, 0.003], [0.003, 0.04]],
            index=["industry", "style"],
            columns=["industry", "style"],
        ),
        industry_columns=("industry",),
        style_columns=("style",),
    )
    point = np.array([0.25, 0.18, 0.57, 0.08, 0.06, 0.1])
    for constraint in captured["constraints"]:
        numeric = np.column_stack(
            [
                (
                    np.atleast_1d(constraint["fun"](point + 1e-6 * direction))
                    - np.atleast_1d(constraint["fun"](point - 1e-6 * direction))
                )
                / 2e-6
                for direction in np.eye(len(point))
            ]
        )
        assert np.atleast_2d(constraint["jac"](point)) == pytest.approx(
            numeric, abs=1e-8, rel=1e-6
        )


@pytest.mark.parametrize("limit", [-0.1, np.nan, True, "0.3", 1.1])
def test_absolute_name_limit_rejects_invalid_units(limit):
    with pytest.raises(ValueError, match="optimizer_config_invalid"):
        optimizer.AbsoluteRiskConfig(name_weight_limit=limit)
