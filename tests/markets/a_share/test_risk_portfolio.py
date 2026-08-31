from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from quant_research_platform.research.portfolios.a_share import (
    FeeSchedule,
    PortfolioOptimizationConfig,
    optimize_a_share_portfolio,
    run_a_share_portfolio_backtest,
)


def test_factor_form_optimizer_enforces_tradability_risk_cost_and_adv() -> None:
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            "alpha": [0.10, 0.02, -1.00],
            "current_weight": [0.5, 0.4, 0.1],
            "benchmark_weight": [0.5, 0.5, 0.0],
            "specific_variance": [0.01, 0.01, 0.02],
            "adv": [1_000_000.0, 1_000_000.0, 100_000.0],
            "eligible": [True, True, False],
            "can_enter": [True, True, True],
            "can_exit": [True, False, True],
            "beta": [1.0, 1.0, 1.2],
            "size": [0.2, -0.2, 1.0],
        }
    )
    covariance = pd.DataFrame(
        [[0.02, 0.0], [0.0, 0.01]],
        index=["beta", "size"],
        columns=["beta", "size"],
    )
    config = PortfolioOptimizationConfig(
        rebalance_frequency="1w",
        portfolio_notional=1_000_000.0,
        max_weight=0.7,
        tracking_error_limit=0.2,
        turnover_limit=0.4,
        adv_participation_limit=0.1,
        active_exposure_limit=0.2,
        risk_penalty=1.0,
        transaction_cost_penalty=1.0,
        spread_slippage_bps=2.0,
        liquidity_impact_bps=1.0,
    )
    schedules = (
        FeeSchedule(date(2023, 1, 1), 2.0, 2.0, 10.0, 0.1),
        FeeSchedule(date(2024, 1, 1), 2.0, 2.0, 5.0, 0.1),
    )

    result = optimize_a_share_portfolio(
        assets,
        exposure_columns=("beta", "size"),
        factor_covariance=covariance,
        fee_schedules=schedules,
        trade_date=date(2024, 6, 1),
        config=config,
    )

    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights["b"] >= 0.4 - 1e-9
    assert result.weights["c"] < 0.1
    assert result.predicted_active_risk <= config.tracking_error_limit + 1e-8
    assert result.turnover <= config.turnover_limit + 1e-8
    assert result.maximum_adv_participation <= config.adv_participation_limit + 1e-8
    assert result.estimated_cost > 0.0
    assert result.rebalance_frequency == "1w"


def test_optimizer_enforces_hard_name_cap_and_psd_risk() -> None:
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "alpha": [0.1, 0.0],
            "current_weight": [0.8, 0.2],
            "benchmark_weight": [0.5, 0.5],
            "specific_variance": [0.01, 0.01],
            "adv": [1_000_000.0, 1_000_000.0],
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
            "beta": [1.0, 1.0],
        }
    )
    config = PortfolioOptimizationConfig(
        "1w", 1_000_000.0, 0.6, 0.5, 0.6, 0.3, 1.0, 1.0, 1.0, 2.0, 1.0
    )
    schedules = (FeeSchedule(date(2024, 1, 1), 2.0, 2.0, 5.0, 0.1),)
    covariance = pd.DataFrame([[0.02]], index=["beta"], columns=["beta"])

    result = optimize_a_share_portfolio(
        assets,
        exposure_columns=("beta",),
        factor_covariance=covariance,
        fee_schedules=schedules,
        trade_date=date(2024, 6, 1),
        config=config,
    )
    assert result.weights.max() <= 0.6 + 1e-8

    blocked = assets.copy()
    blocked.loc[0, "can_exit"] = False
    with pytest.raises(ValueError, match="^a_share_optimizer_overweight_cannot_exit$"):
        optimize_a_share_portfolio(
            blocked,
            exposure_columns=("beta",),
            factor_covariance=covariance,
            fee_schedules=schedules,
            trade_date=date(2024, 6, 1),
            config=config,
        )
    with pytest.raises(ValueError, match="^a_share_factor_covariance_not_psd$"):
        optimize_a_share_portfolio(
            assets,
            exposure_columns=("beta",),
            factor_covariance=pd.DataFrame([[-1.0]], index=["beta"], columns=["beta"]),
            fee_schedules=schedules,
            trade_date=date(2024, 6, 1),
            config=replace(config, max_weight=0.9),
        )


def test_weekly_holdings_drift_between_rebalances_without_hidden_trades() -> None:
    panel = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "asset_return": asset_return,
                "alpha": 0.0,
                "current_weight": 0.5,
                "benchmark_weight": 0.5,
                "specific_variance": 1.0,
                "adv": 1_000_000.0,
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
                "beta": 1.0,
            }
            for timestamp, returns in (
                (pd.Timestamp("2024-01-08"), (1.0, 0.0)),
                (pd.Timestamp("2024-01-09"), (0.0, 1.0)),
            )
            for asset_id, asset_return in zip(("a", "b"), returns, strict=True)
        ]
    )
    result = run_a_share_portfolio_backtest(
        panel,
        exposure_columns=("beta",),
        factor_covariance=pd.DataFrame([[0.0]], index=["beta"], columns=["beta"]),
        fee_schedules=(FeeSchedule(date(2024, 1, 1), 1.0, 1.0, 1.0, 1.0),),
        config=PortfolioOptimizationConfig(
            "1w", 1_000_000.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ),
    )
    path = result["path"]

    assert path["turnover"].tolist() == pytest.approx([0.0, 0.0], abs=1e-8)
    assert path["gross_return"].tolist() == pytest.approx([0.5, 1.0 / 3.0])
    assert path["nav"].tolist() == pytest.approx([1.5, 2.0])
