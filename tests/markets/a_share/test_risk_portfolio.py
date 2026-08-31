from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.portfolios.a_share import (
    FeeSchedule,
    PortfolioOptimizationConfig,
    estimate_a_share_factor_risk,
    optimize_a_share_portfolio,
    run_a_share_portfolio_backtest,
)


def test_factor_form_optimizer_enforces_tradability_risk_cost_and_adv() -> None:
    with pytest.raises(ValueError, match="^a_share_fee_schedule_invalid$"):
        FeeSchedule(date(2024, 1, 1), -1.0, 1.0, 1.0, 1.0)
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
    assert result.component_active_risk.sum() == pytest.approx(
        result.predicted_active_risk
    )


def test_factor_risk_estimator_matches_wls_and_outputs_psd_specific_risk() -> None:
    exposures = {"a": (1.0, -1.0), "b": (1.0, 0.0), "c": (1.0, 1.0)}
    weights = {"a": 1.0, "b": 2.0, "c": 1.0}
    rows = []
    expected = []
    for timestamp, factor_return, residuals in (
        (pd.Timestamp("2024-01-02"), (0.01, 0.02), (0.01, -0.01, 0.01)),
        (pd.Timestamp("2024-01-03"), (0.02, -0.01), (-0.01, 0.02, -0.01)),
        (pd.Timestamp("2024-01-04"), (-0.01, 0.03), (0.02, -0.01, 0.00)),
        (pd.Timestamp("2024-01-05"), (0.03, 0.01), (-0.02, 0.01, 0.02)),
    ):
        cross_section = []
        for position, asset_id in enumerate(exposures):
            beta, size = exposures[asset_id]
            row = {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "asset_return": (
                    beta * factor_return[0]
                    + size * factor_return[1]
                    + residuals[position]
                ),
                "risk_weight": weights[asset_id],
                "beta": beta,
                "size": size,
            }
            rows.append(row)
            cross_section.append(row)
        frame = pd.DataFrame(cross_section)
        root_weight = np.sqrt(frame["risk_weight"].to_numpy())
        expected.append(
            np.linalg.lstsq(
                frame[["beta", "size"]].to_numpy() * root_weight[:, None],
                frame["asset_return"].to_numpy() * root_weight,
                rcond=None,
            )[0]
        )

    estimate = estimate_a_share_factor_risk(
        pd.DataFrame(rows),
        exposure_columns=("beta", "size"),
        weight_column="risk_weight",
        ewma_half_life=2.0,
        diagonal_shrinkage=0.25,
        annualization_factor=1.0,
    )

    np.testing.assert_allclose(
        estimate.factor_returns[["beta", "size"]], np.asarray(expected)
    )
    np.testing.assert_allclose(estimate.factor_covariance, estimate.factor_covariance.T)
    assert np.linalg.eigvalsh(estimate.factor_covariance).min() >= -1e-12
    assert (estimate.specific_variance > 0.0).all()


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
        benchmark_returns=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-08", "2024-01-09"]),
                "benchmark_return": [0.5, 0.5],
            }
        ),
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


def test_drifted_holdings_drive_next_rebalance_cost_turnover_and_capacity() -> None:
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
                (pd.Timestamp("2024-01-09"), (0.0, 0.0)),
                (pd.Timestamp("2024-01-15"), (0.0, 0.0)),
            )
            for asset_id, asset_return in zip(("a", "b"), returns, strict=True)
        ]
    )
    result = run_a_share_portfolio_backtest(
        panel,
        benchmark_returns=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-15"]),
                "benchmark_return": [0.5, 0.0, 0.0],
            }
        ),
        exposure_columns=("beta",),
        factor_covariance=pd.DataFrame([[0.0]], index=["beta"], columns=["beta"]),
        fee_schedules=(FeeSchedule(date(2024, 1, 1), 1.0, 1.0, 1.0, 1.0),),
        config=PortfolioOptimizationConfig(
            "1w", 1_000_000.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ),
    )
    path = result["path"]

    assert path["turnover"].iloc[:2].tolist() == [0.0, 0.0]
    third_turnover = path["turnover"].iloc[2]
    assert third_turnover == pytest.approx(1.0 / 3.0, abs=5e-4)
    assert path["cost"].iloc[2] == pytest.approx(third_turnover * 0.00045)
    assert result["summary"]["capacity_notional"]["1pct_adv"] == pytest.approx(
        20_000.0 / third_turnover
    )


def test_official_benchmark_returns_and_new_zero_weight_asset_are_consumed() -> None:
    rows = []
    for timestamp, assets in (
        (pd.Timestamp("2024-01-08"), (("a", 0.5), ("b", 0.5))),
        (
            pd.Timestamp("2024-01-09"),
            (("a", 0.4), ("b", 0.4), ("c", 0.2)),
        ),
    ):
        rows.extend(
            {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "asset_return": 0.0,
                "alpha": 0.0,
                "current_weight": benchmark_weight,
                "benchmark_weight": benchmark_weight,
                "specific_variance": 1.0,
                "adv": 1_000_000.0,
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
                "beta": 1.0,
            }
            for asset_id, benchmark_weight in assets
        )
    result = run_a_share_portfolio_backtest(
        pd.DataFrame(rows),
        benchmark_returns=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-08", "2024-01-09"]),
                "benchmark_return": [0.01, 0.02],
            }
        ),
        exposure_columns=("beta",),
        factor_covariance=pd.DataFrame([[0.0]], index=["beta"], columns=["beta"]),
        fee_schedules=(FeeSchedule(date(2024, 1, 1), 1.0, 1.0, 1.0, 1.0),),
        config=PortfolioOptimizationConfig(
            "1w", 1_000_000.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ),
    )

    assert result["path"]["benchmark_return"].tolist() == [0.01, 0.02]
    assert result["path"]["active_return"].tolist() == pytest.approx([-0.01, -0.02])
