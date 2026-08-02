from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.exposure import summarize_exposure
from quant_research_platform.compute.dataframe.performance import (
    calculate_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.compute.dataframe.portfolio_returns import (
    simulate_weighted_long_short_returns_and_cost_from_validated_panel,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.portfolios import CostConfig, WeightConfig


def _validated_ranking_panel() -> tuple[pd.DataFrame, ComputePanelSpec]:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    panel = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B", "C"],
            "factor": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        }
    )
    return validate_compute_panel(panel, spec), spec


def test_validated_stream_applies_turnover_costs_and_keeps_market_neutrality() -> None:
    panel, spec = _validated_ranking_panel()

    positions, turnover, returns, _diagnostics, _costs = (
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
            cost_config=CostConfig(slippage_bps=10),
            return_source_column="forward_return",
        )
    )

    assert positions.groupby(["timestamp", "signal"])[
        "position"
    ].sum().abs().max() == pytest.approx(0.0)
    assert turnover["turnover"].tolist() == [2.0, 4.0]
    assert returns["gross_portfolio_return"].round(6).tolist() == [0.02, 0.02]
    assert returns["cost"].round(6).tolist() == [0.002, 0.004]
    assert returns["net_portfolio_return"].round(6).tolist() == [0.018, 0.016]


def test_validated_stream_keeps_long_only_rows_without_short_leg() -> None:
    panel, spec = _validated_ranking_panel()

    _positions, turnover, returns, diagnostics, _costs = (
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(
                portfolio_mode="long_only",
                top_quantile=1 / 3,
                bottom_quantile=1 / 3,
            ),
            cost_config=CostConfig(slippage_bps=10),
            return_source_column="forward_return",
        )
    )

    assert returns["gross_portfolio_return"].round(6).tolist() == [0.03, 0.03]
    assert returns["net_portfolio_return"].round(6).tolist() == [0.029, 0.028]
    assert returns["short_count"].tolist() == [0, 0]
    assert turnover["turnover"].tolist() == [1.0, 2.0]
    assert diagnostics["short_count"].tolist() == [0, 0]


def test_validated_stream_requires_explicit_return_source_column() -> None:
    panel, spec = _validated_ranking_panel()

    with pytest.raises(ValueError, match="portfolio_return_source_column_required"):
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        )


def test_validated_stream_applies_detailed_cost_components() -> None:
    panel, spec = _validated_ranking_panel()

    _positions, _turnover, returns, _diagnostics, costs = (
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
            cost_config=CostConfig(
                buy_cost_bps=10,
                sell_cost_bps=20,
                slippage_bps=2,
                spread_bps=1,
            ),
            return_source_column="forward_return",
        )
    )

    assert costs["buy_cost"].round(6).tolist() == [0.001, 0.002]
    assert costs["sell_cost"].round(6).tolist() == [0.002, 0.004]
    assert costs["slippage_cost"].round(6).tolist() == [0.0004, 0.0008]
    assert costs["spread_cost"].round(6).tolist() == [0.0002, 0.0004]
    assert returns["cost"].round(6).tolist() == [0.0036, 0.0072]


def test_calculate_cumulative_returns_accumulates_by_signal() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02",
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "signal": ["factor", "factor", "alt_factor", "alt_factor"],
            "gross_portfolio_return": [0.03, 0.02, -0.01, 0.04],
            "cost": [0.001, 0.002, 0.0, 0.001],
            "net_portfolio_return": [0.03, 0.02, -0.01, 0.04],
        }
    )

    cumulative_returns = calculate_cumulative_returns(returns)

    ordered = cumulative_returns.sort_values(["signal", "timestamp"]).reset_index(
        drop=True
    )
    assert ordered["cumulative_net_return"].round(6).tolist() == [
        -0.01,
        0.03,
        0.02,
        0.05,
    ]
    assert ordered["cumulative_gross_return"].round(6).tolist() == [
        -0.01,
        0.03,
        0.02,
        0.05,
    ]
    assert ordered["cumulative_cost"].round(6).tolist() == [0.0, 0.001, 0.002, 0.003]
    assert ordered["drawdown"].round(6).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_summarize_performance_reports_common_portfolio_diagnostics() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "signal": ["factor", "factor", "factor"],
            "gross_portfolio_return": [0.03, -0.02, 0.04],
            "turnover": [1.0, 2.0, 1.5],
            "cost": [0.001, 0.002, 0.0015],
            "net_portfolio_return": [0.029, -0.022, 0.0385],
        }
    )
    cumulative_returns = calculate_cumulative_returns(returns)

    summary = summarize_performance(returns, cumulative_returns)

    row = summary.iloc[0]
    assert row["periods"] == 3
    assert row["final_cumulative_net_return"] == pytest.approx(0.0455)
    assert row["win_rate"] == pytest.approx(2 / 3)
    assert row["loss_rate"] == pytest.approx(1 / 3)
    assert row["sortino_like"] != 0
    assert row["volatility_net_portfolio_return"] >= 0.0
    assert row["downside_volatility"] >= 0.0
    assert row["profit_factor"] > 1
    assert row["average_win"] == pytest.approx((0.029 + 0.0385) / 2)
    assert row["average_loss"] == pytest.approx(-0.022)
    assert row["max_consecutive_wins"] == 1
    assert row["max_consecutive_losses"] == 1
    assert row["best_period_return"] == pytest.approx(0.0385)
    assert row["worst_period_return"] == pytest.approx(-0.022)
    assert row["positive_return_rate"] == pytest.approx(2 / 3)
    assert row["negative_return_rate"] == pytest.approx(1 / 3)
    assert row["max_drawdown"] == pytest.approx(-0.022)
    assert row["drawdown_duration"] >= 1
    assert row["mean_turnover"] == pytest.approx(1.5)
    assert row["max_turnover"] == pytest.approx(2.0)
    assert row["total_cost"] == pytest.approx(0.0045)
    assert row["mean_cost"] == pytest.approx(0.0015)
    assert row["cost_to_gross_return"] == pytest.approx(0.0045 / 0.05)


def test_summarize_exposure_reports_position_diagnostics() -> None:
    positions = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "asset": ["A", "B", "A", "B"],
            "signal": ["factor", "factor", "factor", "factor"],
            "position": [0.6, -0.4, 0.5, -0.5],
        }
    )
    long_short = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "signal": ["factor", "factor"],
            "long_count": [1, 1],
            "short_count": [1, 1],
        }
    )

    exposure = summarize_exposure(positions, long_short)

    row = exposure.iloc[0]
    assert row["mean_gross_exposure"] == pytest.approx(1.0)
    assert row["mean_net_exposure"] == pytest.approx(0.1)
    assert row["max_gross_exposure"] == pytest.approx(1.0)
    assert row["max_abs_net_exposure"] == pytest.approx(0.2)
    assert row["mean_long_count"] == pytest.approx(1.0)
    assert row["mean_short_count"] == pytest.approx(1.0)
    assert row["mean_position_count"] == pytest.approx(2.0)
