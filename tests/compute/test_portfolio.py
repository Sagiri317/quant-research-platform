from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.exposure import summarize_exposure
from quant_research_platform.compute.dataframe.performance import (
    calculate_additive_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.compute.dataframe.portfolio_returns import (
    simulate_weighted_long_short_returns_and_cost_from_validated_panel,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)


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


def test_ordinal_portfolio_has_no_tied_signal_and_keeps_cutoff_ties() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    def result(signals: list[float], assets: list[str]):
        panel = validate_compute_panel(
            pd.DataFrame(
                {
                    "timestamp": ["2026-01-01"] * len(signals),
                    "asset": assets,
                    "factor": signals,
                    "forward_return": np.linspace(-0.03, 0.03, len(signals)),
                }
            ),
            spec,
        )
        return simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(top_quantile=0.2, bottom_quantile=0.2),
            return_source_column="forward_return",
        )

    tied = result([1.0] * 4, ["a", "b", "c", "d"])
    cutoff = result([0.0, 0.0, 1.0, 2.0, 3.0, 3.0], list("abcdef"))
    renamed = result(
        [3.0, 0.0, 2.0, 0.0, 3.0, 1.0],
        ["u", "v", "w", "x", "y", "z"],
    )

    assert tied[0]["position"].tolist() == [0.0]
    assert tied[2]["gross_portfolio_return"].tolist() == [0.0]
    assert sorted(cutoff[0]["position"].tolist()) == [-0.5, -0.5, 0.5, 0.5]
    assert sorted(renamed[0]["position"].tolist()) == [-0.5, -0.5, 0.5, 0.5]


def test_validated_stream_rejects_disappeared_held_asset() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    panel = validate_compute_panel(
        pd.DataFrame(
            {
                "timestamp": ["2026-01-01", "2026-01-01", "2026-01-02"],
                "asset": ["A", "B", "B"],
                "factor": [2.0, 1.0, 3.0],
                "forward_return": [0.01, 0.02, 0.03],
                "can_exit": [True, True, True],
            }
        ),
        spec,
    )

    with pytest.raises(ValueError, match="held_asset_missing_from_portfolio_frame"):
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(portfolio_mode="long_only", top_quantile=0.5),
            tradability_config=TradabilityConfig(can_exit_column="can_exit"),
            cost_config=CostConfig(sell_cost_bps=10),
            return_source_column="forward_return",
        )


def test_validated_stream_requires_explicit_return_source_column() -> None:
    panel, spec = _validated_ranking_panel()

    with pytest.raises(ValueError, match="portfolio_return_source_column_required"):
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        )


def test_validated_stream_validates_only_an_extra_return_source() -> None:
    panel, spec = _validated_ranking_panel()

    with pytest.raises(ValueError, match="panel is missing columns"):
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
            return_source_column="missing_return",
        )
    with pytest.raises(TypeError, match="return source column 'extra_return'"):
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel.assign(extra_return=["0.01"] * len(panel)),
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
            return_source_column="extra_return",
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


def test_calculate_additive_cumulative_returns_accumulates_by_signal() -> None:
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

    additive_cumulative_returns = calculate_additive_cumulative_returns(returns)

    ordered = additive_cumulative_returns.sort_values(
        ["signal", "timestamp"]
    ).reset_index(drop=True)
    assert ordered["additive_cumulative_net_return"].round(6).tolist() == [
        -0.01,
        0.03,
        0.02,
        0.05,
    ]
    assert ordered["additive_cumulative_gross_return"].round(6).tolist() == [
        -0.01,
        0.03,
        0.02,
        0.05,
    ]
    assert ordered["additive_cumulative_cost"].round(6).tolist() == [
        0.0,
        0.001,
        0.002,
        0.003,
    ]
    assert ordered["high_water_mark"].round(6).tolist() == [0.0, 0.03, 0.02, 0.05]
    assert ordered["drawdown"].round(6).tolist() == [-0.01, 0.0, 0.0, 0.0]


def test_calculate_additive_cumulative_returns_rejects_invalid_numeric_input() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "signal": ["factor"],
            "net_portfolio_return": ["not-a-return"],
        }
    )

    with pytest.raises(ValueError, match="net_portfolio_return"):
        calculate_additive_cumulative_returns(returns)
    with pytest.raises(ValueError, match="net_portfolio_return.*numeric"):
        calculate_additive_cumulative_returns(returns.assign(net_portfolio_return=["0.1"]))


def test_performance_final_additive_returns_match_ordered_cumulative_tail_exactly(
) -> None:
    rng = np.random.default_rng(20260823)
    values = rng.normal(0.0, 0.01, 100_000)
    returns = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(values), freq="min"
            ),
            "signal": ["factor"] * len(values),
            "gross_portfolio_return": values,
            "turnover": np.ones(len(values)),
            "cost": np.zeros(len(values)),
            "net_portfolio_return": values,
        }
    )

    additive = calculate_additive_cumulative_returns(returns)
    summary = summarize_performance(returns).iloc[0]
    additive_tail = additive.iloc[-1]

    assert (
        summary["final_additive_cumulative_gross_return"]
        == additive_tail["additive_cumulative_gross_return"]
    )
    assert (
        summary["final_additive_cumulative_net_return"]
        == additive_tail["additive_cumulative_net_return"]
    )


def test_summarize_performance_empty_frame_keeps_stable_typed_schema() -> None:
    empty_returns = pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "signal": pd.Series([], dtype="string"),
            "gross_portfolio_return": pd.Series([], dtype="float64"),
            "turnover": pd.Series([], dtype="float64"),
            "cost": pd.Series([], dtype="float64"),
            "net_portfolio_return": pd.Series([], dtype="float64"),
        }
    )
    nonempty_returns = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "signal": ["factor"],
            "gross_portfolio_return": [0.01],
            "turnover": [1.0],
            "cost": [0.0],
            "net_portfolio_return": [0.01],
        }
    )

    empty_summary = summarize_performance(empty_returns)
    nonempty_summary = summarize_performance(nonempty_returns)

    assert list(empty_summary.columns) == list(nonempty_summary.columns)
    assert empty_summary.dtypes.tolist() == nonempty_summary.dtypes.tolist()


def test_returns_reject_segment_collision_with_formal_return_column() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "signal": ["factor"],
            "gross_portfolio_return": [0.03],
            "net_portfolio_return": [0.03],
        }
    )

    with pytest.raises(ValueError, match="gross_portfolio_return"):
        calculate_additive_cumulative_returns(
            returns,
            segment_columns=("gross_portfolio_return",),
        )


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
    summary = summarize_performance(returns)

    row = summary.iloc[0]
    assert row["periods"] == 3
    assert row["final_additive_cumulative_net_return"] == pytest.approx(0.0455)
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


def test_performance_variance_uses_exact_variation_state() -> None:
    def make_returns(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=len(values)),
                "signal": ["factor"] * len(values),
                "gross_portfolio_return": values,
                "net_portfolio_return": values,
                "cost": np.zeros(len(values)),
                "turnover": np.ones(len(values)),
            }
        )

    constant = summarize_performance(make_returns(np.full(3, 1e16))).iloc[0]
    tiny = summarize_performance(
        make_returns(np.array([0.0, 1e-320, 2e-320]))
    ).iloc[0]

    assert constant["volatility_net_portfolio_return"] == 0.0
    assert pd.isna(constant["sharpe_like"])
    assert constant["sharpe_like_reason"] == "zero_denominator"
    assert tiny["volatility_net_portfolio_return"] > 0.0
    assert np.isfinite(float(tiny["sharpe_like"]))


def test_performance_higher_moments_are_scale_invariant() -> None:
    def summarize(values: np.ndarray) -> pd.Series:
        return summarize_performance(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=len(values)),
                    "signal": ["factor"] * len(values),
                    "gross_portfolio_return": values,
                    "net_portfolio_return": values,
                    "cost": np.zeros(len(values)),
                    "turnover": np.ones(len(values)),
                }
            )
        ).iloc[0]

    expected = summarize(np.array([0.0, 1.0, 2.0, 4.0]))
    tiny = summarize(np.array([0.0, 1.0, 2.0, 4.0]) * 1e-200)
    huge = summarize(np.array([0.0, 1.0, 2.0, 4.0]) * 1e200)
    constant = summarize(np.full(4, 1e200))

    for actual in (tiny, huge):
        assert actual["skew"] == pytest.approx(expected["skew"])
        assert actual["kurtosis"] == pytest.approx(expected["kurtosis"])
        assert pd.isna(actual["skew_reason"])
        assert pd.isna(actual["kurtosis_reason"])
    assert pd.isna(constant["skew"])
    assert constant["skew_reason"] == "zero_variance"
    assert pd.isna(constant["kurtosis"])
    assert constant["kurtosis_reason"] == "zero_variance"


@pytest.mark.parametrize(
    ("values", "payoff_reason", "profit_factor", "profit_factor_reason"),
    [
        ([0.01, 0.01], "no_negative_returns", None, "no_negative_returns"),
        ([-0.01, -0.02], "no_positive_returns", 0.0, None),
    ],
)
def test_summarize_performance_keeps_win_loss_undefined_reasons_truthful(
    values: list[float],
    payoff_reason: str,
    profit_factor: float | None,
    profit_factor_reason: str | None,
) -> None:
    returns = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(values)),
            "signal": ["factor"] * len(values),
            "gross_portfolio_return": values,
            "turnover": [1.0] * len(values),
            "cost": [0.0] * len(values),
            "net_portfolio_return": values,
        }
    )

    summary = summarize_performance(returns)
    row = summary.iloc[0]

    assert pd.isna(row["payoff_ratio"])
    assert row["payoff_ratio_reason"] == payoff_reason
    if profit_factor is None:
        assert pd.isna(row["profit_factor"])
    else:
        assert row["profit_factor"] == pytest.approx(profit_factor)
    if profit_factor_reason is None:
        assert pd.isna(row["profit_factor_reason"])
    else:
        assert row["profit_factor_reason"] == profit_factor_reason
    assert pd.isna(row["skew"])
    assert row["skew_reason"] == "insufficient_observations"
    assert pd.isna(row["kurtosis"])
    assert row["kurtosis_reason"] == "insufficient_observations"
    if all(value > 0 for value in values):
        assert pd.isna(row["sharpe_like"])
        assert row["sharpe_like_reason"] == "zero_denominator"
        assert pd.isna(row["sortino_like"])
        assert row["sortino_like_reason"] == "zero_denominator"
        assert pd.isna(row["calmar_like"])
        assert row["calmar_like_reason"] == "zero_denominator"
    for column in (
        "sharpe_like",
        "sortino_like",
        "calmar_like",
        "skew",
        "kurtosis",
        "profit_factor",
        "payoff_ratio",
        "average_win",
        "average_loss",
    ):
        assert summary[column].dtype == "float64"
    for column in (
        "sharpe_like_reason",
        "sortino_like_reason",
        "calmar_like_reason",
        "skew_reason",
        "kurtosis_reason",
        "profit_factor_reason",
        "payoff_ratio_reason",
        "average_win_reason",
        "average_loss_reason",
    ):
        assert str(summary[column].dtype) == "string"


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
    with pytest.raises(ValueError, match="long_count.*numeric"):
        summarize_exposure(
            positions,
            long_short.assign(long_count=["1", "1"]),
        )
    with pytest.raises(ValueError, match="short_count.*numeric"):
        summarize_exposure(
            positions,
            long_short.assign(short_count=["1", "1"]),
        )
