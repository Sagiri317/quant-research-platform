from __future__ import annotations

from pathlib import Path

import pandas as pd

import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec, WeightConfig
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.report_output_frames import (
    make_artifact_signal_spec,
)


def make_output_edge_prediction_spec() -> PredictionSpec:
    return PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )


def make_output_edge_portfolio_spec() -> PortfolioSpec:
    return PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        return_source_column="label",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        weight_config=WeightConfig(method="rank_weighted_quantile"),
    )


def make_output_edge_portfolio_artifacts() -> dict[str, pd.DataFrame]:
    timestamp = pd.Timestamp("2024-01-02")
    return {
        "positions": pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp],
                "asset": ["asset_a", "asset_b"],
                "signal": ["signal", "signal"],
                "position": [1.0, -1.0],
            }
        ),
        "turnover": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "turnover": [2.0],
            }
        ),
        "cost_breakdown": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "buy_turnover": [1.0],
                "sell_turnover": [1.0],
                "turnover": [2.0],
                "buy_cost": [0.0],
                "sell_cost": [0.0],
                "slippage_cost": [0.0],
                "spread_cost": [0.0],
                "impact_cost": [0.0],
                "total_cost": [0.0],
                "cost_model": ["scalar_cost_bps"],
            }
        ),
        "portfolio_returns": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "return_source_column": ["label"],
                "portfolio_mode": ["long_short"],
                "long_return": [0.02],
                "short_return": [-0.01],
                "gross_portfolio_return": [0.03],
                "turnover": [2.0],
                "cost": [0.0],
                "net_portfolio_return": [0.03],
                "long_count": [1],
                "short_count": [1],
            }
        ),
        "additive_cumulative_returns": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "gross_portfolio_return": [0.03],
                "net_portfolio_return": [0.03],
                "cost": [0.0],
                "additive_cumulative_gross_return": [0.03],
                "additive_cumulative_net_return": [0.03],
                "additive_cumulative_cost": [0.0],
                "high_water_mark": [0.03],
                "drawdown": [0.0],
            }
        ),
        "performance_summary": pd.DataFrame(
            {
                "signal": ["signal"],
                "periods": [1],
                "final_additive_cumulative_gross_return": [0.03],
                "final_additive_cumulative_net_return": [0.03],
                "mean_net_portfolio_return": [0.03],
                "volatility_net_portfolio_return": [0.0],
                "downside_volatility": [0.0],
                "sharpe_like": [None],
                "sortino_like": [None],
                "calmar_like": [None],
                "best_period_return": [0.03],
                "worst_period_return": [0.03],
                "skew": [None],
                "kurtosis": [None],
                "var_95": [0.03],
                "cvar_95": [0.03],
                "tail_ratio": [1.0],
                "positive_return_rate": [1.0],
                "negative_return_rate": [0.0],
                "win_rate": [1.0],
                "loss_rate": [0.0],
                "profit_factor": [None],
                "payoff_ratio": [None],
                "average_win": [0.03],
                "average_loss": [None],
                "max_consecutive_wins": [1],
                "max_consecutive_losses": [0],
                "max_drawdown": [0.0],
                "drawdown_duration": [0],
                "max_drawdown_start": [None],
                "max_drawdown_valley": [None],
                "max_drawdown_recovery": [None],
                "mean_turnover": [2.0],
                "max_turnover": [2.0],
                "total_cost": [0.0],
                "mean_cost": [0.0],
                "cost_to_gross_return": [0.0],
                "cost_to_net_return": [0.0],
                "sharpe_like_reason": ["zero_denominator"],
                "sortino_like_reason": ["zero_denominator"],
                "calmar_like_reason": ["zero_denominator"],
                "skew_reason": ["insufficient_observations"],
                "kurtosis_reason": ["insufficient_observations"],
                "tail_ratio_reason": [None],
                "profit_factor_reason": ["no_negative_returns"],
                "payoff_ratio_reason": ["no_negative_returns"],
                "average_win_reason": [None],
                "average_loss_reason": ["no_negative_returns"],
                "cost_to_gross_return_reason": [None],
                "cost_to_net_return_reason": [None],
            }
        ),
        "exposure_summary": pd.DataFrame(
            {
                "signal": ["signal"],
                "mean_gross_exposure": [2.0],
                "mean_net_exposure": [0.0],
                "max_gross_exposure": [2.0],
                "max_abs_net_exposure": [0.0],
                "mean_long_count": [1.0],
                "mean_short_count": [1.0],
                "mean_position_count": [2.0],
            }
        ),
        "concentration": pd.DataFrame(
            {
                "signal": ["signal"],
                "periods": [1],
                "mean_hhi": [0.5],
                "max_hhi": [0.5],
                "mean_top_weight": [0.5],
                "max_top_weight": [0.5],
                "mean_position_count": [2.0],
            }
        ),
        "period_breakdown": pd.DataFrame(
            {
                "signal": ["signal"],
                "year": [2024],
                "month": [1],
                "periods": [1],
                "gross_portfolio_return": [0.03],
                "net_portfolio_return": [0.03],
                "cost": [0.0],
                "turnover": [2.0],
                "hit_rate": [1.0],
            }
        ),
        "diagnostics": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "method": ["rank_weighted_quantile"],
                "universe_count": [2],
                "eligible_count": [2],
                "can_enter_count": [2],
                "can_exit_count": [2],
                "entry_blocked_count": [0],
                "exit_blocked_count": [0],
                "long_count": [1],
                "short_count": [1],
                "fallback_reason": [None],
            }
        ),
        "capacity_diagnostics": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "row_count": [2],
                "positive_amount_row_count": [2],
                "excluded_non_positive_amount_count": [0],
                "gross_weight": [2.0],
                "covered_gross_weight": [2.0],
                "weighted_amount_proxy": [100.0],
                "weighted_float_market_cap_proxy": [1000.0],
            }
        ),
        "numeric_exposure_diagnostics": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "exposure_column": ["beta"],
                "exposure_value": [0.0],
                "row_count": [2],
                "gross_weight": [2.0],
            }
        ),
        "categorical_exposure_diagnostics": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "exposure_column": ["industry"],
                "exposure_value": ["tech"],
                "asset_count": [2],
                "gross_weight": [2.0],
                "net_weight": [0.0],
                "long_weight": [1.0],
                "short_weight": [-1.0],
            }
        ),
    }


def write_output_edge_portfolio_chunks(
    artifacts: dict[str, pd.DataFrame],
    *,
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    row_artifacts = tuple(
        name
        for name in output_edge_portfolio_row_artifact_names()
        if name != "additive_cumulative_returns"
    )
    return portfolio_outputs._write_portfolio_output_chunks(
        ({artifact_name: artifacts[artifact_name] for artifact_name in row_artifacts},),
        summary_factory=lambda: {
            artifact_name: artifacts[artifact_name]
            for artifact_name in artifacts
            if artifact_name not in (*row_artifacts, "additive_cumulative_returns")
        },
        additive_cumulative_returns_factory=lambda: iter(
            (artifacts["additive_cumulative_returns"],)
        ),
        spec=spec,
        output_spec=output_spec,
        layout=layout,
    )


def output_edge_portfolio_row_artifact_names() -> tuple[str, ...]:
    return (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "additive_cumulative_returns",
        "diagnostics",
        "capacity_diagnostics",
        "numeric_exposure_diagnostics",
        "categorical_exposure_diagnostics",
    )


def make_output_edge_report_spec(
    output_definition_id: str,
    run_id: str,
    run_type: str,
) -> OutputSpec:
    return OutputSpec(
        output_id=f"{run_id}_report",
        output_definition_id=output_definition_id,
        run_id=run_id,
        market_id=MarketId("a_share"),
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def make_output_edge_evaluation_diagnostics(
    *,
    min_timestamp: str = "2024-01-02",
    max_timestamp: str = "2024-01-02",
) -> EvaluationDiagnostics:
    return EvaluationDiagnostics(
        filter_mode="eligible",
        filter_columns_used=("eligible",),
        filter_definition="eligible == True",
        rows_input=2,
        rows_after_filter_mode=2,
        rows_after_signal_label_dropna=2,
        metric_rows=1,
        grouped_label_value_rows=1,
        ic_summary_rows=1,
        group_spread_summary_rows=1,
        timestamp_count=1,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        n_groups=2,
        skipped_timestamp_count=0,
    )
