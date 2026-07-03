from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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


def _prediction_spec() -> PredictionSpec:
    return PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )


def _portfolio_spec() -> PortfolioSpec:
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


def _portfolio_artifacts() -> dict[str, pd.DataFrame]:
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
        "cumulative_returns": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "gross_portfolio_return": [0.03],
                "net_portfolio_return": [0.03],
                "cost": [0.0],
                "cumulative_gross_return": [0.03],
                "cumulative_net_return": [0.03],
                "cumulative_cost": [0.0],
                "high_water_mark": [0.03],
                "drawdown": [0.0],
            }
        ),
        "performance_summary": pd.DataFrame(
            {
                "signal": ["signal"],
                "periods": [1],
                "final_cumulative_gross_return": [0.03],
                "final_cumulative_net_return": [0.03],
                "mean_net_portfolio_return": [0.03],
                "volatility_net_portfolio_return": [0.0],
                "downside_volatility": [0.0],
                "sharpe_like": [0.0],
                "sortino_like": [0.0],
                "calmar_like": [0.0],
                "best_period_return": [0.03],
                "worst_period_return": [0.03],
                "skew": [0.0],
                "kurtosis": [0.0],
                "var_95": [0.03],
                "cvar_95": [0.03],
                "tail_ratio": [0.0],
                "positive_return_rate": [1.0],
                "negative_return_rate": [0.0],
                "win_rate": [1.0],
                "loss_rate": [0.0],
                "profit_factor": [0.0],
                "payoff_ratio": [0.0],
                "average_win": [0.03],
                "average_loss": [0.0],
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


def _write_portfolio_chunks_for_test(
    artifacts: dict[str, pd.DataFrame],
    *,
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    row_artifacts = _portfolio_row_artifact_names()
    return portfolio_outputs._write_portfolio_output_chunks(
        ({artifact_name: artifacts[artifact_name] for artifact_name in row_artifacts},),
        summary_factory=lambda: {
            artifact_name: artifacts[artifact_name]
            for artifact_name in artifacts
            if artifact_name not in row_artifacts
        },
        spec=spec,
        output_spec=output_spec,
        layout=layout,
    )


def _portfolio_row_artifact_names() -> tuple[str, ...]:
    return (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
        "capacity_diagnostics",
        "numeric_exposure_diagnostics",
        "categorical_exposure_diagnostics",
    )


def _report_spec(
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


def build_evaluation_diagnostics() -> EvaluationDiagnostics:
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
        min_timestamp="2024-01-02",
        max_timestamp="2024-01-02",
        n_groups=2,
        skipped_timestamp_count=0,
    )
def test_portfolio_outputs_write_manifest_and_local_scope(
    tmp_path: Path,
) -> None:
    spec = _portfolio_spec()
    artifacts = _portfolio_artifacts()
    output_bundle = _write_portfolio_chunks_for_test(
        artifacts,
        spec=spec,
        output_spec=_report_spec(
            "portfolio_outputs",
            spec.portfolio_id,
            "portfolio",
        ),
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert output_bundle.as_posix().endswith("outputs/portfolio_demo_report")
    assert manifest["schema_id"] == "qrp-output-manifest"
    assert manifest["output_id"] == "portfolio_demo_report"
    assert manifest["output_definition_id"] == "portfolio_outputs"
    assert manifest["run_id"] == "portfolio_demo"
    assert manifest["metadata"]["portfolio_id"] == "portfolio_demo"
    assert manifest["metadata"]["research_dataset_id"] == "research_demo"
    assert manifest["metadata"]["signals"] == [
        {
            "signal_id": "signal",
            "source_kind": "feature",
            "source_column": "feature",
            "direction": "higher_is_better",
            "transform": "identity",
        }
    ]
    assert manifest["metadata"]["row_counts"]["positions"] == 2
    assert manifest["metadata"]["row_counts"]["portfolio_returns"] == 1
    assert manifest["metadata"]["weight_config"]["method"] == "rank_weighted_quantile"
    assert manifest["files"] == [
        "positions.parquet",
        "turnover.parquet",
        "cost_breakdown.parquet",
        "portfolio_returns.parquet",
        "cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
        "capacity_diagnostics.parquet",
        "numeric_exposure_diagnostics.parquet",
        "categorical_exposure_diagnostics.parquet",
    ]


def test_portfolio_outputs_write_row_artifacts_from_chunks(
    tmp_path: Path,
) -> None:
    spec = _portfolio_spec()
    artifacts = _portfolio_artifacts()
    row_artifacts = (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
        "capacity_diagnostics",
        "numeric_exposure_diagnostics",
        "categorical_exposure_diagnostics",
    )
    chunks = (
        {
            artifact_name: artifacts[artifact_name].iloc[:1].copy()
            for artifact_name in row_artifacts
        },
        {
            artifact_name: artifacts[artifact_name].iloc[1:].copy()
            for artifact_name in row_artifacts
        },
    )
    output_bundle = portfolio_outputs._write_portfolio_output_chunks(
        chunks,
        summary_factory=lambda: {
            artifact_name: artifacts[artifact_name]
            for artifact_name in artifacts
            if artifact_name not in row_artifacts
        },
        spec=spec,
        output_spec=_report_spec(
            "portfolio_outputs",
            spec.portfolio_id,
            "portfolio",
        ),
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["row_counts"]["positions"] == 2
    assert manifest["metadata"]["row_counts"]["turnover"] == 1
    assert len(pd.read_parquet(output_bundle / "positions.parquet")) == 2
    assert len(pd.read_parquet(output_bundle / "portfolio_returns.parquet")) == 1


def test_portfolio_outputs_reject_bad_frames(tmp_path: Path) -> None:
    spec = _portfolio_spec()
    report_spec = _report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    layout = StorageLayout(project_root=tmp_path)

    missing_positions = _portfolio_artifacts()
    missing_positions["positions"] = missing_positions["positions"].drop(
        columns=["asset"]
    )
    with pytest.raises(ValueError, match="missing_portfolio_positions_columns"):
        _write_portfolio_chunks_for_test(
            missing_positions,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    extra_returns = _portfolio_artifacts()
    extra_returns["portfolio_returns"] = extra_returns["portfolio_returns"].assign(
        extra=1
    )
    with pytest.raises(ValueError, match="unregistered_portfolio_portfolio_returns"):
        _write_portfolio_chunks_for_test(
            extra_returns,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    wrong_return_source = _portfolio_artifacts()
    wrong_return_source["portfolio_returns"] = wrong_return_source[
        "portfolio_returns"
    ].assign(return_source_column="other_return")
    with pytest.raises(ValueError, match="portfolio_return_source_column_mismatch"):
        _write_portfolio_chunks_for_test(
            wrong_return_source,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    wrong_signal = _portfolio_artifacts()
    wrong_signal["portfolio_returns"] = wrong_signal["portfolio_returns"].assign(
        signal="other_signal"
    )
    with pytest.raises(ValueError, match="portfolio_artifact_signal_mismatch"):
        _write_portfolio_chunks_for_test(
            wrong_signal,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


