from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.artifacts import outputs
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec, WeightConfig
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.report_output_frames import (
    make_artifact_evaluation_group_spread_summary_frame,
    make_artifact_evaluation_grouped_label_values_frame,
    make_artifact_evaluation_ic_summary_frame,
    make_artifact_evaluation_metrics_frame,
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
def test_evaluation_outputs_validation_rejects_bad_inputs(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")

    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns"):
        outputs.write_evaluation_outputs(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=build_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_ic_summary_columns"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=pd.DataFrame({"signal": ["signal"]}),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=build_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_group_spread_summary"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=pd.DataFrame({"signal": ["signal"]}),
            evaluation_diagnostics=build_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_evaluation_outputs_rejects_inconsistent_diagnostics(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")
    layout = StorageLayout(project_root=tmp_path)

    bad_metric_rows = replace(build_evaluation_diagnostics(), metric_rows=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_metric_rows,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_filter_mode = replace(build_evaluation_diagnostics(), filter_mode="unfiltered")
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_filter_mode,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_count = replace(build_evaluation_diagnostics(), timestamp_count=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_count,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    skipped_metrics = make_artifact_evaluation_metrics_frame().assign(
        n_obs=[1]
    )
    bad_skipped = replace(build_evaluation_diagnostics(), skipped_timestamp_count=0)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            skipped_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_skipped,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_metrics = make_artifact_evaluation_metrics_frame().assign(
        timestamp=["bad-date"]
    )
    bad_timestamp_diagnostics = replace(
        build_evaluation_diagnostics(),
        timestamp_count=0,
        min_timestamp=None,
        max_timestamp=None,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            bad_timestamp_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_diagnostics,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_evaluation_outputs_rejects_bad_keys_features_groups_and_counts(
    tmp_path: Path,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")
    layout = StorageLayout(project_root=tmp_path)

    duplicate_metrics = pd.concat(
        [
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_metrics_frame(),
        ],
        ignore_index=True,
    )
    duplicate_diagnostics = replace(
        build_evaluation_diagnostics(),
        metric_rows=2,
        rows_after_signal_label_dropna=4,
    )
    with pytest.raises(ValueError, match="duplicate_evaluation_metrics_keys"):
        outputs.write_evaluation_outputs(
            duplicate_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=duplicate_diagnostics,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_feature_metrics = make_artifact_evaluation_metrics_frame().assign(
        signal="other"
    )
    with pytest.raises(ValueError, match="evaluation_artifact_signal_label_mismatch"):
        outputs.write_evaluation_outputs(
            bad_feature_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=build_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_group = make_artifact_evaluation_grouped_label_values_frame().assign(group=3)
    with pytest.raises(ValueError, match="evaluation_group_out_of_range"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            bad_group,
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=build_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_row_chain = replace(build_evaluation_diagnostics(), rows_after_filter_mode=1)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_row_chain,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_evaluation_outputs_writes_label_value_artifact_name(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")

    output_bundle = outputs.write_evaluation_outputs(
        make_artifact_evaluation_metrics_frame(),
        make_artifact_evaluation_grouped_label_values_frame(),
        ic_summary=make_artifact_evaluation_ic_summary_frame(),
        group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=build_evaluation_diagnostics(),
        spec=spec,
        output_spec=report_spec,
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert (output_bundle / "grouped_label_values.parquet").is_file()
    assert (output_bundle / "ic_summary.parquet").is_file()
    assert (output_bundle / "group_spread_summary.parquet").is_file()
    assert not (output_bundle / "grouped_returns.parquet").exists()
    assert manifest["schema_id"] == "qrp-output-manifest"
    assert manifest["files"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
    ]

