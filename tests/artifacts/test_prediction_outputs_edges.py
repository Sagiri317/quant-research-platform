from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.artifacts import outputs
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
    make_artifact_prediction_output_frame,
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
def test_prediction_output_validation_rejects_bad_frames(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)
    valid = make_artifact_prediction_output_frame()

    with pytest.raises(ValueError, match="missing_prediction_output_columns"):
        outputs.write_prediction_outputs(
            valid.drop(columns=["prediction_score"]),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="unregistered_prediction_output_columns"):
        outputs.write_prediction_outputs(
            valid.assign(extra=1),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        outputs.write_prediction_outputs(
            pd.concat([valid, valid], ignore_index=True),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_require_standard_metadata_attrs(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="missing_model_metadata"):
        outputs.write_prediction_outputs(
            make_artifact_prediction_output_frame(with_metadata=False),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
    )

    incomplete_model = make_artifact_prediction_output_frame()
    incomplete_model.attrs["model_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            incomplete_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    incomplete_prediction = make_artifact_prediction_output_frame()
    incomplete_prediction.attrs["prediction_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_prediction_metadata"):
        outputs.write_prediction_outputs(
            incomplete_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


@pytest.mark.parametrize("include_generic_metadata", [False, True])
def test_prediction_outputs_require_prediction_metadata_attrs(
    tmp_path: Path,
    include_generic_metadata: bool,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)
    frame = make_artifact_prediction_output_frame(with_metadata=False)
    frame.attrs["model_metadata"] = make_artifact_prediction_output_frame().attrs[
        "model_metadata"
    ]
    if include_generic_metadata:
        frame.attrs["metadata"] = {"contract_version": "0.1"}

    with pytest.raises(ValueError, match="missing_prediction_metadata"):
        outputs.write_prediction_outputs(
            frame,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_inconsistent_metadata(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_rows = make_artifact_prediction_output_frame()
    bad_rows.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "prediction_rows": 2,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_rows,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_window = make_artifact_prediction_output_frame()
    bad_window.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "test_start": "2024-01-03",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_window,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_train_end = make_artifact_prediction_output_frame()
    bad_train_end.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "train_end": "2024-01-02",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_train_end,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model = make_artifact_prediction_output_frame()
    bad_model.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "coefficient_count": 0,
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            bad_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_feature_columns = make_artifact_prediction_output_frame()
    bad_feature_columns.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "input_columns": ["factor", "other_input"],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_feature_columns,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_tampered_selected_model_input_columns(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-03",
        candidate_model_ids=("candidate_model",),
        model_selection_validation_start="2024-01-02",
        model_selection_validation_end="2024-01-02",
    )
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    tampered = make_artifact_prediction_output_frame().assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    tampered.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "test_start": "2024-01-03",
        "oos_start": "2024-01-03",
        "oos_end": "2024-01-03",
        "model_selection": {
            "status": "selected",
            "configured_model_id": "model_demo",
            "selected_model_id": "candidate_model",
            "selected_model_input_columns": ["other_input"],
            "candidate_model_ids": ["model_demo", "candidate_model"],
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "validation_start": "2024-01-02",
            "validation_end": "2024-01-02",
            "candidate_count": 2,
            "trial_metrics": [
                {
                    "model_id": "model_demo",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 1.0,
                },
                {
                    "model_id": "candidate_model",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 0.0,
                },
            ],
        },
    }
    tampered.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "test_start": "2024-01-03",
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            tampered,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_tampered_model_selection_direction(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-03",
        candidate_model_ids=("candidate_model",),
        model_selection_validation_start="2024-01-02",
        model_selection_validation_end="2024-01-02",
    )
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    tampered = make_artifact_prediction_output_frame().assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    tampered.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "test_start": "2024-01-03",
        "oos_start": "2024-01-03",
        "oos_end": "2024-01-03",
        "model_selection": {
            "status": "selected",
            "configured_model_id": "model_demo",
            "selected_model_id": "candidate_model",
            "selected_model_input_columns": ["factor"],
            "candidate_model_ids": ["model_demo", "candidate_model"],
            "selection_metric": "validation_rankic",
            "selection_direction": "minimize",
            "validation_start": "2024-01-02",
            "validation_end": "2024-01-02",
            "candidate_count": 2,
            "trial_metrics": [
                {
                    "model_id": "model_demo",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 1.0,
                },
                {
                    "model_id": "candidate_model",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 0.0,
                },
            ],
        },
    }
    tampered.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "test_start": "2024-01-03",
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            tampered,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_bad_row_count_chain_and_filter_contract(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_prediction_counts = make_artifact_prediction_output_frame()
    bad_prediction_counts.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "rows_after_input_dropna": 0,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_prediction_counts,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model_counts = make_artifact_prediction_output_frame()
    bad_model_counts.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "rows_after_filter_mode": 1,
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_counts,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_prediction_filter = make_artifact_prediction_output_frame()
    bad_prediction_filter.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "filter_columns_used": [],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_prediction_filter,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model_filter = make_artifact_prediction_output_frame()
    bad_model_filter.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "filter_definition": "no eligible/can_enter/can_exit filter",
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_filter,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_bad_effective_test_end(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_effective_end = make_artifact_prediction_output_frame()
    bad_effective_end.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_effective_end,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_unparseable_prediction_timestamp(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_timestamp = make_artifact_prediction_output_frame()
    bad_timestamp["timestamp"] = ["bad-date"]

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_timestamp,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_effective_test_end_after_requested_end(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    late_prediction = make_artifact_prediction_output_frame().assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    late_prediction.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "test_end": "2024-01-02",
        "oos_end": "2024-01-02",
    }
    late_prediction.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
        "requested_test_end": "2024-01-02",
        "effective_test_end": "2024-01-03",
        "is_final_oos_window": False,
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            late_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_model_train_window_after_spec_train_end(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_model_window = make_artifact_prediction_output_frame()
    bad_model_window.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
    }

    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_window,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )
