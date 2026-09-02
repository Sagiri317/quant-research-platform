from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import (
    write_prediction_outputs,
)
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.models.fitting import ModelFitResult
from quant_research_platform.research.models.production import QuarterlyModelResult
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)
from quant_research_platform.storage.layout import StorageLayout


class _ColumnPredictor:
    def __init__(self, column: str, offset: float = 0.0) -> None:
        self.column = column
        self.offset = offset

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].to_numpy(dtype="float64") + self.offset


class _UnpicklablePredictor(_ColumnPredictor):
    def __getstate__(self) -> object:
        raise RuntimeError("predictor_serialization_failed")


def _metadata(
    columns: tuple[str, ...],
    *,
    algorithm: str = "linear_regression",
    model_id: str = "champion",
    rank_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "model_algorithm": algorithm,
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2014-01-01",
        "train_end": "2018-12-31",
        "input_factor_columns": list(columns),
        "input_feature_columns": [],
        "input_columns": list(columns),
        "input_count": len(columns),
        "label_column": "label",
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": 11,
        "max_train_rows": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
        "coefficient_count": len(columns) + 1,
        "sample_weight_protocol": "equal_date_total_equal_sum_n_train",
        "scaler": None,
        "feature_importance": {column: 1.0 for column in columns},
        "fit_cutoff": "2018-12-31",
        "model_selection": _selection_metadata(
            family_id=algorithm,
            model_id=model_id,
            columns=columns,
        ),
    }
    if rank_parameters is not None:
        result["rank_parameters"] = rank_parameters
    return result


def _selection_metadata(
    *,
    family_id: str,
    model_id: str,
    columns: tuple[str, ...],
) -> dict[str, object]:
    del columns
    trial = {
        "model_id": model_id,
        "validation_rows": 2,
        "validation_ic": 0.1,
        "validation_rankic": 0.2,
        "validation_ic_reason": None,
        "validation_rankic_reason": None,
        "validation_mse": 1.0,
        "resolved_hyperparameters": {},
        "random_seed": 11,
        "n_train": 2,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
    }
    return {
        "status": "selected",
        "selection_metric": "validation_rankic",
        "selection_direction": "maximize",
        "validation_start": "2018-01-01",
        "validation_end": "2018-12-31",
        "candidate_family_ids": [family_id],
        "tie_break_order": [family_id],
        "selected_family_id": family_id,
        "selected_model_id": model_id,
        "families": [
            {
                "family_id": family_id,
                "config_tie_break_order": [model_id],
                "selected_config_id": model_id,
                "trials": [trial],
                "validation_rows": 2,
                "selection_metric_value": 0.2,
                "selection_metric_reason": None,
            }
        ],
    }


def _fold(quarter: int) -> ExperimentFold:
    start = f"2019-{1 + (quarter - 1) * 3:02d}-01"
    end = (pd.Timestamp(start) + pd.offsets.QuarterEnd()).date().isoformat()
    return ExperimentFold(
        fold_id=f"2019-q{quarter}",
        train=TimeWindow("2014-01-01", "2017-12-31"),
        validation=TimeWindow("2018-01-01", "2018-12-31"),
        development_test=TimeWindow(start, end),
    )


def _quarterly_result(
    quarter: int,
    *,
    model_id: str,
    columns: tuple[str, ...],
    predictor: object | None = None,
) -> QuarterlyModelResult:
    model = predictor or _ColumnPredictor(columns[0])
    algorithm = (
        "ic_weighted_rank_ensemble"
        if isinstance(model, ICWeightedRankEnsemble)
        else "equal_weight_rank_ensemble"
        if isinstance(model, RankEnsemblePredictor)
        else "linear_regression"
    )
    return QuarterlyModelResult(
        quarter=_fold(quarter),
        model_id=model_id,
        model_result=ModelFitResult(
            coefficients=np.zeros(len(columns) + 1),
            metadata=_metadata(columns, algorithm=algorithm, model_id=model_id),
            prediction_model=model,
        ),
    )


def _passthrough_session(**attributes):
    session = SimpleNamespace(**attributes)

    @contextmanager
    def filtered_view(_time_window):
        yield session

    session.filtered_view = filtered_view
    return session


def _output_spec(
    *,
    output_definition_id: str,
    output_id: str,
    run_type: str,
    market_id: MarketId | None = None,
) -> OutputSpec:
    return OutputSpec(
        output_id=output_id,
        output_definition_id=output_definition_id,
        run_id="test_run",
        market_id=market_id or MarketId("a_share"),
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _write_prediction_result(layout: StorageLayout, result) -> Path:
    return write_prediction_outputs(
        spec=result.spec,
        output_spec=_output_spec(
            output_definition_id="prediction_outputs",
            output_id=result.spec.prediction_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )


def _research_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000002", "000001", "000002"],
            "timestamp": pd.to_datetime(
                ["2014-01-02", "2014-01-02", "2014-01-03", "2014-01-03"]
            ),
            "mom_20d": [1.0, 2.0, 2.0, 1.0],
            "forward_return_20d": [0.01, 0.03, 0.02, 0.00],
            "eligible": [True, True, True, True],
            "can_enter": [True, True, True, True],
            "can_exit": [True, True, True, True],
            "market_type": [4, 4, 4, 4],
        }
    )


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "mom_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)
