from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.research.models.algorithm_facts import (
    model_algorithm_ids,
    model_smoke_hyperparameters,
)
from quant_research_platform.research.models.prediction import predict_model_frame
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.predictions.outputs import (
    validate_prediction_outputs,
)
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
)


def _model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "factor": [0.1, 0.2, 0.2, 0.4, 0.3, 0.6],
            "feature": [-1.0, 1.0, -0.5, 0.5, -0.25, 0.25],
            "label": [0.01, 0.02, 0.02, 0.04, 0.03, 0.06],
            "eligible": [True] * 6,
            "asset_id": ["a", "b"] * 3,
        }
    )


@pytest.mark.parametrize("algorithm", sorted(model_algorithm_ids()))
def test_all_model_algorithm_ids_fit_and_predict_synthetic_data(
    algorithm: str,
) -> None:
    frame = _model_frame()
    spec = ModelSpec(
        model_id=f"{algorithm}_demo",
        model_algorithm=algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        hyperparameters=model_smoke_hyperparameters(algorithm),
        random_seed=7,
        max_train_rows=len(frame),
    )

    estimator = build_estimator_from_model_spec(spec)
    result = model_fitting.fit_model_result_from_frames(
        (frame,), spec, estimator=build_estimator_from_model_spec(spec)
    )
    predictions = predict_model_frame(frame, spec, result)

    assert estimator is not None
    assert result.metadata["model_algorithm"] == algorithm
    assert result.metadata["input_factor_columns"] == ["factor"]
    assert result.metadata["input_feature_columns"] == ["feature"]
    assert result.metadata["label_column"] == "label"
    assert result.metadata["dependency_status"]["status"] == "available"
    assert result.metadata["test_start"] is None
    assert result.metadata["test_end"] is None
    assert result.metadata["oos_start"] is None
    assert result.metadata["oos_end"] is None
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions["prediction_score"]).any()


def test_training_controls_leave_future_rows_out_of_fit_metadata() -> None:
    frame = _model_frame()
    spec = ModelSpec(
        model_id="linear_regression_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=4,
        purge_gap_steps=1,
    )

    result = model_fitting.fit_model_result_from_frames(
        (frame,), spec, estimator=build_estimator_from_model_spec(spec)
    )

    assert pd.Timestamp(result.metadata["train_end"]) < frame["timestamp"].max()
    assert result.metadata["test_start"] is None
    assert result.metadata["test_end"] is None


def test_model_metadata_records_resolved_parameters_and_backend_version() -> None:
    frame = _model_frame()
    spec = ModelSpec(
        model_id="ridge_regression_demo",
        model_algorithm="ridge_regression",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        hyperparameters={"alpha": 7.0},
        random_seed=11,
        max_train_rows=len(frame),
        purge_gap_steps=1,
    )

    result = model_fitting.fit_model_result_from_frames(
        (frame,),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )

    assert result.metadata["hyperparameters"] == {"alpha": 7.0}
    assert result.metadata["resolved_hyperparameters"] == {"alpha": 7.0}
    assert result.metadata["random_seed"] == 11
    assert result.metadata["purge_gap_steps"] == 1
    assert result.metadata["train_start"] == "2024-01-02"
    assert result.metadata["train_end"] == "2024-01-03"
    assert result.metadata["dependency_status"] == {
        "status": "available",
        "package": "sklearn",
        "version": pytest.importorskip("sklearn").__version__,
    }


def test_prediction_output_helpers_do_not_export_bundle_semantics() -> None:
    frame = _model_frame()
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="linear_regression_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-04",
    )
    model_spec = ModelSpec(
        model_id="linear_regression_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )
    fit = model_fitting.fit_model_result_from_frames(
        (frame.loc[frame["timestamp"] <= "2024-01-02"],),
        model_spec,
        estimator=build_estimator_from_model_spec(model_spec),
    )

    prediction = prediction_frame_from_artifact(
        frame,
        spec,
        fit.coefficients,
        model_spec.input_columns,
        prediction_model=fit.prediction_model,
    )
    validate_prediction_outputs(prediction, spec)

    assert spec.primary_key == ("timestamp", "asset_id")
    assert spec.prediction_column == "prediction_score"
    assert list(prediction.columns) == ["timestamp", "asset_id", "prediction_score"]


def test_prediction_spec_rejects_non_canonical_output_schema() -> None:
    with pytest.raises(ValueError, match="prediction_output_columns_must_be_fixed"):
        PredictionSpec(
            prediction_id="bad_prediction_schema",
            model_id="linear_regression_demo",
            research_dataset_id="research_demo",
            prediction_column="prediction",
            primary_key=("timestamp",),
            train_end="2024-01-02",
            test_start="2024-01-03",
        )
