from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
import quant_research_platform.research.predictions.runner as prediction_runner_module
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.predictions.outputs import prediction_callable
from tests.helpers.research_configuration import configured_models
from tests.research.prediction_output_helpers import write_prediction_output_and_read
from tests.research.workflow_test_support import _research_rows


def test_evaluation_prediction_and_report_boundary(tmp_path: Path) -> None:
    frame = _research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("momentum_20d",),
        label_column="instant_target",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id="pred_demo",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )
    estimator = build_estimator_from_model_spec(model_spec)
    model_result = model_fitting._fit_model_result_from_frames(
        (frame.loc[frame["timestamp"] <= pd.Timestamp(prediction_spec.train_end)],),
        model_spec,
        estimator=estimator,
        fitting_cutoff=prediction_spec.train_end,
    )
    prediction, output_bundle, _ = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        prediction_spec,
        model_spec.input_columns,
        prediction_model=model_result.prediction_model,
        model_metadata=model_result.metadata,
    )
    assert prediction_spec.prediction_column in prediction.columns
    assert prediction["timestamp"].min() >= pd.Timestamp("2014-01-03")
    assert prediction["timestamp"].max() <= pd.Timestamp("2014-01-03")
    assert output_bundle.as_posix().endswith("outputs/pred_demo")


def test_a_share_prediction_requires_persisted_quarter_lineage() -> None:
    with pytest.raises(
        ValueError, match="a_share_persisted_prediction_requires_quarter_lineage"
    ):
        PredictionSpec(
            prediction_id="a_share_prediction",
            model_id="linear_a_share",
            research_dataset_id="demo",
            train_end="2014-01-02",
            test_start="2014-01-03",
            prediction_mode="a_share_oos",
            prediction_column=RAW_PREDICTION_COLUMN,
        )


def test_forest_prediction_keeps_tree_order_without_mutating_fitted_state(monkeypatch):
    import pickle
    import threading

    from joblib import parallel_config
    from joblib.parallel import get_active_backend
    from sklearn.ensemble import RandomForestRegressor, _forest

    from quant_research_platform.research.models.scaler import (
        ModelMatrixScaler,
        ScaledPredictor,
    )

    frame = pd.DataFrame({"x": [0.0, 1.0]})
    forest = RandomForestRegressor(
        n_estimators=3, n_jobs=3, bootstrap=False, random_state=0
    ).fit(frame, [0.0, 0.0])
    for tree, value in zip(forest.estimators_, (1e16, -1e16, 1.0), strict=True):
        tree.tree_.value.fill(value)
    fitted = ScaledPredictor(forest, ModelMatrixScaler.fit(frame))
    frozen_state = pickle.dumps(fitted)
    actual_accumulate = _forest._accumulate_prediction
    caller = threading.current_thread()
    tree_positions = {id(tree): index for index, tree in enumerate(forest.estimators_)}
    completed = [threading.Event() for _ in forest.estimators_]
    predecessors = {0: None, 1: 2, 2: 0}

    def accumulate(predict, *args):
        position = tree_positions[id(predict.__self__)]
        if threading.current_thread() is not caller:
            previous = predecessors[position]
            if previous is not None:
                assert completed[previous].wait(10), "forest worker did not finish"
        actual_accumulate(predict, *args)
        completed[position].set()

    monkeypatch.setattr(_forest, "_accumulate_prediction", accumulate)
    predict = prediction_callable(("x",), fitted)
    with parallel_config(backend="threading", n_jobs=3):
        outer_backend, outer_jobs = get_active_backend()
        first = predict(frame)
        np.testing.assert_array_equal(first, np.full(len(frame), 1.0 / 3.0))
        for event in completed:
            event.clear()
        np.testing.assert_array_equal(predict(frame), first)
        assert get_active_backend() == (outer_backend, outer_jobs)
    assert forest.n_jobs == 3
    assert pickle.dumps(fitted) == frozen_state


def test_a_share_model_ladder_fits_and_predicts_each_retained_algorithm() -> None:
    specs = configured_models("pv_all")
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": [date for date in dates for _ in range(3)],
            "asset_id": [asset for _ in dates for asset in ("a", "b", "c")],
            "eligible": True,
            "model_target": [
                float(date_index + asset_index) / 10.0
                for date_index in range(1, len(dates) + 1)
                for asset_index in range(1, 4)
            ],
            "forward_return_20d_available_at": [
                date + pd.offsets.BDay(20) for date in dates for _ in range(3)
            ],
        }
    )
    for spec in specs:
        for column in spec.input_columns:
            frame[column] = [
                float((date_index + asset_index) % 7 + 1)
                for date_index in range(len(dates))
                for asset_index in range(3)
            ]
    preflight_model_specs(specs)
    for spec in specs:
        result = model_fitting._fit_model_result_from_frames(
            (frame,),
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2020-02-12",
        )
        columns = tuple(dict.fromkeys((*spec.input_columns, "timestamp")))
        prediction = prediction_callable(spec.input_columns, result.prediction_model)(
            frame.loc[:, list(columns)]
        )
        assert len(prediction) == len(frame)
        assert pd.Series(prediction).notna().all()


@pytest.mark.parametrize(
    ("model_algorithm", "expected_package", "wrong_package"),
    (
        ("linear_regression", "internal", "sklearn"),
        ("ridge_regression", "sklearn", "internal"),
    ),
)
def test_prediction_builder_binds_backend_package_to_actual_model_owner(
    model_algorithm: str, expected_package: str, wrong_package: str
) -> None:
    frame = _research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("momentum_20d",),
        label_column="instant_target",
        max_train_rows=len(frame),
    )
    estimator = build_estimator_from_model_spec(model_spec)
    model_result = model_fitting._fit_model_result_from_frames(
        (frame,), model_spec, estimator=estimator, fitting_cutoff="2014-01-02"
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_prediction",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )
    result = prediction_runner_module.build_prediction_result_from_frames(
        (frame,),
        prediction_spec,
        model_spec.input_columns,
        prediction_model=model_result.prediction_model,
        model_metadata=model_result.metadata,
    )
    assert result.model_metadata["dependency_status"]["package"] == expected_package
    wrong_metadata = {
        **model_result.metadata,
        "dependency_status": {
            **model_result.metadata["dependency_status"],
            "package": wrong_package,
        },
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        prediction_runner_module.build_prediction_result_from_frames(
            (frame,),
            prediction_spec,
            model_spec.input_columns,
            prediction_model=model_result.prediction_model,
            model_metadata=wrong_metadata,
        )
