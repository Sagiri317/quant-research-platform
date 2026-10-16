from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.estimators import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.fitting import prediction_callable
from tests.helpers.research_configuration import configured_models
from tests.helpers.research_processed_inputs import _demo_research_rows
from tests.research.direct_model_test_support import fit_frame, predict_frame


def test_native_prediction_preserves_raw_units_rank_and_family(tmp_path: Path) -> None:
    frame = _demo_research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("momentum_20d",),
        target_column="instant_target",
        max_train_rows=len(frame),
    )
    case = fit_frame(
        tmp_path / "fit",
        frame.loc[frame.timestamp.le(pd.Timestamp("2014-01-02"))],
        model_spec,
        market_id="a_share",
    )
    prediction, output, metadata = predict_frame(
        tmp_path / "future",
        frame,
        case,
        window={"start": "2014-01-03", "end": "2014-01-03"},
    )
    assert tuple(prediction.columns) == (
        "timestamp",
        "asset_id",
        "prediction_score",
        "prediction_rank_score",
    )
    assert prediction.timestamp.eq(pd.Timestamp("2014-01-03")).all()
    np.testing.assert_allclose(prediction.prediction_score, [0.02, 0.01])
    assert prediction.prediction_rank_score.tolist() == [0.25, -0.25]
    assert metadata["model_id"] == metadata["family_id"] == model_spec.model_id
    assert output == tmp_path / "future" / "prediction"


@pytest.mark.parametrize("tree_values", [(1.0, 2.0, 3.0), (1e16, -1e16, 1.0)])
def test_forest_parallel_prediction_respects_error_bound_and_fitted_state(
    monkeypatch, tree_values
):
    import pickle
    import threading
    from fractions import Fraction

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
    for tree, value in zip(forest.estimators_, tree_values, strict=True):
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
        expected = float(sum(map(Fraction, tree_values)) / len(tree_values))
        # Forward error for two additions and one final division; the budget scales
        # with actual tree magnitudes, so ordinary data keep a near-epsilon bound.
        eps = np.finfo(float).eps
        gamma = 3 * eps / (1 - 3 * eps)
        budget = gamma * sum(abs(value) for value in tree_values) / len(tree_values)
        np.testing.assert_allclose(first, expected, rtol=0, atol=budget)
        for event in completed:
            event.clear()
        np.testing.assert_allclose(predict(frame), expected, rtol=0, atol=budget)
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
            "model_target_available_at": [
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
    for spec in specs:
        result = model_fitting._fit_model_result(
            model_fitting._collect_training_frame(
                (frame,), spec, fitting_cutoff="2020-02-12"
            ),
            spec,
            estimator=build_estimator_from_model_spec(spec),
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
def test_native_prediction_binds_backend_package_to_actual_model_owner(
    tmp_path, model_algorithm: str, expected_package: str, wrong_package: str
) -> None:
    frame = _demo_research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("momentum_20d",),
        target_column="instant_target",
        max_train_rows=len(frame),
    )
    import json

    case = fit_frame(
        tmp_path / "fit",
        frame.loc[frame.timestamp.le(pd.Timestamp("2014-01-02"))],
        model_spec,
        market_id="a_share",
    )
    _, _, metadata = predict_frame(
        tmp_path / "valid",
        frame,
        case,
        window={"start": "2014-01-03", "end": "2014-01-03"},
    )
    assert (
        not {
            "dependency_status",
            "scaler",
            "feature_importance",
            "hyperparameters",
            "resolved_hyperparameters",
            "model_algorithm",
            "input_columns",
            "target_column",
        }
        & metadata["model_metadata"].keys()
    )
    manifest = case.root / "model.json"
    payload = json.loads(manifest.read_text())
    assert payload["model_metadata"]["dependency_status"]["package"] == expected_package
    payload["model_metadata"]["dependency_status"]["package"] = wrong_package
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        predict_frame(
            tmp_path / "invalid",
            frame,
            case,
            window={"start": "2014-01-03", "end": "2014-01-03"},
        )
    assert not (tmp_path / "invalid" / "prediction").exists()


@pytest.mark.parametrize("start", ["2014-01-01", "2014-01-02"])
def test_native_prediction_must_follow_saved_fit_cutoff(tmp_path, start):
    frame = _demo_research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    case = fit_frame(
        tmp_path / "fit",
        frame.loc[frame.timestamp.eq(pd.Timestamp("2014-01-02"))],
        ModelSpec(
            model_id="cutoff",
            model_algorithm="linear_regression",
            input_factor_columns=("momentum_20d",),
            target_column="instant_target",
        ),
        market_id="a_share",
    )
    with pytest.raises(ValueError, match="prediction_must_follow_fit_cutoff"):
        predict_frame(
            tmp_path / "future",
            frame,
            case,
            window={"start": start, "end": "2014-01-03"},
        )
    assert not (tmp_path / "future" / "prediction").exists()
