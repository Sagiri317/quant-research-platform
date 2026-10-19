"""Model libraries own thread resolution; persisted parameters retain requests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

from quant_research_platform.contracts.models import (
    ModelSpec,
    _resolved_model_hyperparameters_match,
    validate_model_metadata_core,
)
from quant_research_platform.research.models import estimators, fitting

ALGORITHMS = (
    "random_forest_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("requested", [-1000, -1, 1, 65536, None, 0, True, 1.5, "2"])
def test_library_thread_parameters_are_not_hardware_gated(algorithm, requested):
    params = {"n_jobs": requested}
    invalid = type(requested) not in {int, type(None)} or (
        requested == 0 and algorithm == "random_forest_regressor"
    )
    if invalid:
        with pytest.raises(ValueError, match="invalid_model_thread_count"):
            _spec(algorithm, params)
        return
    spec = _spec(algorithm, params)
    assert estimators.build_estimator_from_model_spec(spec).n_jobs == requested
    assert estimators.resolved_model_hyperparameters(spec) == params
    assert _resolved_model_hyperparameters_match(algorithm, params, params)
    assert _resolved_model_hyperparameters_match(algorithm, params, {"n_jobs": 2})
    assert not _resolved_model_hyperparameters_match(
        algorithm, {**params, "n_estimators": 2}, {**params, "n_estimators": 3}
    )


def _spec(algorithm, params):
    return ModelSpec(
        model_id="threads",
        model_algorithm=algorithm,
        input_feature_columns=("x",),
        target_column="label",
        hyperparameters=params,
    )


@pytest.mark.parametrize(
    "algorithm,requested",
    [
        (algorithm, requested)
        for algorithm in ALGORITHMS
        for requested in [-1, None, 1, 0]
        if requested != 0 or algorithm != "random_forest_regressor"
    ]
    + [("linear_regression", None), ("ridge_regression", None)],
)
def test_actual_fit_preserves_thread_configuration_and_metadata(
    algorithm, requested, monkeypatch
):
    params = {"n_estimators": 2, "n_jobs": requested} if algorithm in ALGORITHMS else {}
    if algorithm == "lightgbm_regressor":
        params.update(verbosity=-1, min_child_samples=1)
    spec = _spec(algorithm, params)
    if algorithm in ALGORITHMS:
        default = replace(spec, hyperparameters={})
        assert estimators.build_estimator_from_model_spec(default).n_jobs == -1
    frame = pd.DataFrame(
        dict(
            timestamp=pd.date_range("2024-01-01", periods=8),
            asset_id="a",
            x=np.arange(8.0),
            label=np.arange(8.0) * 2,
            eligible=True,
            label_available_at=pd.Timestamp("2024-01-08"),
        )
    )
    estimator = estimators.build_estimator_from_model_spec(spec)
    observed = []
    if estimator is None:
        original = np.linalg.lstsq

        def fit(*args, **kwargs):
            observed.append(threadpool_info())
            return original(*args, **kwargs)

        monkeypatch.setattr(np.linalg, "lstsq", fit)
    else:
        original = estimator.fit

        def fit(*args, **kwargs):
            observed.append(threadpool_info())
            return original(*args, **kwargs)

        monkeypatch.setattr(estimator, "fit", fit)
    before = threadpool_info()
    result = fitting._fit_model_result(
        fitting._collect_training_frame([frame], spec, fitting_cutoff="2024-01-08"),
        spec,
        estimator=estimator,
    )
    assert observed == [before]
    assert threadpool_info() == before
    validate_model_metadata_core(result.metadata)
    assert result.metadata["resolved_hyperparameters"] == params
    assert np.isfinite(result.prediction_model.predict(frame[["x"]])).all()
    if algorithm in ALGORITHMS:
        assert result.prediction_model.n_jobs == requested
        result.metadata["resolved_hyperparameters"]["n_jobs"] = 2
        validate_model_metadata_core(result.metadata)
        fitting.validate_model_state(result.prediction_model, result.metadata)
        result.metadata["resolved_hyperparameters"]["n_estimators"] = 3
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            validate_model_metadata_core(result.metadata)
        with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
            fitting.validate_model_state(result.prediction_model, result.metadata)
