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

ALGORITHMS = ("xgboost_regressor",)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("requested", [-1000, -1, 1, 65536, None, 0, True, 1.5, "2"])
def test_library_thread_parameters_are_not_hardware_gated(algorithm, requested):
    params = {"n_jobs": requested}
    invalid = type(requested) not in {int, type(None)}
    if invalid:
        with pytest.raises(ValueError, match="invalid_model_thread_count"):
            _spec(algorithm, params)
        return
    spec = _spec(algorithm, params)
    assert estimators.build_estimator_from_model_spec(spec).n_jobs == requested
    assert estimators.resolved_model_hyperparameters(spec) == {
        **params,
        "device": "cuda",
        "tree_method": "hist",
    }
    resolved = {**params, "device": "cuda", "tree_method": "hist"}
    assert _resolved_model_hyperparameters_match(algorithm, params, resolved)
    assert _resolved_model_hyperparameters_match(
        algorithm, params, {**resolved, "n_jobs": 2}
    )
    assert not _resolved_model_hyperparameters_match(
        algorithm, {**params, "n_estimators": 2}, {**resolved, "n_estimators": 3}
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
    ]
    + [
        ("linear_regression", None),
    ],
)
def test_actual_fit_preserves_thread_configuration_and_metadata(
    algorithm, requested, monkeypatch
):
    params = {"n_estimators": 2, "n_jobs": requested} if algorithm in ALGORITHMS else {}
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
    assert result.metadata["resolved_hyperparameters"] == (
        {**params, "device": "cuda", "tree_method": "hist"}
        if algorithm == "xgboost_regressor"
        else params
    )
    assert np.isfinite(result.prediction_model.predict(frame[["x"]])).all()
    if algorithm in ALGORITHMS:
        assert result.prediction_model.n_jobs == requested
