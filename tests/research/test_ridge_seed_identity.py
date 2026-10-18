"""Real backend states retain typed parameters and explicit seeds."""

import numpy as np
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting


@pytest.mark.parametrize("solver", ("sag", "saga"))
@pytest.mark.parametrize("boundary", ("resolved", "state_alpha", "state_seed"))
def test_ridge_resolved_and_native_state_parameters_keep_typed_identity(
    solver, boundary
):
    import copy
    import pickle

    import pandas as pd

    from quant_research_platform.contracts.models import validate_model_metadata_core
    from quant_research_platform.research.models.estimators import (
        build_estimator_from_model_spec,
    )

    rng = np.random.default_rng(14)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120),
            "x": rng.normal(size=120),
            "label": rng.normal(size=120),
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=731,
        hyperparameters={"solver": solver, "alpha": 1, "max_iter": 10000, "tol": 1e-8},
    )
    fitted = fitting._fit_model_result(
        fitting._collect_training_frame(
            (frame,), spec, fitting_cutoff=frame.timestamp.max() + pd.Timedelta(days=2)
        ),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    metadata = validate_model_metadata_core(fitted.metadata)
    predictor = pickle.loads(pickle.dumps(fitted.prediction_model))
    fitting.validate_model_state(predictor, metadata)
    original = pickle.dumps(predictor)
    changed = copy.deepcopy(metadata)
    if boundary == "resolved":
        changed["resolved_hyperparameters"]["alpha"] = 1.0
        with pytest.raises(ValueError, match="^invalid_model_metadata$"):
            validate_model_metadata_core(changed)
    else:
        key, value = (
            ("alpha", 1.0) if boundary == "state_alpha" else ("random_state", 731.0)
        )
        predictor.estimator.set_params(**{key: value})
        with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
            fitting.validate_model_state(predictor, metadata)
    fitting.validate_model_state(pickle.loads(original), metadata)


@pytest.mark.parametrize("seed,alias", ((1, True), (0, 0.0)))
def test_ridge_native_seed_type_aliases_are_rejected(seed, alias):
    import pandas as pd

    from quant_research_platform.research.models.estimators import (
        build_estimator_from_model_spec,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=40),
            "x": np.arange(40.0),
            "label": np.arange(40.0) / 2,
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=seed,
    )
    fitted = fitting._fit_model_result(
        fitting._collect_training_frame(
            (frame,), spec, fitting_cutoff=frame.timestamp.max() + pd.Timedelta(days=2)
        ),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    fitted.prediction_model.estimator.random_state = alias
    with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
        fitting.validate_model_state(fitted.prediction_model, fitted.metadata)


def test_resolved_scientific_parameters_preserve_typed_identity_and_float_zero():
    from quant_research_platform.contracts.models import (
        _resolved_model_hyperparameters_match,
    )

    assert _resolved_model_hyperparameters_match(
        "ridge_regression", {"alpha": -0.0}, {"alpha": 0.0}
    )
    assert not _resolved_model_hyperparameters_match(
        "random_forest_regressor",
        {"n_jobs": -1, "min_samples_split": 2},
        {"n_jobs": 2, "min_samples_split": 2.0},
    )


@pytest.mark.parametrize(
    "algorithm",
    (
        "ridge_regression",
        "lasso_regression",
        "huber_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    ),
)
def test_native_estimator_defaults_and_nan_pass_typed_state_checks(algorithm, tmp_path):
    import pickle

    import pandas as pd

    from quant_research_platform.contracts.models import validate_model_metadata_core

    rng = np.random.default_rng(27)
    x, z = rng.normal(size=(2, 120))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120),
            "x": x,
            "z": z,
            "label": 0.3 * x + 0.2 * z + rng.normal(scale=0.01, size=120),
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    params = {}
    if algorithm in {
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }:
        params.update(n_estimators=2, max_depth=2)
    if algorithm in {
        "random_forest_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }:
        params["n_jobs"] = 1
    if algorithm == "lightgbm_regressor":
        params["verbosity"] = -1
    if algorithm == "ridge_regression":
        params["alpha"] = 0.0
    spec = ModelSpec(
        model_id="probe",
        model_algorithm=algorithm,
        input_factor_columns=("x", "z"),
        target_column="label",
        random_seed=731,
        hyperparameters=params,
    )
    from tests.research.direct_model_test_support import fit_frame

    frame["asset_id"] = "asset"
    case = fit_frame(
        tmp_path, frame, spec, cutoff=frame.timestamp.max() + pd.Timedelta(days=2)
    )
    fitted = case.result
    metadata = validate_model_metadata_core(fitted.metadata)
    predictor = pickle.loads(pickle.dumps(fitted.prediction_model))
    if algorithm == "ridge_regression":
        predictor.estimator.alpha = -0.0
    fitting.validate_model_state(predictor, metadata)


@pytest.mark.parametrize("seed", [None, True, -1, 2**32])
def test_model_seed_rejects_unseeded_or_invalid_requests(seed):
    with pytest.raises((TypeError, ValueError), match="random_seed"):
        ModelSpec(
            model_id="ridge",
            model_algorithm="ridge_regression",
            input_feature_columns=("x",),
            target_column="label",
            random_seed=seed,
        )


@pytest.mark.parametrize("seed", [42, 0, 731])
def test_model_seed_default_override_and_persisted_backend_identity(tmp_path, seed):
    from quant_research_platform.contracts.models import _model_spec
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit
    from tests.research.test_model_explanations import _case

    case, _, _ = _case(tmp_path, "ridge_regression")
    config = dict(case.config, output=str(tmp_path / "seed-model"))
    config["model"] = dict(config["model"])
    if seed == 42:
        del config["model"]["random_seed"]
    else:
        config["model"]["random_seed"] = seed
    assert _model_spec(config["model"]).random_seed == seed
    result, _ = load_model(fit(config))
    assert result.metadata["random_seed"] == seed
    assert result.prediction_model.estimator.random_state == seed
