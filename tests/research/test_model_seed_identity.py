"""Real backend states retain typed parameters and explicit seeds."""

import numpy as np
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting


@pytest.mark.parametrize(
    "algorithm",
    (
        "lasso_regression",
        "xgboost_regressor",
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
        "xgboost_regressor",
    }:
        params.update(n_estimators=2, max_depth=2)
    if algorithm in {
        "xgboost_regressor",
    }:
        params["n_jobs"] = 1
    if algorithm == "lasso_regression":
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
    if algorithm == "lasso_regression":
        predictor.estimator.alpha = -0.0
    fitting.validate_model_state(predictor, metadata)


@pytest.mark.parametrize("seed", [None, True, -1, 2**32])
def test_model_seed_rejects_unseeded_or_invalid_requests(seed):
    with pytest.raises((TypeError, ValueError), match="random_seed"):
        ModelSpec(
            model_id="lasso",
            model_algorithm="lasso_regression",
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

    case, _, _ = _case(tmp_path, "lasso_regression")
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
