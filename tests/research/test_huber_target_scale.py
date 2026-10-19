"""Explicit Huber conditioning preserves economic output units and native facts."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient

from quant_research_platform.artifacts.explanations import read_explanation
from quant_research_platform.contracts.models import (
    ModelSpec,
    model_spec_payload,
    validate_model_facts,
)
from quant_research_platform.research.models.explanations import explain
from quant_research_platform.research.models.fitting import validate_model_state
from quant_research_platform.research.models.persisted import load_model
from tests.research.direct_model_test_support import fit_frame, predict_frame


def spec(scale=10000.0):
    return ModelSpec(
        model_id="conditioned_huber",
        model_algorithm="huber_regression",
        input_factor_columns=("x", "z"),
        target_column="label",
        hyperparameters={"alpha": 0.2, "max_iter": 2000, "tol": 1e-9},
        huber_target_scale=scale,
    )


def frame():
    rng = np.random.default_rng(37)
    x, z, noise = rng.normal(size=(3, 120))
    dates = pd.date_range("2020-01-01", periods=12).repeat(10)
    return pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id=list("abcdefghij") * 12,
            x=x + 10,
            z=z - 5,
            label=(2 * x - z + noise) * 1e-4,
            eligible=True,
            label_available_at=dates,
        )
    )


def test_target_and_regularization_transform_have_the_same_objective_and_gradient():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0.5, 5.0, -0.5, 1.5])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    parameters = np.array([2.0, -1.0, 0.5, 1.0])
    scale = 10000.0
    loss, gradient = _huber_loss_and_gradient(parameters, x, y, 1.35, 0.2, weights)
    # Three zero residuals; one weighted outlier of 2.5, plus 0.2*(2**2+1**2).
    assert loss == pytest.approx(10.0 + 2.0 * (2.0 * 1.35 * 2.5 - 1.35**2) + 1.0)
    np.testing.assert_allclose(
        gradient, [-4.6, -0.4, -5.4, 6.355], rtol=1e-13, atol=1e-13
    )
    scaled_loss, scaled_gradient = _huber_loss_and_gradient(
        parameters * scale, x, y * scale, 1.35, 0.2 / scale, weights
    )
    assert scaled_loss == pytest.approx(scale * loss, rel=1e-13)
    np.testing.assert_allclose(scaled_gradient, gradient, rtol=1e-13, atol=1e-12)


def test_real_fit_inverse_transform_predictions_explanations_and_native_state(
    tmp_path, monkeypatch
):
    data, request = frame(), spec()
    captured = []
    original = HuberRegressor.fit

    def spy(self, matrix, target, **kwargs):
        captured.append(
            (self.alpha, matrix.copy(), target.copy(), kwargs["sample_weight"].copy())
        )
        return original(self, matrix, target, **kwargs)

    monkeypatch.setattr(HuberRegressor, "fit", spy)
    train = data.loc[data.timestamp.le("2020-01-08")]
    case = fit_frame(tmp_path / "fit", train, request)
    assert len(captured) == 1
    alpha, matrix, target, weights = captured[0]
    assert alpha == request.hyperparameters["alpha"] / request.huber_target_scale
    np.testing.assert_allclose(target, train.label.to_numpy() * 10000.0)
    reference = HuberRegressor(**{**request.hyperparameters, "alpha": alpha})
    original(reference, matrix, target, sample_weight=weights)
    estimator = case.result.prediction_model.estimator
    np.testing.assert_allclose(
        estimator.coef_, reference.coef_ / 10000.0, rtol=1e-13, atol=1e-15
    )
    assert estimator.scale_ == pytest.approx(reference.scale_ / 10000.0, rel=1e-13)
    assert estimator.alpha == request.hyperparameters["alpha"]
    assert case.result.metadata["huber_target_scale"] == 10000.0
    assert case.facts["model"]["huber_target_scale"] == 10000.0
    assert case.facts["label_unit"] == "decimal_return"
    reread, facts = load_model(case.root)
    assert facts == case.facts
    window = {"start": "2020-01-09", "end": "2020-01-12"}
    predictions, _, _ = predict_frame(
        tmp_path / "predictions", data, case, window=window
    )
    expected = (
        reference.predict(
            reread.prediction_model.scaler.transform(
                data.loc[data.timestamp.ge("2020-01-09"), ["x", "z"]]
            )
        )
        / 10000.0
    )
    np.testing.assert_allclose(
        predictions.prediction_score, expected, rtol=1e-12, atol=1e-15
    )
    source = tmp_path / "explanation-input.parquet"
    data.to_parquet(source, index=False)
    _, tables = read_explanation(
        explain(
            dict(
                model=str(case.root),
                inputs=str(source),
                input_spec=case.config["input_spec"],
                output=str(tmp_path / "explanation"),
                window=window,
                sample_role="validation",
                methods=["native_importance", "affine"],
                selection={"max_rows": 40, "max_dates": 4, "root_seed": 17},
                output_space="model_output",
            )
        )
    )
    explained = tables["predictions"]
    assert explained.additivity_residual.abs().le(explained.additivity_tolerance).all()
    np.testing.assert_allclose(
        explained.explained_output, explained.prediction_score, rtol=1e-12, atol=1e-15
    )
    broken = deepcopy(case.result.metadata)
    broken["huber_target_scale"] = 1000.0
    with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
        validate_model_state(reread.prediction_model, broken)
    with pytest.raises(ValueError, match="model_facts_metadata_mismatch"):
        validate_model_facts(facts, metadata=broken)


def test_default_scale_retains_existing_native_payload_and_estimator(tmp_path):
    request = spec(1.0)
    assert "huber_target_scale" not in model_spec_payload(request)
    case = fit_frame(tmp_path, frame(), request)
    assert "huber_target_scale" not in case.result.metadata
    assert not hasattr(
        case.result.prediction_model.estimator, "qrp_huber_target_scale_"
    )
    load_model(case.root)


@pytest.mark.parametrize(
    "value", (0.0, -1.0, float("inf"), float("nan"), True, "10000")
)
def test_invalid_target_scales_fail_at_request_boundary(value):
    with pytest.raises(ValueError, match="model_huber_target_scale_invalid"):
        spec(value)


def test_other_algorithms_and_nonconvergence_do_not_silently_change_models(tmp_path):
    with pytest.raises(ValueError, match="model_huber_target_scale_invalid"):
        replace(spec(), model_algorithm="ridge_regression")
    with pytest.raises(ValueError, match="model_fit_non_converged"):
        fit_frame(
            tmp_path,
            frame(),
            replace(spec(), hyperparameters={"max_iter": 1, "tol": 1e-15}),
        )
