"""Real early stopping, best-state restoration, and independent mature validation."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import (
    DEEPLOB_INPUT_COLUMNS,
    ModelSpec,
    validate_model_facts,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config


def stopping_config(tmp_path, algorithm):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["label"] = 0.02 * (frame.x - 2) + 0.01 * frame.timestamp.dt.day
    validation = frame.timestamp.between("2020-01-04", "2020-01-06")
    frame.loc[validation, "label"] = -0.1
    frame.loc[validation, "label_available_at"] += pd.Timedelta(days=1)
    late = frame.timestamp.eq("2020-01-06") & frame.asset_id.eq("a")
    frame.loc[late, "label_available_at"] = pd.Timestamp("2020-01-10")
    config = fit_config(tmp_path, inputs, model_id=algorithm)
    parameters = {
        "device": "cpu",
        "early_stopping_rounds": 1,
    }
    if algorithm == "xgboost_regressor":
        parameters.update(n_estimators=8, max_depth=2, learning_rate=0.3, n_jobs=1)
    else:
        parameters.update(
            hidden_size=4,
            epochs=8,
            learning_rate=0.05,
            sequence_length=2,
            batch_size=64,
        )
    if algorithm.startswith("deeplob"):
        parameters.update(conv_channels=2, inception_channels=2)
        if algorithm == "deeplob_classifier":
            parameters["class_threshold"] = 0.005
        for level in range(1, 11):
            frame[f"ask_px{level}"] = 100 + frame.x * 0.02 + level * 0.01
            frame[f"bid_px{level}"] = 100 + frame.x * 0.02 - level * 0.01
            frame[f"ask_vol{level}"] = 50 + frame.x + level
            frame[f"bid_vol{level}"] = 50 + frame.x + level
        frame["session_date"] = frame.timestamp.dt.normalize()
        frame["segment_id"] = 0
        roles = config["input_spec"]["column_roles"]
        roles.update({column: "feature" for column in DEEPLOB_INPUT_COLUMNS})
        roles.update(session_date="context", segment_id="context")
        config["model"]["input_feature_columns"] = list(DEEPLOB_INPUT_COLUMNS)
    frame.to_parquet(inputs, index=False)
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters=parameters,
    )
    config["validation"] = {
        "windows": [{"start": "2020-01-04", "end": "2020-01-06"}],
        "cutoff": "2020-01-07",
    }
    return config, frame, late


@pytest.mark.parametrize(
    "algorithm",
    [
        "lstm_regressor",
        "transformer_regressor",
        "deeplob_regressor",
        "deeplob_classifier",
        "xgboost_regressor",
    ],
)
def test_best_state_native_roundtrip_and_validation_maturity(tmp_path, algorithm):
    config, frame, late = stopping_config(tmp_path, algorithm)
    model = fit(config)
    result, facts = load_model(model)
    stopped = result.metadata["early_stopping"]
    best = stopped["best_iteration"]
    assert 0 <= best < stopped["iterations"] <= 8
    assert stopped["validation_rows"] == 8
    assert pd.Timestamp(stopped["validation_start"]) == pd.Timestamp("2020-01-04")
    assert pd.Timestamp(stopped["validation_end"]) == pd.Timestamp("2020-01-06")
    assert pd.Timestamp(stopped["label_available_max"]) == pd.Timestamp("2020-01-07")
    assert pd.Timestamp(result.metadata["label_available_max"]) == pd.Timestamp(
        "2020-01-03"
    )
    if algorithm == "xgboost_regressor":
        estimator = result.prediction_model.estimator
        assert stopped["iterations"] < 8
        assert estimator.get_booster().num_boosted_rounds() == best + 1
        losses = estimator.evals_result()["validation_0"]["rmse"]
        assert best == int(np.argmin(losses))
    else:
        curve = result.metadata["learning_curve"]["epochs"]
        assert len(curve) == stopped["iterations"]
        assert best == int(np.argmin([point["validation_loss"] for point in curve]))

    # A separately trained fixed-budget model is an independent checkpoint oracle.
    fixed = deepcopy(config)
    fixed["output"] = str(tmp_path / "fixed")
    fixed.pop("validation")
    parameters = fixed["model"]["hyperparameters"]
    parameters.pop("early_stopping_rounds")
    parameters["n_estimators" if algorithm == "xgboost_regressor" else "epochs"] = (
        best + 1
    )
    reference, _ = load_model(fit(fixed))
    future = frame.loc[frame.timestamp.ge("2020-01-08")]
    model_input = future.loc[:, ["x"]] if algorithm == "xgboost_regressor" else future
    np.testing.assert_allclose(
        result.prediction_model.predict(model_input),
        reference.prediction_model.predict(model_input),
        rtol=2e-6,
        atol=2e-7,
    )
    assert (
        result.prediction_model.scaler.to_payload()
        == reference.prediction_model.scaler.to_payload()
    )

    # The still-immature validation label cannot select the checkpoint.
    frame.loc[late, "label"] = 0.8
    frame.to_parquet(config["inputs"], index=False)
    changed = {**config, "output": str(tmp_path / "late-changed")}
    unchanged, _ = load_model(fit(changed))
    assert unchanged.metadata["early_stopping"] == stopped
    np.testing.assert_allclose(
        unchanged.prediction_model.predict(model_input),
        result.prediction_model.predict(model_input),
        rtol=2e-6,
        atol=2e-7,
    )

    request = {
        "model": str(model),
        "inputs": config["inputs"],
        "output": str(tmp_path / "prediction"),
        "input_spec": config["input_spec"],
        "prediction_window": {"start": "2020-01-07", "end": "2020-01-09"},
        "sample_role": "test",
    }
    with pytest.raises(ValueError, match="prediction_must_follow_validation_cutoff"):
        predict(request)
    request["prediction_window"]["start"] = "2020-01-08"
    predictions, metadata = read_predictions(predict(request))
    assert len(predictions) == 6
    assert metadata["model_metadata"]["early_stopping"] == stopped
    assert metadata["model_facts"]["validation"] == config["validation"]

    tampered = deepcopy(result.metadata)
    tampered["early_stopping"]["label_available_max"] = "2020-01-08"
    with pytest.raises(ValueError, match="invalid_model_validation_maturity"):
        validate_model_facts(facts, tampered)


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "xgboost_regressor"])
def test_stopping_requires_independent_observed_validation(tmp_path, algorithm):
    config, frame, _ = stopping_config(tmp_path, algorithm)
    no_validation = deepcopy(config)
    no_validation.pop("validation")
    with pytest.raises(ValueError, match="early_stopping_validation_required"):
        fit(no_validation)
    overlap = deepcopy(config)
    overlap["output"] = str(tmp_path / "overlap")
    overlap["validation"]["windows"][0]["start"] = "2020-01-03"
    with pytest.raises(ValueError, match="model_validation_training_overlap"):
        fit(overlap)
    frame.loc[frame.timestamp.ge("2020-01-04"), "label_available_at"] = pd.NaT
    frame.to_parquet(config["inputs"], index=False)
    config["output"] = str(tmp_path / "unknown")
    with pytest.raises(ValueError, match="model_validation_sample_empty"):
        fit(config)


@pytest.mark.parametrize(
    "algorithm,patience",
    [
        ("lstm_regressor", 0),
        ("lstm_regressor", True),
        ("xgboost_regressor", -1),
        ("xgboost_regressor", 1.5),
        ("ridge_regression", 2),
    ],
)
def test_early_stopping_parameter_contract(algorithm, patience):
    with pytest.raises(ValueError, match="early_stopping_rounds_invalid"):
        ModelSpec(
            model_id="invalid",
            model_algorithm=algorithm,
            input_feature_columns=("x",),
            target_column="label",
            hyperparameters={"early_stopping_rounds": patience},
        )
