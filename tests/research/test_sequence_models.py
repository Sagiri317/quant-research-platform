"""Real CPU fallback training, causal sequences, and native model round trips."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.research.models.fitting import prediction_callable
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
@pytest.mark.parametrize("grouped", [False, True])
def test_sequence_cpu_fallback_boundaries_and_native_roundtrip(
    tmp_path, algorithm, grouped, monkeypatch
):
    import torch

    from quant_research_platform.artifacts.explanations import read_explanation
    from quant_research_platform.research.models.explanations import explain

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    episodes = ["X", "Y", "X", "X", "X", "X", "X", "Y", "X"]
    frame["episode"] = frame.timestamp.dt.day.map(dict(enumerate(episodes, 1)))
    hidden = frame.timestamp.eq("2020-01-05") & frame.asset_id.eq("a")
    frame.loc[hidden, "x"] = 40.0
    frame.loc[
        hidden | (frame.timestamp.eq("2020-01-08") & frame.asset_id.eq("b")),
        "eligible",
    ] = False
    frame.loc[frame.timestamp.eq("2020-01-06"), "x"] = np.nan
    frame.to_parquet(inputs, index=False, row_group_size=2)
    spec = input_spec()
    spec["column_roles"]["episode"] = "context"
    config = fit_config(tmp_path, inputs)
    config["input_spec"] = spec
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={
            "hidden_size": 4,
            "epochs": 1,
            "sequence_length": 4,
            "sequence_group_columns": ["episode"] if grouped else [],
        },
    )
    config["validation"] = {
        "windows": [{"start": "2020-01-04", "end": "2020-01-09"}],
        "cutoff": "2020-01-09",
    }
    fitted = fit(config)
    result, _ = load_model(fitted)
    model = result.prediction_model
    assert result.metadata["resolved_hyperparameters"]["device"] == "cuda"

    # Independent identities isolate contiguous episodes even when group values
    # repeat. Default models instead skip empty rows and retain asset history.
    reference = frame.copy()
    if grouped:
        segments = [0, 1, 2, 2, 2, 3, 4, 5, 6]
        reference["asset_id"] += reference.timestamp.dt.day.map(
            dict(enumerate(segments, 1))
        ).astype(str)
    expected = model.predict(reference)[frame.timestamp.ge("2020-01-04")]
    future = frame.loc[frame.timestamp.ge("2020-01-04")].reset_index(drop=True)
    np.testing.assert_allclose(
        model.predict(future), expected, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    unordered = future.iloc[::-1].assign(
        timestamp=future.iloc[::-1].timestamp.astype(str)
    )
    np.testing.assert_allclose(
        model.predict(unordered), expected[::-1], rtol=1e-6, atol=1e-6, equal_nan=True
    )
    assert model.predict(future.iloc[:0]).shape == (0,)
    changed = future.copy()
    changed.loc[changed.timestamp.ge("2020-01-07"), "x"] += 100
    before = future.timestamp.lt("2020-01-07")
    np.testing.assert_allclose(
        model.predict(changed)[before], expected[before], rtol=1e-6, atol=1e-6
    )
    changed = future.copy()
    changed.loc[changed.asset_id.eq("b"), "x"] += 100
    other = future.asset_id.ne("b")
    np.testing.assert_allclose(
        model.predict(changed)[other], expected[other], rtol=1e-6, atol=1e-6
    )

    transformed_rows = []
    transform = type(model.scaler).transform

    def measured_transform(self, values):
        transformed_rows.append(len(values))
        return transform(self, values)

    with monkeypatch.context() as measured:
        measured.setattr(type(model.scaler), "transform", measured_transform)
        streamed = model.prediction_function()
        actual = np.concatenate(
            [streamed(future.iloc[i : i + 1]) for i in range(len(future))]
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    # Every new observed row is scaled once, plus the initial continuous tail.
    assert sum(transformed_rows) == int(future.x.notna().sum()) + (3 if grouped else 9)
    with pytest.raises(ValueError, match="sequence_prediction_time_not_increasing"):
        streamed(future.iloc[-1:])

    valid = future.x.notna() & future.eligible
    curve = result.metadata["learning_curve"]
    assert curve["sequence_lengths"]["max"] == (1 if grouped else 3)
    assert curve["epochs"][-1]["validation_loss"] == pytest.approx(
        np.mean((expected[valid] - future.loc[valid, "label"].to_numpy()) ** 2),
        rel=1e-6,
    )
    window = {"start": "2020-01-05", "end": "2020-01-09"}
    selected = valid & future.timestamp.ge(window["start"])
    predictions, _ = read_predictions(
        predict(
            {
                "model": str(fitted),
                "inputs": str(inputs),
                "input_spec": spec,
                "output": str(tmp_path / "pred"),
                "prediction_window": window,
            }
        )
    )
    pd.testing.assert_frame_equal(
        predictions[["timestamp", "asset_id"]],
        future.loc[selected, ["timestamp", "asset_id"]].reset_index(drop=True),
        check_dtype=False,
    )
    np.testing.assert_allclose(
        predictions.prediction_score, expected[selected], rtol=1e-6, atol=1e-6
    )
    _, tables = read_explanation(
        explain(
            {
                "model": str(fitted),
                "inputs": str(inputs),
                "input_spec": {**spec, "column_units": {"label": "decimal_return"}},
                "output": str(tmp_path / "explanation"),
                "methods": ["permutation"],
                "window": window,
                "sample_role": "validation",
                "output_space": "model_output",
                "selection": {"max_rows": 100, "max_dates": None},
                "permutation": {
                    "groups": {"x": ["x"]},
                    "metrics": ["mse"],
                    "axis": "within_timestamp",
                    "cutoff": "2020-01-10",
                    "repeats": 1,
                },
            }
        )
    )
    scored = future.loc[selected].copy()
    scored["squared_error"] = (expected[selected] - scored.label) ** 2
    summary = tables["permutation_summary"]
    assert summary.baseline.iloc[0] == pytest.approx(
        scored.groupby("timestamp").squared_error.mean().mean(), rel=1e-6
    )
    assert summary.observation_count.iloc[0] == len(scored)


def test_xgboost_cpu_training_and_native_roundtrip(tmp_path):
    inputs = data(tmp_path)
    config = fit_config(tmp_path, inputs)
    config["model"].update(
        model_algorithm="xgboost_regressor",
        hyperparameters={"device": "cpu", "n_estimators": 3, "max_depth": 2},
    )
    result, _ = load_model(fit(config))
    assert result.prediction_model.estimator.get_params()["device"] == "cpu"
    predict_frame = prediction_callable(
        tuple(result.metadata["input_columns"]), result.prediction_model
    )
    assert np.isfinite(predict_frame(pd.read_parquet(inputs))).all()


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
@pytest.mark.parametrize("length", [1, 3])
def test_explicit_sequence_groups_and_historical_learning_curve(
    tmp_path, algorithm, length
):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["episode"] = (frame.timestamp.dt.day - 1) // 2
    frame.to_parquet(inputs, index=False)
    config = fit_config(tmp_path, inputs)
    config["input_spec"]["column_roles"]["episode"] = "context"
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={
            "device": "cpu",
            "hidden_size": 4,
            "epochs": 2,
            "sequence_length": length,
            "sequence_group_columns": ["episode"],
        },
    )
    config["validation"] = {
        "windows": [{"start": "2020-01-04", "end": "2020-01-06"}],
        "cutoff": "2020-01-06",
    }
    if algorithm == "transformer_regressor":
        config["candidate_parameters"] = config["model"]["hyperparameters"]
        config["model"]["hyperparameters"] = {}
    result, facts = load_model(fit(config))
    assert facts["validation"] == config["validation"]
    curve = result.metadata["learning_curve"]
    assert curve["sequence_lengths"]["max"] == min(length, 2)
    assert result.prediction_model.scaler.mean_ == pytest.approx((2.0,))
    validation = frame.loc[frame.timestamp.between("2020-01-04", "2020-01-06")]
    predicted = result.prediction_model.predict(validation)
    expected_mse = np.mean((predicted - validation.label.to_numpy()) ** 2)
    assert curve["epochs"][-1]["validation_loss"] == pytest.approx(
        expected_mse, rel=1e-6
    )
    changed = validation.copy()
    changed.loc[changed.episode.eq(1), "x"] += 1000
    np.testing.assert_allclose(
        result.prediction_model.predict(changed)[validation.episode.eq(2)],
        predicted[validation.episode.eq(2)],
        rtol=1e-6,
        atol=1e-6,
    )
    config["output"] = str(tmp_path / "overlap")
    config["validation"]["windows"][0]["start"] = "2020-01-03"
    with pytest.raises(ValueError, match="model_validation_training_overlap"):
        fit(config)
