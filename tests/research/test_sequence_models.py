"""Real CUDA training, causal sequences and native model round trips."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
def test_cuda_sequence_roundtrip_causality_and_asset_isolation(tmp_path, algorithm):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware required for the real training path")
    inputs = data(tmp_path)
    original = pd.read_parquet(inputs)
    hidden = original.timestamp.eq("2020-01-06") & original.asset_id.eq("a")
    original.loc[hidden, "x"] = 40.0
    original.loc[
        hidden | (original.timestamp.eq("2020-01-08") & original.asset_id.eq("b")),
        "eligible",
    ] = False
    original.to_parquet(inputs, index=False)
    config = fit_config(tmp_path, inputs)
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={"hidden_size": 8, "epochs": 2, "sequence_length": 3},
    )
    fitted = fit(config)
    result, _ = load_model(fitted)
    model = result.prediction_model
    frame = pd.read_parquet(inputs)
    frame = frame.loc[frame.timestamp.ge("2020-01-04")].reset_index(drop=True)
    expected = model.predict(frame)
    changed = frame.copy()
    changed.loc[changed.timestamp.ge("2020-01-07"), "x"] += 100
    before = frame.timestamp.lt("2020-01-07")
    np.testing.assert_allclose(
        model.predict(changed)[before], expected[before], rtol=1e-6, atol=1e-6
    )
    changed = frame.copy()
    changed.loc[changed.asset_id.eq("b"), "x"] += 100
    other = frame.asset_id.ne("b")
    np.testing.assert_allclose(
        model.predict(changed)[other], expected[other], rtol=1e-6, atol=1e-6
    )
    stream = model.prediction_function()
    streamed = np.concatenate(
        [stream(part) for _, part in frame.groupby("timestamp", sort=True)]
    )
    np.testing.assert_allclose(streamed, expected, rtol=1e-6, atol=1e-6)
    output = predict(
        {
            "model": str(fitted),
            "inputs": str(inputs),
            "output": str(tmp_path / "pred"),
            "prediction_window": {"start": "2020-01-07", "end": "2020-01-09"},
            "input_spec": input_spec(),
        }
    )
    predictions, _ = read_predictions(output)
    np.testing.assert_allclose(
        predictions.prediction_score,
        expected[~before & frame.eligible],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.metadata["resolved_hyperparameters"]["device"] == "cuda"
    from quant_research_platform.artifacts.explanations import read_explanation
    from quant_research_platform.research.models.explanations import explain

    _, tables = read_explanation(
        explain(
            {
                "model": str(fitted),
                "inputs": str(inputs),
                "input_spec": {
                    **input_spec(),
                    "column_units": {"label": "decimal_return"},
                },
                "output": str(tmp_path / "explanation"),
                "methods": ["permutation"],
                "window": {"start": "2020-01-07", "end": "2020-01-09"},
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
    scored = frame.loc[~before & frame.eligible].copy()
    scored["squared_error"] = (expected[~before & frame.eligible] - scored.label) ** 2
    expected_mse = scored.groupby("timestamp").squared_error.mean().mean()
    assert tables["permutation_summary"].baseline.iloc[0] == pytest.approx(
        expected_mse, rel=1e-5
    )
    assert tables["permutation_summary"].observation_count.iloc[0] == len(scored)


@pytest.mark.parametrize(
    "algorithm", ["xgboost_regressor", "lstm_regressor", "transformer_regressor"]
)
def test_cuda_models_reject_cpu_configuration(algorithm):
    with pytest.raises(ValueError, match="model_cuda_required"):
        ModelSpec(
            model_id="gpu",
            model_algorithm=algorithm,
            input_feature_columns=("x",),
            target_column="y",
            hyperparameters={"device": "cpu"},
        )
