"""Real fit/save/load/predict/select/refit boundaries."""

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict


def input_spec():
    return {
        "data_id": "synthetic",
        "frequency": "1d",
        "grain": "asset_timestamp",
        "primary_key": ["timestamp", "asset_id"],
        "partition_columns": ["year"],
        "column_roles": {
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "feature",
            "label": "label",
            "eligible": "mask",
            "label_available_at": "context",
        },
    }


def data(tmp_path):
    timestamps = pd.date_range("2020-01-01", periods=9)
    frame = pd.DataFrame(
        [
            (t, a, float(i + 1), float((i + 1) * 2 + t.day), True, t)
            for t in timestamps
            for i, a in enumerate(("a", "b", "c"))
        ],
        columns=[
            "timestamp",
            "asset_id",
            "x",
            "label",
            "eligible",
            "label_available_at",
        ],
    )
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)
    return path


def fit_config(tmp_path, inputs, model_id="linear"):
    return {
        "market_id": "a_share",
        "inputs": str(inputs),
        "output": str(tmp_path / model_id),
        "model": {
            "model_id": model_id,
            "model_algorithm": "linear_regression",
            "input_feature_columns": ["x"],
            "target_column": "label",
            "random_seed": 17,
        },
        "fit_windows": [{"start": "2020-01-01", "end": "2020-01-03"}],
        "fit_cutoff": "2020-01-03",
        "label_unit": "decimal_return",
        "input_spec": input_spec(),
    }


def test_explicit_model_roundtrip_freeze_and_repeated_prediction(tmp_path, monkeypatch):
    inputs = data(tmp_path)
    config = fit_config(tmp_path, inputs)
    model = fit(config)
    fitted, facts = load_model(model)
    assert fitted.metadata["rows_after_input_label_dropna"] == 9
    assert facts["fit_windows"] == config["fit_windows"]
    import quant_research_platform.research.models.fitting as fitting

    original = fitting._fit_model_result
    monkeypatch.setattr(
        fitting, "_fit_model_result", lambda *a, **kw: pytest.fail("predict trained")
    )
    prediction_config = {
        "model": str(model),
        "inputs": str(inputs),
        "output": str(tmp_path / "pred"),
        "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
        "sample_role": "validation",
        "input_spec": input_spec(),
    }
    predictions = predict(prediction_config)
    other = predict({**prediction_config, "output": str(tmp_path / "pred-again")})
    pd.testing.assert_frame_equal(
        read_predictions(predictions)[0], read_predictions(other)[0]
    )
    score_path = score(
        {
            "predictions": [str(predictions)],
            "labels": str(inputs),
            "candidate_ids": ["linear"],
            "scoring_window": prediction_config["prediction_window"],
            "validation_windows": [prediction_config["prediction_window"]],
            "cutoff": "2020-01-06",
            "test_window": {"start": "2020-01-07", "end": "2020-01-09"},
            "selection_group": "quarterly",
            "validation_axis": "cross_sectional",
            "output": str(tmp_path / "scores.json"),
            "label_unit": "decimal_return",
            "label_spec": input_spec(),
        }
    )
    selection = select(
        {
            "scores": str(score_path),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["linear"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "refit_cutoff": "2020-01-06",
        }
    )
    monkeypatch.setattr(fitting, "_fit_model_result", original)
    final = fit(
        {
            **config,
            "output": str(tmp_path / "final"),
            "fit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "fit_cutoff": "2020-01-06",
            "selection": str(selection),
            "test_window": {"start": "2020-01-07", "end": "2020-01-09"},
        }
    )
    test = predict(
        {
            **prediction_config,
            "model": str(final),
            "output": str(tmp_path / "test"),
            "sample_role": "test",
            "prediction_window": {"start": "2020-01-07", "end": "2020-01-09"},
        }
    )
    assert len(read_predictions(test)[0]) == 9


def test_prediction_test_requires_refit_freeze(tmp_path):
    inputs = data(tmp_path)
    model = fit(fit_config(tmp_path, inputs))
    with pytest.raises(ValueError, match="frozen_final_refit"):
        predict(
            {
                "model": str(model),
                "inputs": str(inputs),
                "output": str(tmp_path / "pred"),
                "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
                "sample_role": "test",
                "input_spec": input_spec(),
            }
        )


def test_failed_fit_records_failure_without_model(tmp_path):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["label_available_at"] = pd.Timestamp("2030-01-01")
    frame.to_parquet(inputs, index=False)
    config = fit_config(tmp_path, inputs)
    with pytest.raises(ValueError, match="sample_empty"):
        fit(config)
    assert not Path(config["output"]).exists()
    assert (
        json.loads((tmp_path / "linear.trial" / "terminal.json").read_text())["status"]
        == "failed"
    )


def test_actual_refit_sample_count_resolves_regularization_and_seed_is_explicit(
    tmp_path,
):
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.production import fit_model

    frame = pd.read_parquet(data(tmp_path))
    spec = ModelSpec(
        model_id="ridge-fixed",
        model_algorithm="ridge_regression",
        input_feature_columns=("x",),
        target_column="label",
        random_seed=81,
    )
    options = {
        "fit_cutoff": "2020-01-06",
        "candidate_parameters": {"regularization_strength": 0.25},
    }
    first = fit_model(
        [frame],
        spec,
        fit_windows=[{"start": "2020-01-01", "end": "2020-01-03"}],
        **options,
    )
    refit = fit_model(
        [frame],
        spec,
        fit_windows=[{"start": "2020-01-01", "end": "2020-01-06"}],
        **options,
    )
    assert first.metadata["resolved_hyperparameters"]["alpha"] == 0.25 * 9
    assert refit.metadata["resolved_hyperparameters"]["alpha"] == 0.25 * 18
    assert first.metadata["random_seed"] == refit.metadata["random_seed"] == 81


def test_date_fit_window_contains_intraday_rows_before_exact_cutoff(tmp_path):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["timestamp"] += pd.Timedelta(hours=12)
    frame["label_available_at"] = frame.timestamp
    frame.to_parquet(inputs, index=False)
    config = fit_config(tmp_path, inputs)
    config["fit_cutoff"] = "2020-01-03T18:00:00"
    model = fit(config)
    result, _ = load_model(model)
    assert result.metadata["rows_after_input_label_dropna"] == 9
    assert pd.Timestamp(result.metadata["train_end"]) == pd.Timestamp(
        "2020-01-03T12:00:00"
    )
