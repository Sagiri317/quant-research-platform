"""Rank learning, native state, and economic calibration remain training-only."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.rank_ensemble import _label_calibration
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import fit_frame


@pytest.mark.parametrize(
    "algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble", "linear_regression"],
)
@pytest.mark.parametrize("assets", [1, 3])
def test_native_crypto_state_predicts_without_relearning(
    tmp_path, algorithm, assets, monkeypatch
):
    if assets == 1 and "rank" in algorithm:
        assets = 3
    stamps = pd.date_range("2020-01-01", periods=6).repeat(assets)
    x = (
        np.tile(np.arange(assets, dtype=float), 6)
        + np.repeat(np.arange(6), assets) * 0.1
    )
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": [str(i) for i in range(assets)] * 6,
            "x": x,
            "label": x * 0.03,
            "eligible": True,
            "label_available_at": stamps,
        }
    )
    spec = ModelSpec(
        model_id=algorithm,
        model_algorithm=algorithm,
        input_factor_columns=("x",),
        target_column="label",
        random_seed=19,
    )
    case = fit_frame(tmp_path, frame, spec)
    result, facts = load_model(case.root)
    fitting.validate_model_state(result.prediction_model, result.metadata)
    future = frame.assign(
        timestamp=frame.timestamp + pd.Timedelta(days=8),
        label=9999.0,
        label_available_at=pd.Timestamp("2030-01-01"),
    )
    future_path = tmp_path / "future.parquet"
    future.to_parquet(future_path, index=False)

    def fail(*a, **kw):
        pytest.fail("prediction learned state")

    monkeypatch.setattr(fitting, "_fit_model_result", fail)
    output = predict(
        {
            "model": str(case.root),
            "inputs": str(future_path),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "prediction"),
            "prediction_window": {"start": "2020-01-09", "end": "2020-01-14"},
            "sample_role": "validation",
        }
    )
    predictions, metadata = read_predictions(output)
    np.testing.assert_array_equal(
        predictions.prediction_score,
        fitting.prediction_callable(spec.input_columns, result.prediction_model)(
            future
        ),
    )
    if "rank" in algorithm:
        calibration = result.metadata["rank_parameters"]["label_calibration"]
        raw = fitting.prediction_callable(spec.input_columns, result.prediction_model)(
            frame
        )
        expected = _label_calibration(
            raw, frame.label.to_numpy(), equal_timestamp_sample_weights(frame.timestamp)
        )
        assert calibration == list(expected)
        assert result.prediction_model.label_calibration == expected
        assert metadata["model_metadata"]["rank_parameters"][
            "label_calibration"
        ] == list(expected)


def test_rank_retains_missing_inputs_and_only_defined_ic_columns(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b", "c"],
            "x": [1.0, 2.0, 3.0],
            "undefined": [1.0, np.nan, 1.0],
            "label": [1.0, 2.0, 3.0],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-01"),
        }
    )
    spec = ModelSpec(
        model_id="rank",
        model_algorithm="ic_weighted_rank_ensemble",
        input_factor_columns=("x", "undefined"),
        target_column="label",
        random_seed=1,
    )
    result = fit_frame(tmp_path, frame, spec).result
    assert result.metadata["input_columns"] == ["x", "undefined"]
    assert result.prediction_model.columns == ("x", "undefined")
    assert result.prediction_model.weights == (1.0, 0.0)
    assert result.metadata["rank_parameters"]["columns"] == ["x", "undefined"]
    future = pd.DataFrame({"x": [np.nan, np.nan], "undefined": [1.0, np.nan]})
    scores = result.prediction_model.predict(future)
    assert scores[0] == 0.0 and np.isnan(scores[1])


@pytest.mark.parametrize("field", ["fit_cutoff", "train_start", "train_end"])
def test_model_time_metadata_rejects_nat(tmp_path, field):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [1.0, 2.0],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-01"),
        }
    )
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=1,
    )
    case = fit_frame(tmp_path, frame, spec)
    path = case.root / "model.json"
    value = json.loads(path.read_text())
    value["facts" if field == "fit_cutoff" else "model_metadata"][field] = "NaT"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_model(case.root)
