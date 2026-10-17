"""Undefined interventions stay local without changing comparison cohorts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.explanations import read_explanation
from quant_research_platform.research.models.explanations import (
    _output,
    _permutation,
    explain,
)
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
)
from tests.research.test_model_explanations import _case


def _example():
    scaler = ModelMatrixScaler.fit(
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}), missing="impute"
    )
    result = SimpleNamespace(
        prediction_model=LinearScaledPredictor((0.0, 1.0, 1.0, 0.0, 0.0), scaler),
        metadata={
            "input_columns": ["a", "b"],
            "target_column": "label",
            "model_algorithm": "linear_regression",
        },
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": ["x", "y"],
            "a": [1.0, np.nan],
            "b": [np.nan, 2.0],
            "label": [1.0, 2.0],
            "available": pd.Timestamp("2024-01-03"),
        }
    )
    config = {
        "metrics": ["mse", "ic", "rankic"],
        "cutoff": "2024-01-04",
        "axis": "within_timestamp",
        "groups": {"a": ["a"], "b": ["b"]},
        "repeats": 5,
        "root_seed": 42,
    }
    return result, frame, config


def test_missing_intervention_yields_undefined_common_period_not_failure():
    result, frame, config = _example()
    assert np.isfinite(_output(result, frame, "model_output")[1]).all()
    repeats, summary, details = _permutation(
        result,
        {"label_unit": "decimal_return"},
        frame,
        np.arange(2),
        "available",
        config,
        "model_output",
    )
    assert repeats.status.eq("UNDEFINED").all()
    assert repeats.reason.eq("no_common_defined_periods").all()
    assert repeats.observation_count.eq(0).all()
    assert summary.status.eq("UNDEFINED").all()
    assert details["common_metric_periods"] == {"mse": [], "ic": [], "rankic": []}
    assert details["undefined_interventions"]
    assert all(
        row["reason"] == "all_model_inputs_missing"
        for row in details["undefined_interventions"]
    )


def test_baseline_nonfinite_remains_an_error():
    result, frame, config = _example()
    frame.loc[0, ["a", "b"]] = np.nan
    with pytest.raises(ValueError, match="explanation_prediction_nonfinite"):
        _permutation(
            result,
            {"label_unit": "decimal_return"},
            frame,
            np.arange(2),
            "available",
            config,
            "model_output",
        )


def test_bundled_affine_and_importance_survive_missing_interventions(tmp_path):
    case, frame, config = _case(tmp_path)
    future = frame.timestamp.gt("2020-01-05")
    alternating = np.arange(len(frame)) % 2 == 0
    frame.loc[future & alternating, "x"] = np.nan
    frame.loc[future & ~alternating, "y"] = np.nan
    frame.to_parquet(config["inputs"], index=False)
    config["methods"].append("permutation")
    config["permutation"] = {
        "metrics": ["mse"],
        "cutoff": "2020-01-11",
        "axis": "within_timestamp",
        "groups": {"x": ["x"]},
        "repeats": 5,
        "root_seed": 42,
    }
    metadata, tables = read_explanation(explain(config))
    assert tables["predictions"].status.eq("DEFINED").all()
    assert tables["native_importance"].status.eq("DEFINED").all()
    assert tables["permutation_repeats"].status.eq("UNDEFINED").all()
    assert metadata["permutation"]["undefined_interventions"]


def test_retained_period_uses_identical_full_cohort_for_every_intervention():
    result, frame, config = _example()
    clean = frame.copy().assign(
        timestamp=pd.Timestamp("2024-01-03"), a=[1.0, 2.0], b=[3.0, 4.0]
    )
    frame = pd.concat([frame, clean], ignore_index=True)
    config["metrics"] = ["mse"]
    repeats, summary, details = _permutation(
        result,
        {"label_unit": "decimal_return"},
        frame,
        np.arange(4),
        "available",
        config,
        "model_output",
    )
    assert repeats.status.eq("DEFINED").all()
    assert repeats.observation_count.eq(2).all()
    assert repeats.period_count.eq(1).all()
    assert repeats.baseline.nunique() == 1
    baseline = _output(result, clean, "model_output")[1]
    np.testing.assert_allclose(
        repeats.baseline, np.mean((baseline - clean.label.to_numpy()) ** 2)
    )
    assert details["common_metric_periods"]["mse"] == ["2024-01-03 00:00:00"]
    assert all(
        row["period"] == "2024-01-02 00:00:00"
        for row in details["undefined_interventions"]
    )


def test_unexpected_intervention_failure_is_not_hidden(monkeypatch):
    result, frame, config = _example()
    original = LinearScaledPredictor.predict
    calls = 0

    def broken(self, data):
        nonlocal calls
        calls += 1
        scores = original(self, data)
        if calls > 1:
            scores[data.notna().any(axis=1).to_numpy()] = np.nan
        return scores

    monkeypatch.setattr(LinearScaledPredictor, "predict", broken)
    with pytest.raises(ValueError, match="explanation_prediction_nonfinite"):
        _permutation(
            result,
            {"label_unit": "decimal_return"},
            frame,
            np.arange(2),
            "available",
            config,
            "model_output",
        )
