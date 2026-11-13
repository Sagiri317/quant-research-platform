"""Sequence boundaries are causal inputs, not an alternate channel for outcomes."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import _model_spec
from quant_research_platform.research.models.production import fit, fit_model
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def _sequence_config(tmp_path, path, groups, *, candidate=False):
    config = fit_config(tmp_path, path)
    parameters = {
        "device": "cpu",
        "hidden_size": 4,
        "epochs": 1,
        "sequence_length": 3,
        "sequence_group_columns": groups,
    }
    config["model"]["model_algorithm"] = "lstm_regressor"
    if candidate:
        config["candidate_parameters"] = parameters
    else:
        config["model"]["hyperparameters"] = parameters
    return config


@pytest.mark.parametrize(
    "group",
    [
        "label",
        "other_label",
        "label_available_at",
        "other_maturity",
        "can_enter",
        "can_exit",
    ],
)
@pytest.mark.parametrize("candidate", [False, True])
def test_fit_rejects_outcome_and_execution_boundaries(tmp_path, group, candidate):
    path = data(tmp_path)
    frame = pd.read_parquet(path).assign(
        other_label=1.0,
        other_maturity=pd.Timestamp("2030-01-01"),
        can_enter=True,
        can_exit=True,
    )
    future = frame.timestamp.dt.day.gt(3)
    frame.loc[future, "label_available_at"] = pd.Timestamp("2030-01-01")
    frame.to_parquet(path, index=False)
    config = _sequence_config(tmp_path, path, [group], candidate=candidate)
    spec = config["inputs"]["spec"]
    spec["column_roles"].update(
        other_label="label", other_maturity="context", can_enter="mask", can_exit="mask"
    )
    spec["label_availability_columns"] = {
        "label": "label_available_at",
        "other_label": "other_maturity",
    }
    with pytest.raises(
        ValueError, match="sequence_(context_uses_future_outcome|boundary_invalid)"
    ):
        fit(config)
    assert not (tmp_path / "linear").exists()


def test_direct_fit_rejects_its_explicit_maturity_as_sequence_input(tmp_path):
    path = data(tmp_path)
    frame = pd.read_parquet(path).rename(columns={"label_available_at": "maturity"})
    config = _sequence_config(tmp_path, path, ["maturity"])
    with pytest.raises(ValueError, match="sequence_context_uses_future_outcome"):
        fit_model(
            (frame,),
            _model_spec(config["model"]),
            fit_windows=config["fit_windows"],
            fit_cutoff=config["fit_cutoff"],
            label_availability_column="maturity",
        )


@pytest.mark.parametrize("candidate", [False, True])
def test_native_sequence_uses_only_its_declared_causal_boundaries(tmp_path, candidate):
    path = data(tmp_path)
    frame = pd.read_parquet(path).assign(
        session_date=pd.Timestamp("2020-01-01"),
        segment_id="continuous",
        listing_date=pd.Timestamp("2010-01-01"),
        unrelated_label=np.nan,
        unrelated_maturity="unknown",
    )
    frame.to_parquet(path, index=False, row_group_size=2)
    # A selected feature may also be a known regime boundary. It is read once.
    groups = ["session_date", "segment_id", "listing_date", "x"]
    config = _sequence_config(tmp_path, path, groups, candidate=candidate)
    spec = config["inputs"]["spec"]
    spec["column_roles"].update(
        session_date="context",
        segment_id="key",
        listing_date="context",
        unrelated_label="label",
        unrelated_maturity="context",
    )
    spec["label_availability_columns"] = {
        "label": "label_available_at",
        "unrelated_label": "unrelated_maturity",
    }
    model = fit(config)
    request = {
        "model": str(model),
        "inputs": {"path": str(path), "spec": spec},
        "output": str(tmp_path / "prediction"),
        "prediction_window": {"start": "2020-01-04", "end": "2020-01-09"},
    }
    expected, _ = read_predictions(predict(request))
    assert len(expected) == 18
    frame["label_available_at"] = frame.label_available_at.astype(str)
    future = frame.timestamp.dt.day.gt(3)
    frame.loc[future, "label"] = np.inf
    frame.loc[future, "label_available_at"] = "not-a-maturity-clock"
    frame["unrelated_label"] = -np.inf
    frame.to_parquet(path, index=False, row_group_size=2)
    actual, _ = read_predictions(
        predict({**request, "output": str(tmp_path / "prediction-again")})
    )
    pd.testing.assert_frame_equal(actual, expected)

    # Frozen candidate parameters, not the empty pre-resolution recipe, own
    # the prediction context. A new source may not redefine it as an outcome.
    for kind in ("label", "maturity", "mask"):
        changed = deepcopy(spec)
        if kind == "label":
            changed["column_roles"]["segment_id"] = "label"
        elif kind == "maturity":
            changed["label_availability_columns"]["unrelated_label"] = "listing_date"
        else:
            changed["column_roles"]["segment_id"] = "mask"
        with pytest.raises(
            ValueError,
            match="sequence_context_(uses_future_outcome|role_invalid)",
        ):
            predict(
                {
                    **request,
                    "inputs": {"path": str(path), "spec": changed},
                    "output": str(tmp_path / ("invalid-" + kind)),
                }
            )


def test_sequence_boundary_cannot_bypass_explicit_input_projection(tmp_path):
    path = data(tmp_path)
    frame = pd.read_parquet(path).assign(segment_id="session")
    frame.to_parquet(path, index=False)
    config = _sequence_config(tmp_path, path, ["segment_id"])
    config["inputs"]["spec"]["column_roles"]["segment_id"] = "context"
    config["inputs"]["columns"] = list(input_spec()["column_roles"])
    with pytest.raises(ValueError, match="model_input_projection_missing"):
        fit(config)
