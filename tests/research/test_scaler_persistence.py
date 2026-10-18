"""Native saved models preserve fitted floating moments and reject drift."""

import copy
import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import (
    ModelSpec,
    validate_model_metadata_core,
)
from quant_research_platform.research.models.persisted import load_model
from tests.research.direct_model_test_support import fit_frame


def _case(tmp_path):
    stamps = pd.date_range("2020-01-01", periods=3).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": ["a", "b", "c"] * 3,
            "x": [-2.0, 1.0, 2.0] * 3,
            "label": [1.0, 2.0, 3.0] * 3,
            "label_available_at": stamps,
            "eligible": True,
        }
    )
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=11,
    )
    return fit_frame(tmp_path, frame, spec)


@pytest.mark.parametrize("member", ("metadata", "pickle"))
def test_native_models_reject_malformed_or_drifting_scaler_moments(tmp_path, member):
    case = _case(tmp_path)
    if member == "metadata":
        path = case.root / "model.json"
        payload = json.loads(path.read_text())
        payload["model_metadata"]["scaler"]["weighted_mean"][0] += 1.0
        path.write_text(json.dumps(payload))
    else:
        path = case.root / "predictor.pkl"
        predictor = pickle.loads(path.read_bytes())
        predictor = replace(
            predictor,
            scaler=replace(
                predictor.scaler,
                mean_=(predictor.scaler.mean_[0] + 1.0, *predictor.scaler.mean_[1:]),
            ),
        )
        path.write_bytes(pickle.dumps(predictor))
    with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
        load_model(case.root)


def test_native_fit_preserves_frozen_moments_and_nonbinary_center(tmp_path):
    case = _case(tmp_path)
    metadata = case.result.metadata
    for bad in [None, True, "0", [0, 1], np.nan, np.inf]:
        changed = copy.deepcopy(metadata)
        changed["scaler"]["weighted_mean"][0] = bad
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            validate_model_metadata_core(changed)
    predictor = case.result.prediction_model
    scaler = predictor.scaler
    assert scaler.mean_[0] == pytest.approx(1 / 3, rel=1e-14)
    assert case.result.metadata["scaler"]["weighted_mean"] == list(scaler.mean_)
    assert scaler.transform(pd.DataFrame({"x": [0.0]})).x.iloc[0] == pytest.approx(
        -(1 / 3) / scaler.scale_[0], rel=1e-14
    )
    restored, _ = load_model(case.root)
    assert restored.prediction_model.scaler.to_payload() == scaler.to_payload()
    np.testing.assert_allclose(
        restored.prediction_model.predict(case.frame[["x"]]),
        predictor.predict(case.frame[["x"]]),
        rtol=1e-13,
        atol=1e-14,
    )
