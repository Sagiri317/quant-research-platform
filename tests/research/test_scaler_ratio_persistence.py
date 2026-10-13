"""Native saved models keep exact rational scaler offsets and reject drift."""

import copy
import json
import pickle
from dataclasses import replace
from fractions import Fraction

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
            "x": [1e200, 1.0, -1e200] * 3,
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
def test_native_models_reject_old_float_scaler_offsets(tmp_path, member):
    case = _case(tmp_path)
    metadata = case.result.metadata
    for bad in [0.0, [0, 2], [1, 0], [1, -2], [True, 1], [1], [2, 4]]:
        changed = copy.deepcopy(metadata)
        changed["scaler"]["weighted_mean_offset"][0] = bad
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            validate_model_metadata_core(changed)
    if member == "metadata":
        path = case.root / "model.json"
        payload = json.loads(path.read_text())
        payload["model_metadata"]["scaler"]["weighted_mean_offset"] = [
            float(Fraction(*pair))
            for pair in payload["model_metadata"]["scaler"]["weighted_mean_offset"]
        ]
        path.write_text(json.dumps(payload))
    else:
        path = case.root / "predictor.pkl"
        predictor = pickle.loads(path.read_bytes())
        predictor = replace(
            predictor,
            scaler=replace(
                predictor.scaler,
                offset_=tuple(float(value) for value in predictor.scaler.offset_),
            ),
        )
        path.write_bytes(pickle.dumps(predictor))
    with pytest.raises(
        ValueError, match="invalid_model_metadata|model_state_metadata_mismatch"
    ):
        load_model(case.root)


def test_native_fit_preserves_nonbinary_center(tmp_path):
    case = _case(tmp_path)
    predictor = case.result.prediction_model
    scaler = predictor.scaler
    assert Fraction(scaler.origin_[0]) + scaler.offset_[0] == Fraction(1, 3)
    assert (
        Fraction(*case.result.metadata["scaler"]["weighted_mean_offset"][0])
        == scaler.offset_[0]
    )
    assert scaler.transform(pd.DataFrame({"x": [0.0]})).x.iloc[0] == float(
        -Fraction(1, 3) / Fraction(scaler.scale_[0])
    )
    assert np.isfinite(predictor.predict(case.frame[["x"]])).all()
