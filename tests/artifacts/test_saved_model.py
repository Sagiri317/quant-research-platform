"""Saved models load independently of external trial records."""

import shutil

import pandas as pd
import pytest

from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from tests.correctness.test_direct_model_operations import data, fit_config


def test_model_load_is_self_contained_without_external_trial(tmp_path):
    inputs = data(tmp_path)
    model = fit(fit_config(tmp_path, inputs))
    shutil.rmtree(model.with_name(model.name + ".trial"))
    assert load_model(model)[0].metadata["rows_after_input_label_dropna"] == 9


@pytest.mark.parametrize("input_role", ["factor", "feature"])
def test_prediction_roles_reach_scoring(tmp_path, input_role):
    import copy
    import json

    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.research.models.trial_selection import score
    from tests.helpers.output_frames import make_price_volume_prediction_output_frame

    metadata = copy.deepcopy(
        make_price_volume_prediction_output_frame().attrs["prediction_metadata"]
    )
    recipe = metadata["model_facts"]["model"]
    columns = recipe["input_factor_columns"]
    recipe["input_factor_columns"] = []
    recipe["input_feature_columns"] = []
    recipe[f"input_{input_role}_columns"] = columns
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-03"] * 3),
            "asset_id": list("abc"),
            "prediction_score": [0.1, 0.2, 0.3],
            "prediction_rank_score": [-1 / 3, 0.0, 1 / 3],
        }
    )
    prediction = write_prediction_outputs(
        output_path=tmp_path / "prediction", output_chunks=[frame], metadata=metadata
    )
    label = recipe["target_column"]
    availability = metadata["model_metadata"]["label_availability_column"]
    labels = frame[["timestamp", "asset_id"]].assign(
        **{label: [0.1, 0.2, 0.3], availability: frame.timestamp, "eligible": True}
    )
    labels.to_parquet(tmp_path / "labels.parquet", index=False)
    scored = score(
        {
            "predictions": [str(prediction)],
            "candidate_ids": [metadata["model_facts"]["model"]["model_id"]],
            "labels": {
                "path": str(tmp_path / "labels.parquet"),
                "spec": {
                    "data_id": "labels",
                    "frequency": "1d",
                    "grain": "asset_timestamp",
                    "primary_key": ["timestamp", "asset_id"],
                    "partition_columns": [],
                    "column_roles": {
                        "timestamp": "timestamp",
                        "asset_id": "key",
                        "eligible": "mask",
                        label: "label",
                        availability: "context",
                    },
                    "column_units": {label: "decimal_return"},
                },
            },
            "validation_windows": [metadata["prediction_window"]],
            "scoring_window": metadata["prediction_window"],
            "cutoff": "2014-01-03",
            "selection_group": "global",
            "validation_axis": "cross_sectional",
            "output": str(tmp_path / "scores.json"),
        }
    )
    result = json.loads(scored.read_text())
    assert result["common_rows"] == 3
    assert result["metrics"][0]["validation_rankic"] == pytest.approx(1.0)
