"""Saved models load independently of external trial records."""

import shutil

import pandas as pd
import pytest

from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def test_model_load_is_self_contained_without_external_trial(tmp_path):
    inputs = data(tmp_path)
    model = fit(fit_config(tmp_path, inputs))
    shutil.rmtree(model.with_name(model.name + ".trial"))
    assert load_model(model)[0].metadata["rows_after_input_label_dropna"] == 9


def test_published_recipes_with_omitted_defaults_remain_comparable(tmp_path):
    import copy

    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.models import compare_model_facts
    from quant_research_platform.research.models.persisted import save_model
    from tests.helpers.output_frames import make_price_volume_prediction_output_frame

    fitted, facts = load_model(fit(fit_config(tmp_path, data(tmp_path))))
    abbreviated = copy.deepcopy(facts)
    for field in (
        "input_factor_columns",
        "training_weighting",
        "training_scaling",
        "hyperparameters",
        "sample_filter_mode",
    ):
        abbreviated["model"].pop(field)
    saved = save_model(
        tmp_path / "omitted-defaults",
        fitted,
        facts=abbreviated,
        trial_id="omitted-defaults",
    )
    _, restored = load_model(saved)
    compare_model_facts(restored, facts)
    assert restored["model"] != facts["model"]
    assert "training_weighting" not in abbreviated["model"]

    from quant_research_platform.research.predictions.runner import predict

    output = predict(
        {
            "model": str(saved),
            "inputs": {"path": str(tmp_path / "data.parquet"), "spec": input_spec()},
            "output": str(tmp_path / "from-omitted-model"),
            "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
        }
    )
    assert len(read_predictions(output)[0]) == 9

    frame = make_price_volume_prediction_output_frame()
    metadata = copy.deepcopy(frame.attrs["prediction_metadata"])
    expected = copy.deepcopy(metadata["model_facts"])
    metadata["model_facts"]["model"].pop("training_weighting")
    prediction = write_prediction_outputs(
        output_path=tmp_path / "predictions", output_chunks=[frame], metadata=metadata
    )
    _, published = read_predictions(prediction)
    compare_model_facts(published["model_facts"], expected)
    assert published["model_facts"]["model"] != expected["model"]
    assert "training_weighting" not in metadata["model_facts"]["model"]


@pytest.mark.parametrize("input_role", ["factor", "feature"])
def test_prediction_writer_defaults_reach_scoring(tmp_path, input_role):
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
    columns = recipe.pop("input_factor_columns")
    recipe.pop("input_feature_columns")
    recipe[f"input_{input_role}_columns"] = columns
    recipe.pop("sample_filter_mode")
    recipe.pop("hyperparameters")
    metadata.pop("prediction_rows")
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
