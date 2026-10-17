"""A declared model's empty domain is explicit; comparison may not drop it."""

import json

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import fit_frame


def test_empty_family_is_native_empty_without_fabricated_prediction(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b", "c"],
            "x": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-01"),
        }
    )
    spec = ModelSpec(
        model_id="x",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=7,
    )
    case = fit_frame(tmp_path, frame, spec)
    future = frame.assign(timestamp=pd.Timestamp("2020-01-02"), x=np.nan)
    path = tmp_path / "future.parquet"
    future.to_parquet(path, index=False)
    window = {"start": "2020-01-02", "end": "2020-01-02"}
    output = predict(
        {
            "model": str(case.root),
            "inputs": str(path),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    empty, metadata = read_predictions(output)
    assert empty.empty and list(empty) == [
        "timestamp",
        "asset_id",
        "prediction_score",
        "prediction_rank_score",
    ]
    assert (
        metadata["status"] == "empty"
        and metadata["prediction_rows"] == 0
        and metadata["effective_prediction_end"] is None
    )
    scored = score(
        {
            "predictions": [str(output)],
            "candidate_ids": ["x"],
            "labels": str(path),
            "label_spec": case.config["input_spec"],
            "validation_windows": [window],
            "scoring_window": window,
            "cutoff": "2020-01-02",
            "selection_group": "global",
            "validation_axis": "cross_sectional",
            "label_unit": "decimal_return",
            "output": str(tmp_path / "scores.json"),
        }
    )
    result = json.loads(scored.read_text())
    assert result["common_rows"] == 0
    assert result["metrics"][0]["validation_mse"] is None
    selected = select(
        {
            "scores": str(scored),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_mse",
            "selection_direction": "minimize",
            "tie_break_order": ["x"],
            "refit_windows": case.config["fit_windows"],
            "refit_cutoff": "2020-01-02",
        }
    )
    assert json.loads(selected.read_text())["selected_model_id"] is None
