from __future__ import annotations

import numpy as np
import pandas as pd

import quant_research_platform.research.predictions.outputs as prediction_outputs
from quant_research_platform.contracts.models import (
    ModelSpec,
    _model_smoke_hyperparameters,
)
from quant_research_platform.contracts.predictions import PredictionSpec


def sort_filter_mode_evaluation_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.sort_values(list(columns)).reset_index(drop=True)


def make_filter_mode_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * 6,
            "asset_id": [f"asset_{idx}" for idx in range(6)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 6,
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label": [0.01, 0.02, 0.03, 0.04, 0.05, np.nan],
            "eligible": [True, True, False, True, True, True],
            "can_enter": [True, True, True, False, True, True],
            "can_exit": [True, True, True, True, False, True],
        }
    )


def make_filter_mode_prediction_output_scores(
    frames: list[pd.DataFrame],
    *,
    prediction_spec: PredictionSpec,
    model_spec: ModelSpec,
    result,
) -> pd.DataFrame:
    build_state = prediction_outputs._PredictionBuildState()
    output = [
        item.value
        for item in prediction_outputs._iter_prediction_output_frames(
            iter(frames),
            spec=prediction_spec,
            input_columns=model_spec.input_columns,
            predictor=prediction_outputs.prediction_callable(
                model_spec.input_columns,
                result.prediction_model,
            ),
            build_state=build_state,
        )
    ]
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["timestamp", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def make_filter_mode_model_hyperparameters(
    model_algorithm: str,
) -> dict[str, object]:
    return _model_smoke_hyperparameters(model_algorithm)
