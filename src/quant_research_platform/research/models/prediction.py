"""Supervised model prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.rank_ensemble import (
    _ICWeightedRankEnsemble,
    _RankEnsemblePredictor,
)


def predict_model_frame(
    frame: pd.DataFrame,
    spec: ModelSpec,
    result: object,
) -> pd.DataFrame:
    required = set(spec.input_columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    output = frame.copy()
    output["prediction_score"] = _predict_scores(
        frame,
        coefficients=result.coefficients,
        input_columns=spec.input_columns,
        prediction_model=result.prediction_model,
    )
    return output


def _predict_scores(
    frame: pd.DataFrame,
    *,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    prediction_model: object | None,
) -> np.ndarray:
    if prediction_model is not None:
        predicted = prediction_model.predict(
            _prediction_model_input_frame(
                frame,
                input_columns=input_columns,
                prediction_model=prediction_model,
            )
        )
        return np.asarray(predicted, dtype="float64")
    x = frame.loc[:, input_columns].to_numpy(dtype="float64", copy=False)
    return np.asarray(float(coefficients[0]) + x @ coefficients[1:])


def _prediction_model_input_frame(
    frame: pd.DataFrame,
    *,
    input_columns: tuple[str, ...],
    prediction_model: object,
) -> pd.DataFrame:
    columns = input_columns
    if isinstance(prediction_model, (_RankEnsemblePredictor, _ICWeightedRankEnsemble)):
        columns = tuple(dict.fromkeys(("timestamp", *input_columns)))
    return frame.loc[:, list(columns)]
