"""Feature importance extraction for supervised model results."""

from __future__ import annotations

import numpy as np

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)


def linear_feature_importance(
    spec: ModelSpec,
    coefficients: np.ndarray,
) -> dict[str, float]:
    values = np.abs(coefficients[1:])
    if len(values) == 2 * len(spec.input_columns):
        values = values[: len(spec.input_columns)] + values[len(spec.input_columns) :]
    return {
        column: float(abs(value))
        for column, value in zip(spec.input_columns, values, strict=True)
    }


def estimator_feature_importance(
    spec: ModelSpec,
    predictor: object,
) -> dict[str, float] | None:
    if isinstance(predictor, RankEnsemblePredictor):
        weight = 1.0 / len(predictor.columns)
        return {column: weight for column in predictor.columns}
    if isinstance(predictor, ICWeightedRankEnsemble):
        return {
            column: float(abs(weight))
            for column, weight in zip(predictor.columns, predictor.weights, strict=True)
        }
    if hasattr(predictor, "feature_importances_"):
        values = np.asarray(predictor.feature_importances_, dtype="float64")
        if len(values) == 2 * len(spec.input_columns):
            values = (
                values[: len(spec.input_columns)] + values[len(spec.input_columns) :]
            )
        return {
            column: float(value)
            for column, value in zip(spec.input_columns, values, strict=True)
        }
    if hasattr(predictor, "coef_"):
        values = np.abs(np.ravel(np.asarray(predictor.coef_, dtype="float64")))
        if len(values) == 2 * len(spec.input_columns):
            values = (
                values[: len(spec.input_columns)] + values[len(spec.input_columns) :]
            )
        return {
            column: float(abs(value))
            for column, value in zip(spec.input_columns, values, strict=True)
        }
    return None
