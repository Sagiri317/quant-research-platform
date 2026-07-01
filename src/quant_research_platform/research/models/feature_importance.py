"""Feature importance extraction for supervised model results."""

from __future__ import annotations

import numpy as np

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import estimators
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)


def linear_feature_importance(
    spec: ModelSpec,
    coefficients: np.ndarray,
) -> dict[str, float]:
    values = coefficients[1 : len(spec.input_columns) + 1]
    return {
        column: float(abs(value))
        for column, value in zip(spec.input_columns, values, strict=True)
    }


def estimator_feature_importance(
    spec: ModelSpec,
    predictor: object,
) -> dict[str, float] | None:
    if isinstance(predictor, estimators.SingleFactorBaseline):
        return {predictor.column: 1.0}
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
        return {
            column: float(value)
            for column, value in zip(spec.input_columns, values, strict=True)
        }
    if hasattr(predictor, "coef_"):
        values = np.ravel(np.asarray(predictor.coef_, dtype="float64"))
        return {
            column: float(abs(value))
            for column, value in zip(spec.input_columns, values, strict=True)
        }
    return None


__all__ = ["estimator_feature_importance", "linear_feature_importance"]
