"""Estimator factories for supervised model algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.rank_ensemble import (
    _ICWeightedRankEnsembleFactory,
    _RankEnsemblePredictor,
)


@dataclass(frozen=True)
class _LinearRegressionEstimator:
    coefficient_count: int = 0

    def fit_model_result_from_frames(
        self,
        frames: Iterable[pd.DataFrame],
        spec: ModelSpec,
    ):
        del self
        from quant_research_platform.research.models import fitting

        return fitting.fit_linear_model_result_from_frames(frames, spec)


@dataclass(frozen=True)
class _SingleFactorBaseline:
    column: str
    coefficient_count: int = 1

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, self.column].to_numpy(dtype="float64", copy=False)


def _linear_regression_estimator(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    del spec, params
    return _LinearRegressionEstimator()


def _ridge_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    del spec
    from sklearn.linear_model import Ridge

    return Ridge(**params)


def _lasso_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.linear_model import Lasso

    _set_random_state(params, spec)
    return Lasso(**params)


def _elastic_net_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.linear_model import ElasticNet

    _set_random_state(params, spec)
    return ElasticNet(**params)


def _huber_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    del spec
    from sklearn.linear_model import HuberRegressor

    return HuberRegressor(**params)


def _decision_tree_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.tree import DecisionTreeRegressor

    _set_random_state(params, spec)
    return DecisionTreeRegressor(**params)


def _random_forest_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.ensemble import RandomForestRegressor

    _set_random_state(params, spec)
    return RandomForestRegressor(**params)


def _extra_trees_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.ensemble import ExtraTreesRegressor

    _set_random_state(params, spec)
    return ExtraTreesRegressor(**params)


def _gradient_boosting_regressor(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    from sklearn.ensemble import GradientBoostingRegressor

    _set_random_state(params, spec)
    return GradientBoostingRegressor(**params)


def _hist_gradient_boosting_regressor(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    from sklearn.ensemble import HistGradientBoostingRegressor

    _set_random_state(params, spec)
    return HistGradientBoostingRegressor(**params)


def _xgboost_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from xgboost import XGBRegressor

    _set_random_state(params, spec)
    return XGBRegressor(**params)


def _lightgbm_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from lightgbm import LGBMRegressor

    _set_random_state(params, spec)
    return LGBMRegressor(**params)


def _single_factor_baseline(spec: ModelSpec, params: dict[str, object]) -> object:
    del params
    return _SingleFactorBaseline(spec.input_columns[0])


def _equal_weight_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    del params
    return _RankEnsemblePredictor(spec.input_columns)


def _ic_weighted_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    del params
    return _ICWeightedRankEnsembleFactory(spec.input_columns, spec.label_column)


def _set_random_state(params: dict[str, object], spec: ModelSpec) -> None:
    if spec.random_seed is not None:
        params.setdefault("random_state", spec.random_seed)


__all__ = [
    "_SingleFactorBaseline",
    "_decision_tree_regressor",
    "_elastic_net_estimator",
    "_equal_weight_rank_ensemble",
    "_extra_trees_regressor",
    "_gradient_boosting_regressor",
    "_hist_gradient_boosting_regressor",
    "_huber_regression_estimator",
    "_ic_weighted_rank_ensemble",
    "_lasso_regression_estimator",
    "_lightgbm_regressor",
    "_linear_regression_estimator",
    "_random_forest_regressor",
    "_ridge_regression_estimator",
    "_xgboost_regressor",
]
