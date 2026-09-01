"""Estimator factories for supervised model algorithms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.rank_ensemble import (
    EqualWeightRankEnsembleFactory,
    ICWeightedRankEnsembleFactory,
)


@dataclass(frozen=True)
class _LinearRegressionEstimator:
    coefficient_count: int = 0


@dataclass(frozen=True)
class SingleFactorBaseline:
    column: str
    coefficient_count: int = 1

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, self.column].to_numpy(dtype="float64", copy=False)


def linear_regression_estimator(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    del spec, params
    return _LinearRegressionEstimator()


def ridge_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    del spec
    from sklearn.linear_model import Ridge

    return Ridge(**params)


def lasso_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.linear_model import Lasso

    _set_random_state(params, spec)
    return Lasso(**params)


def elastic_net_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.linear_model import ElasticNet

    _set_random_state(params, spec)
    return ElasticNet(**params)


def huber_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    del spec
    from sklearn.linear_model import HuberRegressor

    return HuberRegressor(**params)


def decision_tree_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.tree import DecisionTreeRegressor

    _set_random_state(params, spec)
    return DecisionTreeRegressor(**params)


def random_forest_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.ensemble import RandomForestRegressor

    _set_random_state(params, spec)
    return RandomForestRegressor(**params)


def extra_trees_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.ensemble import ExtraTreesRegressor

    _set_random_state(params, spec)
    return ExtraTreesRegressor(**params)


def gradient_boosting_regressor(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    from sklearn.ensemble import GradientBoostingRegressor

    _set_random_state(params, spec)
    return GradientBoostingRegressor(**params)


def hist_gradient_boosting_regressor(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    from sklearn.ensemble import HistGradientBoostingRegressor

    _set_random_state(params, spec)
    return HistGradientBoostingRegressor(**params)


def xgboost_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from xgboost import XGBRegressor

    _set_random_state(params, spec)
    return XGBRegressor(**params)


def lightgbm_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from lightgbm import LGBMRegressor

    _set_random_state(params, spec)
    return LGBMRegressor(**params)


def single_factor_baseline(spec: ModelSpec, params: dict[str, object]) -> object:
    del params
    return SingleFactorBaseline(spec.input_columns[0])


def equal_weight_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    del params
    return EqualWeightRankEnsembleFactory(spec.input_columns, spec.label_column)


def ic_weighted_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    del params
    return ICWeightedRankEnsembleFactory(spec.input_columns, spec.label_column)


def _set_random_state(params: dict[str, object], spec: ModelSpec) -> None:
    if spec.random_seed is not None:
        params.setdefault("random_state", spec.random_seed)


_MODEL_ESTIMATOR_FACTORIES: dict[
    str, Callable[[ModelSpec, dict[str, object]], object]
] = {
    "linear_regression": linear_regression_estimator,
    "ridge_regression": ridge_regression_estimator,
    "lasso_regression": lasso_regression_estimator,
    "elastic_net": elastic_net_estimator,
    "huber_regression": huber_regression_estimator,
    "decision_tree_regressor": decision_tree_regressor,
    "random_forest_regressor": random_forest_regressor,
    "extra_trees_regressor": extra_trees_regressor,
    "gradient_boosting_regressor": gradient_boosting_regressor,
    "hist_gradient_boosting_regressor": hist_gradient_boosting_regressor,
    "xgboost_regressor": xgboost_regressor,
    "lightgbm_regressor": lightgbm_regressor,
    "single_factor_baseline": single_factor_baseline,
    "equal_weight_rank_ensemble": equal_weight_rank_ensemble,
    "ic_weighted_rank_ensemble": ic_weighted_rank_ensemble,
}
