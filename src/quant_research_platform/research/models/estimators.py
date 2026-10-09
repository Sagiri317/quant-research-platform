"""Estimator factories for supervised model algorithms."""

from __future__ import annotations

from collections.abc import Callable

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.rank_ensemble import (
    EqualWeightRankEnsembleFactory,
    ICWeightedRankEnsembleFactory,
)


def ridge_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.linear_model import Ridge

    _set_random_state(params, spec)
    return Ridge(**params)


def lasso_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
    from sklearn.linear_model import Lasso

    _set_random_state(params, spec)
    return Lasso(**params)


def huber_regression_estimator(spec: ModelSpec, params: dict[str, object]) -> object:
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


def gradient_boosting_regressor(
    spec: ModelSpec,
    params: dict[str, object],
) -> object:
    from sklearn.ensemble import GradientBoostingRegressor

    _set_random_state(params, spec)
    return GradientBoostingRegressor(**params)


def xgboost_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from xgboost import XGBRegressor

    _set_random_state(params, spec)
    return XGBRegressor(**params)


def lightgbm_regressor(spec: ModelSpec, params: dict[str, object]) -> object:
    from lightgbm import LGBMRegressor

    _set_random_state(params, spec)
    return LGBMRegressor(**params)


def equal_weight_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    return EqualWeightRankEnsembleFactory(spec.input_columns, spec.label_column)


def ic_weighted_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    return ICWeightedRankEnsembleFactory(spec.input_columns, spec.label_column)


def _set_random_state(params: dict[str, object], spec: ModelSpec) -> None:
    if spec.random_seed is not None:
        params.setdefault("random_state", spec.random_seed)


_MODEL_ESTIMATOR_FACTORIES: dict[
    str, Callable[[ModelSpec, dict[str, object]], object]
] = {
    "ridge_regression": ridge_regression_estimator,
    "lasso_regression": lasso_regression_estimator,
    "huber_regression": huber_regression_estimator,
    "decision_tree_regressor": decision_tree_regressor,
    "random_forest_regressor": random_forest_regressor,
    "gradient_boosting_regressor": gradient_boosting_regressor,
    "xgboost_regressor": xgboost_regressor,
    "lightgbm_regressor": lightgbm_regressor,
    "equal_weight_rank_ensemble": equal_weight_rank_ensemble,
    "ic_weighted_rank_ensemble": ic_weighted_rank_ensemble,
}
