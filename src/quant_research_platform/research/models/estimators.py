"""Estimator factories for supervised model algorithms."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from quant_research_platform.contracts.models import (
    _PARALLEL_MODEL_ALGORITHMS,
    DEFAULT_MODEL_THREAD_COUNT,
    ModelSpec,
    _model_algorithm_facts,
    _resolved_model_thread_count,
    _thaw_model_hyperparameters,
)
from quant_research_platform.research.models.rank_ensemble import (
    EqualWeightRankEnsembleFactory,
    ICWeightedRankEnsembleFactory,
)
from quant_research_platform.runtime.resources import model_cpu_capacity


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
    return EqualWeightRankEnsembleFactory(spec.input_columns, spec.target_column)


def ic_weighted_rank_ensemble(spec: ModelSpec, params: dict[str, object]) -> object:
    return ICWeightedRankEnsembleFactory(spec.input_columns, spec.target_column)


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


def model_dependency_status(algorithm: str) -> dict[str, str]:
    facts = _model_algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return {"status": "available", "package": "internal"}
    try:
        module = import_module(package)
    except ImportError:
        return {"status": "missing", "package": package, "version": "unavailable"}
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("model_dependency_version_unavailable")
    return {"status": "available", "package": package, "version": version}


def require_model_dependency(algorithm: str) -> dict[str, str]:
    status = model_dependency_status(algorithm)
    if status["status"] == "missing":
        raise ImportError(f"model_dependency_unavailable: {status['package']}")
    return status


def build_estimator_from_model_spec(spec: ModelSpec) -> object | None:
    params = resolved_model_hyperparameters(spec)
    require_model_dependency(spec.model_algorithm)
    if spec.model_algorithm == "linear_regression":
        return None
    factory = _MODEL_ESTIMATOR_FACTORIES.get(spec.model_algorithm)
    if factory is None:
        raise ValueError("unsupported_model_algorithm")
    return factory(spec, params)


def resolved_model_hyperparameters(spec: ModelSpec) -> dict[str, object]:
    params = _thaw_model_hyperparameters(spec.hyperparameters)
    if spec.model_algorithm in _PARALLEL_MODEL_ALGORITHMS and "n_jobs" in params:
        physical, available = model_cpu_capacity()
        params["n_jobs"] = _resolved_model_thread_count(
            params["n_jobs"], physical_cores=physical, available_cores=available
        )
    return params


def model_thread_count(spec: ModelSpec) -> int:
    if spec.model_algorithm not in _PARALLEL_MODEL_ALGORITHMS:
        return DEFAULT_MODEL_THREAD_COUNT
    params = resolved_model_hyperparameters(spec)
    if "n_jobs" not in params:
        return DEFAULT_MODEL_THREAD_COUNT
    return int(params["n_jobs"])
