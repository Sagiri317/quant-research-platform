"""Estimator factories for supervised model algorithms."""

from __future__ import annotations

from importlib import import_module

from quant_research_platform.contracts.models import (
    _PARALLEL_MODEL_ALGORITHMS,
    DEFAULT_MODEL_THREAD_COUNT,
    ModelSpec,
    _model_algorithm_facts,
    _thaw_model_hyperparameters,
)
from quant_research_platform.research.models.rank_ensemble import (
    EqualWeightRankEnsembleFactory,
    ICWeightedRankEnsembleFactory,
)

_MODEL_ESTIMATOR_CLASSES = {
    "ridge_regression": ("sklearn.linear_model", "Ridge"),
    "lasso_regression": ("sklearn.linear_model", "Lasso"),
    "huber_regression": ("sklearn.linear_model", "HuberRegressor"),
    "decision_tree_regressor": ("sklearn.tree", "DecisionTreeRegressor"),
    "random_forest_regressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "gradient_boosting_regressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "xgboost_regressor": ("xgboost", "XGBRegressor"),
    "lightgbm_regressor": ("lightgbm", "LGBMRegressor"),
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
    if spec.model_algorithm == "equal_weight_rank_ensemble":
        return EqualWeightRankEnsembleFactory(spec.input_columns, spec.target_column)
    if spec.model_algorithm == "ic_weighted_rank_ensemble":
        return ICWeightedRankEnsembleFactory(spec.input_columns, spec.target_column)
    module, name = _MODEL_ESTIMATOR_CLASSES[spec.model_algorithm]
    estimator_class = getattr(import_module(module), name)
    if spec.model_algorithm != "huber_regression":
        params.setdefault("random_state", spec.random_seed)
    return estimator_class(**params)


def resolved_model_hyperparameters(spec: ModelSpec) -> dict[str, object]:
    params = _thaw_model_hyperparameters(spec.hyperparameters)
    if spec.model_algorithm in _PARALLEL_MODEL_ALGORITHMS:
        params.setdefault("n_jobs", DEFAULT_MODEL_THREAD_COUNT)
    return params
