"""Construct supported estimators from explicit algorithms, hyperparameters, and
seeds.
"""

from __future__ import annotations

from importlib import import_module

from quant_research_platform.contracts.models import (
    _PARALLEL_MODEL_ALGORITHMS,
    DEFAULT_MODEL_THREAD_COUNT,
    SEQUENCE_ALGORITHMS,
    ModelSpec,
    _model_algorithm_facts,
    _thaw_model_hyperparameters,
)

_MODEL_ESTIMATOR_CLASSES = {
    "lasso_regression": ("sklearn.linear_model", "Lasso"),
    "xgboost_regressor": ("xgboost", "XGBRegressor"),
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
    return {"status": "available", "package": package, "version": module.__version__}


def require_model_dependency(algorithm: str) -> dict[str, str]:
    status = model_dependency_status(algorithm)
    if status["status"] == "missing":
        raise ImportError(f"model_dependency_unavailable: {status['package']}")
    return status


def build_estimator_from_model_spec(spec: ModelSpec) -> object | None:
    params = resolved_model_hyperparameters(spec)
    require_model_dependency(spec.model_algorithm)
    if spec.model_algorithm in {
        "linear_regression",
        *SEQUENCE_ALGORITHMS,
    }:
        return None
    module, name = _MODEL_ESTIMATOR_CLASSES[spec.model_algorithm]
    estimator_class = getattr(import_module(module), name)
    params.setdefault("random_state", spec.random_seed)
    return estimator_class(**params)


def resolved_model_hyperparameters(spec: ModelSpec) -> dict[str, object]:
    params = _thaw_model_hyperparameters(spec.hyperparameters)
    if spec.model_algorithm in _PARALLEL_MODEL_ALGORITHMS:
        params.setdefault("n_jobs", DEFAULT_MODEL_THREAD_COUNT)
    if spec.model_algorithm == "xgboost_regressor":
        params.setdefault("device", "cuda")
        params.setdefault("tree_method", "hist")
    if spec.model_algorithm in SEQUENCE_ALGORITHMS:
        from quant_research_platform.research.models.neural import sequence_parameters

        return sequence_parameters(spec.model_algorithm, params)
    return params
