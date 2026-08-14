"""Deterministic supervised model helpers."""

from __future__ import annotations

from collections.abc import Iterable

import psutil

import quant_research_platform.research.models.algorithm_facts as algorithm_facts
import quant_research_platform.research.models.dependencies as dependencies
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.runtime.resources import current_runtime_resource_policy

_PARALLEL_MODEL_ALGORITHMS = frozenset(
    {
        "random_forest_regressor",
        "extra_trees_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }
)


def build_estimator_from_model_spec(spec: ModelSpec) -> object:
    estimator = _build_estimator(spec)
    facts = algorithm_facts.algorithm_facts(spec.model_algorithm)
    if facts.dependency_package is not None:
        if not dependencies.model_dependency_preflighted(facts.algorithm):
            if current_runtime_resource_policy() is not None:
                raise RuntimeError("model_dependency_not_preflighted")
            dependencies.preflight_model_dependency(facts.algorithm)
    return estimator


def preflight_model_specs(specs: Iterable[ModelSpec]) -> None:
    selected = tuple(specs)
    if current_runtime_resource_policy() is not None:
        if any(
            not dependencies.model_dependency_preflighted(spec.model_algorithm)
            for spec in selected
        ):
            raise RuntimeError("model_dependency_not_preflighted")
        return
    for spec in selected:
        _build_estimator(spec)
    for algorithm in dict.fromkeys(spec.model_algorithm for spec in selected):
        if algorithm_facts.algorithm_facts(algorithm).dependency_package is not None:
            dependencies.preflight_model_dependency(algorithm)


def _build_estimator(spec: ModelSpec) -> object:
    facts = algorithm_facts.algorithm_facts(spec.model_algorithm)
    params = resolved_model_hyperparameters(spec)
    dependencies.require_model_dependency(facts.algorithm)
    if facts.estimator_factory is None:
        raise ValueError("unsupported_model_algorithm")
    return facts.estimator_factory(spec, params)


def resolved_model_hyperparameters(spec: ModelSpec) -> dict[str, object]:
    params = dict(
        algorithm_facts.algorithm_facts(spec.model_algorithm).default_hyperparameters
    )
    params.update(spec.hyperparameters)
    if spec.model_algorithm in _PARALLEL_MODEL_ALGORITHMS:
        params["n_jobs"] = _resolved_model_thread_count(params.get("n_jobs"))
    return params


def model_thread_count(spec: ModelSpec) -> int:
    if spec.model_algorithm not in _PARALLEL_MODEL_ALGORITHMS:
        return 1
    return int(resolved_model_hyperparameters(spec)["n_jobs"])


def _resolved_model_hyperparameters_match(
    algorithm: str,
    hyperparameters: dict[str, object],
    resolved_hyperparameters: dict[str, object],
) -> bool:
    try:
        expected = dict(
            algorithm_facts.algorithm_facts(algorithm).default_hyperparameters
        )
        expected.update(hyperparameters)
        if algorithm in _PARALLEL_MODEL_ALGORITHMS:
            actual_threads = resolved_hyperparameters.get("n_jobs")
            maximum_threads = _resolved_model_thread_count(expected.get("n_jobs"))
            if (
                type(actual_threads) is not int
                or actual_threads <= 0
                or actual_threads > maximum_threads
            ):
                return False
            expected["n_jobs"] = actual_threads
    except ValueError:
        return False
    return resolved_hyperparameters == expected


def physical_core_count() -> int:
    count = psutil.cpu_count(logical=False)
    if type(count) is not int or count <= 0:
        raise RuntimeError("physical_core_count_unavailable")
    return count


def _resolved_model_thread_count(value: object) -> int:
    if type(value) is not int or value == 0:
        raise ValueError("invalid_model_thread_count")
    physical = physical_core_count()
    resolved = value if value > 0 else physical + 1 + value
    if resolved <= 0:
        raise ValueError("invalid_model_thread_count")
    if resolved > physical:
        raise ValueError("model_threads_exceed_physical_cores")
    return resolved
