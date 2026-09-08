"""Deterministic supervised model helpers."""

from __future__ import annotations

from collections.abc import Iterable

import quant_research_platform.research.models.dependencies as dependencies
import quant_research_platform.research.models.estimators as estimators
from quant_research_platform.contracts.models import (
    _PARALLEL_MODEL_ALGORITHMS,
    DEFAULT_MODEL_THREAD_COUNT,
    ModelSpec,
    _model_algorithm_facts,
    _resolved_model_thread_count,
    _thaw_model_hyperparameters,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    model_cpu_capacity,
)


def build_estimator_from_model_spec(spec: ModelSpec) -> object:
    estimator = _build_estimator(spec)
    facts = _model_algorithm_facts(spec.model_algorithm)
    if facts.dependency_package is not None:
        if not dependencies.model_dependency_preflighted(spec.model_algorithm):
            if current_runtime_resource_policy() is not None:
                raise RuntimeError("model_dependency_not_preflighted")
            dependencies.preflight_model_dependency(spec.model_algorithm)
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
        if _model_algorithm_facts(algorithm).dependency_package is not None:
            dependencies.preflight_model_dependency(algorithm)


def _build_estimator(spec: ModelSpec) -> object:
    params = resolved_model_hyperparameters(spec)
    dependencies.require_model_dependency(spec.model_algorithm)
    factory = estimators._MODEL_ESTIMATOR_FACTORIES.get(spec.model_algorithm)
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
