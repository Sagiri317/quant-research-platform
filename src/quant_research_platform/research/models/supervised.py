"""Deterministic supervised model helpers."""

from __future__ import annotations

from collections.abc import Iterable

import quant_research_platform.research.models.dependencies as dependencies
import quant_research_platform.research.models.estimators as estimators
from quant_research_platform.contracts.models import (
    _PARALLEL_MODEL_ALGORITHMS,
    ModelSpec,
    _model_algorithm_facts,
    _model_default_hyperparameters,
    _resolved_model_thread_count,
)
from quant_research_platform.runtime.resources import current_runtime_resource_policy


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
    params = _model_default_hyperparameters(spec.model_algorithm)
    params.update(spec.hyperparameters)
    if spec.model_algorithm in _PARALLEL_MODEL_ALGORITHMS:
        params["n_jobs"] = _resolved_model_thread_count(params.get("n_jobs"))
    return params


def model_thread_count(spec: ModelSpec) -> int:
    if spec.model_algorithm not in _PARALLEL_MODEL_ALGORITHMS:
        return 1
    return int(resolved_model_hyperparameters(spec)["n_jobs"])
