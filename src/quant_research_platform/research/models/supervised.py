"""Deterministic supervised model helpers."""

from __future__ import annotations

from collections.abc import Iterable

import quant_research_platform.research.models.dependencies as dependencies
import quant_research_platform.research.models.estimators as estimators
from quant_research_platform.contracts.models import (
    _PARALLEL_MODEL_ALGORITHMS,
    DEFAULT_MODEL_THREAD_COUNT,
    ModelSpec,
    _resolved_model_thread_count,
    _thaw_model_hyperparameters,
)
from quant_research_platform.runtime.resources import model_cpu_capacity


def preflight_model_specs(specs: Iterable[ModelSpec]) -> None:
    for spec in specs:
        build_estimator_from_model_spec(spec)


def build_estimator_from_model_spec(spec: ModelSpec) -> object | None:
    params = resolved_model_hyperparameters(spec)
    dependencies.require_model_dependency(spec.model_algorithm)
    if spec.model_algorithm == "linear_regression":
        return None
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
