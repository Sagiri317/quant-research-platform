"""Deterministic supervised model helpers."""

from __future__ import annotations

import quant_research_platform.research.models.algorithm_facts as algorithm_facts
import quant_research_platform.research.models.dependencies as dependencies
from quant_research_platform.contracts.models import ModelSpec


def build_estimator_from_model_spec(spec: ModelSpec) -> object:
    facts = algorithm_facts.algorithm_facts(spec.model_algorithm)
    dependencies.require_model_dependency(facts.algorithm)
    if facts.estimator_factory is None:
        raise ValueError("unsupported_model_algorithm")
    return facts.estimator_factory(spec, resolved_model_hyperparameters(spec))


def resolved_model_hyperparameters(spec: ModelSpec) -> dict[str, object]:
    params = dict(
        algorithm_facts.algorithm_facts(spec.model_algorithm).default_hyperparameters
    )
    params.update(spec.hyperparameters)
    return params
