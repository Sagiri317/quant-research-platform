"""Deterministic supervised model helpers."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import (
    algorithm_facts,
    dependencies,
    fitting,
)
from quant_research_platform.research.models.prediction import predict_model_frame

ModelFitResult = fitting.ModelFitResult


def model_algorithm_ids() -> tuple[str, ...]:
    return algorithm_facts.model_algorithm_ids()


def model_algorithm_definition(algorithm: str) -> algorithm_facts.AlgorithmFacts:
    return algorithm_facts.algorithm_facts(algorithm)


def model_default_hyperparameters(algorithm: str) -> dict[str, object]:
    return algorithm_facts.model_default_hyperparameters(algorithm)


def model_smoke_hyperparameters(algorithm: str) -> dict[str, object]:
    return algorithm_facts.model_smoke_hyperparameters(algorithm)


def model_dependency_status(algorithm: str) -> dict[str, str]:
    return dependencies.model_dependency_status(algorithm)


def build_estimator_from_model_spec(spec: ModelSpec) -> object:
    facts = algorithm_facts.algorithm_facts(spec.model_algorithm)
    dependencies.require_model_dependency(facts.algorithm)
    if facts.estimator_factory is None:
        raise ValueError("unsupported_model_algorithm")
    params = dict(facts.default_hyperparameters)
    params.update(spec.hyperparameters)
    return facts.estimator_factory(spec, params)


def fit_model_result(frame: pd.DataFrame, spec: ModelSpec) -> ModelFitResult:
    estimator = build_estimator_from_model_spec(spec)
    return fitting.fit_model_result_from_frames(
        (frame,),
        spec,
        estimator=estimator,
    )


__all__ = [
    "ModelFitResult",
    "build_estimator_from_model_spec",
    "fit_model_result",
    "predict_model_frame",
]
