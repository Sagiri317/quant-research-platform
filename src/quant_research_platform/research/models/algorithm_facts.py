"""Algorithm facts for supervised model dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import quant_research_platform.research.models.estimators as estimators
from quant_research_platform.contracts.models import (
    ModelSpec,
    _model_algorithm_facts,
    _model_algorithm_ids,
    _model_default_hyperparameters,
    _model_smoke_hyperparameters,
)


@dataclass(frozen=True)
class AlgorithmFacts:
    algorithm: str
    dependency_package: str | None = None
    estimator_factory: Callable[[ModelSpec, dict[str, object]], object] | None = None
    default_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    smoke_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    input_kind: str = "feature_inputs"


_ESTIMATOR_FACTORY_NAME_EXCEPTIONS = {
    "linear_regression": "linear_regression_estimator",
    "ridge_regression": "ridge_regression_estimator",
    "lasso_regression": "lasso_regression_estimator",
    "elastic_net": "elastic_net_estimator",
    "huber_regression": "huber_regression_estimator",
}
_MODEL_ALGORITHM_FACTS = {
    algorithm: AlgorithmFacts(
        algorithm=algorithm,
        dependency_package=contract.dependency_package,
        estimator_factory=getattr(
            estimators,
            _ESTIMATOR_FACTORY_NAME_EXCEPTIONS.get(algorithm, algorithm),
        ),
        default_hyperparameters=contract.default_hyperparameters,
        smoke_hyperparameters=contract.smoke_hyperparameters,
        input_kind=contract.input_kind,
    )
    for algorithm in _model_algorithm_ids()
    for contract in (_model_algorithm_facts(algorithm),)
}


def model_algorithm_ids() -> tuple[str, ...]:
    return _model_algorithm_ids()


def algorithm_facts(algorithm: str) -> AlgorithmFacts:
    try:
        return _MODEL_ALGORITHM_FACTS[algorithm]
    except KeyError as exc:
        raise ValueError("unsupported_model_algorithm") from exc


def model_default_hyperparameters(algorithm: str) -> dict[str, object]:
    return _model_default_hyperparameters(algorithm)


def model_smoke_hyperparameters(algorithm: str) -> dict[str, object]:
    return _model_smoke_hyperparameters(algorithm)


__all__ = [
    "AlgorithmFacts",
    "algorithm_facts",
    "model_algorithm_ids",
    "model_default_hyperparameters",
    "model_smoke_hyperparameters",
]
