"""Algorithm facts for supervised model dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import estimators


@dataclass(frozen=True)
class AlgorithmFacts:
    algorithm: str
    dependency_package: str | None = None
    estimator_factory: Callable[[ModelSpec, dict[str, object]], object] | None = None
    default_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    smoke_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    input_kind: str = "feature_inputs"


_MODEL_ALGORITHM_FACTS = {
    "linear_regression": AlgorithmFacts(
        "linear_regression",
        "sklearn",
        estimator_factory=estimators._linear_regression_estimator,
    ),
    "ridge_regression": AlgorithmFacts(
        "ridge_regression",
        "sklearn",
        estimator_factory=estimators._ridge_regression_estimator,
        default_hyperparameters={"alpha": 1.0},
    ),
    "lasso_regression": AlgorithmFacts(
        "lasso_regression",
        "sklearn",
        estimator_factory=estimators._lasso_regression_estimator,
        default_hyperparameters={"alpha": 0.001, "max_iter": 10_000},
    ),
    "elastic_net": AlgorithmFacts(
        "elastic_net",
        "sklearn",
        estimator_factory=estimators._elastic_net_estimator,
        default_hyperparameters={
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 10_000,
        },
    ),
    "huber_regression": AlgorithmFacts(
        "huber_regression",
        "sklearn",
        estimator_factory=estimators._huber_regression_estimator,
        default_hyperparameters={"max_iter": 1_000},
    ),
    "decision_tree_regressor": AlgorithmFacts(
        "decision_tree_regressor",
        "sklearn",
        estimator_factory=estimators._decision_tree_regressor,
    ),
    "random_forest_regressor": AlgorithmFacts(
        "random_forest_regressor",
        "sklearn",
        estimator_factory=estimators._random_forest_regressor,
        default_hyperparameters={"n_estimators": 20, "n_jobs": 1},
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
    ),
    "extra_trees_regressor": AlgorithmFacts(
        "extra_trees_regressor",
        "sklearn",
        estimator_factory=estimators._extra_trees_regressor,
        default_hyperparameters={"n_estimators": 20, "n_jobs": 1},
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
    ),
    "gradient_boosting_regressor": AlgorithmFacts(
        "gradient_boosting_regressor",
        "sklearn",
        estimator_factory=estimators._gradient_boosting_regressor,
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 2},
    ),
    "hist_gradient_boosting_regressor": AlgorithmFacts(
        "hist_gradient_boosting_regressor",
        "sklearn",
        estimator_factory=estimators._hist_gradient_boosting_regressor,
        smoke_hyperparameters={"max_iter": 5, "max_leaf_nodes": 7},
    ),
    "xgboost_regressor": AlgorithmFacts(
        "xgboost_regressor",
        "xgboost",
        estimator_factory=estimators._xgboost_regressor,
        default_hyperparameters={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "device": "cpu",
            "n_jobs": 1,
        },
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
    ),
    "lightgbm_regressor": AlgorithmFacts(
        "lightgbm_regressor",
        "lightgbm",
        estimator_factory=estimators._lightgbm_regressor,
        default_hyperparameters={
            "n_estimators": 20,
            "learning_rate": 0.1,
            "verbosity": -1,
            "n_jobs": 1,
        },
        smoke_hyperparameters={
            "n_estimators": 5,
            "num_leaves": 7,
            "n_jobs": 1,
            "verbosity": -1,
        },
    ),
    "single_factor_baseline": AlgorithmFacts(
        "single_factor_baseline",
        estimator_factory=estimators._single_factor_baseline,
        input_kind="single_factor_input",
    ),
    "equal_weight_rank_ensemble": AlgorithmFacts(
        "equal_weight_rank_ensemble",
        estimator_factory=estimators._equal_weight_rank_ensemble,
        input_kind="rank_ensemble_inputs",
    ),
    "ic_weighted_rank_ensemble": AlgorithmFacts(
        "ic_weighted_rank_ensemble",
        estimator_factory=estimators._ic_weighted_rank_ensemble,
        input_kind="rank_ensemble_inputs",
    ),
}


def model_algorithm_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ALGORITHM_FACTS))


def algorithm_facts(algorithm: str) -> AlgorithmFacts:
    try:
        return _MODEL_ALGORITHM_FACTS[algorithm]
    except KeyError as exc:
        raise ValueError("unsupported_model_algorithm") from exc


def model_default_hyperparameters(algorithm: str) -> dict[str, object]:
    return dict(algorithm_facts(algorithm).default_hyperparameters)


def model_smoke_hyperparameters(algorithm: str) -> dict[str, object]:
    return dict(algorithm_facts(algorithm).smoke_hyperparameters)


__all__ = [
    "AlgorithmFacts",
    "algorithm_facts",
    "model_algorithm_ids",
    "model_default_hyperparameters",
    "model_smoke_hyperparameters",
]
