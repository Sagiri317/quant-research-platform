"""Model contracts for non-DL local research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

import psutil

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    reject_filter_input_columns,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.validation import _validate_column_tuple


@dataclass(frozen=True)
class _ModelAlgorithmFacts:
    dependency_package: str | None = None
    default_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    smoke_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    input_kind: str = "feature_inputs"
    parallel: bool = False


_MODEL_ALGORITHM_FACTS = {
    "linear_regression": _ModelAlgorithmFacts(),
    "ridge_regression": _ModelAlgorithmFacts(
        "sklearn",
        default_hyperparameters={"alpha": 1.0},
    ),
    "lasso_regression": _ModelAlgorithmFacts(
        "sklearn",
        default_hyperparameters={"alpha": 0.001, "max_iter": 10_000},
    ),
    "elastic_net": _ModelAlgorithmFacts(
        "sklearn",
        default_hyperparameters={
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 10_000,
        },
    ),
    "huber_regression": _ModelAlgorithmFacts(
        "sklearn",
        default_hyperparameters={"max_iter": 1_000},
    ),
    "decision_tree_regressor": _ModelAlgorithmFacts(
        "sklearn",
    ),
    "random_forest_regressor": _ModelAlgorithmFacts(
        "sklearn",
        default_hyperparameters={"n_estimators": 20, "n_jobs": -1},
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 3, "n_jobs": -1},
        parallel=True,
    ),
    "extra_trees_regressor": _ModelAlgorithmFacts(
        "sklearn",
        default_hyperparameters={"n_estimators": 20, "n_jobs": -1},
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 3, "n_jobs": -1},
        parallel=True,
    ),
    "gradient_boosting_regressor": _ModelAlgorithmFacts(
        "sklearn",
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 2},
    ),
    "hist_gradient_boosting_regressor": _ModelAlgorithmFacts(
        "sklearn",
        smoke_hyperparameters={"max_iter": 5, "max_leaf_nodes": 7},
    ),
    "xgboost_regressor": _ModelAlgorithmFacts(
        "xgboost",
        default_hyperparameters={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "device": "cpu",
            "n_jobs": -1,
        },
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 2, "n_jobs": -1},
        parallel=True,
    ),
    "lightgbm_regressor": _ModelAlgorithmFacts(
        "lightgbm",
        default_hyperparameters={
            "n_estimators": 20,
            "learning_rate": 0.1,
            "verbosity": -1,
            "n_jobs": -1,
        },
        smoke_hyperparameters={
            "n_estimators": 5,
            "num_leaves": 7,
            "n_jobs": -1,
            "verbosity": -1,
        },
        parallel=True,
    ),
    "single_factor_baseline": _ModelAlgorithmFacts(
        input_kind="single_factor_input",
    ),
    "equal_weight_rank_ensemble": _ModelAlgorithmFacts(
        input_kind="rank_ensemble_inputs",
    ),
    "ic_weighted_rank_ensemble": _ModelAlgorithmFacts(
        input_kind="rank_ensemble_inputs",
    ),
}
_PARALLEL_MODEL_ALGORITHMS = frozenset(
    algorithm for algorithm, facts in _MODEL_ALGORITHM_FACTS.items() if facts.parallel
)


def _model_algorithm_facts(algorithm: str) -> _ModelAlgorithmFacts:
    try:
        return _MODEL_ALGORITHM_FACTS[algorithm]
    except KeyError as exc:
        raise ValueError("unsupported_model_algorithm") from exc


def _model_algorithm_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ALGORITHM_FACTS))


def _model_default_hyperparameters(algorithm: str) -> dict[str, object]:
    return dict(_model_algorithm_facts(algorithm).default_hyperparameters)


def _model_smoke_hyperparameters(algorithm: str) -> dict[str, object]:
    return dict(_model_algorithm_facts(algorithm).smoke_hyperparameters)


def _resolved_model_hyperparameters_match(
    algorithm: str,
    hyperparameters: dict[str, object],
    resolved_hyperparameters: dict[str, object],
) -> bool:
    try:
        expected = _model_default_hyperparameters(algorithm)
        expected.update(hyperparameters)
        if algorithm in _PARALLEL_MODEL_ALGORITHMS:
            requested_threads = expected.get("n_jobs")
            actual_threads = resolved_hyperparameters.get("n_jobs")
            if (
                type(requested_threads) is not int
                or requested_threads == 0
                or type(actual_threads) is not int
                or actual_threads <= 0
                or (requested_threads > 0 and actual_threads > requested_threads)
            ):
                return False
            expected["n_jobs"] = actual_threads
    except ValueError:
        return False
    return resolved_hyperparameters == expected


def _resolved_model_thread_count(value: object) -> int:
    if type(value) is not int or value == 0:
        raise ValueError("invalid_model_thread_count")
    physical = psutil.cpu_count(logical=False)
    if type(physical) is not int or physical <= 0:
        raise RuntimeError("physical_core_count_unavailable")
    resolved = value if value > 0 else physical + 1 + value
    if resolved <= 0:
        raise ValueError("invalid_model_thread_count")
    if resolved > physical:
        raise ValueError("model_threads_exceed_physical_cores")
    return resolved


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    model_id: str
    model_algorithm: str
    input_factor_columns: tuple[str, ...] = ()
    input_feature_columns: tuple[str, ...] = ()
    label_column: str
    sample_filter_mode: SampleFilterMode = "eligible"
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int | None = None
    purge_gap_steps: int = 0
    max_train_rows: int | None = None
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_model_contract_version")
        if not isinstance(self.model_algorithm, str) or not self.model_algorithm:
            raise ValueError("model_algorithm_required")
        validate_sample_filter_mode(self.sample_filter_mode, owner="model")
        _validate_column_tuple(
            self.input_factor_columns,
            "input_factor_columns",
            required=False,
        )
        _validate_column_tuple(
            self.input_feature_columns,
            "input_feature_columns",
            required=False,
        )
        facts = _MODEL_ALGORITHM_FACTS.get(self.model_algorithm)
        if (
            facts is not None
            and facts.input_kind == "single_factor_input"
            and (len(self.input_factor_columns) != 1 or self.input_feature_columns)
        ):
            raise ValueError("single_factor_input_requires_one_factor_and_no_features")
        if not self.input_factor_columns and not self.input_feature_columns:
            raise ValueError("model_input_columns_required")
        overlap = sorted(
            set(self.input_factor_columns).intersection(self.input_feature_columns)
        )
        if overlap:
            raise ValueError("model_factor_feature_overlap: " + ", ".join(overlap))
        reject_filter_input_columns(self.input_factor_columns)
        reject_filter_input_columns(self.input_feature_columns)
        if self.label_column in self.input_columns:
            raise ValueError("model_label_column_in_inputs")
        if self.random_seed is not None and isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer or None")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer or None")
        if (
            self.random_seed is not None
            and "random_state" in self.hyperparameters
            and self.hyperparameters["random_state"] != self.random_seed
        ):
            raise ValueError("model_random_seed_conflicts_with_random_state")
        if not isinstance(self.purge_gap_steps, int) or isinstance(
            self.purge_gap_steps, bool
        ):
            raise TypeError("purge_gap_steps must be an integer")
        if self.purge_gap_steps < 0:
            raise ValueError("purge_gap_steps_must_be_nonnegative")
        if self.max_train_rows is not None and (
            not isinstance(self.max_train_rows, int)
            or isinstance(self.max_train_rows, bool)
        ):
            raise TypeError("max_train_rows must be an integer or None")
        if self.max_train_rows is not None and self.max_train_rows <= 0:
            raise ValueError("max_train_rows_must_be_positive")
        object.__setattr__(
            self,
            "hyperparameters",
            MappingProxyType(
                {
                    key: (
                        int(value)
                        if (
                            key != "n_jobs"
                            and isinstance(value, Integral)
                            and not isinstance(value, bool)
                        )
                        else value
                    )
                    for key, value in self.hyperparameters.items()
                }
            ),
        )

    @property
    def input_columns(self) -> tuple[str, ...]:
        return (*self.input_factor_columns, *self.input_feature_columns)


__all__ = ["ModelSpec"]
