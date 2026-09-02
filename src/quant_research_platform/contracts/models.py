"""Explicit model capability and fitting contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

import numpy as np
import psutil

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    reject_filter_input_columns,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
)


@dataclass(frozen=True)
class _ModelAlgorithmFacts:
    """Capability facts only; research parameters belong to a protocol."""

    dependency_package: str | None = None
    input_kind: str = "feature_inputs"
    parallel: bool = False


_MODEL_ALGORITHM_FACTS = {
    "linear_regression": _ModelAlgorithmFacts(),
    "ridge_regression": _ModelAlgorithmFacts("sklearn"),
    "lasso_regression": _ModelAlgorithmFacts("sklearn"),
    "huber_regression": _ModelAlgorithmFacts("sklearn"),
    "random_forest_regressor": _ModelAlgorithmFacts(
        "sklearn",
        parallel=True,
    ),
    "gradient_boosting_regressor": _ModelAlgorithmFacts("sklearn"),
    "xgboost_regressor": _ModelAlgorithmFacts("xgboost", parallel=True),
    "lightgbm_regressor": _ModelAlgorithmFacts("lightgbm", parallel=True),
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
    if type(algorithm) is not str:
        raise ValueError("unsupported_model_algorithm")
    try:
        return _MODEL_ALGORITHM_FACTS[algorithm]
    except KeyError as exc:
        raise ValueError("unsupported_model_algorithm") from exc


def _model_algorithm_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ALGORITHM_FACTS))


def _freeze_model_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _freeze_model_value(value.item())
    if isinstance(value, np.ndarray):
        return _freeze_model_value(value.tolist())
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("model hyperparameter mapping keys must be strings")
            frozen[key] = _freeze_model_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_model_value(item) for item in value)
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("model hyperparameter values must be finite")
        return value
    if type(value) is str:
        return value
    raise TypeError("unsupported_model_hyperparameter_value")


def _thaw_model_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_model_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_model_value(item) for item in value]
    return value


def _freeze_model_hyperparameters(
    values: Mapping[str, object],
) -> Mapping[str, object]:
    frozen = _freeze_model_value(values)
    if not isinstance(frozen, Mapping):
        raise TypeError("model hyperparameters must be a mapping")
    return frozen


def _thaw_model_hyperparameters(
    values: Mapping[str, object],
) -> dict[str, object]:
    thawed = _thaw_model_value(values)
    if type(thawed) is not dict:
        raise TypeError("model hyperparameters must be a mapping")
    return thawed


def _resolved_model_hyperparameters_match(
    algorithm: str,
    hyperparameters: dict[str, object],
    resolved_hyperparameters: dict[str, object],
) -> bool:
    """Check an artifact against the explicit protocol parameters."""

    try:
        expected = dict(hyperparameters)
        if algorithm in _PARALLEL_MODEL_ALGORITHMS and "n_jobs" in expected:
            requested = expected["n_jobs"]
            actual = resolved_hyperparameters.get("n_jobs")
            if (
                type(requested) is not int
                or requested == 0
                or type(actual) is not int
                or actual <= 0
                or (requested > 0 and actual > requested)
            ):
                return False
            expected["n_jobs"] = actual
    except (TypeError, ValueError):
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
    """A model request with no hidden research defaults."""

    model_id: str
    model_algorithm: str
    input_factor_columns: tuple[str, ...] = ()
    input_feature_columns: tuple[str, ...] = ()
    label_column: str
    sample_filter_mode: SampleFilterMode = "eligible"
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int | None = None
    max_train_rows: int | None = None
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        hyperparameters = _freeze_model_hyperparameters(self.hyperparameters)
        if type(self.contract_version) is not str or self.contract_version != "0.1":
            raise ValueError("unsupported_model_contract_version")
        _validate_non_empty_string(self.model_id, "model_id_required")
        _validate_non_empty_string(self.model_algorithm, "model_algorithm_required")
        facts = _model_algorithm_facts(self.model_algorithm)
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
        _validate_non_empty_string(self.label_column, "model_label_column_required")
        if facts.input_kind == "single_factor_input" and (
            len(self.input_factor_columns) != 1 or self.input_feature_columns
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
        if self.random_seed is not None and (
            type(self.random_seed) is not int or isinstance(self.random_seed, bool)
        ):
            raise TypeError("random_seed must be an integer or None")
        if (
            self.random_seed is not None
            and "random_state" in hyperparameters
            and hyperparameters["random_state"] != self.random_seed
        ):
            raise ValueError("model_random_seed_conflicts_with_random_state")
        if self.max_train_rows is not None and (
            type(self.max_train_rows) is not int
            or isinstance(self.max_train_rows, bool)
        ):
            raise TypeError("max_train_rows must be an integer or None")
        if self.max_train_rows is not None and self.max_train_rows <= 0:
            raise ValueError("max_train_rows_must_be_positive")
        object.__setattr__(self, "hyperparameters", hyperparameters)

    @property
    def input_columns(self) -> tuple[str, ...]:
        return (*self.input_factor_columns, *self.input_feature_columns)
