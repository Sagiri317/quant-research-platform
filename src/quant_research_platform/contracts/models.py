"""Explicit model capability and fitting contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, isfinite
from types import MappingProxyType

import numpy as np
import pandas as pd

from quant_research_platform.contracts.evaluations import (
    CORRELATION_SUMMARY_REASON_CODES,
)
from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    filter_definition,
    reject_filter_input_columns,
    required_filter_columns,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    parse_qrp_json_object,
    qrp_json_mapping_snapshot,
    render_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    is_datetime_value,
    time_value_isoformat,
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
    "decision_tree_regressor": _ModelAlgorithmFacts("sklearn"),
    "random_forest_regressor": _ModelAlgorithmFacts(
        "sklearn",
        parallel=True,
    ),
    "gradient_boosting_regressor": _ModelAlgorithmFacts("sklearn"),
    "xgboost_regressor": _ModelAlgorithmFacts("xgboost", parallel=True),
    "lightgbm_regressor": _ModelAlgorithmFacts("lightgbm", parallel=True),
    "equal_weight_rank_ensemble": _ModelAlgorithmFacts(
        input_kind="rank_ensemble_inputs",
    ),
    "ic_weighted_rank_ensemble": _ModelAlgorithmFacts(
        input_kind="rank_ensemble_inputs",
    ),
}
MODEL_SELECTION_METRIC_DIRECTIONS = {
    "validation_ic": "maximize",
    "validation_rankic": "maximize",
    "validation_mse": "minimize",
}
_PARALLEL_MODEL_ALGORITHMS = frozenset(
    algorithm for algorithm, facts in _MODEL_ALGORITHM_FACTS.items() if facts.parallel
)
DEFAULT_MODEL_THREAD_COUNT = 1


def _model_algorithm_facts(algorithm: str) -> _ModelAlgorithmFacts:
    if type(algorithm) is not str:
        raise ValueError("unsupported_model_algorithm")
    try:
        return _MODEL_ALGORITHM_FACTS[algorithm]
    except KeyError as exc:
        raise ValueError("unsupported_model_algorithm") from exc


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
        if algorithm in _PARALLEL_MODEL_ALGORITHMS:
            requested = expected.get("n_jobs", DEFAULT_MODEL_THREAD_COUNT)
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
        return canonical_identity_bytes(
            resolved_hyperparameters
        ) == canonical_identity_bytes(expected)
    except (TypeError, ValueError):
        return False


def _resolved_model_thread_count(
    value: object, *, physical_cores: int, available_cores: int
) -> int:
    if type(value) is not int or value == 0:
        raise ValueError("invalid_model_thread_count")
    if value > physical_cores:
        raise ValueError("model_threads_exceed_physical_cores")
    resolved = min(value, available_cores) if value > 0 else available_cores + 1 + value
    if resolved <= 0:
        raise ValueError("invalid_model_thread_count")
    return resolved


@dataclass(frozen=True)
class TrainingWeighting:
    """Equal timestamp mass, optionally decayed in observed training periods."""

    method: str = "equal_timestamp"
    half_life_periods: float | None = None

    def __post_init__(self):
        if self.method != "equal_timestamp":
            raise ValueError("training_weighting_method_invalid")
        if self.half_life_periods is not None and (
            type(self.half_life_periods) not in {int, float}
            or not isfinite(self.half_life_periods)
            or self.half_life_periods <= 0
        ):
            raise ValueError("training_weighting_half_life_periods_invalid")

    def payload(self):
        return {"method": self.method, "half_life_periods": self.half_life_periods}


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """A model request with no hidden research defaults."""

    model_id: str
    model_algorithm: str
    input_factor_columns: tuple[str, ...] = ()
    input_feature_columns: tuple[str, ...] = ()
    target_column: str
    sample_filter_mode: SampleFilterMode = "eligible"
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int | None = None
    max_train_rows: int | None = None
    training_weighting: TrainingWeighting = field(default_factory=TrainingWeighting)

    def __post_init__(self) -> None:
        hyperparameters = _freeze_model_hyperparameters(self.hyperparameters)
        weighting = self.training_weighting
        if isinstance(weighting, Mapping):
            weighting = TrainingWeighting(**weighting)
        if type(weighting) is not TrainingWeighting:
            raise ValueError("training_weighting_invalid")
        object.__setattr__(self, "training_weighting", weighting)
        _validate_non_empty_string(self.model_id, "model_id_required")
        _validate_non_empty_string(self.model_algorithm, "model_algorithm_required")
        _model_algorithm_facts(self.model_algorithm)
        if hyperparameters and self.model_algorithm in {
            "linear_regression",
            "equal_weight_rank_ensemble",
            "ic_weighted_rank_ensemble",
        }:
            raise ValueError("model_hyperparameters_not_supported")
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
        _validate_non_empty_string(self.target_column, "model_target_column_required")
        if not self.input_factor_columns and not self.input_feature_columns:
            raise ValueError("model_input_columns_required")
        overlap = sorted(
            set(self.input_factor_columns).intersection(self.input_feature_columns)
        )
        if overlap:
            raise ValueError("model_factor_feature_overlap: " + ", ".join(overlap))
        reject_filter_input_columns(self.input_factor_columns)
        reject_filter_input_columns(self.input_feature_columns)
        if self.target_column in self.input_columns:
            raise ValueError("model_target_column_in_inputs")
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


def validate_model_metadata_core(
    value: object,
    *,
    rank_calibration: bool = False,
) -> dict[str, object]:
    """Validate the shared persisted model metadata contract."""

    if type(value) is not dict:
        raise ValueError("invalid_model_metadata")
    try:
        metadata = parse_qrp_json_object(render_qrp_json_object(value).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_model_metadata") from exc
    required = {
        "model_algorithm",
        "input_factor_columns",
        "input_feature_columns",
        "input_columns",
        "input_count",
        "target_column",
        "sample_filter_mode",
        "hyperparameters",
        "resolved_hyperparameters",
        "random_seed",
        "max_train_rows",
        "dependency_status",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "train_start",
        "training_asset_ids",
        "train_end",
        "filter_columns_used",
        "filter_definition",
        "training_weighting",
        "scaler",
    }
    assets = metadata.get("training_asset_ids")
    if (
        type(assets) is not list
        or any(type(asset) is not str or not asset for asset in assets)
        or len(assets) != len(set(assets))
    ):
        raise ValueError("invalid_model_training_asset_ids")
    if not required.issubset(metadata):
        raise ValueError("invalid_model_metadata")
    allowed = required | {
        "feature_importance",
        "feature_importance_status",
        "rank_parameters",
    }
    if set(metadata).difference(allowed):
        raise ValueError("invalid_model_metadata")
    for key in (
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "input_count",
    ):
        value_for_key = metadata[key]
        if type(value_for_key) is not int or value_for_key < 0:
            raise ValueError("invalid_model_metadata")
    if not (
        0
        < metadata["rows_after_input_label_dropna"]
        <= metadata["rows_after_filter_mode"]
        <= metadata["rows_input"]
    ):
        raise ValueError("invalid_model_metadata")
    if (
        type(metadata["hyperparameters"]) is not dict
        or type(metadata["resolved_hyperparameters"]) is not dict
        or not _resolved_model_hyperparameters_match(
            metadata["model_algorithm"],
            metadata["hyperparameters"],
            metadata["resolved_hyperparameters"],
        )
    ):
        raise ValueError("invalid_model_metadata")
    for key in (
        "input_factor_columns",
        "input_feature_columns",
        "input_columns",
        "filter_columns_used",
    ):
        columns = metadata[key]
        if type(columns) is not list or any(
            type(column) is not str or not column for column in columns
        ):
            raise ValueError("invalid_model_metadata")
    if metadata["input_count"] != len(metadata["input_columns"]):
        raise ValueError("invalid_model_metadata")
    if metadata["input_columns"] != [
        *metadata["input_factor_columns"],
        *metadata["input_feature_columns"],
    ]:
        raise ValueError("invalid_model_metadata")
    dependency_status = metadata["dependency_status"]
    try:
        facts = _model_algorithm_facts(metadata["model_algorithm"])
    except ValueError as exc:
        raise ValueError("invalid_model_metadata") from exc
    expected_package = facts.dependency_package or "internal"
    expected_keys = {"status", "package"}
    if facts.dependency_package is not None:
        expected_keys.add("version")
    if (
        type(dependency_status) is not dict
        or set(dependency_status) != expected_keys
        or dependency_status["status"] != "available"
        or type(dependency_status["package"]) is not str
        or dependency_status["package"] != expected_package
        or (
            facts.dependency_package is not None
            and (
                type(dependency_status["version"]) is not str
                or not dependency_status["version"]
            )
        )
    ):
        raise ValueError("invalid_model_metadata")
    try:
        if metadata["filter_columns_used"] != list(
            required_filter_columns(metadata["sample_filter_mode"])
        ) or metadata["filter_definition"] != filter_definition(
            metadata["sample_filter_mode"]
        ):
            raise ValueError("invalid_model_metadata")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_model_metadata") from exc
    try:
        ModelSpec(
            model_id="persisted_model",
            model_algorithm=metadata["model_algorithm"],
            input_factor_columns=tuple(metadata["input_factor_columns"]),
            input_feature_columns=tuple(metadata["input_feature_columns"]),
            target_column=metadata["target_column"],
            sample_filter_mode=metadata["sample_filter_mode"],
            hyperparameters=metadata["hyperparameters"],
            random_seed=metadata["random_seed"],
            max_train_rows=metadata["max_train_rows"],
            training_weighting=metadata["training_weighting"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_model_metadata") from exc
    for key in ("train_start", "train_end"):
        if type(metadata[key]) is not str:
            raise ValueError("invalid_model_metadata")
        try:
            timestamp = pd.Timestamp(metadata[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_model_metadata") from exc
        if pd.isna(timestamp):
            raise ValueError("invalid_model_metadata")
    train_start = pd.Timestamp(metadata["train_start"])
    train_end = pd.Timestamp(metadata["train_end"])
    if train_start > train_end:
        raise ValueError("invalid_model_metadata")
    scaler = metadata["scaler"]
    if scaler is not None:
        if (
            type(scaler) is not dict
            or set(scaler)
            != {
                "columns",
                "weighted_mean_origin",
                "weighted_mean_offset",
                "weighted_population_scale",
            }
            or scaler["columns"] != metadata["input_columns"]
            or type(scaler["weighted_mean_origin"]) is not list
            or type(scaler["weighted_mean_offset"]) is not list
            or type(scaler["weighted_population_scale"]) is not list
            or len(scaler["columns"]) != len(scaler["weighted_mean_origin"])
            or len(scaler["columns"]) != len(scaler["weighted_mean_offset"])
            or len(scaler["columns"]) != len(scaler["weighted_population_scale"])
            or any(
                type(number) not in {int, float} or not np.isfinite(float(number))
                for number in (
                    *scaler["weighted_mean_origin"],
                    *scaler["weighted_population_scale"],
                )
            )
            or any(
                type(number) not in {int, float}
                or not np.isfinite(float(number))
                or float(number) <= 0.0
                for number in scaler["weighted_population_scale"]
            )
            or any(
                type(ratio) is not list
                or len(ratio) != 2
                or any(type(value) is not int for value in ratio)
                or ratio[1] <= 0
                or gcd(*ratio) != 1
                for ratio in scaler["weighted_mean_offset"]
            )
        ):
            raise ValueError("invalid_model_metadata")
        try:
            for origin, ratio in zip(
                scaler["weighted_mean_origin"],
                scaler["weighted_mean_offset"],
                strict=True,
            ):
                offset = Fraction(*ratio)
                float(offset)
                float(Fraction(float(origin)) + offset)
        except OverflowError:
            raise ValueError("invalid_model_metadata") from None
    rank_algorithms = {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }
    feature_keys = {
        key
        for key in ("feature_importance", "feature_importance_status")
        if key in metadata
    }
    if len(feature_keys) != 1:
        raise ValueError("invalid_model_metadata")
    if "feature_importance" in metadata:
        importance = metadata["feature_importance"]
        if (
            type(importance) is not dict
            or not importance
            or (
                not set(importance).issubset(metadata["input_columns"])
                or (
                    metadata["model_algorithm"] not in rank_algorithms
                    and set(importance) != set(metadata["input_columns"])
                )
            )
            or any(
                type(value) not in {int, float} or not np.isfinite(float(value))
                for value in importance.values()
            )
        ):
            raise ValueError("invalid_model_metadata")
    else:
        if (
            type(metadata["feature_importance_status"]) is not str
            or not metadata["feature_importance_status"]
        ):
            raise ValueError("invalid_model_metadata")
    rank_parameters = metadata.get("rank_parameters")
    if metadata["model_algorithm"] in rank_algorithms:
        expected = (
            {"columns", "directions"}
            if metadata["model_algorithm"] == "equal_weight_rank_ensemble"
            else {"columns", "weights"}
        )
        if rank_calibration:
            expected.add("label_calibration")
        if type(rank_parameters) is not dict or set(rank_parameters) != expected:
            raise ValueError("invalid_model_metadata")
        if rank_calibration:
            calibration = rank_parameters["label_calibration"]
            if (
                type(calibration) is not list
                or len(calibration) != 2
                or any(
                    type(value) not in {int, float} or not np.isfinite(value)
                    for value in calibration
                )
            ):
                raise ValueError("invalid_model_metadata")
        rank_columns = rank_parameters["columns"]
        if (
            type(rank_columns) is not list
            or not rank_columns
            or len(set(rank_columns)) != len(rank_columns)
            or not set(rank_columns).issubset(metadata["input_columns"])
            or (
                metadata["model_algorithm"] == "equal_weight_rank_ensemble"
                and rank_columns != metadata["input_columns"]
            )
        ):
            raise ValueError("invalid_model_metadata")
        value_key = "directions" if "directions" in expected else "weights"
        values = rank_parameters[value_key]
        if type(values) is not list or len(values) != len(rank_columns):
            raise ValueError("invalid_model_metadata")
        if any(
            type(item) not in {int, float} or not np.isfinite(float(item))
            for item in values
        ):
            raise ValueError("invalid_model_metadata")
    elif rank_parameters is not None:
        raise ValueError("invalid_model_metadata")
    return metadata


_MODEL_SELECTION_TRIAL_METRIC_KEYS = frozenset(
    {
        "model_id",
        "validation_rows",
        "validation_ic",
        "validation_rankic",
        "validation_ic_reason",
        "validation_rankic_reason",
        "validation_mse",
    }
) | frozenset(
    f"validation_{suffix}"
    for prefix in ("ic", "rankic")
    for suffix in (
        f"std_{prefix}",
        f"std_{prefix}_reason",
        f"{prefix}ir",
        f"positive_{prefix}_rate",
        f"valid_{prefix}_count",
        f"{prefix}ir_reason",
    )
)


def _validate_a_share_model_family_ids(family_ids) -> None:
    """Keep the A-share champion selector distinct from named model families."""
    if "champion" in family_ids:
        raise ValueError("a_share_model_id_reserved")


def _validate_model_selection_trial_record(trial: object) -> None:
    if type(trial) is not dict or set(trial) != _MODEL_SELECTION_TRIAL_METRIC_KEYS:
        raise ValueError("invalid_model_metadata")
    model_id = trial["model_id"]
    rows = trial["validation_rows"]
    if type(model_id) is not str or not model_id or type(rows) is not int or rows < 1:
        raise ValueError("invalid_model_metadata")
    for metric, reason_key in (
        ("validation_ic", "validation_ic_reason"),
        ("validation_rankic", "validation_rankic_reason"),
    ):
        metric_value = trial[metric]
        reason = trial[reason_key]
        if reason is not None and (
            type(reason) is not str or reason not in CORRELATION_SUMMARY_REASON_CODES
        ):
            raise ValueError("invalid_model_metadata")
        if metric_value is not None and (
            type(metric_value) not in {int, float}
            or not np.isfinite(float(metric_value))
            or abs(float(metric_value)) > 1.0
        ):
            raise ValueError("invalid_model_metadata")
        if metric == "validation_rankic" and reason == "not_applicable":
            if metric_value is not None:
                raise ValueError("invalid_model_metadata")
        elif rows < 2:
            if metric_value is not None or reason != "no_valid_observations":
                raise ValueError("invalid_model_metadata")
        elif metric_value is None:
            if reason is None:
                raise ValueError("invalid_model_metadata")
        elif reason is not None:
            raise ValueError("invalid_model_metadata")
    for prefix in ("ic", "rankic"):
        count = trial[f"validation_valid_{prefix}_count"]
        std = trial[f"validation_std_{prefix}"]
        std_reason = trial[f"validation_std_{prefix}_reason"]
        ratio = trial[f"validation_{prefix}ir"]
        positive = trial[f"validation_positive_{prefix}_rate"]
        reason = trial[f"validation_{prefix}ir_reason"]
        if type(count) is not int or not 0 <= count <= rows:
            raise ValueError("invalid_model_metadata")
        for value in (std, ratio, positive):
            if value is not None and (
                type(value) not in {int, float} or not np.isfinite(value)
            ):
                raise ValueError("invalid_model_metadata")
        if (
            ((std is None) != (std_reason is not None))
            or (
                std_reason is not None
                and (
                    type(std_reason) is not str
                    or std_reason
                    not in {
                        "no_valid_observations",
                        "nonfinite_result",
                        "not_applicable",
                    }
                )
            )
            or (count > 0 and std_reason in {"no_valid_observations", "not_applicable"})
            or (
                count == 0
                and std_reason
                != (
                    "not_applicable"
                    if trial[f"validation_{prefix}_reason"] == "not_applicable"
                    else "no_valid_observations"
                )
            )
            or (std is not None and std < 0)
            or (positive is not None and not 0 <= positive <= 1)
            or (count == 0 and any(v is not None for v in (std, ratio, positive)))
            or (count > 0 and positive is None)
            or ((ratio is None) != (reason is not None))
            or (ratio is not None and (count < 2 or std == 0))
            or (
                reason is not None
                and (
                    type(reason) is not str
                    or reason not in CORRELATION_SUMMARY_REASON_CODES
                )
            )
        ):
            raise ValueError("invalid_model_metadata")
    if (
        type(trial["validation_mse"]) not in {int, float}
        or not np.isfinite(float(trial["validation_mse"]))
        or trial["validation_mse"] < 0
    ):
        raise ValueError("invalid_model_metadata")


def model_spec_payload(spec: object) -> dict[str, object]:
    return {
        "model_id": spec.model_id,
        "model_algorithm": spec.model_algorithm,
        "input_factor_columns": list(spec.input_factor_columns),
        "input_feature_columns": list(spec.input_feature_columns),
        "target_column": spec.target_column,
        "sample_filter_mode": spec.sample_filter_mode,
        "hyperparameters": _thaw_model_hyperparameters(spec.hyperparameters),
        "random_seed": spec.random_seed,
        "max_train_rows": spec.max_train_rows,
        "training_weighting": spec.training_weighting.payload(),
    }


def validate_completed_model_trial_metrics(
    model_id: str, metrics: Mapping[str, object]
) -> None:
    """Validate model metrics at model/report persisted-trial boundaries."""
    if set(metrics) != (_MODEL_SELECTION_TRIAL_METRIC_KEYS - {"model_id"}) | {
        "n_train"
    }:
        raise ValueError("invalid_model_trial_metrics")
    trial = dict(metrics)
    n_train = trial.pop("n_train")
    if type(n_train) is not int or n_train <= 0:
        raise ValueError("invalid_model_trial_metrics")
    trial["model_id"] = model_id
    _validate_model_selection_trial_record(trial)


def _strict_config(config, required, optional=()):
    value = qrp_json_mapping_snapshot(config)
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError("operation_config_fields_invalid")
    return value


def _model_spec(value):
    if not isinstance(value, Mapping):
        raise ValueError("model_spec_required")
    value = dict(value)
    for name in ("input_factor_columns", "input_feature_columns"):
        if name in value:
            if not isinstance(value[name], (list, tuple)):
                raise ValueError("model_input_columns_invalid")
            value[name] = tuple(value[name])
    return ModelSpec(**value)


def _window_payload(window):
    return {
        "start": time_value_isoformat(
            window.start, preserve_datetime=is_datetime_value(window.start)
        ),
        "end": None
        if window.end is None
        else time_value_isoformat(
            window.end, preserve_datetime=is_datetime_value(window.start)
        ),
    }


def _fit_windows(values, cutoff):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("fit_windows_required")
    windows = tuple(
        TimeWindow(**item) if isinstance(item, Mapping) else item for item in values
    )
    last = None
    for window in windows:
        if not isinstance(window, TimeWindow) or window.end is None:
            raise ValueError("finite_fit_window_required")
        start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
        if end > cutoff:
            raise ValueError("fit_window_after_cutoff")
        if last is not None and start <= last:
            raise ValueError("fit_windows_must_be_ordered_disjoint")
        last = end
    return windows


def _resolve_candidate_parameters(algorithm, parameters, n_train):
    values = dict(parameters)
    for name, target, algorithms in (
        ("regularization_strength", "alpha", {"ridge_regression", "huber_regression"}),
        ("min_child_weight_fraction", "min_child_weight", {"xgboost_regressor"}),
    ):
        if name not in values:
            continue
        value = values.pop(name)
        if (
            algorithm not in algorithms
            or target in values
            or type(value) not in {int, float}
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("model_sample_scaled_parameters_invalid")
        values[target] = n_train * value
        if not np.isfinite(values[target]):
            raise ValueError("model_sample_scaled_parameters_unrepresentable")
    return values
