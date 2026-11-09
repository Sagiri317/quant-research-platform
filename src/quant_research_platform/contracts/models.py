"""Explicit model capability and fitting contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

import numpy as np
import pandas as pd

from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.evaluations import (
    CORRELATION_SUMMARY_REASON_CODES,
)
from quant_research_platform.contracts.experiments import DEFAULT_ROOT_SEED
from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    filter_definition,
    reject_filter_input_columns,
    required_filter_columns,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.json_codec import (
    _json_native_object,
    canonical_identity_bytes,
)
from quant_research_platform.contracts.risk import FactorRiskModel
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    is_datetime_window,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
    strict_timestamp_series,
)


@dataclass(frozen=True)
class _ModelAlgorithmFacts:
    """Capability facts only; research parameters belong to a protocol."""

    dependency_package: str | None = None
    parallel: bool = False
    sequence: bool = False


_MODEL_ALGORITHM_FACTS = {
    "linear_regression": _ModelAlgorithmFacts(),
    "lasso_regression": _ModelAlgorithmFacts("sklearn"),
    "ridge_regression": _ModelAlgorithmFacts("sklearn"),
    "xgboost_regressor": _ModelAlgorithmFacts("xgboost", parallel=True),
    "lstm_regressor": _ModelAlgorithmFacts("torch", sequence=True),
    "transformer_regressor": _ModelAlgorithmFacts("torch", sequence=True),
    "deeplob_regressor": _ModelAlgorithmFacts("torch", sequence=True),
    "deeplob_classifier": _ModelAlgorithmFacts("torch", sequence=True),
}
SEQUENCE_ALGORITHMS = frozenset(
    name for name, facts in _MODEL_ALGORITHM_FACTS.items() if facts.sequence
)
DEEPLOB_ALGORITHMS = frozenset({"deeplob_regressor", "deeplob_classifier"})
DEEPLOB_INPUT_COLUMNS = tuple(
    f"{side}_{kind}{level}"
    for level in range(1, 11)
    for side in ("ask", "bid")
    for kind in ("px", "vol")
)
PROBABILITY_COLUMNS = ("probability_down", "probability_flat", "probability_up")
CLASSIFICATION_METRICS = ("log_loss", "brier", "accuracy", "macro_f1")


def sequence_context_columns(algorithm, parameters=None):
    groups = (parameters or {}).get(
        "sequence_group_columns",
        ("session_date", "segment_id") if algorithm in DEEPLOB_ALGORITHMS else (),
    )
    return tuple(groups)


def model_context_window(facts, window):
    """Read only history that can affect this algorithm's prediction window."""
    algorithm = facts["model"]["model_algorithm"]
    if algorithm not in SEQUENCE_ALGORITHMS:
        return window
    start = (
        pd.Timestamp(window.start).normalize()
        if "session_date"
        in sequence_context_columns(algorithm, facts["candidate_parameters"])
        else pd.Timestamp(facts["fit_windows"][-1]["end"])
    )
    return PredictionWindow(
        start.to_pydatetime() if is_datetime_window(window) else start.date(),
        window.end,
    )


def prediction_columns(metadata):
    return (
        "timestamp",
        "asset_id",
        "prediction_score",
        "prediction_rank_score",
        *(
            PROBABILITY_COLUMNS
            if metadata["model_facts"]["model"]["model_algorithm"]
            == "deeplob_classifier"
            else ()
        ),
    )


MODEL_SELECTION_METRIC_DIRECTIONS = {
    "validation_ic": "maximize",
    "validation_rankic": "maximize",
    "validation_mse": "minimize",
    "validation_log_loss": "minimize",
    "validation_brier": "minimize",
    "validation_accuracy": "maximize",
    "validation_macro_f1": "maximize",
}
_PARALLEL_MODEL_ALGORITHMS = frozenset(
    algorithm for algorithm, facts in _MODEL_ALGORITHM_FACTS.items() if facts.parallel
)
DEFAULT_MODEL_THREAD_COUNT = -1


def sequence_parameters(algorithm, parameters):
    defaults = {
        "sequence_length": 20,
        "hidden_size": 32,
        "num_layers": 1,
        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 0.001,
        "dropout": 0.0,
        "device": "cuda",
    }
    if algorithm == "transformer_regressor":
        defaults["n_heads"] = 4
    if algorithm in DEEPLOB_ALGORITHMS:
        defaults.update(
            sequence_length=100, hidden_size=64, conv_channels=32, inception_channels=64
        )
    if algorithm == "deeplob_classifier":
        if "class_threshold" not in parameters:
            raise ValueError("classification_threshold_required")
        defaults["class_threshold"] = parameters["class_threshold"]
    if set(parameters) - (
        set(defaults) | {"sequence_group_columns", "early_stopping_rounds"}
    ):
        raise ValueError("sequence_parameters_unknown")
    defaults.update(parameters)
    groups = defaults.get("sequence_group_columns", ())
    if (
        type(groups) not in (tuple, list)
        or any(
            type(column) is not str
            or not column
            or column in {"asset_id", "timestamp", "eligible"}
            for column in groups
        )
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("sequence_boundary_invalid")
    if "sequence_group_columns" in defaults:
        defaults["sequence_group_columns"] = list(groups)
    for name in (
        "sequence_length",
        "hidden_size",
        "num_layers",
        "batch_size",
        "epochs",
    ):
        if type(defaults[name]) is not int or defaults[name] < 1:
            raise ValueError(f"sequence_parameter_invalid: {name}")
    for name in ("learning_rate", "dropout"):
        value = defaults[name]
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"sequence_parameter_invalid: {name}")
    if defaults["learning_rate"] <= 0 or not 0 <= defaults["dropout"] < 1:
        raise ValueError("sequence_optimizer_parameters_invalid")
    device = defaults["device"]
    if type(device) is not str or not (
        device in {"cpu", "cuda", "auto"}
        or device.startswith("cuda:")
        and device[5:].isdigit()
    ):
        raise ValueError("model_device_invalid")
    if algorithm == "transformer_regressor":
        heads = defaults["n_heads"]
        if type(heads) is not int or heads < 1 or defaults["hidden_size"] % heads:
            raise ValueError("transformer_heads_invalid")
    if algorithm in DEEPLOB_ALGORITHMS:
        for name in ("conv_channels", "inception_channels"):
            if type(defaults[name]) is not int or defaults[name] < 1:
                raise ValueError("deeplob_channels_invalid")
    if algorithm == "deeplob_classifier":
        threshold = defaults["class_threshold"]
        if (
            type(threshold) not in (int, float)
            or not np.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError("classification_threshold_invalid")
    return defaults


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
    """Check scientific parameters; execution parallelism is descriptive metadata."""

    try:
        expected = dict(hyperparameters)
        actual = dict(resolved_hyperparameters)
        if algorithm in _PARALLEL_MODEL_ALGORITHMS:
            expected.pop("n_jobs", None)
            actual.pop("n_jobs", None)
        if algorithm == "xgboost_regressor":
            expected.setdefault("device", "cuda")
            expected.setdefault("tree_method", "hist")
        if algorithm in SEQUENCE_ALGORITHMS:
            expected = sequence_parameters(algorithm, expected)
        return canonical_identity_bytes(actual) == canonical_identity_bytes(expected)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class TrainingWeighting:
    """Equal timestamp mass, optionally decayed over market-calendar sessions."""

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
    """A model request with explicit resolved research parameters."""

    model_id: str
    model_algorithm: str
    input_factor_columns: tuple[str, ...] = ()
    input_feature_columns: tuple[str, ...] = ()
    target_column: str
    sample_filter_mode: SampleFilterMode = "eligible"
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int = DEFAULT_ROOT_SEED
    training_scaling: bool | None = None
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
        if "early_stopping_rounds" in hyperparameters:
            patience = hyperparameters["early_stopping_rounds"]
            if (
                self.model_algorithm not in {*SEQUENCE_ALGORITHMS, "xgboost_regressor"}
                or patience is not None
                and (type(patience) is not int or patience < 1)
            ):
                raise ValueError("early_stopping_rounds_invalid")
            if (
                patience is not None
                and self.model_algorithm == "xgboost_regressor"
                and (
                    hyperparameters.get("booster", "gbtree") not in {"gbtree", "dart"}
                    or hyperparameters.get("callbacks") is not None
                )
            ):
                raise ValueError("early_stopping_booster_or_callbacks_invalid")
        if (
            self.model_algorithm in _PARALLEL_MODEL_ALGORITHMS
            and "n_jobs" in hyperparameters
        ):
            jobs = hyperparameters["n_jobs"]
            if jobs is not None and type(jobs) is not int:
                raise ValueError("invalid_model_thread_count")
        scaling = self.training_scaling
        if scaling is None:
            scaling = self.model_algorithm in {
                "linear_regression",
                "lasso_regression",
                "ridge_regression",
                *SEQUENCE_ALGORITHMS,
            }
        if type(scaling) is not bool:
            raise ValueError("model_training_scaling_invalid")
        object.__setattr__(self, "training_scaling", scaling)
        if hyperparameters and self.model_algorithm == "linear_regression":
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
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")
        if not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be a uint32-compatible integer")
        if (
            "random_state" in hyperparameters
            and hyperparameters["random_state"] != self.random_seed
        ):
            raise ValueError("model_random_seed_conflicts_with_random_state")
        object.__setattr__(self, "hyperparameters", hyperparameters)
        if self.model_algorithm in SEQUENCE_ALGORITHMS:
            sequence_parameters(
                self.model_algorithm, _thaw_model_hyperparameters(hyperparameters)
            )
        if (
            self.model_algorithm in DEEPLOB_ALGORITHMS
            and self.input_columns != DEEPLOB_INPUT_COLUMNS
        ):
            raise ValueError("deeplob_requires_ordered_ten_level_book")

    @property
    def input_columns(self) -> tuple[str, ...]:
        return (*self.input_factor_columns, *self.input_feature_columns)


def _validate_early_stopping(value):
    fields = {
        "best_iteration",
        "iterations",
        "validation_rows",
        "validation_start",
        "validation_end",
        "label_available_max",
        "validation_cutoff",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("invalid_model_early_stopping")
    if any(
        type(value[key]) is not int
        for key in ("best_iteration", "iterations", "validation_rows")
    ) or not (
        0 <= value["best_iteration"] < value["iterations"]
        and value["validation_rows"] > 0
    ):
        raise ValueError("invalid_model_early_stopping")
    clocks = strict_timestamp_series(
        pd.Series(
            [
                value[key]
                for key in (
                    "validation_start",
                    "validation_end",
                    "label_available_max",
                    "validation_cutoff",
                )
            ]
        ),
        error="invalid_model_validation_maturity",
    )
    if not (
        clocks.iloc[0] <= clocks.iloc[1] <= clocks.iloc[3]
        and clocks.iloc[1] <= clocks.iloc[2] <= clocks.iloc[3]
    ):
        raise ValueError("invalid_model_validation_maturity")
    return clocks


def validate_model_metadata_core(value: object) -> dict[str, object]:
    """Validate the shared persisted model metadata contract."""

    if type(value) is not dict:
        raise ValueError("invalid_model_metadata")
    try:
        metadata = _json_native_object(value)
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
        "dependency_status",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "train_start",
        "training_asset_ids",
        "train_end",
        "label_availability_column",
        "label_available_max",
        "filter_columns_used",
        "filter_definition",
        "training_weighting",
        "training_scaling",
        "scaler",
    }
    _validate_non_empty_string(
        metadata.get("label_availability_column"),
        "model_label_availability_column_required",
    )
    strict_timestamp_series(
        pd.Series([metadata.get("label_available_max")]),
        error="invalid_model_label_maturity",
    )
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
        "training_target_mean",
        "learning_curve",
        "early_stopping",
    }
    if set(metadata).difference(allowed):
        raise ValueError("invalid_model_metadata")
    stopping = metadata.get("early_stopping")
    if stopping is not None:
        _validate_early_stopping(stopping)
    if "learning_curve" in metadata:
        curve = metadata["learning_curve"]
        if (
            type(curve) is not dict
            or set(curve)
            != {
                "metric",
                "training_loss_scope",
                "validation_loss_scope",
                "epochs",
                "sequence_lengths",
            }
            or metadata["model_algorithm"] not in SEQUENCE_ALGORITHMS
        ):
            raise ValueError("invalid_learning_curve")
        points = curve["epochs"]
        parameters = sequence_parameters(
            metadata["model_algorithm"], metadata["hyperparameters"]
        )
        if type(points) is not list or not (
            len(points) == parameters["epochs"]
            if parameters.get("early_stopping_rounds") is None
            else stopping is not None
            and 1 <= len(points) <= parameters["epochs"]
            and stopping["iterations"] == len(points)
        ):
            raise ValueError("invalid_learning_curve")
        for epoch, point in enumerate(points, 1):
            if (
                type(point) is not dict
                or set(point) != {"epoch", "training_loss", "validation_loss"}
                or point["epoch"] != epoch
            ):
                raise ValueError("invalid_learning_curve")
            for name in ("training_loss", "validation_loss"):
                loss = point[name]
                if loss is None and name == "validation_loss" and stopping is None:
                    continue
                if type(loss) not in (int, float) or not np.isfinite(loss) or loss < 0:
                    raise ValueError("invalid_learning_curve")
        if stopping is not None and stopping["best_iteration"] != min(
            range(len(points)), key=lambda i: points[i]["validation_loss"]
        ):
            raise ValueError("invalid_learning_curve")
    if "training_target_mean" in metadata and (
        type(metadata["training_target_mean"]) not in {int, float}
        or not np.isfinite(metadata["training_target_mean"])
    ):
        raise ValueError("invalid_model_training_target_mean")
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
    if (stopping is not None) != (
        metadata["hyperparameters"].get("early_stopping_rounds") is not None
    ):
        raise ValueError("invalid_model_early_stopping")
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
            training_weighting=metadata["training_weighting"],
            training_scaling=metadata["training_scaling"],
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
    if type(metadata["training_scaling"]) is not bool:
        raise ValueError("invalid_model_metadata")
    scaler = metadata["scaler"]
    if scaler is not None:
        if (
            type(scaler) is not dict
            or set(scaler) - {"medians", "native_missing"}
            != {
                "columns",
                "scaling",
                "weighted_mean",
                "weighted_population_scale",
            }
            or type(scaler.get("scaling")) is not bool
            or scaler["scaling"] != metadata["training_scaling"]
            or (
                "native_missing" in scaler
                and (scaler["native_missing"] is not True or "medians" in scaler)
            )
            or ("medians" in scaler and type(scaler["medians"]) is not list)
            or scaler["columns"] != metadata["input_columns"]
            or (
                "medians" in scaler
                and (
                    len(scaler["medians"]) != len(metadata["input_columns"])
                    or not all(
                        type(value) in {int, float} and np.isfinite(value)
                        for value in scaler["medians"]
                    )
                )
            )
            or type(scaler["weighted_mean"]) is not list
            or type(scaler["weighted_population_scale"]) is not list
            or len(scaler["columns"]) != len(scaler["weighted_mean"])
            or len(scaler["columns"]) != len(scaler["weighted_population_scale"])
            or any(
                type(number) not in {int, float} or not np.isfinite(float(number))
                for number in (
                    *scaler["weighted_mean"],
                    *scaler["weighted_population_scale"],
                )
            )
            or any(
                type(number) not in {int, float}
                or not np.isfinite(float(number))
                or float(number) <= 0.0
                for number in scaler["weighted_population_scale"]
            )
        ):
            raise ValueError("invalid_model_metadata")
        if not scaler["scaling"] and (
            any(value != 0 for value in scaler["weighted_mean"])
            or any(value != 1 for value in scaler["weighted_population_scale"])
        ):
            raise ValueError("invalid_model_metadata")
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
            or set(importance) != set(metadata["input_columns"])
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
    classification_keys = {f"validation_{name}" for name in CLASSIFICATION_METRICS}
    if type(trial) is not dict or set(trial) not in (
        _MODEL_SELECTION_TRIAL_METRIC_KEYS,
        _MODEL_SELECTION_TRIAL_METRIC_KEYS | classification_keys,
    ):
        raise ValueError("invalid_model_metadata")
    model_id = trial["model_id"]
    rows = trial["validation_rows"]
    if classification_keys <= set(trial):
        validate_classification_metrics(
            {name: trial[f"validation_{name}"] for name in CLASSIFICATION_METRICS}, rows
        )
    if type(model_id) is not str or not model_id or type(rows) is not int or rows < 0:
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
            or (
                ratio is not None
                and (
                    count < 2
                    or trial[f"validation_{prefix}"] is None
                    or std is None
                    or std == 0
                )
            )
            or (
                reason is not None
                and (
                    type(reason) is not str
                    or reason not in CORRELATION_SUMMARY_REASON_CODES
                )
            )
        ):
            raise ValueError("invalid_model_metadata")
    if trial["validation_mse"] is not None and (
        rows == 0
        or type(trial["validation_mse"]) not in {int, float}
        or not np.isfinite(float(trial["validation_mse"]))
        or trial["validation_mse"] < 0
    ):
        raise ValueError("invalid_model_metadata")


def validate_classification_metrics(metrics, rows):
    if set(metrics) != set(CLASSIFICATION_METRICS) or type(rows) is not int or rows < 0:
        raise ValueError("classification_metrics_invalid")
    for name, value in metrics.items():
        if rows == 0:
            if value is not None:
                raise ValueError("classification_metrics_invalid")
        elif (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or value < 0
            or (name in {"accuracy", "macro_f1"} and value > 1)
            or (name == "brier" and value > 2 + 1e-6)
        ):
            raise ValueError("classification_metrics_invalid")


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
        "training_weighting": spec.training_weighting.payload(),
        "training_scaling": spec.training_scaling,
    }


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
        end = pd.Timestamp(window.end)
        if end > cutoff:
            raise ValueError("fit_window_after_cutoff")
        lower, upper, closed = time_window_timestamp_interval(window)
        if last is not None and (lower < last[0] or (lower == last[0] and last[1])):
            raise ValueError("fit_windows_must_be_ordered_disjoint")
        last = (upper, closed)
    return windows


def _resolve_candidate_parameters(algorithm, parameters, n_train):
    values = dict(parameters)
    if "min_child_weight_fraction" in values:
        value = values.pop("min_child_weight_fraction")
        if (
            algorithm != "xgboost_regressor"
            or "min_child_weight" in values
            or type(value) not in {int, float}
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("model_sample_scaled_parameters_invalid")
        values["min_child_weight"] = n_train * value
        if not np.isfinite(values["min_child_weight"]):
            raise ValueError("model_sample_scaled_parameters_unrepresentable")
    return values


def validate_model_facts(facts, metadata=None, *, expected=None):
    """Validate and compare scientific model recipes independently of storage paths."""
    required = {
        "model",
        "market_id",
        "family_id",
        "candidate_parameters",
        "label_unit",
        "input_units",
    }
    if not isinstance(facts, Mapping) or not required <= set(facts):
        raise ValueError("model_facts_invalid")
    from quant_research_platform.contracts.ids import validate_market_id

    validate_market_id(facts["market_id"])
    _validate_non_empty_string(facts["family_id"], "model_family_id_invalid")
    spec = _model_spec(facts["model"])
    parameters = facts["candidate_parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("model_candidate_parameters_invalid")
    _freeze_model_hyperparameters(parameters)
    _resolve_candidate_parameters(spec.model_algorithm, parameters, 1)
    if spec.hyperparameters and canonical_identity_bytes(
        parameters
    ) != canonical_identity_bytes(_thaw_model_hyperparameters(spec.hyperparameters)):
        raise ValueError("model_candidate_parameters_mismatch")
    if type(facts["label_unit"]) is not str or not facts["label_unit"].strip():
        raise ValueError("model_label_unit_required")
    units = facts["input_units"]
    if not isinstance(units, Mapping) or set(units) - set(spec.input_columns):
        raise ValueError("model_input_units_invalid")
    for unit in units.values():
        if type(unit) is not str or not unit.strip():
            raise ValueError("model_input_units_invalid")
    _validate_target_definition(facts.get("target_definition"), spec.target_column)
    assets = facts.get("asset_ids")
    if assets is not None and (
        type(assets) is not list
        or not assets
        or any(type(asset) is not str or not asset for asset in assets)
        or len(assets) != len(set(assets))
    ):
        raise ValueError("model_training_asset_scope_invalid")
    if expected is not None:
        validate_model_facts(expected)
        for key in (
            "market_id",
            "family_id",
            "asset_ids",
            "candidate_parameters",
            "label_unit",
            "input_units",
            "target_definition",
        ):
            if facts.get(key) != expected.get(key):
                raise ValueError(f"model_scientific_facts_mismatch: {key}")
        if model_spec_payload(spec) != model_spec_payload(
            _model_spec(expected["model"])
        ):
            raise ValueError("model_scientific_facts_mismatch: model")
    if metadata is None:
        return
    if "rows_after_input_label_dropna" in metadata:
        expected_parameters = _resolve_candidate_parameters(
            spec.model_algorithm, parameters, metadata["rows_after_input_label_dropna"]
        )
        if canonical_identity_bytes(expected_parameters) != canonical_identity_bytes(
            metadata["hyperparameters"]
        ):
            raise ValueError("model_sample_scaled_parameters_mismatch")
        for key in (
            "model_algorithm",
            "target_column",
            "sample_filter_mode",
            "random_seed",
            "training_scaling",
        ):
            if getattr(spec, key) != metadata[key]:
                raise ValueError("model_facts_metadata_mismatch")
        if (
            spec.training_weighting.payload() != metadata["training_weighting"]
            or list(spec.input_columns) != metadata["input_columns"]
        ):
            raise ValueError("model_facts_metadata_mismatch")
    training_assets = metadata["training_asset_ids"]
    if (
        type(training_assets) is not list
        or not training_assets
        or any(type(asset) is not str or not asset for asset in training_assets)
        or len(training_assets) != len(set(training_assets))
    ):
        raise ValueError("native_model_training_asset_ids_required")
    if assets is not None and not set(training_assets) <= set(assets):
        raise ValueError("model_training_asset_scope_mismatch")
    validate_model_fit_facts(facts, metadata)


_PREDICTION_LOSS_REASONS = (
    "prediction_direction_score_not_return",
    "prediction_target_mismatch",
    "target_unit_missing",
    "prediction_target_unit_mismatch",
    "prediction_target_definition_missing",
    "prediction_target_definition_mismatch",
)


def _target_definition(declared, target, manifest=None):
    """Freeze only the consumed label definition, never its storage identity."""
    source = target
    transform, clip = "raw", "none"
    if target == "model_target":
        transform, clip = declared.get("target_transform"), declared.get("target_clip")
        if transform is None or clip is None:
            return (
                (manifest or {})
                .get("producer_diagnostics", {})
                .get("target_definitions", {})
                .get(target)
            )
        source = declared.get("target_label_column", "forward_return_20d")
    semantics = (
        (manifest or {})
        .get("producer_diagnostics", {})
        .get("label_semantics", {})
        .get(source)
    )
    return {
        "source_label": source,
        "target_transform": transform,
        "target_clip": clip,
        "label_semantics": None if semantics is None else dict(semantics),
        **(
            {
                "risk_model": FactorRiskModel.from_mapping(
                    declared.get("risk_model", {})
                ).to_dict()
            }
            if transform == "risk_residual"
            else {}
        ),
    }


def _comparable_target_definition(definition):
    if definition is None or definition.get("target_transform") != "risk_residual":
        return definition
    model = FactorRiskModel.from_mapping(definition.get("risk_model", {}))
    # Covariance windows do not change the cross-sectional label regression.
    return {
        **definition,
        "risk_model": {
            "style_columns": list(model.style_columns),
            "config": {
                name: getattr(model.config, name)
                for name in ("weight_column", "weighting", "minimum_cross_section")
            },
        },
    }


def _prediction_loss_reason(
    facts, target, declared, manifest=None, *, model_target=None, target_unit=None
):
    """Check label-scale loss comparability independently of rank/IC evidence."""
    if facts["model"]["model_algorithm"] == "deeplob_classifier":
        return "prediction_direction_score_not_return"
    return _prediction_target_reason(
        facts,
        target,
        declared,
        manifest,
        model_target=model_target,
        target_unit=target_unit,
    )


def _prediction_target_reason(
    facts, target, declared, manifest=None, *, model_target=None, target_unit=None
):
    """Compare the economic target independently of the model output scale."""
    trained_target = model_target or facts["model"]["target_column"]
    trained = _comparable_target_definition(facts.get("target_definition"))
    evaluated = _comparable_target_definition(
        _target_definition(declared, target, manifest)
    )
    if trained_target != target and (trained is None or trained != evaluated):
        return "prediction_target_mismatch"
    unit = declared.get("column_units", {}).get(target, target_unit)
    if unit is None or facts.get("label_unit") is None:
        return "target_unit_missing"
    if facts["label_unit"] != unit:
        return "prediction_target_unit_mismatch"
    if (
        trained is None
        or evaluated is None
        or (trained["label_semantics"] is None)
        != (evaluated["label_semantics"] is None)
    ):
        return "prediction_target_definition_missing"
    if trained != evaluated:
        return "prediction_target_definition_mismatch"
    return None


def _validate_target_definition(definition, target):
    if definition is None:
        return
    if (
        type(definition) is not dict
        or set(definition) - {"risk_model"}
        != {"source_label", "target_transform", "target_clip", "label_semantics"}
        or type(definition["source_label"]) is not str
        or not definition["source_label"]
        or target != "model_target"
        and definition["source_label"] != target
        or type(definition["target_transform"]) is not str
        or type(definition["target_clip"]) is not str
        or definition["target_transform"]
        not in (
            {
                "raw",
                "date_zscore",
                "percentile",
                "industry_size_residual",
                "risk_residual",
            }
            if target == "model_target"
            else {"raw"}
        )
        or definition["target_clip"]
        not in (
            {"none", "mad", "quantile_1_99"} if target == "model_target" else {"none"}
        )
        or definition["label_semantics"] is not None
        and type(definition["label_semantics"]) is not dict
    ):
        raise ValueError("model_target_definition_invalid")
    if "risk_model" in definition:
        if definition["target_transform"] != "risk_residual":
            raise ValueError("model_target_definition_invalid")
        FactorRiskModel.from_mapping(definition["risk_model"])
    canonical_identity_bytes(definition)


def validate_model_fit_facts(facts, metadata):
    """Validate chronology using only realized training and decision facts."""
    cutoff = strict_timestamp_series(
        pd.Series([facts["fit_cutoff"]]), error="invalid_fit_cutoff"
    ).iloc[0]
    maturity = strict_timestamp_series(
        pd.Series([metadata["label_available_max"]]),
        error="invalid_model_label_maturity",
    ).iloc[0]
    if maturity > cutoff:
        raise ValueError("model_training_label_after_fit_cutoff")
    windows = _fit_windows(facts["fit_windows"], cutoff)
    bounds = strict_timestamp_series(
        pd.Series([metadata["train_start"], metadata["train_end"]]),
        error="invalid_model_training_window",
    )
    if bounds.iloc[0] > bounds.iloc[1] or bounds.iloc[1] > cutoff:
        raise ValueError("model_training_window_invalid")
    from quant_research_platform.contracts.time import time_window_timestamp_interval

    def contains(window, timestamp):
        start, end, closed = time_window_timestamp_interval(window)
        return pd.Timestamp(start) <= timestamp and (
            timestamp <= pd.Timestamp(end) if closed else timestamp < pd.Timestamp(end)
        )

    if not all(any(contains(window, bound) for window in windows) for bound in bounds):
        raise ValueError("model_training_window_outside_declared_fit")
    enabled = facts["candidate_parameters"].get("early_stopping_rounds") is not None
    stopping = metadata.get("early_stopping")
    if enabled != (stopping is not None):
        raise ValueError("invalid_model_early_stopping")
    if stopping is not None:
        clocks = _validate_early_stopping(stopping)
        validation = facts.get("validation")
        if type(validation) is not dict or set(validation) != {"windows", "cutoff"}:
            raise ValueError("model_validation_facts_required")
        validation_cutoff = strict_timestamp_series(
            pd.Series([validation["cutoff"]]), error="invalid_validation_cutoff"
        ).iloc[0]
        validation_windows = _fit_windows(validation["windows"], validation_cutoff)
        if (
            clocks.iloc[3] != validation_cutoff
            or cutoff >= pd.Timestamp(validation_windows[0].start)
            or cutoff >= clocks.iloc[0]
            or not all(
                any(contains(window, bound) for window in validation_windows)
                for bound in clocks.iloc[:2]
            )
        ):
            raise ValueError("model_validation_training_overlap")
    decision = facts.get("selection_cutoff")
    if (
        decision is not None
        and strict_timestamp_series(
            pd.Series([decision]), error="invalid_selection_cutoff"
        ).iloc[0]
        > cutoff
    ):
        raise ValueError("model_selection_after_fit_cutoff")


def _validate_model_prediction_window(facts, window):
    """Require predictions to follow the actual fitting decision."""
    if pd.Timestamp(window.start) <= pd.Timestamp(facts["fit_cutoff"]):
        raise ValueError("prediction_must_follow_fit_cutoff")
    if facts["candidate_parameters"].get("early_stopping_rounds") is not None:
        validation = facts.get("validation")
        if not isinstance(validation, Mapping) or "cutoff" not in validation:
            raise ValueError("model_validation_facts_required")
        cutoff = strict_timestamp_series(
            pd.Series([validation["cutoff"]]), error="invalid_validation_cutoff"
        ).iloc[0]
        if pd.Timestamp(window.start) <= cutoff:
            raise ValueError("prediction_must_follow_validation_cutoff")


def validate_model_input_roles(
    declared,
    spec,
    *,
    market_id,
    include_target=False,
    selected_columns=None,
    input_units=None,
):
    """Check consumed roles and explicit market/unit conflicts in the opened input."""
    validate_input_market(declared, market_id)
    consumed = {*spec.input_columns, *((spec.target_column,) if include_target else ())}
    if selected_columns is not None and consumed - set(selected_columns):
        raise ValueError("model_input_projection_missing")
    roles = declared["column_roles"]
    _validate_asset_time_key(declared["primary_key"], roles)
    for column in spec.input_factor_columns:
        if roles.get(column) != "factor":
            raise ValueError("model_factor_role_mismatch")
    for column in spec.input_feature_columns:
        if roles.get(column) != "feature":
            raise ValueError("model_feature_role_mismatch")
    if include_target and roles.get(spec.target_column) != "label":
        raise ValueError("model_target_role_mismatch")
    actual_units = declared.get("column_units", {})
    for column, unit in (input_units or {}).items():
        if column in actual_units and actual_units[column] != unit:
            raise ValueError(f"model_input_unit_mismatch: {column}")
