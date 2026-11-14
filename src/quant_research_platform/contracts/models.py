"""Explicit model capability and fitting contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

import numpy as np
import pandas as pd

from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.experiments import (
    DEFAULT_ROOT_SEED,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import (
    FILTER_COLUMNS,
    SampleFilterMode,
    reject_filter_input_columns,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.ids import validate_input_market
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
        "weight_decay": 0.0,
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
            or column in {"asset_id", "timestamp", *FILTER_COLUMNS}
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
    ):
        if type(defaults[name]) is not int or defaults[name] < 1:
            raise ValueError(f"sequence_parameter_invalid: {name}")
    epochs = defaults["epochs"]
    if epochs is None:
        if defaults.get("early_stopping_rounds") is None:
            raise ValueError("unbounded_training_requires_early_stopping")
    elif type(epochs) is not int or epochs < 1:
        raise ValueError("sequence_parameter_invalid: epochs")
    for name in ("learning_rate", "dropout", "weight_decay"):
        value = defaults[name]
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"sequence_parameter_invalid: {name}")
    if (
        defaults["learning_rate"] <= 0
        or not 0 <= defaults["dropout"] < 1
        or defaults["weight_decay"] < 0
    ):
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
            self.model_algorithm == "xgboost_regressor"
            and "n_estimators" in hyperparameters
            and hyperparameters["n_estimators"] is None
            and hyperparameters.get("early_stopping_rounds") is None
        ):
            raise ValueError("unbounded_training_requires_early_stopping")
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
            if self.target_column in sequence_context_columns(
                self.model_algorithm, hyperparameters
            ):
                raise ValueError("sequence_context_uses_future_outcome")
        if (
            self.model_algorithm in DEEPLOB_ALGORITHMS
            and self.input_columns != DEEPLOB_INPUT_COLUMNS
        ):
            raise ValueError("deeplob_requires_ordered_ten_level_book")

    @property
    def input_columns(self) -> tuple[str, ...]:
        return (*self.input_factor_columns, *self.input_feature_columns)


def _validate_a_share_model_family_ids(family_ids) -> None:
    """Keep the A-share champion selector distinct from named model families."""
    if "champion" in family_ids:
        raise ValueError("a_share_model_id_reserved")


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


def compare_model_facts(facts, expected):
    """Compare the published recipe with a new refit or portfolio request."""
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
    if facts["model"] != expected["model"]:
        raise ValueError("model_scientific_facts_mismatch: model")


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
            {"risk_model": declared["risk_model"]}
            if transform == "risk_residual"
            else {}
        ),
    }


def _comparable_target_definition(definition):
    if definition is None or definition.get("target_transform") != "risk_residual":
        return definition
    model = definition["risk_model"]
    # Covariance windows do not change the cross-sectional label regression.
    return {
        **definition,
        "risk_model": {
            "style_columns": model["style_columns"],
            "config": {
                name: model["config"][name]
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
    recipe,
    *,
    market_id,
    include_target=False,
    selected_columns=None,
    input_units=None,
    hyperparameters=None,
):
    """Check consumed roles and explicit market/unit conflicts in the opened input."""
    validate_input_market(declared, market_id)
    factors = recipe.get("input_factor_columns", ())
    features = recipe.get("input_feature_columns", ())
    groups = sequence_context_columns(
        recipe["model_algorithm"],
        recipe.get("hyperparameters", {})
        if hyperparameters is None
        else hyperparameters,
    )
    consumed = {
        *factors,
        *features,
        *groups,
        *((recipe["target_column"],) if include_target else ()),
    }
    if selected_columns is not None and consumed - set(selected_columns):
        raise ValueError("model_input_projection_missing")
    roles = declared["column_roles"]
    _validate_asset_time_key(declared["primary_key"], roles)
    labels = {recipe["target_column"]} | {
        column for column, role in roles.items() if role == "label"
    }
    availability = declared.get("label_availability_columns", {})
    maturity = {
        availability.get(label, training_availability_columns(label)[0])
        for label in labels
    }
    if set(groups) & (labels | maturity | set(FILTER_COLUMNS)):
        raise ValueError("sequence_context_uses_future_outcome")
    if any(
        roles.get(column) not in {"context", "key", "factor", "feature"}
        for column in groups
    ):
        raise ValueError("sequence_context_role_invalid")
    for column in factors:
        if roles.get(column) != "factor":
            raise ValueError("model_factor_role_mismatch")
    for column in features:
        if roles.get(column) != "feature":
            raise ValueError("model_feature_role_mismatch")
    if include_target and roles.get(recipe["target_column"]) != "label":
        raise ValueError("model_target_role_mismatch")
    actual_units = declared.get("column_units", {})
    for column, unit in (input_units or {}).items():
        if column in actual_units and actual_units[column] != unit:
            raise ValueError(f"model_input_unit_mismatch: {column}")
