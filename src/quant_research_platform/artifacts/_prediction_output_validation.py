"""Prediction output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts._output_common_validation import (
    require_finite_metadata_number,
    require_metadata_string,
    require_metadata_string_list,
    require_non_negative_metadata_int,
    require_optional_metadata_string,
    validate_filter_metadata,
    validate_metadata_row_count_chain,
)
from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.contracts.evaluations import (
    CORRELATION_SUMMARY_REASON_CODES,
)
from quant_research_platform.contracts.models import (
    ModelSpec,
    _resolved_model_hyperparameters_match,
)
from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
    PredictionSpec,
)
from quant_research_platform.contracts.validation import strict_timestamp_series

_MODEL_METADATA_BASE_KEYS = frozenset(
    {
        "model_algorithm",
        "sample_filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "oos_start",
        "oos_end",
        "input_factor_columns",
        "input_feature_columns",
        "input_columns",
        "input_count",
        "label_column",
        "hyperparameters",
        "resolved_hyperparameters",
        "random_seed",
        "max_train_rows",
        "purge_gap_steps",
        "dependency_status",
        "coefficient_count",
    }
)
_MODEL_METADATA_FEATURE_KEYS = frozenset(
    {
        "feature_importance",
        "feature_importance_status",
    }
)
_MODEL_METADATA_OPTIONAL_KEYS = frozenset({"model_selection"})
_MODEL_SELECTION_KEYS = frozenset(
    {
        "status",
        "configured_model_id",
        "selected_model_id",
        "selected_model_input_columns",
        "candidate_model_ids",
        "selection_metric",
        "selection_direction",
        "validation_start",
        "validation_end",
        "candidate_count",
        "trial_metrics",
    }
)
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
)
_MODEL_SELECTION_REASON_CODES = CORRELATION_SUMMARY_REASON_CODES
_PREDICTION_METADATA_KEYS = frozenset(
    {
        "prediction_mode",
        "sample_filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_dropna",
        "prediction_rows",
        "train_end",
        "test_start",
        "requested_test_end",
        "effective_test_end",
        "is_oos",
        "is_final_oos_window",
        "input_columns",
        "input_count",
        "prediction_column",
    }
)
_PREDICTION_OUTPUT_METADATA_KEYS = frozenset(
    {
        "prediction_id",
        "model_id",
        "research_dataset_id",
        "prediction_mode",
        "is_oos",
        "train_end",
        "test_start",
        "test_end",
        "is_final_oos_window",
        "prediction_column",
        "primary_key",
        "sample_filter_mode",
        "model_metadata",
        "prediction_metadata",
    }
)


def validate_model_metadata_shape(metadata: dict[str, object]) -> dict[str, object]:
    if type(metadata) is not dict:
        raise ValueError("invalid_model_metadata")
    metadata_keys = set(metadata)
    feature_keys = metadata_keys.intersection(_MODEL_METADATA_FEATURE_KEYS)
    extra_keys = (
        metadata_keys.difference(_MODEL_METADATA_BASE_KEYS)
        .difference(_MODEL_METADATA_FEATURE_KEYS)
        .difference(_MODEL_METADATA_OPTIONAL_KEYS)
    )
    if not _MODEL_METADATA_BASE_KEYS.issubset(metadata_keys) or len(feature_keys) != 1:
        raise ValueError("invalid_model_metadata")
    if extra_keys:
        raise ValueError("invalid_model_metadata")
    require_non_negative_metadata_int(metadata, "rows_input", "invalid_model_metadata")
    require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_model_metadata",
    )
    require_non_negative_metadata_int(
        metadata,
        "rows_after_input_label_dropna",
        "invalid_model_metadata",
    )
    input_factor_columns = require_metadata_string_list(
        metadata,
        "input_factor_columns",
        "invalid_model_metadata",
        allow_empty=True,
    )
    input_feature_columns = require_metadata_string_list(
        metadata,
        "input_feature_columns",
        "invalid_model_metadata",
        allow_empty=True,
    )
    input_columns = require_metadata_string_list(
        metadata,
        "input_columns",
        "invalid_model_metadata",
    )
    input_count = require_non_negative_metadata_int(
        metadata,
        "input_count",
        "invalid_model_metadata",
    )
    coefficient_count = require_non_negative_metadata_int(
        metadata,
        "coefficient_count",
        "invalid_model_metadata",
    )
    require_non_negative_metadata_int(
        metadata,
        "purge_gap_steps",
        "invalid_model_metadata",
    )
    if metadata["max_train_rows"] is not None:
        require_non_negative_metadata_int(
            metadata,
            "max_train_rows",
            "invalid_model_metadata",
        )
    if metadata["random_seed"] is not None and type(metadata["random_seed"]) is not int:
        raise ValueError("invalid_model_metadata")
    if type(metadata["hyperparameters"]) is not dict or type(
        metadata["resolved_hyperparameters"]
    ) is not dict:
        raise ValueError("invalid_model_metadata")
    try:
        render_qrp_json_object(
            {
                "hyperparameters": metadata["hyperparameters"],
                "resolved_hyperparameters": metadata["resolved_hyperparameters"],
            }
        )
    except ValueError as exc:
        raise ValueError("invalid_model_metadata") from exc
    model_algorithm = require_metadata_string(
        metadata,
        "model_algorithm",
        "invalid_model_metadata",
    )
    if not _resolved_model_hyperparameters_match(
        model_algorithm,
        metadata["hyperparameters"],
        metadata["resolved_hyperparameters"],
    ):
        raise ValueError("invalid_model_metadata")
    dependency_status = metadata["dependency_status"]
    if (
        type(dependency_status) is not dict
        or set(dependency_status) != {"status", "package", "version"}
        or dependency_status["status"] != "available"
        or type(dependency_status["package"]) is not str
        or not dependency_status["package"]
        or type(dependency_status["version"]) is not str
        or not dependency_status["version"]
    ):
        raise ValueError("invalid_model_metadata")
    if "feature_importance" in metadata:
        feature_importance = metadata["feature_importance"]
        if type(feature_importance) is not dict or not feature_importance:
            raise ValueError("invalid_model_metadata")
        for column, value in feature_importance.items():
            if type(column) is not str or not column:
                raise ValueError("invalid_model_metadata")
            require_finite_metadata_number(
                {"value": value}, "value", "invalid_model_metadata"
            )
        if set(feature_importance) != set(input_columns):
            raise ValueError("invalid_model_metadata")
    if "feature_importance_status" in metadata:
        require_metadata_string(
            metadata,
            "feature_importance_status",
            "invalid_model_metadata",
        )
    if "model_selection" in metadata:
        validate_model_selection_metadata_shape(metadata["model_selection"])
    if (
        input_columns != [*input_factor_columns, *input_feature_columns]
        or input_count != len(input_columns)
        or coefficient_count < 1
    ):
        raise ValueError("invalid_model_metadata")
    require_metadata_string(metadata, "sample_filter_mode", "invalid_model_metadata")
    require_metadata_string(metadata, "filter_definition", "invalid_model_metadata")
    require_metadata_string(metadata, "label_column", "invalid_model_metadata")
    require_metadata_string(metadata, "train_start", "invalid_model_metadata")
    require_metadata_string(metadata, "train_end", "invalid_model_metadata")
    require_optional_metadata_string(
        metadata,
        "test_start",
        "invalid_model_metadata",
    )
    require_optional_metadata_string(metadata, "test_end", "invalid_model_metadata")
    require_optional_metadata_string(metadata, "oos_start", "invalid_model_metadata")
    require_optional_metadata_string(metadata, "oos_end", "invalid_model_metadata")
    require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_model_metadata",
        allow_empty=True,
    )
    return metadata


def validate_model_selection_metadata_shape(metadata: object) -> None:
    if type(metadata) is not dict or set(metadata) != _MODEL_SELECTION_KEYS:
        raise ValueError("invalid_model_metadata")
    if metadata["status"] != "selected":
        raise ValueError("invalid_model_metadata")
    for key in (
        "configured_model_id",
        "selected_model_id",
        "selection_metric",
        "selection_direction",
        "validation_start",
        "validation_end",
    ):
        require_metadata_string(metadata, key, "invalid_model_metadata")
    if metadata["selection_direction"] not in {"maximize", "minimize"}:
        raise ValueError("invalid_model_metadata")
    if (
        MODEL_SELECTION_METRIC_DIRECTIONS.get(metadata["selection_metric"])
        != metadata["selection_direction"]
    ):
        raise ValueError("invalid_model_metadata")
    candidate_model_ids = require_metadata_string_list(
        metadata,
        "candidate_model_ids",
        "invalid_model_metadata",
    )
    require_metadata_string_list(
        metadata,
        "selected_model_input_columns",
        "invalid_model_metadata",
    )
    candidate_count = require_non_negative_metadata_int(
        metadata,
        "candidate_count",
        "invalid_model_metadata",
    )
    if candidate_count < 1 or metadata["selected_model_id"] not in candidate_model_ids:
        raise ValueError("invalid_model_metadata")
    if candidate_count != len(candidate_model_ids):
        raise ValueError("invalid_model_metadata")
    trial_metrics = metadata["trial_metrics"]
    if type(trial_metrics) is not list or len(trial_metrics) != candidate_count:
        raise ValueError("invalid_model_metadata")
    trial_model_ids: list[str] = []
    for trial in trial_metrics:
        if (
            type(trial) is not dict
            or set(trial) != _MODEL_SELECTION_TRIAL_METRIC_KEYS
        ):
            raise ValueError("invalid_model_metadata")
        trial_model_ids.append(
            require_metadata_string(trial, "model_id", "invalid_model_metadata")
        )
        validation_rows = require_non_negative_metadata_int(
            trial,
            "validation_rows",
            "invalid_model_metadata",
        )
        if validation_rows < 1:
            raise ValueError("invalid_model_metadata")
        for key, reason_key in (
            ("validation_ic", "validation_ic_reason"),
            ("validation_rankic", "validation_rankic_reason"),
        ):
            value = trial[key]
            reason = require_optional_metadata_string(
                trial, reason_key, "invalid_model_metadata"
            )
            if reason is not None and reason not in _MODEL_SELECTION_REASON_CODES:
                raise ValueError("invalid_model_metadata")
            if validation_rows < 2:
                if value is not None or reason != "no_valid_observations":
                    raise ValueError("invalid_model_metadata")
                continue
            if value is None:
                if reason is None:
                    raise ValueError("invalid_model_metadata")
            else:
                numeric_value = require_finite_metadata_number(
                    trial,
                    key,
                    "invalid_model_metadata",
                )
                if abs(numeric_value) > 1.0:
                    raise ValueError("invalid_model_metadata")
                if reason is not None:
                    raise ValueError("invalid_model_metadata")
        require_finite_metadata_number(
            trial, "validation_mse", "invalid_model_metadata"
        )
    if trial_model_ids != candidate_model_ids:
        raise ValueError("invalid_model_metadata")


def validate_prediction_output_metadata_shape(
    metadata: object,
) -> dict[str, object]:
    if type(metadata) is not dict or set(metadata) != _PREDICTION_OUTPUT_METADATA_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    for key in (
        "prediction_id",
        "model_id",
        "research_dataset_id",
        "prediction_mode",
        "train_end",
        "test_start",
        "prediction_column",
        "sample_filter_mode",
    ):
        require_metadata_string(metadata, key, "output_manifest_metadata_invalid")
    require_optional_metadata_string(
        metadata, "test_end", "output_manifest_metadata_invalid"
    )
    if type(metadata["is_oos"]) is not bool or type(
        metadata["is_final_oos_window"]
    ) is not bool:
        raise ValueError("output_manifest_metadata_invalid")
    require_metadata_string_list(
        metadata, "primary_key", "output_manifest_metadata_invalid"
    )
    model_metadata = metadata["model_metadata"]
    prediction_metadata = metadata["prediction_metadata"]
    if type(model_metadata) is not dict or type(prediction_metadata) is not dict:
        raise ValueError("output_manifest_metadata_invalid")
    try:
        validate_model_metadata_shape(model_metadata)
        validate_prediction_metadata_shape(prediction_metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc
    if (
        model_metadata["sample_filter_mode"] != metadata["sample_filter_mode"]
        or prediction_metadata["sample_filter_mode"]
        != metadata["sample_filter_mode"]
        or prediction_metadata["prediction_mode"] != metadata["prediction_mode"]
        or prediction_metadata["prediction_column"] != metadata["prediction_column"]
        or prediction_metadata["train_end"] != metadata["train_end"]
        or prediction_metadata["test_start"] != metadata["test_start"]
        or prediction_metadata["requested_test_end"] != metadata["test_end"]
        or prediction_metadata["is_oos"] is not metadata["is_oos"]
        or prediction_metadata["is_final_oos_window"]
        is not metadata["is_final_oos_window"]
    ):
        raise ValueError("output_manifest_metadata_invalid")
    try:
        model_selection = model_metadata.get("model_selection")
        candidate_model_ids: tuple[str, ...] = ()
        model_selection_metric = "validation_rankic"
        model_selection_validation_start = None
        model_selection_validation_end = None
        if type(model_selection) is dict:
            persisted_candidates = model_selection["candidate_model_ids"]
            candidate_model_ids = tuple(persisted_candidates[1:])
            model_selection_metric = model_selection["selection_metric"]
            model_selection_validation_start = model_selection["validation_start"]
            model_selection_validation_end = model_selection["validation_end"]
        persisted_model_id = (
            model_selection["selected_model_id"]
            if type(model_selection) is dict
            else metadata["model_id"]
        )
        ModelSpec(
            model_id=persisted_model_id,
            model_algorithm=model_metadata["model_algorithm"],
            input_factor_columns=tuple(model_metadata["input_factor_columns"]),
            input_feature_columns=tuple(model_metadata["input_feature_columns"]),
            label_column=model_metadata["label_column"],
            sample_filter_mode=model_metadata["sample_filter_mode"],
            hyperparameters=model_metadata["hyperparameters"],
            random_seed=model_metadata["random_seed"],
            purge_gap_steps=model_metadata["purge_gap_steps"],
            max_train_rows=model_metadata["max_train_rows"],
        )
        spec = PredictionSpec(
            prediction_id=metadata["prediction_id"],
            model_id=metadata["model_id"],
            research_dataset_id=metadata["research_dataset_id"],
            train_end=metadata["train_end"],
            test_start=metadata["test_start"],
            test_end=metadata["test_end"],
            prediction_column=metadata["prediction_column"],
            primary_key=tuple(metadata["primary_key"]),
            prediction_mode=metadata["prediction_mode"],
            sample_filter_mode=metadata["sample_filter_mode"],
            is_oos=metadata["is_oos"],
            is_final_oos_window=metadata["is_final_oos_window"],
            candidate_model_ids=candidate_model_ids,
            model_selection_metric=model_selection_metric,
            model_selection_validation_start=model_selection_validation_start,
            model_selection_validation_end=model_selection_validation_end,
        )
        if spec.is_final_oos_window is not metadata["is_final_oos_window"]:
            raise ValueError("prediction_window_not_canonical")
        validate_model_metadata_consistency(model_metadata, spec=spec)
        validate_prediction_metadata_consistency_from_summary(
            prediction_metadata,
            spec=spec,
            row_count=prediction_metadata["prediction_rows"],
            effective_test_end=prediction_metadata["effective_test_end"],
        )
        validate_model_prediction_metadata_consistency(
            model_metadata,
            prediction_metadata,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc
    return metadata


def validate_prediction_metadata_shape(
    metadata: dict[str, object],
) -> dict[str, object]:
    if type(metadata) is not dict or set(metadata) != _PREDICTION_METADATA_KEYS:
        raise ValueError("invalid_prediction_metadata")
    require_non_negative_metadata_int(
        metadata,
        "rows_input",
        "invalid_prediction_metadata",
    )
    require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_prediction_metadata",
    )
    require_non_negative_metadata_int(
        metadata,
        "rows_after_input_dropna",
        "invalid_prediction_metadata",
    )
    require_non_negative_metadata_int(
        metadata,
        "prediction_rows",
        "invalid_prediction_metadata",
    )
    input_columns = require_metadata_string_list(
        metadata,
        "input_columns",
        "invalid_prediction_metadata",
    )
    input_count = require_non_negative_metadata_int(
        metadata,
        "input_count",
        "invalid_prediction_metadata",
    )
    if input_count != len(input_columns):
        raise ValueError("invalid_prediction_metadata")
    require_metadata_string(
        metadata,
        "prediction_mode",
        "invalid_prediction_metadata",
    )
    require_metadata_string(
        metadata,
        "sample_filter_mode",
        "invalid_prediction_metadata",
    )
    require_metadata_string(
        metadata,
        "filter_definition",
        "invalid_prediction_metadata",
    )
    require_metadata_string(metadata, "test_start", "invalid_prediction_metadata")
    require_metadata_string(metadata, "train_end", "invalid_prediction_metadata")
    if metadata["requested_test_end"] is not None:
        require_metadata_string(
            metadata,
            "requested_test_end",
            "invalid_prediction_metadata",
        )
    require_metadata_string(
        metadata,
        "effective_test_end",
        "invalid_prediction_metadata",
    )
    require_metadata_string(
        metadata,
        "prediction_column",
        "invalid_prediction_metadata",
    )
    require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_prediction_metadata",
        allow_empty=True,
    )
    if type(metadata["is_oos"]) is not bool or type(
        metadata["is_final_oos_window"]
    ) is not bool:
        raise ValueError("invalid_prediction_metadata")
    return metadata


def validate_model_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
) -> None:
    input_count = require_non_negative_metadata_int(
        metadata,
        "input_count",
        "model_metadata_inconsistent",
    )
    input_columns = metadata["input_columns"]
    if type(input_columns) is not list or input_count != len(input_columns):
        raise ValueError("model_metadata_inconsistent")
    validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_label_dropna"),
        "model_metadata_inconsistent",
    )
    validate_filter_metadata(
        mode=metadata["sample_filter_mode"],
        columns=metadata["filter_columns_used"],
        definition=metadata["filter_definition"],
        error_name="model_metadata_inconsistent",
    )
    train_start = pd.Timestamp(metadata["train_start"])
    train_end = pd.Timestamp(metadata["train_end"])
    if train_start > train_end or train_end > pd.Timestamp(spec.train_end):
        raise ValueError("model_metadata_inconsistent")
    if (
        metadata["test_start"] != spec.test_start
        or metadata["test_end"] != spec.test_end
        or metadata["oos_start"] != spec.test_start
    ):
        raise ValueError("model_metadata_inconsistent")
    if (
        require_non_negative_metadata_int(
            metadata,
            "coefficient_count",
            "model_metadata_inconsistent",
        )
        <= 0
    ):
        raise ValueError("model_metadata_inconsistent")
    validate_model_selection_metadata_consistency(metadata, spec=spec)


def validate_model_selection_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
) -> None:
    model_selection = metadata.get("model_selection")
    if not spec.candidate_model_ids:
        if model_selection is not None:
            raise ValueError("model_metadata_inconsistent")
        return
    if type(model_selection) is not dict:
        raise ValueError("model_metadata_inconsistent")
    candidate_model_ids = [spec.model_id, *spec.candidate_model_ids]
    if model_selection["configured_model_id"] != spec.model_id:
        raise ValueError("model_metadata_inconsistent")
    if model_selection["candidate_model_ids"] != candidate_model_ids:
        raise ValueError("model_metadata_inconsistent")
    if model_selection["selected_model_id"] not in candidate_model_ids:
        raise ValueError("model_metadata_inconsistent")
    if model_selection["selection_metric"] != spec.model_selection_metric:
        raise ValueError("model_metadata_inconsistent")
    if (
        model_selection["validation_start"]
        != spec.model_selection_validation_start
        or model_selection["validation_end"] != spec.model_selection_validation_end
    ):
        raise ValueError("model_metadata_inconsistent")
    if len(model_selection["trial_metrics"]) != model_selection["candidate_count"]:
        raise ValueError("model_metadata_inconsistent")
    if [
        trial["model_id"] for trial in model_selection["trial_metrics"]
    ] != candidate_model_ids:
        raise ValueError("model_metadata_inconsistent")
    if model_selection["selected_model_input_columns"] != metadata["input_columns"]:
        raise ValueError("model_metadata_inconsistent")


def validate_prediction_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    frame: pd.DataFrame,
) -> None:
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="prediction_metadata_inconsistent"
    )
    effective_test_end = pd.Timestamp(timestamps.max()).date().isoformat()
    validate_prediction_metadata_consistency_from_summary(
        metadata,
        spec=spec,
        row_count=len(frame),
        effective_test_end=effective_test_end,
    )


def validate_prediction_metadata_consistency_from_summary(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    row_count: int,
    effective_test_end: str,
) -> None:
    expected = {
        "prediction_rows": row_count,
        "prediction_column": spec.prediction_column,
        "prediction_mode": spec.prediction_mode,
        "sample_filter_mode": spec.sample_filter_mode,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "requested_test_end": spec.test_end,
        "is_oos": spec.is_oos,
        "is_final_oos_window": spec.is_final_oos_window,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError("prediction_metadata_inconsistent")
    validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_dropna"),
        "prediction_metadata_inconsistent",
    )
    prediction_rows = require_non_negative_metadata_int(
        metadata,
        "prediction_rows",
        "prediction_metadata_inconsistent",
    )
    rows_after_input_dropna = require_non_negative_metadata_int(
        metadata,
        "rows_after_input_dropna",
        "prediction_metadata_inconsistent",
    )
    if prediction_rows > rows_after_input_dropna:
        raise ValueError("prediction_metadata_inconsistent")
    validate_filter_metadata(
        mode=metadata["sample_filter_mode"],
        columns=metadata["filter_columns_used"],
        definition=metadata["filter_definition"],
        error_name="prediction_metadata_inconsistent",
    )
    if metadata["effective_test_end"] != effective_test_end:
        raise ValueError("prediction_metadata_inconsistent")
    requested_test_end = metadata["requested_test_end"]
    if requested_test_end is not None and pd.Timestamp(
        effective_test_end
    ) > pd.Timestamp(requested_test_end):
        raise ValueError("prediction_metadata_inconsistent")


def validate_model_prediction_metadata_consistency(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> None:
    if model_metadata["input_columns"] != prediction_metadata["input_columns"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["input_count"] != prediction_metadata["input_count"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["oos_end"] != prediction_metadata["effective_test_end"]:
        raise ValueError("prediction_metadata_inconsistent")


def model_metadata_with_effective_prediction_end(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **model_metadata,
        "oos_end": prediction_metadata["effective_test_end"],
    }
