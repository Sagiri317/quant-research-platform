"""Prediction output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts._output_common_validation import (
    require_metadata_string,
    require_metadata_string_list,
    require_non_negative_metadata_int,
    require_optional_metadata_string,
    validate_filter_metadata,
    validate_metadata_row_count_chain,
)
from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
    PredictionSpec,
)

_MODEL_METADATA_BASE_KEYS = frozenset(
    {
        "contract_version",
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
        "validation_mse",
    }
)
_PREDICTION_METADATA_KEYS = frozenset(
    {
        "contract_version",
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


def validate_model_metadata_shape(metadata: dict[str, object]) -> dict[str, object]:
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
    if metadata["contract_version"] != "0.1":
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
    if metadata["random_seed"] is not None and not isinstance(
        metadata["random_seed"],
        int,
    ):
        raise ValueError("invalid_model_metadata")
    if not isinstance(metadata["hyperparameters"], dict):
        raise ValueError("invalid_model_metadata")
    if not isinstance(metadata["dependency_status"], dict):
        raise ValueError("invalid_model_metadata")
    if metadata["dependency_status"].get("status") != "available":
        raise ValueError("invalid_model_metadata")
    if "feature_importance" in metadata and not isinstance(
        metadata["feature_importance"],
        dict,
    ):
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
    require_metadata_string(metadata, "model_algorithm", "invalid_model_metadata")
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
    if not isinstance(metadata, dict) or set(metadata) != _MODEL_SELECTION_KEYS:
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
        MODEL_SELECTION_METRIC_DIRECTIONS.get(str(metadata["selection_metric"]))
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
    if candidate_count != len(candidate_model_ids):
        raise ValueError("invalid_model_metadata")
    trial_metrics = metadata["trial_metrics"]
    if not isinstance(trial_metrics, list) or len(trial_metrics) != candidate_count:
        raise ValueError("invalid_model_metadata")
    for trial in trial_metrics:
        if (
            not isinstance(trial, dict)
            or set(trial) != _MODEL_SELECTION_TRIAL_METRIC_KEYS
        ):
            raise ValueError("invalid_model_metadata")
        require_metadata_string(trial, "model_id", "invalid_model_metadata")
        require_non_negative_metadata_int(
            trial,
            "validation_rows",
            "invalid_model_metadata",
        )
        for key in ("validation_ic", "validation_rankic", "validation_mse"):
            if not isinstance(trial[key], int | float):
                raise ValueError("invalid_model_metadata")


def validate_prediction_metadata_shape(
    metadata: dict[str, object],
) -> dict[str, object]:
    if set(metadata) != _PREDICTION_METADATA_KEYS:
        raise ValueError("invalid_prediction_metadata")
    if metadata["contract_version"] != "0.1":
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
    if not isinstance(metadata["is_oos"], bool) or not isinstance(
        metadata["is_final_oos_window"],
        bool,
    ):
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
    if not isinstance(input_columns, list) or input_count != len(input_columns):
        raise ValueError("model_metadata_inconsistent")
    validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_label_dropna"),
        "model_metadata_inconsistent",
    )
    validate_filter_metadata(
        mode=str(metadata["sample_filter_mode"]),
        columns=metadata["filter_columns_used"],
        definition=str(metadata["filter_definition"]),
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
    if not isinstance(model_selection, dict):
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
    if model_selection["selected_model_input_columns"] != metadata["input_columns"]:
        raise ValueError("model_metadata_inconsistent")


def validate_prediction_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    frame: pd.DataFrame,
) -> None:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("prediction_metadata_inconsistent")
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
        mode=str(metadata["sample_filter_mode"]),
        columns=metadata["filter_columns_used"],
        definition=str(metadata["filter_definition"]),
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
