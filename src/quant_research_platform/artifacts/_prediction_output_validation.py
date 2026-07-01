"""Prediction output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts._output_common_validation import (
    _require_metadata_string,
    _require_metadata_string_list,
    _require_non_negative_metadata_int,
    _require_optional_metadata_string,
    _validate_filter_metadata,
    _validate_metadata_row_count_chain,
)
from quant_research_platform.contracts.predictions import PredictionSpec

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


def _validate_model_metadata_shape(metadata: dict[str, object]) -> dict[str, object]:
    metadata_keys = set(metadata)
    feature_keys = metadata_keys.intersection(_MODEL_METADATA_FEATURE_KEYS)
    if (
        not _MODEL_METADATA_BASE_KEYS.issubset(metadata_keys)
        or len(feature_keys) != 1
        or metadata_keys.difference(_MODEL_METADATA_BASE_KEYS).difference(
            _MODEL_METADATA_FEATURE_KEYS
        )
    ):
        raise ValueError("invalid_model_metadata")
    if metadata["contract_version"] != "0.1":
        raise ValueError("invalid_model_metadata")
    _require_non_negative_metadata_int(metadata, "rows_input", "invalid_model_metadata")
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_model_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_input_label_dropna",
        "invalid_model_metadata",
    )
    input_factor_columns = _require_metadata_string_list(
        metadata,
        "input_factor_columns",
        "invalid_model_metadata",
        allow_empty=True,
    )
    input_feature_columns = _require_metadata_string_list(
        metadata,
        "input_feature_columns",
        "invalid_model_metadata",
        allow_empty=True,
    )
    input_columns = _require_metadata_string_list(
        metadata,
        "input_columns",
        "invalid_model_metadata",
    )
    input_count = _require_non_negative_metadata_int(
        metadata,
        "input_count",
        "invalid_model_metadata",
    )
    coefficient_count = _require_non_negative_metadata_int(
        metadata,
        "coefficient_count",
        "invalid_model_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "purge_gap_steps",
        "invalid_model_metadata",
    )
    if metadata["max_train_rows"] is not None:
        _require_non_negative_metadata_int(
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
        _require_metadata_string(
            metadata,
            "feature_importance_status",
            "invalid_model_metadata",
        )
    if (
        input_columns != [*input_factor_columns, *input_feature_columns]
        or input_count != len(input_columns)
        or coefficient_count < 1
    ):
        raise ValueError("invalid_model_metadata")
    _require_metadata_string(metadata, "model_algorithm", "invalid_model_metadata")
    _require_metadata_string(metadata, "sample_filter_mode", "invalid_model_metadata")
    _require_metadata_string(metadata, "filter_definition", "invalid_model_metadata")
    _require_metadata_string(metadata, "label_column", "invalid_model_metadata")
    _require_metadata_string(metadata, "train_start", "invalid_model_metadata")
    _require_metadata_string(metadata, "train_end", "invalid_model_metadata")
    _require_optional_metadata_string(
        metadata,
        "test_start",
        "invalid_model_metadata",
    )
    _require_optional_metadata_string(metadata, "test_end", "invalid_model_metadata")
    _require_optional_metadata_string(metadata, "oos_start", "invalid_model_metadata")
    _require_optional_metadata_string(metadata, "oos_end", "invalid_model_metadata")
    _require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_model_metadata",
        allow_empty=True,
    )
    return metadata


def _validate_prediction_metadata_shape(
    metadata: dict[str, object],
) -> dict[str, object]:
    if set(metadata) != _PREDICTION_METADATA_KEYS:
        raise ValueError("invalid_prediction_metadata")
    if metadata["contract_version"] != "0.1":
        raise ValueError("invalid_prediction_metadata")
    _require_non_negative_metadata_int(
        metadata,
        "rows_input",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_input_dropna",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "prediction_rows",
        "invalid_prediction_metadata",
    )
    input_columns = _require_metadata_string_list(
        metadata,
        "input_columns",
        "invalid_prediction_metadata",
    )
    input_count = _require_non_negative_metadata_int(
        metadata,
        "input_count",
        "invalid_prediction_metadata",
    )
    if input_count != len(input_columns):
        raise ValueError("invalid_prediction_metadata")
    _require_metadata_string(
        metadata,
        "prediction_mode",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "sample_filter_mode",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "filter_definition",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(metadata, "test_start", "invalid_prediction_metadata")
    _require_metadata_string(metadata, "train_end", "invalid_prediction_metadata")
    if metadata["requested_test_end"] is not None:
        _require_metadata_string(
            metadata,
            "requested_test_end",
            "invalid_prediction_metadata",
        )
    _require_metadata_string(
        metadata,
        "effective_test_end",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "prediction_column",
        "invalid_prediction_metadata",
    )
    _require_metadata_string_list(
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


def _validate_model_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
) -> None:
    input_count = int(metadata["input_count"])
    input_columns = metadata["input_columns"]
    if not isinstance(input_columns, list) or input_count != len(input_columns):
        raise ValueError("model_metadata_inconsistent")
    _validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_label_dropna"),
        "model_metadata_inconsistent",
    )
    _validate_filter_metadata(
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
    if int(metadata["coefficient_count"]) <= 0:
        raise ValueError("model_metadata_inconsistent")


def _validate_prediction_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    frame: pd.DataFrame,
) -> None:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("prediction_metadata_inconsistent")
    effective_test_end = pd.Timestamp(timestamps.max()).date().isoformat()
    _validate_prediction_metadata_consistency_from_summary(
        metadata,
        spec=spec,
        row_count=len(frame),
        effective_test_end=effective_test_end,
    )


def _validate_prediction_metadata_consistency_from_summary(
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
    _validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_dropna"),
        "prediction_metadata_inconsistent",
    )
    if int(metadata["prediction_rows"]) > int(metadata["rows_after_input_dropna"]):
        raise ValueError("prediction_metadata_inconsistent")
    _validate_filter_metadata(
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


def _validate_model_prediction_metadata_consistency(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> None:
    if model_metadata["input_columns"] != prediction_metadata["input_columns"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["input_count"] != prediction_metadata["input_count"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["oos_end"] != prediction_metadata["effective_test_end"]:
        raise ValueError("prediction_metadata_inconsistent")


def _model_metadata_with_effective_prediction_end(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **model_metadata,
        "oos_end": prediction_metadata["effective_test_end"],
    }
