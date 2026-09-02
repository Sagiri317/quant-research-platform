"""Prediction output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts.outputs.common_validation import (
    require_metadata_string,
    require_metadata_string_list,
    require_non_negative_metadata_int,
    require_optional_metadata_string,
    validate_filter_metadata,
    validate_metadata_row_count_chain,
)
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    ModelSpec,
    validate_model_metadata_core,
)
from quant_research_platform.contracts.predictions import (
    PREDICTION_MODES,
    PREDICTION_SCORE_COLUMN,
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.run_id import validate_run_id

_MODEL_ARTIFACT_LINEAGE_KEYS = frozenset(
    {
        "model_artifact_id",
        "model_artifact_identity",
        "model_artifact_schema",
    }
)
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
        "training_asset_ids",
        "evaluation_asset_ids",
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
    if "model_artifact_id" in metadata:
        return _validate_model_artifact_lineage(metadata)
    extension_keys = {"test_start", "test_end", "oos_start", "oos_end"}
    if not extension_keys.issubset(metadata):
        raise ValueError("invalid_model_metadata")
    core_metadata = {
        key: value for key, value in metadata.items() if key not in extension_keys
    }
    validate_model_metadata_core(core_metadata)
    for key in extension_keys:
        require_optional_metadata_string(metadata, key, "invalid_model_metadata")
    return metadata


def _validate_model_artifact_lineage(metadata: dict[str, object]) -> dict[str, object]:
    if set(metadata) != _MODEL_ARTIFACT_LINEAGE_KEYS:
        raise ValueError("invalid_model_metadata")
    model_artifact_id = require_metadata_string(
        metadata,
        "model_artifact_id",
        "invalid_model_metadata",
    )
    try:
        validate_run_id(model_artifact_id)
    except ValueError as exc:
        raise ValueError("invalid_model_metadata") from exc
    identity = metadata["model_artifact_identity"]
    if (
        type(identity) is not str
        or len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
    ):
        raise ValueError("invalid_model_metadata")
    if metadata["model_artifact_schema"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID:
        raise ValueError("invalid_model_metadata")
    return metadata


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
        "test_start",
        "prediction_column",
        "sample_filter_mode",
    ):
        require_metadata_string(metadata, key, "output_manifest_metadata_invalid")
    require_optional_metadata_string(
        metadata, "test_end", "output_manifest_metadata_invalid"
    )
    if metadata["train_end"] is not None:
        require_metadata_string(
            metadata, "train_end", "output_manifest_metadata_invalid"
        )
    elif metadata["prediction_mode"] != "a_share_oos":
        raise ValueError("output_manifest_metadata_invalid")
    if (
        metadata["prediction_mode"] not in PREDICTION_MODES
        or (
            metadata["prediction_mode"] == "a_share_oos"
            and metadata["prediction_column"] != RAW_PREDICTION_COLUMN
        )
        or (
            metadata["prediction_mode"] == "minimal_oos"
            and metadata["prediction_column"] != PREDICTION_SCORE_COLUMN
        )
    ):
        raise ValueError("output_manifest_metadata_invalid")
    if (
        type(metadata["is_oos"]) is not bool
        or type(metadata["is_final_oos_window"]) is not bool
    ):
        raise ValueError("output_manifest_metadata_invalid")
    require_metadata_string_list(
        metadata, "primary_key", "output_manifest_metadata_invalid"
    )
    require_metadata_string_list(
        metadata,
        "training_asset_ids",
        "output_manifest_metadata_invalid",
        allow_empty=True,
    )
    require_metadata_string_list(
        metadata,
        "evaluation_asset_ids",
        "output_manifest_metadata_invalid",
        allow_empty=True,
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
        (
            "sample_filter_mode" in model_metadata
            and model_metadata["sample_filter_mode"] != metadata["sample_filter_mode"]
        )
        or prediction_metadata["sample_filter_mode"] != metadata["sample_filter_mode"]
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
        if "model_artifact_id" not in model_metadata:
            ModelSpec(
                model_id=persisted_model_id,
                model_algorithm=model_metadata["model_algorithm"],
                input_factor_columns=tuple(model_metadata["input_factor_columns"]),
                input_feature_columns=tuple(model_metadata["input_feature_columns"]),
                label_column=model_metadata["label_column"],
                sample_filter_mode=model_metadata["sample_filter_mode"],
                hyperparameters=model_metadata["hyperparameters"],
                random_seed=model_metadata["random_seed"],
                max_train_rows=model_metadata["max_train_rows"],
            )
        elif model_metadata["model_artifact_id"] != metadata["model_id"]:
            raise ValueError("model_metadata_inconsistent")
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
            training_asset_ids=tuple(metadata["training_asset_ids"]),
            evaluation_asset_ids=tuple(metadata["evaluation_asset_ids"]),
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
    if metadata["train_end"] is not None:
        require_metadata_string(metadata, "train_end", "invalid_prediction_metadata")
    elif metadata["prediction_mode"] != "a_share_oos":
        raise ValueError("invalid_prediction_metadata")
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
    if (
        metadata["prediction_mode"] not in PREDICTION_MODES
        or (
            metadata["prediction_mode"] == "a_share_oos"
            and metadata["prediction_column"] != RAW_PREDICTION_COLUMN
        )
        or (
            metadata["prediction_mode"] == "minimal_oos"
            and metadata["prediction_column"] != PREDICTION_SCORE_COLUMN
        )
    ):
        raise ValueError("invalid_prediction_metadata")
    require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_prediction_metadata",
        allow_empty=True,
    )
    if (
        type(metadata["is_oos"]) is not bool
        or type(metadata["is_final_oos_window"]) is not bool
    ):
        raise ValueError("invalid_prediction_metadata")
    return metadata


def validate_model_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
) -> None:
    if "model_artifact_id" in metadata:
        if metadata["model_artifact_id"] != spec.model_id:
            raise ValueError("model_metadata_inconsistent")
        return
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
    if train_start > train_end or train_end >= pd.Timestamp(spec.test_start):
        raise ValueError("model_metadata_inconsistent")
    if spec.train_end is not None and train_end > pd.Timestamp(spec.train_end):
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
        model_selection["validation_start"] != spec.model_selection_validation_start
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
    if "model_artifact_id" in model_metadata:
        if prediction_metadata["prediction_mode"] != "a_share_oos":
            raise ValueError("prediction_metadata_inconsistent")
        return
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
    if "model_artifact_id" in model_metadata:
        return dict(model_metadata)
    return {
        **model_metadata,
        "oos_end": prediction_metadata["effective_test_end"],
    }
