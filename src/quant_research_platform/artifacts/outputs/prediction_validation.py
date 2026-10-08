"""Prediction output validation."""

from __future__ import annotations

from math import isfinite

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
    CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    ModelSpec,
    _validate_a_share_model_family_ids,
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
        "quarterly_models",
        "champion_by_quarter",
        "selection_group",
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
    if metadata.get("model_artifact_schema") == CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID:
        return _validate_crypto_lineage(metadata)
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
    models = metadata["quarterly_models"]
    champions = metadata["champion_by_quarter"]
    group = metadata["selection_group"]
    if type(group) is not str or group not in {"global", "quarterly"}:
        raise ValueError("invalid_model_metadata")
    if type(models) is not list or not models or type(champions) is not dict:
        raise ValueError("invalid_model_metadata")
    seen = set()
    families = {}
    for model in models:
        if type(model) is not dict or set(model) != {
            "quarter_id",
            "decision_id",
            "prediction_source",
            "family_id",
            "model_id",
            "prediction_window",
        }:
            raise ValueError("invalid_model_metadata")
        quarter, family = model["quarter_id"], model["family_id"]
        source = model["prediction_source"]
        decision_id = model["decision_id"]
        validate_run_id(family)
        if (
            type(quarter) is not str
            or str(pd.Period(quarter, freq="Q")) != quarter
            or type(source) is not str
            or source not in {"validation", "final"}
            or decision_id
            != ("final" if source == "final" or group == "global" else quarter)
            or (quarter, family) in seen
        ):
            raise ValueError("invalid_model_metadata")
        validate_run_id(model["model_id"])
        window = model["prediction_window"]
        if (
            type(window) is not dict
            or set(window) != {"start", "end"}
            or any(
                str(pd.Period(window[bound], freq="Q")) != quarter
                for bound in ("start", "end")
            )
        ):
            raise ValueError("invalid_model_metadata")
        seen.add((quarter, family))
        families.setdefault(quarter, []).append(family)
    if (
        len({item["prediction_source"] for item in models}) != 1
        or set(champions) != set(families)
        or any(
            champions[quarter] not in members for quarter, members in families.items()
        )
        or any(
            members != next(iter(families.values())) for members in families.values()
        )
    ):
        raise ValueError("invalid_model_metadata")
    _validate_a_share_model_family_ids(next(iter(families.values())))
    return metadata


def validate_family_prediction_frame(
    frame: pd.DataFrame, model_metadata: dict[str, object]
) -> None:
    """Match every persisted family/config coordinate to frozen model facts."""
    if frame.empty:
        return
    models = {
        (item["quarter_id"], item["family_id"]): item
        for item in model_metadata["quarterly_models"]
    }
    for (quarter, family), group in frame.groupby(
        ["quarter_id", "family_id"], sort=False, dropna=False
    ):
        model = models.get((quarter, family))
        if model is None or not group["model_id"].eq(model["model_id"]).all():
            raise ValueError("prediction_family_model_misaligned")
        times = pd.to_datetime(group["timestamp"], errors="raise")
        if (
            not times.dt.to_period("Q").astype(str).eq(quarter).all()
            or times.min() < pd.Timestamp(model["prediction_window"]["start"])
            or times.max() > pd.Timestamp(model["prediction_window"]["end"])
        ):
            raise ValueError("prediction_family_window_misaligned")


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
    elif metadata["prediction_mode"] not in {"a_share_oos", "crypto_oos"}:
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
        if "model_artifact_id" not in model_metadata:
            ModelSpec(
                model_id=metadata["model_id"],
                model_algorithm=model_metadata["model_algorithm"],
                input_factor_columns=tuple(model_metadata["input_factor_columns"]),
                input_feature_columns=tuple(model_metadata["input_feature_columns"]),
                label_column=model_metadata["label_column"],
                sample_filter_mode=model_metadata["sample_filter_mode"],
                hyperparameters=model_metadata["hyperparameters"],
                random_seed=model_metadata["random_seed"],
                max_train_rows=model_metadata["max_train_rows"],
            )
        elif (
            model_metadata["model_id"]
            if model_metadata["model_artifact_schema"]
            == CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            else model_metadata["model_artifact_id"]
        ) != metadata["model_id"]:
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
    elif metadata["prediction_mode"] not in {"a_share_oos", "crypto_oos"}:
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
        expected_model = (
            metadata["model_id"]
            if metadata["model_artifact_schema"] == CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            else metadata["model_artifact_id"]
        )
        if expected_model != spec.model_id:
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
        expected_mode = (
            "crypto_oos"
            if model_metadata["model_artifact_schema"]
            == CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            else "a_share_oos"
        )
        if prediction_metadata["prediction_mode"] != expected_mode:
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


def _validate_crypto_lineage(metadata):
    if set(metadata) != {
        "model_artifact_id",
        "model_artifact_identity",
        "model_artifact_schema",
        "model_id",
        "asset_ids",
        "fold_models",
        "score_to_label",
    }:
        raise ValueError("invalid_model_metadata")
    validate_run_id(metadata["model_artifact_id"])
    validate_run_id(metadata["model_id"])
    identity = metadata["model_artifact_identity"]
    if (
        type(identity) is not str
        or len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
    ):
        raise ValueError("invalid_model_metadata")
    require_metadata_string_list(metadata, "asset_ids", "invalid_model_metadata")
    calibration = metadata["score_to_label"]
    if calibration is not None and (
        type(calibration) is not list
        or len(calibration) != 2
        or any(
            type(value) not in {int, float} or not isfinite(value)
            for value in calibration
        )
    ):
        raise ValueError("invalid_model_metadata")
    folds = metadata["fold_models"]
    if type(folds) is not list or len(folds) != 1:
        raise ValueError("invalid_model_metadata")
    seen = set()
    previous = None
    for fold in folds:
        if type(fold) is not dict or set(fold) != {
            "fold_id",
            "prediction_window",
            "fit_windows",
            "fit_cutoff",
            "training_asset_ids",
            "evaluation_asset_ids",
        }:
            raise ValueError("invalid_model_metadata")
        validate_run_id(fold["fold_id"])
        if fold["fold_id"] in seen:
            raise ValueError("invalid_model_metadata")
        seen.add(fold["fold_id"])
        start, end = (
            pd.Timestamp(fold["prediction_window"][key]) for key in ("start", "end")
        )
        if start > end or (previous is not None and start <= previous):
            raise ValueError("invalid_model_metadata")
        previous = end
        windows = fold["fit_windows"]
        if type(windows) is not list or len(windows) != 2:
            raise ValueError("invalid_model_metadata")
        if any(
            type(window) is not dict or set(window) != {"start", "end"}
            for window in windows
        ):
            raise ValueError("invalid_model_metadata")
        bounds = tuple(
            (pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
            for window in windows
        )
        cutoff = pd.Timestamp(fold["fit_cutoff"])
        if (
            any(pd.isna(value) for pair in bounds for value in pair)
            or any(left > right for left, right in bounds)
            or not bounds[0][1] < bounds[1][0]
            or bounds[-1][1] != cutoff
            or cutoff >= start
        ):
            raise ValueError("invalid_model_metadata")
        for key in ("training_asset_ids", "evaluation_asset_ids"):
            require_metadata_string_list(
                fold, key, "invalid_model_metadata", allow_empty=True
            )
    return metadata


def validate_crypto_prediction_frame(frame, model_metadata):
    if frame.empty:
        return
    times = pd.to_datetime(frame["timestamp"], errors="raise")
    valid = pd.Series(False, index=frame.index)
    for fold in model_metadata["fold_models"]:
        scope = fold["evaluation_asset_ids"] or model_metadata["asset_ids"]
        valid |= times.between(
            pd.Timestamp(fold["prediction_window"]["start"]),
            pd.Timestamp(fold["prediction_window"]["end"]),
        ) & frame.asset_id.isin(scope)
    if not valid.all():
        raise ValueError("crypto_prediction_fold_scope_misaligned")
