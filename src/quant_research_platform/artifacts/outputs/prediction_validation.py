"""Prediction-specific scientific metadata and data checks."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.models import validate_model_metadata_core
from quant_research_platform.contracts.predictions import (
    PredictionWindow,
    prediction_window_mask,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)


def validate_prediction_output_metadata_shape(metadata):
    required = {
        "model",
        "inputs",
        "model_id",
        "family_id",
        "prediction_window",
        "effective_prediction_end",
        "sample_role",
        "prediction_rows",
        "label_unit",
        "model_metadata",
        "input_spec",
        "status",
    }
    if (
        type(metadata) is not dict
        or not required <= set(metadata)
        or set(metadata) - required - {"asset_ids", "selection"}
    ):
        raise ValueError("invalid_prediction_metadata")
    selection = metadata.get("selection")
    if selection is not None and (type(selection) is not str or not selection):
        raise ValueError("invalid_prediction_selection_reference")
    if metadata["sample_role"] == "test" and selection is None:
        raise ValueError("test_prediction_selection_reference_required")
    assets = metadata.get("asset_ids")
    if assets is not None and (
        type(assets) is not list
        or not assets
        or len(assets) != len(set(assets))
        or any(type(item) is not str or not item for item in assets)
    ):
        raise ValueError("prediction_asset_ids_invalid")
    for name in ("model", "inputs", "model_id", "family_id", "label_unit"):
        if type(metadata[name]) is not str or not metadata[name]:
            raise ValueError("invalid_prediction_metadata")
    if metadata["sample_role"] not in {"validation", "test"}:
        raise ValueError("invalid_prediction_sample_role")
    if type(metadata["prediction_rows"]) is not int or metadata["prediction_rows"] < 0:
        raise ValueError("invalid_prediction_row_count")
    window_payload = metadata["prediction_window"]
    if type(window_payload) is not dict or set(window_payload) != {"start", "end"}:
        raise ValueError("invalid_prediction_window")
    window = PredictionWindow(**window_payload)
    if metadata["prediction_rows"] == 0:
        if (
            metadata["effective_prediction_end"] is not None
            or metadata["status"] != "empty"
        ):
            raise ValueError("invalid_empty_prediction_metadata")
    else:
        effective = strict_timestamp_series(
            pd.Series([metadata["effective_prediction_end"]]),
            error="invalid_effective_prediction_end",
        )
        if (
            metadata["status"] != "complete"
            or not prediction_window_mask(effective, window).all()
        ):
            raise ValueError("prediction_effective_end_outside_window")
    model = metadata["model_metadata"]
    validate_model_metadata_core(
        model, rank_calibration="label_calibration" in model.get("rank_parameters", {})
    )
    return metadata


def validate_prediction_frame(frame, metadata, *, columns=None):
    expected = columns or (
        "timestamp",
        "asset_id",
        "prediction_score",
        "prediction_rank_score",
    )
    if tuple(frame.columns) != tuple(expected):
        raise ValueError("persisted_prediction_columns_mismatch")
    if (
        metadata.get("asset_ids") is not None
        and not frame.asset_id.isin(metadata["asset_ids"]).all()
    ):
        raise ValueError("prediction_asset_scope_invalid")
    timestamps = strict_timestamp_series(
        frame.timestamp, error="invalid_prediction_timestamp"
    )
    if (
        frame.loc[:, ["timestamp", "asset_id"]].isna().any(axis=None)
        or frame.duplicated(["timestamp", "asset_id"]).any()
    ):
        raise ValueError("prediction_coverage_invalid")
    if any(type(asset) is not str or not asset for asset in frame.asset_id):
        raise ValueError("invalid_prediction_keys")
    for column in ("prediction_score", "prediction_rank_score"):
        if column in frame:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError("invalid_prediction_values")
            strict_finite_numeric_series(
                frame[column], error="invalid_prediction_values"
            )
    window = PredictionWindow(**metadata["prediction_window"])
    if not prediction_window_mask(timestamps, window).all():
        raise ValueError("prediction_window_invalid")
