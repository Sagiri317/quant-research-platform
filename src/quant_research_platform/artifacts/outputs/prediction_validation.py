"""Prediction-specific scientific metadata and data checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import (
    _model_spec,
    _validate_model_prediction_window,
    validate_model_fit_facts,
)
from quant_research_platform.contracts.time import (
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
        "model_facts",
        "input_spec",
        "status",
    }
    if (
        type(metadata) is not dict
        or not required <= set(metadata)
        or set(metadata) - required - {"asset_ids"}
    ):
        raise ValueError("invalid_prediction_metadata")
    assets = metadata.get("asset_ids")
    if assets is not None and (
        type(assets) is not list
        or not assets
        or len(assets) != len(set(assets))
        or any(type(item) is not str or not item for item in assets)
    ):
        raise ValueError("prediction_asset_ids_invalid")
    for name in ("model", "model_id", "family_id", "label_unit"):
        if type(metadata[name]) is not str or not metadata[name]:
            raise ValueError("invalid_prediction_metadata")
    from quant_research_platform.storage.datasets.reading import _dataset_input_options

    _dataset_input_options(metadata["inputs"])
    if type(metadata["sample_role"]) is not str or not metadata["sample_role"].strip():
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
    facts = metadata["model_facts"]
    model = metadata["model_metadata"]
    _validate_prediction_model_facts(facts, model)
    if (
        metadata["model_id"] != facts["model"]["model_id"]
        or metadata["family_id"] != facts["family_id"]
        or metadata["label_unit"] != facts["label_unit"]
    ):
        raise ValueError("prediction_model_facts_mismatch")
    _validate_model_prediction_window(facts, window)
    return metadata


def _validate_prediction_model_facts(facts, realized):
    """Check self-contained score evidence without fitting-state metadata."""
    if type(facts) is not dict or set(facts) - {"target_definition"} != {
        "family_id",
        "asset_ids",
        "market_id",
        "model",
        "fit_windows",
        "fit_cutoff",
        "candidate_parameters",
        "label_unit",
        "test_window",
        "selection_cutoff",
    }:
        raise ValueError("prediction_model_facts_invalid")
    from quant_research_platform.contracts.ids import validate_market_id

    validate_market_id(facts["market_id"])
    if type(facts["family_id"]) is not str or not facts["family_id"]:
        raise ValueError("prediction_model_facts_invalid")
    spec = _model_spec(facts["model"])
    if type(facts["candidate_parameters"]) is not dict:
        raise ValueError("prediction_candidate_parameters_invalid")
    fields = {
        "training_asset_ids",
        "train_start",
        "train_end",
        "label_availability_column",
        "label_available_max",
    }
    is_rank = spec.model_algorithm in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }
    if is_rank:
        fields.add("rank_parameters")
    if type(realized) is not dict or set(realized) != fields:
        raise ValueError("prediction_training_facts_invalid")
    if (
        type(realized["label_availability_column"]) is not str
        or not realized["label_availability_column"]
    ):
        raise ValueError("prediction_label_availability_invalid")
    for assets in (facts["asset_ids"], realized["training_asset_ids"]):
        if assets is not None and (
            type(assets) is not list
            or not assets
            or any(type(asset) is not str or not asset for asset in assets)
            or len(assets) != len(set(assets))
        ):
            raise ValueError("prediction_training_assets_invalid")
    if realized["training_asset_ids"] is None or (
        facts["asset_ids"] is not None
        and not set(realized["training_asset_ids"]) <= set(facts["asset_ids"])
    ):
        raise ValueError("prediction_training_assets_invalid")
    if is_rank:
        parameters = realized["rank_parameters"]
        if type(parameters) is not dict or set(parameters) != {"label_calibration"}:
            raise ValueError("prediction_label_calibration_invalid")
        calibration = parameters["label_calibration"]
        if (
            type(calibration) is not list
            or len(calibration) != 2
            or any(
                type(value) not in {int, float} or not np.isfinite(value)
                for value in calibration
            )
        ):
            raise ValueError("prediction_label_calibration_invalid")
    validate_model_fit_facts(facts, realized)


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
    if (
        "prediction_rank_score" in frame
        and not frame["prediction_rank_score"].between(-0.5, 0.5).all()
    ):
        raise ValueError("invalid_prediction_rank_values")
    window = PredictionWindow(**metadata["prediction_window"])
    if not prediction_window_mask(timestamps, window).all():
        raise ValueError("prediction_window_invalid")
