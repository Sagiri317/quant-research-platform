"""Prediction-specific scientific metadata and data checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import (
    PROBABILITY_COLUMNS,
    _validate_model_prediction_window,
    prediction_columns,
    validate_model_facts,
)
from quant_research_platform.contracts.time import (
    PredictionWindow,
    prediction_window_mask,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)

PREDICTION_FACT_ALIASES = ("model_id", "family_id", "label_unit")


def validate_prediction_output_metadata_shape(metadata):
    """Validate either persisted shape and restore the public metadata view."""
    required = {
        "model",
        "inputs",
        "prediction_window",
        "effective_prediction_end",
        "sample_role",
        "prediction_rows",
        "model_metadata",
        "model_facts",
        "input_spec",
        "status",
    }
    if (
        type(metadata) is not dict
        or not required <= set(metadata)
        or set(metadata) - required - {"asset_ids", *PREDICTION_FACT_ALIASES}
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
    for name in ("model", *PREDICTION_FACT_ALIASES):
        if name in metadata and (type(metadata[name]) is not str or not metadata[name]):
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
    _validate_prediction_training_facts(facts, model)
    aliases = {
        "model_id": facts["model"]["model_id"],
        "family_id": facts["family_id"],
        "label_unit": facts["label_unit"],
    }
    if any(
        name in metadata and metadata[name] != value for name, value in aliases.items()
    ):
        raise ValueError("prediction_model_facts_mismatch")
    _validate_model_prediction_window(facts, window)
    metadata.update(aliases)
    return metadata


def _validate_prediction_training_facts(facts, realized):
    """Check self-contained score evidence without fitting-state metadata."""
    if type(facts) is not dict or set(facts) - {"target_definition", "validation"} != {
        "family_id",
        "asset_ids",
        "market_id",
        "model",
        "fit_windows",
        "fit_cutoff",
        "candidate_parameters",
        "label_unit",
        "input_units",
        "selection_cutoff",
    }:
        raise ValueError("prediction_model_facts_invalid")
    fields = {
        "training_asset_ids",
        "train_start",
        "train_end",
        "label_availability_column",
        "label_available_max",
    }
    if type(realized) is dict and "early_stopping" in realized:
        fields.add("early_stopping")
    if type(realized) is dict and "training_target_mean" in realized:
        fields.add("training_target_mean")
        if type(realized["training_target_mean"]) not in {
            int,
            float,
        } or not np.isfinite(realized["training_target_mean"]):
            raise ValueError("prediction_training_target_mean_invalid")
    if type(realized) is not dict or set(realized) != fields:
        raise ValueError("prediction_training_facts_invalid")
    if (
        type(realized["label_availability_column"]) is not str
        or not realized["label_availability_column"]
    ):
        raise ValueError("prediction_label_availability_invalid")
    validate_model_facts(facts, realized)


def validate_prediction_frame(frame, metadata, *, columns=None):
    """Validate rows and return timestamps; the stream index owns uniqueness."""
    expected = columns or prediction_columns(metadata)
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
    if frame.loc[:, ["timestamp", "asset_id"]].isna().any(axis=None):
        raise ValueError("prediction_coverage_invalid")
    if any(type(asset) is not str or not asset for asset in frame.asset_id):
        raise ValueError("invalid_prediction_keys")
    for column in ("prediction_score", "prediction_rank_score", *PROBABILITY_COLUMNS):
        if column in frame:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError("invalid_prediction_values")
            strict_finite_numeric_series(
                frame[column], error="invalid_prediction_values"
            )
    for column in PROBABILITY_COLUMNS:
        if column in frame and not frame[column].between(0, 1).all():
            raise ValueError("invalid_prediction_probabilities")
    if set(PROBABILITY_COLUMNS) <= set(frame):
        if not np.allclose(
            frame[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0, atol=1e-6, rtol=0
        ):
            raise ValueError("invalid_prediction_probabilities")
        if "prediction_score" in frame and not np.allclose(
            frame.prediction_score,
            frame.probability_up - frame.probability_down,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("prediction_direction_score_mismatch")
    if (
        "prediction_rank_score" in frame
        and not frame["prediction_rank_score"].between(-0.5, 0.5).all()
    ):
        raise ValueError("invalid_prediction_rank_values")
    window = PredictionWindow(**metadata["prediction_window"])
    if not prediction_window_mask(timestamps, window).all():
        raise ValueError("prediction_window_invalid")
    return timestamps
