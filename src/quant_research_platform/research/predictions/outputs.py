"""Select valid prediction rows while retaining causal context for sequence models."""

from quant_research_platform.contracts.models import DEEPLOB_ALGORITHMS
from quant_research_platform.contracts.time import prediction_window_mask
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.deeplob import complete_book_rows


def prediction_input_mask(frame, metadata):
    """Identify rows with enough observed inputs for this model's output."""
    if metadata["model_algorithm"] in DEEPLOB_ALGORITHMS:
        return complete_book_rows(frame)
    return frame.loc[:, metadata["input_columns"]].notna().any(axis=1)


def prediction_rows(frame, metadata, window, *, include_context=False):
    """Select observed model inputs while retaining temporarily ineligible
    sequence context.
    """
    if (
        frame.loc[:, ["timestamp", "asset_id"]].isna().any(axis=None)
        or frame.duplicated(["timestamp", "asset_id"]).any()
    ):
        raise ValueError("invalid_prediction_keys")
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_prediction_timestamp"
    )
    mask = prediction_window_mask(timestamps, window)
    scoped = frame.loc[mask].copy()
    scoped["timestamp"] = timestamps.loc[mask]
    filtered = scoped
    if not include_context:
        filtered, _ = apply_filter_mode(
            scoped,
            mode=metadata["sample_filter_mode"],
            metadata_key="sample_filter_mode",
        )
    for column in metadata["input_columns"]:
        strict_finite_numeric_series(
            filtered[column],
            error=f"invalid_prediction_input: {column}",
            allow_null=True,
        )
    if not include_context:
        filtered = filtered.loc[prediction_input_mask(filtered, metadata)]
    return filtered.sort_values(["timestamp", "asset_id"], kind="stable")
