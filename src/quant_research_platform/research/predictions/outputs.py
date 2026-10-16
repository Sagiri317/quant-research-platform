"""Select each model's own eligible prediction domain."""

from quant_research_platform.contracts.time import prediction_window_mask
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.research.features.transforms import (
    mask_unobserved_features,
    model_input_validity_columns,
    model_observed_input_mask,
)
from quant_research_platform.research.filtering import apply_filter_mode


def prediction_rows(frame, metadata, window, *, declared_columns=()):
    """The model's own economic prediction domain, independent of competitors."""
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
    filtered, _ = apply_filter_mode(
        scoped, mode=metadata["sample_filter_mode"], metadata_key="sample_filter_mode"
    )
    validity = model_input_validity_columns(
        tuple(metadata["input_columns"]), metadata["model_algorithm"], declared_columns
    )
    for column in tuple(dict.fromkeys((*metadata["input_columns"], *validity))):
        strict_finite_numeric_series(
            filtered[column],
            error=f"invalid_prediction_input: {column}",
            allow_null=True,
        )
    observed = model_observed_input_mask(
        filtered, metadata["input_columns"], declared_columns
    )
    filtered = mask_unobserved_features(
        filtered, metadata["input_columns"], declared_columns
    )
    return filtered.loc[observed].sort_values(["timestamp", "asset_id"], kind="stable")
