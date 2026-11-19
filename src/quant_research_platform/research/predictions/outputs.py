"""Select valid prediction rows while retaining causal context for sequence models."""

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.outputs.manifest import read_output_manifest
from quant_research_platform.artifacts.outputs.prediction import _iter_prediction_member
from quant_research_platform.contracts.models import DEEPLOB_ALGORITHMS
from quant_research_platform.contracts.time import prediction_window_mask
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.deeplob import complete_book_rows


def read_rolling_predictions(paths, *, window, asset_ids):
    """Join consumed forecasts after their fit and selection cutoffs."""
    frames, metadata = [], []
    for path in paths:
        facts = read_output_manifest(Path(path))["metadata"]
        model = facts["model_facts"]
        cutoffs = [model["fit_cutoff"]]
        if model.get("selection_cutoff") is not None:
            cutoffs.append(model["selection_cutoff"])
        if model["candidate_parameters"].get("early_stopping_rounds") is not None:
            cutoffs.append(model["validation"]["cutoff"])
        cutoff = max(pd.Timestamp(value) for value in cutoffs)
        consumed = False
        with closing_iterator(
            _iter_prediction_member(
                Path(path),
                facts,
                columns=("timestamp", "asset_id", "prediction_score"),
                time_window=window,
                asset_ids=asset_ids,
            )
        ) as chunks:
            for frame in chunks:
                if frame.timestamp.min() <= cutoff:
                    raise ValueError("policy_prediction_not_out_of_sample")
                frames.append(frame)
                consumed = True
        if consumed:
            metadata.append(facts)
    predictions = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "asset_id": pd.Series(dtype="str"),
                "prediction_score": pd.Series(dtype="float64"),
            }
        )
    )
    if predictions.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("hft_prediction_keys_overlap")
    return predictions, metadata


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
    if not include_context:
        filtered = filtered.loc[prediction_input_mask(filtered, metadata)]
    return filtered.sort_values(["timestamp", "asset_id"], kind="stable")


def centered_target_scores(scores, timestamps, definition):
    """Center only a date-standardized target, over all eligible scored rows."""
    if (definition or {}).get("target_transform") != "date_zscore":
        return scores
    return scores - scores.groupby(timestamps, sort=False).transform("mean")
