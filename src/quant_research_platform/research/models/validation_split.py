"""Training window controls for supervised model fitting."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.validation import strict_timestamp_series


def trim_training_buffer(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if spec.max_train_rows is None or frame.empty:
        return frame
    result = frame.copy()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"],
        error="invalid_model_training_timestamp",
    )
    result = result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if spec.purge_gap_steps <= 0:
        return result.tail(spec.max_train_rows).reset_index(drop=True)
    training_candidates = _purge_trailing_timestamps(
        result,
        spec.purge_gap_steps,
        group_column="asset_id" if "asset_id" in result.columns else None,
    )
    keep = result.index.isin(training_candidates.tail(spec.max_train_rows).index)
    keep |= ~result.index.isin(training_candidates.index)
    return result.loc[keep].reset_index(drop=True)


def apply_training_controls(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"],
        error="invalid_model_training_timestamp",
    )
    result = result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    result = _purge_trailing_timestamps(
        result,
        spec.purge_gap_steps,
        group_column="asset_id" if "asset_id" in result.columns else None,
    )
    if spec.max_train_rows is not None and len(result) > spec.max_train_rows:
        result = result.tail(spec.max_train_rows)
    return result.reset_index(drop=True)


def _purge_trailing_timestamps(
    frame: pd.DataFrame,
    purge_gap_steps: int,
    *,
    group_column: str | None = None,
) -> pd.DataFrame:
    if purge_gap_steps <= 0 or frame.empty:
        return frame
    timestamps = strict_timestamp_series(
        frame["timestamp"],
        error="invalid_model_training_timestamp",
    )
    if group_column is not None:
        trailing_rank = timestamps.groupby(
            frame[group_column],
            sort=False,
            dropna=False,
        ).rank(method="dense", ascending=False)
        return frame.loc[trailing_rank > purge_gap_steps]
    unique_timestamps = tuple(sorted(timestamps.dropna().unique()))
    if purge_gap_steps >= len(unique_timestamps):
        return frame.iloc[0:0]
    cutoff = unique_timestamps[-purge_gap_steps]
    return frame.loc[timestamps < cutoff]
