"""Explicit training-window controls for non-formal bounded experiments."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.validation import strict_timestamp_series


def trim_training_buffer(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Keep an explicit trailing row cap when a non-formal caller requests one."""

    if spec.max_train_rows is None or frame.empty:
        return frame
    result = frame.copy()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"],
        error="invalid_model_training_timestamp",
    )
    return (
        result.sort_values("timestamp", kind="mergesort")
        .tail(spec.max_train_rows)
        .reset_index(drop=True)
    )


def apply_training_controls(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Apply only the generic, explicitly requested row cap."""

    if frame.empty:
        return frame
    result = frame.copy()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"],
        error="invalid_model_training_timestamp",
    )
    result = result.sort_values("timestamp", kind="mergesort")
    if spec.max_train_rows is not None:
        result = result.tail(spec.max_train_rows)
    return result.reset_index(drop=True)
