"""Signal materialization and normalization for research workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.validation import (
    _finite_numeric_series,
    strict_timestamp_series,
)


def normalize_signal_frame(
    frame: pd.DataFrame,
    signals: tuple[SignalSpec, ...],
    *,
    timestamp_column: str = "timestamp",
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    for signal in signals:
        _validate_signal_source_kind(signal, source_columns_by_kind)
        result[signal.signal_id] = _normalize_signal_values(
            result,
            signal=signal,
            timestamp_column=timestamp_column,
        )
    consumed_sources = {
        signal.source_column
        for signal in signals
        if signal.source_column != signal.signal_id
        and signal.source_column in frame.columns
    }
    if not consumed_sources:
        return result
    return result.drop(columns=sorted(consumed_sources))


def _validate_signal_source_kind(
    signal: SignalSpec,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None,
) -> None:
    if source_columns_by_kind is None:
        return
    declared = set(source_columns_by_kind.get(signal.source_kind, ()))
    if signal.source_column not in declared:
        raise ValueError(
            "invalid_signal_source_kind_column: "
            f"{signal.source_kind}:{signal.source_column}"
        )


def _normalize_signal_values(
    frame: pd.DataFrame,
    *,
    signal: SignalSpec,
    timestamp_column: str,
) -> pd.Series:
    if signal.source_column not in frame:
        raise ValueError(f"missing_signal_source_column: {signal.source_column}")
    group_timestamps = None
    if signal.transform in {"rank", "zscore"}:
        group_timestamps = strict_timestamp_series(
            frame[timestamp_column],
            error="invalid_signal_timestamps",
        )
    if not pd.api.types.is_numeric_dtype(frame[signal.source_column]):
        raise ValueError(f"invalid_signal_values: {signal.source_column}")
    values = _finite_numeric_series(
        frame[signal.source_column],
        error=f"invalid_signal_values: {signal.source_column}",
        allow_null=True,
    )
    if signal.transform == "rank":
        values = values.groupby(group_timestamps, sort=False).rank(
            method="average",
        )
    elif signal.transform == "zscore":
        grouped = values.groupby(group_timestamps, sort=False)
        values = grouped.transform(_stable_zscore)
    elif signal.transform != "identity":
        raise ValueError(f"unsupported_signal_transform: {signal.transform}")
    if signal.direction == "lower_is_better":
        values = -values.astype("float64")
    elif signal.direction != "higher_is_better":
        raise ValueError(f"unsupported_signal_direction: {signal.direction}")
    return values
