"""Signal materialization and normalization for research workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.signals import SignalSpec


def materialize_signal_frame(
    frame: pd.DataFrame,
    signals: tuple[SignalSpec, ...],
    *,
    timestamp_column: str = "timestamp",
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    return normalize_signal_frame(
        frame,
        signals,
        timestamp_column=timestamp_column,
        source_columns_by_kind=source_columns_by_kind,
    )


def normalize_signal_frame(
    frame: pd.DataFrame,
    signals: tuple[SignalSpec, ...],
    *,
    timestamp_column: str = "timestamp",
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    consumed_source_columns: set[str] = set()
    for signal in signals:
        _validate_signal_source_kind(signal, source_columns_by_kind)
        result[signal.signal_id] = _normalize_signal_values(
            result,
            signal=signal,
            timestamp_column=timestamp_column,
        )
        if signal.source_column != signal.signal_id:
            consumed_source_columns.add(signal.source_column)
    return result.drop(
        columns=[column for column in consumed_source_columns if column in result],
    )


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
    values = pd.to_numeric(frame[signal.source_column], errors="coerce")
    if signal.transform == "rank":
        values = values.groupby(frame[timestamp_column], sort=False).rank(
            method="average",
        )
    elif signal.transform == "zscore":
        grouped = values.groupby(frame[timestamp_column], sort=False)
        mean = grouped.transform("mean")
        std = grouped.transform(lambda item: item.std(ddof=0))
        values = (values - mean) / std.replace(0, pd.NA)
    elif signal.transform != "identity":
        raise ValueError(f"unsupported_signal_transform: {signal.transform}")
    if signal.direction == "lower_is_better":
        values = -values
    elif signal.direction != "higher_is_better":
        raise ValueError(f"unsupported_signal_direction: {signal.direction}")
    return values


__all__ = ["materialize_signal_frame", "normalize_signal_frame"]
