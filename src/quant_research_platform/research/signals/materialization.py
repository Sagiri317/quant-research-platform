"""Signal materialization and normalization for research workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)


def normalize_signal_frame(
    frame: pd.DataFrame,
    signals: tuple[SignalSpec, ...],
    *,
    timestamp_column: str = "timestamp",
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
    resource_lease: MemoryLease | None = None,
) -> pd.DataFrame:
    working_bytes = _signal_normalization_working_bytes(frame, signals)
    policy = current_runtime_resource_policy()
    reservation = (
        policy.lease("research.signal.normalize", working_bytes)
        if policy is not None and resource_lease is None
        else nullcontext()
    )
    with reservation:
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


def _signal_normalization_working_bytes(
    frame: pd.DataFrame,
    signals: tuple[SignalSpec, ...],
) -> int:
    row_count = len(frame)
    index_bytes = int(frame.index.memory_usage(deep=True))
    float_series_bytes = index_bytes + 8 * row_count
    added_signal_bytes = (
        8 * row_count * sum(signal.signal_id not in frame.columns for signal in signals)
    )
    result_bytes = pandas_frame_payload_bytes(frame) + added_signal_bytes
    transform_bytes = 0
    for signal in signals:
        numeric_values = float_series_bytes
        assignment_values = 8 * row_count
        group_codes = float_series_bytes
        if signal.transform == "zscore":
            candidate = (
                numeric_values
                + group_codes
                + float_series_bytes  # group means
                + float_series_bytes  # group standard deviations
                + float_series_bytes  # nonzero standard deviations
                + float_series_bytes  # centered values
                + float_series_bytes  # normalized values
                + row_count  # zero-standard-deviation mask
                + assignment_values
            )
        elif signal.transform == "rank":
            candidate = (
                numeric_values
                + group_codes
                + float_series_bytes  # stable rank workspace
                + float_series_bytes  # ranked values
                + assignment_values
            )
        else:
            candidate = numeric_values + assignment_values
        if signal.direction == "lower_is_better":
            candidate += float_series_bytes
        transform_bytes = max(transform_bytes, candidate)
    consumed_sources = {
        signal.source_column
        for signal in signals
        if signal.source_column != signal.signal_id
        and signal.source_column in frame.columns
    }
    drop_output_bytes = result_bytes if consumed_sources else 0
    return result_bytes + transform_bytes + drop_output_bytes


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
    values = strict_finite_numeric_series(
        frame[signal.source_column],
        error=f"invalid_signal_values: {signal.source_column}",
        allow_null=True,
    ).astype("float64")
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
        values = -values
    elif signal.direction != "higher_is_better":
        raise ValueError(f"unsupported_signal_direction: {signal.direction}")
    return values
