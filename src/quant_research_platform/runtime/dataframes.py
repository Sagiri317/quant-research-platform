"""Timestamp-complete DataFrame streams shared by research owners."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator


def iter_complete_time_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
) -> Iterator[pd.DataFrame]:
    """Keep each timestamp together, buffering only the unfinished boundary."""
    pending: list[pd.DataFrame] = []
    pending_timestamp: pd.Timestamp | None = None
    emitted_max: pd.Timestamp | None = None
    first_empty: pd.DataFrame | None = None
    with closing_iterator(frames):
        for frame in frames:
            if timestamp_column not in frame.columns:
                raise ValueError("missing_time_chunk_timestamp_column")
            if frame.empty:
                if first_empty is None:
                    first_empty = frame
                continue
            timestamps = strict_timestamp_series(
                frame[timestamp_column], error="invalid_time_chunk_timestamp_values"
            )
            if not timestamps.is_monotonic_increasing:
                order = timestamps.argsort(kind="mergesort")
                frame = frame.iloc[order].reset_index(drop=True)
                timestamps = timestamps.iloc[order].reset_index(drop=True)
            current_min = pd.Timestamp(timestamps.iloc[0])
            if emitted_max is not None and current_min <= emitted_max:
                raise ValueError("non_monotonic_time_chunk_order")
            start = 0
            if pending:
                if current_min < pending_timestamp:
                    raise ValueError("non_monotonic_time_chunk_order")
                if current_min == pending_timestamp:
                    start = int(timestamps.searchsorted(current_min, side="right"))
                    pending.append(frame.iloc[:start])
                    if start == len(frame):
                        continue
                complete = (
                    pending[0]
                    if len(pending) == 1
                    else pd.concat(pending, ignore_index=True)
                )
                try:
                    yield complete
                except GeneratorExit:
                    return
                emitted_max = pending_timestamp
                pending.clear()
            last_timestamp = pd.Timestamp(timestamps.iloc[-1])
            last_start = int(timestamps.searchsorted(last_timestamp, side="left"))
            if last_start > start:
                complete = frame.iloc[start:last_start]
                try:
                    yield complete
                except GeneratorExit:
                    return
                emitted_max = pd.Timestamp(timestamps.iloc[last_start - 1])
            pending.append(frame.iloc[last_start:])
            pending_timestamp = last_timestamp
        if pending:
            complete = (
                pending[0]
                if len(pending) == 1
                else pd.concat(pending, ignore_index=True)
            )
            try:
                yield complete
            except GeneratorExit:
                return
        elif first_empty is not None:
            empty = first_empty.iloc[:0].copy()
            try:
                yield empty
            except GeneratorExit:
                return


def iter_complete_session_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
) -> Iterator[pd.DataFrame]:
    """Yield one complete contiguous timestamp session at a time."""
    with closing_iterator(
        iter_complete_time_frames(frames, timestamp_column=timestamp_column)
    ) as complete_frames:
        for frame in complete_frames:
            if frame.empty:
                try:
                    yield frame
                except GeneratorExit:
                    return
                continue
            timestamps = strict_timestamp_series(
                frame[timestamp_column], error="invalid_time_chunk_timestamp_values"
            ).to_numpy(dtype="datetime64[ns]", copy=False)
            stops = np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1
            start = 0
            for stop in stops:
                session = frame.iloc[start:stop]
                try:
                    yield session
                except GeneratorExit:
                    return
                start = int(stop)
            session = frame.iloc[start:]
            try:
                yield session
            except GeneratorExit:
                return
