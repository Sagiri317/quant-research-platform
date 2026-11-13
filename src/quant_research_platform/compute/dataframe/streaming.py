"""Timestamp-complete DataFrame streams for batched calculations."""

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
    one_timestamp: bool = False,
) -> Iterator[pd.DataFrame]:
    """Keep complete timestamps together, optionally yielding each one separately."""
    pending: list[tuple[pd.DataFrame, pd.Series]] = []
    pending_timestamp: pd.Timestamp | None = None
    emitted_max: pd.Timestamp | None = None
    first_empty: pd.DataFrame | None = None

    def completed():
        complete = (
            pending[0][0]
            if len(pending) == 1
            else pd.concat([frame for frame, _ in pending], ignore_index=True)
        )
        if not one_timestamp:
            yield complete
            return
        timestamps = np.concatenate(
            [times.to_numpy(dtype="datetime64[ns]", copy=False) for _, times in pending]
        )
        stops = np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1
        start = 0
        for stop in stops:
            yield complete.iloc[start:stop]
            start = int(stop)
        yield complete.iloc[start:]

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
            if pending:
                if current_min < pending_timestamp:
                    raise ValueError("non_monotonic_time_chunk_order")
            last_timestamp = pd.Timestamp(timestamps.iloc[-1])
            if pending and last_timestamp == pending_timestamp:
                pending.append((frame, timestamps))
                continue
            last_start = int(timestamps.searchsorted(last_timestamp, side="left"))
            if last_start:
                pending.append((frame.iloc[:last_start], timestamps.iloc[:last_start]))
            if pending:
                yield from completed()
                emitted_max = (
                    pd.Timestamp(timestamps.iloc[last_start - 1])
                    if last_start
                    else pending_timestamp
                )
                pending.clear()
            pending.append((frame.iloc[last_start:], timestamps.iloc[last_start:]))
            pending_timestamp = last_timestamp
        if pending:
            yield from completed()
        elif first_empty is not None:
            empty = first_empty.iloc[:0].copy()
            yield empty


def iter_complete_session_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
) -> Iterator[pd.DataFrame]:
    """Yield one complete contiguous timestamp session at a time."""
    yield from iter_complete_time_frames(
        frames, timestamp_column=timestamp_column, one_timestamp=True
    )
