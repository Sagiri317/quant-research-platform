"""Internal memory-bound chunk helpers for research compute layers."""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil

import pandas as pd

from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)


def iter_time_sized_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    axis: ChunkAxis = ChunkAxis.TIME,
) -> Iterator[pd.DataFrame]:
    """Yield timestamp-session chunks sized by the shared runtime policy."""

    if timestamp_column not in frame.columns:
        raise ValueError("missing_time_chunk_timestamp_column")
    if frame.empty:
        yield frame
        return

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_time_chunk_timestamp_values")

    sessions = pd.Index(timestamps.dropna().unique()).sort_values()
    if len(sessions) <= 1:
        yield frame
        return

    estimated_bytes = _estimated_bytes_per_session(frame, session_count=len(sessions))
    plan = ResourceChunkPlanner(RuntimeResourcePolicy.from_environment()).plan_sessions(
        axis=axis,
        session_count=len(sessions),
        estimated_bytes_per_session=estimated_bytes,
    )
    if plan.chunk_count <= 1:
        yield frame
        return

    target = plan.target_core_session_count
    for start in range(0, len(sessions), target):
        window_sessions = sessions[start : start + target]
        yield frame.loc[timestamps.isin(window_sessions)]


def _estimated_bytes_per_session(frame: pd.DataFrame, *, session_count: int) -> int:
    total = int(frame.memory_usage(index=True, deep=True).sum())
    return max(1, ceil(total / max(1, session_count)))
