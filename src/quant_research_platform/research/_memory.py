"""Internal memory-bound chunk helpers for research compute layers."""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil

import pandas as pd

from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
)


def iter_time_sized_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    axis: ChunkAxis = ChunkAxis.TIME,
    resource_policy: RuntimeResourcePolicy | None = None,
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
    policy = resource_policy or active_runtime_resource_policy()
    plan = ResourceChunkPlanner(policy).plan_sessions(
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


def iter_complete_time_sized_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
    axis: ChunkAxis = ChunkAxis.TIME,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield chunks without splitting any timestamp session across inputs."""

    pending: pd.DataFrame | None = None
    pending_min: pd.Timestamp | None = None
    pending_max: pd.Timestamp | None = None
    emitted_max: pd.Timestamp | None = None
    saw_empty = False

    for frame in frames:
        if timestamp_column not in frame.columns:
            raise ValueError("missing_time_chunk_timestamp_column")
        if frame.empty:
            saw_empty = True
            continue
        current_timestamps = _validated_time_chunk_timestamps(
            frame,
            timestamp_column=timestamp_column,
        )
        current_min = current_timestamps.min()
        current_max = current_timestamps.max()
        if emitted_max is not None and current_min <= emitted_max:
            raise ValueError("non_monotonic_time_chunk_order")
        if pending is None:
            pending = frame
            pending_min = current_min
            pending_max = current_max
            continue
        if pending_max is None or pending_min is None:
            raise ValueError("invalid_time_chunk_state")
        if current_min < pending_max:
            raise ValueError("non_monotonic_time_chunk_order")
        if current_min == pending_max:
            pending = _concat_time_boundary_frames(
                pending,
                frame,
                resource_policy=resource_policy,
            )
            pending_min = min(pending_min, current_min)
            pending_max = max(pending_max, current_max)
            continue
        yield from iter_time_sized_frames(
            pending,
            timestamp_column=timestamp_column,
            axis=axis,
            resource_policy=resource_policy,
        )
        emitted_max = pending_max
        pending = frame
        pending_min = current_min
        pending_max = current_max

    if pending is None:
        if saw_empty:
            yield pd.DataFrame(columns=[timestamp_column])
        return
    yield from iter_time_sized_frames(
        pending,
        timestamp_column=timestamp_column,
        axis=axis,
        resource_policy=resource_policy,
    )


def _estimated_bytes_per_session(frame: pd.DataFrame, *, session_count: int) -> int:
    total = int(frame.memory_usage(index=True, deep=True).sum())
    return max(1, ceil(total / max(1, session_count)))


def _validated_time_chunk_timestamps(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.Series:
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_time_chunk_timestamp_values")
    return timestamps


def _concat_time_boundary_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> pd.DataFrame:
    result = pd.concat([left, right], ignore_index=True)
    policy = resource_policy or active_runtime_resource_policy()
    estimated_bytes = int(result.memory_usage(index=True, deep=True).sum())
    if estimated_bytes > policy.effective_chunk_bytes:
        raise ValueError("time_chunk_boundary_overlap_exceeds_memory_budget")
    return result
