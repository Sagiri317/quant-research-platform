"""Internal memory-bound chunk helpers for research compute layers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from math import ceil

import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator


def iter_time_sized_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield timestamp-session chunks sized by the shared runtime policy."""

    bound_policy = bind_runtime_resource_policy(resource_policy)

    def activate() -> Iterator[pd.DataFrame]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            yield from _iter_time_sized_frames_with_policy(
                frame,
                timestamp_column=timestamp_column,
                resource_policy=policy,
            )

    return activate()


def _iter_time_sized_frames_with_policy(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:

    if timestamp_column not in frame.columns:
        raise ValueError("missing_time_chunk_timestamp_column")
    if frame.empty:
        yield frame
        return

    policy = resource_policy

    timestamps = strict_timestamp_series(
        frame[timestamp_column], error="invalid_time_chunk_timestamp_values"
    )

    sessions = pd.Index(timestamps.dropna().unique()).sort_values()
    if len(sessions) <= 1:
        _require_time_chunk_budget(frame, policy=policy)
        yield frame
        return

    estimated_bytes = _estimated_bytes_per_session(frame, session_count=len(sessions))
    target = max(
        1,
        min(
            len(sessions),
            policy.payload_target_bytes // estimated_bytes,
        ),
    )
    if target >= len(sessions):
        _require_time_chunk_budget(frame, policy=policy)
        yield frame
        return

    for start in range(0, len(sessions), target):
        window_sessions = sessions[start : start + target]
        output = frame.loc[timestamps.isin(window_sessions)]
        _require_time_chunk_budget(output, policy=policy)
        yield output


def iter_complete_time_sized_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield chunks without splitting any timestamp session across inputs."""

    with ExitStack() as pending_capture_cleanup:
        pending_capture_cleanup.enter_context(closing_dataset_iterator(frames))
        bound_policy = bind_runtime_resource_policy(resource_policy)
        activation_cleanup = pending_capture_cleanup.pop_all()

    def activate() -> Iterator[pd.DataFrame]:
        with ExitStack() as pending_entry_cleanup:
            pending_entry_cleanup.enter_context(activation_cleanup)
            policy = bound_policy or active_runtime_resource_policy()
            with runtime_resource_policy_scope(policy):
                with pending_entry_cleanup.pop_all():
                    yield from _iter_complete_time_sized_frames_with_policy(
                        frames,
                        timestamp_column=timestamp_column,
                        resource_policy=policy,
                    )

    return activate()


def _iter_complete_time_sized_frames_with_policy(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:

    pending: pd.DataFrame | None = None
    pending_min: pd.Timestamp | None = None
    pending_max: pd.Timestamp | None = None
    emitted_max: pd.Timestamp | None = None
    saw_empty = False
    first_empty: pd.DataFrame | None = None

    for frame in frames:
        if timestamp_column not in frame.columns:
            raise ValueError("missing_time_chunk_timestamp_column")
        if frame.empty:
            saw_empty = True
            if first_empty is None:
                first_empty = frame
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
        yield from _iter_time_sized_frames_with_policy(
            pending,
            timestamp_column=timestamp_column,
            resource_policy=resource_policy,
        )
        emitted_max = pending_max
        pending = frame
        pending_min = current_min
        pending_max = current_max

    if pending is None:
        if saw_empty and first_empty is not None:
            yield first_empty.iloc[0:0].copy()
        return
    yield from _iter_time_sized_frames_with_policy(
        pending,
        timestamp_column=timestamp_column,
        resource_policy=resource_policy,
    )


def _estimated_bytes_per_session(frame: pd.DataFrame, *, session_count: int) -> int:
    total = pandas_frame_payload_bytes(frame)
    return max(1, ceil(total / max(1, session_count)))


def _validated_time_chunk_timestamps(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.Series:
    return strict_timestamp_series(
        frame[timestamp_column], error="invalid_time_chunk_timestamp_values"
    )


def _concat_time_boundary_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    policy = resource_policy
    retained_payload_bytes = _frame_bytes(left) + _frame_bytes(right)
    if retained_payload_bytes > policy.task_memory_budget_bytes:
        raise ValueError("research_time_boundary_payloads_exceed_task_memory_budget")
    result = pd.concat([left, right], ignore_index=True)
    if (
        retained_payload_bytes + _frame_bytes(result)
        > policy.task_memory_budget_bytes
    ):
        raise ValueError("research_time_boundary_payloads_exceed_task_memory_budget")
    return result


def _frame_bytes(frame: pd.DataFrame) -> int:
    return pandas_frame_payload_bytes(frame)


def _require_time_chunk_budget(
    frame: pd.DataFrame,
    *,
    policy: RuntimeResourcePolicy,
) -> None:
    if _frame_bytes(frame) > policy.payload_target_bytes:
        raise ValueError("research_indivisible_time_unit_exceeds_payload_target")
