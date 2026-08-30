"""Internal memory-bound chunk helpers for research compute layers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    clear_exception_frames,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)


def _iter_contiguous_session_frames(
    frame: pd.DataFrame,
    *,
    timestamps: pd.Series,
) -> Iterator[pd.DataFrame]:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    frame_bytes = pandas_frame_payload_bytes(frame)
    remaining = policy.operation_headroom_bytes
    if frame_bytes <= remaining:
        yield frame
        return

    start = 0
    while start < len(frame):
        remaining = policy.operation_headroom_bytes
        stop = start
        output_bound = 0
        while stop < len(frame):
            session = timestamps.iloc[stop]
            session_stop = stop + 1
            while (
                session_stop < len(frame) and timestamps.iloc[session_stop] == session
            ):
                session_stop += 1
            session_bound = pandas_frame_payload_bytes(frame.iloc[stop:session_stop])
            if output_bound + session_bound > remaining:
                if stop == start:
                    explicit_cap = policy.explicit_memory_cap_bytes
                    if (
                        explicit_cap is not None
                        and policy.used_memory_bytes + session_bound > explicit_cap
                    ):
                        raise ValueError(
                            "research_indivisible_time_unit_exceeds_task_memory_budget"
                        )
                    output_bound = session_bound
                    stop = session_stop
                break
            output_bound += session_bound
            stop = session_stop
        with policy.lease("research.time.output", output_bound):
            output = frame.iloc[start:stop].copy()
            if pandas_frame_payload_bytes(output) > output_bound:
                raise ValueError(
                    "research_indivisible_time_unit_exceeds_task_memory_budget"
                )
            yield output
        del output
        start = stop


def iter_complete_time_sized_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield chunks without splitting any timestamp session across inputs."""

    bound_policy = resource_policy or current_runtime_resource_policy()

    def activate() -> Iterator[pd.DataFrame]:
        with runtime_resource_policy_scope(bound_policy), closing_iterator(frames):
            yield from _iter_complete_time_sized_frames_impl(
                frames,
                timestamp_column=timestamp_column,
            )

    return activate()


def iter_complete_session_frames(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield one complete contiguous timestamp session at a time."""

    complete_frames = iter_complete_time_sized_frames(
        frames,
        timestamp_column=timestamp_column,
        resource_policy=resource_policy,
    )

    def activate() -> Iterator[pd.DataFrame]:
        with closing_iterator(complete_frames):
            for frame in complete_frames:
                if frame.empty:
                    yield frame
                    continue
                policy = current_runtime_resource_policy()
                if policy is None:
                    raise RuntimeError("runtime_resource_policy_not_active")
                boundary_bytes = len(frame) * (
                    np.dtype(np.bool_).itemsize + np.dtype(np.intp).itemsize
                )
                inspection_bytes = (
                    _time_chunk_operation_bytes(
                        frame,
                        timestamp_column=timestamp_column,
                    )
                    + boundary_bytes
                )
                with policy.lease("research.session.inspect", inspection_bytes):
                    timestamps = values = stops = None
                    try:
                        try:
                            timestamps = strict_timestamp_series(
                                frame[timestamp_column],
                                error="invalid_time_chunk_timestamp_values",
                            )
                        except BaseException as error:
                            clear_exception_frames(error)
                            raise
                        values = timestamps.to_numpy(
                            dtype="datetime64[ns]",
                            copy=False,
                        )
                        stops = np.flatnonzero(values[1:] != values[:-1]) + 1
                        if len(stops) == 0:
                            yield frame
                            continue
                        start = 0
                        for raw_stop in stops:
                            stop = int(raw_stop)
                            yield frame.iloc[start:stop]
                            start = stop
                        yield frame.iloc[start:]
                    finally:
                        timestamps = values = stops = None

    return activate()


def _iter_complete_time_sized_frames_impl(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
) -> Iterator[pd.DataFrame]:
    resource_policy = current_runtime_resource_policy()
    if resource_policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    pending: OwnedValue[pd.DataFrame] | None = None
    pending_timestamp: pd.Timestamp | None = None
    emitted_max: pd.Timestamp | None = None
    saw_empty = False
    first_empty: pd.DataFrame | None = None

    try:
        for frame in frames:
            if timestamp_column not in frame.columns:
                raise ValueError("missing_time_chunk_timestamp_column")
            if frame.empty:
                saw_empty = True
                if first_empty is None:
                    first_empty = frame
                continue
            operation_bytes = _time_chunk_operation_bytes(
                frame,
                timestamp_column=timestamp_column,
            )
            if (
                resource_policy.explicit_memory_cap_bytes is not None
                and resource_policy.used_memory_bytes + operation_bytes
                > resource_policy.explicit_memory_cap_bytes
            ):
                raise ValueError(
                    "research_indivisible_time_unit_exceeds_task_memory_budget"
                )
            with ExitStack() as frame_work:
                frame_work.enter_context(
                    resource_policy.lease("research.time.inspect", operation_bytes)
                )
                timestamps = strict_timestamp_series(
                    frame[timestamp_column],
                    error="invalid_time_chunk_timestamp_values",
                )
                if not timestamps.is_monotonic_increasing:
                    sort_bytes = _time_sort_working_set_bytes(frame)
                    if (
                        resource_policy.explicit_memory_cap_bytes is not None
                        and resource_policy.used_memory_bytes + sort_bytes
                        > resource_policy.explicit_memory_cap_bytes
                    ):
                        raise ValueError(
                            "research_indivisible_time_unit_exceeds_task_memory_budget"
                        )
                    frame_work.enter_context(
                        resource_policy.lease("research.time.sort", sort_bytes)
                    )
                    frame = frame.sort_values(
                        timestamp_column,
                        kind="mergesort",
                        ignore_index=True,
                    )
                    if pandas_frame_payload_bytes(frame) > sort_bytes:
                        raise ValueError(
                            "research_indivisible_time_unit_exceeds_task_memory_budget"
                        )
                    timestamps = strict_timestamp_series(
                        frame[timestamp_column],
                        error="invalid_time_chunk_timestamp_values",
                    )
                current_min = pd.Timestamp(timestamps.iloc[0])
                if emitted_max is not None and current_min <= emitted_max:
                    raise ValueError("non_monotonic_time_chunk_order")
                start = 0
                if pending is not None:
                    if pending_timestamp is None or current_min < pending_timestamp:
                        raise ValueError("non_monotonic_time_chunk_order")
                    if current_min == pending_timestamp:
                        first_stop = 1
                        while (
                            first_stop < len(frame)
                            and timestamps.iloc[first_stop] == current_min
                        ):
                            first_stop += 1
                        right = frame.iloc[:first_stop]
                        combined_bound = pandas_frame_payload_bytes(
                            pending.value
                        ) + pandas_frame_payload_bytes(right)
                        if (
                            resource_policy.explicit_memory_cap_bytes is not None
                            and resource_policy.used_memory_bytes + combined_bound
                            > resource_policy.explicit_memory_cap_bytes
                        ):
                            raise ValueError(
                                "research_time_boundary_payloads_exceed_task_memory_budget"
                            )
                        pending.lease.grow(combined_bound)
                        combined = pd.concat(
                            [pending.value, right],
                            ignore_index=True,
                        )
                        combined_bytes = pandas_frame_payload_bytes(combined)
                        if combined_bytes > combined_bound:
                            raise ValueError(
                                "research_time_boundary_payloads_exceed_task_memory_budget"
                            )
                        pending.value = combined
                        pending.lease.shrink_to(combined_bytes)
                        if first_stop == len(frame):
                            continue
                        yield pending.value
                        emitted_max = pending_timestamp
                        pending.release(owner="research.time.pending")
                        pending = None
                        pending_timestamp = None
                        start = first_stop
                    else:
                        yield pending.value
                        emitted_max = pending_timestamp
                        pending.release(owner="research.time.pending")
                        pending = None
                        pending_timestamp = None

                last_timestamp = pd.Timestamp(timestamps.iloc[-1])
                last_start = len(frame) - 1
                while (
                    last_start > start
                    and timestamps.iloc[last_start - 1] == last_timestamp
                ):
                    last_start -= 1
                if last_start > start:
                    prefix = frame.iloc[start:last_start]
                    prefix_timestamps = timestamps.iloc[start:last_start]
                    yield from _iter_contiguous_session_frames(
                        prefix,
                        timestamps=prefix_timestamps,
                    )
                    emitted_max = pd.Timestamp(prefix_timestamps.iloc[-1])
                tail = frame.iloc[last_start:]
                tail_bytes = pandas_frame_payload_bytes(tail)
                if (
                    resource_policy.explicit_memory_cap_bytes is not None
                    and resource_policy.used_memory_bytes + tail_bytes
                    > resource_policy.explicit_memory_cap_bytes
                ):
                    raise ValueError(
                        "research_indivisible_time_unit_exceeds_task_memory_budget"
                    )
                pending_lease = resource_policy.lease(
                    "research.time.pending",
                    tail_bytes,
                )
                try:
                    pending_value = tail.copy()
                    if pandas_frame_payload_bytes(pending_value) > tail_bytes:
                        raise ValueError(
                            "research_indivisible_time_unit_exceeds_task_memory_budget"
                        )
                    pending = OwnedValue(pending_value, pending_lease)
                except BaseException as primary:
                    try:
                        pending_lease.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "research pending lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    raise
                pending_timestamp = last_timestamp

        if pending is None:
            if saw_empty and first_empty is not None:
                yield first_empty.iloc[0:0].copy()
            return
        yield pending.value
    finally:
        if pending is not None:
            pending.release(owner="research.time.pending")


def _selected_frame_payload_bytes(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
) -> int:
    return int(frame.index.memory_usage(deep=True)) + sum(
        int(frame[column].memory_usage(index=False, deep=True)) for column in columns
    )


def _time_chunk_operation_bytes(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> int:
    return int(frame[timestamp_column].memory_usage(index=False, deep=True)) + 40 * len(
        frame
    )


def _time_sort_working_set_bytes(frame: pd.DataFrame) -> int:
    row_count = len(frame)
    index_bytes = row_count * np.dtype(np.intp).itemsize
    return 2 * pandas_frame_payload_bytes(frame) + 2 * index_bytes
