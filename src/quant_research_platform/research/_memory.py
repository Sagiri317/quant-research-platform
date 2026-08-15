"""Internal memory-bound chunk helpers for research compute layers."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator


class _ResearchFrameAccumulator:
    """Pre-admit one required complete research DataFrame result."""

    def __init__(
        self,
        owner: str,
        *,
        resource_policy: RuntimeResourcePolicy | None,
    ) -> None:
        self._owner = owner
        self._policy = resource_policy
        self._frames: list[pd.DataFrame] = []
        self._retained = ExitStack()
        self._columns: tuple[object, ...] | None = None
        self._dtypes: tuple[object, ...] | None = None
        self._empty_frame: pd.DataFrame | None = None
        self._output_data_bytes = 0
        self._output_index_bytes = 0
        self._shallow_bytes = 0
        self._row_count = 0
        self._closed = False

    @property
    def row_count(self) -> int:
        return self._row_count

    def append(self, frame: pd.DataFrame) -> None:
        if self._closed:
            raise RuntimeError("research_frame_accumulator_closed")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("invalid_research_frame")
        columns = tuple(frame.columns)
        dtypes = tuple(frame.dtypes)
        if self._columns is None:
            self._columns = columns
            self._dtypes = dtypes
        elif columns != self._columns or dtypes != self._dtypes:
            raise ValueError("research_frame_accumulator_schema_mismatch")
        if frame.empty:
            if self._empty_frame is None:
                self._empty_frame = frame
            return
        frame_bytes = pandas_frame_payload_bytes(frame)
        output_data_bytes = int(
            frame.memory_usage(index=False, deep=True).sum()
        )
        output_index_bytes = int(
            pd.RangeIndex(self._row_count + len(frame)).memory_usage(deep=True)
        )
        output_index_delta = max(
            0,
            output_index_bytes - self._output_index_bytes,
        )
        shallow_bytes = int(frame.memory_usage(index=True, deep=False).sum())
        policy = self._active_policy()
        pending = ExitStack()
        try:
            if policy is not None:
                pending.enter_context(
                    policy.reserve(
                        f"{self._owner}.retained",
                        frame_bytes,
                    )
                )
                pending.enter_context(
                    policy.reserve(
                        f"{self._owner}.result",
                        output_data_bytes,
                    )
                )
                if output_index_delta:
                    pending.enter_context(
                        policy.reserve(
                            f"{self._owner}.result_index",
                            output_index_delta,
                        )
                    )
            self._retained.enter_context(pending.pop_all())
        finally:
            pending.close()
        self._frames.append(frame)
        self._output_data_bytes += output_data_bytes
        self._output_index_bytes = output_index_bytes
        self._shallow_bytes += shallow_bytes
        self._row_count += len(frame)

    def materialize(self) -> pd.DataFrame:
        if self._closed:
            raise RuntimeError("research_frame_accumulator_closed")
        if not self._frames:
            return (
                pd.DataFrame()
                if self._empty_frame is None
                else self._empty_frame.copy()
            )
        policy = self._active_policy()
        with ExitStack() as workspace:
            if policy is not None:
                workspace.enter_context(
                    policy.reserve(
                        f"{self._owner}.concat",
                        self._shallow_bytes,
                    )
                )
            result = pd.concat(self._frames, ignore_index=True)
        if len(result) != self._row_count:
            raise RuntimeError("research_frame_accumulator_row_count_mismatch")
        if (
            tuple(result.columns) != self._columns
            or tuple(result.dtypes) != self._dtypes
        ):
            raise RuntimeError("research_frame_accumulator_schema_changed")
        if pandas_frame_payload_bytes(result) > (
            self._output_data_bytes + self._output_index_bytes
        ):
            raise RuntimeError("research_frame_accumulator_bound_exceeded")
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary = sys.exception()
        self._frames.clear()
        self._empty_frame = None
        try:
            self._retained.close()
        except BaseException:
            if primary is None:
                raise

    def __enter__(self) -> _ResearchFrameAccumulator:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _active_policy(self) -> RuntimeResourcePolicy | None:
        active = current_runtime_resource_policy()
        if self._policy is not None and active is not self._policy:
            raise RuntimeError("runtime_resource_policy_conflict")
        return self._policy


def _iter_contiguous_session_frames(
    frame: pd.DataFrame,
    *,
    timestamps: pd.Series,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    policy = resource_policy
    frame_bytes = _frame_bytes(frame)
    remaining = policy.task_memory_budget_bytes - policy.reserved_memory_bytes
    if frame_bytes <= remaining:
        yield frame
        return

    start = 0
    while start < len(frame):
        remaining = policy.task_memory_budget_bytes - policy.reserved_memory_bytes
        stop = start
        output_bound = 0
        while stop < len(frame):
            session = timestamps.iloc[stop]
            session_stop = stop + 1
            while (
                session_stop < len(frame) and timestamps.iloc[session_stop] == session
            ):
                session_stop += 1
            session_bound = _frame_bytes(frame.iloc[stop:session_stop])
            if output_bound + session_bound > remaining:
                if stop == start:
                    raise ValueError(
                        "research_indivisible_time_unit_exceeds_task_memory_budget"
                    )
                break
            output_bound += session_bound
            stop = session_stop
        with policy.reserve("research.time.output", output_bound):
            output = frame.iloc[start:stop].copy()
            if _frame_bytes(output) > output_bound:
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

    with ExitStack() as pending_capture_cleanup:
        pending_capture_cleanup.enter_context(closing_dataset_iterator(frames))
        bound_policy = bind_runtime_resource_policy(
            resource_policy,
            allow_implicit_active=True,
        )
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
        with closing_dataset_iterator(complete_frames):
            for frame in complete_frames:
                if frame.empty:
                    yield frame
                    continue
                policy = active_runtime_resource_policy()
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
                with policy.reserve("research.session.inspect", inspection_bytes):
                    timestamps = strict_timestamp_series(
                        frame[timestamp_column],
                        error="invalid_time_chunk_timestamp_values",
                    )
                    values = timestamps.to_numpy(dtype="datetime64[ns]", copy=False)
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

    return activate()


def _iter_complete_time_sized_frames_with_policy(
    frames: Iterator[pd.DataFrame],
    *,
    timestamp_column: str,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    pending: pd.DataFrame | None = None
    pending_timestamp: pd.Timestamp | None = None
    emitted_max: pd.Timestamp | None = None
    saw_empty = False
    first_empty: pd.DataFrame | None = None
    pending_reservation = ExitStack()

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
                resource_policy.reserved_memory_bytes + operation_bytes
                > resource_policy.task_memory_budget_bytes
            ):
                raise ValueError(
                    "research_indivisible_time_unit_exceeds_task_memory_budget"
                )
            with ExitStack() as frame_work:
                frame_work.enter_context(
                    resource_policy.reserve("research.time.inspect", operation_bytes)
                )
                timestamps = strict_timestamp_series(
                    frame[timestamp_column],
                    error="invalid_time_chunk_timestamp_values",
                )
                if not timestamps.is_monotonic_increasing:
                    sort_bytes = _time_sort_working_set_bytes(frame)
                    if (
                        resource_policy.reserved_memory_bytes + sort_bytes
                        > resource_policy.task_memory_budget_bytes
                    ):
                        raise ValueError(
                            "research_indivisible_time_unit_exceeds_task_memory_budget"
                        )
                    frame_work.enter_context(
                        resource_policy.reserve("research.time.sort", sort_bytes)
                    )
                    frame = frame.sort_values(
                        timestamp_column,
                        kind="mergesort",
                        ignore_index=True,
                    )
                    if _frame_bytes(frame) > sort_bytes:
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
                        combined = _concat_time_boundary_frames(
                            pending,
                            frame.iloc[:first_stop],
                            resource_policy=resource_policy,
                        )
                        pending_reservation.close()
                        pending_reservation = ExitStack()
                        combined_bytes = _frame_bytes(combined)
                        pending_reservation.enter_context(
                            resource_policy.reserve(
                                "research.time.pending",
                                combined_bytes,
                            )
                        )
                        pending = combined
                        if first_stop == len(frame):
                            continue
                        yield pending
                        emitted_max = pending_timestamp
                        pending = None
                        pending_timestamp = None
                        pending_reservation.close()
                        pending_reservation = ExitStack()
                        start = first_stop
                    else:
                        yield pending
                        emitted_max = pending_timestamp
                        pending = None
                        pending_timestamp = None
                        pending_reservation.close()
                        pending_reservation = ExitStack()

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
                        resource_policy=resource_policy,
                    )
                    emitted_max = pd.Timestamp(prefix_timestamps.iloc[-1])
                tail = frame.iloc[last_start:]
                tail_bytes = _frame_bytes(tail)
                if (
                    resource_policy.reserved_memory_bytes + tail_bytes
                    > resource_policy.task_memory_budget_bytes
                ):
                    raise ValueError(
                        "research_indivisible_time_unit_exceeds_task_memory_budget"
                    )
                pending_reservation.enter_context(
                    resource_policy.reserve(
                        "research.time.pending",
                        tail_bytes,
                    )
                )
                pending = tail.copy()
                if _frame_bytes(pending) > tail_bytes:
                    raise ValueError(
                        "research_indivisible_time_unit_exceeds_task_memory_budget"
                    )
                pending_timestamp = last_timestamp

        if pending is None:
            if saw_empty and first_empty is not None:
                yield first_empty.iloc[0:0].copy()
            return
        yield pending
    finally:
        pending_reservation.close()


def _concat_time_boundary_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    policy = resource_policy
    retained_payload_bytes = _frame_bytes(left) + _frame_bytes(right)
    if (
        policy.reserved_memory_bytes + retained_payload_bytes
        > policy.task_memory_budget_bytes
    ):
        raise ValueError("research_time_boundary_payloads_exceed_task_memory_budget")
    with runtime_resource_policy_scope(policy), policy.reserve(
        "research.time.boundary",
        retained_payload_bytes,
    ):
        result = pd.concat([left, right], ignore_index=True)
        if _frame_bytes(result) > retained_payload_bytes:
            raise ValueError(
                "research_time_boundary_payloads_exceed_task_memory_budget"
            )
    return result


def _frame_bytes(frame: pd.DataFrame) -> int:
    return pandas_frame_payload_bytes(frame)


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
    return _frame_bytes(frame) + 2 * index_bytes
