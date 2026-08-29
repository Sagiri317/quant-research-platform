"""A-share processed materialization planning helpers."""

from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from hashlib import sha256
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import (
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow, require_date_window
from quant_research_platform.contracts.validation import require_non_negative_int
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
)
from quant_research_platform.markets.implementations.a_share.fields import (
    field_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io as materialization_io,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.inspection import _inspect_dataset_owned
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
from quant_research_platform.storage.exact_keys import (
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage.identity import require_artifact_root
from quant_research_platform.storage.layout import StorageLayout


def filter_output_sessions(
    frame: pd.DataFrame,
    sessions: Sequence[pd.Timestamp],
    *,
    columns: Sequence[str] | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    selected_columns = tuple(frame.columns) if columns is None else tuple(columns)
    filter_state_bytes = len(frame) * (
        np.dtype(np.bool_).itemsize + np.dtype(np.intp).itemsize
    )
    output_bound = (
        pandas_frame_payload_bytes(frame) + len(frame) * np.dtype(np.intp).itemsize
    )
    with (
        runtime_resource_policy_scope(resource_policy),
        resource_policy.lease(
            "a_share.output_filter",
            filter_state_bytes,
        ),
    ):
        if any(
            type(timestamp) is not pd.Timestamp
            or timestamp.tz is not None
            or timestamp != timestamp.normalize()
            for timestamp in frame["timestamp"]
        ):
            raise ValueError("invalid_output_session_timestamp")
        output = frame.loc[
            frame["timestamp"].isin(sessions),
            selected_columns,
        ].copy()
        if pandas_frame_payload_bytes(output) > output_bound:
            raise materialization_io.MaterializationCapacityExceeded(
                "a_share_known_payloads_exceed_task_memory_budget"
            )
        return output


def require_dataset_coverage(
    root: Path,
    time_window: TimeWindow,
    *,
    sessions: pd.DatetimeIndex,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    require_date_window(time_window, context="a_share_dataset_coverage")
    first, stop = _target_session_bounds(sessions, time_window)
    coverage_window = TimeWindow(sessions[first].date(), sessions[stop - 1].date())
    expected_sessions = sessions[first:stop]
    coverage_state_bytes = len(expected_sessions) * np.dtype(np.bool_).itemsize
    with resource_policy.lease("a_share.planning.coverage", coverage_state_bytes):
        covered = np.zeros(len(expected_sessions), dtype=np.bool_)
        owned_inspection = _inspect_dataset_owned(
            root,
            resource_policy=resource_policy,
        )
        try:
            if "timestamp" not in owned_inspection.value.columns:
                raise ValueError("time_window_not_supported")
        finally:
            owned_inspection.release(owner="a_share.planning.calendar.inspection")
        with open_dataset_session(
            root,
            columns=("timestamp",),
            time_window=coverage_window,
            resource_policy=resource_policy,
        ) as session:
            chunks = iter_dataset_chunks(
                session,
                target_chunk_bytes=resource_policy.planning_headroom_bytes,
            )
            with closing_iterator(chunks):
                for chunk in chunks:
                    for raw_timestamp in chunk["timestamp"]:
                        try:
                            timestamp = pd.Timestamp(raw_timestamp)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(
                                "invalid_dataset_session_timestamp"
                            ) from exc
                        if pd.isna(timestamp):
                            raise ValueError("invalid_dataset_session_timestamp")
                        if timestamp.tz is not None:
                            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
                        timestamp = timestamp.normalize()
                        position = int(expected_sessions.searchsorted(timestamp))
                        if (
                            position < len(expected_sessions)
                            and expected_sessions[position] == timestamp
                        ):
                            covered[position] = True
        missing_count = 0
        first_missing = None
        for position, is_covered in enumerate(covered):
            if is_covered:
                continue
            missing_count += 1
            if first_missing is None:
                first_missing = expected_sessions[position].date()
        if missing_count:
            raise ValueError(
                "insufficient_dataset_coverage: "
                f"missing_sessions={missing_count} "
                f"first_missing_session={first_missing}"
            )


def iter_active_materialization_batches(
    root: Path,
    *,
    asset_column: str = "asset_id",
    time_window: TimeWindow,
    sessions: Sequence[pd.Timestamp],
    resource_policy: RuntimeResourcePolicy,
    valid_bar_offsets: tuple[int, int] | None = None,
    output_row_bound_bytes: int | None = None,
    materialize: Callable[..., pd.DataFrame],
) -> Iterator[object]:
    bound_policy = resource_policy
    if current_runtime_resource_policy() is not bound_policy:
        raise RuntimeError("a_share_materialization_policy_scope_required")
    if valid_bar_offsets is not None:
        if type(valid_bar_offsets) is not tuple or len(valid_bar_offsets) != 2:
            raise ValueError("invalid_valid_bar_offset")
        for offset in valid_bar_offsets:
            require_non_negative_int(
                offset, error="invalid_valid_bar_offset"
            )
    if (
        isinstance(sessions, pd.DatetimeIndex)
        and type(sessions) is not pd.DatetimeIndex
    ):
        raise ValueError("invalid_session_dates")
    if output_row_bound_bytes is not None and (
        type(output_row_bound_bytes) is not int or output_row_bound_bytes <= 0
    ):
        raise ValueError("invalid_a_share_output_row_bound")
    assert bound_policy is not None
    session_context = sessions
    first_session, stop_session = _target_session_bounds(
        session_context,
        time_window,
    )
    target_count = stop_session - first_session
    target_slice_bytes = (
        0
        if first_session == 0 and stop_session == len(session_context)
        else target_count * np.dtype(object).itemsize
    )
    target_sessions = (
        session_context
        if target_slice_bytes == 0
        else session_context[first_session:stop_session]
    )

    def produce_under_policy() -> Iterator[object]:
        assert bound_policy is not None
        require_artifact_root(root)
        pointer_bytes = np.dtype(object).itemsize
        active_index = ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
            scratch_parent=root.parent,
        )
        with (
            bound_policy.lease(
                "a_share.active_index",
                active_index.retained_memory_bytes,
            ),
            active_index as index,
        ):
            discovery_headroom = (
                bound_policy.memory_budget_bytes
                - bound_policy.used_memory_bytes
            )
            if discovery_headroom <= 0:
                raise materialization_io.MaterializationCapacityExceeded(
                    "a_share_known_payloads_exceed_task_memory_budget"
                )
            discovery_target = discovery_headroom
            with open_dataset_session(
                root,
                columns=(asset_column,),
                time_window=time_window,
                resource_policy=bound_policy,
            ) as session:
                chunks = iter_dataset_chunks(
                    session,
                    target_chunk_bytes=discovery_target,
                )
                with closing_iterator(chunks):
                    for chunk in chunks:
                        if asset_column not in chunk:
                            raise ValueError("missing_processed_panel_asset_id")
                        source_values = chunk[asset_column]

                        def active_asset_rows(
                            values: pd.Series = source_values,
                        ) -> Iterator[tuple[str, bytes, bytes]]:
                            for value in values:
                                if not isinstance(value, str) or not value:
                                    raise ValueError("invalid_processed_panel_asset_id")
                                encoded_bound = sys.getsizeof(b"") + 4 * len(value)
                                available_bytes = (
                                    bound_policy.memory_budget_bytes
                                    - bound_policy.used_memory_bytes
                                )
                                if encoded_bound > available_bytes:
                                    raise (
                                        materialization_io.MaterializationCapacityExceeded(
                                            "a_share_known_payloads_exceed_task_memory_budget"
                                        )
                                    )
                                encoded = value.encode("utf-8")
                                yield "active-assets", encoded, encoded

                        index.add_identity_payload_rows(rows=active_asset_rows())
                        del source_values, chunk

            batch_state_bytes = sys.getsizeof([]) + sys.getsizeof(())

            def emit_asset_batch(
                output: tuple[str, ...],
            ) -> Iterator[object]:
                if valid_bar_offsets is None:
                    yield from _iter_adaptive_materialization_batches(
                        asset_ids=output,
                        sessions=target_sessions,
                        session_context=session_context,
                        output_row_bound_bytes=output_row_bound_bytes,
                        materialize=materialize,
                    )
                    return
                valid_bar_index = ExactPrimaryKeyIndex(
                    primary_key=("asset_id", "timestamp"),
                    timestamp_columns=("timestamp",),
                    scratch_parent=root.parent,
                )
                with (
                    bound_policy.lease(
                        "a_share.valid_bar_index",
                        valid_bar_index.retained_memory_bytes,
                    ),
                    valid_bar_index,
                ):
                    _build_valid_bar_key_index(
                        root,
                        index=valid_bar_index,
                        resource_policy=bound_policy,
                        asset_ids=output,
                        time_window=None,
                    )
                    yield from _iter_adaptive_materialization_batches(
                        asset_ids=output,
                        sessions=target_sessions,
                        session_context=session_context,
                        valid_bar_index=valid_bar_index,
                        valid_bar_offsets=valid_bar_offsets,
                        output_row_bound_bytes=output_row_bound_bytes,
                        materialize=materialize,
                    )

            batch_available = (
                bound_policy.memory_budget_bytes
                - bound_policy.used_memory_bytes
                - batch_state_bytes
            )
            if batch_available <= 0:
                raise materialization_io.MaterializationCapacityExceeded(
                    "a_share_known_payloads_exceed_task_memory_budget"
                )
            asset_batch_limit = batch_available - max(1, output_row_bound_bytes or 1)
            if asset_batch_limit <= 0:
                raise materialization_io.MaterializationCapacityExceeded(
                    "a_share_known_payloads_exceed_task_memory_budget"
                )
            asset_batch: list[str] = []
            asset_batch_bytes = 0
            payloads = index.identity_payload_values(namespace="active-assets")
            asset_reservations = ExitStack()
            try:
                with closing_iterator(payloads):
                    for payload in payloads:
                        item_bytes = (
                            3 * pointer_bytes
                            + sys.getsizeof(b"")
                            + sys.getsizeof("")
                            + 5 * len(payload)
                        )
                        if item_bytes > asset_batch_limit:
                            raise materialization_io.MaterializationCapacityExceeded(
                                "a_share_known_payloads_exceed_task_memory_budget"
                            )
                        if asset_batch and (
                            asset_batch_bytes + item_bytes > asset_batch_limit
                        ):
                            output = tuple(asset_batch)
                            yield from emit_asset_batch(output)
                            del output
                            asset_batch.clear()
                            asset_batch_bytes = 0
                            asset_reservations.close()
                            asset_reservations = ExitStack()
                        asset_reservations.enter_context(
                            bound_policy.lease("a_share.asset_batch", item_bytes)
                        )
                        asset_batch.append(payload.decode("utf-8"))
                        asset_batch_bytes += item_bytes
                    if asset_batch:
                        output = tuple(asset_batch)
                        yield from emit_asset_batch(output)
                        del output
                        asset_batch.clear()
            finally:
                asset_reservations.close()

    def activate_under_policy() -> Iterator[object]:
        assert bound_policy is not None
        yield from produce_under_policy()

    def activate() -> Iterator[object]:
        assert bound_policy is not None
        with runtime_resource_policy_scope(bound_policy):
            yield from activate_under_policy()

    return activate()


def _iter_adaptive_materialization_batches(
    *,
    asset_ids: tuple[object, ...],
    sessions: Sequence[pd.Timestamp],
    session_context: Sequence[pd.Timestamp],
    valid_bar_index: ExactPrimaryKeyIndex | None = None,
    valid_bar_offsets: tuple[int, int] | None = None,
    output_row_bound_bytes: int | None = None,
    materialize: Callable[..., pd.DataFrame],
) -> Iterator[object]:
    """Pack assets then sessions from one immutable soft throughput target."""
    policy = current_runtime_resource_policy()
    pointer_bytes = np.dtype(object).itemsize
    remaining = policy.memory_budget_bytes - policy.used_memory_bytes
    if remaining <= 0:
        raise materialization_io.MaterializationCapacityExceeded(
            "a_share_known_payloads_exceed_task_memory_budget"
        )
    producer_headroom_bytes = 128 * 1024
    producer_workspace_row_bound = (
        None
        if output_row_bound_bytes is None
        else 3 * output_row_bound_bytes
    )
    producer_output_row_bound = (
        None
        if output_row_bound_bytes is None
        else 4 * output_row_bound_bytes
    )
    cell_limit = (
        len(asset_ids) * len(sessions)
        if output_row_bound_bytes is None
        else max(
            1,
            max(0, remaining - producer_headroom_bytes)
            // (producer_output_row_bound + producer_workspace_row_bound),
        )
    )
    asset_start = 0
    asset_span = min(
        len(asset_ids),
        max(1, cell_limit // max(1, len(sessions))),
    )
    session_span_limit = min(
        len(sessions),
        max(1, cell_limit // asset_span),
    )
    while asset_start < len(asset_ids):
        asset_stop = min(len(asset_ids), asset_start + asset_span)
        session_start = 0
        while session_start < len(sessions):
            session_stop = min(
                len(sessions),
                session_start + session_span_limit,
            )
            planned_slice_bytes = (
                0
                if asset_start == 0 and asset_stop == len(asset_ids)
                else (asset_stop - asset_start) * pointer_bytes
            ) + (
                0
                if session_start == 0 and session_stop == len(sessions)
                else (session_stop - session_start) * pointer_bytes
            )
            if (
                policy.used_memory_bytes
                + planned_slice_bytes
                > policy.memory_budget_bytes
            ):
                raise materialization_io.MaterializationCapacityExceeded(
                    "a_share_known_payloads_exceed_task_memory_budget"
                )
            planned_output_bound = (
                max(1, producer_output_row_bound or 1)
                * max(1, (asset_stop - asset_start) * (session_stop - session_start))
                + producer_headroom_bytes
            )
            lease = yield planned_output_bound
            with policy.lease("a_share.batch.slice", planned_slice_bytes):
                current_assets = (
                    asset_ids
                    if asset_start == 0 and asset_stop == len(asset_ids)
                    else asset_ids[asset_start:asset_stop]
                )
                current_sessions = (
                    sessions
                    if session_start == 0 and session_stop == len(sessions)
                    else sessions[session_start:session_stop]
                )
                current_keys: pd.DataFrame | None = None
                owned_keys: OwnedValue[pd.DataFrame] | None = None
                try:
                    if valid_bar_index is None:
                        frame = materialize(
                            current_assets,
                            current_sessions,
                            session_context,
                        )
                    else:
                        assert valid_bar_offsets is not None
                        owned_keys = bounded_valid_bar_key_coverage(
                            valid_bar_index,
                            asset_ids=current_assets,
                            core_sessions=current_sessions,
                            lookback_bars=valid_bar_offsets[0],
                            lookahead_bars=valid_bar_offsets[1],
                            resource_policy=policy,
                        )
                        owned_keys.lease.transfer("a_share.materialization.keys")
                        current_keys = owned_keys.value
                        frame = materialize(
                            current_assets,
                            current_sessions,
                            session_context,
                            current_keys,
                        )
                except (
                    DatasetReadCapacityExceeded,
                    materialization_io.MaterializationCapacityExceeded,
                ):
                    raise materialization_io.MaterializationCapacityExceeded(
                        "a_share_known_payloads_exceed_task_memory_budget"
                    ) from None
                finally:
                    if owned_keys is not None:
                        owned_keys.release(owner="a_share.materialization.keys")
                    del current_assets, current_sessions, current_keys
            actual_output_bound = pandas_frame_payload_bytes(frame) + 64 * 1024
            if actual_output_bound > lease.reserved_bytes:
                raise ValueError("a_share_producer_bound_invalid")
            lease.shrink_to(actual_output_bound)
            yield OwnedValue(frame, lease)
            del frame
            session_start = session_stop
        asset_start = asset_stop


def session_batch_window(
    sessions: pd.DatetimeIndex,
    current_sessions: Sequence[pd.Timestamp],
    *,
    warmup_sessions: int = 0,
    lookahead_sessions: int = 0,
) -> TimeWindow:
    require_non_negative_int(
        warmup_sessions, error="invalid_a_share_session_offset"
    )
    require_non_negative_int(
        lookahead_sessions, error="invalid_a_share_session_offset"
    )
    first_session = current_sessions[0]
    last_session = current_sessions[-1]
    first = bisect_left(sessions, first_session)
    last = bisect_left(sessions, last_session)
    if first == len(sessions) or sessions[first] != first_session:
        raise KeyError(first_session)
    if last == len(sessions) or sessions[last] != last_session:
        raise KeyError(last_session)
    start = sessions[max(0, first - warmup_sessions)].date()
    end = sessions[min(len(sessions) - 1, last + lookahead_sessions)].date()
    return TimeWindow(start, end)


def _valid_bar_asset_namespace(asset_id: object) -> str:
    if type(asset_id) is not str or not asset_id:
        raise ValueError("invalid_processed_panel_asset_id")
    return "valid-bar:" + sha256(asset_id.encode("utf-8")).hexdigest()


def _valid_bar_timestamp_payload(timestamp_value: int) -> bytes:
    return (timestamp_value + (1 << 63)).to_bytes(8, "big")


def _valid_bar_timestamp_value(payload: bytes) -> int:
    if len(payload) != 8:
        raise ValueError("invalid_valid_bar_timestamp_payload")
    return int.from_bytes(payload, "big") - (1 << 63)


def _build_valid_bar_key_index(
    root: Path,
    *,
    index: ExactPrimaryKeyIndex,
    resource_policy: RuntimeResourcePolicy,
    asset_ids: tuple[object, ...] | None = None,
    time_window: TimeWindow | None = None,
) -> None:
    with open_dataset_session(
        root,
        columns=("asset_id", "timestamp"),
        time_window=time_window,
        resource_policy=resource_policy,
        row_filters=(
            ()
            if asset_ids is None
            else (("asset_id", "in", tuple(asset_ids)),)
        ),
    ) as session:
        frames = iter_dataset_chunks(
            session,
            target_chunk_bytes=resource_policy.planning_headroom_bytes,
        )
        with closing_iterator(frames):
            for frame in frames:
                missing = {"asset_id", "timestamp"}.difference(frame.columns)
                if missing:
                    raise ValueError("panel_key_columns_missing")
                inserted = index.add_identity_payload_rows(
                    rows=_iter_valid_bar_identity_rows(frame)
                )
                if inserted != len(frame):
                    raise ValueError("duplicate_panel_keys")
                del frame


def _require_valid_bar_timestamp(timestamp: object) -> None:
    if (
        type(timestamp) is not pd.Timestamp
        or timestamp.tz is not None
        or timestamp != timestamp.normalize()
    ):
        raise ValueError("invalid_processed_panel_timestamp")


def _iter_valid_bar_identity_rows(
    frame: pd.DataFrame,
) -> Iterator[tuple[str, bytes, bytes]]:
    for asset_id, timestamp in zip(frame["asset_id"], frame["timestamp"], strict=True):
        namespace = _valid_bar_asset_namespace(asset_id)
        _require_valid_bar_timestamp(timestamp)
        payload = _valid_bar_timestamp_payload(timestamp.value)
        yield namespace, payload, payload


def bounded_valid_bar_key_coverage(
    index: ExactPrimaryKeyIndex,
    *,
    asset_ids: tuple[object, ...],
    core_sessions: Sequence[pd.Timestamp],
    lookback_bars: int,
    lookahead_bars: int,
    resource_policy: RuntimeResourcePolicy,
) -> OwnedValue[pd.DataFrame]:
    """Query current assets' exact sparse boundary from one built disk index."""

    require_non_negative_int(lookback_bars, error="invalid_valid_bar_offset")
    require_non_negative_int(
        lookahead_bars, error="invalid_valid_bar_offset"
    )
    if not asset_ids:
        raise ValueError("invalid_processed_panel_asset_id")
    try:
        for asset_id in asset_ids:
            _valid_bar_asset_namespace(asset_id)
        if any(
            asset_ids[position - 1] >= asset_ids[position]
            for position in range(1, len(asset_ids))
        ):
            raise ValueError("invalid_processed_panel_asset_order")
    except TypeError as exc:
        raise ValueError("invalid_processed_panel_asset_order") from exc
    if len(core_sessions) == 0:
        raise ValueError("missing_core_sessions")
    if any(
        type(session) is not pd.Timestamp
        or session.tz is not None
        or session != session.normalize()
        for session in core_sessions
    ) or any(
        core_sessions[position - 1] >= core_sessions[position]
        for position in range(1, len(core_sessions))
    ):
        raise ValueError("invalid_processed_panel_timestamp")
    first_core = core_sessions[0].value
    last_core = core_sessions[-1].value
    slot_count = lookback_bars + 2 + lookahead_bars
    boundary_state_bytes = int(len(asset_ids) * slot_count * np.dtype("int64").itemsize)
    if (
        resource_policy.used_memory_bytes
        + boundary_state_bytes
        > resource_policy.memory_budget_bytes
    ):
        raise materialization_io.MaterializationCapacityExceeded(
            "a_share_known_payloads_exceed_task_memory_budget"
        )
    with ExitStack() as state:
        state.enter_context(runtime_resource_policy_scope(resource_policy))
        state.enter_context(
            resource_policy.lease(
                "a_share.valid_bar.boundary",
                boundary_state_bytes,
            )
        )
        boundary_timestamps = np.full(
            (len(asset_ids), slot_count),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )
        missing_value = np.iinfo(np.int64).min
        core_offset = lookback_bars
        after_offset = core_offset + 2
        for asset_position, asset_id in enumerate(asset_ids):
            asset_state = boundary_timestamps[asset_position]
            before_seen = 0
            core_first = missing_value
            core_last = missing_value
            after_count = 0
            values = index.identity_payload_values(
                namespace=_valid_bar_asset_namespace(asset_id)
            )
            with closing_iterator(values):
                for payload in values:
                    timestamp_value = _valid_bar_timestamp_value(payload)
                    if timestamp_value < first_core:
                        if lookback_bars:
                            asset_state[before_seen % lookback_bars] = timestamp_value
                            before_seen += 1
                    elif timestamp_value <= last_core:
                        if core_first == missing_value:
                            core_first = timestamp_value
                        core_last = timestamp_value
                    elif after_count < lookahead_bars:
                        asset_state[after_offset + after_count] = timestamp_value
                        after_count += 1
                    else:
                        break
            retained_before_count = min(before_seen, lookback_bars)
            if retained_before_count and before_seen > lookback_bars:
                _rotate_left_int64_prefix(
                    asset_state,
                    length=retained_before_count,
                    shift=before_seen % lookback_bars,
                )
            if core_first != missing_value:
                asset_state[core_offset] = core_first
                if core_last != core_first:
                    asset_state[core_offset + 1] = core_last
        flat_boundaries = boundary_timestamps.reshape(-1)
        output_count = sum(value != missing_value for value in flat_boundaries)
        output_state_bytes = int(
            output_count * (np.dtype(np.int64).itemsize + np.dtype(object).itemsize)
        )
        if (
            resource_policy.used_memory_bytes
            + output_state_bytes
            > resource_policy.memory_budget_bytes
        ):
            raise materialization_io.MaterializationCapacityExceeded(
                "a_share_known_payloads_exceed_task_memory_budget"
            )
        output_lease = resource_policy.lease(
            "a_share.valid_bar.keys",
            max(1, output_state_bytes + 128 * 1024),
        )
        try:
            selected_asset_values = np.empty(output_count, dtype=object)
            timestamp_values = np.empty(output_count, dtype=np.int64)
            output_position = 0
            for position, value in enumerate(flat_boundaries):
                if value == missing_value:
                    continue
                selected_asset_values[output_position] = asset_ids[
                    position // slot_count
                ]
                timestamp_values[output_position] = value
                output_position += 1
            output = pd.DataFrame(
                {
                    "asset_id": selected_asset_values,
                    "timestamp": timestamp_values.view("datetime64[ns]"),
                },
                copy=False,
            )
            actual_output_bound = (
                int(output.memory_usage(index=True, deep=False).sum()) + 64 * 1024
            )
            output_lease.shrink_to(actual_output_bound)
        except BaseException as primary:
            try:
                output_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "a_share valid-bar key lease close also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        return OwnedValue(output, output_lease)


def available_trading_sessions(
    layout: StorageLayout,
    source_id: SourceId,
    *,
    time_window: TimeWindow,
    lookback_sessions: int,
    lookahead_sessions: int,
    resource_policy: RuntimeResourcePolicy,
) -> OwnedValue[pd.DatetimeIndex]:
    require_date_window(time_window, context="a_share_trading_sessions")
    require_non_negative_int(
        lookback_sessions, error="invalid_a_share_session_offset"
    )
    require_non_negative_int(
        lookahead_sessions, error="invalid_a_share_session_offset"
    )
    calendar_root = layout.interim_data_root(source_id, "trading_calendar")
    owned_inspection = _inspect_dataset_owned(
        calendar_root,
        resource_policy=resource_policy,
    )
    try:
        calendar_row_count = int(owned_inspection.value.row_count)
        persisted_payload_bytes = int(
            owned_inspection.value.manifest.get("file_size_bytes", 0) or 0
        )
    finally:
        owned_inspection.release(owner="a_share.planning.coverage.inspection")
    maximum_session_state_bytes = calendar_row_count * np.dtype(np.int64).itemsize
    boolean_state_bytes = calendar_row_count * np.dtype(np.bool_).itemsize
    position_state_bytes = calendar_row_count * np.dtype(np.intp).itemsize
    maximum_selected_output_bytes = maximum_session_state_bytes
    compact_peak_bytes = (
        maximum_session_state_bytes
        + boolean_state_bytes
        + position_state_bytes
        + maximum_selected_output_bytes
    )
    if compact_peak_bytes > resource_policy.planning_headroom_bytes:
        raise materialization_io.MaterializationCapacityExceeded(
            "a_share_known_payloads_exceed_task_memory_budget"
        )
    session_lease = resource_policy.lease(
        "a_share.trading_sessions",
        max(1, maximum_session_state_bytes),
    )
    try:
        fixed_calendar_row_bytes = (
            np.dtype(np.int64).itemsize
            + np.dtype(np.bool_).itemsize
            + np.dtype("datetime64[ns]").itemsize
        )
        transform_row_bytes = (
            2 * np.dtype(np.bool_).itemsize + np.dtype(np.int64).itemsize
        )
        reader_headroom_bytes = resource_policy.planning_headroom_bytes
        if reader_headroom_bytes <= 0:
            raise materialization_io.MaterializationCapacityExceeded(
                "a_share_known_payloads_exceed_task_memory_budget"
            )
        initial_read_target_bytes = max(
            1,
            maximum_session_state_bytes
            + ceil(persisted_payload_bytes / max(1, calendar_row_count)),
        )
        read_target_bytes = min(
            reader_headroom_bytes,
            initial_read_target_bytes,
            max(
                1,
                reader_headroom_bytes
                * fixed_calendar_row_bytes
                // (fixed_calendar_row_bytes + transform_row_bytes),
            ),
        )
        with open_dataset_session(
            calendar_root,
            resource_policy=resource_policy,
            columns=("source_market_type", "is_trading_day", "timestamp"),
            time_window=None,
        ) as session:
            chunks = iter_dataset_chunks(
                session,
                target_chunk_bytes=read_target_bytes,
            )
            session_values = np.empty(calendar_row_count, dtype=np.int64)
            session_position = 0
            with closing_iterator(chunks):
                for calendar in chunks:
                    source_market_type = calendar["source_market_type"]
                    trading = calendar["is_trading_day"]
                    timestamps = calendar["timestamp"]
                    if source_market_type.dtype != np.dtype(np.int64):
                        raise ValueError("invalid_source_market_type")
                    if trading.dtype != np.dtype(np.bool_):
                        raise ValueError("invalid_is_trading_day")
                    if timestamps.dtype != np.dtype("datetime64[ns]"):
                        raise ValueError("invalid_trading_calendar_timestamp")
                    chunk_rows = len(calendar)
                    actual_transform_scratch_bytes = chunk_rows * transform_row_bytes
                    market_values = source_market_type.to_numpy(copy=False)
                    with resource_policy.lease(
                        "a_share.calendar.transform",
                        actual_transform_scratch_bytes,
                    ):
                        selected = np.zeros(chunk_rows, dtype=np.bool_)
                        for market_type in CURRENT_MARKET_TYPES:
                            np.logical_or(
                                selected,
                                market_values == market_type,
                                out=selected,
                            )
                        np.logical_and(
                            selected,
                            trading.to_numpy(copy=False),
                            out=selected,
                        )
                        selected_count = int(np.count_nonzero(selected))
                        if session_position + selected_count > calendar_row_count:
                            raise materialization_io.MaterializationCapacityExceeded(
                                "a_share_known_payloads_exceed_task_memory_budget"
                            )
                        selected_timestamps = timestamps.to_numpy(copy=False).view(
                            np.int64
                        )[selected]
                        day_ns = pd.Timedelta(days=1).value
                        np.floor_divide(
                            selected_timestamps,
                            day_ns,
                            out=selected_timestamps,
                        )
                        np.multiply(
                            selected_timestamps,
                            day_ns,
                            out=selected_timestamps,
                        )
                        session_values[
                            session_position : session_position + selected_count
                        ] = selected_timestamps
                        session_position += selected_count
                    del (
                        calendar,
                        source_market_type,
                        trading,
                        timestamps,
                        selected,
                        selected_timestamps,
                    )

        values = session_values[:session_position]
        values.sort(kind="heapsort")
        session_lease.grow(
            boolean_state_bytes
            + position_state_bytes
            + maximum_selected_output_bytes
        )
        keep = np.empty(len(values), dtype=np.bool_)
        if len(values):
            keep[0] = True
            keep[1:] = values[1:] != values[:-1]
        positions = np.flatnonzero(keep)
        unique_count = len(positions)
        start_ns = pd.Timestamp(time_window.start).value
        end_ns = pd.Timestamp(time_window.end).value
        raw_first = int(values.searchsorted(start_ns))
        raw_stop = int(values.searchsorted(end_ns, side="right"))
        first = max(0, int(positions.searchsorted(raw_first)) - lookback_sessions)
        stop = min(
            unique_count,
            int(positions.searchsorted(raw_stop)) + lookahead_sessions,
        )
        compact_values = values[positions[first:stop]]
        result = pd.DatetimeIndex(compact_values, copy=False)
        del session_values, values, keep, positions
        session_lease.shrink_to(result.nbytes)
    except BaseException as primary:
        try:
            session_lease.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "trading sessions lease release also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    return OwnedValue(result, session_lease)


def _rotate_left_int64_prefix(
    values: np.ndarray,
    *,
    length: int,
    shift: int,
) -> None:
    if shift == 0:
        return
    _reverse_int64_range(values, 0, shift)
    _reverse_int64_range(values, shift, length)
    _reverse_int64_range(values, 0, length)


def _reverse_int64_range(values: np.ndarray, start: int, stop: int) -> None:
    right = stop - 1
    while start < right:
        values[start], values[right] = values[right], values[start]
        start += 1
        right -= 1


def _target_session_bounds(
    sessions: Sequence[pd.Timestamp],
    time_window: TimeWindow,
) -> tuple[int, int]:
    require_date_window(time_window, context="a_share_target_sessions")
    first = bisect_left(sessions, pd.Timestamp(time_window.start))
    stop = bisect_right(sessions, pd.Timestamp(time_window.end))
    if first == stop:
        raise ValueError("no_trading_sessions_in_time_window")
    return first, stop


def require_target_sessions(
    sessions: pd.DatetimeIndex,
    time_window: TimeWindow,
) -> None:
    _target_session_bounds(sessions, time_window)


def registered_output_buffer_bytes(
    spec: ProcessedDataSpec,
    *,
    row_count: int,
) -> int:
    """Return registered output buffers plus publication key temporaries."""

    if type(row_count) is not int or row_count < 0:
        raise ValueError("invalid_a_share_output_row_count")
    fields = field_catalog()
    bytes_per_row = np.dtype(np.intp).itemsize
    for column in spec.required_columns:
        if (
            spec.kind in {ProcessedDataKind.FACTOR, ProcessedDataKind.LABEL}
            and column not in spec.primary_key
        ):
            dtype = "float64"
            nullable = False
        else:
            field = fields.get(column)
            dtype = field.dtype.lower()
            nullable = field.nullable
        if dtype in {"string", "date"}:
            bytes_per_row += np.dtype(object).itemsize
        elif dtype in {"bool", "boolean"}:
            bytes_per_row += np.dtype(np.bool_).itemsize
        elif dtype.startswith("timestamp"):
            bytes_per_row += np.dtype("datetime64[ns]").itemsize
        else:
            bytes_per_row += np.dtype(dtype).itemsize
        if nullable and not dtype.startswith("float"):
            bytes_per_row += np.dtype(np.bool_).itemsize
    bytes_per_row += (
        len(spec.primary_key) * np.dtype(object).itemsize
        + 2 * np.dtype(np.intp).itemsize
    )
    return row_count * bytes_per_row
