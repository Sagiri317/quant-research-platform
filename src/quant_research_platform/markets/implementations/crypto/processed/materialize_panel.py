"""Crypto market panel materialization."""

from __future__ import annotations

import sys
from collections.abc import Hashable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import replace

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.markets.implementations.crypto.ids import (
    canonical_crypto_asset_id,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation as validation,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    panel_input_reducers,
    panels,
    source_ids,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _OwnedValue,
    _release_owned_value,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    open_dataset_session,
    parquet_filters_for_time_window,
    prune_paths_for_time_window,
)
from quant_research_platform.storage.datasets.writing import (
    DatasetWriteReceipt,
    _write_dataset_owned,
)
from quant_research_platform.storage.identity import require_artifact_root
from quant_research_platform.storage.layout import StorageLayout


def write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    cleanup_stack: ExitStack | None = None,
) -> DatasetInspection:
    root = require_artifact_root(
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    )
    with runtime_resource_policy_scope(resource_policy), ExitStack() as session_stack:
        bar_data_id = source_ids.market_bars_source_data_id(spec)
        trade_data_id = source_ids.aggregate_trades_source_data_id(spec)
        depth_data_id = source_ids.depth_levels_source_data_id(spec)
        input_sessions: dict[str, DatasetSession] = {}
        for data_id, input_time_window in (
            (bar_data_id, time_window),
            (trade_data_id, None),
            (depth_data_id, None),
        ):
            input_sessions[data_id] = session_stack.enter_context(
                open_dataset_session(
                    layout.interim_data_root(source_id, data_id),
                    columns=None,
                    time_window=input_time_window,
                    resource_policy=resource_policy,
                )
            )
        input_identities = validation.input_dataset_identities(input_sessions)
        chunks = _iter_panel_chunks(
            spec,
            input_sessions=input_sessions,
            time_window=time_window,
            resource_policy=resource_policy,
        )

        def validate_staged_dataset(receipt: DatasetWriteReceipt) -> None:
            validation.validate_panel_materialization(
                receipt,
                bar_session=input_sessions[bar_data_id],
                spec=spec,
                expected_input_identities=input_identities,
                resource_policy=resource_policy,
            )
            for session in input_sessions.values():
                session.verify()

        owned = _write_dataset_owned(
            root,
            chunks,
            spec=spec,
            instance_metadata={"input_dataset_identities": input_identities},
            staged_validator=validate_staged_dataset,
        )
        result = owned.value

        def release_result() -> None:
            nonlocal result
            result = None
            _release_owned_value(
                owned,
                owner="crypto.panel.write.result",
            )

        if cleanup_stack is None:
            _release_owned_value(owned, owner="crypto.panel.write.result")
        else:
            cleanup_stack.callback(release_result)
        return result


def _iter_panel_chunks(
    spec: ProcessedDataSpec,
    *,
    input_sessions: Mapping[str, DatasetSession],
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[object]:
    if any(
        session.resource_policy is not resource_policy
        for session in input_sessions.values()
    ):
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        available_bytes = max(1, resource_policy.available_memory_bytes)
        chunk_target_bytes = max(1, available_bytes // 8)
        producer_bound = max(1, available_bytes // 2)
        bar_session = input_sessions[source_ids.market_bars_source_data_id(spec)]
        trade_session = input_sessions[source_ids.aggregate_trades_source_data_id(spec)]
        depth_session = input_sessions[source_ids.depth_levels_source_data_id(spec)]
        with validation.crypto_asset_session_views(
            bar_session,
            columns=("source_asset_id",),
            target_chunk_bytes=chunk_target_bytes,
            error="invalid_crypto_panel_source_physical_asset_grouping",
            key_error="invalid_crypto_matching_bar_source_asset_id",
        ) as source_asset_views:
            source_assets = _canonical_source_assets(
                tuple(key for key, _view in source_asset_views)
            )
            view_by_source_asset = {key[0]: view for key, view in source_asset_views}
            order_state = validation.OrderedDependencyState()
            lease: MemoryLease | None = None
            for source_asset_id in source_assets:
                trade_continuity_state = (
                    panel_input_reducers.AggregateTradeContinuityState()
                )
                asset_session = view_by_source_asset[source_asset_id]
                with closing_iterator(
                    iter_dataset_chunks(
                        asset_session,
                        target_chunk_bytes=chunk_target_bytes,
                    )
                ) as bar_chunks:
                    while True:
                        if lease is None:
                            lease = yield producer_bound
                        while True:
                            try:
                                bars = next(bar_chunks)
                            except StopIteration:
                                bars = None
                                break
                            if not bars.empty:
                                break
                            del bars
                        if bars is None:
                            break
                        _validate_matching_bar_keys(bars)
                        prepared_bars = panels.prepare_panel_bars(bars)
                        del bars
                        prepared_bars = validation.validated_ordered_dependency_chunk(
                            prepared_bars,
                            state=order_state,
                            error="out_of_order_crypto_panel_dependency",
                        )
                        prepared_bar_bytes = pandas_frame_payload_bytes(prepared_bars)
                        if prepared_bar_bytes + 256 * 1024 > lease.reserved_bytes:
                            raise ValueError("crypto_panel_bound_invalid")
                        with (
                            closing_iterator(
                                _iter_matching_trade_chunks(
                                    trade_session,
                                    prepared_bars,
                                    target_chunk_bytes=chunk_target_bytes,
                                )
                            ) as trade_chunks,
                            closing_iterator(
                                _iter_matching_depth_chunks(
                                    depth_session,
                                    prepared_bars,
                                    target_chunk_bytes=chunk_target_bytes,
                                )
                            ) as depth_chunks,
                            ExitStack() as output_stack,
                        ):
                            output = panels.calculate_market_panel_from_chunks(
                                prepared_bars,
                                trade_chunks=trade_chunks,
                                depth_chunks=depth_chunks,
                                payload_target_bytes=resource_policy.available_memory_bytes,
                                trade_continuity_state=trade_continuity_state,
                                max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
                                _output_stack=output_stack,
                            )
                            output_bytes = pandas_frame_payload_bytes(output)
                            actual_bound = (
                                output_bytes + prepared_bar_bytes + 256 * 1024
                            )
                            if actual_bound > lease.reserved_bytes:
                                raise ValueError("crypto_panel_bound_invalid")
                            lease.shrink_to(actual_bound)
                            del prepared_bars
                            yield _OwnedValue(output, lease)
                            del output
                            lease = None
        for session in input_sessions.values():
            session.verify()


def _canonical_source_assets(
    keys: tuple[tuple[object, ...], ...],
) -> tuple[str, ...]:
    source_by_canonical: dict[str, str] = {}
    for key in keys:
        if len(key) != 1 or not isinstance(key[0], str) or not key[0]:
            raise ValueError("invalid_crypto_matching_bar_source_asset_id")
        source_value = key[0]
        canonical = canonical_crypto_asset_id(source_value)
        existing = source_by_canonical.get(canonical)
        if existing is not None:
            if existing != source_value:
                raise ValueError("duplicate_crypto_canonical_asset_id")
            continue
        source_by_canonical[canonical] = source_value
    return tuple(
        source_by_canonical[canonical] for canonical in sorted(source_by_canonical)
    )


def _iter_matching_trade_chunks(
    session: DatasetSession,
    bars: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    asset_ids, read_window = _matching_read_scope(
        bars,
        max_quote_age=pd.Timedelta(0),
    )
    yield from _iter_matching_chunks(
        session,
        asset_ids=asset_ids,
        time_window=read_window,
        source_asset_error="invalid_crypto_matching_trade_source_asset_id",
        timestamp_error="invalid_crypto_matching_trade_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    )


def _iter_matching_depth_chunks(
    session: DatasetSession,
    bars: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    max_quote_age = panels.DEFAULT_MAX_QUOTE_AGE
    asset_ids, read_window = _matching_read_scope(
        bars,
        max_quote_age=max_quote_age,
    )
    yield from _iter_matching_chunks(
        session,
        asset_ids=asset_ids,
        time_window=read_window,
        source_asset_error="invalid_crypto_matching_depth_source_asset_id",
        timestamp_error="invalid_crypto_matching_depth_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    )


def _matching_read_scope(
    bars: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
) -> tuple[set[str], TimeWindow]:
    asset_ids = set(
        _strict_source_asset_ids(
            bars,
            error="invalid_crypto_matching_bar_source_asset_id",
        )
    )
    start = strict_timestamp_series(
        bars["timestamp"],
        error="invalid_crypto_matching_bar_timestamp",
    ).min()
    end = strict_timestamp_series(
        bars["close_time"],
        error="invalid_crypto_matching_bar_close_time",
    ).max()
    if len(asset_ids) != 1:
        raise ValueError("invalid_crypto_matching_bar_source_asset_id")
    return (
        asset_ids,
        TimeWindow(
            (start - max_quote_age).to_pydatetime(),
            end.to_pydatetime(),
        ),
    )


def _validate_matching_bar_keys(bars: pd.DataFrame) -> None:
    _strict_source_asset_ids(
        bars,
        error="invalid_crypto_matching_bar_source_asset_id",
    )
    strict_timestamp_series(
        bars["timestamp"],
        error="invalid_crypto_matching_bar_timestamp",
    )
    strict_timestamp_series(
        bars["close_time"],
        error="invalid_crypto_matching_bar_close_time",
    )


def _iter_matching_chunks(
    session: DatasetSession,
    *,
    asset_ids: set[str],
    time_window: TimeWindow,
    source_asset_error: str,
    timestamp_error: str,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    view = _matching_session_view(
        session,
        time_window=time_window,
        asset_ids=asset_ids,
    )
    try:
        with closing_iterator(
            iter_dataset_chunks(
                view,
                target_chunk_bytes=target_chunk_bytes,
            )
        ) as chunks:
            for chunk in chunks:
                if chunk.empty:
                    continue
                if "timestamp" not in chunk or not pd.api.types.is_datetime64_any_dtype(
                    chunk["timestamp"].dtype
                ):
                    raise ValueError(timestamp_error)
                if any(pd.isna(value) for value in chunk["timestamp"]):
                    raise ValueError(timestamp_error)
                _strict_source_asset_ids(chunk, error=source_asset_error)
                yield chunk
    finally:
        if view.path_selection is not None:
            primary = sys.exception()
            try:
                session.inventory.release_path_selection(view.path_selection)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "crypto matching path-selection cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )


def _matching_session_view(
    session: DatasetSession,
    *,
    time_window: TimeWindow,
    asset_ids: set[str],
) -> DatasetSession:
    path_selection = session.inventory.new_path_selection()
    try:
        for path in session.all_paths:
            if prune_paths_for_time_window(
                session.data_root,
                (path,),
                spec=session.spec,
                time_window=time_window,
            ):
                session.inventory.select_path(path_selection, path)
        filters = tuple(
            parquet_filters_for_time_window(
                timestamp_column=session.timestamp_column,
                time_window=time_window,
            )
            or ()
        )
        filters += (("source_asset_id", "==", next(iter(asset_ids))),)
        view = replace(
            session,
            path_selection=path_selection,
            time_window=time_window,
            parquet_filters=filters or None,
            owns_inventory=False,
        )
        view.verify()
    except BaseException as primary:
        try:
            session.inventory.release_path_selection(path_selection)
        except BaseException as cleanup_error:
            primary.add_note(
                "crypto matching path-selection cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    return view


def _strict_source_asset_ids(frame: pd.DataFrame, *, error: str) -> pd.Series:
    if "source_asset_id" not in frame.columns:
        raise ValueError(error)
    values = frame["source_asset_id"]
    for value in values:
        if (
            value is None
            or isinstance(value, (list, dict, set))
            or not isinstance(value, Hashable)
            or not isinstance(value, str)
            or value == ""
        ):
            raise ValueError(error)
    return values
