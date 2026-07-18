"""Crypto market panel materialization."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from contextlib import ExitStack, closing
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
    materialization_resources,
    materialization_validation,
    panel_input_reducers,
    panels,
    source_ids,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    iter_dataset_chunks,
    open_dataset_session,
    parquet_filters_for_time_window,
    prune_paths_for_time_window,
)
from quant_research_platform.storage._dataset_writing import (
    DatasetWriteReceipt,
    write_dataset,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.layout import StorageLayout


def write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with runtime_resource_policy_scope(resource_policy), ExitStack() as session_stack:
        bar_data_id = source_ids.market_bars_source_data_id(spec)
        trade_data_id = source_ids.aggregate_trades_source_data_id(spec)
        depth_data_id = source_ids.depth_levels_source_data_id(spec)
        input_sessions = {
            bar_data_id: session_stack.enter_context(
                open_dataset_session(
                    layout.interim_data_root(source_id, bar_data_id),
                    columns=None,
                    time_window=time_window,
                    resource_policy=resource_policy,
                ),
            ),
            trade_data_id: session_stack.enter_context(
                open_dataset_session(
                    layout.interim_data_root(source_id, trade_data_id),
                    columns=None,
                    time_window=None,
                    resource_policy=resource_policy,
                ),
            ),
            depth_data_id: session_stack.enter_context(
                open_dataset_session(
                    layout.interim_data_root(source_id, depth_data_id),
                    columns=None,
                    time_window=None,
                    resource_policy=resource_policy,
                ),
            ),
        }
        input_identities = materialization_validation.input_dataset_identities(
            input_sessions
        )
        chunks = materialization_resources.iter_disk_guarded_chunks(
            _iter_panel_chunks(
                spec,
                input_sessions=input_sessions,
                time_window=time_window,
                resource_policy=resource_policy,
            ),
            scratch_parent=root.parent,
        )

        def validate_staged_dataset(receipt: DatasetWriteReceipt) -> None:
            materialization_validation.validate_panel_materialization(
                receipt,
                bar_session=input_sessions[bar_data_id],
                spec=spec,
                expected_input_identities=input_identities,
                resource_policy=resource_policy,
            )
            for session in input_sessions.values():
                session.verify()

        return write_dataset(
            root,
            chunks,
            spec=spec,
            instance_metadata={"input_dataset_identities": input_identities},
            staged_validator=validate_staged_dataset,
            resource_policy=resource_policy,
        )


def _iter_panel_chunks(
    spec: ProcessedDataSpec,
    *,
    input_sessions: Mapping[str, DatasetSession],
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    chunk_target_bytes = resource_policy.payload_target_bytes
    bar_session = input_sessions[source_ids.market_bars_source_data_id(spec)]
    trade_session = input_sessions[source_ids.aggregate_trades_source_data_id(spec)]
    depth_session = input_sessions[source_ids.depth_levels_source_data_id(spec)]
    with materialization_validation.crypto_asset_session_views(
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
        order_state = materialization_validation.OrderedDependencyState()
        for source_asset_id in source_assets:
            trade_continuity_state = (
                panel_input_reducers.AggregateTradeContinuityState()
            )
            asset_session = view_by_source_asset[source_asset_id]
            for bars in iter_dataset_chunks(
                asset_session,
                target_chunk_bytes=chunk_target_bytes,
            ):
                if bars.empty:
                    continue
                _validate_matching_bar_keys(bars)
                prepared_bars = panels.prepare_panel_bars(bars)
                prepared_bars = (
                    materialization_validation.validated_ordered_dependency_chunk(
                        prepared_bars,
                        state=order_state,
                        error="out_of_order_crypto_panel_dependency",
                    )
                )
                materialization_resources.require_frame_within_budget(
                    prepared_bars,
                    target_bytes=resource_policy.payload_target_bytes,
                    error="crypto_indivisible_reducer_unit_exceeds_payload_target",
                )
                with (
                    closing(
                        _iter_matching_trade_chunks(
                            trade_session,
                            prepared_bars,
                            target_chunk_bytes=chunk_target_bytes,
                        )
                    ) as trade_chunks,
                    closing(
                        _iter_matching_depth_chunks(
                            depth_session,
                            prepared_bars,
                            target_chunk_bytes=chunk_target_bytes,
                        )
                    ) as depth_chunks,
                ):
                    output = panels.calculate_market_panel_from_chunks(
                        prepared_bars,
                        trade_chunks=trade_chunks,
                        depth_chunks=depth_chunks,
                        payload_target_bytes=resource_policy.payload_target_bytes,
                        trade_continuity_state=trade_continuity_state,
                        max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
                    ).loc[:, spec.required_columns]
                materialization_resources.require_frame_within_budget(
                    output,
                    target_bytes=resource_policy.payload_target_bytes,
                    error="crypto_indivisible_reducer_unit_exceeds_payload_target",
                )
                yield output
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
    asset_ids, start, end, read_window = _matching_read_scope(
        bars,
        max_quote_age=pd.Timedelta(0),
    )
    yield from _iter_matching_chunks(
        session,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=pd.Timedelta(0),
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
    asset_ids, start, end, read_window = _matching_read_scope(
        bars,
        max_quote_age=max_quote_age,
    )
    yield from _iter_matching_chunks(
        session,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=max_quote_age,
        source_asset_error="invalid_crypto_matching_depth_source_asset_id",
        timestamp_error="invalid_crypto_matching_depth_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    )


def _matching_read_scope(
    bars: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
) -> tuple[set[str], pd.Timestamp, pd.Timestamp, TimeWindow]:
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
    return (
        asset_ids,
        start,
        end,
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
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_window: TimeWindow,
    max_quote_age: pd.Timedelta,
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
        for chunk in iter_dataset_chunks(
            view,
            target_chunk_bytes=target_chunk_bytes,
        ):
            if chunk.empty:
                continue
            matching = chunk.loc[
                _matching_window_mask(
                    chunk,
                    asset_ids=asset_ids,
                    start=start,
                    end=end,
                    max_quote_age=max_quote_age,
                    source_asset_error=source_asset_error,
                    timestamp_error=timestamp_error,
                )
            ].copy()
            if not matching.empty:
                yield matching
    finally:
        if view.path_selection is not None:
            session.inventory.release_path_selection(view.path_selection)


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
        if len(asset_ids) == 1:
            filters += (("source_asset_id", "==", next(iter(asset_ids))),)
        view = replace(
            session,
            path_selection=path_selection,
            time_window=time_window,
            parquet_filters=filters or None,
            owns_inventory=False,
        )
        view.verify()
    except BaseException:
        session.inventory.release_path_selection(path_selection)
        raise
    return view


def _matching_window_mask(
    frame: pd.DataFrame,
    *,
    asset_ids: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_quote_age: pd.Timedelta,
    source_asset_error: str,
    timestamp_error: str,
) -> pd.Series:
    timestamps = strict_timestamp_series(frame["timestamp"], error=timestamp_error)
    source_ids = _strict_source_asset_ids(frame, error=source_asset_error)
    mask = source_ids.isin(asset_ids)
    return mask & timestamps.ge(start - max_quote_age) & timestamps.le(end)


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


__all__ = ["write_panel"]
