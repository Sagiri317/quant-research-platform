"""Crypto market panel materialization."""

from __future__ import annotations

from collections.abc import Hashable, Iterator

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
)
from quant_research_platform.storage._dataset_reading import (
    build_dataset_equality_group_plan,
    build_dataset_read_context,
    dataset_context_for_equality_group,
    iter_dataset_chunks,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    write_dataset,
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
    input_identities = materialization_validation.input_dataset_identities(
        materialization_validation.panel_input_inspections(
            layout=layout,
            source_id=source_id,
            spec=spec,
        )
    )
    materialization_resources.require_materialization_disk_headroom(
        resource_policy,
        root.parent,
    )
    chunks = materialization_resources.iter_disk_guarded_chunks(
        _iter_panel_chunks(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            resource_policy=resource_policy,
        ),
        policy=resource_policy,
        scratch_parent=root.parent,
    )
    write_dataset(
        root,
        chunks,
        spec=spec,
        instance_metadata={"input_dataset_identities": input_identities},
    )
    inspection = inspect_dataset(root, time_window=time_window)
    materialization_validation.validate_panel_materialization(
        inspection,
        layout=layout,
        source_id=source_id,
        spec=spec,
        time_window=time_window,
        expected_input_identities=input_identities,
        resource_policy=resource_policy,
    )
    return inspection


def _iter_panel_chunks(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    resource_policy.require_valid_chunk_budget()
    budget = materialization_resources.MaterializationBudget.from_policy(
        resource_policy
    )
    chunk_target_bytes = budget.physical_batch_bytes
    bar_root = layout.interim_data_root(
        source_id,
        source_ids.market_bars_source_data_id(spec),
    )
    trade_root = layout.interim_data_root(
        source_id,
        source_ids.aggregate_trades_source_data_id(spec),
    )
    depth_root = layout.interim_data_root(
        source_id,
        source_ids.depth_levels_source_data_id(spec),
    )
    bar_context = build_dataset_read_context(
        bar_root,
        columns=None,
        time_window=time_window,
    )
    asset_index_budget = budget.dependency_key_index_bytes // 2
    group_plan = build_dataset_equality_group_plan(
        bar_context,
        columns=("source_asset_id",),
        target_chunk_bytes=chunk_target_bytes,
        target_state_bytes=asset_index_budget,
    )
    source_assets = _canonical_source_assets(
        group_plan.keys,
        target_state_bytes=budget.dependency_key_index_bytes - asset_index_budget,
    )
    for source_asset_id in source_assets:
        group_context = dataset_context_for_equality_group(
            bar_context,
            group_plan,
            (source_asset_id,),
        )
        for bars in iter_dataset_chunks(
            group_context,
            target_chunk_bytes=chunk_target_bytes,
        ):
            if bars.empty:
                continue
            _validate_matching_bar_keys(bars)
            prepared_bars = panels._prepare_panel_bars(bars)
            materialization_resources.require_frame_within_budget(
                prepared_bars,
                target_bytes=budget.allocations["core_window"],
                error="crypto_panel_core_window_exceeds_resource_budget",
            )
            trade_aggregates = _read_trade_aggregates(
                trade_root,
                prepared_bars,
                target_chunk_bytes=chunk_target_bytes,
                target_state_bytes=budget.reducer_state_bytes,
                target_output_bytes=budget.trade_output_bytes,
            )
            materialization_resources.require_frame_within_budget(
                trade_aggregates,
                target_bytes=budget.trade_output_bytes,
                error="crypto_trade_output_exceeds_resource_budget",
            )
            quote_snapshots = _read_quote_snapshots(
                depth_root,
                prepared_bars,
                target_chunk_bytes=chunk_target_bytes,
                target_state_bytes=budget.reducer_state_bytes,
                target_output_bytes=budget.depth_output_bytes,
            )
            materialization_resources.require_frame_within_budget(
                quote_snapshots,
                target_bytes=budget.depth_output_bytes,
                error="crypto_depth_output_exceeds_resource_budget",
            )
            output = panels._calculate_market_panel(
                prepared_bars,
                trade_aggregates=trade_aggregates,
                quote_snapshots=quote_snapshots,
                max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
            ).loc[:, spec.required_columns]
            materialization_resources.require_frame_within_budget(
                output,
                target_bytes=budget.materialized_output_bytes,
                error="crypto_panel_output_exceeds_resource_budget",
            )
            yield output


def _canonical_source_assets(
    keys: tuple[tuple[object, ...], ...],
    *,
    target_state_bytes: int,
) -> tuple[str, ...]:
    source_by_canonical: dict[str, str] = {}
    state_bytes = 0
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
        state_bytes += 256 + 2 * (len(canonical) + len(source_value))
        if state_bytes > target_state_bytes:
            raise ValueError("crypto_panel_asset_index_exceeds_resource_budget")
    return tuple(
        source_by_canonical[canonical]
        for canonical in sorted(source_by_canonical)
    )


def _read_trade_aggregates(
    dataset_root,
    bars: pd.DataFrame,
    *,
    target_chunk_bytes: int,
    target_state_bytes: int,
    target_output_bytes: int,
) -> pd.DataFrame:
    asset_ids, start, end, read_window = _matching_read_scope(
        bars,
        max_quote_age=pd.Timedelta(0),
    )
    reducer = panel_input_reducers.AggregateTradeReducer(
        bars=bars,
        target_state_bytes=target_state_bytes,
        target_output_bytes=target_output_bytes,
        normalize_chunk=panels.normalize_trade_chunk,
    )
    for chunk in _iter_matching_chunks(
        dataset_root,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=pd.Timedelta(0),
        source_asset_error="invalid_crypto_matching_trade_source_asset_id",
        timestamp_error="invalid_crypto_matching_trade_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    ):
        reducer.add_chunk(chunk)
    return reducer.finish()


def _read_quote_snapshots(
    dataset_root,
    bars: pd.DataFrame,
    *,
    target_chunk_bytes: int,
    target_state_bytes: int,
    target_output_bytes: int,
) -> pd.DataFrame:
    max_quote_age = panels.DEFAULT_MAX_QUOTE_AGE
    asset_ids, start, end, read_window = _matching_read_scope(
        bars,
        max_quote_age=max_quote_age,
    )
    reducer = panel_input_reducers.DepthSnapshotReducer(
        bars=bars,
        target_state_bytes=target_state_bytes,
        target_output_bytes=target_output_bytes,
        normalize_chunk=panels.normalize_depth_chunk,
        finalize_snapshot=panels.finalize_depth_snapshot,
    )
    for chunk in _iter_matching_chunks(
        dataset_root,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=max_quote_age,
        source_asset_error="invalid_crypto_matching_depth_source_asset_id",
        timestamp_error="invalid_crypto_matching_depth_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    ):
        reducer.add_chunk(chunk)
    return reducer.finish()


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
    start = _strict_timestamp_column(
        bars,
        "timestamp",
        "invalid_crypto_matching_bar_timestamp",
    ).min()
    end = _strict_timestamp_column(
        bars,
        "close_time",
        "invalid_crypto_matching_bar_close_time",
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
    _strict_timestamp_column(
        bars,
        "timestamp",
        "invalid_crypto_matching_bar_timestamp",
    )
    _strict_timestamp_column(
        bars,
        "close_time",
        "invalid_crypto_matching_bar_close_time",
    )


def _iter_matching_chunks(
    dataset_root,
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
    context = build_dataset_read_context(
        dataset_root,
        columns=None,
        time_window=time_window,
        row_filters=(
            (("source_asset_id", "==", next(iter(asset_ids))),)
            if len(asset_ids) == 1
            else ()
        ),
    )
    for chunk in iter_dataset_chunks(
        context,
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
    timestamps = _strict_timestamp_column(frame, "timestamp", timestamp_error)
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


def _strict_timestamp_column(
    frame: pd.DataFrame,
    column: str,
    error: str,
) -> pd.Series:
    return strict_timestamp_series(frame[column], error=error)


__all__ = ["write_panel"]
