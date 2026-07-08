"""Crypto market panel materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.processed import (
    panels,
    source_ids,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
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
    write_dataset(
        root,
        _iter_panel_chunks(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            resource_policy=resource_policy,
        ),
        spec=spec,
    )
    return inspect_dataset(root, time_window=time_window)


def _iter_panel_chunks(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    resource_policy.require_valid_chunk_budget()
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
    for bars_chunk in read_dataset(
        bar_root,
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
    ):
        if bars_chunk.empty:
            continue
        matching_trades = _read_matching_chunks(
            trade_root,
            bars_chunk,
            max_quote_age=pd.Timedelta(0),
        )
        matching_depth = _read_matching_chunks(
            depth_root,
            bars_chunk,
            max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
        )
        for bars in _split_frame_by_policy(bars_chunk, resource_policy):
            if bars.empty:
                continue
            trades = _filter_matching_frame(
                matching_trades,
                bars,
                max_quote_age=pd.Timedelta(0),
            )
            depth = _filter_matching_frame(
                matching_depth,
                bars,
                max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
            )
            yield panels.compute_market_panel(bars, trades=trades, depth=depth).loc[
                :,
                spec.required_columns,
            ]


def _read_matching_chunks(
    dataset_root,
    bars: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
) -> pd.DataFrame:
    asset_ids = set(bars["source_asset_id"].astype(str))
    start = pd.to_datetime(bars["timestamp"], errors="coerce").min()
    end = pd.to_datetime(bars["close_time"], errors="coerce").max()
    read_window = TimeWindow(
        (start - max_quote_age).to_pydatetime(),
        end.to_pydatetime(),
    )
    chunks = _iter_matching_chunks(
        dataset_root,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=max_quote_age,
    )
    try:
        first = next(chunks)
    except StopIteration:
        return pd.DataFrame()
    return pd.concat(_prepend(first, chunks), ignore_index=True)


def _iter_matching_chunks(
    dataset_root,
    *,
    asset_ids: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_window: TimeWindow,
    max_quote_age: pd.Timedelta,
) -> Iterator[pd.DataFrame]:
    for chunk in read_dataset(
        dataset_root,
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
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
            )
        ].copy()
        if not matching.empty:
            yield matching


def _filter_matching_frame(
    frame: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
) -> pd.DataFrame:
    if frame.empty or bars.empty:
        return pd.DataFrame(columns=frame.columns)
    asset_ids = set(bars["source_asset_id"].astype(str))
    start = pd.to_datetime(bars["timestamp"], errors="coerce").min()
    end = pd.to_datetime(bars["close_time"], errors="coerce").max()
    return frame.loc[
        _matching_window_mask(
            frame,
            asset_ids=asset_ids,
            start=start,
            end=end,
            max_quote_age=max_quote_age,
        )
    ].copy()


def _matching_window_mask(
    frame: pd.DataFrame,
    *,
    asset_ids: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_quote_age: pd.Timedelta,
) -> pd.Series:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    mask = frame["source_asset_id"].astype(str).isin(asset_ids)
    return mask & timestamps.ge(start - max_quote_age) & timestamps.le(end)


def _prepend(
    first: pd.DataFrame,
    remaining: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    yield first
    yield from remaining


def _split_frame_by_policy(
    frame: pd.DataFrame,
    policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    if frame.empty:
        return
    estimated_bytes = max(1, int(frame.memory_usage(deep=True).sum()))
    estimated_bytes_per_row = max(1, estimated_bytes // max(1, len(frame)))
    plan = ResourceChunkPlanner(policy).plan_sessions(
        axis=ChunkAxis.ROW_GROUP,
        session_count=len(frame),
        estimated_bytes_per_session=estimated_bytes_per_row,
    )
    step = max(1, plan.target_core_session_count)
    for start in range(0, len(frame), step):
        yield frame.iloc[start : start + step].copy()


__all__ = ["write_panel"]
