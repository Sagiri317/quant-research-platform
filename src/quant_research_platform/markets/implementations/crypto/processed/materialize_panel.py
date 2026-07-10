"""Crypto market panel materialization."""

from __future__ import annotations

from collections.abc import Hashable, Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation,
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
    input_identities = materialization_validation.input_dataset_identities(
        materialization_validation.panel_input_inspections(
            layout=layout,
            source_id=source_id,
            spec=spec,
        )
    )
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
            source_asset_error="invalid_crypto_matching_trade_source_asset_id",
            timestamp_error="invalid_crypto_matching_trade_timestamp",
        )
        matching_depth = _read_matching_chunks(
            depth_root,
            bars_chunk,
            max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
            source_asset_error="invalid_crypto_matching_depth_source_asset_id",
            timestamp_error="invalid_crypto_matching_depth_timestamp",
        )
        for bars in _split_frame_by_policy(bars_chunk, resource_policy):
            if bars.empty:
                continue
            trades = _filter_matching_frame(
                matching_trades,
                bars,
                max_quote_age=pd.Timedelta(0),
                source_asset_error="invalid_crypto_matching_trade_source_asset_id",
                timestamp_error="invalid_crypto_matching_trade_timestamp",
            )
            depth = _filter_matching_frame(
                matching_depth,
                bars,
                max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
                source_asset_error="invalid_crypto_matching_depth_source_asset_id",
                timestamp_error="invalid_crypto_matching_depth_timestamp",
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
    source_asset_error: str,
    timestamp_error: str,
) -> pd.DataFrame:
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
        source_asset_error=source_asset_error,
        timestamp_error=timestamp_error,
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
    source_asset_error: str,
    timestamp_error: str,
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
                source_asset_error=source_asset_error,
                timestamp_error=timestamp_error,
            )
        ].copy()
        if not matching.empty:
            yield matching


def _filter_matching_frame(
    frame: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
    source_asset_error: str,
    timestamp_error: str,
) -> pd.DataFrame:
    if frame.empty or bars.empty:
        return pd.DataFrame(columns=frame.columns)
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
    return frame.loc[
        _matching_window_mask(
            frame,
            asset_ids=asset_ids,
            start=start,
            end=end,
            max_quote_age=max_quote_age,
            source_asset_error=source_asset_error,
            timestamp_error=timestamp_error,
        )
    ].copy()


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
