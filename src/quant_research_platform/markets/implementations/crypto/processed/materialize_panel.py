"""Crypto market panel materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import MARKET_ID
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    PANEL_COLUMNS,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.sources.binance import ids as binance_ids
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
    if source_id != binance_ids.SOURCE_ID:
        raise ValueError(f"unsupported_crypto_source: {source_id}")
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
    bar_root = layout.interim_data_root(source_id, binance_ids.market_bars_data_id())
    trade_root = layout.interim_data_root(
        source_id,
        binance_ids.aggregate_trades_data_id(),
    )
    depth_root = layout.interim_data_root(source_id, binance_ids.depth_levels_data_id())
    for bars_chunk in read_dataset(
        bar_root,
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
    ):
        for bars in _split_frame_by_policy(bars_chunk, resource_policy):
            if bars.empty:
                continue
            trades = _read_matching_chunks(trade_root, bars, time_window=time_window)
            depth = _read_matching_chunks(depth_root, bars, time_window=time_window)
            yield _build_panel(bars, trades, depth).loc[:, spec.required_columns]


def _read_matching_chunks(
    dataset_root,
    bars: pd.DataFrame,
    *,
    time_window: TimeWindow,
) -> pd.DataFrame:
    asset_ids = set(bars["source_asset_id"].astype(str))
    start = pd.to_datetime(bars["timestamp"], errors="coerce").min()
    end = pd.to_datetime(bars["timestamp"], errors="coerce").max()
    chunks = _iter_matching_chunks(
        dataset_root,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=time_window,
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
) -> Iterator[pd.DataFrame]:
    for chunk in read_dataset(
        dataset_root,
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
    ):
        if chunk.empty:
            continue
        timestamps = pd.to_datetime(chunk["timestamp"], errors="coerce")
        mask = chunk["source_asset_id"].astype(str).isin(asset_ids)
        mask &= timestamps.ge(start) & timestamps.le(end + pd.Timedelta(minutes=1))
        matching = chunk.loc[mask].copy()
        if not matching.empty:
            yield matching


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


def _build_panel(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    depth: pd.DataFrame,
) -> pd.DataFrame:
    panel = bars.copy()
    panel["market_id"] = str(MARKET_ID)
    panel["asset_id"] = panel["source_asset_id"].astype(str)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel["taker_sell_volume"] = (
        _numeric(panel, "volume") - _numeric(panel, "taker_buy_volume")
    )
    panel = panel.merge(
        _signed_trade_volume(trades),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["signed_volume"] = panel["signed_volume"].fillna(0.0)
    panel = panel.merge(
        _depth_quotes(depth),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["mid_price"] = (panel["best_bid"] + panel["best_ask"]) / 2.0
    panel["eligible"] = _numeric(panel, "close").gt(0.0)
    return (
        _stabilize_panel_schema(panel.loc[:, PANEL_COLUMNS])
        .sort_values(["market_id", "asset_id", "timestamp"])
        .reset_index(drop=True)
    )


def _signed_trade_volume(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["asset_id", "timestamp", "signed_volume"])
    frame = trades.copy()
    frame["asset_id"] = frame["source_asset_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.floor(
        "min"
    )
    frame["signed_quantity"] = _numeric(frame, "signed_quantity")
    return (
        frame.groupby(["asset_id", "timestamp"], as_index=False, sort=False)[
            "signed_quantity"
        ]
        .sum()
        .rename(columns={"signed_quantity": "signed_volume"})
    )


def _depth_quotes(depth: pd.DataFrame) -> pd.DataFrame:
    if depth.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "timestamp",
                "best_bid",
                "best_ask",
                "bid_quantity",
                "ask_quantity",
                "bid_depth",
                "ask_depth",
            ]
        )
    frame = depth.copy()
    frame["asset_id"] = frame["source_asset_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["minute"] = frame["timestamp"].dt.floor("min")
    latest = frame.groupby(["asset_id", "minute"], sort=False)["timestamp"].transform(
        "max"
    )
    frame = frame.loc[frame["timestamp"].eq(latest)].copy()
    frame["price"] = _numeric(frame, "price")
    frame["quantity"] = _numeric(frame, "quantity")
    best = frame.loc[frame["level"].eq(1)].pivot_table(
        index=["asset_id", "minute"],
        columns="side",
        values=["price", "quantity"],
        aggfunc="first",
    )
    depth_totals = frame.pivot_table(
        index=["asset_id", "minute"],
        columns="side",
        values="quantity",
        aggfunc="sum",
    )
    quotes = pd.DataFrame(index=best.index)
    quotes["best_bid"] = best.get(("price", "bid"))
    quotes["best_ask"] = best.get(("price", "ask"))
    quotes["bid_quantity"] = best.get(("quantity", "bid"))
    quotes["ask_quantity"] = best.get(("quantity", "ask"))
    quotes["bid_depth"] = depth_totals.get("bid")
    quotes["ask_depth"] = depth_totals.get("ask")
    return quotes.reset_index().rename(columns={"minute": "timestamp"})


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _stabilize_panel_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_id"] = result["market_id"].astype(str)
    result["asset_id"] = result["asset_id"].astype(str)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["eligible"] = result["eligible"].astype(bool)
    for column in PANEL_COLUMNS:
        if column in {"market_id", "asset_id", "timestamp", "eligible"}:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    return result


__all__ = ["write_panel"]
