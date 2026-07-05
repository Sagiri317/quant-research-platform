"""Crypto market panel materialization."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import MARKET_ID
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    PANEL_COLUMNS,
)
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.storage.datasets import (
    DatasetInspection,
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
) -> DatasetInspection:
    if source_id != binance_ids.SOURCE_ID:
        raise ValueError(f"unsupported_crypto_source: {source_id}")
    bars = read_dataset(
        layout.interim_data_root(source_id, binance_ids.market_bars_data_id()),
        time_window=time_window,
    )
    trades = read_dataset(
        layout.interim_data_root(source_id, binance_ids.aggregate_trades_data_id()),
        time_window=time_window,
    )
    depth = read_dataset(
        layout.interim_data_root(source_id, binance_ids.depth_levels_data_id()),
        time_window=time_window,
    )
    panel = _build_panel(bars, trades, depth)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    write_dataset(root, panel.loc[:, spec.required_columns], spec=spec)
    return inspect_dataset(root, time_window=time_window)


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
    fallback_signed_volume = panel["taker_buy_volume"] - panel["taker_sell_volume"]
    panel["signed_volume"] = panel["signed_volume"].fillna(fallback_signed_volume)
    panel = panel.merge(
        _depth_quotes(depth),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["mid_price"] = (panel["best_bid"] + panel["best_ask"]) / 2.0
    panel["eligible"] = _numeric(panel, "close").gt(0.0)
    return (
        panel.loc[:, PANEL_COLUMNS]
        .sort_values(["market_id", "asset_id", "timestamp"])
        .reset_index(drop=True)
    )


def _signed_trade_volume(trades: pd.DataFrame) -> pd.DataFrame:
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


__all__ = ["write_panel"]
