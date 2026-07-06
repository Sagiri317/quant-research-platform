"""Pure crypto market-panel calculations."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    canonical_crypto_asset_id,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    PANEL_COLUMNS,
)

DEFAULT_MAX_QUOTE_AGE = pd.Timedelta(seconds=120)


def compute_market_panel(
    bars: pd.DataFrame,
    *,
    trades: pd.DataFrame,
    depth: pd.DataFrame,
    max_quote_age: pd.Timedelta = DEFAULT_MAX_QUOTE_AGE,
) -> pd.DataFrame:
    panel = bars.copy()
    panel["market_id"] = str(MARKET_ID)
    panel["asset_id"] = panel["source_asset_id"].map(canonical_crypto_asset_id)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel["close_time"] = pd.to_datetime(panel["close_time"], errors="coerce")
    panel["taker_sell_volume"] = (
        _numeric(panel, "volume") - _numeric(panel, "taker_buy_volume")
    )
    panel = panel.merge(
        _signed_trade_volume(trades),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["trade_flow_observed"] = panel["trade_flow_observed"].fillna(False)
    zero_trade = _numeric(panel, "trade_count").eq(0) & panel["signed_volume"].isna()
    panel.loc[zero_trade, "signed_volume"] = 0.0
    panel.loc[zero_trade, "trade_flow_observed"] = True
    panel = panel.merge(
        _asof_quotes(panel, depth, max_quote_age=max_quote_age),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["quote_observed"] = panel["quote_observed"].fillna(False)
    stale_or_missing = ~panel["quote_observed"].astype(bool)
    quote_columns = (
        "best_bid",
        "best_ask",
        "bid_quantity",
        "ask_quantity",
        "bid_depth",
        "ask_depth",
        "mid_price",
        "quote_age_seconds",
    )
    panel.loc[stale_or_missing, list(quote_columns)] = pd.NA
    panel["eligible"] = (
        _numeric(panel, "open").gt(0.0)
        & _numeric(panel, "high").ge(_numeric(panel, "low"))
        & _numeric(panel, "close").gt(0.0)
    )
    return (
        _stabilize_panel_schema(panel.loc[:, PANEL_COLUMNS])
        .sort_values(["market_id", "asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )


def _signed_trade_volume(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["asset_id", "timestamp", "signed_volume", "trade_flow_observed"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    frame = trades.copy()
    frame["asset_id"] = frame["source_asset_id"].map(canonical_crypto_asset_id)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.floor(
        "min"
    )
    frame["signed_quantity"] = _numeric(frame, "signed_quantity")
    grouped = (
        frame.groupby(["asset_id", "timestamp"], as_index=False, sort=False)[
            "signed_quantity"
        ]
        .sum()
        .rename(columns={"signed_quantity": "signed_volume"})
    )
    grouped["trade_flow_observed"] = True
    return grouped.loc[:, columns]


def _asof_quotes(
    bars: pd.DataFrame,
    depth: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
) -> pd.DataFrame:
    columns = [
        "asset_id",
        "timestamp",
        "best_bid",
        "best_ask",
        "bid_quantity",
        "ask_quantity",
        "bid_depth",
        "ask_depth",
        "mid_price",
        "quote_observed",
        "quote_age_seconds",
    ]
    if bars.empty:
        return pd.DataFrame(columns=columns)
    if depth.empty:
        result = bars.loc[:, ["asset_id", "timestamp"]].copy()
        for column in columns:
            if column not in result:
                result[column] = pd.NA
        result["quote_observed"] = False
        return result.loc[:, columns]
    quotes = _quote_snapshots(depth)
    result_rows: list[dict[str, object]] = []
    ordered_bars = bars.sort_values(["asset_id", "timestamp"], kind="mergesort")
    for _, bar in ordered_bars.iterrows():
        candidates = quotes.loc[
            quotes["asset_id"].eq(bar["asset_id"])
            & quotes["quote_timestamp"].le(bar["close_time"])
        ]
        row = {"asset_id": bar["asset_id"], "timestamp": bar["timestamp"]}
        if candidates.empty:
            row["quote_observed"] = False
            result_rows.append(row)
            continue
        quote = candidates.iloc[-1]
        age = pd.Timestamp(bar["close_time"]) - pd.Timestamp(quote["quote_timestamp"])
        observed = pd.notna(age) and age <= max_quote_age
        row.update(quote.drop(labels=["quote_timestamp"]).to_dict())
        row["quote_age_seconds"] = age.total_seconds() if observed else pd.NA
        row["quote_observed"] = bool(observed)
        result_rows.append(row)
    return pd.DataFrame(result_rows, columns=columns)


def _quote_snapshots(depth: pd.DataFrame) -> pd.DataFrame:
    frame = depth.copy()
    frame["asset_id"] = frame["source_asset_id"].map(canonical_crypto_asset_id)
    frame["quote_timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["price"] = _numeric(frame, "price")
    frame["quantity"] = _numeric(frame, "quantity")
    best = frame.loc[frame["level"].eq(1)].pivot_table(
        index=["asset_id", "quote_timestamp"],
        columns="side",
        values=["price", "quantity"],
        aggfunc="first",
    )
    depth_totals = frame.pivot_table(
        index=["asset_id", "quote_timestamp"],
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
    quotes["mid_price"] = (quotes["best_bid"] + quotes["best_ask"]) / 2.0
    return quotes.reset_index().sort_values(
        ["asset_id", "quote_timestamp"],
        kind="mergesort",
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _stabilize_panel_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_id"] = result["market_id"].astype(str)
    result["asset_id"] = result["asset_id"].astype(str)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    for column in ("eligible", "trade_flow_observed", "quote_observed"):
        result[column] = result[column].astype(bool)
    for column in PANEL_COLUMNS:
        if column in {
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "trade_flow_observed",
            "quote_observed",
        }:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
    return result


__all__ = ["compute_market_panel"]
