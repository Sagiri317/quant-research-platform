"""Pure crypto market-panel calculations."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype

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
    if not isinstance(max_quote_age, pd.Timedelta) or max_quote_age < pd.Timedelta(0):
        raise ValueError("invalid_crypto_max_quote_age")
    panel = bars.copy()
    panel["market_id"] = str(MARKET_ID)
    panel["asset_id"] = panel["source_asset_id"].map(canonical_crypto_asset_id)
    if panel["asset_id"].isna().any():
        raise ValueError("invalid_crypto_source_asset_id")
    panel["timestamp"] = _timestamp_series(
        panel,
        "timestamp",
        error="invalid_crypto_panel_timestamp",
    )
    panel["close_time"] = _timestamp_series(
        panel,
        "close_time",
        error="invalid_crypto_panel_close_time",
    )
    volume = _numeric(panel, "volume")
    quote_volume = _numeric(panel, "quote_volume")
    trade_count = _numeric(panel, "trade_count")
    taker_buy_volume = _numeric(panel, "taker_buy_volume")
    taker_buy_quote_volume = _numeric(panel, "taker_buy_quote_volume")
    valid_trade_count = trade_count.ge(0.0) & trade_count.mod(1).eq(0.0)
    valid_bar_quantity = (
        volume.ge(0.0)
        & quote_volume.ge(0.0)
        & valid_trade_count
        & taker_buy_volume.ge(0.0)
        & taker_buy_quote_volume.ge(0.0)
        & taker_buy_volume.le(volume)
        & taker_buy_quote_volume.le(quote_volume)
    )
    panel["taker_sell_volume"] = (volume - taker_buy_volume).where(valid_bar_quantity)
    panel = panel.merge(
        _signed_trade_volume(trades),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["trade_flow_observed"] = panel["trade_flow_observed"].fillna(False)
    zero_trade = valid_bar_quantity & trade_count.eq(0) & panel["signed_volume"].isna()
    panel.loc[zero_trade, "signed_volume"] = 0.0
    panel.loc[zero_trade, "trade_flow_observed"] = True
    panel = panel.merge(
        _asof_quotes(panel, depth, max_quote_age=max_quote_age),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["quote_observed"] = panel["quote_observed"].fillna(False)
    stale_or_missing = ~_strict_bool_series(
        panel["quote_observed"],
        error="invalid_crypto_quote_observed",
    )
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
        & _numeric(panel, "high").gt(0.0)
        & _numeric(panel, "low").gt(0.0)
        & _numeric(panel, "high").ge(_numeric(panel, "low"))
        & _numeric(panel, "close").gt(0.0)
        & _numeric(panel, "high").ge(
            pd.concat(
                (_numeric(panel, "open"), _numeric(panel, "close")),
                axis=1,
            ).max(axis=1)
        )
        & _numeric(panel, "low").le(
            pd.concat(
                (_numeric(panel, "open"), _numeric(panel, "close")),
                axis=1,
            ).min(axis=1)
        )
        & valid_bar_quantity
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
    frame = frame.loc[frame["asset_id"].notna()].copy()
    frame["timestamp"] = _timestamp_series(
        frame,
        "timestamp",
        error="invalid_crypto_trade_timestamp",
    ).dt.floor(
        "min"
    )
    frame["signed_quantity"] = pd.to_numeric(frame["signed_quantity"], errors="coerce")
    frame = frame.loc[frame["signed_quantity"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
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
        observed = pd.notna(age) and age <= max_quote_age and bool(quote["quote_valid"])
        row.update(quote.drop(labels=["quote_timestamp", "quote_valid"]).to_dict())
        row["quote_age_seconds"] = age.total_seconds() if observed else pd.NA
        row["quote_observed"] = bool(observed)
        result_rows.append(row)
    return pd.DataFrame(result_rows, columns=columns)


def _quote_snapshots(depth: pd.DataFrame) -> pd.DataFrame:
    frame = depth.copy()
    frame["asset_id"] = frame["source_asset_id"].map(canonical_crypto_asset_id)
    if frame["asset_id"].isna().any():
        raise ValueError("invalid_crypto_source_asset_id")
    _validate_depth_contract(frame)
    frame["quote_timestamp"] = _timestamp_series(
        frame,
        "timestamp",
        error="invalid_crypto_depth_timestamp",
    )
    frame["price"] = _numeric_not_bool(
        frame,
        "price",
        error="invalid_crypto_depth_price",
    )
    frame["quantity"] = _numeric_not_bool(
        frame,
        "quantity",
        error="invalid_crypto_depth_quantity",
    )
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
    quotes["best_bid"] = _pivot_value(best, ("price", "bid"))
    quotes["best_ask"] = _pivot_value(best, ("price", "ask"))
    quotes["bid_quantity"] = _pivot_value(best, ("quantity", "bid"))
    quotes["ask_quantity"] = _pivot_value(best, ("quantity", "ask"))
    quotes["bid_depth"] = _pivot_value(depth_totals, "bid")
    quotes["ask_depth"] = _pivot_value(depth_totals, "ask")
    quotes["mid_price"] = (quotes["best_bid"] + quotes["best_ask"]) / 2.0
    required = ("best_bid", "best_ask", "bid_quantity", "ask_quantity")
    quotes["quote_valid"] = (
        quotes.loc[:, required].notna().all(axis=1)
        & quotes["best_bid"].gt(0.0)
        & quotes["best_ask"].gt(0.0)
        & quotes["bid_quantity"].gt(0.0)
        & quotes["ask_quantity"].gt(0.0)
        & quotes["best_bid"].lt(quotes["best_ask"])
    )
    return quotes.reset_index().sort_values(
        ["asset_id", "quote_timestamp"],
        kind="mergesort",
    )


def _pivot_value(frame: pd.DataFrame, key: object) -> pd.Series:
    if key in frame:
        return pd.to_numeric(frame[key], errors="coerce")
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _required_numeric(frame: pd.DataFrame, column: str, *, error: str) -> pd.Series:
    values = _numeric_not_bool(frame, column, error=error)
    if values.isna().any():
        raise ValueError(error)
    return values


def _numeric_not_bool(frame: pd.DataFrame, column: str, *, error: str) -> pd.Series:
    if frame[column].map(lambda value: isinstance(value, bool)).any():
        raise ValueError(error)
    return pd.to_numeric(frame[column], errors="coerce")


def _timestamp_series(frame: pd.DataFrame, column: str, *, error: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(error)
    return values


def _validate_depth_contract(frame: pd.DataFrame) -> None:
    if not set(frame["side"]).issubset({"bid", "ask"}):
        raise ValueError("invalid_crypto_depth_side")
    levels = pd.to_numeric(frame["level"], errors="coerce")
    if (
        levels.isna().any()
        or levels.le(0).any()
        or levels.mod(1).ne(0).any()
        or frame["level"].map(lambda value: isinstance(value, bool)).any()
    ):
        raise ValueError("invalid_crypto_depth_level")
    if frame.duplicated(["asset_id", "timestamp", "side", "level"]).any():
        raise ValueError("duplicate_crypto_depth_level")
    prices = _numeric_not_bool(frame, "price", error="invalid_crypto_depth_price")
    quantities = _numeric_not_bool(
        frame,
        "quantity",
        error="invalid_crypto_depth_quantity",
    )
    if prices.le(0.0).any():
        raise ValueError("invalid_crypto_depth_price")
    if quantities.lt(0.0).any():
        raise ValueError("invalid_crypto_depth_quantity")


def _stabilize_panel_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_id"] = result["market_id"].astype(str)
    result["asset_id"] = result["asset_id"].astype(str)
    result["timestamp"] = _timestamp_series(
        result,
        "timestamp",
        error="invalid_crypto_panel_timestamp",
    )
    for column in ("eligible", "trade_flow_observed", "quote_observed"):
        result[column] = _strict_bool_series(
            result[column],
            error=f"invalid_crypto_{column}",
        )
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


def _strict_bool_series(series: pd.Series, *, error: str) -> pd.Series:
    if is_bool_dtype(series):
        return series.astype("bool")
    if not series.map(lambda value: isinstance(value, bool)).all():
        raise ValueError(error)
    return pd.Series(series.to_numpy(dtype=bool), index=series.index)


__all__ = ["compute_market_panel"]
