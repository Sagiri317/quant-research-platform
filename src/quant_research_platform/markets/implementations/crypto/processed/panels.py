"""Pure crypto market-panel calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd
from pandas.api.types import is_bool_dtype

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    canonical_crypto_asset_id,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    panel_input_reducers,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    PANEL_COLUMNS,
)

DEFAULT_MAX_QUOTE_AGE = pd.Timedelta(seconds=120)
_TRADE_VOLUME_RELATIVE_TOLERANCE = 1e-9
_TRADE_VOLUME_ABSOLUTE_TOLERANCE = 1e-12
_SIGNED_QUANTITY_RELATIVE_TOLERANCE = 1e-12
_SIGNED_QUANTITY_ABSOLUTE_TOLERANCE = 1e-12


def prepare_panel_bars(bars: pd.DataFrame) -> pd.DataFrame:
    panel = bars.copy().reset_index(drop=True)
    panel["timestamp"] = _strict_timestamp_column(
        panel,
        "timestamp",
        error="invalid_crypto_panel_timestamp",
    )
    panel["open_time"] = _strict_timestamp_column(
        panel,
        "open_time",
        error="invalid_crypto_bar_open_time",
    )
    panel["close_time"] = _strict_timestamp_column(
        panel,
        "close_time",
        error="invalid_crypto_panel_close_time",
    )
    _validate_one_minute_bar_contract(panel)
    panel["market_id"] = str(MARKET_ID)
    panel["asset_id"] = _crypto_asset_id_series(panel["source_asset_id"])
    return panel


def calculate_market_panel_from_chunks(
    prepared_bars: pd.DataFrame,
    *,
    trade_chunks: Iterable[pd.DataFrame],
    depth_chunks: Iterable[pd.DataFrame],
    trade_state_bytes: int,
    trade_output_bytes: int,
    depth_state_bytes: int,
    depth_output_bytes: int,
    max_quote_age: pd.Timedelta = DEFAULT_MAX_QUOTE_AGE,
) -> pd.DataFrame:
    if not isinstance(max_quote_age, pd.Timedelta) or max_quote_age < pd.Timedelta(0):
        raise ValueError("invalid_crypto_max_quote_age")
    trade_reducer = panel_input_reducers.AggregateTradeReducer(
        bars=prepared_bars,
        target_state_bytes=trade_state_bytes,
        target_output_bytes=trade_output_bytes,
        normalize_chunk=normalize_trade_chunk,
    )
    for chunk in trade_chunks:
        if not chunk.empty:
            trade_reducer.add_chunk(chunk)
    trade_aggregates = trade_reducer.finish()
    depth_reducer = panel_input_reducers.DepthSnapshotReducer(
        bars=prepared_bars,
        target_state_bytes=depth_state_bytes,
        target_output_bytes=depth_output_bytes,
        normalize_chunk=_normalize_depth_chunk_for_reducer,
        finalize_snapshot=finalize_depth_snapshot,
    )
    for chunk in depth_chunks:
        if not chunk.empty:
            depth_reducer.add_chunk(chunk)
    quote_snapshots = depth_reducer.finish()
    return _calculate_market_panel(
        prepared_bars,
        trade_aggregates=trade_aggregates,
        quote_snapshots=quote_snapshots,
        max_quote_age=max_quote_age,
    )


def _calculate_market_panel(
    panel: pd.DataFrame,
    *,
    trade_aggregates: pd.DataFrame,
    quote_snapshots: pd.DataFrame,
    max_quote_age: pd.Timedelta,
) -> pd.DataFrame:
    open_price = _required_numeric(panel, "open", error="invalid_crypto_ohlc")
    high_price = _required_numeric(panel, "high", error="invalid_crypto_ohlc")
    low_price = _required_numeric(panel, "low", error="invalid_crypto_ohlc")
    close_price = _required_numeric(panel, "close", error="invalid_crypto_ohlc")
    volume = _required_numeric(panel, "volume", error="invalid_crypto_bar_quantity")
    quote_volume = _required_numeric(
        panel,
        "quote_volume",
        error="invalid_crypto_bar_quantity",
    )
    trade_count = _required_numeric(
        panel,
        "trade_count",
        error="invalid_crypto_trade_count",
    )
    taker_buy_volume = _required_numeric(
        panel,
        "taker_buy_volume",
        error="invalid_crypto_bar_quantity",
    )
    taker_buy_quote_volume = _required_numeric(
        panel,
        "taker_buy_quote_volume",
        error="invalid_crypto_bar_quantity",
    )
    valid_trade_count = trade_count.ge(0.0) & trade_count.mod(1).eq(0.0)
    valid_trade_summary = volume.ge(0.0) & valid_trade_count
    inconsistent_trade_summary = valid_trade_summary & volume.eq(0.0).ne(
        trade_count.eq(0.0)
    )
    if inconsistent_trade_summary.any():
        raise ValueError("invalid_crypto_bar_trade_summary")
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
        trade_aggregates,
        how="left",
        on=("asset_id", "timestamp"),
    )
    feed_present = panel["trade_flow_observed"].notna()
    zero_trade_summary = valid_trade_summary & trade_count.eq(0.0)
    positive_trade_summary = valid_trade_summary & trade_count.gt(0.0)
    quantity_matches = (
        panel["quantity_sum"].sub(volume).abs()
        <= _TRADE_VOLUME_ABSOLUTE_TOLERANCE
        + _TRADE_VOLUME_RELATIVE_TOLERANCE * volume.abs()
    )
    count_matches = panel["covered_trade_count"].eq(trade_count)
    invalid_trade_feed = (zero_trade_summary & feed_present) | (
        positive_trade_summary
        & feed_present
        & ~(quantity_matches & count_matches)
    )
    if invalid_trade_feed.any():
        raise ValueError("invalid_crypto_trade_flow_coverage")
    panel["trade_flow_observed"] = panel["trade_flow_observed"].where(
        panel["trade_flow_observed"].notna(),
        False,
    )
    zero_trade = zero_trade_summary & ~feed_present
    panel.loc[zero_trade, "signed_volume"] = 0.0
    panel.loc[zero_trade, "trade_flow_observed"] = True
    panel = panel.merge(
        _match_quotes_asof_bar_close(
            panel,
            quote_snapshots,
            max_quote_age=max_quote_age,
        ),
        how="left",
        on=("asset_id", "timestamp"),
    )
    panel["quote_observed"] = panel["quote_observed"].where(
        panel["quote_observed"].notna(),
        False,
    )
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
        open_price.gt(0.0)
        & high_price.gt(0.0)
        & low_price.gt(0.0)
        & high_price.ge(low_price)
        & close_price.gt(0.0)
        & high_price.ge(pd.concat((open_price, close_price), axis=1).max(axis=1))
        & low_price.le(pd.concat((open_price, close_price), axis=1).min(axis=1))
        & valid_bar_quantity
    )
    return (
        _stabilize_panel_schema(panel.loc[:, PANEL_COLUMNS])
        .sort_values(["market_id", "asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )


def normalize_trade_chunk(trades: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "source_asset_id",
        "asset_id",
        "event_timestamp",
        "timestamp",
        "aggregate_trade_id",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "signed_quantity",
    )
    if trades.empty:
        return pd.DataFrame(columns=list(columns))
    frame = trades.copy()
    _validate_spot_source_market_type(
        frame,
        error="invalid_crypto_trade_source_market_type",
    )
    frame["asset_id"] = _crypto_asset_id_series(frame["source_asset_id"])
    frame["event_timestamp"] = _strict_timestamp_column(
        frame,
        "timestamp",
        error="invalid_crypto_trade_timestamp",
    )
    frame["timestamp"] = frame["event_timestamp"].dt.floor("min")
    frame["signed_quantity"] = _required_numeric(
        frame,
        "signed_quantity",
        error="invalid_crypto_signed_quantity",
    )
    frame["quantity"] = _required_numeric(
        frame,
        "quantity",
        error="invalid_crypto_trade_flow_coverage",
    )
    if frame["quantity"].le(0.0).any():
        raise ValueError("invalid_crypto_trade_flow_coverage")
    signed_quantity_matches = (
        frame["signed_quantity"].abs().sub(frame["quantity"]).abs()
        <= _SIGNED_QUANTITY_ABSOLUTE_TOLERANCE
        + _SIGNED_QUANTITY_RELATIVE_TOLERANCE * frame["quantity"].abs()
    )
    if not signed_quantity_matches.all():
        raise ValueError("invalid_crypto_trade_flow_coverage")
    for column in (
        "aggregate_trade_id",
        "first_trade_id",
        "last_trade_id",
    ):
        frame[column] = _strict_non_negative_integer_column(
            frame,
            column,
            error="invalid_crypto_trade_flow_coverage",
        )
    if frame["first_trade_id"].gt(frame["last_trade_id"]).any():
        raise ValueError("invalid_crypto_trade_flow_coverage")
    return frame.loc[:, list(columns)]


def _match_quotes_asof_bar_close(
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
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
    left = bars.loc[:, ["asset_id", "timestamp", "close_time"]].copy()
    left["_bar_position"] = range(len(left))
    if quotes.empty:
        result = left
        for column in columns:
            if column not in result:
                result[column] = pd.NA
        result["quote_observed"] = False
        return result.loc[:, columns]
    ordered_left = left.sort_values(
        ["close_time", "asset_id", "_bar_position"],
        kind="mergesort",
    )
    ordered_quotes = quotes.sort_values(
        ["quote_timestamp", "asset_id"],
        kind="mergesort",
    )
    matched = pd.merge_asof(
        ordered_left,
        ordered_quotes,
        left_on="close_time",
        right_on="quote_timestamp",
        by="asset_id",
        tolerance=max_quote_age,
        direction="backward",
        allow_exact_matches=True,
    )
    matched["quote_observed"] = matched["quote_timestamp"].notna() & matched[
        "quote_valid"
    ].fillna(False)
    age_seconds = (
        matched["close_time"] - matched["quote_timestamp"]
    ).dt.total_seconds()
    matched["quote_age_seconds"] = age_seconds.where(matched["quote_observed"])
    return (
        matched.sort_values("_bar_position", kind="mergesort")
        .reset_index(drop=True)
        .loc[:, columns]
    )


def normalize_depth_chunk(depth: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "source_asset_id",
        "asset_id",
        "quote_timestamp",
        "side",
        "level",
        "price",
        "quantity",
    )
    if depth.empty:
        return pd.DataFrame(columns=list(columns))
    frame = depth.copy()
    _validate_spot_source_market_type(
        frame,
        error="invalid_crypto_depth_source_market_type",
    )
    frame["asset_id"] = _crypto_asset_id_series(frame["source_asset_id"])
    frame["quote_timestamp"] = _strict_timestamp_column(
        frame,
        "timestamp",
        error="invalid_crypto_depth_timestamp",
    )
    _validate_depth_level_values(frame)
    return frame.loc[:, list(columns)]


def _normalize_depth_chunk_for_reducer(depth: pd.DataFrame) -> pd.DataFrame:
    return normalize_depth_chunk(depth).reset_index(drop=True)


def finalize_depth_snapshot(
    asset_id: str,
    quote_timestamp: pd.Timestamp,
    sides: Mapping[str, Mapping[int, tuple[float, float]]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "asset_id": asset_id,
        "quote_timestamp": quote_timestamp,
        "best_bid": pd.NA,
        "best_ask": pd.NA,
        "bid_quantity": pd.NA,
        "ask_quantity": pd.NA,
        "bid_depth": pd.NA,
        "ask_depth": pd.NA,
    }
    for side, levels in sides.items():
        ordered_levels = sorted(levels)
        if ordered_levels != list(range(1, len(ordered_levels) + 1)):
            raise ValueError("invalid_crypto_depth_level_sequence")
        prices = [levels[level][0] for level in ordered_levels]
        if side == "bid" and any(
            current >= previous
            for previous, current in zip(prices, prices[1:], strict=False)
        ):
            raise ValueError("invalid_crypto_bid_price_order")
        if side == "ask" and any(
            current <= previous
            for previous, current in zip(prices, prices[1:], strict=False)
        ):
            raise ValueError("invalid_crypto_ask_price_order")
        best_price, best_quantity = levels[1]
        row[f"best_{side}"] = best_price
        row[f"{side}_quantity"] = best_quantity
        row[f"{side}_depth"] = sum(
            levels[level][1] for level in ordered_levels
        )
    required = ("best_bid", "best_ask", "bid_quantity", "ask_quantity")
    quote_valid = all(pd.notna(row[column]) for column in required)
    if quote_valid:
        row["mid_price"] = (float(row["best_bid"]) + float(row["best_ask"])) / 2
        quote_valid = bool(
            float(row["best_bid"]) > 0
            and float(row["best_ask"]) > 0
            and float(row["bid_quantity"]) > 0
            and float(row["ask_quantity"]) > 0
            and float(row["best_bid"]) < float(row["best_ask"])
        )
    else:
        row["mid_price"] = pd.NA
    row["quote_valid"] = quote_valid
    return row


def _crypto_asset_id_series(series: pd.Series) -> pd.Series:
    try:
        values = series.map(canonical_crypto_asset_id)
    except ValueError as exc:
        raise ValueError("invalid_crypto_source_asset_id") from exc
    if values.isna().any():
        raise ValueError("invalid_crypto_source_asset_id")
    return values


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _required_numeric(frame: pd.DataFrame, column: str, *, error: str) -> pd.Series:
    values = _numeric_not_bool(frame, column, error=error)
    if values.isna().any() or not values.map(math.isfinite).all():
        raise ValueError(error)
    return values


def _numeric_not_bool(frame: pd.DataFrame, column: str, *, error: str) -> pd.Series:
    if frame[column].map(lambda value: isinstance(value, bool)).any():
        raise ValueError(error)
    return pd.to_numeric(frame[column], errors="coerce")


def _strict_non_negative_integer_column(
    frame: pd.DataFrame,
    column: str,
    *,
    error: str,
) -> pd.Series:
    values = _numeric_not_bool(frame, column, error=error)
    if (
        values.isna().any()
        or not values.map(math.isfinite).all()
        or values.lt(0).any()
        or values.mod(1).ne(0).any()
    ):
        raise ValueError(error)
    return values.astype("int64")


def _strict_timestamp_column(
    frame: pd.DataFrame,
    column: str,
    *,
    error: str,
) -> pd.Series:
    return strict_timestamp_series(frame[column], error=error)


def _validate_depth_level_values(frame: pd.DataFrame) -> None:
    for side in frame["side"]:
        if not isinstance(side, str) or side not in {"bid", "ask"}:
            raise ValueError("invalid_crypto_depth_side")
    levels = pd.to_numeric(frame["level"], errors="coerce")
    if (
        levels.isna().any()
        or levels.le(0).any()
        or levels.mod(1).ne(0).any()
        or frame["level"].map(lambda value: isinstance(value, bool)).any()
    ):
        raise ValueError("invalid_crypto_depth_level")
    frame["level"] = levels.astype("int64")
    if frame.duplicated(["asset_id", "quote_timestamp", "side", "level"]).any():
        raise ValueError("duplicate_crypto_depth_level")
    prices = _required_numeric(frame, "price", error="invalid_crypto_depth_price")
    quantities = _required_numeric(
        frame,
        "quantity",
        error="invalid_crypto_depth_quantity",
    )
    if prices.le(0.0).any():
        raise ValueError("invalid_crypto_depth_price")
    if quantities.lt(0.0).any():
        raise ValueError("invalid_crypto_depth_quantity")
    frame["price"] = prices
    frame["quantity"] = quantities


def _validate_one_minute_bar_contract(frame: pd.DataFrame) -> None:
    timestamps = frame["timestamp"]
    if (
        not bool(timestamps.eq(frame["open_time"]).all())
        or not bool(timestamps.eq(timestamps.dt.floor("min")).all())
    ):
        raise ValueError("invalid_crypto_bar_open_time")
    expected_close = timestamps + pd.Timedelta(milliseconds=59_999)
    if not bool(frame["close_time"].eq(expected_close).all()):
        raise ValueError("invalid_crypto_bar_close_time")
    _validate_exact_string_column(
        frame,
        column="interval",
        expected="1m",
        error="invalid_crypto_bar_interval",
    )
    _validate_spot_source_market_type(
        frame,
        error="invalid_crypto_bar_source_market_type",
    )


def _validate_spot_source_market_type(frame: pd.DataFrame, *, error: str) -> None:
    _validate_exact_string_column(
        frame,
        column="source_market_type",
        expected="spot",
        error=error,
    )


def _validate_exact_string_column(
    frame: pd.DataFrame,
    *,
    column: str,
    expected: str,
    error: str,
) -> None:
    if column not in frame or not frame[column].map(
        lambda value: isinstance(value, str) and value == expected
    ).all():
        raise ValueError(error)


def _stabilize_panel_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_id"] = result["market_id"].astype(str)
    result["asset_id"] = result["asset_id"].astype(str)
    result["timestamp"] = _strict_timestamp_column(
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


__all__ = [
    "calculate_market_panel_from_chunks",
    "prepare_panel_bars",
]
