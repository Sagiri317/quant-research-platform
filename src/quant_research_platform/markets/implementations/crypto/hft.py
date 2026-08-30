"""Binance HFT panel, factor, and label semantics."""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    tick_to_price,
)
from quant_research_platform.contracts.validation import (
    strict_finite_float,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN,
    HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import OwnedValue

HFT_GRID_MS = 100
_NANOSECONDS_PER_MILLISECOND = 1_000_000


def l1_ofi(
    previous_bid: tuple[int, float],
    current_bid: tuple[int, float],
    previous_ask: tuple[int, float],
    current_ask: tuple[int, float],
) -> float:
    """Cont-style L1 OFI over exact integer price ticks."""

    previous_bid_price, previous_bid_qty = previous_bid
    current_bid_price, current_bid_qty = current_bid
    previous_ask_price, previous_ask_qty = previous_ask
    current_ask_price, current_ask_qty = current_ask
    if current_bid_price > previous_bid_price:
        bid_term = current_bid_qty
    elif current_bid_price == previous_bid_price:
        bid_term = current_bid_qty - previous_bid_qty
    else:
        bid_term = -previous_bid_qty
    if current_ask_price < previous_ask_price:
        ask_term = -current_ask_qty
    elif current_ask_price == previous_ask_price:
        ask_term = previous_ask_qty - current_ask_qty
    else:
        ask_term = previous_ask_qty
    return bid_term + ask_term


def iter_panel_frames(
    book_events: Iterable[BookState | OwnedValue[BookState] | BookBoundary],
    trades: Iterable[TradeEvent],
    *,
    market_id: str,
    asset_id: str,
    start_ms: int,
    end_ms: int,
    book_coverage: Mapping[str, tuple[int, int]],
    trade_coverage: Mapping[str, tuple[int, int]],
    batch_rows: int,
    interval_ms: int = HFT_GRID_MS,
) -> Iterator[pd.DataFrame]:
    """Sample one fully covered depth segment and trade connection per grid."""

    if type(start_ms) is not int or type(end_ms) is not int or end_ms < start_ms:
        raise ValueError("hft_panel_window_invalid")
    if interval_ms != HFT_GRID_MS:
        raise ValueError("hft_panel_interval_invalid")
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError("hft_panel_batch_rows_invalid")
    if (
        type(market_id) is not str
        or not market_id
        or type(asset_id) is not str
        or not asset_id
    ):
        raise ValueError("hft_panel_identity_invalid")
    buffers = _panel_column_buffers()
    active_books = iter(book_events)
    active_trades = iter(trades)
    with closing_iterator(active_books), closing_iterator(active_trades):
        next_book: BookState | OwnedValue[BookState] | BookBoundary | None = None
        next_trade: TradeEvent | None = None
        current_state: BookState | None = None
        current_owned: OwnedValue[BookState] | None = None
        previous_bbo: tuple[tuple[int, float], tuple[int, float]] | None = None
        previous_book_time: int | None = None
        previous_trade_time: int | None = None
        try:
            next_book = next(active_books, None)
            next_trade = next(active_trades, None)
            for grid_time in range(start_ms, end_ms + 1, interval_ms):
                interval_start = max(0, grid_time - HFT_GRID_MS)
                grid_time_ns = grid_time * _NANOSECONDS_PER_MILLISECOND
                interval_start_ns = interval_start * _NANOSECONDS_PER_MILLISECOND
                book_ofi = 0.0
                trade_count = 0
                trade_volume = 0.0
                signed_trade_volume = 0.0
                taker_buy_volume = 0.0
                taker_sell_volume = 0.0
                trade_capture_seq_min: int | None = None
                trade_capture_seq_max: int | None = None
                trade_connections = tuple(
                    connection_id
                    for connection_id, bounds in trade_coverage.items()
                    if bounds[0] <= interval_start_ns and bounds[1] > grid_time_ns
                )
                while (
                    next_book is not None
                    and _book_observed_time(next_book) <= grid_time_ns
                ):
                    event_time = _book_observed_time(next_book)
                    if (
                        previous_book_time is not None
                        and event_time < previous_book_time
                    ):
                        raise ValueError("hft_book_time_non_monotonic")
                    previous_book_time = event_time
                    event = next_book
                    next_book = next(active_books, None)
                    if isinstance(event, BookBoundary):
                        if current_owned is not None:
                            current_owned.release(owner="crypto.hft.book")
                        current_owned = None
                        current_state = None
                        previous_bbo = None
                        book_ofi = 0.0
                        continue
                    state = _book_value(event)
                    if (
                        current_state is None
                        or state.segment_id != current_state.segment_id
                    ):
                        previous_bbo = None
                        book_ofi = 0.0
                    if (
                        previous_bbo is not None
                        and state.best_bid is not None
                        and state.best_ask is not None
                        and event_time > interval_start_ns
                    ):
                        book_ofi += l1_ofi(
                            previous_bbo[0],
                            state.best_bid,
                            previous_bbo[1],
                            state.best_ask,
                        )
                    if current_owned is not None:
                        current_owned.release(owner="crypto.hft.book")
                    current_owned = event if isinstance(event, OwnedValue) else None
                    current_state = state
                    if state.best_bid is not None and state.best_ask is not None:
                        previous_bbo = (state.best_bid, state.best_ask)
                while next_trade is not None:
                    if not isinstance(next_trade, TradeEvent):
                        raise TypeError("hft_trade_event_invalid")
                    if next_trade.observed_time_ns > grid_time_ns:
                        break
                    trade_time = next_trade.observed_time_ns
                    if (
                        previous_trade_time is not None
                        and trade_time < previous_trade_time
                    ):
                        raise ValueError("hft_trade_time_non_monotonic")
                    previous_trade_time = trade_time
                    trade = next_trade
                    next_trade = next(active_trades, None)
                    quantity = strict_finite_float(
                        trade.quantity,
                        error="hft_trade_quantity_invalid",
                    )
                    visible = strict_finite_float(
                        trade.visible_quantity,
                        error="hft_trade_visible_quantity_invalid",
                    )
                    if (
                        quantity <= 0
                        or visible < 0
                        or visible > quantity
                        or not isinstance(trade.buyer_is_maker, bool)
                    ):
                        raise ValueError("hft_trade_values_invalid")
                    if (
                        current_state is not None
                        and len(trade_connections) == 1
                        and trade.connection_id == trade_connections[0]
                        and trade.observed_time_ns > interval_start_ns
                        and trade.observed_time_ns <= grid_time_ns
                    ):
                        trade_count += 1
                        trade_volume += quantity
                        trade_capture_seq_min = (
                            trade.capture_seq
                            if trade_capture_seq_min is None
                            else min(trade_capture_seq_min, trade.capture_seq)
                        )
                        trade_capture_seq_max = (
                            trade.capture_seq
                            if trade_capture_seq_max is None
                            else max(trade_capture_seq_max, trade.capture_seq)
                        )
                        if trade.buyer_is_maker:
                            signed_trade_volume -= quantity
                            taker_sell_volume += quantity
                        else:
                            signed_trade_volume += quantity
                            taker_buy_volume += quantity
                if current_state is None:
                    continue
                book_bounds = book_coverage.get(current_state.segment_id)
                if (
                    book_bounds is None
                    or book_bounds[0] > interval_start_ns
                    or book_bounds[1] <= grid_time_ns
                ):
                    continue
                if len(trade_connections) != 1:
                    continue
                _append_panel_columns(
                    buffers,
                    current_state,
                    grid_time,
                    market_id=market_id,
                    asset_id=asset_id,
                    trade_connection_id=trade_connections[0],
                    trade_capture_seq_min=trade_capture_seq_min,
                    trade_capture_seq_max=trade_capture_seq_max,
                    book_ofi=book_ofi,
                    trade_count=trade_count,
                    trade_volume=trade_volume,
                    signed_trade_volume=signed_trade_volume,
                    taker_buy_volume=taker_buy_volume,
                    taker_sell_volume=taker_sell_volume,
                )
                if len(buffers["timestamp"]) >= batch_rows:
                    yield pd.DataFrame(buffers)
                    buffers = _panel_column_buffers()
            if buffers["timestamp"]:
                yield pd.DataFrame(buffers)
        finally:
            if current_owned is not None:
                current_owned.release(owner="crypto.hft.book")
            if isinstance(next_book, OwnedValue):
                next_book.release(owner="crypto.hft.book")


def _book_value(event: BookState | OwnedValue[BookState]) -> BookState:
    value = event.value if isinstance(event, OwnedValue) else event
    if not isinstance(value, BookState):
        raise TypeError("hft_book_event_invalid")
    return value


def _book_observed_time(
    event: BookState | OwnedValue[BookState] | BookBoundary,
) -> int:
    return (
        event.value.observed_time_ns
        if isinstance(event, OwnedValue)
        else event.observed_time_ns
    )


def _panel_column_buffers() -> dict[str, list[Any]]:
    return {
        column: []
        for column in (
            "market_id",
            "asset_id",
            "timestamp",
            "segment_id",
            "book_segment_id",
            "book_connection_id",
            "trade_connection_id",
            "book_capture_seq",
            "trade_capture_seq_min",
            "trade_capture_seq_max",
            "book_observed_time_ms",
            "exchange_event_time_ms",
            "transaction_time_ms",
            "local_receive_time",
            "book_receive_monotonic_ns",
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "mid",
            "bid_depth_1",
            "ask_depth_1",
            "bid_depth_5",
            "ask_depth_5",
            "bid_depth_10",
            "ask_depth_10",
            "book_ofi_100ms",
            "trade_count_100ms",
            "trade_volume_100ms",
            "observed_signed_trade_volume_100ms",
            "observed_taker_buy_volume_100ms",
            "observed_taker_sell_volume_100ms",
            "eligible",
        )
    }


def _append_panel_columns(
    buffers: dict[str, list[Any]],
    state: BookState,
    grid_time: int,
    *,
    market_id: str,
    asset_id: str,
    trade_connection_id: str,
    trade_capture_seq_min: int | None,
    trade_capture_seq_max: int | None,
    book_ofi: float,
    trade_count: int,
    trade_volume: float,
    signed_trade_volume: float,
    taker_buy_volume: float,
    taker_sell_volume: float,
) -> None:
    bids = state.bids[:10]
    asks = state.asks[:10]
    if not bids or not asks:
        raise ValueError("hft_panel_one_sided_book")
    bid_price = tick_to_price(bids[0][0], state.tick_size)
    bid_quantity = bids[0][1]
    ask_price = tick_to_price(asks[0][0], state.tick_size)
    ask_quantity = asks[0][1]
    mid = (bid_price + ask_price) / 2.0
    bid_depth_1 = sum(quantity for _price, quantity in bids[:1])
    ask_depth_1 = sum(quantity for _price, quantity in asks[:1])
    bid_depth_5 = sum(quantity for _price, quantity in bids[:5])
    ask_depth_5 = sum(quantity for _price, quantity in asks[:5])
    bid_depth_10 = sum(quantity for _price, quantity in bids)
    ask_depth_10 = sum(quantity for _price, quantity in asks)
    buffers["market_id"].append(market_id)
    buffers["asset_id"].append(asset_id)
    buffers["timestamp"].append(_timestamp(grid_time))
    buffers["segment_id"].append(f"{state.segment_id}:{trade_connection_id}")
    buffers["book_segment_id"].append(state.segment_id)
    buffers["book_connection_id"].append(state.connection_id)
    buffers["trade_connection_id"].append(trade_connection_id)
    buffers["book_capture_seq"].append(state.capture_seq)
    buffers["trade_capture_seq_min"].append(trade_capture_seq_min)
    buffers["trade_capture_seq_max"].append(trade_capture_seq_max)
    buffers["book_observed_time_ms"].append(state.observed_time_ms)
    buffers["exchange_event_time_ms"].append(state.event_time_ms)
    buffers["transaction_time_ms"].append(state.transaction_time_ms)
    buffers["local_receive_time"].append(_receive_timestamp(state.receive_time))
    buffers["book_receive_monotonic_ns"].append(state.receive_monotonic_ns)
    buffers["update_id"].append(state.update_id)
    buffers["best_bid_price"].append(bid_price)
    buffers["best_bid_qty"].append(bid_quantity)
    buffers["best_ask_price"].append(ask_price)
    buffers["best_ask_qty"].append(ask_quantity)
    buffers["mid"].append(mid)
    buffers["bid_depth_1"].append(bid_depth_1)
    buffers["ask_depth_1"].append(ask_depth_1)
    buffers["bid_depth_5"].append(bid_depth_5)
    buffers["ask_depth_5"].append(ask_depth_5)
    buffers["bid_depth_10"].append(bid_depth_10)
    buffers["ask_depth_10"].append(ask_depth_10)
    buffers["book_ofi_100ms"].append(book_ofi)
    buffers["trade_count_100ms"].append(trade_count)
    buffers["trade_volume_100ms"].append(trade_volume)
    buffers["observed_signed_trade_volume_100ms"].append(signed_trade_volume)
    buffers["observed_taker_buy_volume_100ms"].append(taker_buy_volume)
    buffers["observed_taker_sell_volume_100ms"].append(taker_sell_volume)
    buffers["eligible"].append(True)


def _timestamp(value: int) -> dt.datetime:
    return dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=value)


def _receive_timestamp(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("hft_receive_time_invalid") from exc
    return parsed.astimezone(dt.UTC).replace(tzinfo=None)


def iter_factor_frames(frames: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Derive HFT factors from bounded persisted PANEL frames."""

    active_segment: str | None = None
    previous_time: int | None = None
    previous_mid: float | None = None
    return_tail: deque[float] = deque(maxlen=10)
    for source in frames:
        if source.empty:
            continue
        frame = _with_timestamp_ms(source.reset_index(drop=True))
        timestamps, segments, boundaries = _validated_grid_arrays(
            frame,
            active_segment=active_segment,
            previous_time=previous_time,
        )
        mid = _finite_column(frame, "mid", "hft_factor_mid_invalid")
        bid_price = _finite_column(frame, "best_bid_price", "hft_factor_book_invalid")
        ask_price = _finite_column(frame, "best_ask_price", "hft_factor_book_invalid")
        bid_quantity = _finite_column(frame, "best_bid_qty", "hft_factor_book_invalid")
        ask_quantity = _finite_column(frame, "best_ask_qty", "hft_factor_book_invalid")
        spread = ask_price - bid_price
        if bool(np.any(mid <= 0)) or bool(np.any(spread <= 0)):
            raise ValueError("hft_factor_book_invalid")
        depth = {
            levels: (
                _finite_column(
                    frame,
                    f"bid_depth_{levels}",
                    "hft_factor_depth_invalid",
                ),
                _finite_column(
                    frame,
                    f"ask_depth_{levels}",
                    "hft_factor_depth_invalid",
                ),
            )
            for levels in (1, 5, 10)
        }
        bbo_quantity = bid_quantity + ask_quantity
        microprice = np.full(len(frame), np.nan, dtype="float64")
        np.divide(
            ask_price * bid_quantity + bid_price * ask_quantity,
            bbo_quantity,
            out=microprice,
            where=bbo_quantity > 0,
        )
        returns = np.full(len(frame), np.nan, dtype="float64")
        same_segment = ~boundaries
        returns[1:][same_segment[1:]] = (
            mid[1:][same_segment[1:]] / mid[:-1][same_segment[1:]] - 1.0
        )
        if not boundaries[0] and previous_mid is not None:
            returns[0] = mid[0] / previous_mid - 1.0
        volatility = np.full(len(frame), np.nan, dtype="float64")
        for index, value in enumerate(returns):
            if boundaries[index]:
                return_tail.clear()
            if math.isfinite(value):
                return_tail.append(float(value))
            volatility[index] = _sample_std(return_tail)
        output = _identity_frame(frame)
        output["spread"] = spread
        output["relative_spread"] = spread / mid
        for levels in (1, 5, 10):
            bid_depth, ask_depth = depth[levels]
            total = bid_depth + ask_depth
            imbalance = np.full(len(frame), np.nan, dtype="float64")
            np.divide(
                bid_depth - ask_depth,
                total,
                out=imbalance,
                where=total > 0,
            )
            output[f"imbalance_{levels}"] = imbalance
        output["microprice"] = microprice
        output["microprice_deviation"] = microprice - mid
        output["ofi_100ms"] = _finite_column(
            frame, "book_ofi_100ms", "hft_factor_ofi_invalid"
        )
        output["signed_trade_flow_100ms"] = _finite_column(
            frame,
            "observed_signed_trade_volume_100ms",
            "hft_factor_trade_flow_invalid",
        )
        output["taker_buy_volume_100ms"] = _finite_column(
            frame,
            "observed_taker_buy_volume_100ms",
            "hft_factor_trade_flow_invalid",
        )
        output["taker_sell_volume_100ms"] = _finite_column(
            frame,
            "observed_taker_sell_volume_100ms",
            "hft_factor_trade_flow_invalid",
        )
        output["return_100ms"] = returns
        output["realized_volatility_1s"] = volatility
        output["top10_depth_per_spread"] = (depth[10][0] + depth[10][1]) / spread
        yield output
        active_segment = str(segments[-1])
        previous_time = int(timestamps[-1])
        previous_mid = float(mid[-1])


def _sample_std(values: deque[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def iter_label_frames(
    frames: Iterable[pd.DataFrame],
    *,
    horizons_ms: tuple[int, ...],
) -> Iterator[pd.DataFrame]:
    """Emit HFT labels from two-pass bounded PANEL frames."""

    horizons = tuple(horizons_ms)
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(
            type(value) is not int or value <= 0 or value % HFT_GRID_MS
            for value in horizons
        )
    ):
        raise ValueError("hft_label_horizon_invalid")
    tail_rows = max(horizons) // HFT_GRID_MS
    buffer = pd.DataFrame()
    active_segment: str | None = None
    previous_time: int | None = None
    for source in frames:
        if source.empty:
            continue
        frame = _with_timestamp_ms(source.reset_index(drop=True))
        timestamps, segments, _boundaries = _validated_grid_arrays(
            frame,
            active_segment=active_segment,
            previous_time=previous_time,
        )
        combined = (
            frame if buffer.empty else pd.concat((buffer, frame), ignore_index=True)
        )
        combined_segments = combined["segment_id"].to_numpy(copy=False)
        earlier = np.flatnonzero(combined_segments != combined_segments[-1])
        last_start = 0 if not len(earlier) else int(earlier[-1]) + 1
        retained = min(tail_rows, len(combined) - last_start)
        emit_stop = len(combined) - retained
        labeled = _label_frame(combined, horizons=horizons)
        if emit_stop:
            yield labeled.iloc[:emit_stop].reset_index(drop=True)
        buffer = combined.iloc[emit_stop:].copy()
        active_segment = str(segments[-1])
        previous_time = int(timestamps[-1])
    if not buffer.empty:
        yield _label_frame(buffer, horizons=horizons).reset_index(drop=True)


def _validated_grid_arrays(
    frame: pd.DataFrame,
    *,
    active_segment: str | None,
    previous_time: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "timestamp_ms" not in frame or not pd.api.types.is_integer_dtype(
        frame["timestamp_ms"].dtype
    ):
        raise ValueError("hft_grid_time_invalid")
    if "segment_id" not in frame or not pd.api.types.is_string_dtype(
        frame["segment_id"].dtype
    ):
        raise ValueError("hft_grid_segment_invalid")
    if frame["segment_id"].isna().any() or not (
        frame["segment_id"].str.len().gt(0).all()
    ):
        raise ValueError("hft_grid_segment_invalid")
    timestamps = frame["timestamp_ms"].to_numpy(dtype="int64", copy=False)
    if bool(np.any(timestamps < 0)):
        raise ValueError("hft_grid_time_invalid")
    segments = frame["segment_id"].to_numpy(copy=False)
    boundaries = np.empty(len(frame), dtype=bool)
    boundaries[0] = str(segments[0]) != active_segment
    boundaries[1:] = segments[1:] != segments[:-1]
    if (
        not boundaries[0]
        and previous_time is not None
        and timestamps[0] - previous_time != HFT_GRID_MS
    ) or bool(
        np.any((~boundaries[1:]) & (timestamps[1:] - timestamps[:-1] != HFT_GRID_MS))
    ):
        raise ValueError("hft_grid_non_contiguous")
    return timestamps, segments, boundaries


def _with_timestamp_ms(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp_ms" in frame:
        return frame
    if "timestamp" not in frame:
        raise ValueError("hft_grid_time_invalid")
    timestamps = strict_timestamp_series(
        frame["timestamp"],
        error="hft_grid_time_invalid",
    ).to_numpy(dtype="datetime64[ns]", copy=False)
    nanoseconds = timestamps.astype("int64", copy=False)
    if bool(np.any(nanoseconds % _NANOSECONDS_PER_MILLISECOND)):
        raise ValueError("hft_grid_time_invalid")
    result = frame.copy()
    result["timestamp_ms"] = nanoseconds // _NANOSECONDS_PER_MILLISECOND
    return result


def _finite_column(frame: pd.DataFrame, column: str, error: str) -> np.ndarray:
    try:
        values = frame[column].to_numpy(dtype="float64", copy=False)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(values).all():
        raise ValueError(error)
    return values


def _identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        key
        for key in ("market_id", "asset_id", "timestamp", "timestamp_ms")
        if key in frame
    ]
    return frame.loc[:, columns].copy()


def _label_frame(frame: pd.DataFrame, *, horizons: tuple[int, ...]) -> pd.DataFrame:
    mid = pd.Series(
        _finite_column(frame, "mid", "hft_label_mid_invalid"),
        index=frame.index,
    )
    if bool(mid.eq(0).any()):
        raise ValueError("hft_label_mid_invalid")
    output = _identity_frame(frame)
    segment_groups = frame["segment_id"].ne(frame["segment_id"].shift()).cumsum()
    grouped = mid.groupby(segment_groups, sort=False)
    for horizon in horizons:
        future = grouped.shift(-(horizon // HFT_GRID_MS))
        values = future / mid - 1.0
        output[f"future_mid_return_{horizon}ms"] = values
        output[f"future_mid_direction_{horizon}ms"] = np.sign(values)
    if HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS in horizons:
        best_bid_quantity = pd.Series(
            _finite_column(
                frame,
                "best_bid_qty",
                "hft_label_best_bid_quantity_invalid",
            ),
            index=frame.index,
        )
        if bool(best_bid_quantity.le(0).any()):
            raise ValueError("hft_label_best_bid_quantity_invalid")
        taker_sell_volume = pd.Series(
            _finite_column(
                frame,
                "observed_taker_sell_volume_100ms",
                "hft_label_taker_sell_volume_invalid",
            ),
            index=frame.index,
        )
        if bool(taker_sell_volume.lt(0).any()):
            raise ValueError("hft_label_taker_sell_volume_invalid")
        grouped_sell = taker_sell_volume.groupby(segment_groups, sort=False)
        steps = HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS // HFT_GRID_MS
        future_sell_volume = pd.concat(
            tuple(grouped_sell.shift(-step) for step in range(1, steps + 1)),
            axis=1,
        ).sum(axis=1, min_count=steps)
        output[HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN] = (
            future_sell_volume / best_bid_quantity
        ).clip(lower=0.0, upper=1.0)
    return output
