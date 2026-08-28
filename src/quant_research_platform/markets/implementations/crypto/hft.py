"""Binance HFT panel, factor, and label semantics."""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    tick_to_price,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import _OwnedValue

HFT_GRID_MS = 100


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


def iter_panel_rows(
    book_events: Iterable[BookState | _OwnedValue[BookState] | BookBoundary],
    trades: Iterable[TradeEvent],
    *,
    start_ms: int,
    end_ms: int,
    book_coverage: Mapping[str, tuple[int, int]],
    trade_coverage: Mapping[str, tuple[int, int]],
    interval_ms: int = HFT_GRID_MS,
) -> Iterator[dict[str, Any]]:
    """Sample one fully covered depth segment and trade connection per grid."""

    if type(start_ms) is not int or type(end_ms) is not int or end_ms < start_ms:
        raise ValueError("hft_panel_window_invalid")
    if interval_ms != HFT_GRID_MS:
        raise ValueError("hft_panel_interval_invalid")
    active_books = iter(book_events)
    active_trades = iter(trades)
    with closing_iterator(active_books), closing_iterator(active_trades):
        next_book = next(active_books, None)
        next_trade = next(active_trades, None)
        current_state: BookState | None = None
        current_owned: _OwnedValue[BookState] | None = None
        previous_bbo: tuple[tuple[int, float], tuple[int, float]] | None = None
        previous_book_time: int | None = None
        previous_trade_time: int | None = None
        try:
            for grid_time in range(start_ms, end_ms + 1, interval_ms):
                interval_start = max(0, grid_time - HFT_GRID_MS)
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
                    if bounds[0] <= interval_start and bounds[1] > grid_time
                )
                while (
                    next_book is not None
                    and _book_observed_time(next_book) <= grid_time
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
                        _close_owned(current_owned)
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
                        and event_time > interval_start
                    ):
                        book_ofi += l1_ofi(
                            previous_bbo[0],
                            state.best_bid,
                            previous_bbo[1],
                            state.best_ask,
                        )
                    _close_owned(current_owned)
                    current_owned = event if isinstance(event, _OwnedValue) else None
                    current_state = state
                    if state.best_bid is not None and state.best_ask is not None:
                        previous_bbo = (state.best_bid, state.best_ask)
                while next_trade is not None:
                    if not isinstance(next_trade, TradeEvent):
                        raise TypeError("hft_trade_event_invalid")
                    if next_trade.observed_time_ms > grid_time:
                        break
                    trade_time = next_trade.observed_time_ms
                    if (
                        previous_trade_time is not None
                        and trade_time < previous_trade_time
                    ):
                        raise ValueError("hft_trade_time_non_monotonic")
                    previous_trade_time = trade_time
                    trade = next_trade
                    next_trade = next(active_trades, None)
                    quantity = _number(
                        trade.quantity,
                        "hft_trade_quantity_invalid",
                    )
                    visible = _number(
                        trade.visible_quantity,
                        "hft_trade_visible_quantity_invalid",
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
                        and trade.observed_time_ms > interval_start
                        and trade.observed_time_ms <= grid_time
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
                    or book_bounds[0] > interval_start
                    or book_bounds[1] <= grid_time
                ):
                    continue
                if len(trade_connections) != 1:
                    continue
                yield _panel_row(
                    current_state,
                    grid_time,
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
        finally:
            _close_owned(current_owned)
            if isinstance(next_book, _OwnedValue):
                next_book.lease.close()


def _book_value(event: BookState | _OwnedValue[BookState]) -> BookState:
    value = event.value if isinstance(event, _OwnedValue) else event
    if not isinstance(value, BookState):
        raise TypeError("hft_book_event_invalid")
    return value


def _book_observed_time(
    event: BookState | _OwnedValue[BookState] | BookBoundary,
) -> int:
    return (
        event.value.observed_time_ms
        if isinstance(event, _OwnedValue)
        else event.observed_time_ms
    )


def _close_owned(value: _OwnedValue[BookState] | None) -> None:
    if value is not None:
        value.lease.close()


def _number(value: object, error: str) -> float:
    if isinstance(value, bool):
        raise ValueError(error)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not math.isfinite(number):
        raise ValueError(error)
    return number


def _panel_row(
    state: BookState,
    grid_time: int,
    *,
    trade_connection_id: str,
    trade_capture_seq_min: int | None,
    trade_capture_seq_max: int | None,
    book_ofi: float,
    trade_count: int,
    trade_volume: float,
    signed_trade_volume: float,
    taker_buy_volume: float,
    taker_sell_volume: float,
) -> dict[str, Any]:
    bids = tuple(
        (tick_to_price(price_tick, state.tick_size), quantity)
        for price_tick, quantity in state.bids[:10]
    )
    asks = tuple(
        (tick_to_price(price_tick, state.tick_size), quantity)
        for price_tick, quantity in state.asks[:10]
    )
    if not bids or not asks:
        raise ValueError("hft_panel_one_sided_book")
    bid_price, bid_quantity = bids[0]
    ask_price, ask_quantity = asks[0]
    mid = (bid_price + ask_price) / 2.0
    depths = {
        levels: (
            sum(quantity for _, quantity in bids[:levels]),
            sum(quantity for _, quantity in asks[:levels]),
        )
        for levels in (1, 5, 10)
    }
    return {
        "timestamp_ms": grid_time,
        "timestamp": _timestamp(grid_time),
        "segment_id": f"{state.segment_id}:{trade_connection_id}",
        "book_segment_id": state.segment_id,
        "book_connection_id": state.connection_id,
        "trade_connection_id": trade_connection_id,
        "book_capture_seq": state.capture_seq,
        "trade_capture_seq_min": trade_capture_seq_min,
        "trade_capture_seq_max": trade_capture_seq_max,
        "book_observed_time_ms": state.observed_time_ms,
        "exchange_event_time_ms": state.event_time_ms,
        "transaction_time_ms": state.transaction_time_ms,
        "local_receive_time": _receive_timestamp(state.receive_time),
        "book_receive_monotonic_ns": state.receive_monotonic_ns,
        "update_id": state.update_id,
        "best_bid_price": bid_price,
        "best_bid_qty": bid_quantity,
        "best_ask_price": ask_price,
        "best_ask_qty": ask_quantity,
        "mid": mid,
        "bid_depth_1": depths[1][0],
        "ask_depth_1": depths[1][1],
        "bid_depth_5": depths[5][0],
        "ask_depth_5": depths[5][1],
        "bid_depth_10": depths[10][0],
        "ask_depth_10": depths[10][1],
        "book_ofi_100ms": book_ofi,
        "trade_count_100ms": trade_count,
        "trade_volume_100ms": trade_volume,
        "observed_signed_trade_volume_100ms": signed_trade_volume,
        "observed_taker_buy_volume_100ms": taker_buy_volume,
        "observed_taker_sell_volume_100ms": taker_sell_volume,
        "eligible": True,
    }


def _timestamp(value: int) -> dt.datetime:
    return dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=value)


def _receive_timestamp(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("hft_receive_time_invalid") from exc
    return parsed.astimezone(dt.UTC).replace(tzinfo=None)


def iter_factor_rows(
    rows: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Derive factors only from persisted PANEL sufficient statistics."""

    active_segment: str | None = None
    previous_time: int | None = None
    previous_mid: float | None = None
    returns: deque[float] = deque(maxlen=10)
    for row in rows:
        timestamp_ms, segment_id = _grid_identity(row)
        if segment_id != active_segment:
            active_segment = segment_id
            previous_time = None
            previous_mid = None
            returns.clear()
        elif previous_time is not None and timestamp_ms - previous_time != HFT_GRID_MS:
            raise ValueError("hft_grid_non_contiguous")
        mid = _number(row.get("mid"), "hft_factor_mid_invalid")
        bid_price = _number(row.get("best_bid_price"), "hft_factor_book_invalid")
        ask_price = _number(row.get("best_ask_price"), "hft_factor_book_invalid")
        bid_quantity = _number(row.get("best_bid_qty"), "hft_factor_book_invalid")
        ask_quantity = _number(row.get("best_ask_qty"), "hft_factor_book_invalid")
        spread = ask_price - bid_price
        if mid <= 0 or spread <= 0:
            raise ValueError("hft_factor_book_invalid")
        depth = {
            levels: (
                _number(row.get(f"bid_depth_{levels}"), "hft_factor_depth_invalid"),
                _number(row.get(f"ask_depth_{levels}"), "hft_factor_depth_invalid"),
            )
            for levels in (1, 5, 10)
        }
        bbo_quantity = bid_quantity + ask_quantity
        microprice = (
            (ask_price * bid_quantity + bid_price * ask_quantity) / bbo_quantity
            if bbo_quantity > 0
            else math.nan
        )
        current_return = (
            mid / previous_mid - 1.0
            if previous_mid is not None and previous_mid != 0
            else math.nan
        )
        if math.isfinite(current_return):
            returns.append(current_return)
        output = _identity_columns(row)
        output.update(
            {
                "spread": spread,
                "relative_spread": spread / mid,
                "imbalance_1": _imbalance(*depth[1]),
                "imbalance_5": _imbalance(*depth[5]),
                "imbalance_10": _imbalance(*depth[10]),
                "microprice": microprice,
                "microprice_deviation": microprice - mid,
                "ofi_100ms": _number(
                    row.get("book_ofi_100ms"),
                    "hft_factor_ofi_invalid",
                ),
                "signed_trade_flow_100ms": _number(
                    row.get("observed_signed_trade_volume_100ms"),
                    "hft_factor_trade_flow_invalid",
                ),
                "taker_buy_volume_100ms": _number(
                    row.get("observed_taker_buy_volume_100ms"),
                    "hft_factor_trade_flow_invalid",
                ),
                "taker_sell_volume_100ms": _number(
                    row.get("observed_taker_sell_volume_100ms"),
                    "hft_factor_trade_flow_invalid",
                ),
                "return_100ms": current_return,
                "realized_volatility_1s": _sample_std(returns),
                "top10_depth_per_spread": sum(depth[10]) / spread,
            }
        )
        yield output
        previous_mid = mid
        previous_time = timestamp_ms


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return (bid - ask) / total if total > 0 else math.nan


def _sample_std(values: deque[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def iter_label_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    horizons_ms: tuple[int, ...],
) -> Iterator[dict[str, Any]]:
    """Emit exact-grid same-segment labels with bounded pending state."""

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
    max_horizon = max(horizons)
    pending: deque[dict[str, Any]] = deque()
    mids: dict[int, float] = {}
    active_segment: str | None = None
    previous_time: int | None = None

    def flush() -> Iterator[dict[str, Any]]:
        while pending:
            yield _label_row(pending.popleft(), mids=mids, horizons=horizons)

    for source in rows:
        row = dict(source)
        timestamp_ms, segment_id = _grid_identity(row)
        if segment_id != active_segment:
            yield from flush()
            pending.clear()
            mids.clear()
            active_segment = segment_id
            previous_time = None
        elif previous_time is not None and timestamp_ms - previous_time != HFT_GRID_MS:
            raise ValueError("hft_grid_non_contiguous")
        mids[timestamp_ms] = _number(row.get("mid"), "hft_label_mid_invalid")
        pending.append(row)
        while pending and timestamp_ms >= pending[0]["timestamp_ms"] + max_horizon:
            yield _label_row(pending.popleft(), mids=mids, horizons=horizons)
        cutoff = timestamp_ms - max_horizon
        for stale in tuple(value for value in mids if value < cutoff):
            del mids[stale]
        previous_time = timestamp_ms
    yield from flush()


def _label_row(
    row: dict[str, Any],
    *,
    mids: dict[int, float],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    timestamp_ms = row["timestamp_ms"]
    current_mid = _number(row.get("mid"), "hft_label_mid_invalid")
    output = _identity_columns(row)
    for horizon in horizons:
        future_mid = mids.get(timestamp_ms + horizon)
        value = math.nan if future_mid is None else future_mid / current_mid - 1.0
        output[f"future_mid_return_{horizon}ms"] = value
        output[f"future_mid_direction_{horizon}ms"] = (
            math.nan
            if math.isnan(value)
            else 1.0
            if value > 0
            else -1.0
            if value < 0
            else 0.0
        )
    return output


def _grid_identity(row: Mapping[str, Any]) -> tuple[int, str]:
    timestamp_ms = row.get("timestamp_ms")
    segment_id = row.get("segment_id")
    if type(timestamp_ms) is not int or timestamp_ms < 0:
        raise ValueError("hft_grid_time_invalid")
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError("hft_grid_segment_invalid")
    return timestamp_ms, segment_id


def _identity_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("market_id", "asset_id", "timestamp", "timestamp_ms")
        if key in row
    }
