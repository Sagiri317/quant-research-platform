"""Binance HFT panel, factor, and label semantics."""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack
from fractions import Fraction
from itertools import tee
from statistics import stdev
from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    index_to_price,
)
from quant_research_platform.contracts.validation import (
    strict_finite_float,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_CROSS_ASSET_CONTEXT_COLUMNS,
    HFT_CROSS_ASSET_FACTOR_COLUMNS,
    HFT_DERIVATIVE_CONTEXT_COLUMNS,
    HFT_DERIVATIVE_FACTOR_COLUMNS,
    HFT_EXTERNAL_CLOCK_FIELDS,
    HFT_EXTERNAL_FIELDS,
    HFT_FACTOR_COLUMNS,
    HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN,
    HFT_SPOT_CONTEXT_COLUMNS,
    HFT_SPOT_FACTOR_COLUMNS,
    label_catalog,
)

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
    book_events: Iterable[BookState | BookBoundary],
    trades: Iterable[TradeEvent],
    *,
    market_id: str,
    asset_id: str,
    start_ms: int,
    end_ms: int,
    book_coverage: Mapping[str, tuple[int, int]],
    trade_coverage: Callable[[int, int], tuple[str, ...]],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """Sample one fully covered depth segment and trade connection per grid."""

    if type(start_ms) is not int or type(end_ms) is not int or end_ms < start_ms:
        raise ValueError("hft_panel_window_invalid")
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
        next_book: BookState | BookBoundary | None = None
        next_trade: TradeEvent | None = None
        current_state: BookState | None = None
        previous_bbo: tuple[tuple[int, float], tuple[int, float]] | None = None
        previous_book_time: int | None = None
        previous_trade_time: int | None = None
        next_book = next(active_books, None)
        next_trade = next(active_trades, None)
        interval_ns = HFT_GRID_MS * _NANOSECONDS_PER_MILLISECOND
        last_grid_ms = start_ms + (end_ms - start_ms) // HFT_GRID_MS * HFT_GRID_MS
        grid_time = start_ms - HFT_GRID_MS
        while grid_time < last_grid_ms:
            grid_time += HFT_GRID_MS
            interval_start = max(0, grid_time - HFT_GRID_MS)
            grid_time_ns = grid_time * _NANOSECONDS_PER_MILLISECOND
            interval_start_ns = interval_start * _NANOSECONDS_PER_MILLISECOND
            book_ofi = 0.0
            trade_count = 0
            trade_volume = 0.0
            signed_trade_volume = 0.0
            taker_buy_volume = 0.0
            taker_sell_volume = 0.0
            trade_source_sequence_min: int | None = None
            trade_source_sequence_max: int | None = None
            trade_connections = trade_coverage(interval_start_ns, grid_time_ns)
            while (
                next_book is not None and _book_observed_time(next_book) <= grid_time_ns
            ):
                event_time = _book_observed_time(next_book)
                if previous_book_time is not None and event_time < previous_book_time:
                    raise ValueError("hft_book_time_non_monotonic")
                previous_book_time = event_time
                event = next_book
                next_book = next(active_books, None)
                if isinstance(event, BookBoundary):
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
                current_state = state
                if state.best_bid is not None and state.best_ask is not None:
                    previous_bbo = (state.best_bid, state.best_ask)
            while next_trade is not None:
                if not isinstance(next_trade, TradeEvent):
                    raise TypeError("hft_trade_event_invalid")
                if next_trade.observed_time_ns > grid_time_ns:
                    break
                trade_time = next_trade.observed_time_ns
                if previous_trade_time is not None and trade_time < previous_trade_time:
                    raise ValueError("hft_trade_time_non_monotonic")
                previous_trade_time = trade_time
                trade = next_trade
                next_trade = next(active_trades, None)
                quantity = strict_finite_float(
                    trade.quantity,
                    error="hft_trade_quantity_invalid",
                )
                if trade.visible_quantity is not None:
                    visible = strict_finite_float(
                        trade.visible_quantity,
                        error="hft_trade_visible_quantity_invalid",
                    )
                    if visible < 0 or visible > quantity:
                        raise ValueError("hft_trade_values_invalid")
                if quantity <= 0 or not isinstance(trade.buyer_is_maker, bool):
                    raise ValueError("hft_trade_values_invalid")
                if (
                    current_state is not None
                    and len(trade_connections) == 1
                    and trade.observation_id == trade_connections[0]
                    and trade.observed_time_ns > interval_start_ns
                    and trade.observed_time_ns <= grid_time_ns
                ):
                    trade_count += 1
                    trade_volume += quantity
                    trade_source_sequence_min = (
                        trade.source_sequence
                        if trade_source_sequence_min is None
                        else min(trade_source_sequence_min, trade.source_sequence)
                    )
                    trade_source_sequence_max = (
                        trade.source_sequence
                        if trade_source_sequence_max is None
                        else max(trade_source_sequence_max, trade.source_sequence)
                    )
                    if trade.buyer_is_maker:
                        signed_trade_volume -= quantity
                        taker_sell_volume += quantity
                    else:
                        signed_trade_volume += quantity
                        taker_buy_volume += quantity
            book_bounds = (
                None
                if current_state is None
                else book_coverage.get(current_state.segment_id)
            )
            if (
                book_bounds is None
                or book_bounds[0] > interval_start_ns
                or book_bounds[1] <= grid_time_ns
            ):
                # No row can appear before a stream event or book coverage opens.
                next_time_ns = last_grid_ms * _NANOSECONDS_PER_MILLISECOND
                if next_book is not None:
                    next_time_ns = min(next_time_ns, _book_observed_time(next_book))
                if next_trade is not None:
                    next_time_ns = min(next_time_ns, next_trade.observed_time_ns)
                if book_bounds is not None and book_bounds[0] > interval_start_ns:
                    next_time_ns = min(next_time_ns, book_bounds[0] + interval_ns)
                grid_time += (
                    max(0, (next_time_ns - grid_time_ns - 1) // interval_ns)
                    * HFT_GRID_MS
                )
                continue
            if len(trade_connections) != 1:
                continue
            _append_panel_columns(
                buffers,
                current_state,
                grid_time,
                market_id=market_id,
                asset_id=asset_id,
                trade_observation_id=trade_connections[0],
                trade_source_sequence_min=trade_source_sequence_min,
                trade_source_sequence_max=trade_source_sequence_max,
                book_ofi=book_ofi,
                trade_count=trade_count,
                trade_volume=trade_volume,
                signed_trade_volume=signed_trade_volume,
                taker_buy_volume=taker_buy_volume,
                taker_sell_volume=taker_sell_volume,
            )
            if len(buffers["timestamp"]) >= batch_rows:
                _yield_value = _panel_frame(buffers)
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
                buffers = _panel_column_buffers()
        if buffers["timestamp"]:
            _yield_value = _panel_frame(buffers)
            try:
                yield _yield_value
            except GeneratorExit:
                return


def _book_value(event: BookState) -> BookState:
    value = event
    if not isinstance(value, BookState):
        raise TypeError("hft_book_event_invalid")
    return value


def _book_observed_time(
    event: BookState | BookBoundary,
) -> int:
    return event.observed_time_ns


def _panel_frame(buffers):
    for column in (
        "transaction_time_ms",
        "book_receive_monotonic_ns",
        "trade_source_sequence_min",
        "trade_source_sequence_max",
    ):
        buffers[column] = pd.array(buffers[column], dtype="Int64")
    frame = pd.DataFrame(buffers)
    buffers.clear()
    return frame


def _panel_column_buffers() -> dict[str, list[Any]]:
    return {
        column: []
        for column in (
            "market_id",
            "asset_id",
            "timestamp",
            "segment_id",
            "book_segment_id",
            "book_observation_id",
            "trade_observation_id",
            "book_source_sequence",
            "trade_source_sequence_min",
            "trade_source_sequence_max",
            "book_observed_time_ms",
            "exchange_event_time_ms",
            "transaction_time_ms",
            "observed_utc",
            "book_receive_monotonic_ns",
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "mid",
            "book_spread",
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
    trade_observation_id: str,
    trade_source_sequence_min: int | None,
    trade_source_sequence_max: int | None,
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
    bid_price = index_to_price(bids[0][0], state.price_unit)
    bid_quantity = bids[0][1]
    ask_price = index_to_price(asks[0][0], state.price_unit)
    ask_quantity = asks[0][1]
    mid = float((bids[0][0] + asks[0][0]) * Fraction(state.price_unit) / 2)
    spread = index_to_price(asks[0][0] - bids[0][0], state.price_unit)
    bid_depth_1 = sum(quantity for _price, quantity in bids[:1])
    ask_depth_1 = sum(quantity for _price, quantity in asks[:1])
    bid_depth_5 = sum(quantity for _price, quantity in bids[:5])
    ask_depth_5 = sum(quantity for _price, quantity in asks[:5])
    bid_depth_10 = sum(quantity for _price, quantity in bids)
    ask_depth_10 = sum(quantity for _price, quantity in asks)
    buffers["market_id"].append(market_id)
    buffers["asset_id"].append(asset_id)
    buffers["timestamp"].append(_timestamp(grid_time))
    buffers["segment_id"].append(f"{state.segment_id}:{trade_observation_id}")
    buffers["book_segment_id"].append(state.segment_id)
    buffers["book_observation_id"].append(state.observation_id)
    buffers["trade_observation_id"].append(trade_observation_id)
    buffers["book_source_sequence"].append(state.source_sequence)
    buffers["trade_source_sequence_min"].append(trade_source_sequence_min)
    buffers["trade_source_sequence_max"].append(trade_source_sequence_max)
    buffers["book_observed_time_ms"].append(state.observed_time_ms)
    buffers["exchange_event_time_ms"].append(state.event_time_ms)
    buffers["transaction_time_ms"].append(state.transaction_time_ms)
    buffers["observed_utc"].append(pd.Timestamp(state.observed_time_ns, unit="ns"))
    buffers["book_receive_monotonic_ns"].append(state.receive_monotonic_ns)
    buffers["update_id"].append(state.update_id)
    buffers["best_bid_price"].append(bid_price)
    buffers["best_bid_qty"].append(bid_quantity)
    buffers["best_ask_price"].append(ask_price)
    buffers["best_ask_qty"].append(ask_quantity)
    buffers["mid"].append(mid)
    buffers["book_spread"].append(spread)
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


def iter_factor_frames(
    frames: Iterable[pd.DataFrame], *, columns: Iterable[str] | None = None
) -> Iterator[pd.DataFrame]:
    """Derive HFT factors from bounded persisted PANEL frames."""

    selected = set(HFT_FACTOR_COLUMNS if columns is None else columns)
    if not selected <= set(HFT_FACTOR_COLUMNS):
        raise ValueError("hft_factor_columns_invalid")
    needed = set(selected)
    dependencies = {
        "microprice": {"microprice_deviation"},
        "microprice_deviation": {"imbalance_1"},
        "imbalance_1_minus_10": {"imbalance_1", "imbalance_10"},
        "imbalance_5_minus_10": {"imbalance_5", "imbalance_10"},
    }
    for name in (
        "microprice",
        "microprice_deviation",
        "imbalance_1_minus_10",
        "imbalance_5_minus_10",
    ):
        if name in needed:
            needed.update(dependencies[name])
    flow_columns = {
        name
        for name in needed
        if name.startswith(("trade_flow_imbalance_", "depth_normalized_ofi_"))
        or name
        in {
            "ofi_500ms",
            "ofi_1s",
            "ofi_5s",
            "taker_sell_pressure_500ms",
            "taker_buy_pressure_500ms",
        }
    }
    if flow_columns:
        needed.update(
            {
                "ofi_100ms",
                "signed_trade_flow_100ms",
                "taker_buy_volume_100ms",
                "taker_sell_volume_100ms",
            }
        )
    flow_windows = tuple(
        (count, suffix)
        for count, suffix in ((1, "100ms"), (5, "500ms"), (10, "1s"), (50, "5s"))
        if flow_columns
        & (
            {
                f"trade_flow_imbalance_{suffix}",
                f"ofi_{suffix}",
                f"depth_normalized_ofi_{suffix}",
            }
            | (
                {"taker_sell_pressure_500ms", "taker_buy_pressure_500ms"}
                if count == 5
                else set()
            )
        )
    )
    depth_levels = {level for level in (1, 5, 10) if f"imbalance_{level}" in needed}
    if flow_columns or "near_depth_share_1_10" in needed:
        depth_levels.add(1)
    if needed & {"near_depth_share_1_10", "top10_depth_per_spread"}:
        depth_levels.add(10)
    active_segment: str | None = None
    previous_time: int | None = None
    previous_mid: float | None = None
    return_tail: deque[float] = deque(maxlen=10)
    mid_tail: deque[float] = deque(maxlen=6)
    flow_tail: deque[tuple[Fraction, ...]] = deque(maxlen=50)
    flow_totals = dict.fromkeys((count for count, _ in flow_windows), (0,) * 6)
    active_asset = None
    for source in frames:
        if source.empty:
            continue
        frame = _with_timestamp_ms(source.reset_index(drop=True))
        assets = frame["asset_id"].unique()
        if len(assets) != 1:
            raise ValueError("hft_factor_mixed_assets")
        if active_asset != assets[0]:
            active_segment = previous_time = previous_mid = None
            return_tail.clear()
            mid_tail.clear()
            flow_tail.clear()
        active_asset = assets[0]
        timestamps, segments, boundaries = _validated_grid_arrays(
            frame,
            active_segment=active_segment,
            previous_time=previous_time,
        )
        mid = _finite_column(frame, "mid", "hft_factor_mid_invalid")
        spread = _finite_column(frame, "book_spread", "hft_factor_book_invalid")
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
            for levels in depth_levels
        }
        returns = np.full(len(frame), np.nan, dtype="float64")
        same_segment = ~boundaries
        returns[1:][same_segment[1:]] = (
            mid[1:][same_segment[1:]] / mid[:-1][same_segment[1:]] - 1.0
        )
        if not boundaries[0] and previous_mid is not None:
            returns[0] = mid[0] / previous_mid - 1.0
        volatility = np.full(len(frame), np.nan, dtype="float64")
        return_500ms = np.full(len(frame), np.nan, dtype="float64")
        for index, value in (
            enumerate(returns)
            if needed & {"return_500ms", "realized_volatility_1s"}
            else ()
        ):
            if boundaries[index]:
                return_tail.clear()
                mid_tail.clear()
            mid_tail.append(float(mid[index]))
            if len(mid_tail) == 6:
                return_500ms[index] = mid[index] / mid_tail[0] - 1
            if math.isfinite(value):
                return_tail.append(float(value))
            else:
                return_tail.clear()
            if "realized_volatility_1s" in needed and len(return_tail) == 10:
                volatility[index] = stdev(return_tail)
        output = _identity_frame(frame)
        if "spread" in needed:
            output["spread"] = spread
        if "relative_spread" in needed:
            output["relative_spread"] = spread / mid
        for levels in (
            level for level in depth_levels if f"imbalance_{level}" in needed
        ):
            bid_depth, ask_depth = depth[levels]
            imbalance = np.full(len(frame), np.nan, dtype="float64")
            for index, (bid_value, ask_value) in enumerate(
                zip(bid_depth, ask_depth, strict=True)
            ):
                bid = Fraction(float(bid_value))
                ask = Fraction(float(ask_value))
                total = bid + ask
                if total > 0:
                    # Round once; proportional float depths must stay constant.
                    imbalance[index] = float((bid - ask) / total)
            output[f"imbalance_{levels}"] = imbalance
        if "microprice_deviation" in needed:
            microprice_deviation = 0.5 * spread * output["imbalance_1"].to_numpy()
            output["microprice"] = mid + microprice_deviation
            output["microprice_deviation"] = microprice_deviation
        for name, source_name, error in (
            ("ofi_100ms", "book_ofi_100ms", "hft_factor_ofi_invalid"),
            (
                "signed_trade_flow_100ms",
                "observed_signed_trade_volume_100ms",
                "hft_factor_trade_flow_invalid",
            ),
            (
                "taker_buy_volume_100ms",
                "observed_taker_buy_volume_100ms",
                "hft_factor_trade_flow_invalid",
            ),
            (
                "taker_sell_volume_100ms",
                "observed_taker_sell_volume_100ms",
                "hft_factor_trade_flow_invalid",
            ),
        ):
            if name in needed:
                output[name] = _finite_column(frame, source_name, error)
        for name, values in (
            ("return_100ms", returns),
            ("return_500ms", return_500ms),
            ("realized_volatility_1s", volatility),
        ):
            if name in needed:
                output[name] = values
        if "top10_depth_per_spread" in needed:
            output["top10_depth_per_spread"] = [
                float(
                    (Fraction(float(bid)) + Fraction(float(ask)))
                    / Fraction(float(width))
                )
                for bid, ask, width in zip(*depth[10], spread, strict=True)
            ]
        expanded = {
            column: np.full(len(frame), np.nan)
            for column in needed
            if column not in output
        }
        if flow_columns:
            ofi = output["ofi_100ms"].to_numpy()
            signed = output["signed_trade_flow_100ms"].to_numpy()
            buys = output["taker_buy_volume_100ms"].to_numpy()
            sells = output["taker_sell_volume_100ms"].to_numpy()
        for index in range(len(frame)) if flow_columns else ():
            if boundaries[index]:
                flow_tail.clear()
                flow_totals = dict.fromkeys(flow_totals, (0,) * 6)
            bid, ask = depth[1][0][index], depth[1][1][index]
            current = tuple(
                Fraction(float(value))
                for value in (
                    signed[index],
                    buys[index],
                    sells[index],
                    ofi[index],
                    bid,
                    ask,
                )
            )
            for count, totals in flow_totals.items():
                expired = flow_tail[-count] if len(flow_tail) >= count else (0,) * 6
                flow_totals[count] = tuple(
                    total + value - old
                    for total, value, old in zip(totals, current, expired, strict=True)
                )
            flow_tail.append(current)
            for count, suffix in flow_windows:
                if len(flow_tail) < count:
                    continue
                signed_total, buy_total, sell_total, total_ofi, bid_total, ask_total = (
                    flow_totals[count]
                )
                volume = buy_total + sell_total
                if volume > 0 and f"trade_flow_imbalance_{suffix}" in needed:
                    expanded[f"trade_flow_imbalance_{suffix}"][index] = float(
                        signed_total / volume
                    )
                if count > 1:
                    if f"ofi_{suffix}" in needed:
                        expanded[f"ofi_{suffix}"][index] = float(total_ofi)
                    total_depth = bid_total + ask_total
                    if total_depth > 0 and f"depth_normalized_ofi_{suffix}" in needed:
                        expanded[f"depth_normalized_ofi_{suffix}"][index] = float(
                            total_ofi * count / total_depth
                        )
                if count == 5:
                    if bid > 0 and "taker_sell_pressure_500ms" in needed:
                        expanded["taker_sell_pressure_500ms"][index] = float(
                            sell_total / current[4]
                        )
                    if ask > 0 and "taker_buy_pressure_500ms" in needed:
                        expanded["taker_buy_pressure_500ms"][index] = float(
                            buy_total / current[5]
                        )
        if "near_depth_share_1_10" in needed:
            for index, (bid1, ask1, bid10, ask10) in enumerate(
                zip(*depth[1], *depth[10], strict=True)
            ):
                total10 = Fraction(float(bid10)) + Fraction(float(ask10))
                if total10 > 0:
                    expanded["near_depth_share_1_10"][index] = float(
                        (Fraction(float(bid1)) + Fraction(float(ask1))) / total10
                    )
        for level in (1, 5):
            if f"imbalance_{level}_minus_10" in needed:
                expanded[f"imbalance_{level}_minus_10"] = (
                    output[f"imbalance_{level}"] - output["imbalance_10"]
                )
        output = output.assign(**expanded)
        yield output.loc[
            :,
            [name for name in output if name not in HFT_FACTOR_COLUMNS]
            + [name for name in HFT_FACTOR_COLUMNS if name in selected],
        ]
        active_segment = str(segments[-1])
        previous_time = int(timestamps[-1])
        previous_mid = float(mid[-1])


def iter_derivative_factor_frames(frames, *, observations, max_age_ms):
    """Align native perpetual states to one asset's causal, complete 100ms grid."""
    active_segment = previous_time = active_epoch = None
    states = {}
    liquidations = deque()
    oi_values = [np.nan] * 3001
    invalid_prefix = [0] * 3001
    count = invalid = 0
    ratio_fields = (
        (
            "topLongShortPositionRatio",
            "top_position_long_short_ratio",
            "longShortRatio",
        ),
        ("topLongShortAccountRatio", "top_account_long_short_ratio", "longShortRatio"),
        (
            "globalLongShortAccountRatio",
            "global_account_long_short_ratio",
            "longShortRatio",
        ),
        ("takerlongshortRatio", "taker_long_short_ratio", "buySellRatio"),
    )
    with closing_iterator(iter(observations)) as events:
        event = next(events, None)
        for source in frames:
            if source.empty:
                continue
            frame = _with_timestamp_ms(source.reset_index(drop=True))
            timestamps, segments, boundaries = _validated_grid_arrays(
                frame, active_segment=active_segment, previous_time=previous_time
            )
            if frame.asset_id.nunique() != 1:
                raise ValueError("hft_derivative_single_asset_required")
            result = frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
            values = {
                key: (
                    np.full(
                        len(frame), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
                    )
                    if key.endswith(("_observed_at", "_event_at"))
                    else np.full(len(frame), np.nan)
                )
                for key in (
                    *HFT_DERIVATIVE_FACTOR_COLUMNS,
                    *HFT_DERIVATIVE_CONTEXT_COLUMNS,
                )
            }
            for row, stamp in enumerate(timestamps):
                decision_ns = int(stamp) * 1_000_000
                if boundaries[row]:
                    liquidations.clear()
                    count = invalid = 0
                while event is not None and event[0] <= decision_ns:
                    observed_ns, channel, payload, epoch = event
                    if active_epoch is not None and epoch != active_epoch:
                        states.clear()
                        liquidations.clear()
                        count = invalid = 0
                    active_epoch = epoch
                    if channel is None:
                        event = next(events, None)
                        continue
                    states[channel] = (observed_ns, payload)
                    if channel == "forceOrder":
                        # z is observed cumulative filled base quantity, not order q.
                        order = payload["o"]
                        quantity = float(order["z"])
                        if observed_ns > decision_ns - 30_000_000_000 and (
                            count or observed_ns > decision_ns - 100_000_000
                        ):
                            liquidations.append(
                                (
                                    observed_ns,
                                    Fraction(
                                        quantity if order["S"] == "BUY" else -quantity
                                    ),
                                )
                            )
                    event = next(events, None)
                fresh = {}
                for channel, (observed_ns, payload) in states.items():
                    age = (decision_ns - observed_ns) / 1_000_000
                    values[channel + "_age_ms"][row] = age
                    values[channel + "_observed_at"][row] = np.datetime64(
                        observed_ns, "ns"
                    )
                    event_ms = (
                        payload["E"]
                        if channel == "forceOrder"
                        else payload["time"]
                        if channel in {"premiumIndex", "openInterest"}
                        else payload["timestamp"]
                    )
                    values[channel + "_event_at"][row] = np.datetime64(
                        int(event_ms) * 1_000_000, "ns"
                    )
                    if age <= max_age_ms[channel]:
                        fresh[channel] = payload
                premium = fresh.get("premiumIndex")
                if premium is not None:
                    mark, index, mid = (
                        float(premium["markPrice"]),
                        float(premium["indexPrice"]),
                        float(frame["mid"].iloc[row]),
                    )
                    values["mark_mid_basis"][row] = mark / mid - 1
                    values["mark_index_basis"][row] = mark / index - 1
                    values["perp_index_basis"][row] = mid / index - 1
                    values["funding_rate"][row] = float(premium["lastFundingRate"])
                    seconds = (
                        int(premium["nextFundingTime"]) * 1_000_000 - decision_ns
                    ) / 1_000_000_000
                    if seconds >= 0:
                        values["time_to_next_funding"][row] = seconds
                oi = (
                    float(fresh["openInterest"]["openInterest"])
                    if "openInterest" in fresh
                    else np.nan
                )
                if oi > 0:
                    values["log_open_interest"][row] = math.log(oi)
                else:
                    oi = np.nan
                invalid += int(not np.isfinite(oi))
                oi_values[count % 3001] = oi
                invalid_prefix[count % 3001] = invalid
                for lag, suffix in ((300, "30s"), (600, "1m"), (3000, "5m")):
                    old = (count - lag) % 3001
                    if (
                        count >= lag
                        and invalid == invalid_prefix[old]
                        and np.isfinite(oi_values[old])
                    ):
                        values["open_interest_change_" + suffix][row] = (
                            oi / oi_values[old] - 1
                        )
                while (
                    liquidations and liquidations[0][0] <= decision_ns - 30_000_000_000
                ):
                    liquidations.popleft()
                for seconds, suffix in ((5, "5s"), (30, "30s")):
                    if "forceOrder" in fresh and count + 1 >= seconds * 10:
                        window = [
                            quantity
                            for observed, quantity in liquidations
                            if observed > decision_ns - seconds * 1_000_000_000
                        ]
                        total = sum(abs(quantity) for quantity in window)
                        if total > 0:
                            values["observed_force_order_imbalance_" + suffix][row] = (
                                float(sum(window) / total)
                            )
                if "forceOrder" in fresh:
                    values["time_since_observed_force_order"][row] = (
                        decision_ns - states["forceOrder"][0]
                    ) / 1_000_000_000
                for channel, name, field in ratio_fields:
                    if channel in fresh:
                        values[name][row] = float(fresh[channel][field])
                count += 1
            _yield_value = result.assign(**values)
            try:
                yield _yield_value
            except GeneratorExit:
                return
            active_segment, previous_time = str(segments[-1]), int(timestamps[-1])


def _microstructure_context_rows(frames):
    """Reuse the microstructure owner; retain only rows needed for causal context."""
    with closing_iterator(iter(frames)) as source:
        panels, inputs = tee(frame for frame in source if not frame.empty)
        with closing_iterator(
            iter_factor_frames(
                inputs,
                columns=(
                    "return_100ms",
                    "ofi_500ms",
                    "depth_normalized_ofi_500ms",
                    "trade_flow_imbalance_500ms",
                    "return_500ms",
                ),
            )
        ) as factors:
            for panel, calculated in zip(panels, factors, strict=True):
                times = (
                    pd.to_datetime(panel["timestamp"])
                    .to_numpy(dtype="datetime64[ns]")
                    .astype("int64")
                )
                observed = (
                    pd.to_datetime(panel["observed_utc"], utc=True)
                    .to_numpy(dtype="datetime64[ns]")
                    .astype("int64")
                )
                if bool(np.any(observed > times)):
                    raise ValueError("hft_information_observed_after_grid")
                columns = calculated[
                    [
                        "return_100ms",
                        "ofi_500ms",
                        "depth_normalized_ofi_500ms",
                        "trade_flow_imbalance_500ms",
                        "return_500ms",
                    ]
                ].to_numpy(dtype="float64")
                for timestamp, mid, book_time, segment, value in zip(
                    times,
                    panel["mid"].to_numpy(dtype="float64"),
                    observed,
                    panel["segment_id"],
                    columns,
                    strict=True,
                ):
                    _yield_value = {
                        "timestamp": int(timestamp),
                        "mid": float(mid),
                        "return_100ms": value[0],
                        "return_500ms": value[4],
                        "ofi_500ms": value[1],
                        "normalized_ofi_500ms": value[2],
                        "tfi_500ms": value[3],
                        "book_observed_at": int(book_time),
                        "segment": segment,
                    }
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return


def iter_spot_perp_factor_frames(
    perpetual_frames, *, spot_frames, max_age_ms, batch_rows
):
    """Causally align one information asset without changing the execution grid."""
    last = previous = None
    records = []
    with (
        closing_iterator(_microstructure_context_rows(perpetual_frames)) as perpetual,
        closing_iterator(_microstructure_context_rows(spot_frames)) as spot,
    ):
        upcoming = next(spot, None)
        for point in perpetual:
            stamp = point["timestamp"]
            while upcoming is not None and upcoming["timestamp"] <= stamp:
                previous, last = last, upcoming
                upcoming = next(spot, None)
            row = {key: np.nan for key in HFT_SPOT_FACTOR_COLUMNS}
            row.update(
                {
                    key: pd.NaT if key.endswith("_at") else np.nan
                    for key in HFT_SPOT_CONTEXT_COLUMNS
                }
            )
            row["timestamp"] = pd.Timestamp(stamp)
            if last is not None:
                grid_age = (stamp - last["timestamp"]) / 1_000_000
                book_age = (stamp - last["book_observed_at"]) / 1_000_000
                row.update(
                    spot_grid_age_ms=grid_age,
                    spot_book_age_ms=book_age,
                    spot_grid_at=pd.Timestamp(last["timestamp"]),
                    spot_book_observed_at=pd.Timestamp(last["book_observed_at"]),
                )
                if max(grid_age, book_age) <= max_age_ms:
                    row.update(
                        spot_perp_basis=last["mid"] / point["mid"] - 1,
                        spot_return_100ms=last["return_100ms"],
                        spot_return_500ms=last["return_500ms"],
                        spot_ofi_500ms=last["ofi_500ms"],
                        spot_tfi_500ms=last["tfi_500ms"],
                        spot_perp_return_divergence_100ms=last["return_100ms"]
                        - point["return_100ms"],
                        spot_perp_return_divergence_500ms=last["return_500ms"]
                        - point["return_500ms"],
                        spot_perp_normalized_ofi_divergence_500ms=last[
                            "normalized_ofi_500ms"
                        ]
                        - point["normalized_ofi_500ms"],
                        spot_perp_tfi_divergence_500ms=last["tfi_500ms"]
                        - point["tfi_500ms"],
                    )
                    if (
                        previous is not None
                        and previous["segment"] == last["segment"]
                        and last["timestamp"] - previous["timestamp"] == 100_000_000
                        and stamp - previous["book_observed_at"]
                        <= max_age_ms * 1_000_000
                    ):
                        row.update(
                            lagged_spot_return_100ms=previous["return_100ms"],
                            lagged_spot_ofi_500ms=previous["ofi_500ms"],
                            lagged_spot_tfi_500ms=previous["tfi_500ms"],
                        )
            records.append(row)
            if len(records) == batch_rows:
                _yield_value = pd.DataFrame.from_records(records)
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
                records.clear()
        if records:
            _yield_value = pd.DataFrame.from_records(records)
            try:
                yield _yield_value
            except GeneratorExit:
                return


def iter_cross_asset_factor_frames(
    target_frames, *, information_frames, max_age_ms, batch_rows
):
    """Align matured, dimensionless asset states to one target asset's grid."""

    def rows(frames):
        with closing_iterator(iter(frames)) as source:
            for frame in source:
                for row in frame.itertuples(index=False):
                    _yield_value = (
                        pd.Timestamp(row.timestamp).value,
                        row.return_100ms,
                        row.return_500ms,
                        row.depth_normalized_ofi_500ms,
                        row.trade_flow_imbalance_500ms,
                    )
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return

    records = []
    with ExitStack() as readers:
        sources = {
            asset: readers.enter_context(closing_iterator(rows(frames)))
            for asset, frames in information_frames.items()
        }
        upcoming = {asset: next(stream, None) for asset, stream in sources.items()}
        current, previous = {}, {}
        frames = readers.enter_context(closing_iterator(iter(target_frames)))
        for frame in frames:
            for timestamp in frame["timestamp"]:
                stamp = pd.Timestamp(timestamp).value
                row = {key: np.nan for key in HFT_CROSS_ASSET_FACTOR_COLUMNS}
                row.update(
                    {
                        key: pd.NaT if key.endswith("_at") else np.nan
                        for key in HFT_CROSS_ASSET_CONTEXT_COLUMNS
                    }
                )
                row["timestamp"] = pd.Timestamp(stamp)
                for asset, stream in sources.items():
                    point = upcoming[asset]
                    while point is not None and point[0] <= stamp:
                        previous[asset] = current.get(asset)
                        current[asset] = point
                        point = next(stream, None)
                    upcoming[asset] = point
                    last = current.get(asset)
                    if last is None:
                        continue
                    prefix = asset.removesuffix("_usdt") + "_"
                    age = (stamp - last[0]) / 1_000_000
                    row[prefix + "grid_age_ms"] = age
                    row[prefix + "grid_at"] = pd.Timestamp(last[0])
                    if age > max_age_ms:
                        continue
                    for name, value in zip(
                        (
                            "return_100ms",
                            "return_500ms",
                            "normalized_ofi_500ms",
                            "tfi_500ms",
                        ),
                        last[1:],
                        strict=True,
                    ):
                        row[prefix + name] = value
                    prior = previous.get(asset)
                    if (
                        prior is not None
                        and last[0] - prior[0] == 100_000_000
                        and math.isfinite(last[1])
                        and stamp - prior[0] <= max_age_ms * 1_000_000
                    ):
                        row[prefix + "lagged_return_100ms"] = prior[1]
                        row[prefix + "lagged_tfi_500ms"] = prior[4]
                records.append(row)
                if len(records) == batch_rows:
                    _yield_value = pd.DataFrame.from_records(records)
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return
                    records.clear()
        if records:
            _yield_value = pd.DataFrame.from_records(records)
            try:
                yield _yield_value
            except GeneratorExit:
                return


def iter_external_context_frames(
    frames, *, quotes, trades, tickers, venue, base_units_per_amount, max_age_ms
):
    """Align observed external quotes and base/notional trade flow to Binance time."""
    active_segment = previous_time = None
    mids = deque(maxlen=6)
    bins = deque(maxlen=5)
    count = 0
    latest_quote = latest_trade = latest_ticker = None
    if venue == "okx":
        before = Fraction(str(base_units_per_amount["before"]))
        after = Fraction(str(base_units_per_amount["after"]))
        transition_start = (
            pd.Timestamp(base_units_per_amount["transition_start"]).value // 1000
        )
        transition_end = (
            pd.Timestamp(base_units_per_amount["transition_end"]).value // 1000
        )
    else:
        multiplier = Fraction(str(base_units_per_amount))
    with (
        closing_iterator(iter(quotes)) as quote_rows,
        closing_iterator(iter(trades)) as trade_rows,
        closing_iterator(iter(tickers)) as ticker_rows,
    ):
        quote = next(quote_rows, None)
        trade = next(trade_rows, None)
        ticker = next(ticker_rows, None)
        for source in frames:
            if source.empty:
                continue
            frame = _with_timestamp_ms(source.reset_index(drop=True))
            timestamps, segments, boundaries = _validated_grid_arrays(
                frame,
                active_segment=active_segment,
                previous_time=previous_time,
            )
            records = []
            for index, stamp in enumerate(timestamps):
                decision_us = int(stamp) * 1000
                if boundaries[index]:
                    mids.clear()
                    bins.clear()
                    count = 0
                while quote is not None and quote["local_timestamp"] <= decision_us:
                    latest_quote = quote
                    quote = next(quote_rows, None)
                while ticker is not None and ticker["local_timestamp"] <= decision_us:
                    latest_ticker = ticker
                    ticker = next(ticker_rows, None)
                buy = sell = buy_notional = sell_notional = Fraction()
                while trade is not None and trade["local_timestamp"] <= decision_us:
                    latest_trade = trade
                    if trade["local_timestamp"] > decision_us - 100_000:
                        if venue == "okx":
                            event_time = trade["timestamp"]
                            if transition_start <= event_time < transition_end:
                                raise ValueError(
                                    "external_contract_face_value_transition_unproven"
                                )
                            multiplier = (
                                before if event_time < transition_start else after
                            )
                        quantity = Fraction(trade["amount"]) * multiplier
                        notional = quantity * Fraction(trade["price"])
                        if trade["side"] == "buy":
                            buy += quantity
                            buy_notional += notional
                        else:
                            sell += quantity
                            sell_notional += notional
                    trade = next(trade_rows, None)
                bins.append((buy, sell, buy_notional, sell_notional))
                count += 1
                row = {name: np.nan for name in HFT_EXTERNAL_FIELDS}
                row.update(
                    {
                        name: pd.NaT if name.endswith("_at") else np.nan
                        for name in HFT_EXTERNAL_CLOCK_FIELDS
                    }
                )
                for name, observation in (
                    ("quote", latest_quote),
                    ("trade", latest_trade),
                    ("ticker", latest_ticker),
                ):
                    if observation is not None:
                        row[name + "_age_ms"] = (
                            decision_us - observation["local_timestamp"]
                        ) / 1000
                        row[name + "_observed_at"] = pd.Timestamp(
                            observation["local_timestamp"], unit="us"
                        )
                        row[name + "_event_at"] = pd.Timestamp(
                            observation["timestamp"], unit="us"
                        )
                bid = None if latest_quote is None else latest_quote["bid_price"]
                ask = None if latest_quote is None else latest_quote["ask_price"]
                if (
                    bid is not None
                    and ask is not None
                    and 0 < bid < ask
                    and row["quote_age_ms"] <= max_age_ms
                ):
                    mid = bid / 2 + ask / 2
                    mids.append(mid)
                    binance_mid = float(frame["mid"].iloc[index])
                    row.update(
                        mid_usdt=mid,
                        spread_usdt=ask - bid,
                        price_difference=mid / binance_mid - 1,
                        relative_mid_dispersion=abs(mid / 2 - binance_mid / 2)
                        / (mid / 2 + binance_mid / 2),
                    )
                    if len(mids) >= 2:
                        row["return_100ms"] = mid / mids[-2] - 1
                    if len(mids) == 6:
                        row["return_500ms"] = mid / mids[0] - 1
                else:
                    mids.clear()
                if count >= 6:
                    total_buy = sum((value[0] for value in bins), Fraction())
                    total_sell = sum((value[1] for value in bins), Fraction())
                    total = total_buy + total_sell
                    if total > 0:
                        signed = total_buy - total_sell
                        signed_notional = sum(
                            (value[2] - value[3] for value in bins), Fraction()
                        )
                        for name, value in (
                            ("observed_signed_base_flow_500ms", signed),
                            ("observed_signed_notional_flow_500ms", signed_notional),
                            ("observed_tfi_500ms", signed / total),
                        ):
                            try:
                                row[name] = float(value)
                            except OverflowError:
                                pass
                for name in HFT_EXTERNAL_FIELDS:
                    if not math.isfinite(row[name]):
                        row[name] = np.nan
                records.append(
                    {venue + "_" + name: value for name, value in row.items()}
                )
            result = _identity_frame(frame)
            _yield_value = pd.concat(
                (result, pd.DataFrame.from_records(records)), axis=1
            )
            try:
                yield _yield_value
            except GeneratorExit:
                return
            active_segment, previous_time = str(segments[-1]), int(timestamps[-1])


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
        available = (
            pd.to_datetime(frame["timestamp_ms"], unit="ms")
            .groupby(segment_groups, sort=False)
            .shift(-(horizon // HFT_GRID_MS))
            .where(values.notna())
        )
        for kind in ("return", "direction"):
            output[f"future_mid_{kind}_{horizon}ms_available_at"] = available
    proxy = label_catalog().get(HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN)
    if proxy.horizon in horizons:
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
        steps = proxy.horizon // HFT_GRID_MS
        future_sell_volume = pd.concat(
            tuple(grouped_sell.shift(-step) for step in range(1, steps + 1)),
            axis=1,
        ).sum(axis=1, min_count=steps)
        output[HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN] = (
            future_sell_volume / best_bid_quantity
        ).clip(lower=0.0, upper=1.0)
        output[f"{HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN}_available_at"] = (
            pd.to_datetime(frame["timestamp_ms"], unit="ms")
            .groupby(segment_groups, sort=False)
            .shift(-steps)
            .where(future_sell_volume.notna())
        )
    return output
