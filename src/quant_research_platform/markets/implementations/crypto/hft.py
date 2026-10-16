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
    TradeBoundary,
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
    factor_catalog,
    label_catalog,
    spot_dependencies,
)

HFT_GRID_MS = 100
_NANOSECONDS_PER_MILLISECOND = 1_000_000


def _quantity_units(value):
    # Every finite binary64 quantity is an integer multiple of 2**-1074.
    numerator, denominator = value.as_integer_ratio()
    return numerator << (1074 - (denominator.bit_length() - 1))


def _ofi_units(previous_bid, current_bid, previous_ask, current_ask):
    result = 0
    if current_bid[0] >= previous_bid[0]:
        result += current_bid[1]
    if current_bid[0] <= previous_bid[0]:
        result -= previous_bid[1]
    if current_ask[0] <= previous_ask[0]:
        result -= current_ask[1]
    if current_ask[0] >= previous_ask[0]:
        result += previous_ask[1]
    return result


def iter_panel_frames(
    book_events: Iterable[BookState | BookBoundary],
    trades: Iterable[TradeEvent | TradeBoundary],
    *,
    market_id: str,
    asset_id: str,
    start_ms: int,
    end_ms: int,
    book_coverage: Mapping[str, tuple[int, int]],
    trade_coverage: Callable[[int, int], tuple[str, ...]],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """Keep covered book grids; unknown trade bins remain nullable."""

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
        next_trade: TradeEvent | TradeBoundary | None = None
        current_state: BookState | None = None
        previous_bbo: tuple[tuple[int, int], tuple[int, int]] | None = None
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
            book_ofi = 0
            trade_count = 0
            valid_trade_bin = True
            trade_volume = 0
            signed_trade_volume = 0
            taker_buy_volume = 0
            taker_sell_volume = 0
            trade_source_sequence_min: int | None = None
            trade_source_sequence_max: int | None = None
            # Trade sums consume (start, end], unlike the book baseline at start.
            trade_start_ns = min(grid_time_ns, interval_start_ns + 1)
            trade_connections = trade_coverage(trade_start_ns, grid_time_ns)
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
                    book_ofi = 0
                    continue
                state = _book_value(event)
                if (
                    current_state is None
                    or state.segment_id != current_state.segment_id
                ):
                    previous_bbo = None
                    book_ofi = 0
                current_bbo = (
                    (
                        (state.best_bid[0], _quantity_units(state.best_bid[1])),
                        (state.best_ask[0], _quantity_units(state.best_ask[1])),
                    )
                    if state.best_bid is not None and state.best_ask is not None
                    else None
                )
                if (
                    previous_bbo is not None
                    and current_bbo is not None
                    and event_time > interval_start_ns
                ):
                    book_ofi += _ofi_units(
                        previous_bbo[0],
                        current_bbo[0],
                        previous_bbo[1],
                        current_bbo[1],
                    )
                current_state = state
                if current_bbo is not None:
                    previous_bbo = current_bbo
            while next_trade is not None:
                if not isinstance(next_trade, (TradeEvent, TradeBoundary)):
                    raise TypeError("hft_trade_event_invalid")
                if next_trade.observed_time_ns > grid_time_ns:
                    break
                trade_time = next_trade.observed_time_ns
                if previous_trade_time is not None and trade_time < previous_trade_time:
                    raise ValueError("hft_trade_time_non_monotonic")
                previous_trade_time = trade_time
                trade = next_trade
                next_trade = next(active_trades, None)
                if isinstance(trade, TradeBoundary):
                    if trade_start_ns <= trade_time <= grid_time_ns:
                        valid_trade_bin = False
                    continue
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
                    quantity = _quantity_units(quantity)
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
            # The stream may have just consumed a local invalid-observation
            # boundary; refresh the same causal interval after that event.
            trade_connections = trade_coverage(trade_start_ns, grid_time_ns)
            trades_known = len(trade_connections) == 1 and valid_trade_bin
            _append_panel_columns(
                buffers,
                current_state,
                grid_time,
                market_id=market_id,
                asset_id=asset_id,
                trade_observation_id=trade_connections[0] if trades_known else None,
                trade_source_sequence_min=trade_source_sequence_min
                if trades_known
                else None,
                trade_source_sequence_max=trade_source_sequence_max
                if trades_known
                else None,
                book_ofi=_quantity_float(book_ofi),
                trade_count=trade_count if trades_known else None,
                trade_volume=_quantity_float(trade_volume) if trades_known else np.nan,
                signed_trade_volume=_quantity_float(signed_trade_volume)
                if trades_known
                else np.nan,
                taker_buy_volume=_quantity_float(taker_buy_volume)
                if trades_known
                else np.nan,
                taker_sell_volume=_quantity_float(taker_sell_volume)
                if trades_known
                else np.nan,
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


def _quantity_float(units):
    """Only an unrepresentable derived total is unknown, not its source rows."""
    try:
        return units / (1 << 1074)
    except OverflowError:
        return np.nan


def _panel_frame(buffers):
    for column in (
        "transaction_time_ms",
        "book_receive_monotonic_ns",
        "trade_source_sequence_min",
        "trade_source_sequence_max",
        "trade_count_100ms",
    ):
        buffers[column] = pd.array(buffers[column], dtype="Int64")
    buffers["trade_observation_id"] = pd.array(
        buffers["trade_observation_id"], dtype="string"
    )
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
    trade_observation_id: str | None,
    trade_source_sequence_min: int | None,
    trade_source_sequence_max: int | None,
    book_ofi: float,
    trade_count: int | None,
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
    buffers["segment_id"].append(state.segment_id)
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
    buffers["bid_depth_5"].append(bid_depth_5 if math.isfinite(bid_depth_5) else np.nan)
    buffers["ask_depth_5"].append(ask_depth_5 if math.isfinite(ask_depth_5) else np.nan)
    buffers["bid_depth_10"].append(
        bid_depth_10 if math.isfinite(bid_depth_10) else np.nan
    )
    buffers["ask_depth_10"].append(
        ask_depth_10 if math.isfinite(ask_depth_10) else np.nan
    )
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
    """Compute each selected definition from its own inputs and complete windows."""
    selected = set(HFT_FACTOR_COLUMNS if columns is None else columns)
    if not selected <= set(HFT_FACTOR_COLUMNS):
        raise ValueError("hft_factor_columns_invalid")
    definitions = {name: factor_catalog().get(name).input_columns for name in selected}
    required = tuple(
        dict.fromkeys(name for names in definitions.values() for name in names)
    )
    rolling = {
        name: {"100ms": 1, "500ms": 5, "1s": 10, "5s": 50}[name.rsplit("_", 1)[1]]
        for name in selected
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
    # Only aggregate the channels consumed by this definition. Missing samples
    # keep their grid positions, so recovery cannot shrink a market-time window.
    rolling_inputs = {
        name: tuple(
            column
            for column in definitions[name]
            if not ("pressure" in name and column.endswith("depth_1"))
        )
        for name in rolling
    }
    histories = {
        column: deque(maxlen=50)
        for columns in rolling_inputs.values()
        for column in columns
    }
    windows = {
        (column, count)
        for name, count in rolling.items()
        for column in rolling_inputs[name]
    }
    totals = {key: Fraction() for key in windows}
    missing = dict.fromkeys(windows, 0)
    mids = deque(maxlen=6)
    returns = deque(maxlen=10)
    previous_mid = np.nan
    active_segment = previous_time = active_asset = None
    for source in frames:
        if source.empty:
            continue
        frame = _with_timestamp_ms(source.reset_index(drop=True))
        assets = frame["asset_id"].unique()
        if len(assets) != 1:
            raise ValueError("hft_factor_mixed_assets")
        if active_asset != assets[0]:
            active_segment = previous_time = None
        active_asset = assets[0]
        timestamps, segments, boundaries = _validated_grid_arrays(
            frame, active_segment=active_segment, previous_time=previous_time
        )
        inputs = {name: _nullable_numeric_column(frame, name) for name in required}
        output = _identity_frame(frame)
        calculated = {name: np.full(len(frame), np.nan) for name in selected}
        for index in range(len(frame)):
            if boundaries[index]:
                for history in histories.values():
                    history.clear()
                totals = {key: Fraction() for key in windows}
                missing = dict.fromkeys(windows, 0)
                mids.clear()
                returns.clear()
                previous_mid = np.nan
            current = {name: float(values[index]) for name, values in inputs.items()}
            exact = {
                name: Fraction(value) if math.isfinite(value) else None
                for name, value in current.items()
            }
            for (name, count), total in totals.items():
                history = histories[name]
                old = history[-count] if len(history) >= count else Fraction()
                value = exact[name]
                totals[name, count] = total + (value or 0) - (old or 0)
                missing[name, count] += int(value is None) - int(old is None)
            for name, history in histories.items():
                history.append(exact[name])
            mid = current.get("mid", np.nan)
            with np.errstate(over="ignore", invalid="ignore"):
                step_return = (
                    (mid - previous_mid) / previous_mid
                    if previous_mid > 0 and mid > 0
                    else np.nan
                )
            previous_mid = mid
            if mid > 0:
                mids.append(mid)
            else:
                mids.clear()
            if math.isfinite(step_return):
                returns.append(step_return)
            else:
                returns.clear()
            for name, dependencies in definitions.items():
                result = np.nan
                values = [exact[column] for column in dependencies]
                if any(value is None for value in values):
                    continue
                if name in rolling:
                    count = rolling[name]
                    accumulated = rolling_inputs[name]
                    if any(
                        len(histories[column]) < count or missing[column, count]
                        for column in accumulated
                    ):
                        continue
                    sums = {column: totals[column, count] for column in accumulated}
                    if name.startswith("trade_flow_imbalance_"):
                        total = (
                            sums["observed_taker_buy_volume_100ms"]
                            + sums["observed_taker_sell_volume_100ms"]
                        )
                        if total > 0:
                            result = sums["observed_signed_trade_volume_100ms"] / total
                    elif name.startswith("depth_normalized_ofi_"):
                        total = sums["bid_depth_1"] + sums["ask_depth_1"]
                        if total > 0:
                            result = sums["book_ofi_100ms"] * count / total
                    elif name.startswith("ofi_"):
                        result = sums["book_ofi_100ms"]
                    else:
                        volume, depth = dependencies
                        if exact[depth] > 0:
                            result = sums[volume] / exact[depth]
                elif name == "spread":
                    result = values[0]
                elif name == "relative_spread":
                    result = values[0] / values[1]
                elif name.startswith("imbalance_") or name in {
                    "microprice",
                    "microprice_deviation",
                }:
                    bid, ask = values[:2]
                    if bid + ask > 0:
                        result = (bid - ask) / (bid + ask)
                        if "minus" in name:
                            bid10, ask10 = values[2:]
                            result = (
                                result - (bid10 - ask10) / (bid10 + ask10)
                                if bid10 + ask10 > 0
                                else np.nan
                            )
                        elif name in {"microprice", "microprice_deviation"}:
                            result *= exact["book_spread"] / 2
                            if name == "microprice":
                                result += exact["mid"]
                elif name == "top10_depth_per_spread":
                    result = (values[0] + values[1]) / values[2]
                elif name == "near_depth_share_1_10":
                    if values[2] + values[3] > 0:
                        result = (values[0] + values[1]) / (values[2] + values[3])
                elif name == "return_100ms":
                    result = step_return
                elif name == "return_500ms":
                    if len(mids) == 6:
                        result = (mid - mids[0]) / mids[0]
                elif name == "realized_volatility_1s":
                    if len(returns) == 10:
                        result = stdev(returns)
                else:
                    result = values[0]
                try:
                    value = float(result)
                except OverflowError:
                    continue
                if math.isfinite(value):
                    calculated[name][index] = value
        yield output.assign(
            **{
                name: calculated[name]
                for name in HFT_FACTOR_COLUMNS
                if name in selected
            }
        )
        active_segment = str(segments[-1])
        previous_time = int(timestamps[-1])


def _nullable_numeric_column(frame, column, *, signed_mid=False):
    """Localize identifiable non-finite/domain-invalid observations to their cell."""
    if column not in frame:
        return np.full(len(frame), np.nan)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan, copy=True
    )
    valid = np.isfinite(values)
    valid &= ~frame[column].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).to_numpy(dtype=bool)
    if column == "mid" and signed_mid:
        valid &= values != 0
    elif column in {"mid", "book_spread", "best_bid_qty"}:
        valid &= values > 0
    elif column.startswith(("bid_depth_", "ask_depth_", "observed_taker_")):
        valid &= values >= 0
    values[~valid] = np.nan
    return values


def iter_derivative_factor_frames(
    frames, *, observations, max_age_ms, input_delay_ms=0, columns=None
):
    """Align native perpetual states to one asset's causal, complete 100ms grid."""
    selected = set(HFT_DERIVATIVE_FACTOR_COLUMNS if columns is None else columns)
    active_segment = previous_time = active_epoch = None
    flow_start_ns = None
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
                information_ns = decision_ns - input_delay_ms * 1_000_000
                if boundaries[row]:
                    liquidations.clear()
                    count = invalid = 0
                    flow_start_ns = decision_ns - 100_000_000
                while event is not None and event[0] <= information_ns:
                    observed_ns, channel, payload, epoch = event
                    if epoch != active_epoch and (
                        active_epoch is not None or channel is None
                    ):
                        states.clear()
                        liquidations.clear()
                        count = invalid = 0
                        flow_start_ns = max(flow_start_ns, observed_ns)
                    active_epoch = epoch
                    if channel is None:
                        event = next(events, None)
                        continue
                    states[channel] = (observed_ns, payload)
                    if channel == "forceOrder":
                        # z is observed cumulative filled base quantity, not order q.
                        order = payload["o"]
                        quantity = _state_number(order.get("z"), nonnegative=True)
                        if not math.isfinite(quantity) or order.get("S") not in {
                            "BUY",
                            "SELL",
                        }:
                            liquidations.clear()
                            flow_start_ns = max(flow_start_ns, observed_ns)
                            event = next(events, None)
                            continue
                        if observed_ns > information_ns - 30_000_000_000 and (
                            count or observed_ns > information_ns - 100_000_000
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
                    if age <= max_age_ms.get(channel, -1):
                        fresh[channel] = payload
                premium = fresh.get("premiumIndex")
                if premium is not None:
                    mark = _state_number(premium.get("markPrice"), positive=True)
                    index_price = _state_number(
                        premium.get("indexPrice"), positive=True
                    )
                    mid = (
                        _state_number(frame["mid"].iloc[row], positive=True)
                        if selected & {"mark_mid_basis", "perp_index_basis"}
                        and "mid" in frame
                        else np.nan
                    )
                    values["mark_mid_basis"][row] = (mark - mid) / mid
                    values["mark_index_basis"][row] = (mark - index_price) / index_price
                    values["perp_index_basis"][row] = (mid - index_price) / index_price
                    values["funding_rate"][row] = _state_number(
                        premium.get("lastFundingRate")
                    )
                    next_funding = premium.get("nextFundingTime")
                    if next_funding is not None:
                        seconds = (
                            int(next_funding) * 1_000_000 - decision_ns
                        ) / 1_000_000_000
                        if seconds >= 0:
                            values["time_to_next_funding"][row] = seconds
                oi = _state_number(
                    fresh.get("openInterest", {}).get("openInterest"), positive=True
                )
                if oi > 0:
                    values["log_open_interest"][row] = math.log(oi)
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
                            oi - oi_values[old]
                        ) / oi_values[old]
                while (
                    liquidations
                    and liquidations[0][0] <= information_ns - 30_000_000_000
                ):
                    liquidations.popleft()
                for seconds, suffix in ((5, "5s"), (30, "30s")):
                    if (
                        "forceOrder" in fresh
                        and information_ns - flow_start_ns >= seconds * 1_000_000_000
                    ):
                        window = [
                            quantity
                            for observed, quantity in liquidations
                            if observed > information_ns - seconds * 1_000_000_000
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
                        values[name][row] = _state_number(
                            fresh[channel].get(field), nonnegative=True
                        )
                count += 1
            for name in HFT_DERIVATIVE_FACTOR_COLUMNS:
                values[name][~np.isfinite(values[name])] = np.nan
            _yield_value = result.assign(**values)
            try:
                yield _yield_value
            except GeneratorExit:
                return
            active_segment, previous_time = str(segments[-1]), int(timestamps[-1])


def _state_number(value, *, positive=False, nonnegative=False):
    if value is None or isinstance(value, bool):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return (
        value
        if math.isfinite(value)
        and (not positive or value > 0)
        and (not nonnegative or value >= 0)
        else np.nan
    )


def _microstructure_context_rows(frames, *, factors=None):
    """Reuse the microstructure owner; retain only rows needed for causal context."""
    with closing_iterator(iter(frames)) as source:
        panels, inputs = tee(frame for frame in source if not frame.empty)
        selected = (
            (
                "return_100ms",
                "ofi_500ms",
                "depth_normalized_ofi_500ms",
                "trade_flow_imbalance_500ms",
                "return_500ms",
            )
            if factors is None
            else factors
        )
        calculated_frames = (
            iter_factor_frames(inputs, columns=selected) if selected else inputs
        )
        with closing_iterator(calculated_frames) as factors:
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
                    if "observed_utc" in panel
                    else np.full(len(panel), np.iinfo("int64").min, dtype="int64")
                )
                if bool(np.any(observed > times)):
                    raise ValueError("hft_information_observed_after_grid")
                columns = calculated.reindex(
                    columns=[
                        "return_100ms",
                        "ofi_500ms",
                        "depth_normalized_ofi_500ms",
                        "trade_flow_imbalance_500ms",
                        "return_500ms",
                    ]
                ).to_numpy(dtype="float64")
                for timestamp, mid, book_time, segment, value in zip(
                    times,
                    _nullable_numeric_column(panel, "mid"),
                    observed,
                    panel.get("segment_id", [None] * len(panel)),
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
    perpetual_frames,
    *,
    spot_frames,
    max_age_ms,
    batch_rows,
    input_delay_ms=0,
    columns=None,
):
    """Causally align one information asset without changing the execution grid."""
    selected = HFT_SPOT_FACTOR_COLUMNS if columns is None else columns
    last = previous = None
    records = []
    with (
        closing_iterator(
            _microstructure_context_rows(
                perpetual_frames, factors=spot_dependencies(selected, perpetual=True)[1]
            )
        ) as perpetual,
        closing_iterator(
            _microstructure_context_rows(
                spot_frames, factors=spot_dependencies(selected)[1]
            )
        ) as spot,
    ):
        upcoming = next(spot, None)
        for point in perpetual:
            stamp = point["timestamp"]
            while (
                upcoming is not None
                and upcoming["timestamp"] <= stamp - input_delay_ms * 1_000_000
            ):
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
                        spot_perp_basis=(last["mid"] - point["mid"]) / point["mid"],
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
    target_frames, *, information_frames, max_age_ms, batch_rows, input_delay_ms=0
):
    """Align matured, dimensionless asset states to one target asset's grid."""

    def rows(frames):
        with closing_iterator(iter(frames)) as source:
            for frame in source:
                values = [
                    _nullable_numeric_column(frame, name)
                    for name in (
                        "return_100ms",
                        "return_500ms",
                        "depth_normalized_ofi_500ms",
                        "trade_flow_imbalance_500ms",
                    )
                ]
                for row in zip(frame["timestamp"], *values, strict=True):
                    yield (pd.Timestamp(row[0]).value, *row[1:])

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
                    while (
                        point is not None
                        and point[0] <= stamp - input_delay_ms * 1_000_000
                    ):
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
    frames,
    *,
    quotes,
    trades,
    venue,
    base_units_per_amount,
    max_age_ms,
    input_delay_ms=0,
    columns=None,
):
    """Align observed external quotes and base/notional trade flow to Binance time."""
    selected = (
        HFT_EXTERNAL_FIELDS
        if columns is None
        else tuple(name.removeprefix(venue + "_") for name in columns)
    )
    needs_history = any(name.startswith(("return_", "observed_")) for name in selected)
    active_segment = previous_time = None
    mids = deque(maxlen=6)
    bins = deque(maxlen=5)
    count = 0
    latest_quote = latest_trade = None
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
    ):
        quote = next(quote_rows, None)
        trade = next(trade_rows, None)
        for source in frames:
            if source.empty:
                continue
            frame = _with_timestamp_ms(source.reset_index(drop=True))
            if needs_history:
                timestamps, segments, boundaries = _validated_grid_arrays(
                    frame,
                    active_segment=active_segment,
                    previous_time=previous_time,
                )
            else:
                timestamps = frame["timestamp_ms"].to_numpy(dtype="int64")
                if (
                    np.any(timestamps < 0)
                    or np.any(np.diff(timestamps) <= 0)
                    or (previous_time is not None and timestamps[0] <= previous_time)
                ):
                    raise ValueError("hft_grid_time_invalid")
                segments = np.full(len(frame), "pointwise", dtype=object)
                boundaries = np.zeros(len(frame), dtype=bool)
            records = []
            for index, stamp in enumerate(timestamps):
                decision_us = int(stamp) * 1000
                information_us = decision_us - input_delay_ms * 1000
                if boundaries[index]:
                    mids.clear()
                    bins.clear()
                    count = 0
                while quote is not None and quote["local_timestamp"] <= information_us:
                    latest_quote = quote
                    quote = next(quote_rows, None)
                buy = sell = buy_notional = sell_notional = Fraction()
                valid_base_bin = valid_notional_bin = True
                while trade is not None and trade["local_timestamp"] <= information_us:
                    latest_trade = trade
                    if trade["local_timestamp"] > information_us - 100_000:
                        if trade["amount"] is None or trade.get("side") not in {
                            "buy",
                            "sell",
                        }:
                            valid_base_bin = valid_notional_bin = False
                            trade = next(trade_rows, None)
                            continue
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
                        if trade["price"] is None:
                            valid_notional_bin = False
                            notional = Fraction()
                        else:
                            notional = quantity * Fraction(trade["price"])
                        if trade["side"] == "buy":
                            buy += quantity
                            buy_notional += notional
                        else:
                            sell += quantity
                            sell_notional += notional
                    trade = next(trade_rows, None)
                bins.append(
                    (
                        buy,
                        sell,
                        buy_notional,
                        sell_notional,
                        valid_base_bin,
                        valid_notional_bin,
                    )
                )
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
                    mid = (
                        (bid + ask) / 2
                        if ask <= np.finfo(float).max / 2
                        else bid / 2 + ask / 2
                    )
                    mids.append(mid)
                    binance_mid = (
                        _state_number(frame["mid"].iloc[index], positive=True)
                        if "mid" in frame
                        else np.nan
                    )
                    row.update(
                        mid_usdt=mid,
                        spread_usdt=ask - bid,
                        price_difference=(mid - binance_mid) / binance_mid,
                        relative_mid_dispersion=(
                            abs(mid - binance_mid) / (mid + binance_mid)
                            if max(mid, binance_mid) <= np.finfo(float).max / 2
                            else abs(mid / 2 - binance_mid / 2)
                            / (mid / 2 + binance_mid / 2)
                        ),
                    )
                    if len(mids) >= 2:
                        row["return_100ms"] = (mid - mids[-2]) / mids[-2]
                    if len(mids) == 6:
                        row["return_500ms"] = (mid - mids[0]) / mids[0]
                else:
                    mids.clear()
                if (count - 1) * 100 >= 500 + input_delay_ms and all(
                    value[4] for value in bins
                ):
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
                            (
                                "observed_signed_notional_flow_500ms",
                                signed_notional
                                if all(value[5] for value in bins)
                                else np.nan,
                            ),
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
    columns: tuple[str, ...] | None = None,
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
    available = {
        f"future_mid_{kind}_{horizon}ms"
        for horizon in horizons
        for kind in ("return", "direction")
    }
    if 500 in horizons:
        available.add(HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN)
    selected = available if columns is None else set(columns)
    if (
        not selected
        or not selected <= available
        or (columns is not None and len(selected) != len(columns))
    ):
        raise ValueError("hft_label_columns_invalid")
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
        labeled = _label_frame(combined, horizons=horizons, columns=selected)
        if emit_stop:
            yield labeled.iloc[:emit_stop].reset_index(drop=True)
        buffer = combined.iloc[emit_stop:].copy()
        active_segment = str(segments[-1])
        previous_time = int(timestamps[-1])
    if not buffer.empty:
        yield _label_frame(buffer, horizons=horizons, columns=selected).reset_index(
            drop=True
        )


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


def _identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        key
        for key in ("market_id", "asset_id", "timestamp", "timestamp_ms")
        if key in frame
    ]
    return frame.loc[:, columns].copy()


def _label_frame(
    frame: pd.DataFrame, *, horizons: tuple[int, ...], columns: set[str]
) -> pd.DataFrame:
    output = _identity_frame(frame)
    segment_groups = frame["segment_id"].ne(frame["segment_id"].shift()).cumsum()
    mid_horizons = tuple(
        horizon
        for horizon in horizons
        if any(
            f"future_mid_{kind}_{horizon}ms" in columns
            for kind in ("return", "direction")
        )
    )
    if mid_horizons:
        mid = pd.Series(
            _nullable_numeric_column(frame, "mid", signed_mid=True),
            index=frame.index,
        )
        grouped = mid.groupby(segment_groups, sort=False)
    for horizon in mid_horizons:
        future = grouped.shift(-(horizon // HFT_GRID_MS))
        same_sign = future.lt(0).eq(mid.lt(0))
        values = (future.where(same_sign) - mid.where(same_sign)) / mid
        values.loc[~same_sign] = future.loc[~same_sign] / mid.loc[~same_sign] - 1.0
        complete = (
            pd.concat(
                tuple(
                    grouped.shift(-step) for step in range(horizon // HFT_GRID_MS + 1)
                ),
                axis=1,
            )
            .notna()
            .all(axis=1)
        )
        values = values.where(complete & np.isfinite(values))
        available = (
            pd.to_datetime(frame["timestamp_ms"], unit="ms")
            .groupby(segment_groups, sort=False)
            .shift(-(horizon // HFT_GRID_MS))
            .where(values.notna())
        )
        for kind in ("return", "direction"):
            column = f"future_mid_{kind}_{horizon}ms"
            if column in columns:
                output[column] = values if kind == "return" else np.sign(values)
        for kind in ("return", "direction"):
            column = f"future_mid_{kind}_{horizon}ms"
            if column in columns:
                output[column + "_available_at"] = available
    proxy = label_catalog().get(HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN)
    if proxy.output_column in columns:
        best_bid_quantity = pd.Series(
            _nullable_numeric_column(frame, "best_bid_qty"),
            index=frame.index,
        )
        taker_sell_volume = pd.Series(
            _nullable_numeric_column(frame, "observed_taker_sell_volume_100ms"),
            index=frame.index,
        )
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
            .where(output[HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN].notna())
        )
    return output
