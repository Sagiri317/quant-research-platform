"""Research-only event-time taker and public-L2 maker simulations."""

from __future__ import annotations

import heapq
import json
import math
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    price_to_tick,
    tick_to_price,
)
from quant_research_platform.policy.lifecycle import closing_iterator


@dataclass(frozen=True)
class TakerConfig:
    order_quantity: float
    position_size: float
    latency_ms: int
    taker_fee_bps: float
    threshold_bps: float
    markout_horizon_ms: int

    def __post_init__(self) -> None:
        _positive(self.order_quantity, "hft_taker_order_quantity_invalid")
        _positive(self.position_size, "hft_taker_position_size_invalid")
        _nonnegative_integer(self.latency_ms, "hft_taker_latency_invalid")
        _nonnegative(self.taker_fee_bps, "hft_taker_fee_invalid")
        _nonnegative(self.threshold_bps, "hft_taker_threshold_invalid")
        _positive_integer(
            self.markout_horizon_ms,
            "hft_markout_horizon_invalid",
        )


@dataclass(frozen=True)
class MakerConfig:
    order_quantity: float
    max_position: float
    maker_fee_bps: float
    inventory_target: float
    inventory_skew: float
    markout_horizon_ms: int

    def __post_init__(self) -> None:
        _positive(self.order_quantity, "hft_maker_order_quantity_invalid")
        _positive(self.max_position, "hft_maker_position_size_invalid")
        _finite(self.maker_fee_bps, "hft_maker_fee_invalid")
        target = _finite(
            self.inventory_target,
            "hft_maker_inventory_target_invalid",
        )
        if abs(target) > self.max_position:
            raise ValueError("hft_maker_inventory_target_invalid")
        _nonnegative(self.inventory_skew, "hft_maker_inventory_skew_invalid")
        _positive_integer(
            self.markout_horizon_ms,
            "hft_markout_horizon_invalid",
        )


def _finite(value: object, error: str) -> float:
    if isinstance(value, bool):
        raise ValueError(error)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not math.isfinite(number):
        raise ValueError(error)
    return number


def _positive(value: object, error: str) -> float:
    number = _finite(value, error)
    if number <= 0:
        raise ValueError(error)
    return number


def _nonnegative(value: object, error: str) -> float:
    number = _finite(value, error)
    if number < 0:
        raise ValueError(error)
    return number


def _nonnegative_integer(value: object, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(error)
    return value


def _positive_integer(value: object, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(error)
    return value


@dataclass
class _Metrics:
    fill_events: int = 0
    fill_quantity: float = 0.0
    fees: float = 0.0
    turnover: float = 0.0
    slippage_quantity: float = 0.0
    slippage_weighted: float = 0.0
    spread_capture_pnl: float = 0.0
    defined_markout_quantity: float = 0.0
    markout_weighted_usd: float = 0.0
    markout_weighted_bps: float = 0.0
    markout_pnl: float = 0.0
    adverse_selection_cost: float = 0.0
    undefined_markouts: int = 0

    def observe_fill(self, fill: Mapping[str, Any]) -> None:
        quantity = float(fill["fill_qty"])
        self.fill_events += 1
        self.fill_quantity += quantity
        self.fees += float(fill["fee_usd"])
        self.turnover += float(fill["notional_usd"])
        if "slippage_bps" in fill:
            self.slippage_quantity += quantity
            self.slippage_weighted += quantity * float(fill["slippage_bps"])
        if "spread_capture_usd_per_unit" in fill:
            self.spread_capture_pnl += quantity * float(
                fill["spread_capture_usd_per_unit"]
            )

    def finalize_markout(self, fill: Mapping[str, Any]) -> None:
        quantity = float(fill["fill_qty"])
        markout = fill.get("markout_usd_per_unit")
        if markout is None:
            self.undefined_markouts += 1
            return
        value = float(markout)
        self.defined_markout_quantity += quantity
        self.markout_weighted_usd += quantity * value
        self.markout_weighted_bps += quantity * float(fill["markout_bps"])
        self.markout_pnl += float(fill["markout_pnl_usd"])
        self.adverse_selection_cost += float(fill["adverse_selection_cost_usd"])

    def payload(self) -> dict[str, Any]:
        return {
            "fill_events": self.fill_events,
            "fill_quantity": self.fill_quantity,
            "fees_usd": self.fees,
            "turnover_usd": self.turnover,
            "average_slippage_bps": (
                self.slippage_weighted / self.slippage_quantity
                if self.slippage_quantity
                else None
            ),
            "spread_capture_pnl_usd": self.spread_capture_pnl,
            "average_markout_usd_per_unit": (
                self.markout_weighted_usd / self.defined_markout_quantity
                if self.defined_markout_quantity
                else None
            ),
            "average_markout_bps": (
                self.markout_weighted_bps / self.defined_markout_quantity
                if self.defined_markout_quantity
                else None
            ),
            "markout_pnl_usd": (
                self.markout_pnl if self.defined_markout_quantity else None
            ),
            "adverse_selection_cost_usd": (
                self.adverse_selection_cost if self.defined_markout_quantity else None
            ),
            "markout_undefined_fill_events": self.undefined_markouts,
        }


class _PendingMarkouts:
    """Disk-backed FIFO for exact markouts without retained fill history."""

    def __init__(
        self,
        metrics: _Metrics,
        on_fill: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._read_offset = 0
        self._next: tuple[dict[str, Any], int] | None = None
        self._metrics = metrics
        self._on_fill = on_fill

    def __enter__(self) -> _PendingMarkouts:
        return self

    def __exit__(
        self,
        _exception_type: object,
        primary: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.expire_all()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                "execution pending markout flush also failed: "
                f"{type(cleanup_error).__name__}"
            )
        finally:
            self._file.close()

    def add(self, fill: dict[str, Any]) -> None:
        self._file.seek(0, 2)
        self._file.write(
            json.dumps(fill, separators=(",", ":"), allow_nan=False).encode("utf-8")
            + b"\n"
        )
        self._file.flush()

    def resolve(self, state: BookState) -> None:
        future_mid = _mid(state)
        while (fill := self._peek()) is not None:
            if fill["segment_id"] != state.segment_id:
                self._finalize(fill, future_mid=None)
                self._pop()
                continue
            if state.research_time_ms < fill["markout_target_time_ms"]:
                break
            self._finalize(fill, future_mid=future_mid)
            self._pop()

    def expire_all(self) -> None:
        while (fill := self._peek()) is not None:
            self._finalize(fill, future_mid=None)
            self._pop()

    def _peek(self) -> dict[str, Any] | None:
        if self._next is not None:
            return self._next[0]
        self._file.seek(self._read_offset)
        line = self._file.readline()
        if not line:
            return None
        self._next = (json.loads(line), self._file.tell())
        return self._next[0]

    def _pop(self) -> None:
        assert self._next is not None
        self._read_offset = self._next[1]
        self._next = None

    def _finalize(
        self,
        fill: dict[str, Any],
        *,
        future_mid: float | None,
    ) -> None:
        if future_mid is not None:
            signed_markout = (
                future_mid - fill["fill_price"]
                if fill["side"] == "buy"
                else fill["fill_price"] - future_mid
            )
            fill["markout_usd_per_unit"] = signed_markout
            fill["markout_bps"] = signed_markout / fill["fill_price"] * 10_000.0
            fill["markout_pnl_usd"] = signed_markout * fill["fill_qty"]
            fill["adverse_selection_cost_usd"] = (
                max(0.0, -signed_markout) * fill["fill_qty"]
            )
        else:
            fill["markout_usd_per_unit"] = None
            fill["markout_bps"] = None
            fill["markout_pnl_usd"] = None
            fill["adverse_selection_cost_usd"] = None
        self._metrics.finalize_markout(fill)
        if self._on_fill is not None:
            self._on_fill(fill)


def _checked_book_events(
    events: Iterable[BookState | BookBoundary],
) -> Iterator[BookState | BookBoundary]:
    previous: int | None = None
    for event in events:
        if not isinstance(event, BookState | BookBoundary):
            raise TypeError("hft_execution_book_event_invalid")
        timestamp = event.research_time_ms
        if previous is not None and timestamp < previous:
            raise ValueError("hft_execution_time_non_monotonic")
        previous = timestamp
        yield event


def _checked_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    previous: int | None = None
    for source in decisions:
        row = dict(source)
        timestamp = row.get("timestamp_ms")
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("hft_execution_decision_time_invalid")
        if previous is not None and timestamp <= previous:
            raise ValueError("hft_execution_decision_time_non_monotonic")
        previous = timestamp
        yield row


def _mid(state: BookState) -> float:
    if not state.bids or not state.asks:
        raise ValueError("hft_execution_one_sided_book")
    bid = tick_to_price(state.bids[0][0], state.tick_size)
    ask = tick_to_price(state.asks[0][0], state.tick_size)
    if ask <= bid:
        raise ValueError("hft_execution_crossed_book")
    return (bid + ask) / 2.0


def _covered(
    coverage: Mapping[str, tuple[int, int]],
    segment_id: str,
    timestamp_ms: int,
) -> bool:
    bounds = coverage.get(segment_id)
    return bool(bounds is not None and bounds[0] <= timestamp_ms <= bounds[1])


def _resolve_observed_markouts(
    pending: _PendingMarkouts,
    state: BookState,
    coverage: Mapping[str, tuple[int, int]],
) -> None:
    bounds = coverage.get(state.segment_id)
    if bounds is None or state.research_time_ms > bounds[1]:
        pending.expire_all()
    elif state.research_time_ms >= bounds[0]:
        pending.resolve(state)


def simulate_taker(
    book_events: Iterable[BookState | BookBoundary],
    decisions: Iterable[Mapping[str, Any]],
    *,
    prediction_column: str,
    config: TakerConfig,
    observed_coverage: Mapping[str, tuple[int, int]],
    on_fill: Callable[[dict[str, Any]], None] | None = None,
    on_equity: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Use the latest actually known valid L2 state at effective time."""

    events = iter(_checked_book_events(book_events))
    active_decisions = iter(_checked_decisions(decisions))
    next_event = next(events, None)
    latest_state: BookState | None = None
    metrics = _Metrics()
    position = 0.0
    cash_gross = 0.0
    fees = 0.0
    created_orders = 0
    filled_orders = 0
    unfilled_quantity = 0.0
    equity_rows = 0
    max_abs_position = 0.0
    last_gross: float | None = None
    last_net: float | None = None
    market_event_count = 0

    with (
        closing_iterator(events),
        closing_iterator(active_decisions),
        _PendingMarkouts(metrics, on_fill) as pending,
    ):

        def consume_until(target: int) -> None:
            nonlocal latest_state, market_event_count, next_event
            while next_event is not None and next_event.research_time_ms <= target:
                market_event_count += 1
                if isinstance(next_event, BookBoundary):
                    latest_state = None
                    pending.expire_all()
                else:
                    latest_state = next_event
                    _resolve_observed_markouts(
                        pending,
                        next_event,
                        observed_coverage,
                    )
                next_event = next(events, None)

        for row in active_decisions:
            decision_time = row["timestamp_ms"]
            prediction = _finite(
                row.get(prediction_column),
                "hft_prediction_invalid",
            )
            effective_time = decision_time + config.latency_ms
            consume_until(effective_time)
            current_mid: float | None = None
            if latest_state is not None and _covered(
                observed_coverage,
                latest_state.segment_id,
                effective_time,
            ):
                decision_segment = row.get("segment_id")
                if (
                    decision_segment is None
                    or decision_segment == latest_state.segment_id
                ):
                    current_mid = _mid(latest_state)
                    bid = tick_to_price(
                        latest_state.bids[0][0],
                        latest_state.tick_size,
                    )
                    ask = tick_to_price(
                        latest_state.asks[0][0],
                        latest_state.tick_size,
                    )
                    edge_threshold = (
                        (ask - bid) / current_mid / 2.0
                        + config.taker_fee_bps / 10_000.0
                        + config.threshold_bps / 10_000.0
                    )
                    target_position = (
                        config.position_size
                        if prediction > edge_threshold
                        else -config.position_size
                        if prediction < -edge_threshold
                        else 0.0
                    )
                    delta = target_position - position
                    if delta:
                        created_orders += 1
                        side = "buy" if delta > 0 else "sell"
                        levels = latest_state.asks if delta > 0 else latest_state.bids
                        requested = min(abs(delta), config.order_quantity)
                        remaining = requested
                        notional = 0.0
                        levels_consumed = 0
                        for price_tick, available in levels:
                            quantity = min(remaining, available)
                            if quantity <= 0:
                                continue
                            price = tick_to_price(price_tick, latest_state.tick_size)
                            notional += quantity * price
                            remaining -= quantity
                            levels_consumed += 1
                            if remaining <= 0:
                                break
                        filled = requested - remaining
                        unfilled_quantity += remaining
                        if filled > 0:
                            filled_orders += 1
                            fill_price = notional / filled
                            fee = notional * config.taker_fee_bps / 10_000.0
                            signed = 1.0 if delta > 0 else -1.0
                            position += signed * filled
                            cash_gross -= signed * notional
                            fees += fee
                            fill = {
                                "order_id": created_orders,
                                "timestamp_ms": effective_time,
                                "effective_state_time_ms": (
                                    latest_state.research_time_ms
                                ),
                                "segment_id": latest_state.segment_id,
                                "side": side,
                                "fill_price": fill_price,
                                "fill_qty": filled,
                                "levels_consumed": levels_consumed,
                                "notional_usd": notional,
                                "fee_usd": fee,
                                "slippage_bps": (
                                    signed
                                    * (fill_price - current_mid)
                                    / current_mid
                                    * 10_000.0
                                ),
                                "markout_target_time_ms": (
                                    effective_time + config.markout_horizon_ms
                                ),
                            }
                            metrics.observe_fill(fill)
                            pending.add(fill)
            gross = None if current_mid is None else cash_gross + position * current_mid
            net = None if gross is None else gross - fees
            equity = {
                "timestamp_ms": decision_time,
                "position": position,
                "gross_pnl_usd": gross,
                "fees_usd": fees,
                "net_pnl_usd": net,
            }
            if on_equity is not None:
                on_equity(equity)
            equity_rows += 1
            max_abs_position = max(max_abs_position, abs(position))
            last_gross = gross
            last_net = net
        while next_event is not None:
            market_event_count += 1
            if isinstance(next_event, BookBoundary):
                latest_state = None
                pending.expire_all()
            else:
                latest_state = next_event
                _resolve_observed_markouts(
                    pending,
                    next_event,
                    observed_coverage,
                )
            next_event = next(events, None)
    return {
        "simulation": "event_time_target_position_taker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "unfilled_orders": created_orders - filled_orders,
        "unfilled_quantity": unfilled_quantity,
        "equity_rows": equity_rows,
        "gross_pnl_usd": last_gross,
        "net_pnl_usd": last_net,
        "max_abs_position": max_abs_position,
        "market_event_count": market_event_count,
        "latency_ms": config.latency_ms,
        "taker_fee_bps": config.taker_fee_bps,
        "order_quantity": config.order_quantity,
        "position_size": config.position_size,
        "threshold_bps": config.threshold_bps,
        "markout_horizon_ms": config.markout_horizon_ms,
        "latency_fidelity": "full_l2_observed_event",
        "assumption": "historical_book_zero_market_impact",
        **metrics.payload(),
    }


def taker_latency_sweep(
    state_factory: Callable[[], Iterable[BookState | BookBoundary]],
    decision_factory: Callable[[], Iterable[Mapping[str, Any]]],
    *,
    prediction_column: str,
    config: TakerConfig,
    latency_scenarios_ms: tuple[int, ...],
    observed_coverage: Mapping[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    if any(
        isinstance(value, bool) or type(value) is not int or value < 0
        for value in latency_scenarios_ms
    ):
        raise ValueError("hft_latency_scenario_invalid")
    return [
        simulate_taker(
            state_factory(),
            decision_factory(),
            prediction_column=prediction_column,
            config=replace(config, latency_ms=latency_ms),
            observed_coverage=observed_coverage,
        )
        for latency_ms in latency_scenarios_ms
    ]


@dataclass
class _MakerOrder:
    order_id: int
    side: str
    price_tick: int
    remaining: float
    queue_ahead: float
    initial_queue_ahead: float
    segment_id: str
    tick_size: str
    created_time_ms: int
    had_fill: bool = False


def _checked_trades(
    trades: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[int, tuple[int, float, bool, int]]]:
    previous: int | None = None
    for record in trades:
        payload = record.get("payload")
        tick_size = record.get("tick_size")
        if not isinstance(payload, Mapping) or not isinstance(tick_size, str):
            raise ValueError("hft_execution_trade_payload_invalid")
        timestamp = payload.get("T")
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("hft_execution_trade_time_invalid")
        if previous is not None and timestamp < previous:
            raise ValueError("hft_execution_time_non_monotonic")
        previous = timestamp
        quantity = _positive(
            payload.get("q"),
            "hft_execution_trade_quantity_invalid",
        )
        maker_buyer = payload.get("m")
        if not isinstance(maker_buyer, bool):
            raise ValueError("hft_execution_trade_side_invalid")
        yield (
            timestamp,
            (
                price_to_tick(
                    payload.get("p"),
                    tick_size,
                    error="hft_execution_trade_price_invalid",
                ),
                quantity,
                maker_buyer,
                timestamp,
            ),
        )


def _market_events(
    books: Iterable[BookState | BookBoundary],
    trades: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[int, int, object]]:
    book_events = (
        (event.research_time_ms, 1, event) for event in _checked_book_events(books)
    )
    trade_events = (
        (timestamp, 0, trade) for timestamp, trade in _checked_trades(trades)
    )
    yield from heapq.merge(
        trade_events,
        book_events,
        key=lambda item: (item[0], item[1]),
    )


def simulate_maker(
    book_events: Iterable[BookState | BookBoundary],
    trades: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    *,
    config: MakerConfig,
    observed_coverage: Mapping[str, tuple[int, int]],
    on_fill: Callable[[dict[str, Any]], None] | None = None,
    on_equity: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Simulate an estimated-queue maker without claiming MBO/FIFO truth."""

    events = iter(_market_events(book_events, trades))
    active_decisions = iter(_checked_decisions(decisions))
    next_event = next(events, None)
    latest_state: BookState | None = None
    active: dict[str, _MakerOrder] = {}
    metrics = _Metrics()
    position = 0.0
    cash_gross = 0.0
    fees = 0.0
    created_orders = 0
    filled_orders = 0
    cancelled_orders = 0
    partial_fill_events = 0
    trade_through_fill_events = 0
    equity_rows = 0
    max_abs_position = 0.0
    market_event_count = 0
    last_event_time = 0
    last_mid: float | None = None
    last_gross: float | None = None
    last_net: float | None = None
    all_lifetime_count = 0
    all_lifetime_sum = 0
    filled_lifetime_count = 0
    filled_lifetime_sum = 0

    def finish_order(order: _MakerOrder, timestamp_ms: int, *, cancelled: bool) -> None:
        nonlocal all_lifetime_count, all_lifetime_sum, cancelled_orders
        nonlocal filled_lifetime_count, filled_lifetime_sum
        lifetime = max(0, timestamp_ms - order.created_time_ms)
        all_lifetime_count += 1
        all_lifetime_sum += lifetime
        if order.had_fill:
            filled_lifetime_count += 1
            filled_lifetime_sum += lifetime
        if cancelled:
            cancelled_orders += 1

    with (
        closing_iterator(events),
        closing_iterator(active_decisions),
        _PendingMarkouts(metrics, on_fill) as pending,
    ):

        def cancel_active(timestamp_ms: int) -> None:
            for order in active.values():
                finish_order(order, timestamp_ms, cancelled=True)
            active.clear()

        def process_book(event: BookState | BookBoundary) -> None:
            nonlocal last_mid, latest_state
            if isinstance(event, BookBoundary):
                cancel_active(event.research_time_ms)
                latest_state = None
                last_mid = None
                pending.expire_all()
                return
            if latest_state is not None and event.segment_id != latest_state.segment_id:
                cancel_active(event.research_time_ms)
            latest_state = event
            last_mid = _mid(event)
            _resolve_observed_markouts(
                pending,
                event,
                observed_coverage,
            )

        def process_trade(trade: tuple[int, float, bool, int]) -> None:
            nonlocal cash_gross, fees, filled_orders, partial_fill_events
            nonlocal position, trade_through_fill_events
            if latest_state is None:
                return
            trade_tick, quantity, maker_buyer, timestamp_ms = trade
            if not _covered(
                observed_coverage,
                latest_state.segment_id,
                timestamp_ms,
            ):
                return
            mid = _mid(latest_state)
            for side in ("buy", "sell"):
                order = active.get(side)
                if order is None or order.segment_id != latest_state.segment_id:
                    continue
                same_price = trade_tick == order.price_tick
                trade_through = (
                    side == "buy" and maker_buyer and trade_tick < order.price_tick
                ) or (
                    side == "sell" and not maker_buyer and trade_tick > order.price_tick
                )
                same_price_depletion = same_price and (
                    (side == "buy" and maker_buyer)
                    or (side == "sell" and not maker_buyer)
                )
                if trade_through:
                    fill_quantity = order.remaining
                    order.queue_ahead = 0.0
                    reason = "trade_through"
                elif same_price_depletion:
                    consumed = min(order.queue_ahead, quantity)
                    order.queue_ahead -= consumed
                    fill_quantity = min(order.remaining, quantity - consumed)
                    reason = "same_price_depletion"
                else:
                    continue
                if fill_quantity <= 0:
                    continue
                remaining_before = order.remaining
                fill_price = tick_to_price(order.price_tick, order.tick_size)
                notional = fill_quantity * fill_price
                fee = notional * config.maker_fee_bps / 10_000.0
                signed = 1.0 if side == "buy" else -1.0
                fill = {
                    "order_id": order.order_id,
                    "timestamp_ms": timestamp_ms,
                    "segment_id": order.segment_id,
                    "side": side,
                    "fill_reason": reason,
                    "fill_price": fill_price,
                    "fill_qty": fill_quantity,
                    "notional_usd": notional,
                    "fee_usd": fee,
                    "estimated_queue_ahead_at_join": order.initial_queue_ahead,
                    "estimated_queue_ahead_before_fill": order.queue_ahead,
                    "spread_capture_usd_per_unit": signed * (mid - fill_price),
                    "markout_target_time_ms": (
                        timestamp_ms + config.markout_horizon_ms
                    ),
                }
                metrics.observe_fill(fill)
                pending.add(fill)
                position += signed * fill_quantity
                cash_gross -= signed * notional
                fees += fee
                if not order.had_fill:
                    order.had_fill = True
                    filled_orders += 1
                order.remaining -= fill_quantity
                if fill_quantity < remaining_before:
                    partial_fill_events += 1
                if trade_through:
                    trade_through_fill_events += 1
                if order.remaining <= 0:
                    active.pop(side, None)
                    finish_order(order, timestamp_ms, cancelled=False)
                break

        def process_until(target: int) -> None:
            nonlocal last_event_time, market_event_count, next_event
            while next_event is not None and next_event[0] <= target:
                market_event_count += 1
                last_event_time = next_event[0]
                _, kind, item = next_event
                if kind == 0:
                    process_trade(item)
                else:
                    process_book(item)
                next_event = next(events, None)

        for row in active_decisions:
            decision_time = row["timestamp_ms"]
            process_until(decision_time)
            current_mid: float | None = None
            if latest_state is None or not _covered(
                observed_coverage,
                latest_state.segment_id,
                decision_time,
            ):
                cancel_active(decision_time)
            else:
                current_mid = _mid(latest_state)
                desired = {
                    "buy": latest_state.bids[0][0],
                    "sell": latest_state.asks[0][0],
                }
                for side, price_tick in desired.items():
                    order = active.get(side)
                    if order is not None and order.price_tick != price_tick:
                        active.pop(side)
                        finish_order(order, decision_time, cancelled=True)
                inventory_ratio = (
                    position - config.inventory_target
                ) / config.max_position
                sizes = {
                    "buy": min(
                        config.order_quantity
                        * max(0.0, 1.0 - config.inventory_skew * inventory_ratio),
                        max(0.0, config.max_position - position),
                    ),
                    "sell": min(
                        config.order_quantity
                        * max(0.0, 1.0 + config.inventory_skew * inventory_ratio),
                        max(0.0, config.max_position + position),
                    ),
                }
                for side, size in sizes.items():
                    if side in active or size <= 0:
                        continue
                    created_orders += 1
                    queue_ahead = (
                        latest_state.bids[0][1]
                        if side == "buy"
                        else latest_state.asks[0][1]
                    )
                    active[side] = _MakerOrder(
                        order_id=created_orders,
                        side=side,
                        price_tick=desired[side],
                        remaining=size,
                        queue_ahead=queue_ahead,
                        initial_queue_ahead=queue_ahead,
                        segment_id=latest_state.segment_id,
                        tick_size=latest_state.tick_size,
                        created_time_ms=decision_time,
                    )
            gross = None if current_mid is None else cash_gross + position * current_mid
            net = None if gross is None else gross - fees
            equity = {
                "timestamp_ms": decision_time,
                "position": position,
                "gross_pnl_usd": gross,
                "fees_usd": fees,
                "net_pnl_usd": net,
            }
            if on_equity is not None:
                on_equity(equity)
            equity_rows += 1
            max_abs_position = max(max_abs_position, abs(position))
            last_gross = gross
            last_net = net
        while next_event is not None:
            process_until(next_event[0])
        cancel_active(last_event_time)
        if latest_state is not None:
            last_mid = _mid(latest_state)
            last_gross = cash_gross + position * last_mid
            last_net = last_gross - fees
        else:
            last_gross = None
            last_net = None
    return {
        "simulation": "event_ordered_public_l2_maker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "unfilled_orders": created_orders - filled_orders,
        "partial_fill_events": partial_fill_events,
        "trade_through_fill_events": trade_through_fill_events,
        "equity_rows": equity_rows,
        "gross_pnl_usd": last_gross,
        "net_pnl_usd": last_net,
        "max_abs_position": max_abs_position,
        "inventory_revaluation_pnl_usd": (
            None if last_gross is None else last_gross - metrics.spread_capture_pnl
        ),
        "average_order_lifetime_ms": (
            all_lifetime_sum / all_lifetime_count if all_lifetime_count else None
        ),
        "average_filled_order_lifetime_ms": (
            filled_lifetime_sum / filled_lifetime_count
            if filled_lifetime_count
            else None
        ),
        "market_event_count": market_event_count,
        "maker_fee_bps": config.maker_fee_bps,
        "order_quantity": config.order_quantity,
        "max_position": config.max_position,
        "inventory_target": config.inventory_target,
        "inventory_skew": config.inventory_skew,
        "markout_horizon_ms": config.markout_horizon_ms,
        "queue_model": "estimated_queue_ahead",
        "queue_assumption": (
            "public MBP aggregate depth; cancellations do not advance queue"
        ),
        "same_timestamp_policy": "trade_before_book_and_requote",
        "assumption": "historical_book_zero_market_impact",
        **metrics.payload(),
    }
