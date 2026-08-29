"""Research-only event-time taker and public-L2 maker simulations."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    tick_to_price,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    OwnedValue,
    require_free_disk_headroom,
)

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_PENDING_MARKOUT_DISK_CHECK_BYTES = 1024 * 1024


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
        self._temp_root = Path(tempfile.gettempdir())
        require_free_disk_headroom(
            self._temp_root,
            reserve_bytes=2 * _PENDING_MARKOUT_DISK_CHECK_BYTES,
        )
        self._disk_bytes_remaining = _PENDING_MARKOUT_DISK_CHECK_BYTES
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
        cleanup_failure: BaseException | None = None
        try:
            self.expire_all()
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "execution pending markout flush also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            else:
                cleanup_failure = cleanup_error
        try:
            self._file.close()
        except BaseException as close_error:
            if primary is not None:
                primary.add_note(
                    "execution pending markout handle close also failed: "
                    f"{type(close_error).__name__}"
                )
            elif cleanup_failure is None:
                cleanup_failure = close_error
            else:
                cleanup_failure.add_note(
                    "execution pending markout handle close also failed: "
                    f"{type(close_error).__name__}"
                )
        if cleanup_failure is not None:
            raise cleanup_failure

    def add(self, fill: dict[str, Any]) -> None:
        payload = (
            json.dumps(fill, separators=(",", ":"), allow_nan=False).encode("utf-8")
            + b"\n"
        )
        if len(payload) > self._disk_bytes_remaining:
            admitted = max(_PENDING_MARKOUT_DISK_CHECK_BYTES, len(payload))
            require_free_disk_headroom(
                self._temp_root,
                reserve_bytes=2 * admitted,
            )
            self._disk_bytes_remaining = admitted
        self._disk_bytes_remaining -= len(payload)
        self._file.seek(0, 2)
        self._file.write(payload)
        self._file.flush()

    def resolve(
        self,
        state: BookState,
        *,
        previous_state: BookState | None,
        coverage: Mapping[str, tuple[int, int]],
    ) -> None:
        current_mid = _mid(state)
        previous_mid = None if previous_state is None else _mid(previous_state)
        while (fill := self._peek()) is not None:
            if fill["segment_id"] != state.segment_id:
                self._finalize(fill, future_mid=None)
                self._pop()
                continue
            target = fill["markout_target_time_ns"]
            if state.observed_time_ns < target:
                break
            if target == state.observed_time_ns:
                markout_mid = current_mid
            elif (
                previous_state is not None
                and previous_state.segment_id == fill["segment_id"]
                and _covered(coverage, fill["segment_id"], target)
            ):
                markout_mid = previous_mid
            else:
                markout_mid = None
            self._finalize(fill, future_mid=markout_mid)
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


def _book_value(event: BookState | OwnedValue[BookState]) -> BookState:
    value = event.value if isinstance(event, OwnedValue) else event
    if not isinstance(value, BookState):
        raise TypeError("hft_execution_book_event_invalid")
    return value


def _book_key(
    event: BookState | OwnedValue[BookState] | BookBoundary,
) -> tuple[int, int, int]:
    value = event.value if isinstance(event, OwnedValue) else event
    return (
        value.observed_time_ns,
        value.receive_monotonic_ns,
        value.capture_seq,
    )


def _checked_book_events(
    events: Iterable[BookState | OwnedValue[BookState] | BookBoundary],
) -> Iterator[BookState | OwnedValue[BookState] | BookBoundary]:
    previous: tuple[int, int, int] | None = None
    active = iter(events)
    with closing_iterator(active):
        for event in active:
            if isinstance(event, OwnedValue):
                value = event.value
            else:
                value = event
            if not isinstance(value, BookState | BookBoundary):
                raise TypeError("hft_execution_book_event_invalid")
            key = (
                value.observed_time_ns,
                value.receive_monotonic_ns,
                value.capture_seq,
            )
            if previous is not None and key < previous:
                raise ValueError("hft_execution_time_non_monotonic")
            previous = key
            yield event


def _checked_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    previous: int | None = None
    active = iter(decisions)
    with closing_iterator(active):
        for source in active:
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
    timestamp_ns: int,
) -> bool:
    bounds = coverage.get(segment_id)
    return bool(bounds is not None and bounds[0] <= timestamp_ns < bounds[1])


def _resolve_observed_markouts(
    pending: _PendingMarkouts,
    state: BookState,
    coverage: Mapping[str, tuple[int, int]],
    *,
    previous_state: BookState | None,
) -> None:
    bounds = coverage.get(state.segment_id)
    if bounds is None or state.observed_time_ns >= bounds[1]:
        pending.expire_all()
    elif state.observed_time_ns >= bounds[0]:
        pending.resolve(
            state,
            previous_state=previous_state,
            coverage=coverage,
        )


def simulate_taker(
    book_events: Iterable[BookState | OwnedValue[BookState] | BookBoundary],
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
    next_event = None
    latest_state: BookState | None = None
    latest_owned: OwnedValue[BookState] | None = None
    terminal_mid: float | None = None
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
        try:
            next_event = next(events, None)

            def consume_event(
                event: BookState | OwnedValue[BookState] | BookBoundary,
            ) -> None:
                nonlocal latest_state, latest_owned, terminal_mid
                if isinstance(event, BookBoundary):
                    if latest_owned is not None:
                        latest_owned.release(owner="crypto.execution.book")
                    latest_owned = None
                    latest_state = None
                    pending.expire_all()
                    return
                previous_state = latest_state
                state = _book_value(event)
                if latest_owned is not None:
                    latest_owned.release(owner="crypto.execution.book")
                latest_owned = event if isinstance(event, OwnedValue) else None
                latest_state = state
                if _covered(
                    observed_coverage,
                    state.segment_id,
                    state.observed_time_ns,
                ):
                    terminal_mid = _mid(state)
                _resolve_observed_markouts(
                    pending,
                    state,
                    observed_coverage,
                    previous_state=previous_state,
                )

            def consume_until(target: int) -> None:
                nonlocal market_event_count, next_event
                while next_event is not None and _book_key(next_event)[0] <= target:
                    event = next_event
                    next_event = None
                    market_event_count += 1
                    consume_event(event)
                    next_event = next(events, None)

            for row in active_decisions:
                decision_time = row["timestamp_ms"]
                prediction = _finite(
                    row.get(prediction_column),
                    "hft_prediction_invalid",
                )
                effective_time = decision_time + config.latency_ms
                effective_time_ns = effective_time * _NANOSECONDS_PER_MILLISECOND
                consume_until(effective_time_ns)
                current_mid: float | None = None
                if latest_state is not None and _covered(
                    observed_coverage,
                    latest_state.segment_id,
                    effective_time_ns,
                ):
                    decision_segment = row.get(
                        "book_segment_id",
                        row.get("segment_id"),
                    )
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
                            levels = (
                                latest_state.asks
                                if delta > 0
                                else latest_state.bids
                            )
                            requested = min(abs(delta), config.order_quantity)
                            remaining = requested
                            notional = 0.0
                            levels_consumed = 0
                            for price_tick, available in levels:
                                quantity = min(remaining, available)
                                if quantity <= 0:
                                    continue
                                price = tick_to_price(
                                    price_tick,
                                    latest_state.tick_size,
                                )
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
                                    "decision_time_ms": decision_time,
                                    "effective_time_ms": effective_time,
                                    "state_observed_time_ms": (
                                        latest_state.observed_time_ms
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
                                    "markout_target_time_ns": (
                                        effective_time_ns
                                        + config.markout_horizon_ms
                                        * _NANOSECONDS_PER_MILLISECOND
                                    ),
                                }
                                metrics.observe_fill(fill)
                                pending.add(fill)
                gross = (
                    None
                    if current_mid is None
                    else cash_gross + position * current_mid
                )
                net = None if gross is None else gross - fees
                equity = {
                    "timestamp_ms": effective_time,
                    "decision_time_ms": decision_time,
                    "effective_time_ms": effective_time,
                    "state_observed_time_ms": (
                        None
                        if latest_state is None
                        else latest_state.observed_time_ms
                    ),
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
                event = next_event
                next_event = None
                market_event_count += 1
                consume_event(event)
                next_event = next(events, None)
            if terminal_mid is None:
                last_gross = None
                last_net = None
            else:
                last_gross = cash_gross + position * terminal_mid
                last_net = last_gross - fees
        finally:
            if latest_owned is not None:
                latest_owned.release(owner="crypto.execution.book")
            if isinstance(next_event, OwnedValue):
                next_event.release(owner="crypto.execution.book")
    return {
        "simulation": "event_time_target_position_taker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "zero_fill_orders": created_orders - filled_orders,
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
    state_factory: Callable[
        [], Iterable[BookState | OwnedValue[BookState] | BookBoundary]
    ],
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
    trade_connection_id: str
    tick_size: str
    created_time_ns: int
    valid_until_ns: int
    had_fill: bool = False


def _checked_trades(
    trades: Iterable[TradeEvent],
) -> Iterator[TradeEvent]:
    previous: tuple[int, int, int] | None = None
    active = iter(trades)
    with closing_iterator(active):
        for trade in active:
            if not isinstance(trade, TradeEvent):
                raise TypeError("hft_execution_trade_event_invalid")
            key = (
                trade.observed_time_ns,
                trade.receive_monotonic_ns,
                trade.capture_seq,
            )
            if previous is not None and key < previous:
                raise ValueError("hft_execution_time_non_monotonic")
            previous = key
            _positive(trade.quantity, "hft_execution_trade_quantity_invalid")
            visible = _finite(
                trade.visible_quantity,
                "hft_execution_trade_visible_quantity_invalid",
            )
            if visible < 0 or visible > trade.quantity:
                raise ValueError("hft_execution_trade_visible_quantity_invalid")
            if not isinstance(trade.buyer_is_maker, bool):
                raise ValueError("hft_execution_trade_side_invalid")
            yield trade


def _market_events(
    books: Iterable[BookState | OwnedValue[BookState] | BookBoundary],
    trades: Iterable[TradeEvent],
) -> Iterator[tuple[tuple[int, int, int], object]]:
    book_events = iter(_checked_book_events(books))
    trade_events = iter(_checked_trades(trades))
    book_next: BookState | OwnedValue[BookState] | BookBoundary | None = None
    trade_next: TradeEvent | None = None
    pending_event: object | None = None
    with closing_iterator(book_events), closing_iterator(trade_events):
        try:
            book_next = next(book_events, None)
            trade_next = next(trade_events, None)
            while book_next is not None or trade_next is not None:
                if trade_next is None or (
                    book_next is not None and _book_key(book_next) <= (
                        trade_next.observed_time_ns,
                        trade_next.receive_monotonic_ns,
                        trade_next.capture_seq,
                    )
                ):
                    pending_event = book_next
                    assert book_next is not None
                    key = _book_key(book_next)
                    book_next = next(book_events, None)
                else:
                    pending_event = trade_next
                    assert trade_next is not None
                    key = (
                        trade_next.observed_time_ns,
                        trade_next.receive_monotonic_ns,
                        trade_next.capture_seq,
                    )
                    trade_next = next(trade_events, None)
                yield key, pending_event
                pending_event = None
        finally:
            if isinstance(pending_event, OwnedValue):
                pending_event.release(owner="crypto.execution.market_event")
            if isinstance(book_next, OwnedValue):
                book_next.release(owner="crypto.execution.market_event")


def _checked_quote_schedule(
    rows: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    previous: int | None = None
    active = iter(rows)
    with closing_iterator(active):
        for source in active:
            row = dict(source)
            if "prediction_score" in row:
                raise ValueError("hft_maker_prediction_forbidden")
            for key in (
                "asset_id",
                "segment_id",
                "book_segment_id",
                "trade_connection_id",
            ):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError("hft_maker_quote_schedule_identity_invalid")
            if row["segment_id"] != (
                f"{row['book_segment_id']}:{row['trade_connection_id']}"
            ):
                raise ValueError("hft_maker_quote_schedule_identity_invalid")
            timestamp = row.get("timestamp_ms")
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError("hft_maker_quote_schedule_time_invalid")
            if previous is not None and timestamp <= previous:
                raise ValueError("hft_maker_quote_schedule_time_non_monotonic")
            previous = timestamp
            yield row


def simulate_maker(
    book_events: Iterable[BookState | OwnedValue[BookState] | BookBoundary],
    trades: Iterable[TradeEvent],
    quote_schedule: Iterable[Mapping[str, Any]],
    *,
    config: MakerConfig,
    observed_coverage: Mapping[str, tuple[int, int]],
    trade_coverage: Mapping[str, tuple[int, int]],
    on_fill: Callable[[dict[str, Any]], None] | None = None,
    on_equity: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Simulate the model-free public-L2 maker baseline."""

    events = iter(_market_events(book_events, trades))
    active_decisions = iter(_checked_quote_schedule(quote_schedule))
    next_event = None
    latest_state: BookState | None = None
    latest_owned: OwnedValue[BookState] | None = None
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
    last_event_time_ns = 0
    last_mid: float | None = None
    last_gross: float | None = None
    last_net: float | None = None
    all_lifetime_count = 0
    all_lifetime_sum_ns = 0
    filled_lifetime_count = 0
    filled_lifetime_sum_ns = 0

    def finish_order(order: _MakerOrder, timestamp_ns: int, *, cancelled: bool) -> None:
        nonlocal all_lifetime_count, all_lifetime_sum_ns, cancelled_orders
        nonlocal filled_lifetime_count, filled_lifetime_sum_ns
        lifetime_ns = max(0, timestamp_ns - order.created_time_ns)
        all_lifetime_count += 1
        all_lifetime_sum_ns += lifetime_ns
        if order.had_fill:
            filled_lifetime_count += 1
            filled_lifetime_sum_ns += lifetime_ns
        if cancelled:
            cancelled_orders += 1

    with (
        closing_iterator(events),
        closing_iterator(active_decisions),
        _PendingMarkouts(metrics, on_fill) as pending,
        ExitStack() as cleanup,
    ):

        def cleanup_books() -> None:
            if latest_owned is not None:
                latest_owned.release(owner="crypto.execution.book")
            if next_event is not None and isinstance(next_event[1], OwnedValue):
                next_event[1].release(owner="crypto.execution.book")

        cleanup.callback(cleanup_books)
        next_event = next(events, None)

        def cancel_active(timestamp_ns: int) -> None:
            for order in active.values():
                finish_order(
                    order,
                    min(timestamp_ns, order.valid_until_ns),
                    cancelled=True,
                )
            active.clear()

        def expire_active(timestamp_ns: int) -> None:
            for side, order in tuple(active.items()):
                if timestamp_ns >= order.valid_until_ns:
                    active.pop(side)
                    finish_order(
                        order,
                        order.valid_until_ns,
                        cancelled=True,
                    )

        def process_book(
            event: BookState | OwnedValue[BookState] | BookBoundary,
        ) -> None:
            nonlocal last_mid, latest_state, latest_owned
            if isinstance(event, BookBoundary):
                cancel_active(event.observed_time_ns)
                if latest_owned is not None:
                    latest_owned.release(owner="crypto.execution.book")
                latest_owned = None
                latest_state = None
                last_mid = None
                pending.expire_all()
                return
            previous_state = latest_state
            state = _book_value(event)
            if (
                latest_state is not None
                and state.segment_id != latest_state.segment_id
            ):
                cancel_active(state.observed_time_ns)
            elif latest_state is not None:
                desired = {
                    "buy": state.bids[0][0],
                    "sell": state.asks[0][0],
                }
                for side, price_tick in desired.items():
                    order = active.get(side)
                    if order is not None and order.price_tick != price_tick:
                        active.pop(side)
                        finish_order(
                            order,
                            state.observed_time_ns,
                            cancelled=True,
                        )
            if latest_owned is not None:
                latest_owned.release(owner="crypto.execution.book")
            latest_owned = event if isinstance(event, OwnedValue) else None
            latest_state = state
            last_mid = _mid(state)
            _resolve_observed_markouts(
                pending,
                state,
                observed_coverage,
                previous_state=previous_state,
            )

        def process_trade(trade: TradeEvent) -> None:
            nonlocal cash_gross, fees, filled_orders, partial_fill_events
            nonlocal max_abs_position, position, trade_through_fill_events
            if latest_state is None:
                return
            trade_tick = trade.price_tick
            quantity = trade.visible_quantity
            maker_buyer = trade.buyer_is_maker
            timestamp_ns = trade.observed_time_ns
            if quantity <= 0:
                return
            if not _covered(
                observed_coverage,
                latest_state.segment_id,
                timestamp_ns,
            ):
                return
            if not _covered(trade_coverage, trade.connection_id, timestamp_ns):
                return
            mid = _mid(latest_state)
            for side in ("buy", "sell"):
                order = active.get(side)
                if (
                    order is None
                    or order.segment_id != latest_state.segment_id
                    or order.trade_connection_id != trade.connection_id
                ):
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
                    "timestamp_ms": timestamp_ns // _NANOSECONDS_PER_MILLISECOND,
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
                        timestamp_ns // _NANOSECONDS_PER_MILLISECOND
                        + config.markout_horizon_ms
                    ),
                    "markout_target_time_ns": (
                        timestamp_ns
                        + config.markout_horizon_ms
                        * _NANOSECONDS_PER_MILLISECOND
                    ),
                }
                position += signed * fill_quantity
                cash_gross -= signed * notional
                fees += fee
                max_abs_position = max(max_abs_position, abs(position))
                if not order.had_fill:
                    order.had_fill = True
                    filled_orders += 1
                order.remaining -= fill_quantity
                if fill_quantity < remaining_before:
                    partial_fill_events += 1
                if trade_through:
                    trade_through_fill_events += 1
                metrics.observe_fill(fill)
                pending.add(fill)
                if order.remaining <= 0:
                    active.pop(side, None)
                    finish_order(order, timestamp_ns, cancelled=False)
                break

        def process_until(target: int) -> None:
            nonlocal last_event_time_ns, market_event_count, next_event
            while next_event is not None and next_event[0][0] <= target:
                market_event_count += 1
                last_event_time_ns = next_event[0][0]
                expire_active(last_event_time_ns)
                _, item = next_event
                if isinstance(item, TradeEvent):
                    process_trade(item)
                else:
                    process_book(item)
                next_event = next(events, None)

        for row in active_decisions:
            decision_time = row["timestamp_ms"]
            decision_time_ns = decision_time * _NANOSECONDS_PER_MILLISECOND
            process_until(decision_time_ns)
            expire_active(decision_time_ns)
            current_mid: float | None = None
            if (
                latest_state is None
                or not _covered(
                    observed_coverage,
                    latest_state.segment_id,
                    decision_time_ns,
                )
                or row["book_segment_id"] != latest_state.segment_id
                or not _covered(
                    trade_coverage,
                    row["trade_connection_id"],
                    decision_time_ns,
                )
            ):
                cancel_active(decision_time_ns)
            else:
                if any(
                    order.trade_connection_id != row["trade_connection_id"]
                    for order in active.values()
                ):
                    cancel_active(decision_time_ns)
                current_mid = _mid(latest_state)
                desired = {
                    "buy": latest_state.bids[0][0],
                    "sell": latest_state.asks[0][0],
                }
                for side, price_tick in desired.items():
                    order = active.get(side)
                    if order is not None and order.price_tick != price_tick:
                        active.pop(side)
                        finish_order(order, decision_time_ns, cancelled=True)
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
                    book_bounds = observed_coverage[latest_state.segment_id]
                    trade_bounds = trade_coverage[row["trade_connection_id"]]
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
                        trade_connection_id=row["trade_connection_id"],
                        tick_size=latest_state.tick_size,
                        created_time_ns=decision_time_ns,
                        valid_until_ns=min(book_bounds[1], trade_bounds[1]),
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
            process_until(next_event[0][0])
        for order in active.values():
            finish_order(order, order.valid_until_ns, cancelled=True)
        active.clear()
        if latest_state is not None:
            last_mid = _mid(latest_state)
            last_gross = cash_gross + position * last_mid
            last_net = last_gross - fees
        else:
            last_gross = None
            last_net = None
        if latest_owned is not None:
            latest_owned.release(owner="crypto.execution.book")
        latest_owned = None
    return {
        "simulation": "event_ordered_public_l2_maker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "zero_fill_orders": created_orders - filled_orders,
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
            all_lifetime_sum_ns
            / _NANOSECONDS_PER_MILLISECOND
            / all_lifetime_count
            if all_lifetime_count
            else None
        ),
        "average_filled_order_lifetime_ms": (
            filled_lifetime_sum_ns
            / _NANOSECONDS_PER_MILLISECOND
            / filled_lifetime_count
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
        "same_timestamp_policy": "local_observation_order",
        "maker_policy_id": "public_l2_inventory_maker_baseline",
        "maker_prediction_used": False,
        "assumption": "historical_book_zero_market_impact",
        **metrics.payload(),
    }
