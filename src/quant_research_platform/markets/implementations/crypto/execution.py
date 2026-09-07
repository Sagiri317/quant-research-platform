"""Research-only event-time taker and public-L2 maker simulations."""

from __future__ import annotations

import json
import struct
import tempfile
from array import array
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quant_research_platform.contracts.json_codec import (
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    tick_to_price,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_ECONOMIC_SUMMARY_FIELDS,
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.validation import (
    clear_exception_frames,
    strict_finite_float,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.research.protocols.crypto_hft import HFT_MODEL_HORIZON_MS
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    require_free_disk_headroom,
)

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_UTC_DAY_NS = 86_400_000_000_000
_PENDING_MARKOUT_DISK_CHECK_BYTES = 1024 * 1024
_PENDING_MARKOUT_LENGTH = struct.Struct(">Q")
_PENDING_MARKOUT_JSON_OVERHEAD_BYTES = 64 * 1024
_EXECUTION_WORKSPACE_BYTES = 256 * 1024


@dataclass(frozen=True)
class TakerConfig:
    order_quantity: float
    position_size: float
    latency_ms: int
    taker_fee_bps: float
    markout_horizon_ms: int

    def __post_init__(self) -> None:
        _positive(self.order_quantity, "hft_taker_order_quantity_invalid")
        _positive(self.position_size, "hft_taker_position_size_invalid")
        _nonnegative_integer(self.latency_ms, "hft_taker_latency_invalid")
        _nonnegative(self.taker_fee_bps, "hft_taker_fee_invalid")
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
    latency_ms: int

    def __post_init__(self) -> None:
        _positive(self.order_quantity, "hft_maker_order_quantity_invalid")
        _positive(self.max_position, "hft_maker_position_size_invalid")
        strict_finite_float(self.maker_fee_bps, error="hft_maker_fee_invalid")
        target = strict_finite_float(
            self.inventory_target,
            error="hft_maker_inventory_target_invalid",
        )
        if abs(target) > self.max_position:
            raise ValueError("hft_maker_inventory_target_invalid")
        _nonnegative(self.inventory_skew, "hft_maker_inventory_skew_invalid")
        _positive_integer(
            self.markout_horizon_ms,
            "hft_markout_horizon_invalid",
        )
        _nonnegative_integer(self.latency_ms, "hft_maker_latency_invalid")


def _positive(value: object, error: str) -> float:
    number = strict_finite_float(value, error=error)
    if number <= 0:
        raise ValueError(error)
    return number


def _nonnegative(value: object, error: str) -> float:
    number = strict_finite_float(value, error=error)
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


def _funding_consumer(events: Iterable[FundingEvent], economics: PortfolioEconomics):
    if type(events) is not tuple or any(
        type(event) is not FundingEvent for event in events
    ):
        raise ValueError("portfolio_funding_schedule_invalid")
    timestamps = tuple(event.timestamp_ns for event in events)
    if timestamps != tuple(sorted(set(timestamps))):
        raise ValueError("portfolio_funding_schedule_invalid")
    pending = iter(events)
    next_event = next(pending, None)

    def consume(timestamp_ns: int) -> None:
        nonlocal next_event
        while next_event is not None and next_event.timestamp_ns <= timestamp_ns:
            economics.funding(next_event)
            next_event = next(pending, None)

    return consume


def _pending_markout_max_json_bytes(fill: Mapping[str, Any]) -> int:
    total = 2
    for index, (key, value) in enumerate(fill.items()):
        total += (1 if index else 0) + 2 + 12 * len(key) + 1
        if isinstance(value, str):
            total += 2 + 12 * len(value)
        elif value is None:
            total += 4
        elif isinstance(value, bool):
            total += 5
        elif type(value) is int:
            digits = max(1, (abs(value).bit_length() * 30103) // 100000 + 1)
            total += digits + (1 if value < 0 else 0)
        elif type(value) is float:
            total += 24
        else:
            raise TypeError("hft_pending_markout_value_invalid")
    return total


@dataclass
class _Metrics:
    fill_events: int = 0
    fill_quantity: float = 0.0
    turnover: float = 0.0
    slippage_quantity: float = 0.0
    slippage_weighted: float = 0.0
    spread_capture_pnl: float = 0.0
    defined_markout_quantity: float = 0.0
    markout_weighted: float = 0.0
    markout_weighted_bps: float = 0.0
    markout_pnl: float = 0.0
    adverse_selection_cost: float = 0.0
    undefined_markouts: int = 0

    def observe_fill(self, fill: Mapping[str, Any]) -> None:
        quantity = float(fill["fill_qty"])
        self.fill_events += 1
        self.fill_quantity += quantity
        self.turnover += float(fill["notional"])
        if "slippage_bps" in fill:
            self.slippage_quantity += quantity
            self.slippage_weighted += quantity * float(fill["slippage_bps"])
        if "spread_capture_per_unit" in fill:
            self.spread_capture_pnl += quantity * float(fill["spread_capture_per_unit"])

    def finalize_markout(self, fill: Mapping[str, Any]) -> None:
        quantity = float(fill["fill_qty"])
        markout = fill.get("markout_per_unit")
        if markout is None:
            self.undefined_markouts += 1
            return
        value = float(markout)
        self.defined_markout_quantity += quantity
        self.markout_weighted += quantity * value
        self.markout_weighted_bps += quantity * float(fill["markout_bps"])
        self.markout_pnl += float(fill["markout_pnl"])
        self.adverse_selection_cost += float(fill["adverse_selection_cost"])

    def payload(self) -> dict[str, Any]:
        return {
            "fill_events": self.fill_events,
            "fill_quantity": self.fill_quantity,
            "turnover": self.turnover,
            "average_slippage_bps": (
                self.slippage_weighted / self.slippage_quantity
                if self.slippage_quantity
                else None
            ),
            "spread_capture_pnl": self.spread_capture_pnl,
            "average_markout_per_unit": (
                self.markout_weighted / self.defined_markout_quantity
                if self.defined_markout_quantity
                else None
            ),
            "average_markout_bps": (
                self.markout_weighted_bps / self.defined_markout_quantity
                if self.defined_markout_quantity
                else None
            ),
            "markout_pnl": (
                self.markout_pnl if self.defined_markout_quantity else None
            ),
            "adverse_selection_cost": (
                self.adverse_selection_cost if self.defined_markout_quantity else None
            ),
            "markout_undefined_fill_events": self.undefined_markouts,
        }


class _PendingMarkouts:
    """Disk-backed FIFO for exact markouts without retained fill history."""

    def __init__(
        self,
        metrics: _Metrics,
        on_fill: Callable[[Mapping[str, Any]], None] | None,
        resource_policy: RuntimeResourcePolicy,
    ) -> None:
        self._temp_root = Path(tempfile.gettempdir())
        require_free_disk_headroom(
            self._temp_root,
            reserve_bytes=2 * _PENDING_MARKOUT_DISK_CHECK_BYTES,
        )
        self._disk_bytes_remaining = _PENDING_MARKOUT_DISK_CHECK_BYTES
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._read_offset = 0
        self._next: tuple[OwnedValue[dict[str, Any]], int] | None = None
        self._metrics = metrics
        self._on_fill = on_fill
        self._resource_policy = resource_policy

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
            if primary is None:
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
            self._discard_next()
        except BaseException as release_error:
            note = (
                "execution pending markout lease release also failed: "
                f"{type(release_error).__name__}"
            )
            if primary is not None:
                primary.add_note(note)
            elif cleanup_failure is None:
                cleanup_failure = release_error
            else:
                cleanup_failure.add_note(note)
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
        max_payload_bytes = _pending_markout_max_json_bytes(fill)
        working_set_bytes = (
            json_record_working_set_bytes(max_payload_bytes)
            + _PENDING_MARKOUT_JSON_OVERHEAD_BYTES
        )
        with self._resource_policy.lease(
            "crypto.execution.pending_markout.encode",
            working_set_bytes,
        ):
            payload = json.dumps(
                fill,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if len(payload) > max_payload_bytes:
                raise AssertionError("hft_pending_markout_json_bound_invalid")
            record_bytes = _PENDING_MARKOUT_LENGTH.size + len(payload)
            if record_bytes > self._disk_bytes_remaining:
                admitted = max(_PENDING_MARKOUT_DISK_CHECK_BYTES, record_bytes)
                require_free_disk_headroom(
                    self._temp_root,
                    reserve_bytes=2 * admitted,
                )
                self._disk_bytes_remaining = admitted
            self._disk_bytes_remaining -= record_bytes
            self._file.seek(0, 2)
            self._file.write(_PENDING_MARKOUT_LENGTH.pack(len(payload)))
            self._file.write(payload)
            self._file.flush()
            del payload

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
                try:
                    self._finalize(fill, future_mid=None)
                finally:
                    del fill
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
            try:
                self._finalize(fill, future_mid=markout_mid)
            finally:
                del fill
                self._pop()

    def expire_all(self) -> None:
        while (fill := self._peek()) is not None:
            try:
                self._finalize(fill, future_mid=None)
            finally:
                del fill
                self._pop()

    def _peek(self) -> dict[str, Any] | None:
        if self._next is not None:
            return self._next[0].value
        self._file.seek(self._read_offset)
        header = self._file.read(_PENDING_MARKOUT_LENGTH.size)
        if not header:
            return None
        if len(header) != _PENDING_MARKOUT_LENGTH.size:
            raise RuntimeError("hft_pending_markout_record_invalid")
        payload_bytes = _PENDING_MARKOUT_LENGTH.unpack(header)[0]
        del header
        if payload_bytes <= 0:
            raise RuntimeError("hft_pending_markout_record_invalid")
        working_set_bytes = (
            json_record_working_set_bytes(payload_bytes)
            + _PENDING_MARKOUT_JSON_OVERHEAD_BYTES
        )
        lease = self._resource_policy.lease(
            "crypto.execution.pending_markout.decode",
            working_set_bytes,
        )
        try:
            payload = self._file.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise RuntimeError("hft_pending_markout_record_invalid")
            try:
                fill = parse_qrp_json_object(payload)
            except ValueError as error:
                raise RuntimeError("hft_pending_markout_record_invalid") from error
            lease.transfer("crypto.execution.pending_markout")
            self._next = (OwnedValue(fill, lease), self._file.tell())
            return fill
        except BaseException as primary:
            try:
                lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "execution pending markout lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise

    def _pop(self) -> None:
        assert self._next is not None
        owned, self._read_offset = self._next
        self._next = None
        owned.release(owner="crypto.execution.pending_markout")

    def _discard_next(self) -> None:
        if self._next is None:
            return
        owned = self._next[0]
        self._next = None
        owned.release(owner="crypto.execution.pending_markout")

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
            fill["markout_per_unit"] = signed_markout
            fill["markout_bps"] = signed_markout / fill["fill_price"] * 10_000.0
            fill["markout_pnl"] = signed_markout * fill["fill_qty"]
            fill["adverse_selection_cost"] = (
                max(0.0, -signed_markout) * fill["fill_qty"]
            )
        else:
            fill["markout_per_unit"] = None
            fill["markout_bps"] = None
            fill["markout_pnl"] = None
            fill["adverse_selection_cost"] = None
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


def _taker_order_quantity(
    state, *, prediction, config, buy_limit, sell_limit, funding_per_unit
):
    """Maximize a piecewise-linear gain over the observed continuous depth domain."""
    mid = _mid(state)
    forecast = _positive(mid * (1.0 + prediction), "hft_prediction_price_invalid")
    best_quantity = best_gain = 0.0
    for sign, levels, limit in (
        (1.0, state.asks, buy_limit),
        (-1.0, state.bids, sell_limit),
    ):
        remaining = min(config.order_quantity, limit)
        quantity = gain = 0.0
        for tick, available in levels:
            size = min(remaining, available)
            if size <= 0:
                continue
            price = tick_to_price(tick, state.tick_size)
            gain += size * (
                sign * (forecast - price + funding_per_unit)
                - price * config.taker_fee_bps / 10_000.0
            )
            quantity += size
            remaining -= size
            if gain > best_gain:
                best_quantity, best_gain = sign * quantity, gain
            if remaining <= 0:
                break
    return best_quantity, best_gain


def _record_execution_result(economics, result, timestamp, mark):
    if result["net_pnl"] is not None:
        diagnostics = {
            key: value
            for key, value in result.items()
            if key not in PORTFOLIO_ECONOMIC_SUMMARY_FIELDS
        }
        economics.mark(
            timestamp,
            mark,
            diagnostics=(
                PortfolioDiagnostic(
                    economics.asset_id if mark is not None else PORTFOLIO_CASH_ASSET_ID,
                    None,
                    "execution_summary",
                    text=json.dumps(diagnostics, sort_keys=True, allow_nan=False),
                ),
            ),
        )
    return result


def _advance_utc_boundaries(
    target, next_day, advance_events, economics, current_book, coverage
):
    while next_day is not None and next_day <= target:
        advance_events(next_day)
        book = current_book()
        mark = (
            _mid(book)
            if book is not None and _covered(coverage, book.segment_id, next_day)
            else None
        )
        if mark is not None or economics.quantity == 0:
            economics.mark(
                next_day,
                mark,
                diagnostics=(
                    PortfolioDiagnostic(
                        economics.asset_id
                        if mark is not None
                        else PORTFOLIO_CASH_ASSET_ID,
                        None,
                        "utc_day_boundary",
                        number=float(next_day // _UTC_DAY_NS),
                    ),
                ),
            )
        next_day += _UTC_DAY_NS
    advance_events(target)
    return next_day


def simulate_taker(
    book_events: Iterable[BookState | OwnedValue[BookState] | BookBoundary],
    decisions: Iterable[Mapping[str, Any]],
    *,
    prediction_column: str,
    prediction_horizon_ms: int,
    config: TakerConfig,
    observed_coverage: Mapping[str, tuple[int, int]],
    resource_policy: RuntimeResourcePolicy,
    capital: PortfolioCapital,
    asset_id: str,
    funding_events: Iterable[FundingEvent],
    funding_forecasts: Iterable[FundingForecast],
    decision_windows_ns: tuple[tuple[int, int], ...] | None = None,
    on_ledger: Callable[[PortfolioLedgerBatch], None] | None = None,
    on_fill: Callable[[Mapping[str, Any]], None] | None = None,
    on_equity: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Freeze the net-gain maximizing order at decision time, then execute later."""

    _positive_integer(prediction_horizon_ms, "hft_prediction_horizon_invalid")
    if decision_windows_ns is not None:
        previous_end = -1
        if type(decision_windows_ns) is not tuple or not decision_windows_ns:
            raise ValueError("hft_execution_windows_invalid")
        for window in decision_windows_ns:
            if (
                type(window) is not tuple
                or len(window) != 2
                or any(type(value) is not int for value in window)
                or not previous_end < window[0] <= window[1]
            ):
                raise ValueError("hft_execution_windows_invalid")
            previous_end = window[1]

    events = iter(_checked_book_events(book_events))
    active_decisions = iter(_checked_decisions(decisions))
    forecasts = iter(funding_forecasts)
    active_forecast = None
    next_forecast = None
    forecast_timestamp = -1
    next_event = None
    latest_state: BookState | None = None
    latest_owned: OwnedValue[BookState] | None = None
    terminal_mid: float | None = None
    terminal_timestamp = 0
    action_timestamp = 0
    metrics = _Metrics()
    economics = PortfolioEconomics(capital, asset_id, on_ledger)
    apply_funding = _funding_consumer(funding_events, economics)

    def resolved_fill(fill):
        economics.diagnostic(dict(fill))
        if on_fill is not None:
            on_fill(fill)

    blocked_decisions = undefined_decisions = 0
    orders = deque()
    reserved_buy = reserved_sell = 0.0
    created_orders = 0
    filled_orders = 0
    unfilled_quantity = 0.0
    equity_rows = 0
    max_abs_position = 0.0
    last_gross: float | None = None
    last_net: float | None = None
    market_event_count = 0
    window_index = 0
    window_open = False
    cancelled_orders = 0
    next_day = (
        None
        if decision_windows_ns is None
        else ((decision_windows_ns[0][0] + _UTC_DAY_NS - 1) // _UTC_DAY_NS)
        * _UTC_DAY_NS
    )

    with (
        resource_policy.lease(
            "crypto.execution.taker.workspace",
            _EXECUTION_WORKSPACE_BYTES,
        ),
        resource_policy.lease("crypto.execution.taker.orders", 1) as order_lease,
        closing_iterator(events),
        closing_iterator(active_decisions),
        closing_iterator(forecasts),
        _PendingMarkouts(metrics, resolved_fill, resource_policy) as pending,
    ):
        try:
            next_event = next(events, None)
            next_forecast = next(forecasts, None)

            def consume_event(
                event: BookState | OwnedValue[BookState] | BookBoundary,
            ) -> None:
                nonlocal latest_state, latest_owned, terminal_mid, terminal_timestamp
                terminal_timestamp = _book_key(event)[0]
                apply_funding(terminal_timestamp)
                if isinstance(event, BookBoundary):
                    if latest_owned is not None:
                        latest_owned.release(owner="crypto.execution.book")
                    latest_owned = None
                    latest_state = None
                    if economics.quantity != 0.0:
                        terminal_mid = None
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
                if (
                    decision_windows_ns is not None
                    and decision_windows_ns[0][0]
                    <= state.observed_time_ns
                    <= decision_windows_ns[-1][1]
                    and _covered(
                        observed_coverage, state.segment_id, state.observed_time_ns
                    )
                ):
                    economics.mark(state.observed_time_ns, _mid(state))

            def consume_until(target: int) -> None:
                nonlocal market_event_count, next_event
                while next_event is not None and _book_key(next_event)[0] <= target:
                    event = next_event
                    next_event = None
                    market_event_count += 1
                    consume_event(event)
                    next_event = next(events, None)

            def execute_order(order):
                nonlocal reserved_buy, reserved_sell, filled_orders, unfilled_quantity
                nonlocal equity_rows, max_abs_position, last_gross, last_net
                (
                    order_id,
                    decision_time,
                    effective_time,
                    segment,
                    signed_quantity,
                    gain,
                    decision_mid,
                ) = order
                effective_time_ns = effective_time * _NANOSECONDS_PER_MILLISECOND
                if signed_quantity > 0:
                    reserved_buy -= signed_quantity
                else:
                    reserved_sell -= -signed_quantity
                apply_funding(effective_time_ns)
                requested = abs(signed_quantity)
                remaining = requested
                current_mid = None
                if (
                    latest_state is not None
                    and latest_state.segment_id == segment
                    and _covered(observed_coverage, segment, effective_time_ns)
                ):
                    current_mid = _mid(latest_state)
                    levels = (
                        latest_state.asks if signed_quantity > 0 else latest_state.bids
                    )
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
                    if filled > 0:
                        filled_orders += 1
                        fill_price = notional / filled
                        fee = notional * config.taker_fee_bps / 10_000.0
                        signed = 1.0 if signed_quantity > 0 else -1.0
                        fill_id = economics.fill(
                            timestamp_ns=effective_time_ns,
                            order_id=str(order_id),
                            quantity=signed * filled,
                            price=fill_price,
                            mark_price=current_mid,
                            fee=fee,
                        )
                        fill = dict(
                            fill_id=fill_id,
                            order_id=order_id,
                            timestamp_ms=effective_time,
                            decision_time_ms=decision_time,
                            effective_time_ms=effective_time,
                            state_observed_time_ms=latest_state.observed_time_ms,
                            segment_id=segment,
                            side="buy" if signed_quantity > 0 else "sell",
                            fill_price=fill_price,
                            fill_qty=filled,
                            requested_quantity=requested,
                            decision_mid=decision_mid,
                            expected_net_gain=gain,
                            levels_consumed=levels_consumed,
                            notional=notional,
                            fee=fee,
                            slippage_bps=signed
                            * (fill_price - current_mid)
                            / current_mid
                            * 10_000.0,
                            markout_target_time_ms=effective_time
                            + config.markout_horizon_ms,
                            markout_target_time_ns=effective_time_ns
                            + config.markout_horizon_ms * _NANOSECONDS_PER_MILLISECOND,
                        )
                        metrics.observe_fill(fill)
                        pending.add(fill)
                unfilled_quantity += remaining
                marked = (
                    None
                    if current_mid is None
                    else economics.mark(effective_time_ns, current_mid)
                )
                last_gross = None if marked is None else marked["gross_pnl"]
                last_net = None if marked is None else marked["net_pnl"]
                max_abs_position = max(max_abs_position, abs(economics.quantity))

            def advance_events(target_ns):
                nonlocal action_timestamp
                while (
                    orders and orders[0][2] * _NANOSECONDS_PER_MILLISECOND <= target_ns
                ):
                    consume_until(orders[0][2] * _NANOSECONDS_PER_MILLISECOND)
                    order = orders.popleft()
                    try:
                        execute_order(order)
                    finally:
                        order = None
                    order_lease.shrink_to(max(1, len(orders) * 2048))
                consume_until(target_ns)
                apply_funding(target_ns)
                action_timestamp = target_ns

            def advance(target_ns):
                nonlocal next_day
                next_day = _advance_utc_boundaries(
                    target_ns,
                    next_day,
                    advance_events,
                    economics,
                    lambda: latest_state,
                    observed_coverage,
                )

            def boundary(timestamp_ns, *, closing):
                nonlocal reserved_buy, reserved_sell, cancelled_orders
                nonlocal unfilled_quantity
                advance(timestamp_ns)
                cancelled_count = len(orders) if closing else 0
                cancelled_quantity = (
                    sum(abs(order[4]) for order in orders) if closing else 0.0
                )
                if closing:
                    cancelled_orders += len(orders)
                    unfilled_quantity += cancelled_quantity
                    orders.clear()
                    reserved_buy = reserved_sell = 0.0
                    order_lease.shrink_to(1)
                mark = (
                    _mid(latest_state)
                    if latest_state is not None
                    and _covered(
                        observed_coverage, latest_state.segment_id, timestamp_ns
                    )
                    else None
                )
                if mark is None and economics.quantity != 0.0:
                    raise PortfolioTargetBlocked(
                        "portfolio_window_boundary_valuation_missing"
                    )
                diagnostic_asset = (
                    asset_id if mark is not None else PORTFOLIO_CASH_ASSET_ID
                )
                economics.mark(
                    timestamp_ns,
                    mark,
                    diagnostics=(
                        PortfolioDiagnostic(
                            diagnostic_asset,
                            None,
                            "oos_window_end" if closing else "oos_window_start",
                            number=float(window_index),
                        ),
                        *(
                            (
                                PortfolioDiagnostic(
                                    diagnostic_asset,
                                    None,
                                    "cancelled_order_count",
                                    number=float(cancelled_count),
                                ),
                                PortfolioDiagnostic(
                                    diagnostic_asset,
                                    None,
                                    "cancelled_order_quantity",
                                    number=float(cancelled_quantity),
                                ),
                            )
                            if closing
                            else ()
                        ),
                    ),
                )

            for row in active_decisions:
                decision_time = row["timestamp_ms"]
                decision_ns = decision_time * _NANOSECONDS_PER_MILLISECOND
                if decision_windows_ns is not None:
                    while (
                        window_index < len(decision_windows_ns)
                        and decision_ns > decision_windows_ns[window_index][1]
                    ):
                        if not window_open:
                            boundary(
                                decision_windows_ns[window_index][0], closing=False
                            )
                        boundary(decision_windows_ns[window_index][1], closing=True)
                        window_index += 1
                        window_open = False
                    if (
                        window_index == len(decision_windows_ns)
                        or decision_ns < decision_windows_ns[window_index][0]
                    ):
                        raise ValueError("hft_decision_outside_oos_window")
                    if not window_open:
                        boundary(decision_windows_ns[window_index][0], closing=False)
                        window_open = True
                prediction = strict_finite_float(
                    row.get(prediction_column), error="hft_prediction_invalid"
                )
                advance(decision_ns)
                while next_forecast is not None:
                    if (
                        type(next_forecast) is not FundingForecast
                        or next_forecast.available_at_ns <= forecast_timestamp
                    ):
                        raise ValueError("portfolio_funding_forecast_order_invalid")
                    if next_forecast.available_at_ns > decision_ns:
                        break
                    active_forecast = next_forecast
                    forecast_timestamp = active_forecast.available_at_ns
                    next_forecast = next(forecasts, None)
                decision_mid = None
                selected = 0.0
                gain = None
                funding_per_unit = None
                segment = None
                reason = "decision_book_unavailable"
                if latest_state is not None and _covered(
                    observed_coverage, latest_state.segment_id, decision_ns
                ):
                    decision_mid = _mid(latest_state)
                    segment = row.get(
                        "book_segment_id",
                        row.get("segment_id", latest_state.segment_id),
                    )
                    reason = "decision_segment_mismatch"
                    if segment == latest_state.segment_id:
                        reason = "forecast_expired_before_execution"
                        if config.latency_ms < prediction_horizon_ms:
                            forecast_end = (
                                decision_time + prediction_horizon_ms
                            ) * _NANOSECONDS_PER_MILLISECOND
                            funding_per_unit = None
                            if (
                                active_forecast is not None
                                and active_forecast.known_until_ns >= forecast_end
                            ):
                                cashflows = tuple(
                                    value
                                    for timestamp, value in active_forecast.cashflows
                                    if (decision_time + config.latency_ms)
                                    * _NANOSECONDS_PER_MILLISECOND
                                    < timestamp
                                    <= forecast_end
                                )
                                if all(value is not None for value in cashflows):
                                    funding_per_unit = sum(cashflows)
                            if funding_per_unit is None:
                                blocked_decisions += 1
                                reason = "portfolio_expected_funding_missing"
                            else:
                                selected, gain = _taker_order_quantity(
                                    latest_state,
                                    prediction=prediction,
                                    config=config,
                                    buy_limit=max(
                                        0.0,
                                        config.position_size
                                        - economics.quantity
                                        - reserved_buy,
                                    ),
                                    sell_limit=max(
                                        0.0,
                                        config.position_size
                                        + economics.quantity
                                        - reserved_sell,
                                    ),
                                    funding_per_unit=funding_per_unit,
                                )
                                reason = (
                                    "positive_expected_net_gain"
                                    if selected
                                    else "no_positive_expected_net_gain"
                                )
                if gain is None and reason != "portfolio_expected_funding_missing":
                    undefined_decisions += 1
                marked = None
                if decision_mid is not None:
                    marked = economics.mark(
                        decision_ns,
                        decision_mid,
                        diagnostics=(
                            PortfolioDiagnostic(
                                asset_id,
                                None,
                                "decision_requested_quantity",
                                number=selected,
                            ),
                            PortfolioDiagnostic(
                                asset_id, None, "expected_net_gain", number=gain
                            ),
                            PortfolioDiagnostic(
                                asset_id,
                                None,
                                "expected_funding_cashflow",
                                number=None
                                if funding_per_unit is None
                                else selected * funding_per_unit,
                            ),
                            PortfolioDiagnostic(
                                asset_id,
                                None,
                                "prediction_horizon_ms",
                                number=float(prediction_horizon_ms),
                            ),
                            PortfolioDiagnostic(
                                asset_id, None, "decision_reason", text=reason
                            ),
                        ),
                    )
                last_gross = None if marked is None else marked["gross_pnl"]
                last_net = None if marked is None else marked["net_pnl"]
                if on_equity is not None:
                    on_equity(
                        dict(
                            timestamp_ms=decision_time,
                            decision_time_ms=decision_time,
                            effective_time_ms=decision_time + config.latency_ms,
                            state_observed_time_ms=None
                            if latest_state is None
                            else latest_state.observed_time_ms,
                            position=economics.quantity,
                            gross_pnl=last_gross,
                            fees=-economics.cost_cashflows,
                            net_pnl=last_net,
                            requested_quantity=selected,
                            expected_net_gain=gain,
                            decision_reason=reason,
                        )
                    )
                equity_rows += 1
                if selected:
                    required = (len(orders) + 1) * 2048
                    if required > order_lease.reserved_bytes:
                        order_lease.grow(required - order_lease.reserved_bytes)
                    created_orders += 1
                    orders.append(
                        (
                            created_orders,
                            decision_time,
                            decision_time + config.latency_ms,
                            segment,
                            selected,
                            gain,
                            decision_mid,
                        )
                    )
                    if selected > 0:
                        reserved_buy += selected
                    else:
                        reserved_sell -= selected
                    advance(decision_ns)
            if decision_windows_ns is not None:
                while window_index < len(decision_windows_ns):
                    if not window_open:
                        boundary(decision_windows_ns[window_index][0], closing=False)
                    boundary(decision_windows_ns[window_index][1], closing=True)
                    window_index += 1
                    window_open = False
            else:
                while orders:
                    advance(orders[0][2] * _NANOSECONDS_PER_MILLISECOND)
                while next_event is not None:
                    consume_until(_book_key(next_event)[0])
            end_timestamp = max(
                action_timestamp, terminal_timestamp, economics.last_timestamp
            )
            pending.expire_all()
            terminal_mid = (
                _mid(latest_state)
                if latest_state is not None
                and _covered(observed_coverage, latest_state.segment_id, end_timestamp)
                else None
            )
            if terminal_mid is None and economics.quantity != 0.0:
                last_gross = None
                last_net = None
            else:
                marked = economics.mark(
                    end_timestamp,
                    terminal_mid,
                )
                last_gross, last_net = marked["gross_pnl"], marked["net_pnl"]
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            orders.clear()
            latest_state = None
            if latest_owned is not None:
                latest_owned.release(owner="crypto.execution.book")
            if isinstance(next_event, OwnedValue):
                next_event.release(owner="crypto.execution.book")
    result = {
        "simulation": "decision_time_net_gain_taker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "zero_fill_orders": created_orders - filled_orders,
        "unfilled_quantity": unfilled_quantity,
        "equity_rows": equity_rows,
        "gross_pnl": last_gross,
        "net_pnl": last_net,
        "max_abs_position": max_abs_position,
        "market_event_count": market_event_count,
        "prediction_horizon_ms": prediction_horizon_ms,
        "selection_status": "BLOCKED"
        if blocked_decisions
        else "UNDEFINED"
        if undefined_decisions
        else "COMPLETE",
        "blocked_decisions": blocked_decisions,
        "undefined_decisions": undefined_decisions,
        "expected_gain_basis": (
            "decision_forecast_mark_minus_l2_entry_and_fee_plus_known_funding"
        ),
        "quantity_domain": "continuous_l2_depth_with_order_and_pending_position_bounds",
        "latency_ms": config.latency_ms,
        "taker_fee_bps": config.taker_fee_bps,
        "order_quantity": config.order_quantity,
        "position_size": config.position_size,
        "markout_horizon_ms": config.markout_horizon_ms,
        "latency_fidelity": "full_l2_observed_event",
        "assumption": "historical_book_zero_market_impact",
        **metrics.payload(),
        "fees": -economics.cost_cashflows,
        "funding": economics.funding_cashflows,
        "initial_capital": capital.initial_capital,
        "valuation_currency": capital.valuation_currency,
        "candidate_id": capital.candidate_id,
        "nav": None if last_net is None else capital.initial_capital + last_net,
        "capital_return": None
        if last_net is None
        else last_net / capital.initial_capital,
    }
    return _record_execution_result(economics, result, end_timestamp, terminal_mid)


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
    effective_at_ns: int
    had_fill: bool = False


def _maker_join(order, state, observed_coverage, trade_coverage):
    if (
        state is None
        or state.segment_id != order.segment_id
        or not _covered(observed_coverage, order.segment_id, order.effective_at_ns)
        or not _covered(
            trade_coverage, order.trade_connection_id, order.effective_at_ns
        )
    ):
        return "activation_coverage_missing"
    if (order.side == "buy" and order.price_tick >= state.asks[0][0]) or (
        order.side == "sell" and order.price_tick <= state.bids[0][0]
    ):
        return "post_only_crossing_rejected"
    levels = state.bids if order.side == "buy" else state.asks
    if (order.side == "buy" and order.price_tick < levels[-1][0]) or (
        order.side == "sell" and order.price_tick > levels[-1][0]
    ):
        return "activation_queue_depth_unknown"
    order.queue_ahead = sum(
        quantity for tick, quantity in levels if tick == order.price_tick
    )
    order.initial_queue_ahead = order.queue_ahead
    return None


def _maker_trade_fill(order, trade):
    if trade.visible_quantity <= 0:
        return 0.0, "zero_visible_trade_quantity"
    if trade.connection_id != order.trade_connection_id:
        return 0.0, "trade_connection_mismatch"
    same_direction = (order.side == "buy") == trade.buyer_is_maker
    if not same_direction:
        return 0.0, "opposite_trade_direction"
    through = (order.side == "buy" and trade.price_tick < order.price_tick) or (
        order.side == "sell" and trade.price_tick > order.price_tick
    )
    if through:
        order.queue_ahead = 0.0
        return order.remaining, "trade_through"
    if trade.price_tick != order.price_tick:
        return 0.0, "different_trade_price"
    consumed = min(order.queue_ahead, trade.visible_quantity)
    order.queue_ahead -= consumed
    return min(
        order.remaining, trade.visible_quantity - consumed
    ), "same_price_depletion"


def _maker_new_order(
    order_id, side, quantity, state, row, config, observed_coverage, trade_coverage
):
    timestamp = row["timestamp_ms"] * _NANOSECONDS_PER_MILLISECOND
    return _MakerOrder(
        order_id=order_id,
        side=side,
        price_tick=(state.bids if side == "buy" else state.asks)[0][0],
        remaining=quantity,
        queue_ahead=0.0,
        initial_queue_ahead=0.0,
        segment_id=row["book_segment_id"],
        trade_connection_id=row["trade_connection_id"],
        tick_size=state.tick_size,
        created_time_ns=timestamp,
        valid_until_ns=min(
            timestamp + HFT_MODEL_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND,
            observed_coverage[row["book_segment_id"]][1],
            trade_coverage[row["trade_connection_id"]][1],
        ),
        effective_at_ns=timestamp + config.latency_ms * _NANOSECONDS_PER_MILLISECOND,
    )


class MakerCalibration:
    """Frozen paired responses with only the full-horizon curve retained."""

    def __init__(self, quantities, errors, fills, *, resource_policy):
        self.quantities = quantities
        self.errors = errors
        self.fills = fills
        self.policy = resource_policy
        self.complete_curve = None
        self.lease = resource_policy.lease(
            "crypto.maker.calibration.curve",
            max(1, len(quantities) * 96 + len(fills) * 40),
        )
        try:
            self.complete_curve = self._compress(quantities)
            self.lease.shrink_to(
                max(1, sum(value.nbytes for value in self.complete_curve))
            )
        except BaseException:
            self.complete_curve = None
            _close_lease_preserving_primary(
                self.lease, owner="crypto.maker.calibration.curve"
            )
            raise

    def close(self):
        self.complete_curve = None
        self.quantities = self.errors = self.fills = None
        self.lease.close()

    def _compress(self, quantities):
        values, inverse, counts = np.unique(
            quantities, return_inverse=True, return_counts=True
        )
        errors = np.bincount(inverse, weights=self.errors, minlength=len(values))
        return values, counts, errors

    def _curve(self, lifetime_ns):
        if lifetime_ns >= HFT_MODEL_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND:
            return self.complete_curve
        quantities = np.zeros(len(self.quantities))
        selected = self.fills[:, 1] < lifetime_ns
        np.maximum.at(
            quantities,
            self.fills[selected, 0].astype(np.int64),
            self.fills[selected, 2],
        )
        return self._compress(quantities)

    def select(
        self,
        *,
        limit,
        side,
        mid,
        price,
        prediction,
        fee_bps,
        lifetime_ns,
        funding,
        fixed_quantity=None,
    ):
        with self.policy.lease(
            "crypto.maker.calibration.query",
            max(1, (len(self.quantities) + len(self.fills)) * 256),
        ):
            # Every truncated response breakpoint is already a cumulative fill size.
            choices = (
                np.array([0.0, fixed_quantity])
                if fixed_quantity is not None
                else np.unique(
                    np.concatenate(([0.0, limit], self.quantities, self.fills[:, 2]))
                )
            )
            choices = choices[(choices >= 0.0) & (choices <= limit)]
            sign = 1.0 if side == "buy" else -1.0
            forecast = _positive(
                mid * (1.0 + prediction), "hft_prediction_price_invalid"
            )
            edge = sign * (forecast - price) - price * fee_bps / 10_000.0

            def response(curve, weights):
                filled, _count, _errors = curve
                weighted = np.concatenate(([0.0], np.cumsum(filled * weights)))
                total = np.concatenate(([0.0], np.cumsum(weights)))
                indexes = np.searchsorted(filled, choices, side="right")
                return weighted[indexes] + choices * (total[-1] - total[indexes])

            curve = self._curve(lifetime_ns)
            gains = response(curve, curve[1] * edge + sign * mid * curve[2])
            del curve
            for elapsed, cash in funding:
                curve = self._curve(min(lifetime_ns, elapsed))
                gains += sign * cash * response(curve, curve[1])
                del curve
            gains /= len(self.quantities)
            index = int(np.argmax(gains))
            return float(choices[index]), float(gains[index])


def replay_maker_calibration(
    book_events,
    trades,
    decisions,
    *,
    config,
    observed_coverage,
    trade_coverage,
    cutoff_ns,
    resource_policy,
    on_sample,
):
    """Replay independent validation quotes with the same passive fill primitives."""
    events = iter(_market_events(book_events, trades))
    rows = iter(_checked_quote_schedule(decisions))
    latest_state = latest_owned = next_event = None
    samples = []
    counts = {"buy": 0, "sell": 0}
    excluded = 0
    with (
        closing_iterator(events),
        closing_iterator(rows),
        resource_policy.lease("crypto.maker.calibration.pending", 1) as lease,
        ExitStack() as cleanup,
    ):

        def cleanup_books():
            if latest_owned is not None:
                latest_owned.release(owner="crypto.maker.calibration.book")
            if next_event is not None and isinstance(next_event[1], OwnedValue):
                next_event[1].release(owner="crypto.maker.calibration.book")

        cleanup.callback(cleanup_books)
        next_event = next(events, None)

        def finish(sample, reason):
            order = sample["order"]
            on_sample(
                side=order.side,
                sample_id=sample["sample_id"],
                decision_ns=order.created_time_ns,
                prediction_end_ns=order.valid_until_ns,
                available_ns=sample["available_ns"],
                prediction=sample["prediction"],
                observed_return=sample["observed_return"],
                filled=config.order_quantity - order.remaining,
                quote_quantity=config.order_quantity,
                terminal_reason=reason,
                fills=sample["fills"],
            )
            samples.remove(sample)

        def release_completed():
            lease.shrink_to(
                max(1, sum(2048 + len(sample["fills"]) * 16 for sample in samples))
            )

        def process_until(target):
            nonlocal latest_state, latest_owned, next_event
            while True:
                timers = [
                    min(
                        sample["order"].valid_until_ns,
                        sample["cancel_at"]
                        if sample["cancel_at"] is not None
                        else sample["order"].valid_until_ns,
                        sample["order"].effective_at_ns
                        if not sample["activated"]
                        else sample["order"].valid_until_ns,
                    )
                    for sample in samples
                ]
                event_time = next_event[0][0] if next_event is not None else None
                due = min(
                    (*timers, *((event_time,) if event_time is not None else ())),
                    default=None,
                )
                if due is None or due > target:
                    break
                for sample in tuple(samples):
                    if sample["order"].valid_until_ns <= due:
                        finish(sample, "expired")
                if event_time == due:
                    event = next_event[1]
                    if isinstance(event, BookBoundary):
                        for sample in tuple(samples):
                            finish(sample, "book_boundary")
                        if latest_owned is not None:
                            latest_owned.release(owner="crypto.maker.calibration.book")
                        latest_owned = latest_state = None
                    elif isinstance(event, TradeEvent):
                        for sample in tuple(samples):
                            order = sample["order"]
                            if (
                                not sample["activated"]
                                or latest_state is None
                                or latest_state.segment_id != order.segment_id
                            ):
                                continue
                            quantity, _reason = _maker_trade_fill(order, event)
                            if quantity > 0:
                                lease.grow(64)
                                order.remaining -= quantity
                                sample["fills"].extend(
                                    (
                                        float(due - order.created_time_ns),
                                        config.order_quantity - order.remaining,
                                    )
                                )
                                if order.remaining <= 0:
                                    finish(sample, "filled")
                    else:
                        state = _book_value(event)
                        if (
                            latest_state is not None
                            and state.segment_id != latest_state.segment_id
                        ):
                            for sample in tuple(samples):
                                finish(sample, "book_boundary")
                        if latest_owned is not None:
                            latest_owned.release(owner="crypto.maker.calibration.book")
                        latest_owned = event if isinstance(event, OwnedValue) else None
                        latest_state = state
                    next_event = next(events, None)
                    release_completed()
                    continue
                for sample in tuple(samples):
                    order = sample["order"]
                    if sample["cancel_at"] is not None and sample["cancel_at"] <= due:
                        finish(sample, "replaced")
                    elif not sample["activated"] and order.effective_at_ns <= due:
                        reason = _maker_join(
                            order, latest_state, observed_coverage, trade_coverage
                        )
                        if reason is None:
                            sample["activated"] = True
                        else:
                            finish(sample, reason)
                release_completed()

        try:
            for row in rows:
                timestamp = row["timestamp_ms"] * _NANOSECONDS_PER_MILLISECOND
                end = timestamp + HFT_MODEL_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND
                if row["label_available_at_ns"] > cutoff_ns or end > cutoff_ns:
                    raise ValueError("maker_calibration_outcome_not_mature")
                process_until(timestamp)
                valid = (
                    latest_state is not None
                    and latest_state.segment_id == row["book_segment_id"]
                    and _covered(observed_coverage, row["book_segment_id"], timestamp)
                    and _covered(observed_coverage, row["book_segment_id"], end)
                    and _covered(trade_coverage, row["trade_connection_id"], timestamp)
                    and _covered(trade_coverage, row["trade_connection_id"], end)
                )
                gate = row["proxy_prediction"] >= 0.5
                for sample in samples:
                    order = sample["order"]
                    desired = (
                        None
                        if not valid
                        else (
                            latest_state.bids
                            if order.side == "buy"
                            else latest_state.asks
                        )[0][0]
                    )
                    if (
                        not gate
                        or desired != order.price_tick
                        or order.remaining != config.order_quantity
                    ):
                        cancellation = (
                            timestamp + config.latency_ms * _NANOSECONDS_PER_MILLISECOND
                        )
                        sample["cancel_at"] = (
                            cancellation
                            if sample["cancel_at"] is None
                            else min(sample["cancel_at"], cancellation)
                        )
                if not valid or not gate:
                    excluded += 1
                    continue
                for side in ("buy", "sell"):
                    lease.grow(2048)
                    order = _maker_new_order(
                        counts[side],
                        side,
                        config.order_quantity,
                        latest_state,
                        row,
                        config,
                        observed_coverage,
                        trade_coverage,
                    )
                    samples.append(
                        {
                            "order": order,
                            "sample_id": counts[side],
                            "prediction": row["price_prediction"],
                            "observed_return": row["price_label"],
                            "available_ns": row["label_available_at_ns"],
                            "activated": False,
                            "cancel_at": None,
                            "fills": array("d"),
                        }
                    )
                    counts[side] += 1
                process_until(timestamp)
            process_until(cutoff_ns)
            if samples:
                raise ValueError("maker_calibration_outcome_not_mature")
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            samples.clear()
    return {"samples": counts, "excluded_decisions": excluded}


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
            visible = strict_finite_float(
                trade.visible_quantity,
                error="hft_execution_trade_visible_quantity_invalid",
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
                    book_next is not None
                    and _book_key(book_next)
                    <= (
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
                "book_segment_id",
                "trade_connection_id",
            ):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError("hft_maker_quote_schedule_identity_invalid")
            timestamp = row.get("timestamp_ms")
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError("hft_maker_quote_schedule_time_invalid")
            if previous is not None and timestamp <= previous:
                raise ValueError("hft_maker_quote_schedule_time_non_monotonic")
            previous = timestamp
            yield row


def simulate_maker(
    book_events,
    trades,
    quote_schedule,
    *,
    config: MakerConfig,
    observed_coverage,
    trade_coverage,
    resource_policy,
    capital,
    asset_id,
    funding_events,
    funding_forecasts=(),
    calibrations=None,
    decision_windows_ns=None,
    on_ledger=None,
    on_fill=None,
    on_equity=None,
):
    """Freeze passive quotes at decisions; replay delayed replacements and expiry."""
    events = iter(_market_events(book_events, trades))
    decisions = iter(_checked_quote_schedule(quote_schedule))
    forecasts = iter(funding_forecasts)
    active = {}
    commands = deque()
    metrics = _Metrics()
    economics = PortfolioEconomics(capital, asset_id, on_ledger)
    apply_funding = _funding_consumer(funding_events, economics)
    latest_state = latest_owned = next_event = None
    active_forecast = next_forecast = None
    forecast_timestamp = -1
    created_orders = filled_orders = cancelled_orders = 0
    partial_fill_events = trade_through_fill_events = post_only_rejections = 0
    equity_rows = market_event_count = blocked_decisions = undefined_decisions = 0
    max_abs_position = 0.0
    all_lifetime_count = all_lifetime_sum_ns = 0
    filled_lifetime_count = filled_lifetime_sum_ns = 0
    decision_time_ns = last_event_time_ns = 0
    window_index = 0
    window_open = False
    next_day = (
        None
        if decision_windows_ns is None
        else ((decision_windows_ns[0][0] + _UTC_DAY_NS - 1) // _UTC_DAY_NS)
        * _UTC_DAY_NS
    )

    def resolved_fill(fill):
        economics.diagnostic(dict(fill))
        if on_fill is not None:
            on_fill(fill)

    def finish_order(order, timestamp_ns, *, cancelled):
        nonlocal all_lifetime_count, all_lifetime_sum_ns, cancelled_orders
        nonlocal filled_lifetime_count, filled_lifetime_sum_ns
        lifetime = max(0, timestamp_ns - order.created_time_ns)
        all_lifetime_count += 1
        all_lifetime_sum_ns += lifetime
        if order.had_fill:
            filled_lifetime_count += 1
            filled_lifetime_sum_ns += lifetime
        if cancelled:
            cancelled_orders += 1

    with (
        resource_policy.lease(
            "crypto.execution.maker.workspace", _EXECUTION_WORKSPACE_BYTES
        ),
        resource_policy.lease("crypto.execution.maker.commands", 1) as command_lease,
        closing_iterator(events),
        closing_iterator(decisions),
        closing_iterator(forecasts),
        _PendingMarkouts(metrics, resolved_fill, resource_policy) as pending,
        ExitStack() as cleanup,
    ):

        def cleanup_books():
            if latest_owned is not None:
                latest_owned.release(owner="crypto.execution.book")
            if next_event is not None and isinstance(next_event[1], OwnedValue):
                next_event[1].release(owner="crypto.execution.book")

        cleanup.callback(cleanup_books)
        next_event = next(events, None)
        next_forecast = next(forecasts, None)

        def update_command_lease():
            command_lease.shrink_to(max(1, (len(commands) + len(active)) * 1024))

        def cancel_all(timestamp):
            for order in active.values():
                finish_order(
                    order, min(timestamp, order.valid_until_ns), cancelled=True
                )
            active.clear()
            for _time, _side, order in commands:
                if order is not None:
                    finish_order(
                        order, min(timestamp, order.valid_until_ns), cancelled=True
                    )
            commands.clear()
            update_command_lease()

        def expire_active(timestamp):
            for side, order in tuple(active.items()):
                if order.valid_until_ns <= timestamp:
                    active.pop(side)
                    finish_order(order, order.valid_until_ns, cancelled=True)

        def observed_mark(timestamp):
            return (
                _mid(latest_state)
                if latest_state is not None
                and _covered(observed_coverage, latest_state.segment_id, timestamp)
                else None
            )

        def process_book(event):
            nonlocal latest_state, latest_owned
            if isinstance(event, BookBoundary):
                cancel_all(event.observed_time_ns)
                if latest_owned is not None:
                    latest_owned.release(owner="crypto.execution.book")
                latest_owned = latest_state = None
                pending.expire_all()
                return
            previous = latest_state
            state = _book_value(event)
            if previous is not None and previous.segment_id != state.segment_id:
                cancel_all(state.observed_time_ns)
            if latest_owned is not None:
                latest_owned.release(owner="crypto.execution.book")
            latest_owned = event if isinstance(event, OwnedValue) else None
            latest_state = state
            _resolve_observed_markouts(
                pending, state, observed_coverage, previous_state=previous
            )
            if (
                decision_windows_ns is None
                or decision_windows_ns[0][0]
                <= state.observed_time_ns
                <= decision_windows_ns[-1][1]
            ) and _covered(observed_coverage, state.segment_id, state.observed_time_ns):
                economics.mark(state.observed_time_ns, _mid(state))

        def process_trade(trade):
            nonlocal filled_orders, partial_fill_events, trade_through_fill_events
            nonlocal max_abs_position
            if (
                latest_state is None
                or not _covered(
                    observed_coverage, latest_state.segment_id, trade.observed_time_ns
                )
                or not _covered(
                    trade_coverage, trade.connection_id, trade.observed_time_ns
                )
            ):
                return
            for side in ("buy", "sell"):
                order = active.get(side)
                if order is None or order.segment_id != latest_state.segment_id:
                    continue
                quantity, reason = _maker_trade_fill(order, trade)
                if quantity <= 0:
                    continue
                price = tick_to_price(order.price_tick, order.tick_size)
                sign = 1.0 if side == "buy" else -1.0
                fee = quantity * price * config.maker_fee_bps / 10_000.0
                mid = _mid(latest_state)
                fill = {
                    "order_id": order.order_id,
                    "timestamp_ms": trade.observed_time_ns
                    // _NANOSECONDS_PER_MILLISECOND,
                    "segment_id": order.segment_id,
                    "side": side,
                    "fill_reason": reason,
                    "fill_price": price,
                    "fill_qty": quantity,
                    "notional": quantity * price,
                    "fee": fee,
                    "estimated_queue_ahead_at_join": order.initial_queue_ahead,
                    "estimated_queue_ahead_before_fill": order.queue_ahead,
                    "spread_capture_per_unit": sign * (mid - price),
                    "markout_target_time_ms": trade.observed_time_ns
                    // _NANOSECONDS_PER_MILLISECOND
                    + config.markout_horizon_ms,
                    "markout_target_time_ns": trade.observed_time_ns
                    + config.markout_horizon_ms * _NANOSECONDS_PER_MILLISECOND,
                }
                fill["fill_id"] = economics.fill(
                    timestamp_ns=trade.observed_time_ns,
                    order_id=str(order.order_id),
                    quantity=sign * quantity,
                    price=price,
                    mark_price=mid,
                    fee=fee,
                )
                if not order.had_fill:
                    filled_orders += 1
                    order.had_fill = True
                if quantity < order.remaining:
                    partial_fill_events += 1
                if reason == "trade_through":
                    trade_through_fill_events += 1
                order.remaining -= quantity
                max_abs_position = max(max_abs_position, abs(economics.quantity))
                metrics.observe_fill(fill)
                pending.add(fill)
                if order.remaining <= 0:
                    active.pop(side)
                    finish_order(order, trade.observed_time_ns, cancelled=False)
                break

        def process_events_until(target):
            nonlocal \
                next_event, \
                market_event_count, \
                last_event_time_ns, \
                post_only_rejections
            while (next_event is not None and next_event[0][0] <= target) or (
                commands and commands[0][0] <= target
            ):
                event_time = next_event[0][0] if next_event is not None else None
                command_time = commands[0][0] if commands else None
                stamp = (
                    event_time
                    if command_time is None
                    else command_time
                    if event_time is None
                    else min(event_time, command_time)
                )
                if stamp > target:
                    break
                apply_funding(stamp)
                expire_active(stamp)
                if event_time == stamp:
                    market_event_count += 1
                    last_event_time_ns = stamp
                    item = next_event[1]
                    if isinstance(item, TradeEvent):
                        process_trade(item)
                    else:
                        process_book(item)
                    next_event = next(events, None)
                    continue
                _effective, side, new_order = commands.popleft()
                old = active.pop(side, None)
                if old is not None:
                    finish_order(old, stamp, cancelled=True)
                if new_order is not None:
                    reason = (
                        "quote_expired_before_activation"
                        if new_order.valid_until_ns <= stamp
                        else _maker_join(
                            new_order, latest_state, observed_coverage, trade_coverage
                        )
                    )
                    if reason is None:
                        active[side] = new_order
                    else:
                        finish_order(
                            new_order,
                            min(stamp, new_order.valid_until_ns),
                            cancelled=True,
                        )
                        post_only_rejections += int(
                            reason == "post_only_crossing_rejected"
                        )
                        mark = observed_mark(stamp)
                        if mark is not None or economics.quantity == 0:
                            economics.mark(
                                stamp,
                                mark,
                                diagnostics=(
                                    PortfolioDiagnostic(
                                        asset_id
                                        if mark is not None
                                        else PORTFOLIO_CASH_ASSET_ID,
                                        None,
                                        "maker_activation_rejection",
                                        text=reason,
                                    ),
                                ),
                            )
                update_command_lease()
            apply_funding(target)
            expire_active(target)

        def process_until(target):
            nonlocal next_day
            next_day = _advance_utc_boundaries(
                target,
                next_day,
                process_events_until,
                economics,
                lambda: latest_state,
                observed_coverage,
            )

        def boundary(timestamp, *, closing):
            process_until(timestamp)
            count = len(active) + sum(
                order is not None for _time, _side, order in commands
            )
            quantity = sum(order.remaining for order in active.values()) + sum(
                order.remaining for _time, _side, order in commands if order is not None
            )
            if closing:
                cancel_all(timestamp)
            mark = observed_mark(timestamp)
            if mark is None and economics.quantity != 0:
                raise PortfolioTargetBlocked(
                    "portfolio_window_boundary_valuation_missing"
                )
            diagnostic_asset = asset_id if mark is not None else PORTFOLIO_CASH_ASSET_ID
            economics.mark(
                timestamp,
                mark,
                diagnostics=(
                    PortfolioDiagnostic(
                        diagnostic_asset,
                        None,
                        "oos_window_end" if closing else "oos_window_start",
                        number=float(window_index),
                    ),
                    *(
                        (
                            PortfolioDiagnostic(
                                diagnostic_asset,
                                None,
                                "cancelled_order_count",
                                number=float(count),
                            ),
                            PortfolioDiagnostic(
                                diagnostic_asset,
                                None,
                                "cancelled_order_quantity",
                                number=quantity,
                            ),
                        )
                        if closing
                        else ()
                    ),
                ),
            )

        try:
            for row in decisions:
                decision_time_ns = row["timestamp_ms"] * _NANOSECONDS_PER_MILLISECOND
                if decision_windows_ns is not None:
                    while (
                        window_index < len(decision_windows_ns)
                        and decision_time_ns > decision_windows_ns[window_index][1]
                    ):
                        if not window_open:
                            boundary(
                                decision_windows_ns[window_index][0], closing=False
                            )
                        boundary(decision_windows_ns[window_index][1], closing=True)
                        window_index += 1
                        window_open = False
                    if (
                        window_index == len(decision_windows_ns)
                        or decision_time_ns < decision_windows_ns[window_index][0]
                    ):
                        raise ValueError("hft_decision_outside_oos_window")
                    if not window_open:
                        boundary(decision_windows_ns[window_index][0], closing=False)
                        window_open = True
                process_until(decision_time_ns)
                while next_forecast is not None:
                    if (
                        type(next_forecast) is not FundingForecast
                        or next_forecast.available_at_ns <= forecast_timestamp
                    ):
                        raise ValueError("portfolio_funding_forecast_order_invalid")
                    if next_forecast.available_at_ns > decision_time_ns:
                        break
                    active_forecast = next_forecast
                    forecast_timestamp = active_forecast.available_at_ns
                    next_forecast = next(forecasts, None)
                mid = observed_mark(decision_time_ns)
                valid = (
                    mid is not None
                    and latest_state.segment_id == row["book_segment_id"]
                    and _covered(
                        trade_coverage, row["trade_connection_id"], decision_time_ns
                    )
                )
                if not valid:
                    undefined_decisions += 1
                if calibrations is not None:
                    for column in ("price_prediction", "proxy_prediction"):
                        if row[column] is not None:
                            strict_finite_float(
                                row[column], error="hft_maker_prediction_invalid"
                            )
                gate = calibrations is None or (
                    row["proxy_prediction"] is not None
                    and row["proxy_prediction"] >= 0.5
                    and row["price_prediction"] is not None
                )
                forecast_end = (
                    decision_time_ns
                    + HFT_MODEL_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND
                )
                funding = []
                funding_known = True
                if calibrations is not None and valid and gate:
                    funding_known = (
                        active_forecast is not None
                        and active_forecast.known_until_ns >= forecast_end
                    )
                    if funding_known:
                        funding = [
                            (stamp - decision_time_ns, cash)
                            for stamp, cash in active_forecast.cashflows
                            if decision_time_ns < stamp <= forecast_end
                        ]
                        funding_known = all(
                            cash is not None for _stamp, cash in funding
                        )
                    if not funding_known:
                        blocked_decisions += 1
                diagnostics = []
                for side, sign in (("buy", 1.0), ("sell", -1.0)):
                    desired_price = (
                        None
                        if not valid
                        else (
                            latest_state.bids if side == "buy" else latest_state.asks
                        )[0][0]
                    )
                    headroom = max(0.0, config.max_position - sign * economics.quantity)
                    lifetime = HFT_MODEL_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND
                    if decision_windows_ns is not None:
                        lifetime = min(
                            lifetime,
                            decision_windows_ns[window_index][1] - decision_time_ns,
                        )
                    allowed = (
                        valid
                        and gate
                        and funding_known
                        and config.latency_ms * _NANOSECONDS_PER_MILLISECOND < lifetime
                    )

                    def choose(
                        limit,
                        side=side,
                        sign=sign,
                        lifetime=lifetime,
                        allowed=allowed,
                        row=row,
                        mid=mid,
                        desired_price=desired_price,
                        funding=funding,
                        fixed_quantity=None,
                    ):
                        if not allowed:
                            return 0.0, 0.0
                        limit = min(config.order_quantity, limit)
                        if calibrations is None:
                            skew = (
                                economics.quantity - config.inventory_target
                            ) / config.max_position
                            return min(
                                limit,
                                config.order_quantity
                                * max(0.0, 1.0 - sign * config.inventory_skew * skew),
                            ), None
                        return calibrations[row["fold_id"]][side].select(
                            limit=limit,
                            side=side,
                            mid=mid,
                            price=tick_to_price(desired_price, latest_state.tick_size),
                            prediction=row["price_prediction"],
                            fee_bps=config.maker_fee_bps,
                            lifetime_ns=lifetime,
                            funding=funding,
                            fixed_quantity=fixed_quantity,
                        )

                    quantity, gain = (
                        choose(headroom) if calibrations is None else (0.0, 0.0)
                    )
                    pending_side = [
                        order
                        for _time, command_side, order in commands
                        if command_side == side
                    ]
                    current = pending_side[-1] if pending_side else active.get(side)
                    reserved = sum(
                        order.remaining for order in pending_side if order is not None
                    )
                    if side in active:
                        reserved += active[side].remaining
                    if config.latency_ms == 0:
                        reserved = 0.0
                    keep = (
                        current is not None
                        and quantity > 0
                        and current.price_tick == desired_price
                        and current.remaining == quantity
                        and current.valid_until_ns > decision_time_ns
                    )
                    if calibrations is not None:
                        keep_quantity = keep_gain = 0.0
                        if current is not None and current.price_tick == desired_price:
                            remaining_lifetime = min(
                                lifetime, current.valid_until_ns - decision_time_ns
                            )
                            keep_quantity, keep_gain = choose(
                                headroom,
                                lifetime=remaining_lifetime,
                                allowed=(
                                    valid
                                    and gate
                                    and funding_known
                                    and max(
                                        0, current.effective_at_ns - decision_time_ns
                                    )
                                    < remaining_lifetime
                                ),
                                fixed_quantity=current.remaining,
                            )
                        quantity, gain = choose(max(0.0, headroom - reserved))
                        keep = keep_gain > 0 and keep_gain >= gain
                        if keep:
                            quantity, gain = keep_quantity, keep_gain
                    if not keep:
                        if calibrations is None:
                            quantity, gain = choose(max(0.0, headroom - reserved))
                        if quantity > 0 or current is not None:
                            order = None
                            if quantity > 0:
                                created_orders += 1
                                order = _maker_new_order(
                                    created_orders,
                                    side,
                                    quantity,
                                    latest_state,
                                    row,
                                    config,
                                    observed_coverage,
                                    trade_coverage,
                                )
                            effective = (
                                decision_time_ns
                                + config.latency_ms * _NANOSECONDS_PER_MILLISECOND
                            )
                            command_lease.grow(1024)
                            commands.append((effective, side, order))
                    reason = (
                        "decision_book_or_trade_unavailable"
                        if not valid
                        else "proxy_gate_closed"
                        if not gate
                        else "portfolio_expected_funding_missing"
                        if not funding_known
                        else "keep_queue_without_renewal"
                        if keep
                        else "positive_expected_net_gain"
                        if quantity and calibrations is not None
                        else "fixed_quote_diagnostic"
                        if quantity
                        else "no_positive_expected_net_gain"
                    )
                    diagnostic_asset = (
                        asset_id if mid is not None else PORTFOLIO_CASH_ASSET_ID
                    )
                    diagnostics.extend(
                        (
                            PortfolioDiagnostic(
                                diagnostic_asset,
                                None,
                                f"maker_{side}_requested_quantity",
                                number=quantity,
                            ),
                            PortfolioDiagnostic(
                                diagnostic_asset,
                                None,
                                f"maker_{side}_expected_net_gain",
                                number=gain,
                            ),
                            PortfolioDiagnostic(
                                diagnostic_asset,
                                None,
                                f"maker_{side}_selection_reason",
                                text=reason,
                            ),
                        )
                    )
                marked = None
                if mid is not None or economics.quantity == 0:
                    marked = economics.mark(
                        decision_time_ns, mid, diagnostics=tuple(diagnostics)
                    )
                process_until(decision_time_ns)
                if on_equity is not None:
                    on_equity(
                        {
                            "timestamp_ms": row["timestamp_ms"],
                            "position": economics.quantity,
                            "gross_pnl": None
                            if marked is None
                            else marked["gross_pnl"],
                            "net_pnl": None if marked is None else marked["net_pnl"],
                            "fees": -economics.cost_cashflows,
                        }
                    )
                equity_rows += 1
            if decision_windows_ns is not None:
                while window_index < len(decision_windows_ns):
                    if not window_open:
                        boundary(decision_windows_ns[window_index][0], closing=False)
                    boundary(decision_windows_ns[window_index][1], closing=True)
                    window_index += 1
                    window_open = False
                end_timestamp = decision_windows_ns[-1][1]
            else:
                while next_event is not None or commands:
                    times = ([next_event[0][0]] if next_event is not None else []) + (
                        [commands[0][0]] if commands else []
                    )
                    process_until(min(times))
                end_timestamp = max(
                    economics.last_timestamp, last_event_time_ns, decision_time_ns
                )
                for order in active.values():
                    finish_order(order, order.valid_until_ns, cancelled=True)
                active.clear()
            pending.expire_all()
            terminal_mid = observed_mark(end_timestamp)
            marked = (
                None
                if terminal_mid is None and economics.quantity != 0
                else economics.mark(end_timestamp, terminal_mid)
            )
            last_gross = None if marked is None else marked["gross_pnl"]
            last_net = None if marked is None else marked["net_pnl"]
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            active.clear()
            commands.clear()
    result = {
        "simulation": "event_ordered_public_l2_maker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "zero_fill_orders": created_orders - filled_orders,
        "partial_fill_events": partial_fill_events,
        "trade_through_fill_events": trade_through_fill_events,
        "post_only_rejections": post_only_rejections,
        "equity_rows": equity_rows,
        "gross_pnl": last_gross,
        "net_pnl": last_net,
        "max_abs_position": max_abs_position,
        "inventory_revaluation_pnl": None
        if last_gross is None
        else last_gross - metrics.spread_capture_pnl,
        "average_order_lifetime_ms": all_lifetime_sum_ns
        / _NANOSECONDS_PER_MILLISECOND
        / all_lifetime_count
        if all_lifetime_count
        else None,
        "average_filled_order_lifetime_ms": filled_lifetime_sum_ns
        / _NANOSECONDS_PER_MILLISECOND
        / filled_lifetime_count
        if filled_lifetime_count
        else None,
        "market_event_count": market_event_count,
        "maker_fee_bps": config.maker_fee_bps,
        "order_quantity": config.order_quantity,
        "max_position": config.max_position,
        "inventory_target": config.inventory_target,
        "inventory_skew": config.inventory_skew,
        "latency_ms": config.latency_ms,
        "quote_horizon_ms": HFT_MODEL_HORIZON_MS,
        "markout_horizon_ms": config.markout_horizon_ms,
        "queue_model": "estimated_queue_ahead",
        "queue_assumption": (
            "public MBP aggregate depth; cancellations do not advance queue"
        ),
        "same_timestamp_policy": "local_observation_order_then_delayed_commands",
        "maker_policy_id": "paired_validation_calibrated_post_only_maker"
        if calibrations is not None
        else "public_l2_inventory_maker_baseline",
        "maker_prediction_used": calibrations is not None,
        "selection_status": "BLOCKED"
        if blocked_decisions
        else "UNDEFINED"
        if undefined_decisions
        else "COMPLETE",
        "blocked_decisions": blocked_decisions,
        "undefined_decisions": undefined_decisions,
        "assumption": "historical_book_zero_market_impact",
        **metrics.payload(),
        "fees": -economics.cost_cashflows,
        "funding": economics.funding_cashflows,
        "initial_capital": capital.initial_capital,
        "valuation_currency": capital.valuation_currency,
        "candidate_id": capital.candidate_id,
        "nav": None if last_net is None else capital.initial_capital + last_net,
        "capital_return": None
        if last_net is None
        else last_net / capital.initial_capital,
    }
    return _record_execution_result(economics, result, end_timestamp, terminal_mid)
