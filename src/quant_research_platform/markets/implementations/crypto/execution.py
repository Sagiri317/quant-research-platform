"""Research-only event-time taker and public-L2 maker simulations."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from types import SimpleNamespace
from typing import Any

import numpy as np

from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeBoundary,
    TradeEvent,
    index_to_price,
    observation_order_key,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_ECONOMIC_SUMMARY_FIELDS,
    ExecutedTrade,
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
    PortfolioValuation,
)
from quant_research_platform.contracts.validation import (
    strict_finite_float,
)
from quant_research_platform.policy.lifecycle import closing_iterator


class PortfolioEconomics:
    """Bounded linear-perpetual state shared by maker and taker execution."""

    def __init__(
        self,
        capital: PortfolioCapital,
        asset_id: str,
        emit: Callable[[PortfolioLedgerBatch], None] | None,
    ) -> None:
        if capital.valuation_method != "linear_perpetual" or not asset_id:
            raise ValueError("portfolio_valuation_method_invalid")
        self.capital = capital
        self.asset_id = asset_id
        self.emit = emit
        self.quantity = Fraction()
        self._cost_basis = Fraction()
        self._cash = Fraction(capital.initial_capital)
        self._economics = _LedgerEconomics(capital)
        self.sequence = 0
        self.fill_count = 0
        self.last_timestamp = 0
        self.last_mark: float | None = None
        self._unvalued_funding: list[PortfolioCashFlow] = []

    @property
    def cost_basis(self):
        return float(self._cost_basis)

    @property
    def cash(self):
        return float(self._cash)

    @property
    def cost_cashflows(self):
        return -float(self._economics.totals["fees"])

    @property
    def funding_cashflows(self):
        return float(self._economics.totals["funding"])

    def _values(
        self, timestamp: int, phase: str, mark: float | None
    ) -> tuple[PortfolioValuation, ...]:
        cash = PortfolioValuation(
            timestamp, PORTFOLIO_CASH_ASSET_ID, phase, self.cash, self.cash
        )
        if mark is None:
            return (cash,)
        return (
            PortfolioValuation(
                timestamp,
                self.asset_id,
                phase,
                self.quantity,
                float(self.quantity * Fraction(mark) - self._cost_basis),
                mark,
                self.cost_basis,
            ),
            cash,
        )

    def fill(
        self,
        *,
        timestamp_ns: int,
        order_id: str,
        quantity: float | Fraction,
        price: float | Fraction,
        mark_price: float,
        fee: float,
    ) -> str:
        if (
            not np.isfinite([mark_price, fee]).all()
            or (not isinstance(price, Fraction) and not np.isfinite(price))
            or (not isinstance(quantity, Fraction) and not np.isfinite(quantity))
            or not quantity
            or min(price, mark_price) <= 0
        ):
            raise ValueError("portfolio_execution_fill_invalid")
        self._settle_funding(timestamp_ns, mark_price)
        before = self._values(timestamp_ns, "before", mark_price)
        self.fill_count += 1
        fill_id = f"fill:{self.fill_count}"
        quantity = Fraction(quantity)
        notional = float(quantity * Fraction(price))
        trade = ExecutedTrade(
            fill_id, order_id, timestamp_ns, self.asset_id, quantity, notional
        )
        realized = Fraction()
        if self.quantity and (self.quantity > 0) != (quantity > 0):
            closed = min(abs(self.quantity), abs(quantity))
            entry = self._cost_basis / self.quantity
            realized = (
                closed * (1 if self.quantity > 0 else -1) * (Fraction(price) - entry)
            )
            remaining = self.quantity + quantity
            self._cost_basis = remaining * (
                entry
                if remaining != 0 and (remaining > 0) == (self.quantity > 0)
                else Fraction(price)
            )
        else:
            self._cost_basis += quantity * Fraction(price)
        self.quantity += quantity
        realized = float(realized)
        self._cash += Fraction(realized) - Fraction(fee)
        cashflows = (
            PortfolioCashFlow(
                f"{fill_id}:settlement",
                timestamp_ns,
                self.asset_id,
                fill_id,
                "realized_pnl",
                float(realized),
            ),
            PortfolioCashFlow(
                f"{fill_id}:fee",
                timestamp_ns,
                self.asset_id,
                fill_id,
                "commission_cost",
                -fee,
            ),
        )
        self._emit((trade,), cashflows, before, timestamp_ns, mark_price)
        return fill_id

    def funding(self, event: FundingEvent, *, mark_price: float | None) -> None:
        """Keep settlement cash evidence separate from the causal book valuation."""
        if self.quantity == 0 or event.rate == 0:
            return
        if event.rate is None or event.mark_price is None:
            raise PortfolioTargetBlocked("portfolio_required_funding_missing")
        amount = float(
            -self.quantity * Fraction(event.mark_price) * Fraction(event.rate)
        )
        self._unvalued_funding.append(
            PortfolioCashFlow(
                f"funding:{event.timestamp_ns}",
                event.timestamp_ns,
                self.asset_id,
                None,
                "funding",
                amount,
            )
        )
        self._settle_funding(event.timestamp_ns, mark_price)

    @property
    def unvalued_funding(self):
        return {
            "unvalued_funding_cashflow": float(
                sum(
                    (Fraction(row.amount) for row in self._unvalued_funding), Fraction()
                )
            ),
            "unvalued_funding_event_count": len(self._unvalued_funding),
        }

    def _settle_funding(self, timestamp, mark):
        if (
            not self._unvalued_funding
            or mark is None
            or timestamp < self._unvalued_funding[-1].timestamp_ns
        ):
            return
        if not np.isfinite(mark) or mark <= 0:
            raise ValueError("portfolio_valuation_mark_invalid")
        # An unquoted interval retains its cash facts without inventing a NAV.
        before = self._values(self.last_timestamp, "before", self.last_mark)
        cashflows = tuple(self._unvalued_funding)
        self._cash += sum((Fraction(row.amount) for row in cashflows), Fraction())
        self._emit((), cashflows, before, timestamp, mark)
        self._unvalued_funding.clear()

    def mark(
        self,
        timestamp_ns: int,
        mark_price: float | None,
        *,
        diagnostics: tuple[PortfolioDiagnostic, ...] = (),
    ) -> dict[str, float]:
        if (mark_price is None and self.quantity != 0.0) or (
            mark_price is not None and (not np.isfinite(mark_price) or mark_price <= 0)
        ):
            raise ValueError("portfolio_valuation_mark_invalid")
        self._settle_funding(timestamp_ns, mark_price)
        before = self._values(timestamp_ns, "before", mark_price)
        self._emit((), (), before, timestamp_ns, mark_price, diagnostics)
        summary = self._economics.summary()
        return {
            name: summary[name]
            for name in (
                "gross_pnl",
                "net_pnl",
                "nav",
                "capital_return",
                "fees",
                "funding",
            )
        }

    def diagnostic(self, fill: dict[str, object]) -> None:
        fill_id = str(fill["fill_id"])
        diagnostics = tuple(
            PortfolioDiagnostic(
                self.asset_id,
                fill_id,
                name,
                number=float(value) if type(value) in {int, float} else None,
                text=value if isinstance(value, str) else None,
            )
            for name, value in fill.items()
            if name not in {"fill_id", "fill_price", "fill_qty", "notional", "fee"}
            and (value is None or type(value) in {int, float, str})
        )
        before = self._values(self.last_timestamp, "before", self.last_mark)
        self._emit((), (), before, self.last_timestamp, self.last_mark, diagnostics)

    def _emit(self, trades, cashflows, before, timestamp, mark, diagnostics=()) -> None:
        batch = PortfolioLedgerBatch(
            self.sequence,
            trades,
            cashflows,
            (
                *before,
                *self._values(timestamp, "after", mark),
                *self._values(timestamp, "end", mark),
            ),
            diagnostics,
        )
        self._economics.consume(batch)
        self.sequence += 1
        self.last_timestamp = timestamp
        self.last_mark = mark
        if self.emit is not None:
            self.emit(batch)


MAKER_QUOTE_HORIZON_MS = 500
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_UTC_DAY_NS = 86_400_000_000_000


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
    inventory_target: float | None
    inventory_skew: float | None
    markout_horizon_ms: int
    latency_ms: int

    def __post_init__(self) -> None:
        _positive(self.order_quantity, "hft_maker_order_quantity_invalid")
        _positive(self.max_position, "hft_maker_position_size_invalid")
        strict_finite_float(self.maker_fee_bps, error="hft_maker_fee_invalid")
        if self.inventory_target is not None:
            target = strict_finite_float(
                self.inventory_target,
                error="hft_maker_inventory_target_invalid",
            )
            if abs(target) > self.max_position:
                raise ValueError("hft_maker_inventory_target_invalid")
        if self.inventory_skew is not None:
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


class _AccountPathStop(Exception):
    """Stop only the exact account path whose next state cannot be identified."""

    def __init__(self, reason, timestamp_ns):
        self.reason, self.timestamp_ns = reason, timestamp_ns
        super().__init__(reason)


@contextmanager
def _known_account_path(stopped):
    try:
        yield
    except _AccountPathStop as error:
        stopped.append(error)


def _funding_consumer(events, economics, coverage, current_book, book_coverage):
    if type(events) is not tuple or any(
        type(event) is not FundingEvent for event in events
    ):
        raise ValueError("portfolio_funding_schedule_invalid")
    timestamps = tuple(event.timestamp_ns for event in events)
    if timestamps != tuple(sorted(set(timestamps))):
        raise ValueError("portfolio_funding_schedule_invalid")
    pending = iter(events)
    next_event = next(pending, None)
    previous = 0

    def consume(timestamp_ns):
        nonlocal next_event, previous
        stop = None
        if economics.quantity and coverage is not None:
            cursor = previous + 1
            for start, end in coverage:
                if end < cursor:
                    continue
                if start > cursor or cursor > timestamp_ns:
                    break
                cursor = end + 1
            if cursor <= timestamp_ns:
                stop = cursor
        known_until = timestamp_ns if stop is None else stop - 1
        while next_event is not None and next_event.timestamp_ns <= known_until:
            if economics.quantity and (
                next_event.rate is None
                or next_event.mark_price is None
                and next_event.rate != 0
            ):
                raise _AccountPathStop(
                    "portfolio_required_funding_missing", next_event.timestamp_ns
                )
            book = current_book()
            mark = (
                _mid(book)
                if book is not None
                and book.observed_time_ns <= next_event.timestamp_ns
                and _covered(book_coverage, book.segment_id, next_event.timestamp_ns)
                else None
            )
            economics.funding(next_event, mark_price=mark)
            next_event = next(pending, None)
        if stop is not None:
            raise _AccountPathStop("portfolio_required_funding_coverage_missing", stop)
        previous = max(previous, timestamp_ns)

    return consume


def _weighted_mean(
    mean: float, old_quantity: float, value: float, quantity: float
) -> float:
    total = old_quantity + quantity
    old_weight, weight = old_quantity / total, quantity / total
    if (mean < 0) == (value < 0):
        # Update the larger weight so rounding the other ratio to one cannot cancel it.
        if quantity > old_quantity:
            return value + (mean - value) * old_weight
        return mean + (value - mean) * weight
    return mean * old_weight + value * weight


@dataclass
class _Metrics:
    fill_events: int = 0
    fill_quantity: float = 0.0
    turnover: float = 0.0
    slippage_quantity: float = 0.0
    slippage_mean: float = 0.0
    spread_capture_pnl: float = 0.0
    defined_markout_quantity: float = 0.0
    markout_weighted: float = 0.0
    markout_mean_bps: float = 0.0
    markout_pnl: float = 0.0
    adverse_selection_cost: float = 0.0
    undefined_markouts: int = 0

    def observe_fill(self, fill: Mapping[str, Any]) -> None:
        quantity = float(fill["fill_qty"])
        self.fill_events += 1
        self.fill_quantity += quantity
        self.turnover += float(fill["notional"])
        if "slippage_bps" in fill:
            self.slippage_mean = _weighted_mean(
                self.slippage_mean,
                self.slippage_quantity,
                float(fill["slippage_bps"]),
                quantity,
            )
            self.slippage_quantity += quantity
        if "spread_capture_per_unit" in fill:
            self.spread_capture_pnl += quantity * float(fill["spread_capture_per_unit"])

    def finalize_markout(self, fill: Mapping[str, Any]) -> None:
        quantity = float(fill["fill_qty"])
        markout = fill.get("markout_per_unit")
        if markout is None:
            self.undefined_markouts += 1
            return
        value = float(markout)
        self.markout_mean_bps = _weighted_mean(
            self.markout_mean_bps,
            self.defined_markout_quantity,
            float(fill["markout_bps"]),
            quantity,
        )
        self.defined_markout_quantity += quantity
        self.markout_weighted += quantity * value
        self.markout_pnl += float(fill["markout_pnl"])
        self.adverse_selection_cost += float(fill["adverse_selection_cost"])

    def payload(self) -> dict[str, Any]:
        return {
            "fill_events": self.fill_events,
            "fill_quantity": self.fill_quantity,
            "turnover": self.turnover,
            "average_slippage_bps": (
                self.slippage_mean if self.slippage_quantity else None
            ),
            "spread_capture_pnl": self.spread_capture_pnl,
            "average_markout_per_unit": (
                self.markout_weighted / self.defined_markout_quantity
                if self.defined_markout_quantity
                else None
            ),
            "average_markout_bps": (
                self.markout_mean_bps if self.defined_markout_quantity else None
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
    """FIFO of fills awaiting their exact future markout."""

    def __init__(self, metrics, on_fill, *, stop_context=None):
        self._pending = deque()
        self._metrics = metrics
        self._on_fill = on_fill
        self._stop_context = stop_context

    def __enter__(self) -> _PendingMarkouts:
        return self

    def __exit__(self, _exception_type, primary, _traceback):
        if isinstance(primary, _AccountPathStop) and self._stop_context is not None:
            state, coverage = self._stop_context()
            self.advance(primary.timestamp_ns - 1, state, coverage)
        if primary is None or isinstance(primary, _AccountPathStop):
            self.expire_all(
                reason="observation_truncated" if primary is not None else None
            )

    def add(self, fill):
        self._pending.append(dict(fill))

    def advance(
        self,
        timestamp_ns: int,
        state: BookState | None,
        coverage: Mapping[str, tuple[int, int]],
    ) -> None:
        """Mature fills at a reached clock using only the last observed book."""
        while (fill := self._peek()) is not None:
            target = fill["markout_target_time_ns"]
            if target > timestamp_ns:
                break
            future_mid = (
                _mid(state)
                if state is not None
                and state.segment_id == fill["segment_id"]
                and state.observed_time_ns <= target
                and _covered(coverage, fill["segment_id"], target)
                else None
            )
            try:
                self._finalize(fill, future_mid=future_mid)
            finally:
                del fill
                self._pop()

    def expire_all(self, *, reason=None) -> None:
        while (fill := self._peek()) is not None:
            try:
                if reason is not None:
                    fill["markout_reason"] = reason
                self._finalize(fill, future_mid=None)
            finally:
                del fill
                self._pop()

    def _peek(self):
        return self._pending[0] if self._pending else None

    def _pop(self):
        self._pending.popleft()

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


def _checked_book_events(
    events: Iterable[BookState | BookBoundary],
) -> Iterator[BookState | BookBoundary]:
    previous: tuple[int, ...] | None = None
    active = iter(events)
    with closing_iterator(active):
        for event in active:
            value = event
            if not isinstance(value, BookState | BookBoundary):
                raise TypeError("hft_execution_book_event_invalid")
            key = observation_order_key(value)
            if previous is not None and key < previous:
                raise ValueError("hft_execution_time_non_monotonic")
            previous = key
            try:
                yield event
            except GeneratorExit:
                return


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
            try:
                yield row
            except GeneratorExit:
                return


def _mid(state: BookState) -> float:
    if not state.bids or not state.asks:
        raise ValueError("hft_execution_one_sided_book")
    bid, ask = state.bids[0][0], state.asks[0][0]
    index_to_price(bid, state.price_unit)
    index_to_price(ask, state.price_unit)
    if ask <= bid:
        raise ValueError("hft_execution_crossed_book")
    return float((bid + ask) * Fraction(state.price_unit) / 2)


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
    # The previous timestamp is complete; equal-time updates still need replaying.
    pending.advance(state.observed_time_ns - 1, previous_state, coverage)
    bounds = coverage.get(state.segment_id)
    if (
        bounds is None
        or state.observed_time_ns >= bounds[1]
        or previous_state is not None
        and previous_state.segment_id != state.segment_id
    ):
        pending.expire_all()


def _execution_fee(quantity, price, fee_bps):
    """Charge actual executed consideration without rescuing float overflow."""
    if isinstance(quantity, Fraction):
        return quantity * Fraction(price) * Fraction(fee_bps) / 10_000
    fee = quantity * price * fee_bps / 10_000.0
    if not math.isfinite(fee):
        raise ValueError("hft_execution_fee_unrepresentable")
    return fee


def _position_headroom(limit, quantity, reserved=0):
    """Round the exact remaining exposure down to an admissible float order size."""
    capacity = max(
        Fraction(), Fraction(limit) - Fraction(quantity) - Fraction(reserved)
    )
    try:
        rounded = float(capacity)
    except OverflowError:
        return math.nextafter(math.inf, 0.0)
    if Fraction(rounded) > capacity:
        rounded = math.nextafter(rounded, 0.0)
    return rounded


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
            price = index_to_price(tick, state.price_unit)
            fee_per_unit = price * config.taker_fee_bps / 10_000.0
            unit_gain = sign * (forecast - price + funding_per_unit) - fee_per_unit
            increment = size * unit_gain
            gain += increment
            if not math.isfinite(gain):
                raise ValueError("hft_expected_gain_unrepresentable")
            quantity += size
            remaining -= size
            if gain > best_gain:
                best_quantity, best_gain = sign * quantity, gain
            if remaining <= 0:
                break
    return best_quantity, best_gain


def _record_execution_result(economics, result, timestamp, mark):
    result.update(economics.unvalued_funding)
    if result["net_pnl"] is None:
        # Repeat the last exact prefix valuation, never a stale endpoint estimate.
        timestamp, mark = economics.last_timestamp, economics.last_mark
        prefix = economics.mark(timestamp, mark)
        result.update(
            {
                f"known_prefix_{key}": value
                for key, value in prefix.items()
                if key != "fees"
            }
        )
        result["observed_totals_scope"] = "known_prefix"
        result["known_prefix_valuation_timestamp_ns"] = timestamp
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
    book_events: Iterable[BookState | BookBoundary],
    decisions: Iterable[Mapping[str, Any]],
    *,
    prediction_column: str,
    prediction_horizon_ms: int,
    config: TakerConfig,
    observed_coverage: Mapping[str, tuple[int, int]],
    capital: PortfolioCapital,
    asset_id: str,
    funding_events: Iterable[FundingEvent],
    funding_forecasts: Iterable[FundingForecast],
    funding_coverage=None,
    decision_windows_ns: tuple[tuple[int, int], ...] | None = None,
    observation_window_ns: tuple[int, int] | None = None,
    on_ledger: Callable[[PortfolioLedgerBatch], None] | None = None,
) -> dict[str, Any]:
    """Freeze the net-gain maximizing order at decision time, then execute later."""

    observation_start = (
        None if observation_window_ns is None else observation_window_ns[0]
    )
    observation_stop = (
        None if observation_window_ns is None else observation_window_ns[1]
    )
    truncated = observation_stop is not None and (
        decision_windows_ns is None or observation_stop <= decision_windows_ns[-1][1]
    )
    first_market_gap_ns = None
    censored_count = censored_pending = censored_zero_fill = 0
    censored_quantity = 0.0
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
    terminal_mid: float | None = None
    terminal_timestamp = 0
    action_timestamp = 0
    metrics = _Metrics()
    economics = PortfolioEconomics(capital, asset_id, on_ledger)
    apply_funding = _funding_consumer(
        funding_events,
        economics,
        funding_coverage,
        lambda: latest_state,
        observed_coverage,
    )

    def resolved_fill(fill):
        economics.diagnostic(dict(fill))

    blocked_decisions = undefined_decisions = 0
    funding_required_decisions = 0
    orders = deque()
    reserved_buy = reserved_sell = Fraction()
    created_orders = 0
    filled_orders = 0
    unfilled_quantity = 0.0
    equity_rows = 0
    max_abs_position = 0.0
    last_gross: float | None = None
    last_net: float | None = None
    market_event_count = 0
    window_index = 0
    if observation_start is not None and decision_windows_ns is not None:
        while (
            window_index < len(decision_windows_ns)
            and decision_windows_ns[window_index][1] < observation_start
        ):
            window_index += 1
    window_open = False
    cancelled_orders = 0
    next_day = (
        None
        if decision_windows_ns is None
        else (
            (max(decision_windows_ns[0][0], observation_start or 0) + _UTC_DAY_NS - 1)
            // _UTC_DAY_NS
        )
        * _UTC_DAY_NS
    )

    stopped = []
    with (
        closing_iterator(events),
        closing_iterator(active_decisions),
        closing_iterator(forecasts),
        _known_account_path(stopped),
        _PendingMarkouts(
            metrics,
            resolved_fill,
            stop_context=lambda: (latest_state, observed_coverage),
        ) as pending,
    ):
        next_event = next(events, None)
        next_forecast = next(forecasts, None)

        def consume_event(
            event: BookState | BookBoundary,
        ) -> None:
            nonlocal latest_state, terminal_mid, terminal_timestamp, first_market_gap_ns
            terminal_timestamp = observation_order_key(event)[0]
            apply_funding(terminal_timestamp)
            if isinstance(event, BookBoundary):
                if first_market_gap_ns is None and economics.quantity:
                    first_market_gap_ns = event.observed_time_ns
                pending.advance(
                    event.observed_time_ns - 1, latest_state, observed_coverage
                )
                if orders:
                    raise _AccountPathStop(
                        "portfolio_pending_execution_unknown", event.observed_time_ns
                    )
                latest_state = None
                if economics.quantity != 0.0:
                    terminal_mid = None
                pending.expire_all()
                return
            previous_state = latest_state
            state = event
            latest_state = state
            if _covered(
                observed_coverage,
                state.segment_id,
                state.observed_time_ns,
            ):
                terminal_mid = _mid(state)
                economics._settle_funding(state.observed_time_ns, terminal_mid)
            _resolve_observed_markouts(
                pending,
                state,
                observed_coverage,
                previous_state=previous_state,
            )
            if (
                decision_windows_ns is not None
                and max(decision_windows_ns[0][0], observation_start or 0)
                <= state.observed_time_ns
                <= decision_windows_ns[-1][1]
                and _covered(
                    observed_coverage, state.segment_id, state.observed_time_ns
                )
            ):
                economics.mark(state.observed_time_ns, _mid(state))

        def consume_until(target: int) -> None:
            nonlocal market_event_count, next_event
            while (
                next_event is not None
                and observation_order_key(next_event)[0] <= target
                and (
                    observation_stop is None
                    or observation_order_key(next_event)[0] < observation_stop
                )
            ):
                event = next_event
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
            if (
                latest_state is None
                or latest_state.segment_id != segment
                or not _covered(observed_coverage, segment, effective_time_ns)
            ):
                raise _AccountPathStop(
                    "portfolio_pending_execution_unknown", effective_time_ns
                )
            if signed_quantity > 0:
                reserved_buy -= Fraction(signed_quantity)
            else:
                reserved_sell += Fraction(signed_quantity)
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
                levels = latest_state.asks if signed_quantity > 0 else latest_state.bids
                takes = []
                exact_remaining = False
                for price_index, available in levels:
                    quantity = min(remaining, available)
                    if quantity <= 0:
                        continue
                    price = index_to_price(price_index, latest_state.price_unit)
                    takes.append((quantity, price))
                    rest = remaining - quantity
                    # FastTwoSum preserves lost residuals before consuming later depth.
                    if not exact_remaining and (remaining - rest) - quantity != 0:
                        rest = Fraction(remaining) - Fraction(quantity)
                        exact_remaining = True
                    remaining = rest
                    if remaining <= 0:
                        break
                levels_consumed = len(takes)
                if takes:
                    exact_filled = sum(
                        (Fraction(quantity) for quantity, _ in takes), Fraction()
                    )
                    exact_notional = sum(
                        (
                            Fraction(quantity) * Fraction(price)
                            for quantity, price in takes
                        ),
                        Fraction(),
                    )
                    execution_price = exact_notional / exact_filled
                    try:
                        filled = float(exact_filled)
                        notional = float(exact_notional)
                        fill_price = float(execution_price)
                    except OverflowError:
                        raise ValueError("portfolio_execution_fill_invalid") from None
                    filled_orders += 1
                    try:
                        fee = float(
                            _execution_fee(
                                exact_notional,
                                1.0,
                                config.taker_fee_bps,
                            )
                        )
                    except OverflowError:
                        raise ValueError("hft_execution_fee_unrepresentable") from None
                    signed = 1 if signed_quantity > 0 else -1
                    fill_id = economics.fill(
                        timestamp_ns=effective_time_ns,
                        order_id=str(order_id),
                        quantity=signed * exact_filled,
                        price=execution_price,
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
            unfilled_quantity += float(remaining)
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
            while orders and orders[0][2] * _NANOSECONDS_PER_MILLISECOND <= target_ns:
                consume_until(orders[0][2] * _NANOSECONDS_PER_MILLISECOND)
                execute_order(orders[0])
                orders.popleft()
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
                reserved_buy = reserved_sell = Fraction()
            mark = (
                _mid(latest_state)
                if latest_state is not None
                and _covered(observed_coverage, latest_state.segment_id, timestamp_ns)
                else None
            )
            if mark is None and observation_window_ns is not None and not closing:
                raise PortfolioTargetBlocked(
                    "portfolio_episode_initial_valuation_missing"
                )
            if mark is None and economics.quantity != 0.0:
                raise _AccountPathStop(
                    "portfolio_window_boundary_valuation_missing", timestamp_ns
                )
            diagnostic_asset = asset_id if mark is not None else PORTFOLIO_CASH_ASSET_ID
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

        if observation_start is not None and (
            decision_windows_ns is None
            or observation_start < decision_windows_ns[window_index][0]
        ):
            advance(observation_start)
            if latest_state is None or not _covered(
                observed_coverage, latest_state.segment_id, observation_start
            ):
                raise PortfolioTargetBlocked(
                    "portfolio_episode_initial_valuation_missing"
                )
            economics.mark(observation_start, _mid(latest_state))
        for row in active_decisions:
            decision_time = row["timestamp_ms"]
            decision_ns = decision_time * _NANOSECONDS_PER_MILLISECOND
            if observation_start is not None and decision_ns < observation_start:
                continue
            if observation_stop is not None and decision_ns >= observation_stop:
                break
            if decision_windows_ns is not None:
                while (
                    window_index < len(decision_windows_ns)
                    and decision_ns > decision_windows_ns[window_index][1]
                ):
                    if not window_open:
                        boundary(
                            max(
                                decision_windows_ns[window_index][0],
                                observation_start or 0,
                            ),
                            closing=False,
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
                    boundary(
                        max(
                            decision_windows_ns[window_index][0],
                            observation_start or 0,
                        ),
                        closing=False,
                    )
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
                # Input segment IDs describe sample lineage, not this execution source.
                segment = latest_state.segment_id
                reason = "forecast_expired_before_execution"
                if config.latency_ms < prediction_horizon_ms:
                    forecast_end = (
                        decision_time + prediction_horizon_ms
                    ) * _NANOSECONDS_PER_MILLISECOND
                    funding_required_decisions += 1
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
                            buy_limit=_position_headroom(
                                config.position_size,
                                economics.quantity,
                                reserved_buy,
                            ),
                            sell_limit=_position_headroom(
                                config.position_size,
                                -economics.quantity,
                                reserved_sell,
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
            equity_rows += 1
            if selected:
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
                    reserved_buy += Fraction(selected)
                else:
                    reserved_sell -= Fraction(selected)
                advance(decision_ns)
        if decision_windows_ns is not None:
            while window_index < len(decision_windows_ns):
                begin, end = decision_windows_ns[window_index]
                if observation_stop is not None and begin >= observation_stop:
                    break
                if not window_open:
                    boundary(max(begin, observation_start or 0), closing=False)
                if observation_stop is not None and end >= observation_stop:
                    break
                boundary(end, closing=True)
                window_index += 1
                window_open = False
        if truncated:
            advance(observation_stop - 1)
            censored_count = censored_pending = censored_zero_fill = len(orders)
            censored_quantity = sum(abs(order[4]) for order in orders)
            end_timestamp = max(terminal_timestamp, economics.last_timestamp)
        elif decision_windows_ns is None:
            while orders:
                advance(orders[0][2] * _NANOSECONDS_PER_MILLISECOND)
            while next_event is not None:
                consume_until(observation_order_key(next_event)[0])
            end_timestamp = max(
                action_timestamp, terminal_timestamp, economics.last_timestamp
            )
        else:
            end_timestamp = max(
                action_timestamp, terminal_timestamp, economics.last_timestamp
            )
        pending.advance(
            observation_stop - 1 if truncated else end_timestamp,
            latest_state,
            observed_coverage,
        )
        pending.expire_all(reason="observation_truncated" if truncated else None)
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
    if stopped:
        truncated = True
        end_timestamp, terminal_mid = economics.last_timestamp, economics.last_mark
        last_gross = last_net = None
        censored_count = censored_pending = censored_zero_fill = len(orders)
        censored_quantity = sum(abs(order[4]) for order in orders)
    result = {
        "market_observation_status": "PARTIAL"
        if first_market_gap_ns is not None
        else "COMPLETE",
        "market_observation_reason": "crypto_execution_book_coverage_incomplete"
        if first_market_gap_ns is not None
        else None,
        "first_market_gap_ns": first_market_gap_ns,
        "account_path_status": "PARTIAL" if stopped or last_net is None else "COMPLETE",
        "account_stop_reason": stopped[0].reason
        if stopped
        else "portfolio_terminal_valuation_missing"
        if last_net is None
        else None,
        "account_stop_ns": stopped[0].timestamp_ns
        if stopped
        else end_timestamp
        if last_net is None
        else None,
        "simulation": "decision_time_net_gain_taker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "zero_fill_orders": created_orders - filled_orders - censored_zero_fill,
        "censored_active_order_count": censored_count,
        "censored_pending_command_count": censored_pending,
        "censored_order_quantity": censored_quantity,
        "observation_termination": "TRUNCATED" if truncated else "PLANNED_END",
        "unfilled_quantity": unfilled_quantity,
        "equity_rows": equity_rows,
        "no_eligible_decisions": equity_rows == 0,
        "gross_pnl": last_gross,
        "net_pnl": last_net,
        "max_abs_position": float(max_abs_position),
        "market_event_count": market_event_count,
        "prediction_horizon_ms": prediction_horizon_ms,
        "selection_status": "BLOCKED"
        if blocked_decisions and blocked_decisions == funding_required_decisions
        else "PARTIAL"
        if blocked_decisions
        else "UNDEFINED"
        if undefined_decisions
        else "COMPLETE",
        "blocked_decisions": blocked_decisions,
        "funding_gap_policy": "causal_no_new_risk_partial_forecast_v1",
        "funding_required_decisions": funding_required_decisions,
        "funding_covered_decisions": funding_required_decisions - blocked_decisions,
        "funding_decision_coverage": (
            (funding_required_decisions - blocked_decisions)
            / funding_required_decisions
            if funding_required_decisions
            else None
        ),
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
        "funding": None if last_net is None else economics.funding_cashflows,
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
    price_index: int
    remaining: Fraction
    queue_ahead: Fraction
    initial_queue_ahead: Fraction
    segment_id: str
    trade_observation_id: str
    price_unit: str
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
            trade_coverage, order.trade_observation_id, order.effective_at_ns
        )
    ):
        return "activation_coverage_missing"
    if (order.side == "buy" and order.price_index >= state.asks[0][0]) or (
        order.side == "sell" and order.price_index <= state.bids[0][0]
    ):
        return "post_only_crossing_rejected"
    levels = state.bids if order.side == "buy" else state.asks
    bound = getattr(
        state,
        "known_bid_floor_tick" if order.side == "buy" else "known_ask_ceiling_tick",
        None,
    )
    if bound is None:
        bound = levels[-1][0]
    if (order.side == "buy" and order.price_index < bound) or (
        order.side == "sell" and order.price_index > bound
    ):
        return "activation_queue_depth_unknown"
    order.queue_ahead = sum(
        (Fraction(quantity) for tick, quantity in levels if tick == order.price_index),
        Fraction(),
    )
    order.initial_queue_ahead = order.queue_ahead
    return None


def _maker_trade_fill(order, trade):
    if trade.observation_id != order.trade_observation_id:
        return 0.0, "trade_connection_mismatch"
    same_direction = (order.side == "buy") == trade.buyer_is_maker
    if not same_direction:
        return 0.0, "opposite_trade_direction"
    if trade.visible_quantity == 0:
        return 0.0, "zero_visible_trade_quantity"
    if trade.price_index is None:
        raise PortfolioTargetBlocked("hft_trade_price_unknown")
    through = (order.side == "buy" and trade.price_index < order.price_index) or (
        order.side == "sell" and trade.price_index > order.price_index
    )
    if not through and trade.price_index != order.price_index:
        return 0.0, "different_trade_price"
    if trade.visible_quantity is None:
        raise PortfolioTargetBlocked("hft_trade_visible_quantity_unknown")
    if through:
        order.queue_ahead = Fraction()
        return order.remaining, "trade_through"
    visible = Fraction(trade.visible_quantity)
    consumed = min(order.queue_ahead, visible)
    order.queue_ahead -= consumed
    return min(order.remaining, visible - consumed), "same_price_depletion"


def _maker_next_time(event_time, command_time):
    return (
        event_time
        if command_time is None
        else command_time
        if event_time is None
        else min(event_time, command_time)
    )


def _maker_expired_orders(active, timestamp):
    for side, order in tuple(active.items()):
        if order.valid_until_ns <= timestamp:
            active.pop(side)
            yield order


def _maker_apply_command(active, command, state, observed_coverage, trade_coverage):
    timestamp, side, new_order = command
    old_order = active.pop(side, None)
    reason = None
    if new_order is not None:
        reason = (
            "quote_expired_before_activation"
            if new_order.valid_until_ns <= timestamp
            else _maker_join(new_order, state, observed_coverage, trade_coverage)
        )
        if reason is None:
            active[side] = new_order
    return old_order, reason


def _maker_new_order(order_id, side, quantity, state, row, latency_ms):
    timestamp = row["timestamp_ms"] * _NANOSECONDS_PER_MILLISECOND
    return _MakerOrder(
        order_id=order_id,
        side=side,
        price_index=(state.bids if side == "buy" else state.asks)[0][0],
        remaining=Fraction(quantity),
        queue_ahead=Fraction(),
        initial_queue_ahead=Fraction(),
        segment_id=row["book_segment_id"],
        trade_observation_id=row["trade_observation_id"],
        price_unit=state.price_unit,
        created_time_ns=timestamp,
        valid_until_ns=timestamp
        + MAKER_QUOTE_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND,
        effective_at_ns=timestamp + latency_ms * _NANOSECONDS_PER_MILLISECOND,
    )


class MakerCalibration:
    """Reuse one fold history while recomputing each decision's economic response."""

    def __init__(self, samples, events, *, price_unit):
        self.price_unit = Fraction(price_unit)
        self._source_samples = samples
        self._source_events = events
        self._native = None

    def _load(self):
        if self._native is not None:
            return
        history = {"buy": [], "sell": [], "events": []}
        with closing_iterator(self._source_samples()) as samples:
            for sample in samples:
                history[sample["side"]].append(dict(sample))
        with closing_iterator(self._source_events()) as events:
            for event in events:
                if event.event_id != len(history["events"]):
                    raise ValueError("crypto_calibration_event_order_invalid")
                history["events"].append(event)
        self._native = history

    def samples(self, side):
        self._load()
        yield from self._native[side]

    def events(self, start, stop):
        self._load()
        if not 0 <= start <= stop <= len(self._native["events"]):
            raise ValueError("crypto_calibration_event_coordinates_invalid")
        for index in range(start, stop):
            yield self._native["events"][index]

    def _response(
        self,
        sample,
        *,
        state,
        decision_ns,
        new_order,
        active_order,
        commands,
        lifetime_ns,
        replace_quantity,
        history_scale,
        events,
    ):
        if replace_quantity == 0 and active_order is None and not commands:
            # Validate native history coordinates before skipping economic replay.
            with closing_iterator(events) as events:
                next(events, None)
            return
        side = new_order.side
        start = int(sample["decision_at"])
        end = start + lifetime_ns
        origin = (
            sample["initial_bid_index" if side == "buy" else "initial_ask_index"]
            * history_scale
        )
        current_origin = (state.bids if side == "buy" else state.asks)[0][0]
        book_context = sample["book_segment_id"]
        trade_context = sample["trade_observation_id"]

        def mapped(order):
            return _MakerOrder(
                **{
                    **vars(order),
                    "price_index": origin + order.price_index - current_origin,
                    "segment_id": book_context
                    if order.segment_id == new_order.segment_id
                    else order.segment_id,
                    "trade_observation_id": trade_context
                    if order.trade_observation_id == new_order.trade_observation_id
                    else order.trade_observation_id,
                    "created_time_ns": start + order.created_time_ns - decision_ns,
                    "effective_at_ns": start + order.effective_at_ns - decision_ns,
                    "valid_until_ns": start + order.valid_until_ns - decision_ns,
                }
            )

        active = {} if active_order is None else {side: mapped(active_order)}
        pending = deque(
            (
                start + stamp - decision_ns,
                command_side,
                None if order is None else mapped(order),
            )
            for stamp, command_side, order in commands
        )
        candidate = None
        candidate_reason = None
        if replace_quantity is not None:
            candidate = mapped(new_order) if replace_quantity > 0 else None
            if candidate is not None:
                candidate.remaining = Fraction(replace_quantity)
            pending.append(
                (start + new_order.effective_at_ns - decision_ns, side, candidate)
            )
        observed = {
            book_context: (
                start,
                start + MAKER_QUOTE_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND,
            )
        }
        traded = {trade_context: observed[book_context]}
        current_state = SimpleNamespace(
            segment_id=book_context,
            known_bid_floor_tick=origin
            + (getattr(state, "known_bid_floor_tick", None) or state.bids[-1][0])
            - current_origin,
            known_ask_ceiling_tick=origin
            + (getattr(state, "known_ask_ceiling_tick", None) or state.asks[-1][0])
            - current_origin,
            bids=tuple(
                (origin + tick - current_origin, quantity)
                for tick, quantity in state.bids
            ),
            asks=tuple(
                (origin + tick - current_origin, quantity)
                for tick, quantity in state.asks
            ),
        )
        with closing_iterator(events) as events:
            event = next(events, None)
            while event is not None or pending:
                event_time = event.event_at if event is not None else None
                command_time = pending[0][0] if pending else None
                stamp = _maker_next_time(event_time, command_time)
                # Include known OOS endpoint events; expire each order first at TTL.
                if stamp > end:
                    break
                for _expired in _maker_expired_orders(active, stamp):
                    if _expired is candidate:
                        candidate_reason = "expired"
                if event_time == stamp:
                    kind = event.kind
                    if kind == "boundary":
                        if candidate is not None and candidate_reason is None:
                            candidate_reason = "book_boundary"
                        active.clear()
                        pending.clear()
                        current_state = None
                    elif kind == "book":
                        next_state = (
                            event
                            if history_scale == 1
                            else SimpleNamespace(
                                segment_id=event.segment_id,
                                known_bid_floor_tick=(
                                    getattr(event, "known_bid_floor_tick", None)
                                    or event.bids[-1][0]
                                )
                                * history_scale,
                                known_ask_ceiling_tick=(
                                    getattr(event, "known_ask_ceiling_tick", None)
                                    or event.asks[-1][0]
                                )
                                * history_scale,
                                bids=tuple(
                                    (tick * history_scale, quantity)
                                    for tick, quantity in event.bids
                                ),
                                asks=tuple(
                                    (tick * history_scale, quantity)
                                    for tick, quantity in event.asks
                                ),
                            )
                        )
                        if (
                            current_state is not None
                            and next_state.segment_id != current_state.segment_id
                        ):
                            if candidate is not None and candidate_reason is None:
                                candidate_reason = "book_boundary"
                            active.clear()
                            pending.clear()
                        current_state = next_state
                    else:
                        order = active.get(side)
                        if (
                            order is not None
                            and current_state is not None
                            and order.segment_id == current_state.segment_id
                        ):
                            trade = (
                                event
                                if history_scale == 1
                                else SimpleNamespace(
                                    price_index=None
                                    if event.price_index is None
                                    else event.price_index * history_scale,
                                    visible_quantity=event.visible_quantity,
                                    observation_id=event.observation_id,
                                    buyer_is_maker=event.buyer_is_maker,
                                )
                            )
                            quantity, _reason = _maker_trade_fill(order, trade)
                            if quantity > 0:
                                price = index_to_price(
                                    current_origin + order.price_index - origin,
                                    new_order.price_unit,
                                )
                                _yield_value = (
                                    stamp - start,
                                    quantity,
                                    price,
                                    order is candidate,
                                )
                                try:
                                    yield _yield_value
                                except GeneratorExit:
                                    return
                                order.remaining -= quantity
                                if order.remaining <= 0:
                                    active.pop(side)
                                    if order is candidate:
                                        candidate_reason = "filled"
                    event = next(events, None)
                    continue
                command = pending.popleft()
                _old, reason = _maker_apply_command(
                    active, command, current_state, observed, traded
                )
                if reason in {
                    "activation_queue_depth_unknown",
                    "activation_coverage_missing",
                }:
                    raise PortfolioTargetBlocked(reason)
                if command[2] is candidate and reason is not None:
                    candidate_reason = reason
            return candidate_reason or "expired"

    def select(
        self,
        *,
        state,
        decision_ns,
        new_order,
        active_order,
        commands,
        lifetime_ns,
        prediction,
        fee_bps,
        funding,
    ):
        choices = set()
        curves = {}
        sign = 1.0 if new_order.side == "buy" else -1.0
        mid = _mid(state)
        forecast = _positive(mid * (1.0 + prediction), "hft_prediction_price_invalid")
        limit = new_order.remaining
        price = index_to_price(new_order.price_index, new_order.price_unit)
        fee_per_unit = _execution_fee(1.0, price, fee_bps)
        keep_gain = cancel_gain = 0.0
        sample_count = 0
        # Explicit source collections need not be nested. Historical integer
        # coordinates are reusable only when the current unit exactly divides
        # their frozen unit; taking a fractional ratio's numerator loses prices.
        ratio = self.price_unit / Fraction(new_order.price_unit)
        if ratio.denominator != 1:
            raise ValueError("hft_maker_calibration_price_unit_incompatible")
        history_scale = ratio.numerator

        def records():
            nonlocal keep_gain, cancel_gain, sample_count
            for quantity in (0.0, limit):
                choices.add(quantity)
            samples = self.samples(new_order.side)
            for sample in samples:
                sample_id = sample_count
                sample_count += 1
                error = sample["forecast_error"]
                funded = [0.0] * len(funding)
                for replacement in (None, limit):
                    if replacement is None and active_order is None and not commands:
                        continue
                    cumulative = Fraction()
                    response = self._response(
                        sample,
                        state=state,
                        decision_ns=decision_ns,
                        new_order=new_order,
                        active_order=active_order,
                        commands=commands,
                        lifetime_ns=lifetime_ns,
                        replace_quantity=replacement,
                        history_scale=history_scale,
                        events=self.events(
                            int(sample["event_start"]), int(sample["event_end"])
                        ),
                    )
                    for elapsed, quantity, price, is_new in response:
                        if is_new:
                            cumulative += quantity
                            if 0.0 <= cumulative <= limit:
                                choices.add(cumulative)
                            for index, (stamp, _cash) in enumerate(funding):
                                if elapsed < stamp:
                                    funded[index] = max(funded[index], cumulative)
                        else:
                            fee = _execution_fee(1.0, price, fee_bps)
                            cash = sum(
                                cash for stamp, cash in funding if elapsed < stamp
                            )
                            gain = quantity * (
                                sign * (forecast - price) - fee + sign * mid * error
                            )
                            gain += quantity * sign * cash
                            if not math.isfinite(gain):
                                raise ValueError("hft_expected_gain_unrepresentable")
                            if replacement is None:
                                keep_gain += gain
                            else:
                                cancel_gain += gain
                    if replacement is not None:
                        curves.setdefault("response", []).append(
                            (cumulative, sample_id, error)
                        )
                for index, quantity in enumerate(funded):
                    curves.setdefault(f"funding_{index}", []).append(
                        (quantity, sample_id, 0.0)
                    )

        records()
        choices = sorted(choices)

        def curve(namespace):
            rows = sorted(curves.get(namespace, ()), key=lambda row: (row[0], row[1]))
            for quantity, group in groupby(rows, key=lambda row: row[0]):
                count, error_sum = 0, 0.0
                for _quantity, _sample_id, error in group:
                    count += 1
                    error_sum += error
                yield (quantity, count, error_sum)

        edge = sign * (forecast - price) - fee_per_unit

        def responses(namespace, *, with_error):

            def weight(count, error):
                return count * edge + sign * mid * error if with_error else count

            total = 0.0 if with_error else 0
            groups = tuple(curve(namespace))
            for _quantity, count, error in groups:
                total += weight(count, error)
            weighted = 0.0
            prefix = 0.0 if with_error else 0
            groups = iter(groups)
            current = next(groups, None)
            for choice in choices:
                while current is not None and current[0] <= choice:
                    quantity, count, error = current
                    value = weight(count, error)
                    weighted += quantity * value
                    prefix += value
                    current = next(groups, None)
                yield weighted + choice * (total - prefix)

        streams = [
            responses("response", with_error=True),
            *(
                responses(f"funding_{index}", with_error=False)
                for index in range(len(funding))
            ),
        ]
        best_gain = -float("inf")
        best_quantity = 0.0
        for choice, base, *funded in zip(choices, *streams, strict=True):
            gain = base
            for (_stamp, cash), value in zip(funding, funded, strict=True):
                gain += sign * cash * value
            gain = (gain + cancel_gain) / sample_count
            if not math.isfinite(gain):
                raise ValueError("hft_expected_gain_unrepresentable")
            if gain > best_gain:
                best_gain = gain
                best_quantity = choice
        keep_gain /= sample_count
        keep = bool(active_order is not None or commands) and keep_gain >= best_gain
        gain = float(keep_gain) if keep else float(best_gain)
        if not math.isfinite(gain):
            raise ValueError("hft_expected_gain_unrepresentable")
        return keep, 0.0 if keep else float(best_quantity), gain


def _source_quote_context(row, state, trade_coverage, timestamp):
    """Bind eligible sample lineage to this source after causal event advancement."""
    identities = (
        trade_coverage.covering(timestamp, timestamp)
        if hasattr(trade_coverage, "covering")
        else tuple(
            identity
            for identity, (start, stop) in trade_coverage.items()
            if start <= timestamp < stop
        )
    )
    return {
        **row,
        "book_segment_id": None if state is None else state.segment_id,
        "trade_observation_id": identities[0] if len(identities) == 1 else None,
    }


def replay_maker_calibration(
    book_events,
    trades,
    decisions,
    *,
    order_quantity,
    latency_ms,
    observed_coverage,
    trade_coverage,
    cutoff_ns,
    on_sample,
    on_event,
):
    """Capture one shared mature validation tape and independent reference quotes."""
    _positive(order_quantity, "hft_maker_order_quantity_invalid")
    _nonnegative_integer(latency_ms, "hft_maker_latency_invalid")
    events = iter(_market_events(book_events, trades))
    rows = iter(_checked_quote_schedule(decisions))
    latest_state = next_event = latest_book_id = None
    samples = []
    counts = {"buy": 0, "sell": 0}
    excluded = event_count = 0
    with closing_iterator(events), closing_iterator(rows):
        next_event = next(events, None)

        def record(event):
            nonlocal event_count
            index = event_count
            on_event(event_id=index, event=event)
            event_count += 1
            return index

        def finish(sample):
            order = sample["order"]
            on_sample(
                side=order.side,
                sample_id=counts[order.side],
                decision_ns=order.created_time_ns,
                prediction_end_ns=order.valid_until_ns,
                available_ns=sample["available_ns"],
                prediction=sample["prediction"],
                observed_return=sample["observed_return"],
                filled=float(Fraction(order_quantity) - order.remaining),
                quote_quantity=order_quantity,
                terminal_reason=sample["terminal_reason"] or "expired",
                book_segment_id=order.segment_id,
                trade_observation_id=order.trade_observation_id,
                initial_book_event_id=sample["initial_book_event_id"],
                initial_bid_index=sample["initial_bid_index"],
                initial_ask_index=sample["initial_ask_index"],
                event_start=sample["event_start"],
                event_end=event_count,
            )
            counts[order.side] += 1
            samples.remove(sample)

        def exclude_references(decisions=None):
            nonlocal excluded
            for sample in tuple(samples):
                order = sample["order"]
                if decisions is None or order.created_time_ns in decisions:
                    excluded += order.side == "buy"
                    samples.remove(sample)

        def process_until(target):
            nonlocal latest_state, latest_book_id, next_event
            while True:
                timers = [
                    min(
                        sample["order"].valid_until_ns,
                        sample["order"].effective_at_ns
                        if not sample["attempted"] and sample["terminal_reason"] is None
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
                        finish(sample)
                if event_time == due:
                    event = next_event[1]
                    if isinstance(event, TradeBoundary):
                        exclude_references()
                        latest_book_id = None
                    elif isinstance(event, BookBoundary):
                        exclude_references()
                        latest_state = latest_book_id = None
                    elif isinstance(event, TradeEvent):
                        if samples:
                            record(event)
                        unknown_decisions = set()
                        for sample in samples:
                            order = sample["order"]
                            if (
                                not sample["activated"]
                                or sample["terminal_reason"] is not None
                                or latest_state is None
                                or latest_state.segment_id != order.segment_id
                            ):
                                continue
                            try:
                                quantity, _reason = _maker_trade_fill(order, event)
                            except PortfolioTargetBlocked:
                                unknown_decisions.add(order.created_time_ns)
                                continue
                            order.remaining -= quantity
                            if order.remaining <= 0:
                                sample["terminal_reason"] = "filled"
                        if unknown_decisions:
                            exclude_references(unknown_decisions)
                    else:
                        state = event
                        if (
                            latest_state is not None
                            and state.segment_id != latest_state.segment_id
                        ):
                            exclude_references()
                        latest_state = state
                        latest_book_id = record(state) if samples else None
                    next_event = next(events, None)
                else:
                    unknown_decisions = set()
                    for sample in tuple(samples):
                        order = sample["order"]
                        if (
                            order.created_time_ns not in unknown_decisions
                            and not sample["attempted"]
                            and sample["terminal_reason"] is None
                            and order.effective_at_ns <= due
                        ):
                            sample["attempted"] = True
                            reason = _maker_join(
                                order, latest_state, observed_coverage, trade_coverage
                            )
                            if reason in {
                                "activation_queue_depth_unknown",
                                "activation_coverage_missing",
                            }:
                                unknown_decisions.add(order.created_time_ns)
                                exclude_references({order.created_time_ns})
                                continue
                            sample["activated"] = reason is None
                            sample["terminal_reason"] = reason

        for row in rows:
            timestamp = row["timestamp_ms"] * _NANOSECONDS_PER_MILLISECOND
            end = timestamp + MAKER_QUOTE_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND
            if row["label_available_at_ns"] > cutoff_ns or end > cutoff_ns:
                raise ValueError("maker_calibration_outcome_not_mature")
            process_until(timestamp)
            row = _source_quote_context(row, latest_state, trade_coverage, timestamp)
            valid = (
                latest_state is not None
                and row["trade_observation_id"] is not None
                and _covered(observed_coverage, row["book_segment_id"], timestamp)
                and _covered(observed_coverage, row["book_segment_id"], end - 1)
                and _covered(trade_coverage, row["trade_observation_id"], timestamp)
                and _covered(trade_coverage, row["trade_observation_id"], end - 1)
            )
            if not valid or row["proxy_prediction"] < 0.5:
                excluded += 1
                continue
            if latest_book_id is None:
                latest_book_id = record(latest_state)
            for side in ("buy", "sell"):
                order = _maker_new_order(
                    counts[side],
                    side,
                    order_quantity,
                    latest_state,
                    row,
                    latency_ms,
                )
                samples.append(
                    {
                        "order": order,
                        "prediction": row["price_prediction"],
                        "observed_return": row["price_label"],
                        "available_ns": row["label_available_at_ns"],
                        "activated": False,
                        "attempted": False,
                        "terminal_reason": None,
                        "initial_book_event_id": latest_book_id,
                        "initial_bid_index": latest_state.bids[0][0],
                        "initial_ask_index": latest_state.asks[0][0],
                        "event_start": event_count,
                    }
                )
            process_until(timestamp)
        process_until(cutoff_ns)
        if samples:
            raise ValueError("maker_calibration_outcome_not_mature")
    return {"samples": counts, "excluded_decisions": excluded}


def _checked_trades(
    trades: Iterable[TradeEvent | TradeBoundary],
) -> Iterator[TradeEvent | TradeBoundary]:
    previous: tuple[int, ...] | None = None
    active = iter(trades)
    with closing_iterator(active):
        for trade in active:
            if not isinstance(trade, TradeEvent | TradeBoundary):
                raise TypeError("hft_execution_trade_event_invalid")
            key = observation_order_key(trade)
            if previous is not None and key < previous:
                raise ValueError("hft_execution_time_non_monotonic")
            previous = key
            if isinstance(trade, TradeBoundary):
                yield trade
                continue
            _positive(trade.quantity, "hft_execution_trade_quantity_invalid")
            if trade.visible_quantity is not None:
                visible = strict_finite_float(
                    trade.visible_quantity,
                    error="hft_execution_trade_visible_quantity_invalid",
                )
                if visible < 0 or visible > trade.quantity:
                    raise ValueError("hft_execution_trade_visible_quantity_invalid")
            if not isinstance(trade.buyer_is_maker, bool):
                raise ValueError("hft_execution_trade_side_invalid")
            try:
                yield trade
            except GeneratorExit:
                return


def _market_events(
    books: Iterable[BookState | BookBoundary],
    trades: Iterable[TradeEvent | TradeBoundary],
) -> Iterator[tuple[tuple[int, ...], object]]:
    book_events = iter(_checked_book_events(books))
    trade_events = iter(_checked_trades(trades))
    book_next: BookState | BookBoundary | None = None
    trade_next: TradeEvent | TradeBoundary | None = None
    pending_event: object | None = None
    with closing_iterator(book_events), closing_iterator(trade_events):
        book_next = next(book_events, None)
        trade_next = next(trade_events, None)
        while book_next is not None or trade_next is not None:
            if trade_next is None or (
                book_next is not None
                and observation_order_key(book_next)
                <= observation_order_key(trade_next)
            ):
                pending_event = book_next
                assert book_next is not None
                key = observation_order_key(book_next)
                book_next = next(book_events, None)
            else:
                pending_event = trade_next
                assert trade_next is not None
                key = observation_order_key(trade_next)
                trade_next = next(trade_events, None)
            _yield_value = key, pending_event
            try:
                yield _yield_value
            except GeneratorExit:
                return
            pending_event = None


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
                "trade_observation_id",
            ):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError("hft_maker_quote_schedule_identity_invalid")
            timestamp = row.get("timestamp_ms")
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError("hft_maker_quote_schedule_time_invalid")
            if previous is not None and timestamp <= previous:
                raise ValueError("hft_maker_quote_schedule_time_non_monotonic")
            previous = timestamp
            try:
                yield row
            except GeneratorExit:
                return


def simulate_maker(
    book_events,
    trades,
    quote_schedule,
    *,
    config: MakerConfig,
    observed_coverage,
    trade_coverage,
    capital,
    asset_id,
    funding_events,
    funding_forecasts=(),
    funding_coverage=None,
    calibrations=None,
    decision_windows_ns=None,
    observation_window_ns=None,
    on_ledger=None,
):
    """Freeze passive quotes at decisions; replay delayed replacements and expiry."""
    if calibrations is None and (
        config.inventory_target is None or config.inventory_skew is None
    ):
        raise ValueError("hft_maker_inventory_parameters_required")
    if calibrations is not None and (
        config.inventory_target is not None or config.inventory_skew is not None
    ):
        raise ValueError("hft_maker_calibrated_inventory_parameters_invalid")
    observation_start = (
        None if observation_window_ns is None else observation_window_ns[0]
    )
    observation_stop = (
        None if observation_window_ns is None else observation_window_ns[1]
    )
    truncated = observation_stop is not None and (
        decision_windows_ns is None or observation_stop <= decision_windows_ns[-1][1]
    )
    first_market_gap_ns = None
    censored_count = censored_pending = censored_zero_fill = 0
    censored_quantity = 0.0
    events = iter(_market_events(book_events, trades))
    decisions = iter(_checked_quote_schedule(quote_schedule))
    forecasts = iter(funding_forecasts)
    active = {}
    commands = deque()
    metrics = _Metrics()
    economics = PortfolioEconomics(capital, asset_id, on_ledger)
    apply_funding = _funding_consumer(
        funding_events,
        economics,
        funding_coverage,
        lambda: latest_state,
        observed_coverage,
    )
    latest_state = next_event = None
    trade_observation_id = None
    active_forecast = next_forecast = None
    forecast_timestamp = -1
    created_orders = filled_orders = cancelled_orders = 0
    partial_fill_events = trade_through_fill_events = post_only_rejections = 0
    equity_rows = market_event_count = blocked_decisions = undefined_decisions = 0
    unknown_queue_decisions = 0
    funding_required_decisions = 0
    max_abs_position = 0.0
    all_lifetime_count = all_lifetime_sum_ns = 0
    filled_lifetime_count = filled_lifetime_sum_ns = 0
    decision_time_ns = last_event_time_ns = 0
    window_index = 0
    if observation_start is not None and decision_windows_ns is not None:
        while (
            window_index < len(decision_windows_ns)
            and decision_windows_ns[window_index][1] < observation_start
        ):
            window_index += 1
    window_open = False
    next_day = (
        None
        if decision_windows_ns is None
        else (
            (max(decision_windows_ns[0][0], observation_start or 0) + _UTC_DAY_NS - 1)
            // _UTC_DAY_NS
        )
        * _UTC_DAY_NS
    )

    def resolved_fill(fill):
        economics.diagnostic(dict(fill))

    def observed_order_end(order):
        return min(
            order.valid_until_ns,
            observed_coverage[order.segment_id][1],
            trade_coverage[order.trade_observation_id][1],
        )

    def risk_coverage_stop():
        ends = []
        for order in (
            *active.values(),
            *(order for _, _, order in commands if order is not None),
        ):
            end = min(
                observed_coverage[order.segment_id][1],
                trade_coverage[order.trade_observation_id][1],
            )
            if (
                end < order.valid_until_ns
                and order.effective_at_ns < order.valid_until_ns
            ):
                ends.append(end)
        return min(ends) if ends else None

    def finish_order(order, timestamp_ns, *, cancelled):
        nonlocal all_lifetime_count, all_lifetime_sum_ns, cancelled_orders
        nonlocal filled_lifetime_count, filled_lifetime_sum_ns
        lifetime = max(
            0, min(timestamp_ns, observed_order_end(order)) - order.created_time_ns
        )
        all_lifetime_count += 1
        all_lifetime_sum_ns += lifetime
        if order.had_fill:
            filled_lifetime_count += 1
            filled_lifetime_sum_ns += lifetime
        if cancelled:
            cancelled_orders += 1

    stopped = []
    with (
        closing_iterator(events),
        closing_iterator(decisions),
        closing_iterator(forecasts),
        _known_account_path(stopped),
        _PendingMarkouts(
            metrics,
            resolved_fill,
            stop_context=lambda: (latest_state, observed_coverage),
        ) as pending,
    ):
        next_event = next(events, None)
        next_forecast = next(forecasts, None)

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

        def observe_trade_connection(identity, timestamp):
            nonlocal trade_observation_id
            if trade_observation_id is not None and trade_observation_id != identity:
                if active or any(order is not None for _, _, order in commands):
                    raise _AccountPathStop(
                        "portfolio_pending_execution_unknown", timestamp
                    )
            trade_observation_id = identity

        def expire_orders(timestamp):
            # Only decision-time TTL expires orders; gaps never imply cancellation.
            for order in _maker_expired_orders(active, timestamp):
                finish_order(order, timestamp, cancelled=True)
            for _ in range(len(commands)):
                command = commands.popleft()
                order = command[2]
                if order is not None and order.valid_until_ns <= timestamp:
                    finish_order(order, timestamp, cancelled=True)
                else:
                    commands.append(command)

        def observed_mark(timestamp):
            return (
                _mid(latest_state)
                if latest_state is not None
                and _covered(observed_coverage, latest_state.segment_id, timestamp)
                else None
            )

        def process_book(event):
            nonlocal latest_state, first_market_gap_ns
            if isinstance(event, BookBoundary):
                if first_market_gap_ns is None and economics.quantity:
                    first_market_gap_ns = event.observed_time_ns
                pending.advance(
                    event.observed_time_ns - 1, latest_state, observed_coverage
                )
                if active or any(order is not None for _, _, order in commands):
                    raise _AccountPathStop(
                        "portfolio_pending_execution_unknown", event.observed_time_ns
                    )
                latest_state = None
                pending.expire_all()
                return
            previous = latest_state
            state = event
            if previous is not None and previous.segment_id != state.segment_id:
                if active or any(order is not None for _, _, order in commands):
                    raise _AccountPathStop(
                        "portfolio_pending_execution_unknown", state.observed_time_ns
                    )
            latest_state = state
            if _covered(observed_coverage, state.segment_id, state.observed_time_ns):
                economics._settle_funding(state.observed_time_ns, _mid(state))
            _resolve_observed_markouts(
                pending, state, observed_coverage, previous_state=previous
            )
            if (
                decision_windows_ns is None
                or max(decision_windows_ns[0][0], observation_start or 0)
                <= state.observed_time_ns
                <= decision_windows_ns[-1][1]
            ) and _covered(observed_coverage, state.segment_id, state.observed_time_ns):
                economics.mark(state.observed_time_ns, _mid(state))

        def process_trade(trade):
            nonlocal filled_orders, partial_fill_events, trade_through_fill_events
            nonlocal max_abs_position
            observe_trade_connection(trade.observation_id, trade.observed_time_ns)
            if (
                latest_state is None
                or not _covered(
                    observed_coverage, latest_state.segment_id, trade.observed_time_ns
                )
                or not _covered(
                    trade_coverage, trade.observation_id, trade.observed_time_ns
                )
            ):
                return
            for side in ("buy", "sell"):
                order = active.get(side)
                if order is None or order.segment_id != latest_state.segment_id:
                    continue
                try:
                    quantity, reason = _maker_trade_fill(order, trade)
                except PortfolioTargetBlocked as error:
                    raise _AccountPathStop(str(error), trade.observed_time_ns) from None
                if quantity <= 0:
                    continue
                price = index_to_price(order.price_index, order.price_unit)
                sign = 1 if side == "buy" else -1
                try:
                    fee = float(_execution_fee(quantity, price, config.maker_fee_bps))
                except OverflowError:
                    raise ValueError("hft_execution_fee_unrepresentable") from None
                mid = _mid(latest_state)
                fill = {
                    "order_id": order.order_id,
                    "timestamp_ms": trade.observed_time_ns
                    // _NANOSECONDS_PER_MILLISECOND,
                    "segment_id": order.segment_id,
                    "side": side,
                    "fill_reason": reason,
                    "fill_price": price,
                    "fill_qty": float(quantity),
                    "notional": float(quantity * Fraction(price)),
                    "fee": fee,
                    "estimated_queue_ahead_at_join": float(order.initial_queue_ahead),
                    "estimated_queue_ahead_before_fill": float(order.queue_ahead),
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
                stamp = _maker_next_time(event_time, command_time)
                if stamp > target:
                    break
                stop = risk_coverage_stop()
                if stop is not None and stamp >= stop:
                    apply_funding(stop - 1)
                    raise _AccountPathStop("portfolio_pending_execution_unknown", stop)
                apply_funding(stamp)
                expire_orders(stamp)
                if event_time == stamp:
                    market_event_count += 1
                    last_event_time_ns = stamp
                    item = next_event[1]
                    if isinstance(item, TradeBoundary):
                        if active or any(order is not None for _, _, order in commands):
                            raise _AccountPathStop(
                                "portfolio_pending_execution_unknown", stamp
                            )
                    elif isinstance(item, TradeEvent):
                        process_trade(item)
                    else:
                        process_book(item)
                    next_event = next(events, None)
                    continue
                if not commands or commands[0][0] != stamp:
                    continue
                command = commands.popleft()
                _effective, side, new_order = command
                old, reason = _maker_apply_command(
                    active, command, latest_state, observed_coverage, trade_coverage
                )
                if reason in {
                    "activation_queue_depth_unknown",
                    "activation_coverage_missing",
                }:
                    # Activation is unresolved, not an exchange rejection. Retain
                    # the command as censored exposure at the local account stop.
                    commands.appendleft(command)
                    raise _AccountPathStop(reason, stamp)
                if old is not None:
                    finish_order(old, stamp, cancelled=True)
                if new_order is not None:
                    if reason is not None:
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
            stop = risk_coverage_stop()
            if stop is not None and target >= stop:
                apply_funding(stop - 1)
                raise _AccountPathStop("portfolio_pending_execution_unknown", stop)
            apply_funding(target)
            expire_orders(target)

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
            if mark is None and observation_window_ns is not None and not closing:
                raise PortfolioTargetBlocked(
                    "portfolio_episode_initial_valuation_missing"
                )
            if mark is None and economics.quantity != 0:
                raise _AccountPathStop(
                    "portfolio_window_boundary_valuation_missing", timestamp
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
                                number=float(quantity),
                            ),
                        )
                        if closing
                        else ()
                    ),
                ),
            )

        if observation_start is not None and (
            decision_windows_ns is None
            or observation_start < decision_windows_ns[window_index][0]
        ):
            process_until(observation_start)
            mark = observed_mark(observation_start)
            if mark is None:
                raise PortfolioTargetBlocked(
                    "portfolio_episode_initial_valuation_missing"
                )
            economics.mark(observation_start, mark)
        for row in decisions:
            decision_time_ns = row["timestamp_ms"] * _NANOSECONDS_PER_MILLISECOND
            if observation_start is not None and decision_time_ns < observation_start:
                continue
            if observation_stop is not None and decision_time_ns >= observation_stop:
                break
            if decision_windows_ns is not None:
                while (
                    window_index < len(decision_windows_ns)
                    and decision_time_ns > decision_windows_ns[window_index][1]
                ):
                    if not window_open:
                        boundary(
                            max(
                                decision_windows_ns[window_index][0],
                                observation_start or 0,
                            ),
                            closing=False,
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
                    boundary(
                        max(
                            decision_windows_ns[window_index][0],
                            observation_start or 0,
                        ),
                        closing=False,
                    )
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
            row = _source_quote_context(
                row, latest_state, trade_coverage, decision_time_ns
            )
            observe_trade_connection(row["trade_observation_id"], decision_time_ns)
            mid = observed_mark(decision_time_ns)
            valid = mid is not None and row["trade_observation_id"] is not None
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
                decision_time_ns + MAKER_QUOTE_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND
            )
            lifetime = MAKER_QUOTE_HORIZON_MS * _NANOSECONDS_PER_MILLISECOND
            if decision_windows_ns is not None:
                lifetime = min(
                    lifetime, decision_windows_ns[window_index][1] - decision_time_ns
                )
            funding = []
            funding_known = True
            if (
                calibrations is not None
                and valid
                and gate
                and lifetime > 0
                and (
                    config.latency_ms * _NANOSECONDS_PER_MILLISECOND < lifetime
                    or active
                    or any(order is not None for _time, _side, order in commands)
                )
            ):
                funding_required_decisions += 1
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
                    funding_known = all(cash is not None for _stamp, cash in funding)
                if not funding_known:
                    blocked_decisions += 1
            diagnostics = []
            for side, sign in (("buy", 1), ("sell", -1)):
                unknown_queue_reason = None
                desired_price = (
                    None
                    if not valid
                    else (latest_state.bids if side == "buy" else latest_state.asks)[0][
                        0
                    ]
                )
                headroom = _position_headroom(
                    config.max_position, sign * economics.quantity
                )
                allowed = (
                    valid
                    and gate
                    and funding_known
                    and config.latency_ms * _NANOSECONDS_PER_MILLISECOND < lifetime
                )

                def choose(
                    limit,
                    sign=sign,
                    allowed=allowed,
                ):
                    if not allowed:
                        return 0.0, 0.0
                    limit = min(config.order_quantity, limit)
                    skew = float(
                        (economics.quantity - Fraction(config.inventory_target))
                        / Fraction(config.max_position)
                    )
                    return min(
                        limit,
                        config.order_quantity
                        * max(0.0, 1.0 - sign * config.inventory_skew * skew),
                    ), None

                quantity, gain = (
                    choose(headroom) if calibrations is None else (0.0, 0.0)
                )
                side_commands = [command for command in commands if command[1] == side]
                pending_side = [order for _time, _side, order in side_commands]
                current = pending_side[-1] if pending_side else active.get(side)
                reserved = sum(
                    (
                        Fraction(order.remaining)
                        for order in pending_side
                        if order is not None
                    ),
                    Fraction(),
                )
                if side in active:
                    reserved += Fraction(active[side].remaining)
                if config.latency_ms == 0:
                    reserved = Fraction()
                available = _position_headroom(
                    config.max_position, sign * economics.quantity, reserved
                )
                keep = (
                    current is not None
                    and quantity > 0
                    and current.price_index == desired_price
                    and current.remaining == quantity
                    and current.valid_until_ns > decision_time_ns
                )
                if calibrations is not None:
                    if valid and gate and funding_known and lifetime > 0:
                        proposed = _maker_new_order(
                            0,
                            side,
                            min(config.order_quantity, available) if allowed else 0.0,
                            latest_state,
                            row,
                            config.latency_ms,
                        )
                        try:
                            keep, quantity, gain = calibrations[row["fold_id"]].select(
                                state=latest_state,
                                decision_ns=decision_time_ns,
                                new_order=proposed,
                                active_order=active.get(side),
                                commands=side_commands,
                                lifetime_ns=lifetime,
                                prediction=row["price_prediction"],
                                fee_bps=config.maker_fee_bps,
                                funding=funding,
                            )
                        except PortfolioTargetBlocked as error:
                            if str(error) not in {
                                "activation_queue_depth_unknown",
                                "activation_coverage_missing",
                                "hft_trade_visible_quantity_unknown",
                            }:
                                raise
                            unknown_queue_reason = str(error)
                            unknown_queue_decisions += 1
                            keep, quantity, gain = True, 0.0, None
                        if keep and current is not None:
                            quantity = current.remaining
                if not keep:
                    if calibrations is None:
                        quantity, gain = choose(available)
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
                                config.latency_ms,
                            )
                        effective = (
                            decision_time_ns
                            + config.latency_ms * _NANOSECONDS_PER_MILLISECOND
                        )
                        commands.append((effective, side, order))
                if not funding_known:
                    gain = None
                reason = (
                    unknown_queue_reason
                    if unknown_queue_reason is not None
                    else "decision_book_or_trade_unavailable"
                    if not valid
                    else "proxy_gate_closed"
                    if not gate
                    else "portfolio_expected_funding_missing"
                    if not funding_known
                    else "keep_existing_schedule_without_renewal"
                    if keep and calibrations is not None
                    else "keep_queue_without_renewal"
                    if keep
                    else "cancel_replace_maximizes_expected_net_gain"
                    if quantity and calibrations is not None
                    else "cancel_maximizes_expected_net_gain"
                    if calibrations is not None and current is not None
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
                            number=float(quantity),
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
            if mid is not None or economics.quantity == 0:
                economics.mark(decision_time_ns, mid, diagnostics=tuple(diagnostics))
            process_until(decision_time_ns)
            equity_rows += 1
        if decision_windows_ns is not None:
            while window_index < len(decision_windows_ns):
                begin, end = decision_windows_ns[window_index]
                if observation_stop is not None and begin >= observation_stop:
                    break
                if not window_open:
                    boundary(max(begin, observation_start or 0), closing=False)
                if observation_stop is not None and end >= observation_stop:
                    break
                boundary(end, closing=True)
                window_index += 1
                window_open = False
        if truncated:
            process_until(observation_stop - 1)
            censored_orders = (
                *active.values(),
                *(order for _time, _side, order in commands if order is not None),
            )
            censored_count = len(censored_orders)
            censored_pending = len(commands)
            censored_zero_fill = sum(not order.had_fill for order in censored_orders)
            censored_quantity = float(sum(order.remaining for order in censored_orders))
            end_timestamp = max(
                economics.last_timestamp, last_event_time_ns, observation_start or 0
            )
        elif decision_windows_ns is not None:
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
            stop = risk_coverage_stop()
            if stop is not None:
                apply_funding(stop - 1)
                raise _AccountPathStop("portfolio_pending_execution_unknown", stop)
            for order in active.values():
                finish_order(order, order.valid_until_ns, cancelled=True)
            active.clear()
        pending.advance(
            observation_stop - 1 if truncated else end_timestamp,
            latest_state,
            observed_coverage,
        )
        pending.expire_all(reason="observation_truncated" if truncated else None)
        terminal_mid = observed_mark(end_timestamp)
        marked = (
            None
            if terminal_mid is None and economics.quantity != 0
            else economics.mark(end_timestamp, terminal_mid)
        )
        last_gross = None if marked is None else marked["gross_pnl"]
        last_net = None if marked is None else marked["net_pnl"]
    if stopped:
        truncated = True
        end_timestamp, terminal_mid = economics.last_timestamp, economics.last_mark
        last_gross = last_net = None
        censored_orders = (
            *active.values(),
            *(order for _, _, order in commands if order is not None),
        )
        censored_count = len(censored_orders)
        censored_pending = len(commands)
        censored_zero_fill = sum(not order.had_fill for order in censored_orders)
        censored_quantity = float(sum(order.remaining for order in censored_orders))
    result = {
        "market_observation_status": "PARTIAL"
        if first_market_gap_ns is not None
        else "COMPLETE",
        "market_observation_reason": "crypto_execution_book_coverage_incomplete"
        if first_market_gap_ns is not None
        else None,
        "first_market_gap_ns": first_market_gap_ns,
        "account_path_status": "PARTIAL" if stopped or last_net is None else "COMPLETE",
        "account_stop_reason": stopped[0].reason
        if stopped
        else "portfolio_terminal_valuation_missing"
        if last_net is None
        else None,
        "account_stop_ns": stopped[0].timestamp_ns
        if stopped
        else end_timestamp
        if last_net is None
        else None,
        "simulation": "event_ordered_public_l2_maker",
        "created_orders": created_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "zero_fill_orders": created_orders - filled_orders - censored_zero_fill,
        "censored_active_order_count": censored_count,
        "censored_pending_command_count": censored_pending,
        "censored_order_quantity": censored_quantity,
        "observation_termination": "TRUNCATED" if truncated else "PLANNED_END",
        "partial_fill_events": partial_fill_events,
        "trade_through_fill_events": trade_through_fill_events,
        "post_only_rejections": post_only_rejections,
        "equity_rows": equity_rows,
        "no_eligible_decisions": equity_rows == 0,
        "gross_pnl": last_gross,
        "net_pnl": last_net,
        "max_abs_position": float(max_abs_position),
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
        **(
            {
                "inventory_target": config.inventory_target,
                "inventory_skew": config.inventory_skew,
            }
            if calibrations is None
            else {}
        ),
        "latency_ms": config.latency_ms,
        "quote_horizon_ms": MAKER_QUOTE_HORIZON_MS,
        "markout_horizon_ms": config.markout_horizon_ms,
        "queue_model": "estimated_queue_ahead",
        "queue_assumption": (
            "public MBP aggregate depth; cancellations do not advance queue; "
            "positive visible trade-through fills all remaining quantity "
            "at the resting price; "
            "same-price fills use visible quantity after queue depletion; "
            "zero visible quantity neither fills nor advances queue; "
            "unknown visible quantity blocks; historical replay assumes small orders "
            "with no market impact; order and position limits "
            "do not validate this assumption"
        ),
        "same_timestamp_policy": "local_observation_order_then_delayed_commands",
        "maker_policy_id": "paired_validation_calibrated_post_only_maker"
        if calibrations is not None
        else "public_l2_inventory_maker_baseline",
        "maker_prediction_used": calibrations is not None,
        "unknown_queue_decisions": unknown_queue_decisions,
        "selection_status": "PARTIAL"
        if unknown_queue_decisions
        else "BLOCKED"
        if blocked_decisions and blocked_decisions == funding_required_decisions
        else "PARTIAL"
        if blocked_decisions
        else "UNDEFINED"
        if undefined_decisions
        else "COMPLETE",
        "blocked_decisions": blocked_decisions,
        "funding_gap_policy": "causal_no_new_risk_partial_forecast_v1",
        "funding_required_decisions": funding_required_decisions,
        "funding_covered_decisions": funding_required_decisions - blocked_decisions,
        "funding_decision_coverage": (
            (funding_required_decisions - blocked_decisions)
            / funding_required_decisions
            if funding_required_decisions
            else None
        ),
        "undefined_decisions": undefined_decisions,
        "assumption": "historical_book_zero_market_impact",
        **metrics.payload(),
        "fees": -economics.cost_cashflows,
        "funding": None if last_net is None else economics.funding_cashflows,
        "initial_capital": capital.initial_capital,
        "valuation_currency": capital.valuation_currency,
        "candidate_id": capital.candidate_id,
        "nav": None if last_net is None else capital.initial_capital + last_net,
        "capital_return": None
        if last_net is None
        else last_net / capital.initial_capital,
    }
    return _record_execution_result(economics, result, end_timestamp, terminal_mid)
