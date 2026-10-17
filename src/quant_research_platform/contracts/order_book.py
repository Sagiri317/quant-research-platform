"""Normalized order-book contracts and exact tick-price conversion."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import isfinite


@dataclass(frozen=True)
class BookState:
    source_sequence: int
    event_time_ms: int
    transaction_time_ms: int | None
    observed_time_ns: int
    receive_time: str
    receive_monotonic_ns: int | None
    update_id: int
    segment_id: str
    observation_id: str
    market_type: str
    price_unit: str
    bids: tuple[tuple[int, float], ...]
    asks: tuple[tuple[int, float], ...]

    # Bounds certify absence as zero only inside the emitted side's known range.
    # Producers that omit them certify no more than their visible levels.
    known_bid_floor_tick: int | None = None
    known_ask_ceiling_tick: int | None = None

    @property
    def observed_time_ms(self) -> int:
        return self.observed_time_ns // 1_000_000

    @property
    def best_bid(self) -> tuple[int, float] | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> tuple[int, float] | None:
        return self.asks[0] if self.asks else None


@dataclass(frozen=True)
class BookBoundary:
    source_sequence: int
    event_time_ms: int
    transaction_time_ms: int | None
    observed_time_ns: int
    receive_time: str
    receive_monotonic_ns: int | None
    segment_id: str
    observation_id: str
    market_type: str
    reason: str

    @property
    def observed_time_ms(self) -> int:
        return self.observed_time_ns // 1_000_000


@dataclass(frozen=True)
class TradeEvent:
    source_sequence: int
    event_time_ms: int
    transaction_time_ms: int
    observed_time_ns: int
    receive_time: str
    receive_monotonic_ns: int | None
    observation_id: str
    market_type: str
    price_unit: str
    price_index: int
    quantity: float
    visible_quantity: float | None
    buyer_is_maker: bool

    @property
    def observed_time_ms(self) -> int:
        return self.observed_time_ns // 1_000_000


@dataclass(frozen=True)
class TradeBoundary:
    """A located invalid trade observation; it cannot supply executable liquidity."""

    source_sequence: int
    event_time_ms: int
    transaction_time_ms: int
    observed_time_ns: int
    receive_time: str
    receive_monotonic_ns: int | None
    observation_id: str
    market_type: str
    reason: str

    @property
    def observed_time_ms(self) -> int:
        return self.observed_time_ns // 1_000_000


def validate_price_unit(value: object) -> str:
    _positive_decimal(value, error="order_book_price_unit_invalid")
    return value


def price_to_index(value: object, price_unit: str, *, error: str) -> int:
    unit = _positive_decimal(price_unit, error="order_book_price_unit_invalid")
    return _price_to_index(value, unit, error=error)


def _price_to_index(value: object, unit: Decimal | Fraction, *, error: str) -> int:
    price = _positive_decimal(value, error=error)
    index = Fraction(price) / Fraction(unit)
    if index.denominator != 1:
        raise ValueError(error)
    return index.numerator


def index_to_price(price_index: int, price_unit: str) -> float:
    unit = _positive_decimal(price_unit, error="order_book_price_unit_invalid")
    if type(price_index) is not int:
        raise ValueError("order_book_price_index_invalid")
    try:
        price = float(price_index * Fraction(unit))
    except OverflowError:
        raise ValueError("order_book_price_index_invalid") from None
    if not isfinite(price):
        raise ValueError("order_book_price_index_invalid")
    return price


def price_decimal_places(value: object) -> int:
    """Return exact required decimal places without Decimal context rounding."""
    decimal = _positive_decimal(value, error="order_book_price_invalid")
    parts = decimal.as_tuple()
    trailing = 0
    for digit in reversed(parts.digits):
        if digit:
            break
        trailing += 1
    return max(0, -parts.exponent - trailing)


def _positive_decimal(value: object, *, error: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ValueError(error)
    try:
        decimal = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if (
        not decimal.is_finite()
        or decimal <= 0
        or not 0.0 < float(decimal) < float("inf")
    ):
        raise ValueError(error)
    return decimal


def observation_order_key(
    event: BookState | BookBoundary | TradeEvent | TradeBoundary,
) -> tuple[int, ...]:
    """Use only the selected source's real clocks and stable native order."""
    if event.receive_monotonic_ns is None:
        return event.observed_time_ns, event.source_sequence
    return event.observed_time_ns, event.receive_monotonic_ns, event.source_sequence
