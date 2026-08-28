"""Normalized order-book contracts and exact tick-price conversion."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class BookState:
    capture_seq: int
    event_time_ms: int
    transaction_time_ms: int | None
    observed_time_ms: int
    receive_time: str
    receive_monotonic_ns: int
    update_id: int
    segment_id: str
    connection_id: str
    market_type: str
    tick_size: str
    bids: tuple[tuple[int, float], ...]
    asks: tuple[tuple[int, float], ...]

    @property
    def best_bid(self) -> tuple[int, float] | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> tuple[int, float] | None:
        return self.asks[0] if self.asks else None


@dataclass(frozen=True)
class BookBoundary:
    capture_seq: int
    event_time_ms: int
    transaction_time_ms: int | None
    observed_time_ms: int
    receive_time: str
    receive_monotonic_ns: int
    segment_id: str
    connection_id: str
    market_type: str
    reason: str



@dataclass(frozen=True)
class TradeEvent:
    capture_seq: int
    event_time_ms: int
    transaction_time_ms: int
    observed_time_ms: int
    receive_time: str
    receive_monotonic_ns: int
    connection_id: str
    market_type: str
    tick_size: str
    price_tick: int
    quantity: float
    visible_quantity: float
    buyer_is_maker: bool


def validate_tick_size(value: object) -> str:
    text, _decimal = _tick_size(value)
    return text


def price_to_tick(value: object, tick_size: str, *, error: str) -> int:
    _, tick = _tick_size(tick_size)
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise ValueError(error)
    try:
        price = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(error) from exc
    if not price.is_finite() or price <= 0:
        raise ValueError(error)
    ticks = price / tick
    integral = ticks.to_integral_value()
    if ticks != integral:
        raise ValueError(error)
    return int(integral)


def tick_to_price(price_tick: int, tick_size: str) -> float:
    _, tick = _tick_size(tick_size)
    if type(price_tick) is not int:
        raise ValueError("order_book_price_tick_invalid")
    return float(Decimal(price_tick) * tick)


def _tick_size(value: object) -> tuple[str, Decimal]:
    if not isinstance(value, str) or not value:
        raise ValueError("order_book_tick_size_invalid")
    try:
        decimal = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("order_book_tick_size_invalid") from exc
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError("order_book_tick_size_invalid")
    return value, decimal
