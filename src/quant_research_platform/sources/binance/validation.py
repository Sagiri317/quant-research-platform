"""Strict Binance source-local validation helpers."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
from pandas.errors import OutOfBoundsDatetime

BINANCE_SPOT_MARKET_TYPE = "spot"
BINANCE_DATA_KINDS = frozenset({"klines", "agg_trades", "depth"})
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+$")
_NANOSECOND_TIMESTAMP_MAX_MILLISECONDS = (
    pd.Timestamp.max.value // pd.Timedelta(milliseconds=1).value
)


def strict_non_empty_string(value: object, *, error: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(error)
    return value


def strict_symbol(value: object, *, error: str = "invalid_binance_symbol") -> str:
    symbol = strict_non_empty_string(value, error=error)
    if _SYMBOL_PATTERN.fullmatch(symbol) is None:
        raise ValueError(error)
    return symbol


def strict_data_kind(
    value: object,
    *,
    error: str = "unsupported_binance_data_kind",
) -> str:
    data_kind = strict_non_empty_string(value, error=error)
    if data_kind not in BINANCE_DATA_KINDS:
        raise ValueError(error)
    return data_kind


def strict_market_type(
    value: object,
    *,
    error: str = "unsupported_binance_market_type",
) -> str:
    market_type = strict_non_empty_string(value, error=error)
    if market_type != BINANCE_SPOT_MARKET_TYPE:
        raise ValueError(error)
    return market_type


def strict_finite_float(value: object, *, error: str) -> float:
    if isinstance(value, bool):
        raise ValueError(error)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not math.isfinite(number):
        raise ValueError(error)
    return number


def strict_positive_float(value: object, *, error: str) -> float:
    number = strict_finite_float(value, error=error)
    if number <= 0:
        raise ValueError(error)
    return number


def strict_non_negative_float(value: object, *, error: str) -> float:
    number = strict_finite_float(value, error=error)
    if number < 0:
        raise ValueError(error)
    return number


def strict_non_negative_integer(value: object, *, error: str) -> int:
    number = _strict_integer(value, error=error)
    if number < 0:
        raise ValueError(error)
    return number


def strict_positive_integer(value: object, *, error: str) -> int:
    number = _strict_integer(value, error=error)
    if number <= 0:
        raise ValueError(error)
    return number


def _strict_integer(value: object, *, error: str) -> int:
    if isinstance(value, bool):
        raise ValueError(error)
    if isinstance(value, int):
        return value
    raise ValueError(error)


def strict_boolean(value: object, *, error: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(error)
    return value


def strict_millisecond_timestamp(value: object, *, error: str) -> pd.Timestamp:
    milliseconds = strict_non_negative_integer(value, error=error)
    if milliseconds > _NANOSECOND_TIMESTAMP_MAX_MILLISECONDS:
        raise ValueError(error)
    try:
        return pd.Timestamp(milliseconds, unit="ms", tz="UTC").as_unit("ns")
    except (OutOfBoundsDatetime, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc


def strict_iso_timestamp(value: object, *, error: str) -> pd.Timestamp:
    if not isinstance(value, str) or not _has_utc_offset(value):
        raise ValueError(error)
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(error)
    return pd.Timestamp(timestamp)


def first_present(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError("missing_binance_raw_payload_field")


def depth_side_levels(
    raw_levels: object,
    *,
    side: str,
    error: str,
) -> list[tuple[float, float]]:
    if raw_levels is None:
        return []
    if side not in {"bid", "ask"}:
        raise ValueError(error)
    if not isinstance(raw_levels, list | tuple):
        raise ValueError(error)
    levels: list[tuple[float, float]] = []
    previous_price: float | None = None
    for raw_level in raw_levels:
        if not isinstance(raw_level, list | tuple) or len(raw_level) != 2:
            raise ValueError(error)
        price = strict_positive_float(raw_level[0], error=error)
        quantity = strict_non_negative_float(raw_level[1], error=error)
        if previous_price is not None:
            if side == "bid" and price >= previous_price:
                raise ValueError(error)
            if side == "ask" and price <= previous_price:
                raise ValueError(error)
        previous_price = price
        levels.append((price, quantity))
    return levels


def _has_utc_offset(value: str) -> bool:
    if value.endswith("Z"):
        return True
    if len(value) < 6:
        return False
    offset = value[-6:]
    return (
        offset[0] in {"+", "-"}
        and offset[1:3].isdigit()
        and offset[3] == ":"
        and offset[4:6].isdigit()
    )
