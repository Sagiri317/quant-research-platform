"""Binance capture plan contract.

This module defines local capture-plan validation only. It does not call any
Binance plugin, network endpoint, or credentialed API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

BINANCE_SPOT_MARKET_TYPE = "spot"
BINANCE_DATA_KINDS = frozenset({"klines", "agg_trades", "depth"})
MAX_LIMIT = 1_000
MAX_DEPTH_LIMIT = 5_000
MAX_CHUNK_SIZE = 100_000
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+$")


@dataclass(frozen=True, kw_only=True)
class BinanceCapturePlan:
    capture_id: str
    market_type: str
    data_kind: str
    symbols: tuple[str, ...]
    interval: str
    start_time: str
    end_time: str
    limit: int
    depth_limit: int
    chunk_size: int
    time_zone: str = "UTC"

    def __post_init__(self) -> None:
        if self.market_type != BINANCE_SPOT_MARKET_TYPE:
            raise ValueError("unsupported_binance_market_type")
        if self.data_kind not in BINANCE_DATA_KINDS:
            raise ValueError("unsupported_binance_data_kind")
        if self.time_zone != "UTC":
            raise ValueError("unsupported_binance_time_zone")
        if not self.capture_id:
            raise ValueError("binance_capture_id_required")
        if not self.symbols or any(not symbol for symbol in self.symbols):
            raise ValueError("binance_symbols_required")
        if any(not _SYMBOL_PATTERN.match(symbol) for symbol in self.symbols):
            raise ValueError("invalid_binance_symbol")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("duplicate_binance_symbols")
        if self.interval != "1m":
            raise ValueError("unsupported_binance_interval")
        if self.limit <= 0 or self.depth_limit <= 0 or self.chunk_size <= 0:
            raise ValueError("binance_capture_positive_limits_required")
        if (
            self.limit > MAX_LIMIT
            or self.depth_limit > MAX_DEPTH_LIMIT
            or self.chunk_size > MAX_CHUNK_SIZE
        ):
            raise ValueError("binance_capture_limit_too_large")
        if not self.start_time or not self.end_time:
            raise ValueError("binance_capture_time_window_required")
        start = _parse_utc_timestamp(
            self.start_time,
            error="invalid_binance_start_time",
        )
        end = _parse_utc_timestamp(self.end_time, error="invalid_binance_end_time")
        if start >= end:
            raise ValueError("invalid_binance_capture_time_order")


def _parse_utc_timestamp(value: str, *, error: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(error)
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(error)
    return timestamp


__all__ = ["BINANCE_DATA_KINDS", "BINANCE_SPOT_MARKET_TYPE", "BinanceCapturePlan"]
