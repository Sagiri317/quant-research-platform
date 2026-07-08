"""Binance capture plan contract.

This module defines local capture-plan validation only. It does not call any
Binance plugin, network endpoint, or credentialed API.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.sources.binance.validation import (
    BINANCE_DATA_KINDS,
    BINANCE_SPOT_MARKET_TYPE,
    strict_capture_id,
    strict_data_kind,
    strict_iso_timestamp,
    strict_market_type,
    strict_optional_interval_by_data_kind,
    strict_positive_integer,
    strict_symbol_tuple,
)

MAX_LIMIT = 1_000
MAX_DEPTH_LIMIT = 5_000
MAX_CHUNK_SIZE = 100_000


@dataclass(frozen=True, kw_only=True)
class BinanceCapturePlan:
    capture_id: str
    market_type: str
    data_kind: str
    symbols: tuple[str, ...]
    interval: str | None
    start_time: str
    end_time: str
    limit: int
    depth_limit: int
    chunk_size: int
    time_zone: str = "UTC"

    def __post_init__(self) -> None:
        market_type = strict_market_type(self.market_type)
        data_kind = strict_data_kind(self.data_kind)
        if self.time_zone != "UTC":
            raise ValueError("unsupported_binance_time_zone")
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "data_kind", data_kind)
        object.__setattr__(self, "capture_id", strict_capture_id(self.capture_id))
        object.__setattr__(self, "symbols", strict_symbol_tuple(self.symbols))
        object.__setattr__(
            self,
            "interval",
            strict_optional_interval_by_data_kind(
                data_kind=data_kind,
                interval=self.interval,
            ),
        )
        limit = strict_positive_integer(
            self.limit,
            error="binance_capture_positive_limits_required",
        )
        depth_limit = strict_positive_integer(
            self.depth_limit,
            error="binance_capture_positive_limits_required",
        )
        chunk_size = strict_positive_integer(
            self.chunk_size,
            error="binance_capture_positive_limits_required",
        )
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "depth_limit", depth_limit)
        object.__setattr__(self, "chunk_size", chunk_size)
        if (
            limit > MAX_LIMIT
            or depth_limit > MAX_DEPTH_LIMIT
            or chunk_size > MAX_CHUNK_SIZE
        ):
            raise ValueError("binance_capture_limit_too_large")
        if not self.start_time or not self.end_time:
            raise ValueError("binance_capture_time_window_required")
        start = strict_iso_timestamp(
            self.start_time,
            error="invalid_binance_start_time",
        )
        end = strict_iso_timestamp(self.end_time, error="invalid_binance_end_time")
        if start >= end:
            raise ValueError("invalid_binance_capture_time_order")


__all__ = ["BINANCE_DATA_KINDS", "BINANCE_SPOT_MARKET_TYPE", "BinanceCapturePlan"]
