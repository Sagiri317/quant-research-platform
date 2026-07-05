"""Binance capture plan contract.

This module defines local capture-plan validation only. It does not call any
Binance plugin, network endpoint, or credentialed API.
"""

from __future__ import annotations

from dataclasses import dataclass

BINANCE_SPOT_MARKET_TYPE = "spot"
BINANCE_DATA_KINDS = frozenset({"klines", "agg_trades", "depth"})


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
        if self.interval != "1m":
            raise ValueError("unsupported_binance_interval")
        if self.limit <= 0 or self.depth_limit <= 0 or self.chunk_size <= 0:
            raise ValueError("binance_capture_positive_limits_required")
        if not self.start_time or not self.end_time:
            raise ValueError("binance_capture_time_window_required")


__all__ = ["BINANCE_DATA_KINDS", "BINANCE_SPOT_MARKET_TYPE", "BinanceCapturePlan"]
