"""Binance source ids."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.ids import SourceId

SOURCE_ID = SourceId("binance")


def market_bars_data_id(frequency: DataFrequency | str = DataFrequency.MINUTE) -> str:
    if DataFrequency(frequency) is not DataFrequency.MINUTE:
        raise ValueError("unsupported_binance_market_bars_frequency")
    return "market_bars_1m"


def aggregate_trades_data_id() -> str:
    return "aggregate_trades_tick"


def depth_levels_data_id() -> str:
    return "depth_levels_tick"
