"""Crypto processed source-input id selection."""

from __future__ import annotations

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.sources.binance import ids as binance_ids

MARKET_BARS_SOURCE_DATA_ID = binance_ids.market_bars_data_id()
AGGREGATE_TRADES_SOURCE_DATA_ID = binance_ids.aggregate_trades_data_id()
DEPTH_LEVELS_SOURCE_DATA_ID = binance_ids.depth_levels_data_id()


def market_bars_source_data_id(spec: ProcessedDataSpec) -> str:
    return _exactly_one(
        spec,
        expected=MARKET_BARS_SOURCE_DATA_ID,
        error="panel_market_bars_source_input_not_unique",
    )


def aggregate_trades_source_data_id(spec: ProcessedDataSpec) -> str:
    return _exactly_one(
        spec,
        expected=AGGREGATE_TRADES_SOURCE_DATA_ID,
        error="panel_aggregate_trades_source_input_not_unique",
    )


def depth_levels_source_data_id(spec: ProcessedDataSpec) -> str:
    return _exactly_one(
        spec,
        expected=DEPTH_LEVELS_SOURCE_DATA_ID,
        error="panel_depth_levels_source_input_not_unique",
    )


def _exactly_one(
    spec: ProcessedDataSpec,
    *,
    expected: str,
    error: str,
) -> str:
    _require_exact_source_input_set(spec)
    matches = tuple(
        data_id for data_id in spec.source_input_data_ids if data_id == expected
    )
    if len(matches) != 1:
        raise ValueError(error)
    return matches[0]


def _require_exact_source_input_set(spec: ProcessedDataSpec) -> None:
    values = tuple(spec.source_input_data_ids)
    expected = (
        MARKET_BARS_SOURCE_DATA_ID,
        AGGREGATE_TRADES_SOURCE_DATA_ID,
        DEPTH_LEVELS_SOURCE_DATA_ID,
    )
    if len(values) != len(expected) or set(values) != set(expected):
        raise ValueError("panel_source_input_data_ids_mismatch")


__all__ = [
    "AGGREGATE_TRADES_SOURCE_DATA_ID",
    "DEPTH_LEVELS_SOURCE_DATA_ID",
    "MARKET_BARS_SOURCE_DATA_ID",
    "aggregate_trades_source_data_id",
    "depth_levels_source_data_id",
    "market_bars_source_data_id",
]
