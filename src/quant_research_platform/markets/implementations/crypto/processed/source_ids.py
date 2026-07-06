"""Crypto processed source-input id selection."""

from __future__ import annotations

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.sources.binance import ids as binance_ids


def market_bars_source_data_id(spec: ProcessedDataSpec) -> str:
    return _exactly_one(
        spec,
        expected=binance_ids.market_bars_data_id(),
        error="panel_market_bars_source_input_not_unique",
    )


def aggregate_trades_source_data_id(spec: ProcessedDataSpec) -> str:
    return _exactly_one(
        spec,
        expected=binance_ids.aggregate_trades_data_id(),
        error="panel_aggregate_trades_source_input_not_unique",
    )


def depth_levels_source_data_id(spec: ProcessedDataSpec) -> str:
    return _exactly_one(
        spec,
        expected=binance_ids.depth_levels_data_id(),
        error="panel_depth_levels_source_input_not_unique",
    )


def _exactly_one(
    spec: ProcessedDataSpec,
    *,
    expected: str,
    error: str,
) -> str:
    matches = tuple(
        data_id for data_id in spec.source_input_data_ids if data_id == expected
    )
    if len(matches) != 1:
        raise ValueError(error)
    return matches[0]


__all__ = [
    "aggregate_trades_source_data_id",
    "depth_levels_source_data_id",
    "market_bars_source_data_id",
]
