"""Binance source/interim field roles."""

from __future__ import annotations

from quant_research_platform.contracts.columns import ColumnRole


def field_role(column: str, *, primary_key: tuple[str, ...]) -> ColumnRole:
    if column == "timestamp":
        return ColumnRole.TIMESTAMP
    if column in primary_key:
        return ColumnRole.KEY
    return ColumnRole.CONTEXT


__all__ = ["field_role"]
