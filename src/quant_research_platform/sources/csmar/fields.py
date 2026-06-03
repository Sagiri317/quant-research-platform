"""CSMAR source/interim field catalog helpers."""

from __future__ import annotations

from quant_research_platform.contracts.columns import ColumnRole

SOURCE_IDENTIFIER_FIELDS = {
    "stock_code",
    "index_code",
    "trade_date",
    "share_change_date",
    "asset_record_date",
    "release_date",
    "base_date",
    "start_date",
}


def csmar_field_role(column: str) -> ColumnRole:
    if column in SOURCE_IDENTIFIER_FIELDS:
        return ColumnRole.SOURCE
    return ColumnRole.CONTEXT


__all__ = ["SOURCE_IDENTIFIER_FIELDS", "csmar_field_role"]
