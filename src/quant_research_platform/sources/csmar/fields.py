"""CSMAR source/interim field catalog helpers."""

from __future__ import annotations

from quant_research_platform.contracts.columns import ColumnRole

SOURCE_IDENTIFIER_FIELDS = {
    "source_asset_id",
    "stock_code",
    "index_code",
    "constituent_asset_id",
    "trade_date",
    "share_change_date",
    "asset_record_date",
    "calendar_date",
    "base_date",
    "start_date",
    "source_market_type",
}


def field_role(column: str) -> ColumnRole:
    if column in SOURCE_IDENTIFIER_FIELDS:
        return ColumnRole.SOURCE
    return ColumnRole.CONTEXT
