"""CSMAR raw-to-canonical-interim helpers."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.implementations.a_share.adapter import (
    AShareMarketAdapter,
)

CSMAR_SOURCE_ID = "csmar"
INDEX_METADATA_TIME_COLUMNS: tuple[str, ...] = (
    "release_date",
    "base_date",
    "start_date",
)


def normalize_daily_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 0.1 canonical interim keys to a normalized daily market frame."""

    return normalize_asset_time_frame(frame, time_column="trade_date")


def normalize_asset_time_frame(
    frame: pd.DataFrame,
    *,
    time_column: str,
    source_asset_column: str = "stock_code",
) -> pd.DataFrame:
    """Add canonical A-share asset/time fields to stock-level CSMAR interim data."""

    return AShareMarketAdapter().with_daily_bar_canonical_fields(
        frame,
        source_id=CSMAR_SOURCE_ID,
        source_asset_column=source_asset_column,
        session_date_column=time_column,
        asset_namespace="stock",
    )


def normalize_index_time_frame(
    frame: pd.DataFrame,
    *,
    time_column: str = "trade_date",
    source_asset_column: str = "index_code",
) -> pd.DataFrame:
    """Add canonical fields for A-share benchmark/index time-series data."""

    return AShareMarketAdapter().with_daily_bar_canonical_fields(
        frame,
        source_id=CSMAR_SOURCE_ID,
        source_asset_column=source_asset_column,
        session_date_column=time_column,
        asset_namespace="index",
    )


def normalize_index_metadata_frame(
    frame: pd.DataFrame,
    *,
    source_asset_column: str = "index_code",
    time_columns: tuple[str, ...] = INDEX_METADATA_TIME_COLUMNS,
) -> pd.DataFrame:
    """Add canonical fields for A-share index metadata without requiring dates."""

    return AShareMarketAdapter().with_metadata_canonical_fields(
        frame,
        source_id=CSMAR_SOURCE_ID,
        source_asset_column=source_asset_column,
        time_columns=time_columns,
        asset_namespace="index",
    )


__all__ = [
    "CSMAR_SOURCE_ID",
    "INDEX_METADATA_TIME_COLUMNS",
    "normalize_asset_time_frame",
    "normalize_daily_market_frame",
    "normalize_index_metadata_frame",
    "normalize_index_time_frame",
]
