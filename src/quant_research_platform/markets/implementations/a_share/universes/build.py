"""Build A-share universe processed data from panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.parquet import write_processed_data

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "session_date",
    "available_at",
    "stock_code",
    "trade_date",
    "eligible",
)


def select_universe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in UNIVERSE_COLUMNS if column in frame.columns]
    return frame[columns].copy()


def write_universe(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    universe = _filter_years(select_universe_columns(frame), start_year, end_year)
    catalog = a_share_processed_catalog()
    spec = catalog.get("a_share_default")
    universe = universe[list(spec.required_columns)].copy()
    data_root = write_processed_data(
        universe,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id="a_share_default",
    )
    return data_root, len(universe)


def _filter_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    result = frame.copy()
    if "trade_date" in result.columns:
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.loc[result["trade_date"].dt.year.between(start_year, end_year)]
    sort_columns = [
        column for column in ("stock_code", "trade_date") if column in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


__all__ = ["UNIVERSE_COLUMNS", "select_universe_columns", "write_universe"]
