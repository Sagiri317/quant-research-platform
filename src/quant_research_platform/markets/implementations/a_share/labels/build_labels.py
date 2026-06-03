"""Build split A-share forward-return label processed data from panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_label_catalog,
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.parquet import (
    parquet_dataset_columns,
    write_processed_data,
)

from .forward_returns import add_forward_return_labels

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "session_date",
    "available_at",
    "stock_code",
    "trade_date",
    "eligible",
    "can_enter",
    "can_exit",
)


def calculate_panel_labels(
    panel: pd.DataFrame,
    *,
    horizons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    label_horizons = horizons or tuple(
        definition.horizon for definition in a_share_label_catalog().list()
    )
    return add_forward_return_labels(
        panel,
        horizons=label_horizons,
        asset_column="stock_code",
        date_column="trade_date",
    )


def select_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    spec = a_share_processed_catalog().get("forward_returns")
    missing = [
        column for column in spec.required_columns if column not in frame.columns
    ]
    if missing:
        raise ValueError("missing_label_output_columns: " + ", ".join(missing))
    return frame[list(spec.required_columns)].copy()


def write_forward_labels(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    labels = select_label_columns(frame)
    if "trade_date" in labels.columns:
        labels["trade_date"] = pd.to_datetime(labels["trade_date"])
        mask = labels["trade_date"].dt.year.between(start_year, end_year)
        labels = labels.loc[mask].copy()
    labels = labels.sort_values(
        [column for column in ("stock_code", "trade_date") if column in labels.columns]
    ).reset_index(drop=True)
    catalog = a_share_processed_catalog()
    data_root = write_processed_data(
        labels,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id="forward_returns",
    )
    return data_root, len(labels)


def select_existing_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    wanted_unique = list(dict.fromkeys(wanted))
    columns = parquet_dataset_columns(path)
    return [column for column in wanted_unique if column in columns]


__all__ = [
    "LABEL_BASE_COLUMNS",
    "calculate_panel_labels",
    "select_existing_columns",
    "select_label_columns",
    "write_forward_labels",
]
