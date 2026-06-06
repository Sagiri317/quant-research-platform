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
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.storage.parquet import (
    parquet_dataset_columns,
    write_processed_data,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
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
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
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
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    mask = labels["timestamp"].dt.year.between(start_year, end_year)
    labels = labels.loc[mask].copy()
    labels = labels.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )
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
