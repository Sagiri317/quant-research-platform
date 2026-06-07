"""Build split A-share forward-return label processed data from panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_label_catalog,
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    DAILY_PANEL_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.storage.parquet import (
    parquet_dataset_columns,
    read_partitioned_parquet_dataset,
    write_processed_data,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)
LABEL_PANEL_INPUT_COLUMNS: tuple[str, ...] = (*LABEL_BASE_COLUMNS, "adj_open")


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


def calculate_forward_returns_full(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    panel_frame: pd.DataFrame | None = None,
) -> tuple[Path, int]:
    panel = _panel_frame_for_label_build(
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
        panel_frame=panel_frame,
    )
    labels = calculate_panel_labels(panel)
    return write_forward_labels(
        labels,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )


def select_existing_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    wanted_unique = list(dict.fromkeys(wanted))
    columns = parquet_dataset_columns(path)
    return [column for column in wanted_unique if column in columns]


def _panel_frame_for_label_build(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    panel_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if panel_frame is None:
        data_root = (
            Path(processed_root)
            / "panels"
            / DAILY_PANEL_PROCESSED_DATA_ID
            / "data"
        )
        available_columns = set(parquet_dataset_columns(data_root))
        missing = [
            column
            for column in LABEL_PANEL_INPUT_COLUMNS
            if column not in available_columns
        ]
        if missing:
            raise ValueError(
                "missing_label_panel_input_columns: " + ", ".join(missing)
            )
        years = _available_panel_years(
            data_root=data_root,
            start_year=start_year,
            end_year=end_year + 1,
        )
        return read_partitioned_parquet_dataset(
            data_root,
            years=years,
            columns=list(LABEL_PANEL_INPUT_COLUMNS),
        )
    missing = [
        column
        for column in LABEL_PANEL_INPUT_COLUMNS
        if column not in panel_frame.columns
    ]
    if missing:
        raise ValueError("missing_label_panel_input_columns: " + ", ".join(missing))
    panel = panel_frame.loc[:, list(LABEL_PANEL_INPUT_COLUMNS)].copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    return panel.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )


def _available_panel_years(
    *,
    data_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[int, ...]:
    years = tuple(
        year
        for year in range(start_year, end_year + 1)
        if list((data_root / f"year={year}").glob("*.parquet"))
    )
    if not years:
        raise ValueError("missing_requested_processed_input")
    return years


__all__ = [
    "LABEL_BASE_COLUMNS",
    "LABEL_PANEL_INPUT_COLUMNS",
    "calculate_panel_labels",
    "calculate_forward_returns_full",
    "select_existing_columns",
    "select_label_columns",
    "write_forward_labels",
]
