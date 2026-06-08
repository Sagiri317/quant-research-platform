"""Build split A-share forward-return label processed data from panels."""

from __future__ import annotations

from collections.abc import Iterator
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
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    write_dataset,
)

from ..processed._dataset_helpers import (
    available_dataset_years,
    chunked_values,
    dataset_asset_ids,
    dataset_row_count,
    read_dataset_asset_chunk,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)
LABEL_PANEL_INPUT_COLUMNS: tuple[str, ...] = (*LABEL_BASE_COLUMNS, "adj_open")


def calculate_forward_returns_asset_batches(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    asset_chunk_size: int | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[Path, int]:
    """Calculate forward-return labels in asset batches with future context."""

    catalog = a_share_processed_catalog()
    spec = catalog.get("forward_returns")
    data_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        "panels",
        DAILY_PANEL_PROCESSED_DATA_ID,
    )
    available_columns = set(inspect_dataset(data_root).columns)
    missing = [
        column
        for column in LABEL_PANEL_INPUT_COLUMNS
        if column not in available_columns
    ]
    if missing:
        raise ValueError("missing_label_panel_input_columns: " + ", ".join(missing))
    input_years = available_dataset_years(
        data_root=data_root,
        start_year=start_year,
        end_year=end_year + 1,
    )
    label_horizons = tuple(
        definition.horizon for definition in a_share_label_catalog().list()
    )
    output_columns = list(spec.required_columns)
    asset_ids = dataset_asset_ids(data_root, years=input_years)
    chunk_size = asset_chunk_size or _planned_asset_chunk_size(
        asset_ids=asset_ids,
        data_root=data_root,
        years=input_years,
        resource_policy=resource_policy,
    )
    row_count = 0

    def _output_chunks() -> Iterator[pd.DataFrame]:
        nonlocal row_count
        for asset_chunk in chunked_values(asset_ids, chunk_size):
            panel = read_dataset_asset_chunk(
                data_root,
                years=input_years,
                columns=LABEL_PANEL_INPUT_COLUMNS,
                asset_ids=asset_chunk,
            )
            if panel.empty:
                continue
            labels = add_forward_return_labels(
                panel,
                horizons=label_horizons,
                asset_column="asset_id",
                date_column="timestamp",
                execution_price_column="adj_open",
            )
            missing = [
                column for column in output_columns if column not in labels.columns
            ]
            if missing:
                raise ValueError(
                    "missing_label_output_columns: " + ", ".join(missing)
                )
            output = labels.loc[:, output_columns].copy()
            output["timestamp"] = pd.to_datetime(output["timestamp"])
            mask = output["timestamp"].dt.year.between(start_year, end_year)
            output = output.loc[mask].copy()
            output = output.sort_values(list(spec.primary_key)).reset_index(drop=True)
            if output.empty:
                continue
            row_count += len(output)
            yield output

    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(
        artifact_root,
        _output_chunks(),
        spec=spec,
    )
    return artifact_root, row_count


def _planned_asset_chunk_size(
    *,
    asset_ids: tuple[str, ...],
    data_root: Path,
    years: tuple[int, ...],
    resource_policy: RuntimeResourcePolicy | None,
) -> int:
    total_rows = dataset_row_count(data_root, years=years)
    rows_per_asset = max(1, total_rows // max(1, len(asset_ids)))
    output_width = len(
        a_share_processed_catalog().get("forward_returns").required_columns
    )
    working_width = len(LABEL_PANEL_INPUT_COLUMNS) + (output_width * 2)
    estimated_bytes_per_asset = max(
        1,
        rows_per_asset * working_width * 16,
    )
    label_horizons = tuple(
        definition.horizon for definition in a_share_label_catalog().list()
    )
    plan = ChunkPlanner(resource_policy).plan(
        axis=ChunkAxis.ASSET,
        item_count=len(asset_ids),
        estimated_bytes_per_item=estimated_bytes_per_asset,
        lookahead_window=max(label_horizons) + 1,
        output_width=output_width,
        disjoint=True,
    )
    return plan.chunk_size




__all__ = [
    "LABEL_BASE_COLUMNS",
    "LABEL_PANEL_INPUT_COLUMNS",
    "calculate_forward_returns_asset_batches",
]
