"""Build A-share context processed data from panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)

CONTEXT_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "market_type",
    "is_st",
    "is_pt",
    "limit_status",
    "listing_age_trading_days",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "total_market_cap",
    "float_market_cap",
    "amount",
    "turnover",
)


def select_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in CONTEXT_COLUMNS if column in frame.columns]
    return frame[columns].copy()


def write_contexts(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    context = _filter_years(select_context_columns(frame), start_year, end_year)
    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_context")
    context = context[list(spec.required_columns)].copy()
    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(artifact_root, context, spec=spec)
    return artifact_root, len(context)


def write_contexts_from_panel_dataset(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_context")
    panel_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        "panels",
        "daily_panel",
    )
    available = set(inspect_dataset(panel_root).columns)
    columns = [column for column in CONTEXT_COLUMNS if column in available]
    row_count = 0

    def _chunks():
        nonlocal row_count
        frames = read_dataset(
            panel_root,
            partition_filter={"year": tuple(range(start_year, end_year + 1))},
            columns=columns,
            mode=DatasetReadMode.CHUNKED,
        )
        for frame in frames:
            context = _filter_years(select_context_columns(frame), start_year, end_year)
            context = context[list(spec.required_columns)].copy()
            if context.empty:
                continue
            row_count += len(context)
            yield context

    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(
        artifact_root,
        _chunks(),
        spec=spec,
    )
    return artifact_root, row_count


def _filter_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.loc[result["timestamp"].dt.year.between(start_year, end_year)]
    sort_columns = ["market_id", "asset_id", "timestamp"]
    return result.sort_values(sort_columns).reset_index(drop=True)


__all__ = [
    "CONTEXT_COLUMNS",
    "select_context_columns",
    "write_contexts",
    "write_contexts_from_panel_dataset",
]
