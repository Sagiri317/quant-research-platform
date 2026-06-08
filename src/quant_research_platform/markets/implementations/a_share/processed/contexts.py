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
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    DatasetAccessMode,
    parquet_dataset_columns,
    read_dataset,
    write_dataset_artifact,
)
from quant_research_platform.storage.layout import processed_data_parquet_root
from quant_research_platform.storage.parquet import write_processed_data

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
    data_root = write_processed_data(
        context,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id="daily_context",
    )
    return data_root, len(context)


def write_contexts_from_panel_dataset(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[Path, int]:
    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_context")
    panel_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    available = set(parquet_dataset_columns(panel_root))
    columns = [column for column in CONTEXT_COLUMNS if column in available]
    row_count = 0

    def _chunks():
        nonlocal row_count
        frames = read_dataset(
            panel_root,
            years=tuple(range(start_year, end_year + 1)),
            columns=columns,
            access_mode=DatasetAccessMode.STREAMING,
        )
        for frame in frames:
            context = _filter_years(select_context_columns(frame), start_year, end_year)
            context = context[list(spec.required_columns)].copy()
            if context.empty:
                continue
            row_count += len(context)
            yield context

    data_root = write_dataset_artifact(
        _chunks(),
        Path(processed_root) / "contexts" / "daily_context",
        catalog=catalog,
        data_id="daily_context",
        access_mode=DatasetAccessMode.STREAMING,
        resource_policy=resource_policy,
    )
    return data_root, row_count


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
