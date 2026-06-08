"""Build A-share mask processed data from panels."""

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
    read_dataset,
    write_dataset_artifact,
)
from quant_research_platform.storage.layout import processed_data_parquet_root
from quant_research_platform.storage.parquet import write_processed_data

MASK_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "as_of": ("a_share_as_of", ("eligible",)),
    "execution": ("a_share_execution", ("can_enter", "can_exit")),
    "evaluation": ("a_share_evaluation", ("eligible", "can_exit")),
}
MASK_KEY_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)


def write_masks(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> dict[str, tuple[Path, int]]:
    outputs: dict[str, tuple[Path, int]] = {}
    for mask_kind, (processed_data_id, mask_columns) in MASK_SPECS.items():
        mask_frame = _filter_years(
            _select_mask_columns(frame, mask_columns),
            start_year,
            end_year,
        )
        catalog = a_share_processed_catalog()
        spec = catalog.get(processed_data_id)
        mask_frame = mask_frame[list(spec.required_columns)].copy()
        data_root = write_processed_data(
            mask_frame,
            layout=layout_from_processed_root(processed_root),
            catalog=catalog,
            processed_data_id=processed_data_id,
        )
        outputs[mask_kind] = (data_root, len(mask_frame))
    return outputs


def write_masks_from_panel_dataset(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, tuple[Path, int]]:
    outputs: dict[str, tuple[Path, int]] = {}
    for mask_kind, (processed_data_id, mask_columns) in MASK_SPECS.items():
        outputs[mask_kind] = _write_mask_from_panel_dataset(
            processed_root,
            processed_data_id=processed_data_id,
            mask_columns=mask_columns,
            start_year=start_year,
            end_year=end_year,
            resource_policy=resource_policy,
        )
    return outputs


def _write_mask_from_panel_dataset(
    processed_root: Path,
    *,
    processed_data_id: str,
    mask_columns: tuple[str, ...],
    start_year: int,
    end_year: int,
    resource_policy: RuntimeResourcePolicy | None,
) -> tuple[Path, int]:
    catalog = a_share_processed_catalog()
    spec = catalog.get(processed_data_id)
    panel_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    read_columns = list(dict.fromkeys((*MASK_KEY_COLUMNS, *mask_columns)))
    row_count = 0

    def _chunks():
        nonlocal row_count
        frames = read_dataset(
            panel_root,
            years=tuple(range(start_year, end_year + 1)),
            columns=read_columns,
            access_mode=DatasetAccessMode.STREAMING,
        )
        for frame in frames:
            mask_frame = _filter_years(
                _select_mask_columns(frame, mask_columns),
                start_year,
                end_year,
            )
            mask_frame = mask_frame[list(spec.required_columns)].copy()
            if mask_frame.empty:
                continue
            row_count += len(mask_frame)
            yield mask_frame

    data_root = write_dataset_artifact(
        _chunks(),
        Path(processed_root) / spec.category / processed_data_id,
        catalog=catalog,
        data_id=processed_data_id,
        access_mode=DatasetAccessMode.STREAMING,
        resource_policy=resource_policy,
    )
    return data_root, row_count


def _select_mask_columns(
    frame: pd.DataFrame,
    mask_columns: tuple[str, ...],
) -> pd.DataFrame:
    columns = [
        column
        for column in (*MASK_KEY_COLUMNS, *mask_columns)
        if column in frame.columns
    ]
    return frame[columns].copy()


def _filter_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.loc[result["timestamp"].dt.year.between(start_year, end_year)]
    sort_columns = ["market_id", "asset_id", "timestamp"]
    return result.sort_values(sort_columns).reset_index(drop=True)


__all__ = [
    "MASK_KEY_COLUMNS",
    "MASK_SPECS",
    "write_masks",
    "write_masks_from_panel_dataset",
]
