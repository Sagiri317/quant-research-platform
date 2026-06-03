"""Historical processed-data migration helpers for A-share references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.adapter import (
    AShareMarketAdapter,
)
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.parquet import (
    read_partitioned_parquet_dataset,
    write_processed_data,
)


@dataclass(frozen=True)
class HistoricalMigrationResult:
    row_counts: dict[str, int] = field(default_factory=dict)


def migrate_reference_factor_label_history(
    *,
    reference_processed_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> HistoricalMigrationResult:
    """Migrate verified historical factor/label-era artifacts into QRP 0.1 layout."""

    row_counts: dict[str, int] = {}
    for year in range(start_year, end_year + 1):
        year_frames = _read_reference_year(reference_processed_root, year)
        as_of_membership = year_frames["mask_as_of"].loc[
            :, ["stock_code", "trade_date", "eligible"]
        ]
        migrated = {
            "daily_context": _canonicalize_daily_frame(year_frames["context"]),
            "a_share_default": _canonicalize_daily_frame(as_of_membership),
            "a_share_as_of": _canonicalize_daily_frame(as_of_membership),
            "a_share_execution": _canonicalize_daily_frame(
                year_frames["mask_execution"].loc[
                    :, ["stock_code", "trade_date", "can_enter", "can_exit"]
                ]
            ),
            "a_share_evaluation": _evaluation_mask_frame(year_frames),
            "price_volume_56": _factor_frame(year_frames),
            "forward_returns": _label_frame(year_frames),
        }
        _write_migrated_frames(migrated, processed_root=processed_root)
        for data_id, frame in migrated.items():
            row_counts[data_id] = row_counts.get(data_id, 0) + len(frame)
    return HistoricalMigrationResult(row_counts=row_counts)


def _read_reference_year(
    reference_processed_root: Path,
    year: int,
) -> dict[str, pd.DataFrame]:
    return {
        "context": _read_reference_dataset(
            reference_processed_root / "context",
            year,
        ),
        "mask_as_of": _read_reference_dataset(
            reference_processed_root / "masks" / "as_of",
            year,
        ),
        "mask_execution": _read_reference_dataset(
            reference_processed_root / "masks" / "execution",
            year,
        ),
        "factors": _read_reference_dataset(
            reference_processed_root / "factors" / "price_volume",
            year,
        ),
        "labels": _read_reference_dataset(
            reference_processed_root / "labels" / "forward_returns",
            year,
        ),
    }


def _read_reference_dataset(root: Path, year: int) -> pd.DataFrame:
    frame = read_partitioned_parquet_dataset(root, years=(year,))
    return _normalize_reference_keys(frame)


def _normalize_reference_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["stock_code"] = (
        result["stock_code"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise")
    return result


def _canonicalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return AShareMarketAdapter().with_daily_bar_canonical_fields(
        frame,
        source_id="reference_migration",
        source_asset_column="stock_code",
        session_date_column="trade_date",
    )


def _evaluation_mask_frame(year_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = year_frames["mask_as_of"].loc[:, ["stock_code", "trade_date", "eligible"]]
    execution = year_frames["mask_execution"].loc[
        :, ["stock_code", "trade_date", "can_exit"]
    ]
    return _canonicalize_daily_frame(
        frame.merge(
            execution,
            on=["stock_code", "trade_date"],
            how="inner",
            validate="one_to_one",
        )
    )


def _factor_frame(year_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = year_frames["factors"]
    for source_name, columns in (
        (
            "context",
            (
                "stock_code",
                "trade_date",
                "market_type",
                "amount",
                "turnover",
                "total_market_cap",
                "float_market_cap",
            ),
        ),
        ("mask_as_of", ("stock_code", "trade_date", "eligible")),
        ("mask_execution", ("stock_code", "trade_date", "can_enter", "can_exit")),
    ):
        frame = frame.merge(
            year_frames[source_name].loc[:, list(columns)],
            on=["stock_code", "trade_date"],
            how="inner",
            validate="one_to_one",
        )
    return _canonicalize_daily_frame(frame)


def _label_frame(year_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = year_frames["labels"]
    for source_name, columns in (
        ("mask_as_of", ("stock_code", "trade_date", "eligible")),
        ("mask_execution", ("stock_code", "trade_date", "can_enter", "can_exit")),
    ):
        frame = frame.merge(
            year_frames[source_name].loc[:, list(columns)],
            on=["stock_code", "trade_date"],
            how="inner",
            validate="one_to_one",
        )
    return _canonicalize_daily_frame(frame)


def _write_migrated_frames(
    frames: dict[str, pd.DataFrame],
    *,
    processed_root: Path,
) -> None:
    catalog = a_share_processed_catalog()
    layout = layout_from_processed_root(processed_root)
    for data_id, frame in frames.items():
        spec = catalog.get(data_id)
        write_processed_data(
            frame.loc[:, list(spec.required_columns)],
            layout=layout,
            catalog=catalog,
            processed_data_id=data_id,
        )


__all__ = [
    "HistoricalMigrationResult",
    "migrate_reference_factor_label_history",
]
