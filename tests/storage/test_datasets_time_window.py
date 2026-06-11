from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)


def test_read_dataset_time_window_uses_closed_interval(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path, "windowed")
    write_dataset(root, _frame(), spec=_factor_spec("windowed"))

    result = read_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2014-01-02", "2015-01-05"),
    )

    assert list(result["asset_id"]) == ["000001", "000001"]
    assert list(result["value"]) == [1.0, 2.0]


def test_inspect_dataset_time_window_reports_window_scoped_counts(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path, "inspect")
    write_dataset(root, _frame(), spec=_factor_spec("inspect"))

    inspection = inspect_dataset(
        root,
        time_window=TimeWindow("2015-01-01", "2015-12-31"),
    )

    assert inspection.row_count == 1
    assert inspection.date_min == "2015-01-05"
    assert inspection.date_max == "2015-01-05"
    assert inspection.file_count == 1
    assert inspection.partitions == ("year=2015",)
    assert inspection.partition_row_counts == {"year=2015": 1}


def test_fingerprint_dataset_time_window_hashes_window_only(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path, "fingerprint")
    write_dataset(root, _frame(), spec=_factor_spec("fingerprint"))

    full = fingerprint_dataset(root)
    windowed = fingerprint_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2015-01-01", "2015-12-31"),
    )

    assert full.row_count == 3
    assert windowed.row_count == 1
    assert windowed.column_names == ("asset_id", "value")
    assert windowed.content_hash != full.content_hash


def test_compare_datasets_time_window_compares_window_only(tmp_path: Path) -> None:
    left = _artifact_root(tmp_path, "left")
    right = _artifact_root(tmp_path, "right")
    write_dataset(left, _frame(), spec=_factor_spec("left"))
    changed = _frame()
    changed.loc[changed["timestamp"] == pd.Timestamp("2016-01-04"), "value"] = 99.0
    write_dataset(right, changed, spec=_factor_spec("right"))

    comparison = compare_datasets(
        left,
        right,
        columns=["asset_id", "timestamp", "value"],
        time_window=TimeWindow("2014-01-01", "2015-12-31"),
    )

    assert comparison.matches is True
    assert comparison.status == "MATCH"


def test_time_window_without_timestamp_role_raises(tmp_path: Path) -> None:
    root = tmp_path / "data" / "processed" / "a_share" / "annotation" / "reference"
    spec = ProcessedDataSpec(
        data_id="reference",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.ANNOTATION,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_REFERENCE,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "name": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id"),
        partition_columns=(),
    )
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["000001"],
                "name": ["demo"],
            }
        ),
        spec=spec,
    )

    with pytest.raises(ValueError, match="time_window_not_supported"):
        read_dataset(root, time_window=TimeWindow("2014-01-01", "2014-12-31"))


def test_write_dataset_existing_root_raises_dataset_exists(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path, "exists")
    write_dataset(root, _frame(), spec=_factor_spec("exists"))

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(root, _frame(), spec=_factor_spec("exists"))


def _artifact_root(tmp_path: Path, data_id: str) -> Path:
    return tmp_path / "data" / "processed" / "a_share" / "factor" / data_id


def _factor_spec(data_id: str) -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000001", "000001"],
            "timestamp": pd.to_datetime(["2014-01-02", "2015-01-05", "2016-01-04"]),
            "value": [1.0, 2.0, 3.0],
        }
    )
