from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import datasets
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)


def test_storage_public_surface_keeps_physical_execution_knobs_internal() -> None:
    forbidden = {
        "partition_filter",
        "_partition_filter",
        "year_filter",
        "_year_filter",
        "force",
        "overwrite",
        "dependency_policy",
        "_dataset_planning_metadata",
    }
    public_functions = (
        datasets.write_dataset,
        datasets.read_dataset,
        datasets.inspect_dataset,
        datasets.fingerprint_dataset,
        datasets.compare_datasets,
    )

    for function in public_functions:
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    assert not hasattr(datasets, "_dataset_planning_metadata")


def test_storage_duplicate_primary_key_fails_closed(tmp_path: Path) -> None:
    frame = pd.concat([_frame().iloc[:1], _frame().iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            _artifact_root(tmp_path),
            frame,
            spec=_processed_spec(),
        )


def test_storage_chunked_write_read_keeps_schema_keys_and_sort_stable(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    frame = _frame()
    write_dataset(
        root,
        (chunk for chunk in (frame.iloc[:2], frame.iloc[2:])),
        spec=_processed_spec(),
    )

    result = read_dataset(root)

    assert list(result.columns) == list(frame.columns)
    assert result.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    pd.testing.assert_frame_equal(result, frame)


def test_inspect_dataset_time_window_reports_only_scope_contributors(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_processed_spec())

    inspection = inspect_dataset(
        root,
        time_window=TimeWindow("2020-01-03", "2020-01-06"),
    )

    assert inspection.row_count == 2
    assert inspection.date_min == "2020-01-06"
    assert inspection.date_max == "2020-01-06"
    assert inspection.file_count >= 1


def _processed_spec() -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id="correctness_demo",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "factor_value": ColumnRole.FACTOR,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )


def _artifact_root(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / "correctness_demo"
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_b", "asset_a", "asset_b"],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-06", "2020-01-06"]
            ),
            "factor_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
