from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    DatasetAccessMode,
    compare_dataset_fingerprints,
    fingerprint_dataset,
    read_dataset,
    write_dataset,
)


def _catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            market_id=MarketId("a_share"),
            category="factors",
            processed_data_id="x",
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
    )
    return catalog


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["000002", "000001", "000001"],
            "timestamp": pd.to_datetime(["2015-01-05", "2014-01-02", "2015-01-05"]),
            "value": [2.0, 1.0, math.nan],
        }
    )


def _data_root(tmp_path: Path, name: str = "x") -> Path:
    return tmp_path / "data" / "processed" / "a_share" / "factors" / name / "data"


def test_write_dataset_materialized_and_streaming_are_equivalent(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    materialized_root = _data_root(tmp_path, "x")
    streaming_root = _data_root(tmp_path, "x_stream")
    streaming_spec = ProcessedDataSpec(
        market_id=MarketId("a_share"),
        category="factors",
        processed_data_id="x_stream",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=catalog.get("x").column_roles,
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )
    catalog.register(streaming_spec)
    frame = _frame()

    write_dataset(
        frame,
        materialized_root,
        catalog=catalog,
        data_id="x",
        access_mode=DatasetAccessMode.MATERIALIZED,
    )
    write_dataset(
        (chunk for chunk in [frame.iloc[:1].copy(), frame.iloc[1:].copy()]),
        streaming_root,
        catalog=catalog,
        data_id="x_stream",
        access_mode=DatasetAccessMode.STREAMING,
    )

    materialized = read_dataset(
        materialized_root,
        catalog=catalog,
        data_id="x",
        access_mode=DatasetAccessMode.MATERIALIZED,
    )
    streaming = read_dataset(
        streaming_root,
        catalog=catalog,
        data_id="x_stream",
        access_mode=DatasetAccessMode.MATERIALIZED,
    )

    pd.testing.assert_frame_equal(
        materialized.reset_index(drop=True),
        streaming.reset_index(drop=True),
    )
    assert list(materialized.columns) == ["market_id", "asset_id", "timestamp", "value"]
    assert list(materialized["asset_id"]) == ["000001", "000001", "000002"]


def test_read_dataset_streaming_honors_columns_and_partitions(tmp_path: Path) -> None:
    catalog = _catalog()
    write_dataset(
        _frame(),
        _data_root(tmp_path),
        catalog=catalog,
        data_id="x",
        access_mode=DatasetAccessMode.MATERIALIZED,
    )

    chunks = list(
        read_dataset(
            _data_root(tmp_path),
            catalog=catalog,
            data_id="x",
            years=(2014,),
            columns=["market_id", "asset_id", "timestamp"],
            access_mode=DatasetAccessMode.STREAMING,
            resource_policy=RuntimeResourcePolicy(
                available_memory_bytes=10_000,
                memory_budget_fraction=0.70,
                chunk_target_bytes=1,
                memory_safety_factor=0.60,
                min_free_disk_bytes=0,
            ),
        )
    )

    assert len(chunks) == 1
    assert list(chunks[0].columns) == ["market_id", "asset_id", "timestamp"]
    assert chunks[0]["timestamp"].dt.year.tolist() == [2014]


def test_write_dataset_detects_cross_chunk_duplicate_and_null_keys(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    frame = _frame()
    duplicate = pd.concat([frame.iloc[:1], frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_primary_key"):
        write_dataset(
            (chunk for chunk in [duplicate.iloc[:1], duplicate.iloc[1:]]),
            _data_root(tmp_path),
            catalog=catalog,
            data_id="x",
            access_mode=DatasetAccessMode.STREAMING,
        )

    null_key = frame.copy()
    null_key.loc[0, "asset_id"] = None
    with pytest.raises(ValueError, match="null_primary_key_columns: asset_id"):
        write_dataset(
            null_key,
            _data_root(tmp_path),
            catalog=catalog,
            data_id="x",
            access_mode=DatasetAccessMode.MATERIALIZED,
        )


def test_write_dataset_rejects_non_finite_values_across_chunks(tmp_path: Path) -> None:
    catalog = _catalog()
    frame = _frame()
    frame.loc[0, "value"] = math.inf

    with pytest.raises(ValueError, match="non_finite_columns: value"):
        write_dataset(
            (chunk for chunk in [frame.iloc[:1], frame.iloc[1:]]),
            _data_root(tmp_path),
            catalog=catalog,
            data_id="x",
            access_mode=DatasetAccessMode.STREAMING,
        )


def test_fingerprint_is_independent_of_part_order(tmp_path: Path) -> None:
    catalog = _catalog()
    first_root = _data_root(tmp_path, "x")
    second_root = _data_root(tmp_path, "x_reordered")
    second_spec = ProcessedDataSpec(
        market_id=MarketId("a_share"),
        category="factors",
        processed_data_id="x_reordered",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=catalog.get("x").column_roles,
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )
    catalog.register(second_spec)

    frame = _frame()
    write_dataset(frame, first_root, catalog=catalog, data_id="x")
    write_dataset(
        frame.iloc[::-1].reset_index(drop=True),
        second_root,
        catalog=catalog,
        data_id="x_reordered",
    )

    first = fingerprint_dataset(first_root, catalog=catalog, data_id="x")
    second = fingerprint_dataset(second_root, catalog=catalog, data_id="x_reordered")
    comparison = compare_dataset_fingerprints(first, second)

    assert first.fingerprint_schema_version == "0.1"
    assert first.content_hash == second.content_hash
    assert comparison.matches
    assert comparison.unexplained_delta == 0
