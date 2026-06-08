from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage.datasets as datasets_module
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import (
    platform_entity_time_key,
    platform_interim_column_names,
)
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.markets.implementations.a_share.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.runtime.privacy import is_private_boundary_path
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    write_internal_artifact_manifest,
    write_layer_manifest,
    write_manifest,
    write_run_manifest,
)
from quant_research_platform.storage.parquet import (
    write_interim_data,
    write_partitioned_parquet_dataset,
    write_processed_data,
    write_research_dataset,
)


def test_partitioned_parquet_and_manifest_contract(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["000001", "000002"],
            "timestamp": pd.to_datetime(["2014-01-02", "2015-01-05"]),
            "value": [1.0, 2.0],
        }
    )
    data_root = tmp_path / "data" / "processed" / "a_share" / "panels" / "x" / "data"
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            market_id=MarketId("a_share"),
            category="panels",
            processed_data_id="x",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "value": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )

    write_partitioned_parquet_dataset(frame, data_root, catalog=catalog, data_id="x")
    manifest = write_manifest(
        data_root.parent / "_manifest" / "manifest.json",
        {
            "schema_version": "0.1",
            "primary_key": ["market_id", "asset_id", "timestamp"],
        },
    )

    assert (data_root / "year=2014" / "part-000.parquet").is_file()
    assert (data_root / "year=2015" / "part-000.parquet").is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == "0.1"
    assert is_private_boundary_path(tmp_path / "data" / "processed" / "a_share")
    assert not is_private_boundary_path(tmp_path / "public" / "x" / "data")


def test_manifest_writer_sanitizes_public_summary_payload(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "reports" / "private_real" / "a_share" / "run" / "manifest.json",
        {
            "path": r"C:\Users\DELL\data\raw\csmar\000001.SZ",
            "traceback": "Traceback (most recent call last): /mnt/data/raw",
            "token": "token=abcdef123456",
        },
    )
    text = manifest.read_text(encoding="utf-8")

    assert "C:" not in text
    assert "/mnt/" not in text
    assert "000001" not in text
    assert "abcdef123456" not in text


def test_manifest_writer_rejects_existing_target_file(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share"
        / "run"
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_manifest(manifest, {"schema_version": "0.1"})

    assert manifest.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_private_evidence_has_no_dataset_manifest_writer_api() -> None:
    import quant_research_platform.storage.manifests as manifests

    assert not hasattr(manifests, "write_dataset_manifest")


def test_layer_manifest_detects_missing_dataset_manifests(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    present = layer_root / "a_share" / "factors" / "present"
    missing = layer_root / "a_share" / "labels" / "missing"
    write_internal_artifact_manifest(
        present / "_manifest" / "manifest.json",
        _artifact_manifest_payload(
            dataset_id="present",
            logical_layer="processed",
            row_count=3,
            date_min="2014-01-02",
            date_max="2014-01-03",
        ),
    )

    manifest = write_layer_manifest(
        layer_root=layer_root,
        logical_layer="processed",
        run_id="run",
        dataset_roots=(present, missing),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["dataset_count"] == 1
    assert payload["total_rows"] == 3
    assert payload["datasets"] == [
        {
            "dataset_id": "present",
            "manifest_path": (
                "data/processed/a_share/factors/present/_manifest/manifest.json"
            ),
            "manifest_summary": {
                "row_count": 3,
                "date_min": "2014-01-02",
                "date_max": "2014-01-03",
                "duplicate_primary_key_count": None,
                "validation_status": "pass",
            },
        }
    ]
    assert payload["missing_dataset_manifests"] == [
        "data/processed/a_share/labels/missing/_manifest/manifest.json"
    ]
    assert payload["layer_validation_status"] == "fail"


def test_layer_manifest_accepts_artifact_writer_schema_manifest(
    tmp_path: Path,
) -> None:
    layer_root = tmp_path / "data" / "processed"
    dataset = layer_root / "a_share" / "factors" / "price_volume"
    write_internal_artifact_manifest(
        dataset / "_manifest" / "manifest.json",
        {
            "schema_version": "0.1",
            "market_id": "a_share",
            "category": "factors",
            "processed_data_id": "price_volume",
            "primary_key": ["market_id", "asset_id", "timestamp"],
            "partition_columns": ["year"],
            "required_columns": ["market_id", "asset_id", "timestamp", "mom_20d"],
            "column_roles": {
                "market_id": "key",
                "asset_id": "key",
                "timestamp": "timestamp",
                "mom_20d": "factor",
            },
        },
    )

    manifest = write_layer_manifest(
        layer_root=layer_root,
        logical_layer="processed",
        run_id="s8",
        dataset_roots=(dataset,),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["dataset_count"] == 1
    assert payload["datasets"][0]["dataset_id"] == "price_volume"
    assert payload["datasets"][0]["manifest_summary"]["validation_status"] is None
    assert payload["failed_dataset_validations"] == []
    assert payload["layer_validation_status"] == "pass"


def test_run_manifest_aggregates_layers(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    evidence = tmp_path / "data" / "evidence"
    processed.mkdir(parents=True)
    evidence.mkdir(parents=True)
    write_layer_manifest(
        layer_root=processed,
        logical_layer="processed",
        run_id="run",
        dataset_roots=(),
    )
    write_layer_manifest(
        layer_root=evidence,
        logical_layer="evidence",
        run_id="run",
        dataset_roots=(),
    )

    manifest = write_run_manifest(
        manifests_root=tmp_path / "manifests",
        run_id="run",
        output_root=Path(".run") / "s1",
        layer_manifest_paths=(
            processed / "_manifests" / "layer_manifest.json",
            evidence / "_manifests" / "layer_manifest.json",
        ),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["run_id"] == "run"
    assert payload["output_root"] == ".run/s1"
    assert payload["layer_count"] == 2
    assert payload["run_validation_status"] == "pass"


def test_processed_writer_resolves_spec_and_layout(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            market_id=MarketId("a_share"),
            category="panels",
            processed_data_id="daily_panel",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "value": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )

    data_root = write_processed_data(
        frame,
        layout=StorageLayout(project_root=tmp_path),
        catalog=catalog,
        processed_data_id="daily_panel",
    )
    manifest = json.loads(
        (data_root.parent / "_manifest" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert data_root == (
        tmp_path / "data" / "processed" / "a_share" / "panels" / "daily_panel" / "data"
    )
    assert manifest["schema_version"] == "0.1"
    assert manifest["grain"] == "asset_timestamp"
    assert manifest["primary_key"] == ["market_id", "asset_id", "timestamp"]


def test_processed_writer_rejects_non_finite_factor_values(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2014-01-02", "2015-01-05"]),
            "factor_a": [1.0, float("nan")],
            "factor_b": [float("inf"), 2.0],
        }
    )
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
                "factor_a": ColumnRole.FACTOR,
                "factor_b": ColumnRole.FACTOR,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )

    with pytest.raises(ValueError, match="non_finite_columns: factor_b"):
        write_processed_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            processed_data_id="x",
        )

    artifact_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "x"
    assert not artifact_root.exists()


def test_layer_manifest_surfaces_dataset_validation_failure_from_targeted_stats(
    tmp_path: Path,
) -> None:
    layer_root = tmp_path / "data" / "processed"
    dataset = layer_root / "a_share" / "factors" / "bad_factor"
    write_internal_artifact_manifest(
        dataset / "_manifest" / "manifest.json",
        {
            "schema_version": "0.1",
            "run_id": "artifact-writer",
            "logical_layer": "processed",
            "dataset_id": "bad_factor",
            "primary_key": ["market_id", "asset_id", "timestamp"],
            "row_count": 2,
            "date_min": "2014-01-02",
            "date_max": "2014-01-03",
            "null_count_summary": {
                "market_id": 0,
                "asset_id": 0,
                "timestamp": 1,
            },
            "non_finite_count_summary": {"factor_a": 1},
            "duplicate_primary_key_count": 1,
            "validation_status": "fail",
        },
    )

    manifest = write_layer_manifest(
        layer_root=layer_root,
        logical_layer="processed",
        run_id="run",
        dataset_roots=(dataset,),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["layer_validation_status"] == "fail"
    assert payload["failed_dataset_validations"] == ["bad_factor"]
    assert payload["datasets"][0]["manifest_summary"] == {
        "row_count": 2,
        "date_min": "2014-01-02",
        "date_max": "2014-01-03",
        "duplicate_primary_key_count": 1,
        "validation_status": "fail",
    }


def test_interim_writer_resolves_spec_and_layout(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "session_date": pd.to_datetime(["2014-01-02"]),
            "available_at": pd.to_datetime(["2014-01-02"]),
            "source_id": ["csmar"],
            "source_asset_id": ["000001"],
            "close": [1.0],
        }
    )
    catalog = InterimDataCatalog()
    catalog.register(
        InterimDataSpec(
            market_id=MarketId("a_share"),
            source_id=SourceId("csmar"),
            interim_data_id="daily_market",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                **{
                    name: ColumnRole.TIMESTAMP
                    if name in {"timestamp", "session_date", "available_at"}
                    else ColumnRole.SOURCE
                    if name in {"source_id", "source_asset_id"}
                    else ColumnRole.KEY
                    for name in platform_interim_column_names()
                },
                "close": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )

    data_root = write_interim_data(
        frame,
        layout=StorageLayout(project_root=tmp_path),
        catalog=catalog,
        interim_data_id="daily_market",
    )

    assert data_root == (
        tmp_path / "data" / "interim" / "csmar" / "daily_market" / "data"
    )
    assert (data_root / "year=2014" / "part-000.parquet").is_file()


def test_interim_manifest_preserves_source_id_schema_fields(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "source_id": ["csmar", "csmar"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "calendar_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "source_market_type": [1, 1],
            "is_trading_day": [False, True],
            "source_day_of_week": [1, 2],
            "source_state": ["C", "O"],
        }
    )
    catalog = InterimDataCatalog()
    catalog.register(
        InterimDataSpec(
            market_id=MarketId("a_share"),
            source_id=SourceId("csmar"),
            interim_data_id="trading_calendar",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.SOURCE_MARKET_CALENDAR_DATE,
            column_roles={
                "market_id": ColumnRole.KEY,
                "source_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "calendar_date": ColumnRole.TIMESTAMP,
                "source_market_type": ColumnRole.KEY,
                "is_trading_day": ColumnRole.CONTEXT,
                "source_day_of_week": ColumnRole.CONTEXT,
                "source_state": ColumnRole.CONTEXT,
            },
            primary_key=(
                "market_id",
                "source_id",
                "source_market_type",
                "calendar_date",
            ),
            partition_columns=("year",),
        )
    )

    data_root = write_interim_data(
        frame,
        layout=StorageLayout(project_root=tmp_path),
        catalog=catalog,
        interim_data_id="trading_calendar",
    )
    manifest = json.loads(
        (data_root.parent / "_manifest" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    spec_json = json.loads((data_root.parent / "spec.json").read_text("utf-8"))

    for payload in (manifest, spec_json):
        assert payload["source_id"] == "csmar"
        assert payload["grain"] == "source_market_calendar_date"
        assert payload["primary_key"] == [
            "market_id",
            "source_id",
            "source_market_type",
            "calendar_date",
        ]
        assert "source_id" in payload["required_columns"]
        assert payload["column_roles"]["source_id"] == "key"
        assert "[private]" not in payload["primary_key"]
        assert "[private]" not in payload["required_columns"]
        assert "[private]" not in payload["column_roles"]


def test_internal_interim_manifest_rejects_dangerous_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe_internal_artifact_metadata"):
        write_internal_artifact_manifest(
            tmp_path / "data" / "interim" / "csmar" / "bad" / "spec.json",
            {
                "schema_version": "0.1",
                "source_id": "csmar",
                "primary_key": ["market_id", "source_id", "timestamp"],
                "raw_path": r"C:\Users\DELL\data\raw\csmar\export.csv",
            },
        )

    assert not (tmp_path / "data" / "interim" / "csmar" / "bad" / "spec.json").exists()


def test_research_dataset_writer_resolves_spec_and_layout(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "mom_20d": [1.0],
            "forward_return_20d": [0.01],
        }
    )
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            market_id=MarketId("a_share"),
            dataset_id="demo",
            processed_data_ids=(
                PRICE_VOLUME_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
            ),
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )

    data_root = write_research_dataset(
        frame,
        layout=StorageLayout(project_root=tmp_path),
        catalog=catalog,
        dataset_id="demo",
    )

    assert data_root == (
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "research_datasets"
        / "demo"
        / "data"
    )
    assert (data_root.parent / "spec.json").is_file()
    assert (data_root.parent / "_manifest" / "manifest.json").is_file()


def test_parquet_writer_rejects_missing_required_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
        }
    )
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

    with pytest.raises(ValueError, match="missing_required_columns"):
        write_partitioned_parquet_dataset(
            frame,
            tmp_path / "data" / "processed" / "a_share" / "factors" / "x" / "data",
            catalog=catalog,
            data_id="x",
        )


def test_parquet_writer_rejects_unregistered_output_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
            "unknown": [2.0],
        }
    )
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

    with pytest.raises(ValueError, match="unregistered_output_columns"):
        write_partitioned_parquet_dataset(
            frame,
            tmp_path / "data" / "processed" / "a_share" / "factors" / "x" / "data",
            catalog=catalog,
            data_id="x",
        )


def test_parquet_writer_rejects_public_output_root(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )
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

    with pytest.raises(ValueError, match="outside_private_boundary"):
        write_partitioned_parquet_dataset(
            frame,
            tmp_path / "public" / "factors" / "x" / "data",
            catalog=catalog,
            data_id="x",
        )


def test_partitioned_writer_rejects_processed_root_not_derived_from_spec(
    tmp_path: Path,
) -> None:
    frame, catalog = _processed_writer_case()
    invalid_roots = [
        tmp_path / "data" / "processed" / "other_market" / "factors" / "x" / "data",
        tmp_path / "data" / "processed" / "a_share" / "other_category" / "x" / "data",
        tmp_path / "data" / "processed" / "a_share" / "factors" / "other_id" / "data",
    ]

    for root in invalid_roots:
        with pytest.raises(ValueError, match="invalid_processed_output_root"):
            write_partitioned_parquet_dataset(frame, root, catalog=catalog, data_id="x")

        assert not root.exists()
        assert list(root.parent.glob(".data.tmp-*")) == []


def test_partitioned_writer_accepts_processed_root_when_parent_contains_data(
    tmp_path: Path,
) -> None:
    frame, catalog = _processed_writer_case()
    project_root = tmp_path / "outer" / "data" / "checkout" / "repo"
    root = project_root / "data" / "processed" / "a_share" / "factors" / "x" / "data"

    write_partitioned_parquet_dataset(frame, root, catalog=catalog, data_id="x")

    assert (root / "year=2014" / "part-000.parquet").is_file()


def test_partitioned_writer_accepts_interim_root_when_parent_contains_data(
    tmp_path: Path,
) -> None:
    frame, catalog = _interim_writer_case()
    project_root = tmp_path / "outer" / "data" / "checkout" / "repo"
    root = project_root / "data" / "interim" / "csmar" / "daily_market" / "data"

    write_partitioned_parquet_dataset(
        frame,
        root,
        catalog=catalog,
        data_id="daily_market",
    )

    assert (root / "year=2014" / "part-000.parquet").is_file()


def test_partitioned_writer_fails_closed_for_invalid_data_suffixes(
    tmp_path: Path,
) -> None:
    frame, catalog = _processed_writer_case()
    project_root = tmp_path / "outer" / "data" / "checkout" / "repo"
    invalid_roots = [
        project_root
        / "data"
        / "processed"
        / "a_share"
        / "factors"
        / "x"
        / "data"
        / "extra",
        project_root
        / "data"
        / "processed"
        / "a_share"
        / "factors"
        / "x"
        / "data"
        / "repo"
        / "data"
        / "processed"
        / "a_share"
        / "factors"
        / "x"
        / "data",
    ]

    for root in invalid_roots:
        with pytest.raises(ValueError, match="invalid_processed_output_root"):
            write_partitioned_parquet_dataset(frame, root, catalog=catalog, data_id="x")


def test_partitioned_writer_rejects_interim_root_not_derived_from_spec(
    tmp_path: Path,
) -> None:
    frame, catalog = _interim_writer_case()
    invalid_roots = [
        tmp_path / "data" / "interim" / "other_source" / "daily_market" / "data",
        tmp_path / "data" / "interim" / "csmar" / "other_interim" / "data",
    ]

    for root in invalid_roots:
        with pytest.raises(ValueError, match="invalid_interim_output_root"):
            write_partitioned_parquet_dataset(
                frame,
                root,
                catalog=catalog,
                data_id="daily_market",
            )

        assert not root.exists()
        assert list(root.parent.glob(".data.tmp-*")) == []


def test_partitioned_writer_rejects_research_root_not_derived_from_spec(
    tmp_path: Path,
) -> None:
    frame, catalog = _research_dataset_writer_case()
    invalid_roots = [
        tmp_path
        / "data"
        / "processed"
        / "other_market"
        / "research_datasets"
        / "demo"
        / "data",
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "research_datasets"
        / "other_dataset"
        / "data",
    ]

    for root in invalid_roots:
        with pytest.raises(ValueError, match="invalid_research_dataset_output_root"):
            write_partitioned_parquet_dataset(
                frame,
                root,
                catalog=catalog,
                data_id="demo",
            )

        assert not root.exists()
        assert list(root.parent.glob(".data.tmp-*")) == []


def test_parquet_writer_rejects_existing_output_root_without_partial_write(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )
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
    root = tmp_path / "data" / "processed" / "a_share" / "factors" / "x" / "data"
    root.mkdir(parents=True)
    sentinel = root / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_partitioned_parquet_dataset(frame, root, catalog=catalog, data_id="x")

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (root / "year=2014").exists()
    assert list(root.parent.glob("*backup*")) == []


def test_processed_writer_rejects_existing_manifest_before_creating_data(
    tmp_path: Path,
) -> None:
    frame, catalog = _processed_writer_case()
    artifact_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "x"
    manifest = artifact_root / "_manifest" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_processed_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            processed_data_id="x",
        )

    assert not (artifact_root / "data").exists()
    assert manifest.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert list(artifact_root.parent.glob(".x.tmp-*")) == []


def test_processed_writer_rejects_existing_data_before_writing_manifest_or_spec(
    tmp_path: Path,
) -> None:
    frame, catalog = _processed_writer_case()
    artifact_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "x"
    data_root = artifact_root / "data"
    data_root.mkdir(parents=True)
    (data_root / "sentinel.txt").write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_processed_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            processed_data_id="x",
        )

    assert (data_root / "sentinel.txt").read_text(encoding="utf-8") == "existing\n"
    assert not (artifact_root / "_manifest" / "manifest.json").exists()
    assert not (artifact_root / "spec.json").exists()


def test_processed_writer_rejects_existing_spec_before_creating_data(
    tmp_path: Path,
) -> None:
    frame, catalog = _processed_writer_case()
    artifact_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "x"
    spec_path = artifact_root / "spec.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_processed_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            processed_data_id="x",
        )

    assert not (artifact_root / "data").exists()
    assert spec_path.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert not (artifact_root / "_manifest" / "manifest.json").exists()


def test_interim_writer_rejects_existing_manifest_before_creating_data(
    tmp_path: Path,
) -> None:
    frame, catalog = _interim_writer_case()
    artifact_root = tmp_path / "data" / "interim" / "csmar" / "daily_market"
    manifest = artifact_root / "_manifest" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_interim_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            interim_data_id="daily_market",
        )

    assert not (artifact_root / "data").exists()
    assert manifest.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_interim_writer_rejects_existing_data_before_writing_manifest_or_spec(
    tmp_path: Path,
) -> None:
    frame, catalog = _interim_writer_case()
    artifact_root = tmp_path / "data" / "interim" / "csmar" / "daily_market"
    data_root = artifact_root / "data"
    data_root.mkdir(parents=True)
    (data_root / "sentinel.txt").write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_interim_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            interim_data_id="daily_market",
        )

    assert (data_root / "sentinel.txt").read_text(encoding="utf-8") == "existing\n"
    assert not (artifact_root / "_manifest" / "manifest.json").exists()
    assert not (artifact_root / "spec.json").exists()


def test_research_dataset_writer_rejects_existing_spec_before_creating_data(
    tmp_path: Path,
) -> None:
    frame, catalog = _research_dataset_writer_case()
    artifact_root = (
        tmp_path / "data" / "processed" / "a_share" / "research_datasets" / "demo"
    )
    spec_path = artifact_root / "spec.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_research_dataset(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            dataset_id="demo",
        )

    assert not (artifact_root / "data").exists()
    assert spec_path.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert not (artifact_root / "_manifest" / "manifest.json").exists()


def test_research_dataset_writer_rejects_existing_data_before_writing_metadata(
    tmp_path: Path,
) -> None:
    frame, catalog = _research_dataset_writer_case()
    artifact_root = (
        tmp_path / "data" / "processed" / "a_share" / "research_datasets" / "demo"
    )
    data_root = artifact_root / "data"
    data_root.mkdir(parents=True)
    (data_root / "sentinel.txt").write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_research_dataset(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            dataset_id="demo",
        )

    assert (data_root / "sentinel.txt").read_text(encoding="utf-8") == "existing\n"
    assert not (artifact_root / "spec.json").exists()
    assert not (artifact_root / "_manifest" / "manifest.json").exists()


def test_processed_writer_cleans_artifact_temp_root_after_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, catalog = _processed_writer_case()
    artifact_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "x"

    def fail_parquet_write(_frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_parquet_failure")

    monkeypatch.setattr(datasets_module, "_write_parquet_file", fail_parquet_write)

    with pytest.raises(RuntimeError, match="simulated_parquet_failure"):
        write_processed_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            processed_data_id="x",
        )

    assert not artifact_root.exists()
    assert list(artifact_root.parent.glob(".x.tmp-*")) == []


def test_interim_writer_cleans_artifact_temp_root_after_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, catalog = _interim_writer_case()
    artifact_root = tmp_path / "data" / "interim" / "csmar" / "daily_market"

    def fail_parquet_write(_frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_parquet_failure")

    monkeypatch.setattr(datasets_module, "_write_parquet_file", fail_parquet_write)

    with pytest.raises(RuntimeError, match="simulated_parquet_failure"):
        write_interim_data(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            interim_data_id="daily_market",
        )

    assert not artifact_root.exists()
    assert list(artifact_root.parent.glob(".daily_market.tmp-*")) == []


def test_research_dataset_writer_cleans_artifact_temp_root_after_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, catalog = _research_dataset_writer_case()
    artifact_root = (
        tmp_path / "data" / "processed" / "a_share" / "research_datasets" / "demo"
    )

    def fail_parquet_write(_frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_parquet_failure")

    monkeypatch.setattr(datasets_module, "_write_parquet_file", fail_parquet_write)

    with pytest.raises(RuntimeError, match="simulated_parquet_failure"):
        write_research_dataset(
            frame,
            layout=StorageLayout(project_root=tmp_path),
            catalog=catalog,
            dataset_id="demo",
        )

    assert not artifact_root.exists()
    assert list(artifact_root.parent.glob(".demo.tmp-*")) == []


def _processed_writer_case() -> tuple[pd.DataFrame, ProcessedDataCatalog]:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )
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
    return frame, catalog


def _artifact_manifest_payload(
    *,
    dataset_id: str,
    logical_layer: str,
    row_count: int,
    date_min: str | None = None,
    date_max: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "run_id": "artifact-writer",
        "logical_layer": logical_layer,
        "dataset_id": dataset_id,
        "primary_key": ["market_id", "asset_id", "timestamp"],
        "partition_columns": ["year"],
        "columns": ["market_id", "asset_id", "timestamp", "value"],
        "row_count": row_count,
        "partition_count": 0,
        "date_min": date_min,
        "date_max": date_max,
        "null_count_summary": {
            "market_id": 0,
            "asset_id": 0,
            "timestamp": 0,
        },
        "non_finite_count_summary": {},
        "duplicate_primary_key_count": None,
        "validation_status": "pass",
    }


def _interim_writer_case() -> tuple[pd.DataFrame, InterimDataCatalog]:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "session_date": pd.to_datetime(["2014-01-02"]),
            "available_at": pd.to_datetime(["2014-01-02"]),
            "source_id": ["csmar"],
            "source_asset_id": ["000001"],
            "close": [1.0],
        }
    )
    catalog = InterimDataCatalog()
    catalog.register(
        InterimDataSpec(
            market_id=MarketId("a_share"),
            source_id=SourceId("csmar"),
            interim_data_id="daily_market",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                **{
                    name: ColumnRole.TIMESTAMP
                    if name in {"timestamp", "session_date", "available_at"}
                    else ColumnRole.SOURCE
                    if name in {"source_id", "source_asset_id"}
                    else ColumnRole.KEY
                    for name in platform_interim_column_names()
                },
                "close": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return frame, catalog


def _research_dataset_writer_case() -> tuple[pd.DataFrame, ResearchDatasetCatalog]:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "mom_20d": [1.0],
            "forward_return_20d": [0.01],
        }
    )
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            market_id=MarketId("a_share"),
            dataset_id="demo",
            processed_data_ids=(
                PRICE_VOLUME_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
            ),
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return frame, catalog
