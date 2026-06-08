from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage.datasets as datasets
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.manifests import write_internal_artifact_manifest


def _spec(processed_data_id: str = "x") -> ProcessedDataSpec:
    return ProcessedDataSpec(
        market_id=MarketId("a_share"),
        category="factors",
        processed_data_id=processed_data_id,
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
            "asset_id": ["000002", "000001", "000001"],
            "timestamp": pd.to_datetime(["2015-01-05", "2014-01-02", "2015-01-05"]),
            "value": [2.0, 1.0, math.nan],
        }
    )


def _artifact_root(tmp_path: Path, name: str = "x") -> Path:
    return tmp_path / "data" / "processed" / "a_share" / "factors" / name


def _write(root: Path, frame: pd.DataFrame | list[pd.DataFrame]) -> None:
    write_dataset(
        root,
        frame if isinstance(frame, pd.DataFrame) else iter(frame),
        spec=_spec(root.name),
    )


def _spec_payload(spec: ProcessedDataSpec) -> dict[str, object]:
    return {
        "schema_version": spec.schema_version,
        "market_id": str(spec.market_id),
        "frequency": spec.frequency.value,
        "grain": spec.grain.value,
        "primary_key": list(spec.primary_key),
        "partition_columns": list(spec.partition_columns),
        "required_columns": list(spec.required_columns),
        "column_roles": {
            column: role.value for column, role in sorted(spec.column_roles.items())
        },
        "category": spec.category,
        "processed_data_id": spec.processed_data_id,
    }


def test_write_dataset_writes_complete_artifact_root_and_returns_none(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)

    result = write_dataset(root, _frame(), spec=_spec())

    assert result is None
    assert (root / "spec.json").is_file()
    assert (root / "_manifest" / "manifest.json").is_file()
    assert (root / "data" / "year=2014" / "part-000.parquet").is_file()
    assert (root / "data" / "year=2015" / "part-000.parquet").is_file()


def test_write_dataset_accepts_dataframe_and_dataframe_chunks(tmp_path: Path) -> None:
    materialized_root = _artifact_root(tmp_path, "materialized")
    chunked_root = _artifact_root(tmp_path, "chunked")
    frame = _frame()

    write_dataset(materialized_root, frame, spec=_spec("materialized"))
    write_dataset(
        chunked_root,
        (chunk for chunk in [frame.iloc[:1].copy(), frame.iloc[1:].copy()]),
        spec=_spec("chunked"),
    )

    materialized = read_dataset(materialized_root)
    chunked = read_dataset(chunked_root)
    pd.testing.assert_frame_equal(
        materialized.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        chunked.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
    )
    assert fingerprint_dataset(materialized_root).content_hash == fingerprint_dataset(
        chunked_root
    ).content_hash


def test_write_dataset_rejects_invalid_and_empty_chunks(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="invalid_dataset_chunk"):
        write_dataset(
            _artifact_root(tmp_path, "invalid"),
            iter([_frame().iloc[:1], {"not": "a frame"}]),
            spec=_spec("invalid"),
        )

    with pytest.raises(ValueError, match="empty_dataset"):
        write_dataset(
            _artifact_root(tmp_path, "empty"),
            iter([_frame().iloc[:0].copy()]),
            spec=_spec("empty"),
        )


def test_write_dataset_rejects_primary_key_and_infinite_value_failures(
    tmp_path: Path,
) -> None:
    frame = _frame()
    duplicate = pd.concat([frame.iloc[:1], frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            _artifact_root(tmp_path, "duplicate"),
            iter([duplicate.iloc[:1], duplicate.iloc[1:]]),
            spec=_spec("duplicate"),
        )

    null_key = frame.copy()
    null_key.loc[0, "asset_id"] = None
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        write_dataset(
            _artifact_root(tmp_path, "null_key"),
            null_key,
            spec=_spec("null_key"),
        )

    infinite = frame.copy()
    infinite.loc[0, "value"] = -math.inf
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            _artifact_root(tmp_path, "infinite"),
            iter([infinite.iloc[:1], infinite.iloc[1:]]),
            spec=_spec("infinite"),
        )


def test_write_dataset_overwrite_policy_and_staged_failure(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())
    original = read_dataset(root)

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(root, _frame(), spec=_spec())

    bad = _frame()
    bad.loc[0, "asset_id"] = None
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        write_dataset(root, bad, spec=_spec(), overwrite=True)

    pd.testing.assert_frame_equal(read_dataset(root), original)
    replacement = _frame()
    replacement["value"] = [101.0, 102.0, 103.0]
    write_dataset(root, replacement, spec=_spec(), overwrite=True)
    assert read_dataset(root)["value"].min() >= 101.0


def test_read_dataset_materialized_and_chunked_from_artifact_root(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())

    materialized = read_dataset(root, mode=DatasetReadMode.MATERIALIZED)
    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))

    assert list(materialized.columns) == ["market_id", "asset_id", "timestamp", "value"]
    assert len(chunks) == 2
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)


def test_read_dataset_accepts_canonical_artifact_not_created_by_writer(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    spec = _spec()
    part = root / "data" / "year=2014" / "part-000.parquet"
    part.parent.mkdir(parents=True)
    _frame().iloc[[1]].to_parquet(part, index=False)
    write_internal_artifact_manifest(root / "spec.json", _spec_payload(spec))
    write_internal_artifact_manifest(
        root / "_manifest" / "manifest.json",
        {
            "schema_version": spec.schema_version,
            "dataset_id": spec.processed_data_id,
            "row_count": 1,
            "file_count": 1,
            "validation_status": "pass",
        },
    )

    frame = read_dataset(root)

    assert list(frame.columns) == ["market_id", "asset_id", "timestamp", "value"]
    assert frame["asset_id"].tolist() == ["000001"]


def test_read_dataset_rejects_data_root_missing_columns_and_partitions(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())

    with pytest.raises(ValueError, match="dataset_root_must_be_artifact_root"):
        read_dataset(root / "data")

    with pytest.raises(ValueError, match="dataset_column_not_found"):
        read_dataset(root, columns=["market_id", "missing"])

    selected = read_dataset(
        root,
        columns=["asset_id", "timestamp"],
        partition_filter={"year": [2014]},
    )
    assert list(selected.columns) == ["asset_id", "timestamp"]
    assert selected["timestamp"].dt.year.tolist() == [2014]

    with pytest.raises(ValueError, match="partition_key_not_found"):
        read_dataset(root, partition_filter={"month": [1]})

    with pytest.raises(ValueError, match="partition_value_not_found"):
        read_dataset(root, partition_filter={"year": [2020]})


def test_inspect_dataset_returns_minimal_metadata(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())

    inspection = inspect_dataset(root)

    assert inspection.root == root
    assert inspection.data_root == root / "data"
    assert inspection.spec["processed_data_id"] == "x"
    assert inspection.columns == ("market_id", "asset_id", "timestamp", "value")
    assert inspection.schema_hash
    assert inspection.file_count == 2
    assert inspection.manifest["row_count"] == 3
    assert not hasattr(inspection, "schema")


def test_dataset_planning_metadata_counts_rows_by_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())

    def read_parquet_blocked(*args, **kwargs):
        raise AssertionError("planning metadata must not read row contents")

    def fingerprint_blocked(*args, **kwargs):
        raise AssertionError("planning metadata must not fingerprint rows")

    monkeypatch.setattr(datasets.pd, "read_parquet", read_parquet_blocked)
    monkeypatch.setattr(datasets, "fingerprint_dataset", fingerprint_blocked)

    metadata = datasets._dataset_planning_metadata(
        root,
        partition_filter={"year": (2014,)},
    )

    assert metadata.row_count == 1
    assert metadata.file_count == 1
    assert metadata.partitions == ("year=2014",)


def test_dataset_planning_metadata_counts_multiple_partitions(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())

    metadata = datasets._dataset_planning_metadata(
        root,
        partition_filter={"year": (2014, 2015)},
    )

    assert metadata.row_count == 3
    assert metadata.file_count == 2
    assert metadata.partitions == ("year=2014", "year=2015")


def test_dataset_planning_metadata_missing_partition_propagates_known_error(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())

    with pytest.raises(ValueError, match="partition_value_not_found"):
        datasets._dataset_planning_metadata(
            root,
            partition_filter={"year": (2020,)},
        )


def test_dataset_planning_metadata_corrupt_structure_does_not_return_zero(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())
    (root / "spec.json").unlink()

    with pytest.raises(ValueError, match="dataset_spec_missing"):
        datasets._dataset_planning_metadata(root)


def test_inspect_and_fingerprint_detect_schema_mismatch_across_parts(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    write_dataset(root, _frame(), spec=_spec())
    bad_part = root / "data" / "year=2016" / "part-000.parquet"
    bad_part.parent.mkdir(parents=True)
    _frame().assign(extra=1).to_parquet(bad_part, index=False)

    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        inspect_dataset(root)
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        fingerprint_dataset(root)


def test_fingerprint_is_deterministic_under_file_order_variation(
    tmp_path: Path,
) -> None:
    first_root = _artifact_root(tmp_path, "first")
    second_root = _artifact_root(tmp_path, "second")
    frame = _frame()

    write_dataset(first_root, frame, spec=_spec("first"))
    write_dataset(
        second_root,
        frame.iloc[::-1].reset_index(drop=True),
        spec=_spec("second"),
    )

    first = fingerprint_dataset(first_root)
    second = fingerprint_dataset(second_root)
    assert first.content_hash == second.content_hash
    assert first.partition_fingerprints == second.partition_fingerprints


def test_compare_datasets_statuses_for_semantic_and_schema_differences(
    tmp_path: Path,
) -> None:
    left_root = _artifact_root(tmp_path, "left")
    right_root = _artifact_root(tmp_path, "right")
    schema_root = _artifact_root(tmp_path, "schema")
    write_dataset(left_root, _frame(), spec=_spec("left"))
    write_dataset(
        right_root,
        iter([_frame().iloc[:1], _frame().iloc[1:]]),
        spec=_spec("right"),
    )

    comparison = compare_datasets(left_root, right_root)
    assert comparison.status == "MATCH"
    assert comparison.matches
    assert comparison.layout_delta["file_count"] == {"left": 2, "right": 3}

    wider_spec = ProcessedDataSpec(
        market_id=MarketId("a_share"),
        category="factors",
        processed_data_id="schema",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **_spec().column_roles,
            "other": ColumnRole.FACTOR,
        },
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )
    write_dataset(schema_root, _frame().assign(other=1.0), spec=wider_spec)

    schema_comparison = compare_datasets(left_root, schema_root)
    assert schema_comparison.status == "SCHEMA_MISMATCH"
    assert not schema_comparison.matches


def test_storage_datasets_public_api_is_hard_cutover() -> None:
    import quant_research_platform.storage as storage
    import quant_research_platform.storage.datasets as datasets

    def old_name(*parts: str) -> str:
        return "".join(parts)

    assert datasets.__all__ == [
        "DatasetReadMode",
        "DatasetInspection",
        "DatasetFingerprint",
        "DatasetComparison",
        "write_dataset",
        "read_dataset",
        "inspect_dataset",
        "fingerprint_dataset",
        "compare_datasets",
    ]
    for banned in (
        old_name("Dataset", "AccessMode"),
        old_name("Dataset", "WriteResult"),
        old_name("Data", "SpecProvider"),
        old_name("Dataset", "ManifestAccumulator"),
        old_name("compare_dataset", "_fingerprints"),
        old_name("parquet_dataset", "_columns"),
        old_name("write_dataset", "_artifact"),
        old_name("write_dataset", "_contents"),
    ):
        assert not hasattr(datasets, banned)
    assert not hasattr(storage, "write_dataset")
