from __future__ import annotations

import json
import math
from inspect import signature
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage.datasets as datasets
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
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def _input_fingerprint_payload(
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_hash": "a" * 64,
        "row_count": 3,
        "key_columns": ["market_id", "asset_id", "timestamp"],
        "content_hash": "b" * 64,
        "partition_fingerprints": [
            {"partition": "year=2014", "row_count": 1, "content_hash": "c" * 64}
        ],
        "file_count": 1,
        "column_names": ["market_id", "asset_id", "timestamp", "value"],
    }
    if extra:
        payload.update(extra)
    return payload


def count_candidate_data_file_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"count": 0}
    original = dataset_reading.candidate_data_files

    def count_candidate_data_files(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dataset_reading,
        "candidate_data_files",
        count_candidate_data_files,
    )
    return calls


def test_write_dataset_writes_complete_artifact_root(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)

    result = write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    spec_payload = json.loads((root / "spec.json").read_text(encoding="utf-8"))

    assert result is None
    assert (root / "data" / "year=2014" / "part-000.parquet").is_file()
    assert (root / "data" / "year=2015" / "part-000.parquet").is_file()
    assert manifest["columns"] == ["market_id", "asset_id", "timestamp", "value"]
    assert manifest["schema_id"] == "qrp-dataset-manifest"
    assert manifest["partition_row_counts"] == {"year=2014": 1, "year=2015": 2}
    assert manifest["file_count"] == 2
    assert manifest["row_count"] == 3
    assert spec_payload["data_id"] == "x"
    assert spec_payload["schema_id"] == "qrp-dataset-spec"
    assert spec_payload["kind"] == "factor"
    assert "category" not in spec_payload
    assert "processed_data_id" not in spec_payload


def test_write_dataset_partitions_without_groupby_sort_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_groupby(*args: object, **kwargs: object) -> object:
        raise AssertionError("partition_write_must_not_groupby_full_frame")

    monkeypatch.setattr(pd.DataFrame, "groupby", fail_groupby)
    root = make_dataset_artifact_root(tmp_path, "no_groupby_partition")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("no_groupby_partition"),
    )

    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["partition_row_counts"] == {"year=2014": 1, "year=2015": 2}


def test_write_dataset_merges_allowed_instance_metadata(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    input_fingerprints = {"demo_input": _input_fingerprint_payload()}

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(),
        instance_metadata={"input_fingerprints": input_fingerprints},
    )

    inspection = inspect_dataset(root)
    assert inspection.manifest["input_fingerprints"] == input_fingerprints


def test_write_dataset_rejects_malformed_input_fingerprints_metadata(
    tmp_path: Path,
) -> None:
    cases = [
        {"demo_input": _input_fingerprint_payload(extra={"unexpected": "x"})},
        {"demo_input": _input_fingerprint_payload(extra={"stock_code_sample": "x"})},
        {"demo_input": _input_fingerprint_payload(extra={"row_count": math.nan})},
        {
            "demo_input": _input_fingerprint_payload(
                extra={
                    "partition_fingerprints": [
                        {
                            "partition": "year=2014",
                            "row_count": 1,
                            "content_hash": "c" * 64,
                            "sample": ["x"],
                        }
                    ]
                }
            )
        },
    ]

    for index, input_fingerprints in enumerate(cases):
        with pytest.raises(ValueError, match="invalid_input_fingerprints_metadata"):
            write_dataset(
                make_dataset_artifact_root(
                    tmp_path, f"bad_fingerprint_metadata_{index}"
                ),
                make_default_dataset_frame(),
                spec=make_processed_dataset_spec(f"bad_fingerprint_metadata_{index}"),
                instance_metadata={"input_fingerprints": input_fingerprints},
            )


def test_write_dataset_rejects_unsupported_reserved_and_unsafe_instance_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported_dataset_instance_metadata_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "unsupported_metadata"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("unsupported_metadata"),
            instance_metadata={"other_metadata": {}},
        )

    with pytest.raises(ValueError, match="reserved_dataset_instance_metadata_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "reserved_metadata"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("reserved_metadata"),
            instance_metadata={"row_count": 999},
        )

    unsafe_root = make_dataset_artifact_root(tmp_path, "unsafe_metadata")
    unsafe_fingerprint = _input_fingerprint_payload(
        extra={"content_hash": "data/raw/internal-provider/export.csv"}
    )
    with pytest.raises(ValueError, match="unsafe_internal_artifact_metadata"):
        write_dataset(
            unsafe_root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("unsafe_metadata"),
            instance_metadata={
                "input_fingerprints": {"demo_input": unsafe_fingerprint}
            },
        )
    assert not unsafe_root.exists()


def test_write_dataset_accepts_dataframe_and_dataframe_chunks(tmp_path: Path) -> None:
    materialized_root = make_dataset_artifact_root(tmp_path, "materialized")
    chunked_root = make_dataset_artifact_root(tmp_path, "chunked")
    frame = make_default_dataset_frame()

    write_dataset(
        materialized_root, frame, spec=make_processed_dataset_spec("materialized")
    )
    write_dataset(
        chunked_root,
        (chunk for chunk in [frame.iloc[:1].copy(), frame.iloc[1:].copy()]),
        spec=make_processed_dataset_spec("chunked"),
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
    assert (
        fingerprint_dataset(materialized_root).content_hash
        == fingerprint_dataset(chunked_root).content_hash
    )


def test_write_dataset_preserves_schema_for_empty_chunks(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_schema")
    empty_chunk = make_default_dataset_frame().iloc[:0].copy()

    write_dataset(
        root, iter([empty_chunk]), spec=make_processed_dataset_spec("empty_schema")
    )

    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    materialized = read_dataset(root)
    inspection = inspect_dataset(root)
    assert manifest["row_count"] == 0
    assert manifest["columns"] == ["market_id", "asset_id", "timestamp", "value"]
    assert manifest["file_count"] == 1
    assert materialized.empty
    assert materialized.columns.tolist() == manifest["columns"]
    assert inspection.row_count == 0
    assert inspection.columns == tuple(manifest["columns"])


def test_write_dataset_rejects_invalid_empty_and_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="invalid_dataset_chunk"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "invalid"),
            iter([make_default_dataset_frame().iloc[:1], {"not": "a frame"}]),
            spec=make_processed_dataset_spec("invalid"),
        )

    with pytest.raises(ValueError, match="empty_dataset_schema_missing"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "empty_iterable"),
            iter(()),
            spec=make_processed_dataset_spec("empty_iterable"),
        )

    duplicate = pd.concat(
        [make_default_dataset_frame().iloc[:1], make_default_dataset_frame().iloc[:1]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "duplicate"),
            duplicate,
            spec=make_processed_dataset_spec("duplicate"),
        )

    null_key = make_default_dataset_frame()
    null_key.loc[0, "asset_id"] = None
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "null_key"),
            null_key,
            spec=make_processed_dataset_spec("null_key"),
        )

    infinite = make_default_dataset_frame()
    infinite.loc[0, "value"] = -math.inf
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "infinite"),
            infinite,
            spec=make_processed_dataset_spec("infinite"),
        )

    missing_required = make_default_dataset_frame().drop(columns=["value"])
    missing_root = make_dataset_artifact_root(tmp_path, "missing_required")
    with pytest.raises(ValueError, match="missing_required_columns: value"):
        write_dataset(
            missing_root,
            missing_required,
            spec=make_processed_dataset_spec("missing_required"),
        )
    assert not missing_root.exists()


def test_write_dataset_rejects_existing_root_without_overwrite(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    original = read_dataset(root)

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(
            root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
        )

    pd.testing.assert_frame_equal(read_dataset(root), original)


def test_write_dataset_rejects_generic_tmp_artifact_root_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="invalid_processed_dataset_root"):
        write_dataset(
            tmp_path / "data" / "processed" / "a_share" / "factor" / ".tmp-user",
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec(),
        )


def test_write_dataset_allows_writer_owned_tmp_artifact_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / ".x.tmp-user"

    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )

    assert inspect_dataset(root).spec["data_id"] == "x"


def test_read_dataset_materialized_chunked_projection_and_time_window(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )

    materialized = read_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2014-01-01", "2014-12-31"),
    )
    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))

    assert list(materialized.columns) == ["asset_id", "value"]
    assert materialized["asset_id"].tolist() == ["asset_001"]
    assert len(chunks) == 2
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)


def test_read_dataset_reuses_one_candidate_file_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "read_plan_reuse")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("read_plan_reuse"),
    )
    calls = count_candidate_data_file_calls(monkeypatch)

    read_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2015-01-01", "2015-01-05"),
    )

    assert calls["count"] == 1


def test_fingerprint_dataset_reuses_one_candidate_file_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint_plan_reuse")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("fingerprint_plan_reuse"),
    )
    calls = count_candidate_data_file_calls(monkeypatch)

    fingerprint_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2015-01-01", "2015-01-05"),
    )

    assert calls["count"] == 1


def test_read_dataset_manifest_failures_are_closed(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )

    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_schema_hash_mismatch"):
        read_dataset(root)


@pytest.mark.parametrize(
    ("relative_path", "replacement", "error"),
    (
        ("spec.json", {}, "dataset_spec_schema_id_invalid"),
        (
            "spec.json",
            {"schema_id": ""},
            "dataset_spec_schema_id_invalid",
        ),
        (
            "spec.json",
            {"schema_id": "qrp-dataset-spec/0.1"},
            "dataset_spec_schema_id_invalid",
        ),
        (
            "spec.json",
            {
                "schema_id": "qrp-dataset-spec",
                "metadata": {"schema_id": "nested/version"},
            },
            "dataset_spec_schema_id_invalid",
        ),
        ("_manifest/manifest.json", {}, "dataset_manifest_schema_id_invalid"),
        (
            "_manifest/manifest.json",
            {"schema_id": ""},
            "dataset_manifest_schema_id_invalid",
        ),
        (
            "_manifest/manifest.json",
            {"schema_id": "qrp-dataset-manifest/0.1"},
            "dataset_manifest_schema_id_invalid",
        ),
        (
            "_manifest/manifest.json",
            {
                "schema_id": "qrp-dataset-manifest",
                "metadata": {"schema_id": "nested/version"},
            },
            "dataset_manifest_schema_id_invalid",
        ),
    ),
)
def test_read_dataset_rejects_invalid_qrp_json_schema_id(
    tmp_path: Path,
    relative_path: str,
    replacement: dict[str, str],
    error: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    path = root / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema_id", None)
    payload.update(replacement)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        read_dataset(root)


def test_read_dataset_manifest_columns_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["columns"] = ["market_id", "asset_id", "timestamp", "bogus"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_columns_mismatch"):
        read_dataset(root)


def test_read_dataset_manifest_row_count_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = int(manifest["row_count"]) + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_row_count_mismatch"):
        read_dataset(root)


def test_read_dataset_rejects_data_root_missing_and_missing_projection_columns(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )

    with pytest.raises(ValueError, match="dataset_column_not_found"):
        read_dataset(root, columns=["asset_id", "missing"])

    data_root = root / "data"
    renamed = root / "data_missing"
    data_root.rename(renamed)
    with pytest.raises(ValueError, match="dataset_data_dir_missing"):
        read_dataset(root)


def test_inspect_dataset_returns_expanded_metadata(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )

    inspection = inspect_dataset(root)

    assert inspection.root == root
    assert inspection.data_root == root / "data"
    assert inspection.spec["data_id"] == "x"
    assert inspection.spec["kind"] == "factor"
    assert inspection.columns == ("market_id", "asset_id", "timestamp", "value")
    assert inspection.schema_hash
    assert inspection.file_count == 2
    assert inspection.row_count == 3
    assert inspection.partitions == ("year=2014", "year=2015")
    assert inspection.partition_row_counts == {"year=2014": 1, "year=2015": 2}
    assert inspection.date_min == "2014-01-02"
    assert inspection.date_max == "2015-01-05"
    assert not hasattr(inspection, "schema")


def test_write_dataset_preserves_factor_and_label_nan_profiles(tmp_path: Path) -> None:
    factor_root = make_dataset_artifact_root(tmp_path, "factor_nan")
    label_root = make_dataset_artifact_root(
        tmp_path,
        "label_x",
        kind=ProcessedDataKind.LABEL,
    )
    label_frame = make_default_dataset_frame().rename(columns={"value": "label"})

    write_dataset(
        factor_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("factor_nan"),
    )
    write_dataset(
        label_root,
        label_frame,
        spec=make_processed_dataset_spec(
            "label_x",
            kind=ProcessedDataKind.LABEL,
            value_column="label",
        ),
    )

    factor_manifest = json.loads(
        (factor_root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    label_manifest = json.loads(
        (label_root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert factor_manifest["factor_nan_profile"] == {"value": 1}
    assert factor_manifest["label_nan_profile"] == {}
    assert label_manifest["factor_nan_profile"] == {}
    assert label_manifest["label_nan_profile"] == {"label": 1}


def test_inspect_and_fingerprint_detect_schema_mismatch_across_parts(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    bad_part = root / "data" / "year=2016" / "part-000.parquet"
    bad_part.parent.mkdir(parents=True)
    make_default_dataset_frame().assign(extra=1).to_parquet(bad_part, index=False)

    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        inspect_dataset(root)
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        fingerprint_dataset(root)


def test_compare_datasets_statuses_for_semantic_and_schema_differences(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path, "left")
    right_root = make_dataset_artifact_root(tmp_path, "right")
    schema_root = make_dataset_artifact_root(tmp_path, "schema")
    write_dataset(
        left_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("left"),
    )
    write_dataset(
        right_root,
        iter(
            [
                make_default_dataset_frame().iloc[:1],
                make_default_dataset_frame().iloc[1:],
            ]
        ),
        spec=make_processed_dataset_spec("right"),
    )

    comparison = compare_datasets(left_root, right_root)
    assert comparison.status == "MATCH"
    assert comparison.matches
    assert comparison.layout_delta["file_count"] == {"left": 2, "right": 3}

    wider_spec = ProcessedDataSpec(
        data_id="schema",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **make_processed_dataset_spec().column_roles,
            "other": ColumnRole.FACTOR,
        },
        primary_key=make_processed_dataset_spec().primary_key,
        partition_columns=("year",),
    )
    write_dataset(
        schema_root, make_default_dataset_frame().assign(other=1.0), spec=wider_spec
    )

    schema_comparison = compare_datasets(left_root, schema_root)
    assert schema_comparison.status == "SCHEMA_MISMATCH"
    assert not schema_comparison.matches


def test_compare_datasets_ignores_allowed_instance_metadata_for_same_content(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path, "left_metadata")
    right_root = make_dataset_artifact_root(tmp_path, "right_metadata")
    write_dataset(
        left_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("left_metadata"),
    )
    write_dataset(
        right_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("right_metadata"),
        instance_metadata={
            "input_fingerprints": {"demo_input": _input_fingerprint_payload()}
        },
    )

    comparison = compare_datasets(left_root, right_root)

    assert comparison.status == "MATCH"
    assert comparison.matches


def test_storage_datasets_public_api_is_hard_cutover() -> None:
    import quant_research_platform.storage as storage

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
    assert not hasattr(datasets, "_dataset_planning_metadata")
    assert not hasattr(storage, "write_dataset")


def test_storage_datasets_public_signatures_do_not_expose_physical_controls() -> None:
    public_signatures = {
        "write_dataset": signature(write_dataset),
        "read_dataset": signature(read_dataset),
        "inspect_dataset": signature(inspect_dataset),
        "fingerprint_dataset": signature(fingerprint_dataset),
        "compare_datasets": signature(compare_datasets),
    }

    assert list(public_signatures["write_dataset"].parameters) == [
        "dataset_root",
        "data",
        "spec",
        "instance_metadata",
    ]
    assert list(public_signatures["read_dataset"].parameters) == [
        "dataset_root",
        "columns",
        "time_window",
        "mode",
    ]
    assert list(public_signatures["inspect_dataset"].parameters) == [
        "dataset_root",
        "time_window",
    ]
    assert list(public_signatures["fingerprint_dataset"].parameters) == [
        "dataset_root",
        "columns",
        "time_window",
    ]
    assert list(public_signatures["compare_datasets"].parameters) == [
        "left_root",
        "right_root",
        "columns",
        "time_window",
    ]
