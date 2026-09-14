from __future__ import annotations

import gc
import json
import struct
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.storage.datasets.fingerprinting as dataset_fingerprinting
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_payload_from_frame,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.dataset_test_support import (
    _input_dataset_identity_payload,
    _persisted_table,
)
from tests.storage.low_level_fixtures import (
    fixture_dataset_content_hash,
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


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
    assert chunks
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)


def test_internal_physical_reader_uses_bounded_record_batches(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path,
        "bounded_reader",
        market_id="dummy",
    )
    row_count = 4_096
    frame = pd.DataFrame(
        {
            "market_id": ["dummy"] * row_count,
            "asset_id": [f"asset_{index:04d}" for index in range(row_count)],
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="min"),
            "value": [float(index) for index in range(row_count)],
        }
    )
    ordered_input = pd.concat(
        (frame.iloc[row_count // 2 :].copy(), frame.iloc[: row_count // 2].copy()),
        ignore_index=True,
    )
    ordered_input["timestamp"] = ordered_input["timestamp"].astype("datetime64[ns]")
    write_dataset(
        root,
        (
            frame.iloc[row_count // 2 :].copy(),
            frame.iloc[: row_count // 2].copy(),
        ),
        spec=make_processed_dataset_spec("bounded_reader", market_id="dummy"),
    )
    target_chunk_bytes = 512 * 1024
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as context:
        chunks = list(
            dataset_reading.iter_dataset_chunks(
                context,
                target_chunk_bytes=target_chunk_bytes,
            )
        )

    assert len(chunks) > 1
    assert all(
        pandas_frame_payload_bytes(chunk) <= target_chunk_bytes for chunk in chunks
    )
    pd.testing.assert_frame_equal(
        pd.concat(chunks, ignore_index=True),
        ordered_input,
    )

    window = TimeWindow(
        frame.loc[28, "timestamp"],
        frame.loc[36, "timestamp"],
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=("asset_id", "value"),
        time_window=window,
    ) as projected_context:
        projected = pd.concat(
            dataset_reading.iter_dataset_chunks(
                projected_context,
                target_chunk_bytes=target_chunk_bytes,
            ),
            ignore_index=True,
        )
    expected_projected = ordered_input.loc[
        ordered_input["timestamp"].between(
            pd.Timestamp(window.start),
            pd.Timestamp(window.end),
        ),
        ["asset_id", "value"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(projected, expected_projected)


def test_internal_physical_reader_allows_single_row_over_soft_target(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path,
        "oversized_reader_row",
        market_id="dummy",
    )
    frame = pd.DataFrame(
        {
            "market_id": ["dummy"],
            "asset_id": ["asset_a"],
            "timestamp": [pd.Timestamp("2026-01-01")],
            "value": ["x" * 8_192],
        }
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(
            "oversized_reader_row",
            market_id="dummy",
        ),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as context:
        chunks = list(
            dataset_reading.iter_dataset_chunks(
                context,
                target_chunk_bytes=512,
            )
        )
    expected = frame.copy()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), expected)


def test_bounded_dataset_read_path_never_calls_full_file_pandas_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "no_full_file_reader")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("no_full_file_reader"),
    )

    def fail_read_parquet(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded_dataset_read_must_not_call_pd_read_parquet")

    monkeypatch.setattr(dataset_reading.pd, "read_parquet", fail_read_parquet)

    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))

    assert sum(len(chunk) for chunk in chunks) == 3


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
            "dataset_spec_fields_invalid",
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
            "dataset_manifest_fields_invalid",
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("row_count", True, "manifest_row_count_invalid"),
        ("row_count", "3", "manifest_row_count_invalid"),
        ("row_count", 3.0, "manifest_row_count_invalid"),
        ("file_count", True, "manifest_file_count_invalid"),
        (
            "columns",
            ["market_id", "asset_id", "timestamp", "value", "value"],
            "manifest_columns_invalid",
        ),
        ("schema_hash", "A" * 64, "manifest_schema_hash_invalid"),
        ("partition_count", True, "manifest_partition_count_invalid"),
        (
            "partition_row_counts",
            {"year=2014": True, "year=2015": 2},
            "manifest_partition_row_counts_invalid",
        ),
        (
            "partitions",
            ["year=2014", "year=2015", "year=2015"],
            "manifest_partitions_invalid",
        ),
        ("date_min", "2016-01-01", "manifest_date_range_invalid"),
        ("validation_status", "pass", "manifest_validation_status_invalid"),
        ("unknown", "value", "dataset_manifest_fields_invalid"),
    ),
)
def test_read_dataset_rejects_strict_manifest_type_and_shape_mutations(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        read_dataset(root)


def test_read_dataset_rejects_forged_recursive_physical_schema_identity(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["physical_schema"][0]["nullable"] = not manifest["physical_schema"][0][
        "nullable"
    ]
    manifest["schema_hash"] = canonical_json_hash(manifest["physical_schema"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_physical_schema_mismatch"):
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
    bad_frame = make_default_dataset_frame().assign(extra=1)
    dataset_writing.write_parquet_file(
        _persisted_table(bad_frame),
        bad_part,
    )

    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_hash"] = fixture_dataset_content_hash(root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

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
        (
            make_default_dataset_frame().iloc[:1],
            make_default_dataset_frame().iloc[1:],
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
            "input_dataset_identities": {
                "demo_input": _input_dataset_identity_payload()
            }
        },
    )

    comparison = compare_datasets(left_root, right_root)

    assert comparison.status == "MATCH"
    assert comparison.matches


def test_empty_materialized_and_chunked_reads_preserve_stored_dtypes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_dtypes")
    expected = make_default_dataset_frame().iloc[:0].copy()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    expected["asset_id"] = pd.Categorical(
        expected["asset_id"],
        categories=["asset_001", "asset_002"],
    )
    write_dataset(
        root,
        expected,
        spec=make_processed_dataset_spec("empty_dtypes"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        materialized = dataset_reading.read_dataset(
            root,
            mode=DatasetReadMode.MATERIALIZED,
            resource_policy=policy,
        )
        chunked = list(
            dataset_reading.read_dataset(
                root,
                mode=DatasetReadMode.CHUNKED,
                resource_policy=policy,
            )
        )

    assert len(chunked) == 1
    assert (
        materialized.dtypes.astype(str).to_dict()
        == expected.dtypes.astype(str).to_dict()
    )
    assert (
        chunked[0].dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()
    )


def test_time_window_filter_is_applied_before_arrow_to_pandas_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "physical_time_filter")
    row_count = 1_024
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": ["asset_a"] * row_count,
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="min"),
            "value": np.arange(row_count, dtype="float64"),
        }
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("physical_time_filter"),
    )
    converted_rows = 0
    original = dataset_reading._arrow_batch_to_bounded_pandas

    def record_rows(batch: pa.RecordBatch, **kwargs):
        nonlocal converted_rows
        converted_rows += batch.num_rows
        yield from original(batch, **kwargs)

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_batch_to_bounded_pandas",
        record_rows,
    )
    target = frame.loc[900, "timestamp"]

    result = read_dataset(
        root,
        time_window=TimeWindow(target.to_pydatetime(), target.to_pydatetime()),
    )

    assert len(result) == 1
    assert converted_rows <= 2


def test_time_window_row_group_pruning_uses_parquet_leaf_path(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "nested_before_timestamp")
    spec = ProcessedDataSpec(
        data_id="nested_before_timestamp",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "context": ColumnRole.CONTEXT,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    target = pd.Timestamp("2026-01-02")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "context": [
                {
                    "first": pd.Timestamp("2000-01-01"),
                    "second": pd.Timestamp("2000-01-02"),
                }
            ],
            "timestamp": [target],
            "value": [1.0],
        }
    )
    write_dataset(root, frame, spec=spec)

    result = read_dataset(
        root,
        time_window=TimeWindow(target.to_pydatetime(), target.to_pydatetime()),
    )

    assert len(result) == 1
    assert result.loc[0, "timestamp"] == target


def test_fingerprint_hashes_ordinary_frame_without_normalized_copy() -> None:
    frame = pd.DataFrame({"value": np.arange(20_000, dtype="int64")})

    observed = tuple(
        dataset_fingerprinting._iter_fingerprint_hashes(
            frame,
            target_bytes=pandas_frame_payload_bytes(frame) + len(frame) * 8,
        )
    )

    assert len(observed) == 1
    np.testing.assert_array_equal(
        observed[0],
        pd.util.hash_pandas_object(frame, index=False).to_numpy("uint64"),
    )


@pytest.mark.parametrize(
    ("data_type", "values"),
    (
        (pa.list_(pa.int64()), [[1], [2]]),
        (pa.map_(pa.string(), pa.int64()), [[("a", 1)], [("b", 2)]]),
    ),
)
def test_fingerprint_rejects_generalized_nested_values_before_hashing(
    data_type: pa.DataType,
    values: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": values})
    schema = pa.schema([("value", data_type)])
    monkeypatch.setattr(
        pd.util,
        "hash_pandas_object",
        lambda *_args, **_kwargs: pytest.fail("nested_value_was_hashed"),
    )

    with pytest.raises(ValueError, match="unsupported_dataset_fingerprint_value"):
        tuple(
            dataset_fingerprinting._iter_fingerprint_hashes(
                frame,
                target_bytes=pandas_frame_payload_bytes(frame) + 1024,
                arrow_schema=schema,
            )
        )


@pytest.mark.parametrize("data_type", (pa.float32(), pa.float64()))
def test_fingerprint_rejects_float_primary_keys_before_hashing(
    data_type: pa.DataType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"asset_id": [1.0, 2.0]})
    schema = pa.schema([("asset_id", data_type)])
    monkeypatch.setattr(
        pd.util,
        "hash_pandas_object",
        lambda *_args, **_kwargs: pytest.fail("float_primary_key_was_hashed"),
    )

    with pytest.raises(ValueError, match="unsupported_dataset_fingerprint_value"):
        tuple(
            dataset_fingerprinting._iter_fingerprint_hashes(
                frame,
                target_bytes=pandas_frame_payload_bytes(frame) + 1024,
                arrow_schema=schema,
                primary_key=("asset_id",),
            )
        )


def test_fingerprint_hashes_large_primitive_frame_in_bounded_slices() -> None:
    frame = pd.DataFrame({"value": np.arange(20_000, dtype="int64")})
    target_bytes = pandas_frame_payload_bytes(frame) + 8 * 1_000

    observed = tuple(
        dataset_fingerprinting._iter_fingerprint_hashes(
            frame,
            target_bytes=target_bytes,
        )
    )

    assert len(observed) == 20
    np.testing.assert_array_equal(
        np.concatenate(observed),
        pd.util.hash_pandas_object(frame, index=False).to_numpy("uint64"),
    )


def test_fingerprint_releases_hash_array_before_next_source_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": np.arange(16, dtype="int64")})
    arrow_batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    paths = (tmp_path / "data" / "part-0.parquet", tmp_path / "data" / "part-1.parquet")
    hash_references: list[weakref.ReferenceType[np.ndarray]] = []
    released_before_next_batch: list[bool] = []

    def batches(*_args: object, **kwargs: object):
        observer = kwargs["batch_observer"]
        observer(arrow_batch, frame)
        yield paths[0], frame
        gc.collect()
        released_before_next_batch.append(
            bool(hash_references) and hash_references[-1]() is None
        )
        observer(arrow_batch, frame)
        yield paths[1], frame

    def hashes(current: pd.DataFrame, **_kwargs: object):
        values = np.arange(len(current), dtype="uint64")
        hash_references.append(weakref.ref(values))
        yield values
        del values

    monkeypatch.setattr(dataset_fingerprinting, "iter_dataset_file_batches", batches)
    monkeypatch.setattr(dataset_fingerprinting, "_iter_fingerprint_hashes", hashes)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    context = SimpleNamespace(
        resource_policy=policy,
        spec={"primary_key": ["value"]},
        logical_columns=("value",),
        arrow_schema=arrow_batch.schema,
        logical_dtypes=logical_dtype_payload_from_frame(frame),
        time_window=None,
        data_root=tmp_path / "data",
        schema_hash="schema",
        read_paths=paths,
        inventory=SimpleNamespace(retained_memory_bytes=0),
        retained_memory_bytes=0,
    )
    context.physical_columns = context.logical_columns
    context.spec_bytes = json.dumps(context.spec).encode("utf-8")
    context.manifest_bytes = b"{}"

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context,
        resource_policy=policy,
    )

    assert fingerprint.row_count == 2 * len(frame)
    assert released_before_next_batch == [True]


def test_fingerprint_rejects_exhausted_operation_headroom_before_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    traversals = 0

    def forbidden_traversal(*_args: object, **_kwargs: object):
        nonlocal traversals
        traversals += 1
        raise AssertionError("fingerprint_traversed_without_operation_headroom")

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        forbidden_traversal,
    )
    context = SimpleNamespace(
        resource_policy=policy,
        inventory=SimpleNamespace(retained_memory_bytes=policy.memory_budget_bytes),
        retained_memory_bytes=policy.memory_budget_bytes,
        read_paths=(),
    )
    context.logical_columns = context.physical_columns = ("id",)
    context.spec_bytes = b'{"primary_key":["id"]}'
    context.manifest_bytes = b"{}"

    with runtime_resource_policy_scope(policy):
        retained_lease = policy.lease(
            "test.fingerprint.retained",
            policy.memory_budget_bytes,
        )
        try:
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                dataset_fingerprinting.fingerprint_dataset_session(
                    context,
                    resource_policy=policy,
                )
        finally:
            retained_lease.close()

    assert traversals == 0


def test_fingerprint_reserves_many_partition_counters_before_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=28_000,
    )
    traversals = 0

    def forbidden_traversal(*_args: object, **_kwargs: object):
        nonlocal traversals
        traversals += 1
        raise AssertionError("fingerprint_traversed_without_partition_reserve")

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        forbidden_traversal,
    )
    data_root = tmp_path / "data"
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        data_root=data_root,
        read_paths=tuple(
            data_root / f"partition={index:04d}" / "part-000.parquet"
            for index in range(200)
        ),
    )
    context.logical_columns = context.physical_columns = ("id",)
    context.spec_bytes = b'{"primary_key":["id"]}'
    context.manifest_bytes = b"{}"

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_fingerprinting.fingerprint_dataset_session(
            context,
            resource_policy=policy,
        )

    assert traversals == 0


def test_fingerprint_rejects_many_partition_output_overlap_after_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    data_root = tmp_path / "data"
    paths = tuple(
        data_root / f"partition={index:03d}" / "part-000.parquet"
        for index in range(100)
    )
    frame = pd.DataFrame({"id": [1]})
    batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    traversals = 0

    def batches(*_args: object, **kwargs: object):
        nonlocal traversals
        observer = kwargs["batch_observer"]
        for path in paths:
            traversals += 1
            observer(batch, frame)
            yield path, frame

    monkeypatch.setattr(dataset_fingerprinting, "iter_dataset_file_batches", batches)
    context = SimpleNamespace(
        resource_policy=policy,
        data_root=data_root,
        read_paths=paths,
        spec={"primary_key": ["id"]},
        logical_columns=("id",),
        arrow_schema=batch.schema,
        time_window=None,
        schema_hash="schema",
    )
    context.physical_columns = context.logical_columns
    context.spec_bytes = json.dumps(context.spec).encode("utf-8")
    context.manifest_bytes = b"{}"
    partition_state_bytes = dataset_reading.dataset_partition_state_reserve_bytes(
        context
    )
    local_state_bytes = 3 * (struct.calcsize("P") + len(b"id"))
    retained_bytes = policy.memory_budget_bytes - (
        partition_state_bytes + local_state_bytes + 1024
    )

    with runtime_resource_policy_scope(policy):
        retained_lease = policy.lease(
            "test.fingerprint.retained",
            retained_bytes,
        )
        try:
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                dataset_fingerprinting.fingerprint_dataset_session(
                    context,
                    resource_policy=policy,
                )
        finally:
            retained_lease.close()

    assert traversals == 0
