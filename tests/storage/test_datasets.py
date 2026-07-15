from __future__ import annotations

import importlib
import importlib.util
import json
import math
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage._dataset_comparison as dataset_comparison
import quant_research_platform.storage._dataset_fingerprinting as dataset_fingerprinting
import quant_research_platform.storage._dataset_inspection as dataset_inspection
import quant_research_platform.storage._dataset_physical as dataset_physical
import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_schema as dataset_schema
import quant_research_platform.storage._dataset_writing as dataset_writing
import quant_research_platform.storage.atomic as storage_atomic
import quant_research_platform.storage.datasets as datasets
from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_schema import (
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
)
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


def _input_dataset_identity_payload(
    data_id: str = "demo_input",
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "data_id": data_id,
        "spec_hash": "d" * 64,
        "manifest_hash": "e" * 64,
    }
    if extra:
        payload.update(extra)
    return payload


def _persisted_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"qrp.logical-dtypes": logical_dtype_metadata_bytes(
                logical_dtype_payload_from_frame(frame)
            )
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("validation_status", "FAIL"),
        ("duplicate_primary_key_count", 1),
        ("null_count_summary", {"market_id": 1}),
        ("non_finite_count_summary", {"value": 1}),
    ),
)
def test_inspection_recomputes_persisted_summary_instead_of_trusting_manifest(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        read_dataset(root)
    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        fingerprint_dataset(root)

    if field == "validation_status":
        assert inspect_dataset(root).manifest[field] == value
    else:
        with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
            inspect_dataset(root)


@pytest.mark.parametrize(
    "null_count_summary",
    (
        {},
        {"market_id": 0, "asset_id": 0},
    ),
    ids=("empty", "composite-key-member-omitted"),
)
def test_inspection_rejects_missing_primary_key_null_facts(
    tmp_path: Path,
    null_count_summary: dict[str, int],
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["null_count_summary"] = null_count_summary
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        read_dataset(root)
    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        fingerprint_dataset(root)

    with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
        inspect_dataset(root)


@pytest.mark.parametrize(
    "forged_summary",
    (
        {
            "row_count": 4,
            "partition_row_counts": {"year=2014": 2, "year=2015": 2},
        },
        {
            "partition_row_counts": {"year=2014": 2, "year=2015": 1},
        },
        {
            "partitions": ["year=2014", "year=2099"],
            "partition_row_counts": {"year=2014": 1, "year=2099": 2},
        },
        {"file_count": 3},
    ),
    ids=("row-count", "partition-row-counts", "partitions", "file-count"),
)
def test_full_inspection_rejects_forged_persisted_shape_summary(
    tmp_path: Path,
    forged_summary: dict[str, object],
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        require_consumable=False,
    )
    manifest = session.manifest
    manifest.update(forged_summary)
    forged_manifest_bytes = (json.dumps(manifest) + "\n").encode("utf-8")
    (root / "_manifest" / "manifest.json").write_bytes(forged_manifest_bytes)
    forged_session = replace(
        session,
        manifest_bytes=forged_manifest_bytes,
    )

    with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
        dataset_inspection.inspect_dataset_session(
            forged_session,
            resource_policy=active_runtime_resource_policy(),
        )


def test_session_consuming_inspection_does_not_reopen_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        require_consumable=False,
    )
    monkeypatch.setattr(
        dataset_inspection,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("session_inspection_reopened_dataset")
        ),
    )

    inspection = dataset_inspection.inspect_dataset_session(
        session,
        resource_policy=active_runtime_resource_policy(),
    )

    assert inspection.root == root
    assert inspection.row_count == 3
    assert inspection.file_count == 2


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


def test_write_dataset_rejects_parallel_input_fingerprint_lineage(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported_dataset_instance_metadata_key: input_fingerprints",
    ):
        write_dataset(
            make_dataset_artifact_root(tmp_path),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec(),
            instance_metadata={"input_fingerprints": {}},
        )


def test_write_dataset_merges_exact_input_dataset_identities(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "identity_metadata")
    identities = {"demo_input": _input_dataset_identity_payload()}

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("identity_metadata"),
        instance_metadata={"input_dataset_identities": identities},
    )

    inspection = inspect_dataset(root)
    assert inspection.manifest["input_dataset_identities"] == identities


@pytest.mark.parametrize(
    "identities",
    [
        [],
        {"demo_input": _input_dataset_identity_payload(extra={"unexpected": "x"})},
        {"demo_input": _input_dataset_identity_payload(data_id="other_input")},
        {"demo_input": _input_dataset_identity_payload(extra={"spec_hash": 1})},
        {"demo_input": _input_dataset_identity_payload(extra={"manifest_hash": "x"})},
        {1: _input_dataset_identity_payload()},
    ],
)
def test_write_dataset_rejects_malformed_input_dataset_identities(
    tmp_path: Path,
    identities: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_input_dataset_identities_metadata"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "bad_identity_metadata"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("bad_identity_metadata"),
            instance_metadata={"input_dataset_identities": identities},
        )


def test_inspect_dataset_rejects_tampered_input_dataset_identity(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tampered_identity_metadata")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tampered_identity_metadata"),
        instance_metadata={
            "input_dataset_identities": {
                "demo_input": _input_dataset_identity_payload()
            }
        },
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_dataset_identities"]["demo_input"]["unknown"] = "x"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid_input_dataset_identities_metadata"):
        inspect_dataset(root)


def test_inspect_dataset_rejects_parallel_input_fingerprint_lineage(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "parallel_fingerprint_lineage")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("parallel_fingerprint_lineage"),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_fingerprints"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="dataset_manifest_fields_invalid",
    ):
        inspect_dataset(root)


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


@pytest.mark.parametrize(
    "equivalent_asset_ids",
    [
        pd.Series([1, 1.0], dtype="object"),
        pd.Series([np.int64(1), 1], dtype="object"),
        pd.Series([date(2026, 1, 1), datetime(2026, 1, 1)], dtype="object"),
    ],
    ids=["int-float", "numpy-python", "date-datetime"],
)
def test_write_dataset_rejects_keys_equal_after_arrow_conversion(
    tmp_path: Path,
    equivalent_asset_ids: pd.Series,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "arrow_key_identity")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": equivalent_asset_ids,
            "timestamp": [pd.Timestamp("2026-01-02")] * 2,
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("arrow_key_identity"),
        )


def test_writer_key_index_scratch_is_inside_staged_dataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_database_paths: list[Path] = []
    original_tracker = dataset_writing.ExactPrimaryKeyIndex

    class RecordingTracker(original_tracker):
        def add_table(self, table: pa.Table, *, namespace: str = "dataset") -> int:
            observed_database_paths.append(self.database_path)
            return super().add_table(table, namespace=namespace)

    monkeypatch.setattr(dataset_writing, "ExactPrimaryKeyIndex", RecordingTracker)
    root = make_dataset_artifact_root(tmp_path, "staged_key_scratch")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("staged_key_scratch"),
    )

    assert observed_database_paths
    assert all(
        path.is_relative_to(root.parent)
        and path.parts[len(root.parent.parts)].startswith(
            storage_atomic._atomic_directory_scratch_prefix(root)
        )
        for path in observed_database_paths
    )
    assert not any(path.exists() for path in observed_database_paths)
    assert not list(root.rglob(".owner.start"))










def test_single_short_string_key_row_uses_measured_conversion_size() -> None:
    batch = pa.record_batch(
        [
            pa.array(["a_share"], type=pa.large_string()),
            pa.array(["000001"], type=pa.large_string()),
            pa.array([pd.Timestamp("2014-01-02")], type=pa.timestamp("ns")),
        ],
        names=["market_id", "asset_id", "timestamp"],
    )

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=4_096,
        )
    )

    assert len(chunks) == 1
    pd.testing.assert_frame_equal(chunks[0], batch.to_pandas(use_threads=False))


def test_declared_variable_width_member_decodes_multirow_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "declared-member.parquet"
    row_count = 5_000
    pq.write_table(
        pa.table({"value": [f"value-{index:05d}" for index in range(row_count)]}),
        path,
    )
    conversions = 0
    original = dataset_reading._arrow_slice_to_pandas

    def record_conversion(batch: pa.RecordBatch) -> pd.DataFrame:
        nonlocal conversions
        conversions += 1
        return original(batch)

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )
    frames = tuple(
        dataset_reading.iter_declared_parquet_member_batches(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            target_chunk_bytes=64 * 1024,
        )
    )

    assert sum(len(frame) for frame in frames) == row_count
    assert conversions < row_count // 10
    assert all(
        pandas_frame_payload_bytes(frame) <= 64 * 1024 for frame in frames
    )


def test_chunk_iterator_verifies_session_on_close_empty_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "iterator_final_verify")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("iterator_final_verify"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    calls = 0
    original_verify = dataset_reading.DatasetSession.verify

    def record_verify(current: dataset_reading.DatasetSession) -> None:
        nonlocal calls
        if current.root == root:
            calls += 1
        original_verify(current)

    monkeypatch.setattr(dataset_reading.DatasetSession, "verify", record_verify)

    iterator = dataset_reading.iter_dataset_chunks(
        session,
        target_chunk_bytes=1024**2,
    )
    next(iterator)
    before_close = calls
    iterator.close()
    assert calls > before_close

    empty_session = replace(session, read_paths=())
    before_empty = calls
    assert len(
        list(
            dataset_reading.iter_dataset_chunks(
                empty_session,
                target_chunk_bytes=1024**2,
            )
        )
    ) == 1
    assert calls > before_empty

    def fail_batches(*_args, **_kwargs):
        raise RuntimeError("injected_read_failure")
        yield  # pragma: no cover

    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_file_batches",
        fail_batches,
    )
    before_error = calls
    with pytest.raises(RuntimeError, match="injected_read_failure"):
        list(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=1024**2,
            )
        )
    assert calls > before_error


@pytest.mark.parametrize(
    "arrays",
    [
        [pa.nulls(1) for _ in range(500)],
        [pa.array([b"x"], type=pa.binary(1)) for _ in range(500)],
        [
            pa.array([Decimal("1.23")], type=pa.decimal128(10, 2))
            for _ in range(500)
        ],
        [pa.array([date(2026, 1, 1)], type=pa.date32()) for _ in range(500)],
    ],
    ids=["null", "fixed-binary", "decimal", "date"],
)
def test_wide_single_row_fails_after_actual_pandas_payload_check(
    arrays: list[pa.Array],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        arrays,
        names=[f"column_{index}" for index in range(len(arrays))],
    )
    calls = {"count": 0}
    convert = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        return convert(arrow_slice)

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=4_096,
            )
        )

    assert calls["count"] == 1






def test_large_map_row_uses_only_approved_pandas_payload_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        [
            pa.array(
                [[(f"key_{index}", None) for index in range(5_000)]],
                type=pa.map_(pa.string(), pa.null()),
            )
        ],
        names=["payload"],
    )
    calls = {"count": 0}
    convert = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        return convert(arrow_slice)

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=256 * 1024,
        )
    )

    assert calls["count"] == 1
    assert max(pandas_frame_payload_bytes(chunk) for chunk in chunks) <= (
        256 * 1024
    ) // 4


def test_wide_numeric_row_uses_only_approved_pandas_payload_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        [pa.array([1], type=pa.int64()) for _ in range(1_500)],
        names=[f"column_{index}" for index in range(1_500)],
    )
    calls = {"count": 0}
    convert = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        return convert(arrow_slice)

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=64 * 1024,
        )
    )

    assert calls["count"] == 1
    assert max(pandas_frame_payload_bytes(chunk) for chunk in chunks) <= (
        64 * 1024
    ) // 4




def test_materialized_context_fails_before_collecting_all_physical_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yielded = 0
    concat_called = False

    def frames(*_args, **_kwargs):
        nonlocal yielded
        for index in range(4):
            yielded += 1
            yield pd.DataFrame({"value": ["x" * 1_024], "part": [index]})

    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", frames)

    def forbidden_concat(*_args, **_kwargs):
        nonlocal concat_called
        concat_called = True
        raise AssertionError(
            "concat_must_not_run_after_retained_payload_exceeds_budget"
        )

    monkeypatch.setattr(dataset_reading.pd, "concat", forbidden_concat)

    with pytest.raises(
        ValueError,
        match="dataset_materialized_result_exceeds_target_bytes",
    ):
        dataset_reading.materialize_dataset_session(
            SimpleNamespace(),
            target_chunk_bytes=64 * 1024,
            materialized_budget_bytes=2_048,
        )

    assert yielded < 4
    assert concat_called is False


def test_materialized_session_accounts_for_concat_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame({"value": ["payload"]})
    budget = pandas_frame_payload_bytes(expected) * 2

    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((expected,)),
    )

    observed = dataset_reading.materialize_dataset_session(
        SimpleNamespace(verify=lambda: None),
        target_chunk_bytes=budget,
        materialized_budget_bytes=budget,
    )

    pd.testing.assert_frame_equal(observed, expected)


















def test_chunked_and_materialized_reads_share_nullable_numeric_dtype(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "nullable_numeric_dtype")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["value"] = pd.Series([1], dtype="Int64")
    second["value"] = pd.Series([pd.NA], dtype="Int64")

    write_dataset(
        root,
        (first, second),
        spec=make_processed_dataset_spec("nullable_numeric_dtype"),
    )

    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    materialized = read_dataset(root)

    assert {str(chunk["value"].dtype) for chunk in chunks} == {
        str(materialized["value"].dtype)
    }


def test_nullable_uint64_read_and_fingerprint_preserve_large_integer_identity(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path / "left", "nullable_uint64")
    right_root = make_dataset_artifact_root(tmp_path / "right", "nullable_uint64")
    spec = make_processed_dataset_spec("nullable_uint64")
    left = make_default_dataset_frame().iloc[:2].copy()
    right = left.copy()
    left["value"] = pd.array([2**63, pd.NA], dtype="UInt64")
    right["value"] = pd.array([2**63 + 1, pd.NA], dtype="UInt64")

    write_dataset(left_root, left, spec=spec)
    write_dataset(right_root, right, spec=spec)

    left_read = read_dataset(left_root)
    right_read = read_dataset(right_root)
    comparison = compare_datasets(left_root, right_root)

    assert str(left_read["value"].dtype) == "UInt64"
    assert str(right_read["value"].dtype) == "UInt64"
    assert int(left_read["value"].dropna().iloc[0]) == 2**63
    assert int(right_read["value"].dropna().iloc[0]) == 2**63 + 1
    assert comparison.status == "CONTENT_MISMATCH"


def test_nullable_boolean_preserves_extension_dtype_across_read_modes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "nullable_boolean")
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["value"] = pd.array([True, pd.NA], dtype="boolean")
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("nullable_boolean"),
    )

    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    materialized = read_dataset(root)

    assert {str(chunk["value"].dtype) for chunk in chunks} == {"boolean"}
    assert str(materialized["value"].dtype) == "boolean"


@pytest.mark.parametrize(
    "mode",
    [DatasetReadMode.CHUNKED, DatasetReadMode.MATERIALIZED],
)
def test_read_dataset_rejects_empty_column_projection(
    tmp_path: Path,
    mode: DatasetReadMode,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"empty_projection_{mode.value}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"empty_projection_{mode.value}"),
    )

    with pytest.raises(ValueError, match="dataset_columns_empty"):
        read_dataset(root, columns=[], mode=mode)


def test_materialized_read_unions_categorical_dictionaries_across_parts(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "categorical_parts")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["asset_id"] = first["asset_id"].astype("category")
    second["asset_id"] = second["asset_id"].astype("category")
    write_dataset(
        root,
        (first, second),
        spec=make_processed_dataset_spec("categorical_parts"),
    )

    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    materialized = read_dataset(root)

    assert {str(chunk["asset_id"].dtype) for chunk in chunks} == {"category"}
    assert str(materialized["asset_id"].dtype) == "category"
    assert set(materialized["asset_id"].cat.categories) == set(
        pd.concat((first["asset_id"].astype(str), second["asset_id"].astype(str)))
    )
    assert all(
        chunk["asset_id"].dtype == materialized["asset_id"].dtype
        for chunk in chunks
    )




def test_fingerprint_preserves_content_identity_across_unused_categories(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path / "left", "category_identity")
    right_root = make_dataset_artifact_root(tmp_path / "right", "category_identity")
    left = make_default_dataset_frame().iloc[:1].copy()
    right = left.copy()
    value = str(left["asset_id"].iloc[0])
    left["asset_id"] = pd.Categorical(
        left["asset_id"],
        categories=[value, "unused_left"],
    )
    right["asset_id"] = pd.Categorical(
        right["asset_id"],
        categories=[value, "unused_right"],
    )
    spec = make_processed_dataset_spec("category_identity")
    write_dataset(left_root, left, spec=spec)
    write_dataset(right_root, right, spec=spec)

    comparison = compare_datasets(left_root, right_root)

    assert comparison.matches


def test_fingerprint_content_hash_preserves_registered_formula(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint_formula")
    frame = make_default_dataset_frame()
    spec = make_processed_dataset_spec("fingerprint_formula")
    write_dataset(root, frame, spec=spec)

    fingerprint = fingerprint_dataset(root)
    hash_columns = tuple(dict.fromkeys((*spec.primary_key, *spec.required_columns)))
    legacy_readback = pd.concat(
        tuple(
            pd.read_parquet(path)
            for path in sorted((root / "data").rglob("part-*.parquet"))
        ),
        ignore_index=True,
    )
    hashes = pd.util.hash_pandas_object(
        legacy_readback.loc[:, list(hash_columns)],
        index=False,
    ).to_numpy("uint64")
    expected = canonical_json_hash(
        {
            "row_hash_sum": int(hashes.sum(dtype=np.uint64)),
            "row_hash_xor": int(
                np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
            ),
            "row_count": len(frame),
            "columns": list(spec.required_columns),
            "hash_columns": list(hash_columns),
        }
    )

    assert fingerprint.content_hash == expected


def test_writer_rejects_inconsistent_ordered_categorical_domains(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "ordered_categorical_parts")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["asset_id"] = pd.Categorical(
        first["asset_id"],
        categories=[first["asset_id"].iloc[0]],
        ordered=True,
    )
    second["asset_id"] = pd.Categorical(
        second["asset_id"],
        categories=[second["asset_id"].iloc[0]],
        ordered=True,
    )

    with pytest.raises(ValueError, match="dataset_ordered_categorical_mismatch"):
        write_dataset(
            root,
            (first, second),
            spec=make_processed_dataset_spec("ordered_categorical_parts"),
        )

    assert not root.exists()


def test_inspect_and_fingerprint_reuse_physical_batch_reader() -> None:
    inspection_source = Path(
        importlib.import_module(
            "quant_research_platform.storage._dataset_inspection"
        ).__file__
    ).read_text(encoding="utf-8")
    fingerprint_source = Path(
        importlib.import_module(
            "quant_research_platform.storage._dataset_fingerprinting"
        ).__file__
    ).read_text(encoding="utf-8")

    assert "pd.read_parquet" not in inspection_source
    assert "pd.read_parquet" not in fingerprint_source


def test_exact_primary_key_index_is_spillable_exact_and_cleans_up(
    tmp_path: Path,
) -> None:
    module_spec = importlib.util.find_spec(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    assert module_spec is not None, "storage exact key index module is missing"
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    index_root = tmp_path / "index"
    index_root.mkdir()
    index = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id", "timestamp"),
        timestamp_columns=("timestamp",),
        scratch_parent=index_root,
    )
    first = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00", "2026-01-01T00:01:00"]
            ),
        }
    )
    duplicate_same_instant = pd.DataFrame(
        {
            "asset_id": ["asset_a"],
            "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")],
        }
    )

    with index:
        database_path = index.database_path
        assert database_path.is_file()
        assert index.add_frame(first) == 0
        assert index.add_frame(duplicate_same_instant) == 1
        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").write_bytes(b"scratch")

    assert not database_path.exists()
    assert list(index_root.iterdir()) == []

    numeric = module.ExactPrimaryKeyIndex(
        primary_key=("numeric_key",),
        timestamp_columns=(),
        scratch_parent=index_root,
    )
    with numeric:
        assert numeric.add_frame(pd.DataFrame({"numeric_key": [-0.0]})) == 0
        assert numeric.add_frame(pd.DataFrame({"numeric_key": [0.0]})) == 1
    assert list(index_root.iterdir()) == []

    exceptional = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
        scratch_parent=index_root,
    )
    with pytest.raises(RuntimeError, match="stop"):
        with exceptional:
            exceptional_path = exceptional.database_path
            raise RuntimeError("stop")
    assert not exceptional_path.exists()
    assert list(index_root.iterdir()) == []


def test_exact_key_index_cleans_scratch_when_sqlite_connect_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    index = module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    )

    def fail_connect(*_args, **_kwargs):
        raise module.sqlite3.OperationalError("synthetic_connect_failure")

    monkeypatch.setattr(module.sqlite3, "connect", fail_connect)

    with pytest.raises(module.sqlite3.OperationalError):
        index.__enter__()

    assert list(tmp_path.iterdir()) == []


def test_exact_key_index_inserts_unique_rows_from_duplicate_chunk(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert index.add_frame(pd.DataFrame({"key": ["a", "a", "b"]})) == 1
        assert index.add_frame(pd.DataFrame({"key": ["b"]})) == 1


def test_exact_key_index_keeps_sqlite_temp_state_in_memory(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert index._connection is not None
        temp_store = int(
            index._connection.execute("PRAGMA temp_store").fetchone()[0]
        )
        index.add_frame(pd.DataFrame({"key": ["a", "b"]}), namespace="left")
        index.add_frame(pd.DataFrame({"key": ["b", "c"]}), namespace="right")

        assert temp_store == 2
        assert index.difference_counts(
            expected_namespace="left",
            actual_namespace="right",
        ) == (1, 1)

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert " EXCEPT " not in source


def test_exact_primary_key_index_uses_target_arrow_scalar_semantics(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", pa.float64())]),
    ) as numeric:
        assert numeric.add_frame(pd.DataFrame({"key": [Decimal("1.0")]})) == 0
        assert numeric.add_frame(pd.DataFrame({"key": [1.0]})) == 1
        assert numeric.add_frame(pd.DataFrame({"key": [-0.0]})) == 0
        assert numeric.add_frame(pd.DataFrame({"key": [0.0]})) == 1
        for infinite_value in (math.inf, -math.inf):
            with pytest.raises(
                ValueError,
                match="invalid_dataset_primary_key_value",
            ):
                numeric.add_frame(pd.DataFrame({"key": [infinite_value]}))
        for null_value in (None, math.nan):
            with pytest.raises(ValueError, match="dataset_primary_key_null"):
                numeric.add_frame(pd.DataFrame({"key": [null_value]}))

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=("key",),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", pa.timestamp("ns"))]),
    ) as timestamps:
        assert timestamps.add_frame(pd.DataFrame({"key": [date(2026, 1, 1)]})) == 0
        assert (
            timestamps.add_frame(
                pd.DataFrame({"key": [datetime(2026, 1, 1)]})
            )
            == 1
        )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", pa.string())]),
    ) as strings:
        assert not hasattr(strings, "key_schema")
        assert strings.add_frame(pd.DataFrame({"key": ["é"]})) == 0
        assert strings.add_frame(pd.DataFrame({"key": ["e\u0301"]})) == 0


@pytest.mark.parametrize(
    "decimal_value",
    (Decimal("Infinity"), Decimal("-Infinity")),
)
def test_exact_primary_key_scalar_rejects_decimal_infinity(
    decimal_value: Decimal,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    scalar = SimpleNamespace(is_valid=True, as_py=lambda: decimal_value)

    with pytest.raises(ValueError, match="invalid_dataset_primary_key_value"):
        module._canonical_arrow_scalar_payload(
            scalar,
            pa.decimal128(38, 0),
        )


def test_exact_primary_key_timestamp_identity_preserves_arrow_type_and_range(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    unit_types = tuple(pa.timestamp(unit) for unit in ("s", "ms", "us", "ns"))
    unit_payloads = {
        module._canonical_arrow_scalar_payload(pa.scalar(1, type=data_type), data_type)
        for data_type in unit_types
    }
    timezone_types = (
        pa.timestamp("s"),
        pa.timestamp("s", tz="UTC"),
        pa.timestamp("s", tz="Asia/Shanghai"),
    )
    timezone_payloads = {
        module._canonical_arrow_scalar_payload(pa.scalar(1, type=data_type), data_type)
        for data_type in timezone_types
    }

    assert len(unit_payloads) == len(unit_types)
    assert len(timezone_payloads) == len(timezone_types)

    out_of_nanosecond_range = 32_503_680_000
    timestamp_type = pa.timestamp("s", tz="UTC")
    table = pa.table({"key": pa.array([out_of_nanosecond_range], type=timestamp_type)})
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", timestamp_type)]),
    ) as timestamps:
        assert timestamps.add_table(table) == 0
        assert timestamps.add_table(table) == 1


def test_dataset_writer_accepts_dictionary_encoded_primary_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_tracking = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    tracked_schemas: list[pa.Schema] = []
    original_add_table = key_tracking.ExactPrimaryKeyIndex.add_table

    def record_schema(index, table, **kwargs):  # noqa: ANN001
        tracked_schemas.append(table.schema.remove_metadata())
        return original_add_table(index, table, **kwargs)

    monkeypatch.setattr(
        key_tracking.ExactPrimaryKeyIndex,
        "add_table",
        record_schema,
    )
    root = make_dataset_artifact_root(tmp_path, "categorical_keys")
    frame = make_default_dataset_frame()
    frame["asset_id"] = frame["asset_id"].astype("category")

    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("categorical_keys"),
    )

    assert sorted(read_dataset(root)["asset_id"].astype(str).tolist()) == sorted(
        frame["asset_id"].astype(str).tolist()
    )
    part = next((root / "data").rglob("part-*.parquet"))
    persisted_key_schema = pa.schema(
        [pq.ParquetFile(part).schema_arrow.field("asset_id")]
    )
    assert tracked_schemas
    assert all(
        pa.schema([schema.field("asset_id")]) == persisted_key_schema
        for schema in tracked_schemas
    )


def test_exact_key_index_recovers_dead_sibling_scratch(tmp_path: Path) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    stale = tmp_path / ".k-qzzzzzz-stale"
    stale.mkdir()
    for name in ("k.sqlite3", "k.sqlite3-journal", "k.sqlite3-wal", "k.sqlite3-shm"):
        (stale / name).write_bytes(b"scratch")

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert not stale.exists()
        assert index.add_frame(pd.DataFrame({"key": ["safe"]})) == 0

    assert list(tmp_path.iterdir()) == []


def test_exact_key_index_preserves_live_pid_named_sibling(tmp_path: Path) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    ownership = importlib.import_module(
        "quant_research_platform.storage._scratch_ownership"
    )
    live = tmp_path / ownership.owned_scratch_name(".k-", "concurrent")
    live.mkdir()

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert live.is_dir()
        assert index.add_frame(pd.DataFrame({"key": ["safe"]})) == 0

    live.rmdir()
    assert list(tmp_path.iterdir()) == []


def test_exact_key_index_removes_reused_pid_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    ownership = importlib.import_module(
        "quant_research_platform.storage._scratch_ownership"
    )
    live_name = ownership.owned_scratch_name(".k-", "live")
    owner_pid = live_name.split("-", 3)[1]
    reused = tmp_path / f".k-{owner_pid}-0-reused"
    reused.mkdir()
    (reused / "k.sqlite3").write_bytes(b"scratch")
    monkeypatch.setattr(
        ownership.psutil,
        "Process",
        lambda _pid: SimpleNamespace(
            create_time=lambda: reused.stat().st_ctime + 60,
            is_running=lambda: True,
        ),
    )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert not reused.exists()
        assert index.add_frame(pd.DataFrame({"key": ["safe"]})) == 0

    assert list(tmp_path.iterdir()) == []


def test_dataset_write_recovers_stale_key_scratch_without_publishing_it(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "stale_key_scratch")
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(root)
    stale_stage = root.parent / f"{scratch_prefix}qzzzzzz-stale"
    stale_scratch = stale_stage / ".k-qzzzzzz-stale"
    stale_scratch.mkdir(parents=True)
    Path(f"{stale_stage}.target").write_text(
        storage_atomic._atomic_target_identity(root),
        encoding="utf-8",
    )
    for name in (
        "k.sqlite3",
        "k.sqlite3-journal",
        "k.sqlite3-wal",
        "k.sqlite3-shm",
    ):
        (stale_scratch / name).write_bytes(b"scratch")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("stale_key_scratch"),
    )

    assert not stale_stage.exists()
    assert not stale_scratch.exists()
    assert not list(root.rglob("*.sqlite3"))
    assert not list(root.rglob("*.sqlite3-*"))


def test_dataset_write_does_not_remove_live_concurrent_key_scratch(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "live_key_scratch")
    ownership = importlib.import_module(
        "quant_research_platform.storage._scratch_ownership"
    )
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(root)
    live_stage = root.parent / ownership.owned_scratch_name(
        scratch_prefix,
        "concurrent",
    )
    live_scratch = live_stage / ".k-concurrent"
    live_scratch.mkdir(parents=True)
    Path(f"{live_stage}.target").write_text(
        storage_atomic._atomic_target_identity(root),
        encoding="utf-8",
    )
    (live_scratch / "k.sqlite3").write_bytes(b"live")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("live_key_scratch"),
    )

    assert live_scratch.is_dir()


def test_dataset_write_removes_reused_pid_staged_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "reused_pid_stage")
    ownership = importlib.import_module(
        "quant_research_platform.storage._scratch_ownership"
    )
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(root)
    live_name = ownership.owned_scratch_name(
        scratch_prefix,
        "live",
    )
    owner_pid = live_name.removeprefix(scratch_prefix).split("-", 1)[0]
    reused_stage = root.parent / f"{scratch_prefix}{owner_pid}-0-reused"
    reused_scratch = reused_stage / ".k-reused"
    reused_scratch.mkdir(parents=True)
    Path(f"{reused_stage}.target").write_text(
        storage_atomic._atomic_target_identity(root),
        encoding="utf-8",
    )
    (reused_scratch / "k.sqlite3").write_bytes(b"scratch")
    monkeypatch.setattr(
        ownership.psutil,
        "Process",
        lambda _pid: SimpleNamespace(
            create_time=lambda: reused_stage.stat().st_ctime + 60,
            is_running=lambda: True,
        ),
    )

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("reused_pid_stage"),
    )

    assert not reused_scratch.exists()


def test_dataset_write_recovers_dead_stage_when_final_exists(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "published_with_dead_stage")
    spec = make_processed_dataset_spec("published_with_dead_stage")
    write_dataset(root, make_default_dataset_frame(), spec=spec)
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(root)
    stale_stage = root.with_name(
        f"{scratch_prefix}qzzzzzz-stale"
    )
    Path(f"{stale_stage}.target").write_text(
        storage_atomic._atomic_target_identity(root),
        encoding="utf-8",
    )
    stale_scratch = stale_stage / ".k-qzzzzzz-stale"
    stale_scratch.mkdir(parents=True)
    (stale_scratch / "k.sqlite3").write_bytes(b"scratch")

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(root, make_default_dataset_frame(), spec=spec)

    assert not stale_stage.exists()


def test_storage_scratch_ownership_has_one_process_identity_path() -> None:
    key_source = Path(dataset_reading.__file__).with_name(
        "_dataset_key_tracking.py"
    ).read_text(encoding="utf-8")
    writer_source = Path(dataset_writing.__file__).read_text(encoding="utf-8")

    assert ".owner.pid" not in key_source
    assert ".owner.pid" not in writer_source
    assert ".owner.start" not in key_source
    assert ".owner.start" not in writer_source


def test_dataset_writer_uses_single_atomic_directory_lifecycle() -> None:
    writer_source = Path(dataset_writing.__file__).read_text(encoding="utf-8")

    assert "atomic_write_directory" in writer_source
    assert "_staged_write_directory" not in writer_source
    assert "_cleanup_stale_staged_scratch" not in writer_source


def test_scratch_owner_liveness_check_never_signals_processes() -> None:
    source = Path(
        importlib.import_module(
            "quant_research_platform.storage._scratch_ownership"
        ).__file__
    ).read_text(encoding="utf-8")

    assert "os.kill" not in source


def test_exact_key_index_honors_configured_sqlite_cache_budget(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        sqlite_cache_bytes=12 * 1024,
    ) as index:
        assert index._connection is not None
        cache_size = int(index._connection.execute("PRAGMA cache_size").fetchone()[0])

    assert cache_size == -12


def test_dataset_writer_does_not_retain_row_proportional_python_key_state() -> None:
    source = Path(dataset_writing.__file__).read_text(encoding="utf-8")
    schema_source = Path(
        importlib.import_module(
            "quant_research_platform.storage._dataset_schema"
        ).__file__
    ).read_text(encoding="utf-8")

    assert "_seen_key_hashes" not in source
    assert "hash_pandas_object" not in source
    assert "def require_qrp_json_schema_id(" not in schema_source


def test_write_dataset_rejects_duplicate_across_chunks_and_cleans_stage(
    tmp_path: Path,
) -> None:
    frame = make_default_dataset_frame().iloc[:2].reset_index(drop=True)
    root = make_dataset_artifact_root(tmp_path, "duplicate_across_chunks")

    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            root,
            iter((frame, frame.iloc[[0]].copy())),
            spec=make_processed_dataset_spec("duplicate_across_chunks"),
        )

    assert not root.exists()
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(root)
    assert list(root.parent.glob(f"{scratch_prefix}*")) == []


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


def test_write_dataset_rejects_writer_scratch_name_as_final_artifact_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / ".x.tmp-user"

    with pytest.raises(ValueError, match="invalid_processed_dataset_root"):
        write_dataset(
            root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
        )

    assert not root.exists()


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
            "timestamp": pd.date_range(
                "2026-01-01", periods=row_count, freq="min"
            ),
            "value": [float(index) for index in range(row_count)],
        }
    )
    ordered_input = pd.concat(
        (frame.iloc[row_count // 2 :].copy(), frame.iloc[: row_count // 2].copy()),
        ignore_index=True,
    )
    ordered_input["timestamp"] = ordered_input["timestamp"].astype(
        "datetime64[ns]"
    )
    write_dataset(
        root,
        (
            frame.iloc[row_count // 2 :].copy(),
            frame.iloc[: row_count // 2].copy(),
        ),
        spec=make_processed_dataset_spec("bounded_reader", market_id="dummy"),
    )
    assert hasattr(dataset_reading, "iter_dataset_chunks")
    context = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    target_chunk_bytes = 512 * 1024

    chunks = list(
        dataset_reading.iter_dataset_chunks(
            context,
            target_chunk_bytes=target_chunk_bytes,
        )
    )

    assert len(chunks) > 1
    assert all(
        pandas_frame_payload_bytes(chunk) <= target_chunk_bytes
        for chunk in chunks
    )
    pd.testing.assert_frame_equal(
        pd.concat(chunks, ignore_index=True),
        ordered_input,
    )

    window = TimeWindow(
        frame.loc[28, "timestamp"],
        frame.loc[36, "timestamp"],
    )
    projected_context = dataset_reading.open_dataset_session(
        root,
        columns=("asset_id", "value"),
        time_window=window,
    )
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


def test_internal_physical_reader_rejects_single_row_over_target(
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
    assert hasattr(dataset_reading, "iter_dataset_chunks")
    context = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        list(
            dataset_reading.iter_dataset_chunks(
                context,
                target_chunk_bytes=512,
            )
        )


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

    def fail_read_parquet(*args: object, **kwargs: object) -> object:
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
    bad_frame = make_default_dataset_frame().assign(extra=1)
    dataset_writing.write_parquet_file(
        _persisted_table(bad_frame),
        bad_part,
    )

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
            "input_dataset_identities": {
                "demo_input": _input_dataset_identity_payload()
            }
        },
    )

    comparison = compare_datasets(left_root, right_root)

    assert comparison.status == "MATCH"
    assert comparison.matches


def test_fingerprint_supports_writer_accepted_nested_columns(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "nested_values")
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["value"] = [[1, 2], [3]]

    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("nested_values"),
    )

    comparison = compare_datasets(root, root)
    assert comparison.status == "MATCH"
    assert comparison.matches


def test_nested_fingerprint_rejects_actual_canonical_payload_over_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"nested": [["\x00" * (32 * 1024)]]})
    calls = {"count": 0}
    canonicalize = dataset_fingerprinting._canonical_nested_value

    def record_canonicalization(value: object) -> str:
        calls["count"] += 1
        return canonicalize(value)

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_canonical_nested_value",
        record_canonicalization,
    )

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._hashable_fingerprint_frame(
            frame,
            target_bytes=64 * 1024,
        )
    assert calls["count"] == 1


def test_fingerprint_supports_nested_non_finite_float_values(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "nested_non_finite")
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["value"] = [[1.0, math.nan], [math.inf, -0.0]]

    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("nested_non_finite"),
    )

    comparison = compare_datasets(root, root)
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
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "8",
            "QRP_PAYLOAD_TARGET_MB": "4",
        },
    )
    with runtime_resource_policy_scope(policy):
        materialized = read_dataset(root)
        chunked = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))

    assert len(chunked) == 1
    assert materialized.dtypes.astype(str).to_dict() == expected.dtypes.astype(
        str
    ).to_dict()
    assert chunked[0].dtypes.astype(str).to_dict() == expected.dtypes.astype(
        str
    ).to_dict()


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
            "timestamp": pd.date_range(
                "2026-01-01", periods=row_count, freq="min"
            ),
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


def test_fingerprint_nested_normalization_checks_actual_payloads() -> None:
    frame = pd.DataFrame({"nested": [["x" * 50] * 4]})
    canonical = frame.copy()
    canonical["nested"] = canonical["nested"].map(
        dataset_fingerprinting._canonical_nested_value
    )
    input_payload_bytes = pandas_frame_payload_bytes(frame)
    actual_payload_bytes = input_payload_bytes + pandas_frame_payload_bytes(canonical)

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._hashable_fingerprint_frame(
            frame,
            target_bytes=input_payload_bytes,
        )

    observed = dataset_fingerprinting._hashable_fingerprint_frame(
        frame,
        target_bytes=actual_payload_bytes,
    )
    pd.testing.assert_frame_equal(observed, canonical)


def test_public_storage_operations_capture_one_active_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "8",
            "QRP_PAYLOAD_TARGET_MB": "4",
        },
    )
    active_calls = 0
    delegated_policies: list[RuntimeResourcePolicy] = []
    sentinel = object()

    def active_policy() -> RuntimeResourcePolicy:
        nonlocal active_calls
        active_calls += 1
        return policy

    def delegate(*_args, **kwargs):
        delegated_policies.append(kwargs["resource_policy"])
        return sentinel

    monkeypatch.setattr(datasets, "active_runtime_resource_policy", active_policy)
    for name in (
        "_write_dataset",
        "_read_dataset",
        "_inspect_dataset",
        "_fingerprint_dataset",
        "_compare_datasets",
    ):
        monkeypatch.setattr(datasets, name, delegate)

    spec = make_processed_dataset_spec("captured_policy")
    calls = (
        lambda: datasets.write_dataset(Path("ignored"), pd.DataFrame(), spec=spec),
        lambda: datasets.read_dataset(Path("ignored")),
        lambda: datasets.read_dataset(
            Path("ignored"),
            mode=DatasetReadMode.CHUNKED,
        ),
        lambda: datasets.inspect_dataset(Path("ignored")),
        lambda: datasets.fingerprint_dataset(Path("ignored")),
        lambda: datasets.compare_datasets(Path("left"), Path("right")),
    )

    results = tuple(call() for call in calls)
    assert results[0] is None
    assert all(result is sentinel for result in results[1:])
    assert active_calls == len(calls)
    assert delegated_policies == [policy] * len(calls)


def test_storage_operations_propagate_the_captured_payload_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "32",
            "QRP_PAYLOAD_TARGET_MB": "16",
        },
    )
    root = make_dataset_artifact_root(tmp_path, "policy_propagation")
    frame = make_default_dataset_frame()
    writer_targets: list[int] = []
    read_targets: list[int] = []
    inspection_targets: list[int] = []
    fingerprint_physical_targets: list[int] = []
    fingerprint_hash_targets: list[int] = []

    writer_decode = dataset_writing.iter_declared_parquet_member_batches
    read_chunks = dataset_reading.iter_dataset_chunks
    inspect_batches = dataset_inspection.iter_dataset_file_batches
    fingerprint_batches = dataset_fingerprinting.iter_dataset_file_batches
    fingerprint_hashable = dataset_fingerprinting._hashable_fingerprint_frame

    def record_writer_target(*args, **kwargs):
        writer_targets.append(kwargs["target_chunk_bytes"])
        yield from writer_decode(*args, **kwargs)

    def record_read_target(*args, **kwargs):
        read_targets.append(kwargs["target_chunk_bytes"])
        yield from read_chunks(*args, **kwargs)

    def record_inspection_target(*args, **kwargs):
        inspection_targets.append(kwargs["target_chunk_bytes"])
        yield from inspect_batches(*args, **kwargs)

    def record_fingerprint_physical_target(*args, **kwargs):
        fingerprint_physical_targets.append(kwargs["target_chunk_bytes"])
        yield from fingerprint_batches(*args, **kwargs)

    def record_fingerprint_hash_target(*args, **kwargs):
        fingerprint_hash_targets.append(kwargs["target_bytes"])
        return fingerprint_hashable(*args, **kwargs)

    monkeypatch.setattr(
        dataset_writing,
        "iter_declared_parquet_member_batches",
        record_writer_target,
    )
    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", record_read_target)
    monkeypatch.setattr(
        dataset_inspection,
        "iter_dataset_file_batches",
        record_inspection_target,
    )
    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        record_fingerprint_physical_target,
    )
    monkeypatch.setattr(
        dataset_fingerprinting,
        "_hashable_fingerprint_frame",
        record_fingerprint_hash_target,
    )

    with runtime_resource_policy_scope(policy):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("policy_propagation"),
        )
        read_dataset(root)
        list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
        inspect_dataset(
            root,
            time_window=TimeWindow("2014-01-01", "2015-12-31"),
        )
        fingerprint_dataset(root)

    assert writer_targets and set(writer_targets) == {policy.payload_target_bytes}
    assert read_targets == [policy.payload_target_bytes] * 2
    assert inspection_targets and set(inspection_targets) == {
        policy.payload_target_bytes
    }
    assert fingerprint_physical_targets and set(fingerprint_physical_targets) == {
        policy.payload_target_bytes
    }
    assert fingerprint_hash_targets and set(fingerprint_hash_targets) == {
        policy.payload_target_bytes
    }


def test_compare_reuses_one_policy_for_both_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    observed: list[RuntimeResourcePolicy] = []

    def fake_fingerprint(*_args, resource_policy, **_kwargs):
        observed.append(resource_policy)
        return SimpleNamespace(
            schema_hash="schema",
            column_names=("value",),
            row_count=0,
            key_columns=("value",),
            content_hash="content",
            partition_fingerprints=(),
            file_count=1,
        )

    monkeypatch.setattr(dataset_comparison, "fingerprint_dataset", fake_fingerprint)

    comparison = dataset_comparison.compare_datasets(
        left,
        right,
        resource_policy=policy,
    )

    assert comparison.status == "MATCH"
    assert observed == [policy, policy]


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


def test_dataset_session_returns_fresh_contract_views_and_keeps_frozen_truth(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "immutable_session")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("immutable_session"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    expected_primary_key = tuple(session.spec["primary_key"])
    expected_schema = session.schema
    expected_logical = session.logical_dtypes

    session.spec["primary_key"].clear()
    session.manifest["physical_schema"].clear()
    session.schema.clear()
    session.logical_dtypes["columns"].clear()

    assert tuple(session.spec["primary_key"]) == expected_primary_key
    assert session.schema == expected_schema
    assert session.logical_dtypes == expected_logical
    session.verify()


def test_dataset_session_rejects_raw_contract_and_inventory_drift(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_drift")
    frame = make_default_dataset_frame()
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("session_drift"),
    )

    raw_session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    spec_path = root / "spec.json"
    spec_path.write_bytes(raw_session.spec_bytes + b" ")
    with pytest.raises(ValueError, match="dataset_spec_identity_changed"):
        raw_session.verify()
    spec_path.write_bytes(raw_session.spec_bytes)

    inventory_session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    added = root / "data" / "year=2016" / "part-000.parquet"
    dataset_writing.write_parquet_file(
        _persisted_table(frame),
        added,
    )
    with pytest.raises(ValueError, match="dataset_inventory_changed"):
        inventory_session.verify()
    added.unlink()

    removal_session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    removal_session.all_paths[0].unlink()
    with pytest.raises(ValueError, match="dataset_inventory_changed"):
        removal_session.verify()


def test_physical_decoder_rechecks_the_same_handle_after_decode(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "handle_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("handle_drift"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    path = session.all_paths[0]

    def mutate_after_open(_metadata) -> None:  # noqa: ANN001
        path.touch()

    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        list(
            dataset_physical.iter_dataset_part_batches(
                path,
                expected_identity=session.metadata_for(path).identity,
                target_chunk_bytes=1024**2,
                metadata_observer=mutate_after_open,
            )
        )


def test_writer_observes_each_closed_part_once_and_passes_live_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "writer_receipt")
    frame = make_default_dataset_frame()
    observed_paths: list[Path] = []
    decoder = dataset_writing.iter_declared_parquet_member_batches
    receipt_checks: list[bool] = []

    def record_decode(path, **kwargs):  # noqa: ANN001
        observed_paths.append(Path(path))
        yield from decoder(path, **kwargs)

    def validate_receipt(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        receipt.session.verify()
        receipt_checks.append(receipt.exact_primary_key_index.database_path.is_file())
        assert receipt.exact_primary_key_index.add_frame(frame) == len(frame)

    monkeypatch.setattr(
        dataset_writing,
        "iter_declared_parquet_member_batches",
        record_decode,
    )
    dataset_writing.write_dataset(
        root,
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_processed_dataset_spec("writer_receipt"),
        staged_validator=validate_receipt,
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=64 * 1024**2,
            environ={"QRP_TASK_MEMORY_BUDGET_MB": "32"},
        ),
    )

    assert receipt_checks == [True]
    assert len(observed_paths) == len(set(observed_paths))
    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(observed_paths) == manifest["file_count"]


def test_writer_bounds_raw_pandas_arrow_and_persisted_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "bounded_writer")
    row_count = 100
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": [f"asset-{index:03d}" for index in range(row_count)],
            "timestamp": pd.date_range("2026-01-01", periods=row_count),
            "value": ["x" * 20_000 for _ in range(row_count)],
        }
    )
    target = 1024**2
    assert pandas_frame_payload_bytes(frame) > target
    written_table_bytes: list[int] = []
    converted_frame_bytes: list[int] = []
    original_write = dataset_writing.write_parquet_file
    original_decode = dataset_writing.iter_declared_parquet_member_batches

    def record_write(table: pa.Table, path: Path) -> Path:
        written_table_bytes.append(table.nbytes)
        return original_write(table, path)

    def record_decode(*args, **kwargs):
        for current in original_decode(*args, **kwargs):
            converted_frame_bytes.append(pandas_frame_payload_bytes(current))
            yield current

    monkeypatch.setattr(dataset_writing, "write_parquet_file", record_write)
    monkeypatch.setattr(
        dataset_writing,
        "iter_declared_parquet_member_batches",
        record_decode,
    )
    result = dataset_writing.write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("bounded_writer"),
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=2 * 1024**2,
            environ={
                "QRP_TASK_MEMORY_BUDGET_MB": "1",
                "QRP_PAYLOAD_TARGET_MB": "1",
            },
        ),
    )

    assert result.root == root
    assert result.data_root == root / "data"
    assert result.row_count == row_count
    assert written_table_bytes and max(written_table_bytes) <= target
    assert converted_frame_bytes and max(converted_frame_bytes) <= target


def test_private_writer_returns_final_inspection_and_public_writer_returns_none(
    tmp_path: Path,
) -> None:
    private_root = make_dataset_artifact_root(tmp_path, "private_return")
    public_root = make_dataset_artifact_root(tmp_path, "public_return")
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "32"},
    )
    private_result = dataset_writing.write_dataset(
        private_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("private_return"),
        resource_policy=policy,
    )

    assert private_result.root == private_root
    assert private_result.data_root == private_root / "data"
    assert private_result.manifest["row_count"] == private_result.row_count
    with runtime_resource_policy_scope(policy):
        assert (
            write_dataset(
                public_root,
                make_default_dataset_frame(),
                spec=make_processed_dataset_spec("public_return"),
            )
            is None
        )


def test_writer_manifest_uses_persisted_bytes_not_prewrite_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "persisted_truth")
    original = dataset_writing.write_parquet_file

    def write_infinite(table, path):  # noqa: ANN001
        position = table.schema.get_field_index("value")
        persisted = table.set_column(
            position,
            table.schema.field(position),
            pa.array(
                [float("inf")] * table.num_rows,
                type=table.schema.field(position).type,
            ),
        )
        return original(persisted, path)

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_infinite)
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("persisted_truth"),
        )


def test_exact_primary_key_index_supports_nested_arrow_key_identity(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "key": pa.array(
                [
                    {"items": [1, 2], "label": "a"},
                    {"items": [2, 1], "label": "a"},
                ],
                type=pa.struct(
                    [
                        pa.field("items", pa.list_(pa.int64())),
                        pa.field("label", pa.string()),
                    ]
                ),
            )
        }
    )
    with dataset_writing.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert index.add_table(table) == 0
        assert index.add_table(table) == 2


def test_storage_has_one_parquet_constructor_and_no_retired_page_parser() -> None:
    storage_root = Path(dataset_reading.__file__).parent
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in storage_root.glob("*.py")
    }
    constructor_owners = [
        name for name, source in sources.items() if "pq.ParquetFile(" in source
    ]

    assert constructor_owners == ["_dataset_physical.py"]
    combined = "\n".join(sources.values())
    assert "_PHYSICAL_WORKING_SET_PARTS" not in combined
    assert "_read_compact" not in combined
    assert "_skip_compact" not in combined
    assert ".read_bytes(" not in sources["_dataset_physical.py"]
    assert tuple(dataset_physical.DatasetFileIdentity.__dataclass_fields__) == (
        "device",
        "inode",
        "size",
        "modified_ns",
    )


def test_dataset_global_nullable_integer_plan_is_chunk_order_independent(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame()
    numpy_chunk = base.iloc[:1].copy()
    nullable_chunk = base.iloc[1:2].copy()
    numpy_chunk["value"] = pd.Series(
        [1], index=numpy_chunk.index, dtype="int64"
    )
    nullable_chunk["value"] = pd.Series(
        [2], index=nullable_chunk.index, dtype="Int64"
    )
    manifests = []
    for name, chunks in (
        ("numpy_first", (numpy_chunk, nullable_chunk)),
        ("nullable_first", (nullable_chunk, numpy_chunk)),
    ):
        root = make_dataset_artifact_root(tmp_path / name, name)
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
        assert str(read_dataset(root)["value"].dtype) == "Int64"
        manifests.append(
            json.loads(
                (root / "_manifest" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    assert manifests[0]["logical_dtypes"] == manifests[1]["logical_dtypes"]
    assert manifests[0]["logical_dtype_hash"] == manifests[1][
        "logical_dtype_hash"
    ]


def test_all_empty_logical_dtype_plan_is_chunk_order_independent(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame().iloc[:0].copy()
    numpy_empty = base.astype({"value": "int64"})
    nullable_empty = base.astype({"value": "Int64"})
    observed = []
    for name, chunks in (
        ("empty_numpy_first", (numpy_empty, nullable_empty)),
        ("empty_nullable_first", (nullable_empty, numpy_empty)),
    ):
        root = make_dataset_artifact_root(tmp_path / name, name)
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
        frame = read_dataset(root)
        assert frame.empty
        assert str(frame["value"].dtype) == "Int64"
        observed.append(
            json.loads(
                (root / "_manifest" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )["logical_dtypes"]
        )
    assert observed[0] == observed[1]


@pytest.mark.parametrize("reverse", (False, True))
def test_writer_rejects_incompatible_physical_width_in_both_chunk_orders(
    tmp_path: Path,
    reverse: bool,
) -> None:
    base = make_default_dataset_frame()
    narrow = base.iloc[:1].copy()
    wide = base.iloc[1:2].copy()
    narrow["value"] = pd.Series([1], index=narrow.index, dtype="int32")
    wide["value"] = pd.Series([2], index=wide.index, dtype="int64")
    chunks = (wide, narrow) if reverse else (narrow, wide)
    root = make_dataset_artifact_root(tmp_path, f"physical_width_{reverse}")
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        write_dataset(
            root,
            chunks,
            spec=make_processed_dataset_spec(f"physical_width_{reverse}"),
        )


def test_writer_rejects_malformed_persisted_logical_dtype_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "malformed_logical_dtype")
    monkeypatch.setattr(
        dataset_writing,
        "logical_dtype_metadata_bytes",
        lambda _payload: b"{}",
    )
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("malformed_logical_dtype"),
        )


def test_writer_rejects_persisted_category_outside_declared_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_category_domain")
    frame = make_default_dataset_frame()
    frame["asset_id"] = pd.Categorical(frame["asset_id"])
    original = dataset_writing.write_parquet_file

    def write_extra_domain(current, path):  # noqa: ANN001
        persisted = current.to_pandas(use_threads=False)
        persisted["asset_id"] = persisted["asset_id"].cat.add_categories(
            ["FORGED_UNUSED_CATEGORY"]
        )
        persisted.loc[persisted.index[0], "asset_id"] = "FORGED_UNUSED_CATEGORY"
        forged = pa.Table.from_pandas(persisted, preserve_index=False)
        return original(
            forged.replace_schema_metadata(current.schema.metadata),
            path,
        )

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_extra_domain)
    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("forged_category_domain"),
        )


def test_writer_rejects_indivisible_categorical_identity_over_payload_target(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "oversized_category")
    frame = make_default_dataset_frame().iloc[:1].copy()
    value = "x" * (1024**2 + 1)
    frame["asset_id"] = pd.Categorical([value], categories=[value])
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "32",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )
    with pytest.raises(
        ValueError,
        match="dataset_categorical_value_exceeds_payload_target",
    ):
        dataset_writing.write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("oversized_category"),
            resource_policy=policy,
        )


def test_full_inspection_rejects_forged_persisted_date_summary(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_dates")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("forged_dates"),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["date_min"] = "1900-01-01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
        inspect_dataset(root)


def test_writer_receipt_reuses_observed_metadata_without_reopening_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "receipt_no_reopen")

    def forbidden_reopen(*_args, **_kwargs):
        raise AssertionError("writer_receipt_must_not_reopen_staged_parquet")

    monkeypatch.setattr(dataset_reading, "inspect_dataset_part", forbidden_reopen)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("receipt_no_reopen"),
    )


def test_dataset_session_rejects_mutable_filter_values(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "immutable_filter")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("immutable_filter"),
    )
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=(("asset_id", "==", ["A"]),),
        )


@pytest.mark.parametrize("dtype", ("Int64", "boolean", "string"))
def test_dataset_global_extension_dtype_survives_read_scopes(
    tmp_path: Path,
    dtype: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"extension_{dtype}")
    frame = make_default_dataset_frame()
    values = {
        "Int64": [1, 2, 3],
        "boolean": [True, False, True],
        "string": ["x", "y", "z"],
    }[dtype]
    frame["value"] = pd.Series(values, index=frame.index, dtype=dtype)
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(f"extension_{dtype}"),
    )
    materialized = read_dataset(root, columns=("timestamp", "value"))
    chunked = list(
        read_dataset(
            root,
            columns=("timestamp", "value"),
            mode=DatasetReadMode.CHUNKED,
        )
    )
    windowed = read_dataset(
        root,
        columns=("timestamp", "value"),
        time_window=TimeWindow(date(2015, 1, 1), date(2015, 12, 31)),
    )
    assert str(materialized["value"].dtype) == dtype
    assert {str(part["value"].dtype) for part in chunked} == {dtype}
    assert str(windowed["value"].dtype) == dtype


def test_writer_rejects_missing_persisted_logical_dtype_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "missing_logical_dtype")

    def write_without_metadata(table, path):  # noqa: ANN001
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), path)
        return path

    monkeypatch.setattr(
        dataset_writing,
        "write_parquet_file",
        write_without_metadata,
    )
    with pytest.raises(ValueError, match="dataset_logical_dtypes_missing"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("missing_logical_dtype"),
        )


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schema_id":"unknown","columns":[]}\n',
        b'{"schema_id":"qrp-dataset-part-logical-dtypes","columns":[],"x":1}\n',
        b'{"schema_id":"qrp-dataset-part-logical-dtypes","columns":[],"x":1,"x":2}\n',
    ),
)
def test_logical_dtype_metadata_rejects_unknown_and_noncanonical_payloads(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        dataset_writing.parse_logical_dtype_metadata(payload)


@pytest.mark.parametrize(
    "series",
    (
        pd.Series([1], dtype="Int64"),
        pd.Series([True], dtype="boolean"),
        pd.Series(["x"], dtype="string"),
        pd.Series([1], dtype="int64"),
        pd.Series([True], dtype="bool"),
        pd.Series([1.0], dtype="float64"),
        pd.Series(pd.to_datetime(["2026-01-01"])),
        pd.Series(pd.to_timedelta([1], unit="s")),
        pd.Series([object()], dtype="object"),
        pd.Series(pd.Categorical(["x"], categories=["x", "y"])),
    ),
)
def test_logical_dtype_grammar_round_trips_writer_canonical_families(
    series: pd.Series,
) -> None:
    frame = pd.DataFrame({"value": series})
    payload = logical_dtype_payload_from_frame(frame)

    assert dataset_schema.validate_logical_dtype_payload(payload) == payload


@pytest.mark.parametrize(
    "descriptor",
    (
        {"name": "value", "kind": "numpy_integer", "dtype": "int"},
        {"name": "value", "kind": "numpy_float", "dtype": "float"},
        {"name": "value", "kind": "unknown", "dtype": "object"},
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "bytes", "value": "FF"}],
        },
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "decimal", "value": "+1.0"}],
        },
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "decimal", "value": "-0"}],
        },
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "timestamp", "value": "2026-01-01 00:00:00"}],
        },
    ),
)
def test_logical_dtype_grammar_rejects_noncanonical_siblings(
    descriptor: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        dataset_schema.validate_logical_dtype_payload(
            {
                "schema_id": "qrp-dataset-part-logical-dtypes",
                "columns": [descriptor],
            }
        )


@pytest.mark.parametrize("reverse", (False, True), ids=("negative-first", "zero-first"))
@pytest.mark.parametrize("empty_sibling", (False, True), ids=("rows", "empty"))
def test_unordered_decimal_zero_category_has_one_canonical_domain(
    reverse: bool,
    empty_sibling: bool,
) -> None:
    negative = pd.DataFrame()
    negative["value"] = pd.Categorical(
        [Decimal("-0.00")],
        categories=[Decimal("-0.00")],
        ordered=False,
    )
    zero = pd.DataFrame()
    zero["value"] = pd.Categorical(
        [] if empty_sibling else [Decimal("0")],
        categories=[Decimal("0")],
        ordered=False,
    )
    payloads = [
        logical_dtype_payload_from_frame(frame)
        for frame in ((zero, negative) if reverse else (negative, zero))
    ]

    merged = dataset_schema.merge_logical_dtype_payloads(payloads)

    descriptor = merged["columns"][0]
    assert descriptor["categories"] == [{"type": "decimal", "value": "0"}]
    final_dtype = dataset_schema.pandas_dtype_overrides_from_logical_payload(
        merged,
        columns=("value",),
    )["value"]
    assert isinstance(final_dtype, pd.CategoricalDtype)
    assert list(final_dtype.categories) == [Decimal("0")]
    assert final_dtype.categories.size == 1
    materialized = pd.Series(
        [Decimal("-0.00"), Decimal("0")],
        dtype=final_dtype,
    )
    assert materialized.cat.categories.size == 1
    assert materialized.tolist() == [Decimal("0"), Decimal("0")]


@pytest.mark.parametrize("ordered", (False, True))
def test_writer_stops_single_category_domain_before_descriptor_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ordered: bool,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path,
        f"single_domain_early_{ordered}",
    )
    categories = [
        f"category-{index:05d}-" + "x" * 512 for index in range(5_000)
    ]
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": pd.Categorical(
                [categories[0]],
                categories=categories,
                ordered=ordered,
            ),
            "timestamp": [pd.Timestamp("2026-01-01")],
            "value": [1.0],
        }
    )
    conversions = 0
    original_value_payload = dataset_schema._typed_category_value_payload

    def guarded_value_payload(value):  # noqa: ANN001
        nonlocal conversions
        conversions += 1
        if conversions == 6_000:
            raise AssertionError("single_category_domain_tail_materialized")
        return original_value_payload(value)

    monkeypatch.setattr(
        dataset_schema,
        "_typed_category_value_payload",
        guarded_value_payload,
    )
    monkeypatch.setattr(
        dataset_writing,
        "logical_dtype_payload_from_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full_category_descriptor_materialized_before_gate")
        ),
    )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=2 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )

    with pytest.raises(
        ValueError,
        match="dataset_logical_dtype_metadata_exceeds_payload_target",
    ):
        dataset_writing.write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec(f"single_domain_early_{ordered}"),
            resource_policy=policy,
        )
    assert conversions < 6_000
    assert conversions < len(categories) * 2
    assert not root.exists()


@pytest.mark.parametrize(
    "series",
    (
        pd.Series(
            pd.Categorical(["b"], categories=["b", "a"], ordered=False)
        ),
        pd.Series(
            pd.Categorical(["b"], categories=["b", "a"], ordered=True)
        ),
        pd.Series(pd.Categorical([], categories=[], ordered=False)),
        pd.Series([1.0], dtype="float64"),
    ),
)
def test_streamed_logical_dtype_size_matches_final_metadata_bytes(
    series: pd.Series,
) -> None:
    payload = logical_dtype_payload_from_frame(pd.DataFrame({"value": series}))
    skeleton = {
        "schema_id": payload["schema_id"],
        "columns": [
            {
                **descriptor,
                **(
                    {"categories": []}
                    if descriptor["kind"] == "categorical"
                    else {}
                ),
            }
            for descriptor in payload["columns"]
        ],
    }

    def category_payloads(descriptor):  # noqa: ANN001
        if descriptor["kind"] != "categorical":
            raise AssertionError("noncategory_requested_category_payloads")
        return iter(payload["columns"][0]["categories"])

    assert dataset_schema.require_logical_dtype_metadata_within_target(
        skeleton,
        category_payloads=category_payloads,
        target_chunk_bytes=1024**2,
    ) == len(logical_dtype_metadata_bytes(payload))


def test_empty_unordered_category_domain_unions_with_nonempty_domain(
    tmp_path: Path,
) -> None:
    manifests = []
    for name, reverse in (("empty_first", False), ("empty_last", True)):
        root = make_dataset_artifact_root(tmp_path, name)
        empty = make_default_dataset_frame().iloc[:0].copy()
        empty["asset_id"] = pd.Categorical(
            [],
            categories=["empty-only"],
            ordered=False,
        )
        nonempty = make_default_dataset_frame().iloc[:1].copy()
        nonempty["asset_id"] = pd.Categorical(
            ["live-only"],
            categories=["live-only"],
            ordered=False,
        )
        chunks = (nonempty, empty) if reverse else (empty, nonempty)
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
        manifest = json.loads(
            (root / "_manifest" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        session = dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        )
        assert manifest["row_count"] == 1
        assert manifest["file_count"] == len(session.all_paths) == 2
        assert manifest["partition_count"] == 1
        expected_partition = f"year={nonempty['timestamp'].iloc[0].year}"
        assert manifest["partition_row_counts"] == {expected_partition: 1}
        assert [path.name for path in session.all_paths] == [
            "part-000.parquet",
            "part-001.parquet",
        ]
        assert len({path.parent for path in session.all_paths}) == 1
        assert tuple(
            session.metadata_for(path).row_count for path in session.all_paths
        ) == (1, 0)
        assert session.logical_dtypes == manifest["logical_dtypes"]
        assert read_dataset(root)["asset_id"].astype("string").tolist() == [
            "live-only"
        ]
        manifests.append(manifest)

    domains = []
    for manifest in manifests:
        descriptor = next(
            item
            for item in manifest["logical_dtypes"]["columns"]
            if item["name"] == "asset_id"
        )
        domains.append(descriptor["categories"])
    assert domains[0] == domains[1]
    assert {item["value"] for item in domains[0]} == {
        "empty-only",
        "live-only",
    }


@pytest.mark.parametrize("empty_first", (False, True))
def test_empty_ordered_category_domain_mismatch_is_not_ignored(
    tmp_path: Path,
    empty_first: bool,
) -> None:
    name = f"empty_ordered_mismatch_{empty_first}"
    root = make_dataset_artifact_root(tmp_path, name)
    empty = make_default_dataset_frame().iloc[:0].copy()
    empty["asset_id"] = pd.Categorical(
        [],
        categories=["a", "b"],
        ordered=True,
    )
    nonempty = make_default_dataset_frame().iloc[:1].copy()
    nonempty["asset_id"] = pd.Categorical(
        ["a"],
        categories=["a", "c"],
        ordered=True,
    )
    chunks = (empty, nonempty) if empty_first else (nonempty, empty)

    with pytest.raises(ValueError, match="dataset_ordered_categorical_mismatch"):
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
    assert not root.exists()


@pytest.mark.parametrize("empty_chunks", (False, True))
def test_writer_retains_one_incremental_logical_dtype_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_chunks: bool,
) -> None:
    name = f"incremental_dtype_plan_{empty_chunks}"
    root = make_dataset_artifact_root(tmp_path, name)
    base = make_default_dataset_frame().iloc[:1].copy()
    chunks = []
    for index in range(32):
        current = base.iloc[:0].copy() if empty_chunks else base.copy()
        if not empty_chunks:
            current["asset_id"] = f"asset-{index:03d}"
            current["timestamp"] = pd.Timestamp("2026-01-01") + pd.Timedelta(
                days=index
            )
        chunks.append(current)
    retained_column_counts: list[int] = []
    original = dataset_writing._WriteStats.observe_logical_dtype_payload

    def record_plan(self, *args, **kwargs):  # noqa: ANN001
        result = original(self, *args, **kwargs)
        assert self._logical_dtype_plan is not None
        retained_column_counts.append(len(self._logical_dtype_plan["columns"]))
        assert not hasattr(self, "_logical_dtype_parts")
        return result

    monkeypatch.setattr(
        dataset_writing._WriteStats,
        "observe_logical_dtype_payload",
        record_plan,
    )
    write_dataset(root, chunks, spec=make_processed_dataset_spec(name))

    assert retained_column_counts
    assert set(retained_column_counts) == {len(base.columns)}
    assert "empty_logical_parts" not in Path(dataset_writing.__file__).read_text(
        encoding="utf-8"
    )


def test_writer_rejects_cumulative_logical_dtype_metadata_over_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "category_metadata_bound")
    chunks = []
    for chunk_index in range(24):
        categories = [
            f"category-{chunk_index:02d}-{index:04d}-" + "x" * 48
            for index in range(1_000)
        ]
        chunks.append(
            pd.DataFrame(
                {
                    "market_id": ["a_share"],
                    "asset_id": pd.Categorical(
                        [categories[0]],
                        categories=categories,
                    ),
                    "timestamp": [pd.Timestamp("2026-01-01")],
                    "value": [float(chunk_index)],
                }
            )
        )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "32",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )
    stream_calls = 0
    streamed_values = 0
    original_values = dataset_writing.ExactPrimaryKeyIndex.identity_payload_values

    def guarded_values(self, *, namespace):  # noqa: ANN001
        nonlocal stream_calls, streamed_values
        stream_calls += 1
        if stream_calls > 1:
            raise AssertionError("category_domain_materialized_after_failed_gate")
        source = original_values(self, namespace=namespace)
        try:
            for raw in source:
                streamed_values += 1
                if streamed_values == 15_000:
                    raise AssertionError("category_domain_tail_materialized")
                yield raw
        finally:
            source.close()

    monkeypatch.setattr(
        dataset_writing.ExactPrimaryKeyIndex,
        "identity_payload_values",
        guarded_values,
    )

    with pytest.raises(
        ValueError,
        match="dataset_logical_dtype_metadata_exceeds_payload_target",
    ):
        dataset_writing.write_dataset(
            root,
            chunks,
            spec=make_processed_dataset_spec("category_metadata_bound"),
            resource_policy=policy,
        )
    assert stream_calls == 1
    assert streamed_values < 15_000
    assert streamed_values < 24_000
    assert not root.exists()


def test_unordered_category_domain_uses_live_exact_index_namespace(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "category_spill")
    frame = make_default_dataset_frame().iloc[:1].copy()
    categories = [f"asset-{index:05d}" for index in range(2_000)]
    frame["asset_id"] = pd.Categorical(
        [categories[-1]],
        categories=categories,
        ordered=False,
    )
    observed_counts: list[int] = []

    def validate(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        observed_counts.append(
            len(
                list(
                    receipt.exact_primary_key_index.identity_payload_values(
                        namespace="logical-category:asset_id"
                    )
                )
            )
        )

    dataset_writing.write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("category_spill"),
        staged_validator=validate,
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=64 * 1024**2,
            environ={"QRP_TASK_MEMORY_BUDGET_MB": "32"},
        ),
    )
    assert observed_counts == [len(categories)]
    assert len(read_dataset(root)["asset_id"].cat.categories) == len(categories)


def test_writer_rejects_mixed_incompatible_category_values(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "mixed_category")
    frame = make_default_dataset_frame()
    frame["asset_id"] = pd.Categorical([1, "1", 1], categories=[1, "1"])
    with pytest.raises(ValueError, match="unsupported_dataset_logical_dtype"):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("mixed_category"),
        )


def test_session_detects_part_manifest_and_same_size_identity_changes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "identity_changes")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("identity_changes"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest_path.write_bytes(session.manifest_bytes + b" ")
    with pytest.raises(ValueError, match="dataset_manifest_identity_changed"):
        session.verify()
    manifest_path.write_bytes(session.manifest_bytes)

    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    part = session.all_paths[0]
    raw = bytearray(part.read_bytes())
    raw[len(raw) // 2] ^= 1
    part.write_bytes(raw)
    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        session.verify()


def test_session_detects_part_replacement_after_issue(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "part_replacement")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("part_replacement"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    part = session.all_paths[0]
    replacement = tmp_path / "replacement.parquet"
    replacement.write_bytes(part.read_bytes())
    replacement.replace(part)
    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        session.verify()


def test_exact_key_decimal_zero_and_numpy_scalar_identity(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", pa.decimal128(18, 4))]),
    ) as decimals:
        assert decimals.add_frame(pd.DataFrame({"key": [Decimal("-0.0000")]})) == 0
        assert decimals.add_frame(pd.DataFrame({"key": [Decimal("0.0000")]})) == 1
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", pa.int64())]),
    ) as integers:
        assert integers.add_frame(pd.DataFrame({"key": [np.int64(7)]})) == 0
        assert integers.add_frame(pd.DataFrame({"key": [7]})) == 1


def test_staged_validator_wrong_keys_rejects_and_cleans_atomic_state(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "wrong_staged_keys")
    frame = make_default_dataset_frame()

    def reject_wrong_keys(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        wrong = frame.copy()
        wrong["asset_id"] = ["WRONG-A", "WRONG-B", "WRONG-C"]
        receipt.exact_primary_key_index.add_frame(wrong, namespace="expected")
        missing, extra = receipt.exact_primary_key_index.difference_counts(
            expected_namespace="expected",
            actual_namespace="dataset",
        )
        assert (missing, extra) == (len(frame), len(frame))
        raise ValueError("staged_dataset_primary_key_mismatch")

    with pytest.raises(ValueError, match="staged_dataset_primary_key_mismatch"):
        dataset_writing.write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("wrong_staged_keys"),
            staged_validator=reject_wrong_keys,
            resource_policy=RuntimeResourcePolicy.from_environment(
                available_memory_bytes=64 * 1024**2,
                environ={"QRP_TASK_MEMORY_BUDGET_MB": "32"},
            ),
        )
    assert not root.exists()
    assert not list(root.parent.glob(".atomic-*"))
