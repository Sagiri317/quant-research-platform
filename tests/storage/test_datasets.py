from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
from datetime import date, datetime
from decimal import Decimal
from inspect import getsource, signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage._dataset_fingerprinting as dataset_fingerprinting
import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_writing as dataset_writing
import quant_research_platform.storage.atomic as storage_atomic
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
from quant_research_platform.storage._dataset_schema import sha256_json
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
        match="unsupported_dataset_instance_metadata_key: input_fingerprints",
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


def test_physical_batch_size_accounts_for_arrow_to_pandas_expansion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repeated-strings.parquet"
    table = pa.table({"value": ["x"] * 20_000})
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    parquet_file = pq.ParquetFile(path)
    target_chunk_bytes = 16 * 1024

    batch_rows = dataset_reading._target_batch_rows(
        parquet_file,
        columns=("value",),
        target_chunk_bytes=target_chunk_bytes,
    )
    batch = next(
        parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=["value"],
            use_threads=False,
        )
    )
    pandas_bytes = int(
        batch.to_pandas(use_threads=False).memory_usage(index=True, deep=True).sum()
    )

    assert pandas_bytes <= target_chunk_bytes


def test_fixed_width_batch_estimate_uses_parquet_leaf_paths(tmp_path: Path) -> None:
    path = tmp_path / "nested-before-fixed-width.parquet"
    row_count = 10_000
    table = pa.table(
        {
            "metadata": pa.StructArray.from_arrays(
                (
                    pa.array([True] * row_count),
                    pa.array([False] * row_count),
                ),
                names=("first", "second"),
            ),
            "payload": pa.array(
                [b"x" * 1024] * row_count,
                type=pa.binary(1024),
            ),
        }
    )
    pq.write_table(table, path, compression="zstd")
    parquet_file = pq.ParquetFile(path)
    target_chunk_bytes = 64 * 1024

    batch_rows = dataset_reading._target_batch_rows(
        parquet_file,
        columns=("payload",),
        target_chunk_bytes=target_chunk_bytes,
    )
    batch = next(
        parquet_file.iter_batches(
            batch_size=batch_rows,
            columns=["payload"],
            use_threads=False,
        )
    )

    assert batch.nbytes <= target_chunk_bytes // 4


def test_fixed_size_binary_conversion_probes_before_pandas_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        [pa.array([b"x"] * 64, type=pa.binary(1))],
        names=["value"],
    )
    observed_conversion_bytes: list[int] = []
    convert = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        frame = convert(arrow_slice)
        observed_conversion_bytes.append(
            dataset_reading._pandas_frame_bytes(frame)
        )
        return frame

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=2_048,
        )
    )

    assert sum(len(chunk) for chunk in chunks) == 64
    assert max(observed_conversion_bytes) <= 2_048 // 4


def test_single_row_over_conversion_estimate_fails_before_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        [pa.array([b"x"], type=pa.binary(1))],
        names=["value"],
    )
    calls = {"count": 0}

    def record_conversion(_arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("conversion_must_not_run")

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=512,
            )
        )

    assert calls["count"] == 0


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
def test_wide_single_row_fails_before_pandas_conversion(
    arrays: list[pa.Array],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        arrays,
        names=[f"column_{index}" for index in range(len(arrays))],
    )
    calls = {"count": 0}

    def record_conversion(_arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("conversion_must_not_run")

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

    assert calls["count"] == 0


def test_wide_multi_row_bounds_each_pandas_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        [pa.nulls(64) for _ in range(500)],
        names=[f"column_{index}" for index in range(500)],
    )
    observed_conversion_bytes: list[int] = []
    convert = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        frame = convert(arrow_slice)
        observed_conversion_bytes.append(
            dataset_reading._pandas_frame_bytes(frame)
        )
        return frame

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

    assert sum(len(chunk) for chunk in chunks) == 64
    assert max(observed_conversion_bytes) <= (256 * 1024) // 4


def test_wide_struct_row_fails_before_pandas_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    struct_type = pa.struct(
        [pa.field(f"field_{index}", pa.null()) for index in range(1_000)]
    )
    batch = pa.record_batch(
        [pa.array([{}], type=struct_type)],
        names=["payload"],
    )
    calls = {"count": 0}

    def record_conversion(_arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("conversion_must_not_run")

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=60_000,
            )
        )

    assert calls["count"] == 0


def test_large_map_row_fails_before_pandas_conversion(
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

    def record_conversion(_arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("conversion_must_not_run")

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=256 * 1024,
            )
        )

    assert calls["count"] == 0


def test_wide_numeric_row_counts_column_axis_before_pandas_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch(
        [pa.array([1], type=pa.int64()) for _ in range(1_500)],
        names=[f"column_{index}" for index in range(1_500)],
    )
    calls = {"count": 0}

    def record_conversion(_arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("conversion_must_not_run")

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=64 * 1024,
            )
        )

    assert calls["count"] == 0


def test_wide_empty_schema_fails_before_pandas_materialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide-empty.parquet"
    columns = tuple(f"column_{index}" for index in range(1_500))
    pq.write_table(
        pa.table({column: pa.array([], type=pa.int64()) for column in columns}),
        path,
    )
    context = SimpleNamespace(all_paths=(path,))

    with pytest.raises(ValueError, match="dataset_row_exceeds_target_chunk_bytes"):
        dataset_reading._empty_frame(
            context,
            columns,
            target_chunk_bytes=16 * 1024,
        )


def test_materialized_context_fails_before_collecting_all_physical_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yielded = 0

    def frames(*_args, **_kwargs):
        nonlocal yielded
        for index in range(4):
            yielded += 1
            yield pd.DataFrame({"value": ["x" * 1_024], "part": [index]})

    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", frames)

    with pytest.raises(
        ValueError,
        match="dataset_materialized_result_exceeds_target_bytes",
    ):
        dataset_reading.materialize_dataset_context(
            SimpleNamespace(),
            target_chunk_bytes=64 * 1024,
            materialized_budget_bytes=2_048,
        )

    assert yielded < 4


def test_variable_width_batch_probe_precedes_arrow_decode_expansion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dictionary-long-strings.parquet"
    value = "x" * (64 * 1024)
    pq.write_table(
        pa.table({"value": [value] * 1_024}),
        path,
        compression="zstd",
        use_dictionary=True,
    )
    parquet_file = pq.ParquetFile(path)

    assert dataset_reading._target_batch_rows(
        parquet_file,
        columns=("value",),
        target_chunk_bytes=64 * 1024,
    ) == 1


def test_variable_width_probe_adapts_above_probe_size_for_small_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dictionary-small-strings.parquet"
    pq.write_table(
        pa.table({"value": ["x" * 128] * 20_000}),
        path,
        compression="zstd",
        use_dictionary=True,
    )
    target_chunk_bytes = 4 * 1024**2
    requested_batch_sizes: list[int] = []
    raw_batches = dataset_reading._raw_arrow_batches

    def record_raw_batch_size(path, **kwargs):  # noqa: ANN001
        requested_batch_sizes.append(kwargs["batch_size"])
        yield from raw_batches(path, **kwargs)

    monkeypatch.setattr(
        dataset_reading,
        "_raw_arrow_batches",
        record_raw_batch_size,
    )
    maximum_rows = max(
        batch.num_rows
        for batch in dataset_reading._iter_filtered_arrow_batches(
            path,
            columns=("value",),
            parquet_filters=None,
            batch_size=1,
            target_chunk_bytes=target_chunk_bytes,
            probe_variable_width=True,
        )
    )

    assert maximum_rows > dataset_reading._PANDAS_PROBE_ROWS
    assert requested_batch_sizes[0] == 1
    assert max(requested_batch_sizes[1:]) > 1


def test_plain_variable_width_probe_does_not_decode_one_row_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "plain-unique-strings.parquet"
    pq.write_table(
        pa.table(
            {
                "value": [
                    f"value-{index:08d}-" + "x" * 20
                    for index in range(20_000)
                ]
            }
        ),
        path,
        compression="zstd",
        use_dictionary=False,
    )
    requested_batch_sizes: list[int] = []
    raw_batches = dataset_reading._raw_arrow_batches

    def record_raw_batch_size(path, **kwargs):  # noqa: ANN001
        requested_batch_sizes.append(kwargs["batch_size"])
        yield from raw_batches(path, **kwargs)

    monkeypatch.setattr(dataset_reading, "_raw_arrow_batches", record_raw_batch_size)
    tuple(
        dataset_reading._adaptive_variable_width_source_batches(
            path,
            columns=("value",),
            parquet_filters=None,
            component_budget=512 * 1024,
        )
    )

    assert requested_batch_sizes[0] == 1
    assert max(requested_batch_sizes[1:]) > 1_000


def test_parquet_page_is_rejected_before_oversized_decompression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized-data-page.parquet"
    pq.write_table(
        pa.table({"value": ["x" * 1_024] * 10_000}),
        path,
        compression="zstd",
        use_dictionary=False,
        data_page_size=32 * 1024**2,
    )

    with pytest.raises(
        ValueError,
        match="dataset_parquet_page_exceeds_target_chunk_bytes",
    ):
        next(
            dataset_reading.iter_parquet_arrow_batches(
                path,
                columns=("value",),
                target_chunk_bytes=4 * 1024**2,
            )
        )


def test_variable_width_batch_adapts_before_arrow_allocation_when_probe_underestimates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "variable-width-spike.parquet"
    pq.write_table(
        pa.table({"value": ["x", *(["y" * 4096] * 200)]}),
        path,
        compression="zstd",
        use_dictionary=True,
    )

    observed_batch_bytes: list[int] = []
    require_budget = dataset_reading._require_arrow_batch_budget

    def record_budget_check(batch, *, arrow_budget):  # noqa: ANN001
        observed_batch_bytes.append(batch.nbytes)
        require_budget(batch, arrow_budget=arrow_budget)

    monkeypatch.setattr(
        dataset_reading,
        "_require_arrow_batch_budget",
        record_budget_check,
    )
    batches = tuple(
        dataset_reading._iter_filtered_arrow_batches(
            path,
            columns=("value",),
            parquet_filters=None,
            batch_size=1,
            target_chunk_bytes=64 * 1024,
            probe_variable_width=True,
        )
    )

    assert sum(batch.num_rows for batch in batches) == 201
    assert max(observed_batch_bytes) <= 64 * 1024 // 4


def test_variable_width_batch_accounts_for_python_batch_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "variable-width-python-state.parquet"
    pq.write_table(
        pa.table({"value": ["x"] * 2_000}),
        path,
        compression="zstd",
        use_dictionary=True,
    )
    observed_state_bytes: list[int] = []
    coalesce = dataset_reading._coalesced_record_batch

    def record_state(batches, *, arrow_budget):  # noqa: ANN001
        observed_state_bytes.append(sum(sys.getsizeof(batch) for batch in batches))
        return coalesce(batches, arrow_budget=arrow_budget)

    monkeypatch.setattr(
        dataset_reading,
        "_coalesced_record_batch",
        record_state,
    )

    tuple(
        dataset_reading._iter_filtered_arrow_batches(
            path,
            columns=("value",),
            parquet_filters=None,
            batch_size=1,
            target_chunk_bytes=64 * 1024,
            probe_variable_width=True,
        )
    )

    assert max(observed_state_bytes) <= 64 * 1024 // 4


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


def test_categorical_domain_budget_rejects_before_dictionary_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized-dictionary.parquet"
    dictionary = pa.array(
        [f"category-{index:04d}-" + "x" * 1_024 for index in range(1_000)]
    )
    table = pa.table(
        {
            "value": pa.DictionaryArray.from_arrays(
                pa.array([0], type=pa.int32()),
                dictionary,
            )
        }
    )
    pq.write_table(table, path, use_dictionary=True)

    def unexpected_decode(*_args, **_kwargs):
        raise AssertionError("dictionary_decode_started_before_budget_check")
        yield

    monkeypatch.setattr(
        dataset_reading,
        "iter_parquet_arrow_batches",
        unexpected_decode,
    )

    with pytest.raises(
        ValueError,
        match="dataset_categorical_state_exceeds_target_bytes",
    ):
        dataset_reading.dataset_categorical_dtypes(
            (path,),
            ("value",),
            target_state_bytes=64 * 1024,
        )
    source = getsource(dataset_reading.dataset_categorical_dtypes)
    assert ".iter_batches(" not in source
    assert "iter_parquet_arrow_batches(" in source


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
    expected = sha256_json(
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


def test_exact_primary_key_tracker_is_spillable_exact_and_cleans_up(
    tmp_path: Path,
) -> None:
    module_spec = importlib.util.find_spec(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    assert module_spec is not None, "storage exact key tracker module is missing"
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    tracker_root = tmp_path / "tracker"
    tracker_root.mkdir()
    tracker = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id", "timestamp"),
        timestamp_columns=("timestamp",),
        scratch_parent=tracker_root,
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

    with tracker:
        database_path = tracker.database_path
        assert database_path.is_file()
        assert tracker.add_frame(first) == 0
        assert tracker.add_frame(duplicate_same_instant) == 1
        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").write_bytes(b"scratch")

    assert not database_path.exists()
    assert list(tracker_root.iterdir()) == []

    numeric = module.ExactPrimaryKeyIndex(
        primary_key=("numeric_key",),
        timestamp_columns=(),
        scratch_parent=tracker_root,
    )
    with numeric:
        assert numeric.add_frame(pd.DataFrame({"numeric_key": [-0.0]})) == 0
        assert numeric.add_frame(pd.DataFrame({"numeric_key": [0.0]})) == 1
    assert list(tracker_root.iterdir()) == []

    exceptional = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
        scratch_parent=tracker_root,
    )
    with pytest.raises(RuntimeError, match="stop"):
        with exceptional:
            exceptional_path = exceptional.database_path
            raise RuntimeError("stop")
    assert not exceptional_path.exists()
    assert list(tracker_root.iterdir()) == []


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
        assert numeric.add_frame(pd.DataFrame({"key": [math.inf]})) == 0
        assert numeric.add_frame(pd.DataFrame({"key": [math.inf]})) == 1
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
    assert len(chunks) == 2
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)


def test_internal_physical_reader_uses_bounded_record_batches(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path,
        "bounded_reader",
        market_id="dummy",
    )
    frame = pd.DataFrame(
        {
            "market_id": ["dummy"] * 64,
            "asset_id": [f"asset_{index:04d}" for index in range(64)],
            "timestamp": pd.date_range("2026-01-01", periods=64, freq="min"),
            "value": [float(index) for index in range(64)],
        }
    )
    ordered_input = pd.concat(
        (frame.iloc[32:].copy(), frame.iloc[:32].copy()),
        ignore_index=True,
    )
    ordered_input["timestamp"] = ordered_input["timestamp"].astype(
        "datetime64[ns]"
    )
    write_dataset(
        root,
        (frame.iloc[32:].copy(), frame.iloc[:32].copy()),
        spec=make_processed_dataset_spec("bounded_reader", market_id="dummy"),
    )
    assert hasattr(dataset_reading, "iter_dataset_chunks")
    context = dataset_reading.build_dataset_read_context(
        root,
        columns=None,
        time_window=None,
    )
    target_chunk_bytes = 2_048

    chunks = list(
        dataset_reading.iter_dataset_chunks(
            context,
            target_chunk_bytes=target_chunk_bytes,
        )
    )

    assert len(chunks) > 1
    assert all(
        int(chunk.memory_usage(index=True, deep=True).sum()) <= target_chunk_bytes
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
    projected_context = dataset_reading.build_dataset_read_context(
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
    context = dataset_reading.build_dataset_read_context(
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


def test_nested_fingerprint_rejects_escaping_expansion_before_canonical_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"nested": [["\x00" * (32 * 1024)]]})

    def unexpected_canonicalization(_value: object) -> str:
        raise AssertionError("canonical_string_allocated_before_budget_check")

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_canonical_nested_value",
        unexpected_canonicalization,
    )

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._hashable_fingerprint_frame(
            frame,
            target_bytes=64 * 1024,
        )


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
    write_dataset(
        root,
        expected,
        spec=make_processed_dataset_spec("empty_dtypes"),
    )

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


def test_equality_group_plan_indexes_only_candidate_part_paths(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "equality_groups")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("equality_groups"),
    )
    context = dataset_reading.build_dataset_read_context(
        root,
        columns=None,
        time_window=None,
    )

    plan = dataset_reading.build_dataset_equality_group_plan(
        context,
        columns=("market_id", "asset_id"),
        target_chunk_bytes=1024**2,
        target_state_bytes=1024**2,
    )
    group_context = dataset_reading.dataset_context_for_equality_group(
        context,
        plan,
        ("a_share", "asset_002"),
    )
    observed = pd.concat(
        dataset_reading.iter_dataset_chunks(
            group_context,
            target_chunk_bytes=1024**2,
        ),
        ignore_index=True,
    )

    assert plan.keys == (
        ("a_share", "asset_001"),
        ("a_share", "asset_002"),
    )
    assert len(group_context.read_paths) == 1
    assert observed["asset_id"].tolist() == ["asset_002"]


def test_equality_group_plan_state_accounting_work_is_linear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 200
    root = make_dataset_artifact_root(tmp_path, "linear_equality_groups")
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share"] * row_count,
                "asset_id": [f"asset_{index:04d}" for index in range(row_count)],
                "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="min"),
                "value": np.arange(row_count, dtype="float64"),
            }
        ),
        spec=make_processed_dataset_spec("linear_equality_groups"),
    )
    context = dataset_reading.build_dataset_read_context(
        root,
        columns=None,
        time_window=None,
    )
    original = dataset_reading._equality_group_plan_bytes
    observed_work = {"units": 0}

    def record_accounting_work(*args, **kwargs):
        current = args[0]
        observed_work["units"] += len(current) if isinstance(current, dict) else 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dataset_reading,
        "_equality_group_plan_bytes",
        record_accounting_work,
    )

    plan = dataset_reading.build_dataset_equality_group_plan(
        context,
        columns=("market_id", "asset_id"),
        target_chunk_bytes=64 * 1024,
        target_state_bytes=1024**2,
    )

    assert len(plan.keys) == row_count
    assert observed_work["units"] <= row_count * 2


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
