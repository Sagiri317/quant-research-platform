from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
from datetime import date, datetime
from decimal import Decimal
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_writing as dataset_writing
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
        and path.parts[len(root.parent.parts)].startswith(".staged_key_scratch.tmp-")
        for path in observed_database_paths
    )
    assert not any(path.exists() for path in observed_database_paths)


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
) -> None:
    path = tmp_path / "dictionary-small-strings.parquet"
    pq.write_table(
        pa.table({"value": ["x" * 128] * 20_000}),
        path,
        compression="zstd",
        use_dictionary=True,
    )
    target_chunk_bytes = 4 * 1024**2
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
        assert strings.add_frame(pd.DataFrame({"key": ["é"]})) == 0
        assert strings.add_frame(pd.DataFrame({"key": ["e\u0301"]})) == 0


def test_dataset_writer_accepts_dictionary_encoded_primary_keys(
    tmp_path: Path,
) -> None:
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


def test_exact_key_index_recovers_dead_sibling_scratch(tmp_path: Path) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
    stale = tmp_path / ".k-deadbeef"
    stale.mkdir()
    (stale / ".owner.pid").write_text("999999999", encoding="ascii")
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
    live = tmp_path / f".k-{os.getpid()}-concurrent"
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


def test_dataset_write_recovers_stale_key_scratch_without_publishing_it(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "stale_key_scratch")
    stale_stage = root.parent / ".stale_key_scratch.tmp-deadbeef"
    stale_scratch = stale_stage / ".k-deadbeef"
    stale_scratch.mkdir(parents=True)
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

    assert not stale_scratch.exists()
    assert not list(root.rglob("*.sqlite3"))
    assert not list(root.rglob("*.sqlite3-*"))


def test_dataset_write_does_not_remove_live_concurrent_key_scratch(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "live_key_scratch")
    live_stage = root.parent / ".live_key_scratch.tmp-concurrent"
    live_scratch = live_stage / ".k-concurrent"
    live_scratch.mkdir(parents=True)
    (live_stage / ".owner.pid").write_text(str(os.getpid()), encoding="ascii")
    (live_scratch / "k.sqlite3").write_bytes(b"live")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("live_key_scratch"),
    )

    assert live_scratch.is_dir()


def test_staged_owner_liveness_check_never_signals_processes() -> None:
    source = Path(dataset_writing.__file__).read_text(encoding="utf-8")

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

    assert "_seen_key_hashes" not in source
    assert "hash_pandas_object" not in source


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
    assert list(root.parent.glob(".duplicate_across_chunks.tmp-*")) == []


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
    target_chunk_bytes = 1_024

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
            "input_fingerprints": {"demo_input": _input_fingerprint_payload()}
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
