from __future__ import annotations

import builtins
import gc
import importlib
import importlib.util
import json
import math
import os
import struct
import sys
import tracemalloc
import weakref
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from inspect import currentframe, signature
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage._dataset_comparison as dataset_comparison
import quant_research_platform.storage._dataset_fingerprinting as dataset_fingerprinting
import quant_research_platform.storage._dataset_inspection as dataset_inspection
import quant_research_platform.storage._dataset_inventory as dataset_inventory
import quant_research_platform.storage._dataset_observation as dataset_observation
import quant_research_platform.storage._dataset_physical as dataset_physical
import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_schema as dataset_schema
import quant_research_platform.storage._dataset_writing as dataset_writing
import quant_research_platform.storage.atomic as storage_atomic
import quant_research_platform.storage.datasets as datasets
import quant_research_platform.storage.errors as storage_errors
import quant_research_platform.storage.manifests as storage_manifests
from quant_research_platform.contracts._json import (
    canonical_identity_bytes,
    canonical_json_hash,
)
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
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def _issued_test_policy(task_bytes: int = 16 * 1024**2) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=task_bytes * 5 // 4,
    )


def _writer_test_policy(
    working_headroom_bytes: int = 1024**2,
) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=(
            dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
            + dataset_writing.PARQUET_WRITER_NATIVE_BYTES
            + working_headroom_bytes
        ),
    )


_ARROW_SCALAR_LOGICAL_DTYPE_CASES = (
    ("boolean", pa.bool_(), "nullable_boolean", "boolean", True),
    ("int8", pa.int8(), "nullable_integer", "Int8", 1),
    ("int16", pa.int16(), "nullable_integer", "Int16", 1),
    ("int32", pa.int32(), "nullable_integer", "Int32", 1),
    ("int64", pa.int64(), "nullable_integer", "Int64", 1),
    ("uint8", pa.uint8(), "nullable_integer", "UInt8", 1),
    ("uint16", pa.uint16(), "nullable_integer", "UInt16", 1),
    ("uint32", pa.uint32(), "nullable_integer", "UInt32", 1),
    ("uint64", pa.uint64(), "nullable_integer", "UInt64", 1),
    ("float32", pa.float32(), "nullable_float", "Float32", 1.5),
    ("float64", pa.float64(), "nullable_float", "Float64", 1.5),
    (
        "timestamp-s",
        pa.timestamp("s"),
        "datetime",
        "datetime64[s]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-ms",
        pa.timestamp("ms"),
        "datetime",
        "datetime64[ms]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-us",
        pa.timestamp("us"),
        "datetime",
        "datetime64[us]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-ns",
        pa.timestamp("ns"),
        "datetime",
        "datetime64[ns]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-s-utc",
        pa.timestamp("s", tz="UTC"),
        "datetime",
        "datetime64[s, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "timestamp-ms-utc",
        pa.timestamp("ms", tz="UTC"),
        "datetime",
        "datetime64[ms, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "timestamp-us-utc",
        pa.timestamp("us", tz="UTC"),
        "datetime",
        "datetime64[us, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "timestamp-ns-utc",
        pa.timestamp("ns", tz="UTC"),
        "datetime",
        "datetime64[ns, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    ("string", pa.string(), "string", "string", "value"),
    ("large-string", pa.large_string(), "string", "string", "value"),
)

_ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES = tuple(
    case for case in _ARROW_SCALAR_LOGICAL_DTYPE_CASES if case[2] == "datetime"
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
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        require_consumable=False,
    ) as session:
        monkeypatch.setattr(
            dataset_inspection,
            "open_dataset_session",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("session_inspection_reopened_dataset")
            ),
        )

        inspection = dataset_inspection.inspect_dataset_session(
            session,
            resource_policy=session.resource_policy,
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


def test_writer_reserves_complete_fixed_owner_before_sqlite_or_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenInput:
        def __iter__(self) -> ForbiddenInput:
            return self

        def __next__(self) -> pd.DataFrame:
            raise AssertionError("input_pulled_before_storage_owner_admission")

        def close(self) -> None:
            pass

    def reject_connect(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("sqlite_opened_before_storage_owner_admission")

    monkeypatch.setattr(dataset_inventory.sqlite3, "connect", reject_connect)
    policy = capture_test_policy(
        available_memory_bytes=2560,
        explicit_cap_bytes=2048,
    )

    with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "fixed_owner_admission"),
            ForbiddenInput(),
            spec=make_processed_dataset_spec("fixed_owner_admission"),
            resource_policy=policy,
        )


@pytest.mark.parametrize(
    "exit_mode",
    (
        "success",
        "runtime_error",
        "keyboard_interrupt",
        "system_exit",
        "close",
        "target_exists",
    ),
)
def test_writer_closes_actual_input_iterator_on_every_exit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_mode: str,
) -> None:
    frame = make_default_dataset_frame().iloc[:1].copy()

    class CloseableChunks:
        def __init__(self) -> None:
            self._chunks = iter((frame,))
            self.close_calls = 0

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._chunks)

        def close(self) -> None:
            self.close_calls += 1

    source = CloseableChunks()
    failure_types = {
        "runtime_error": RuntimeError,
        "keyboard_interrupt": KeyboardInterrupt,
        "system_exit": SystemExit,
        "close": GeneratorExit,
    }
    failure_type = failure_types.get(exit_mode)
    if failure_type is not None:

        def fail_write(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise failure_type("writer_exit")

        monkeypatch.setattr(dataset_writing, "write_parquet_file", fail_write)
    root = make_dataset_artifact_root(tmp_path, f"writer_close_{exit_mode}")
    if exit_mode == "target_exists":
        dataset_writing.write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
            resource_policy=active_runtime_resource_policy(),
        )
        with pytest.raises(ValueError, match="dataset_exists"):
            dataset_writing.write_dataset(
                root,
                source,
                spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
                resource_policy=active_runtime_resource_policy(),
            )
    elif failure_type is None:
        dataset_writing.write_dataset(
            root,
            source,
            spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
            resource_policy=active_runtime_resource_policy(),
        )
    else:
        with pytest.raises(failure_type, match="writer_exit"):
            dataset_writing.write_dataset(
                root,
                source,
                spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
                resource_policy=active_runtime_resource_policy(),
            )

    assert source.close_calls == 1


@pytest.mark.parametrize(
    ("exit_mode", "expected_error", "expected_message"),
    (
        ("success", OSError, "source_close_failure"),
        ("runtime_error", RuntimeError, "writer_exit"),
        ("keyboard_interrupt", KeyboardInterrupt, "writer_exit"),
        ("system_exit", SystemExit, "writer_exit"),
        ("target_exists", ValueError, "dataset_exists"),
    ),
)
def test_writer_close_failure_precedes_publish_but_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_mode: str,
    expected_error: type[BaseException],
    expected_message: str,
) -> None:
    frame = make_default_dataset_frame().iloc[:1].copy()

    class ThrowingCloseIterator:
        def __init__(self) -> None:
            self._chunks = iter((frame,))
            self.close_calls = 0

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._chunks)

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("source_close_failure")

    class IterableFacade:
        def __init__(self, source: ThrowingCloseIterator) -> None:
            self.source = source
            self.close_calls = 0

        def __iter__(self):  # noqa: ANN204
            return self.source

        def close(self) -> None:
            self.close_calls += 1

    source = ThrowingCloseIterator()
    facade = IterableFacade(source)
    failure_types = {
        "runtime_error": RuntimeError,
        "keyboard_interrupt": KeyboardInterrupt,
        "system_exit": SystemExit,
    }
    failure_type = failure_types.get(exit_mode)
    if failure_type is not None:

        def fail_write(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise failure_type("writer_exit")

        monkeypatch.setattr(dataset_writing, "write_parquet_file", fail_write)

    root = make_dataset_artifact_root(tmp_path, f"writer_close_failure_{exit_mode}")
    if exit_mode == "target_exists":
        dataset_writing.write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec(f"writer_close_failure_{exit_mode}"),
            resource_policy=active_runtime_resource_policy(),
        )

    with pytest.raises(expected_error, match=expected_message):
        dataset_writing.write_dataset(
            root,
            facade,
            spec=make_processed_dataset_spec(f"writer_close_failure_{exit_mode}"),
            resource_policy=active_runtime_resource_policy(),
        )

    assert source.close_calls == 1
    assert facade.close_calls == 0
    assert root.exists() is (exit_mode == "target_exists")


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


def test_categorical_converter_rejects_nonnull_value_outside_declared_domain() -> None:
    batch = pa.record_batch([pa.array(["known", "unknown", None])], names=["value"])

    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=4_096,
                operation_headroom_bytes=16_384,
                dtype_overrides={
                    "value": pd.CategoricalDtype(categories=["known"]),
                },
            )
        )


def test_converter_rejects_before_pandas_allocation_when_bound_exceeds_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch([pa.array([1])], names=["value"])
    converted = False

    def reject_conversion(_slice: pa.RecordBatch) -> pd.DataFrame:
        nonlocal converted
        converted = True
        raise AssertionError("Pandas conversion ran before conservative admission")

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        reject_conversion,
    )
    row_bytes, fixed_bytes = (
        dataset_physical.record_batch_decoded_working_set_bounds(batch)
    )
    conversion_bound = (
        fixed_bytes
        + dataset_reading._PANDAS_FRAME_FIXED_BYTES
        + row_bytes
        + len(batch.schema.names) * dataset_reading._PANDAS_OBJECT_CELL_BYTES
        + dataset_reading._PANDAS_INDEX_BYTES_PER_ROW
    )

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=4_096,
                operation_headroom_bytes=(
                    batch.get_total_buffer_size() + conversion_bound - 1
                ),
            )
        )
    assert not converted


@pytest.mark.parametrize(
    ("headroom_delta", "should_fail"),
    ((0, False), (-1, True)),
)
def test_converter_accounts_for_parent_buffers_and_previous_output(
    headroom_delta: int,
    should_fail: bool,
) -> None:
    batch = pa.record_batch(
        [
            pa.array(
                ["x" * 4_096, "y" * 4_096],
                type=pa.large_string(),
            )
        ],
        names=["value"],
    )
    expected = tuple(
        batch.slice(index, 1).to_pandas(use_threads=False)
        for index in range(batch.num_rows)
    )
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    target_chunk_bytes = max(output_bytes)
    row_bytes, fixed_bytes = (
        dataset_physical.record_batch_decoded_working_set_bounds(batch)
    )
    conversion_row_bytes = (
        row_bytes
        + len(batch.schema.names) * dataset_reading._PANDAS_OBJECT_CELL_BYTES
        + dataset_reading._PANDAS_INDEX_BYTES_PER_ROW
    )
    operation_headroom_bytes = (
        batch.get_total_buffer_size()
        + fixed_bytes
        + dataset_reading._PANDAS_FRAME_FIXED_BYTES
        + conversion_row_bytes
        + output_bytes[0]
        + headroom_delta
    )
    observed_slices: list[tuple[str, int]] = []

    def observe_batch(arrow_slice: pa.RecordBatch, frame: pd.DataFrame) -> None:
        observed_slices.append(
            (
                str(frame.loc[0, "value"]),
                arrow_slice.get_total_buffer_size(),
            )
        )

    converted = dataset_reading._arrow_batch_to_bounded_pandas(
        batch,
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=operation_headroom_bytes,
        batch_observer=observe_batch,
    )

    if should_fail:
        with pytest.raises(
            ValueError,
            match="dataset_working_set_exceeds_task_memory_budget",
        ):
            tuple(converted)
        return

    output = tuple(converted)
    assert len(output) == 2
    for observed, expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)
    assert observed_slices == [
        ("x" * 4_096, batch.get_total_buffer_size()),
        ("y" * 4_096, batch.get_total_buffer_size()),
    ]


def test_declared_member_preserves_output_across_parent_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batches = (
        pa.record_batch([pa.array(["x" * 4_096])], names=["value"]),
        pa.record_batch([pa.array(["y" * 4_096])], names=["value"]),
    )
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches)
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    target_chunk_bytes = max(output_bytes)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter(batches),
    )

    converted = dataset_reading.iter_declared_parquet_member_batches(
        path,
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=64 * 1024,
        expected_identity=SimpleNamespace(),
    )
    output = tuple(converted)
    for observed, expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)


def test_dataset_file_batches_preserve_output_across_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = (tmp_path / "part-0.parquet", tmp_path / "part-1.parquet")
    batches = {
        paths[0]: pa.record_batch([pa.array([1], type=pa.int64())], names=["value"]),
        paths[1]: pa.record_batch([pa.array([2], type=pa.int64())], names=["value"]),
    }
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches.values())
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    target_chunk_bytes = max(output_bytes)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda path, **_kwargs: iter((batches[Path(path)],)),
    )
    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        retained_memory_bytes=0,
        logical_dtypes=logical_dtype_payload_from_frame(expected[0]),
        parquet_filters=None,
        read_paths=paths,
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )

    converted = dataset_reading.iter_dataset_file_batches(
        context,
        columns=("value",),
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=64 * 1024,
    )
    output = tuple(converted)
    assert tuple(path for path, _ in output) == paths
    for (_, observed), expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)


def test_dataset_file_batches_subtract_live_root_reservations_before_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batch = pa.record_batch([pa.array([1], type=pa.int64())], names=["value"])
    expected = batch.to_pandas(use_threads=False)
    policy = _issued_test_policy(1024**2)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter((batch,)),
    )

    def reject_conversion(_batch: pa.RecordBatch) -> pd.DataFrame:
        raise AssertionError("Pandas conversion ran before shared-root admission")

    monkeypatch.setattr(dataset_reading, "_arrow_slice_to_pandas", reject_conversion)
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        parquet_filters=None,
        read_paths=(path,),
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )

    with runtime_resource_policy_scope(policy):
        with policy.reserve("research.time.pending", policy.task_memory_budget_bytes):
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                next(
                    dataset_reading.iter_dataset_file_batches(
                        context,
                        columns=("value",),
                        target_chunk_bytes=4_096,
                    )
                )


def test_dataset_file_batches_do_not_repeat_explicitly_composed_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batch = pa.record_batch([pa.array([1], type=pa.int64())], names=["value"])
    expected = batch.to_pandas(use_threads=False)
    output_bytes = pandas_frame_payload_bytes(expected)
    composed_headroom = 64 * 1024
    policy = _issued_test_policy()
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter((batch,)),
    )
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        parquet_filters=None,
        read_paths=(path,),
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )

    with runtime_resource_policy_scope(policy):
        with policy.reserve(
            "storage.dataset.write",
            policy.task_memory_budget_bytes - composed_headroom,
        ):
            output = tuple(
                dataset_reading.iter_dataset_file_batches(
                    context,
                    columns=("value",),
                    target_chunk_bytes=output_bytes,
                    operation_headroom_bytes=composed_headroom,
                )
            )

    assert len(output) == 1
    assert output[0][0] == path
    pd.testing.assert_frame_equal(output[0][1], expected)


def test_dataset_file_batches_do_not_credit_implicit_live_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batch = pa.record_batch([pa.array([1], type=pa.int64())], names=["value"])
    expected = batch.to_pandas(use_threads=False)
    retained_bytes = 256
    exact_headroom = batch.get_total_buffer_size() + pandas_frame_payload_bytes(
        expected
    )
    policy = _issued_test_policy(exact_headroom + retained_bytes)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter((batch,)),
    )

    def reject_conversion(_batch: pa.RecordBatch) -> pd.DataFrame:
        raise AssertionError("conversion ran after uncomposed reservation was credited")

    monkeypatch.setattr(dataset_reading, "_arrow_slice_to_pandas", reject_conversion)
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=retained_bytes,
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        parquet_filters=None,
        read_paths=(path,),
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )

    with runtime_resource_policy_scope(policy):
        with policy.reserve("research.model.fitted_state", retained_bytes):
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                next(
                    dataset_reading.iter_dataset_file_batches(
                        context,
                        columns=("value",),
                        target_chunk_bytes=pandas_frame_payload_bytes(expected),
                    )
                )


def test_dataset_file_batches_do_not_double_count_reserved_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batches = tuple(
        pa.record_batch([pa.array([index])], names=["value"])
        for index in range(2)
    )
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches)
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    policy = _issued_test_policy()
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter(batches),
    )
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        logical_dtypes=logical_dtype_payload_from_frame(expected[0]),
        parquet_filters=None,
        read_paths=(path,),
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )
    converted = dataset_reading.iter_dataset_file_batches(
        context,
        columns=("value",),
        target_chunk_bytes=max(output_bytes),
        operation_headroom_bytes=64 * 1024,
    )

    with dataset_reading.closing_dataset_iterator(converted):
        first_path, first = next(converted)
        with runtime_resource_policy_scope(policy):
            with policy.reserve("research.time.pending", output_bytes[0]):
                second_path, second = next(converted)

    assert first_path == path
    assert second_path == path
    pd.testing.assert_frame_equal(first, expected[0])
    pd.testing.assert_frame_equal(second, expected[1])


def test_dataset_file_batches_subtract_new_reservation_before_next_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batches = (
        pa.record_batch([pa.array(["x"])], names=["value"]),
        pa.record_batch([pa.array(["y" * (64 * 1024)])], names=["value"]),
    )
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches)
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    policy = _issued_test_policy()
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter(batches),
    )
    original_conversion = dataset_reading._arrow_slice_to_pandas
    conversion_count = 0

    def reject_second_conversion(batch: pa.RecordBatch) -> pd.DataFrame:
        nonlocal conversion_count
        conversion_count += 1
        if conversion_count == 2:
            raise AssertionError("second conversion ran before shared-root admission")
        return original_conversion(batch)

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        reject_second_conversion,
    )
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        logical_dtypes=logical_dtype_payload_from_frame(expected[0]),
        parquet_filters=None,
        read_paths=(path,),
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )
    converted = dataset_reading.iter_dataset_file_batches(
        context,
        columns=("value",),
        target_chunk_bytes=max(output_bytes),
    )

    with dataset_reading.closing_dataset_iterator(converted):
        next(converted)
        with runtime_resource_policy_scope(policy):
            remaining = (
                policy.task_memory_budget_bytes - policy.reserved_memory_bytes
            )
            assert remaining > 0
            with policy.reserve(
                "research.time.pending",
                remaining,
            ):
                with pytest.raises(
                    ValueError,
                    match="dataset_working_set_exceeds_task_memory_budget",
                ):
                    next(converted)

    assert conversion_count == 1


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
    assert all(pandas_frame_payload_bytes(frame) <= 64 * 1024 for frame in frames)


def test_dataset_file_batches_refresh_headroom_after_internal_release(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retained-output.parquet"
    row_count = 8_192
    expected = pd.DataFrame({"value": np.arange(row_count, dtype=np.int64)})
    pq.write_table(
        dataset_physical.table_with_decoded_row_bounds(pa.Table.from_pandas(expected)),
        path,
        row_group_size=row_count,
    )
    target_chunk_bytes = 16 * 1024
    headroom_bytes = 300 * 1024
    policy = _issued_test_policy(4 * 1024**2)
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        parquet_filters=None,
        read_paths=(path,),
        metadata_for=lambda _path: SimpleNamespace(
            identity=dataset_physical.dataset_file_identity(path)
        ),
        verify=lambda: None,
    )

    output = tuple(
        dataset_reading.iter_dataset_file_batches(
            context,
            columns=("value",),
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=headroom_bytes,
        )
    )
    frames = tuple(frame for _, frame in output)

    assert len(frames) < row_count // 10
    pd.testing.assert_frame_equal(
        pd.concat(frames, ignore_index=True),
        expected,
    )


def test_variable_width_decode_rejects_before_first_arrow_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "skewed-variable-width.parquet"
    values = ["x"] * 1023 + ["y" * 1024**2]
    pq.write_table(pa.table({"value": values}), path)
    real_parquet_file = dataset_physical.pq.ParquetFile
    pulled = False

    class ObservedParquetFile:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self._inner = real_parquet_file(*args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def iter_batches(self, **kwargs):  # noqa: ANN003, ANN202
            nonlocal pulled
            for batch in self._inner.iter_batches(**kwargs):
                pulled = True
                raise AssertionError("Arrow decode ran before conservative admission")
                yield batch

        def close(self) -> None:
            self._inner.close()

    monkeypatch.setattr(dataset_physical.pq, "ParquetFile", ObservedParquetFile)

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        list(
            dataset_physical.iter_dataset_part_batches(
                path,
                expected_identity=dataset_physical.dataset_file_identity(path),
                target_chunk_bytes=8 * 1024,
                operation_headroom_bytes=64 * 1024,
            )
        )

    assert not pulled


def test_categorical_converter_releases_temporary_codes_after_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch([pa.array(["known"] * 10_000)], names=["value"])
    code_references: list[weakref.ReferenceType[np.ndarray]] = []
    original_get_indexer = pd.Index.get_indexer

    def observe_get_indexer(
        index,
        target,
        *args,
        **kwargs,  # noqa: ANN001, ANN002, ANN003
    ):  # noqa: ANN202
        codes = original_get_indexer(index, target, *args, **kwargs)
        if index.tolist() == ["known"]:
            code_references.append(weakref.ref(codes))
        return codes

    monkeypatch.setattr(pd.Index, "get_indexer", observe_get_indexer)
    converted = dataset_reading._arrow_batch_to_bounded_pandas(
        batch,
        target_chunk_bytes=1024**2,
        operation_headroom_bytes=4 * 1024**2,
        dtype_overrides={
            "value": pd.CategoricalDtype(categories=["known"]),
        },
    )

    output = next(converted)
    gc.collect()

    assert not output.empty
    assert code_references
    assert all(reference() is None for reference in code_references)
    converted.close()


@pytest.mark.parametrize(
    ("values", "target_chunk_bytes"),
    (
        (tuple("x" * 32 for _ in range(512)), 16 * 1024),
        (tuple([*("x" * 32 for _ in range(511)), "y" * 8_000]), 16 * 1024),
    ),
    ids=("uniform", "skewed"),
)
def test_arrow_batch_converter_preflights_without_duplicate_observer_effects(
    values: tuple[str, ...],
    target_chunk_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_ids = tuple(range(len(values)))
    batch = pa.record_batch(
        [pa.array(row_ids), pa.array(values)],
        names=["row_id", "value"],
    )
    conversions = 0
    persisted_rows: list[int] = []
    logical_rows: list[int] = []
    original = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        nonlocal conversions
        conversions += 1
        return original(arrow_slice)

    monkeypatch.setattr(dataset_reading, "_arrow_slice_to_pandas", record_conversion)

    output = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=target_chunk_bytes,
            dtype_overrides={"value": "string"},
            persisted_frame_observer=lambda frame: persisted_rows.extend(
                frame["row_id"].tolist()
            ),
            batch_observer=lambda _batch, frame: logical_rows.extend(
                frame["row_id"].tolist()
            ),
        )
    )

    assert conversions < len(values) // 10
    assert len(output) < len(values) // 10
    assert persisted_rows == list(row_ids)
    assert logical_rows == list(row_ids)
    assert tuple(row for frame in output for row in frame["row_id"]) == row_ids
    assert all(str(frame["value"].dtype) == "string" for frame in output)
    assert all(
        pandas_frame_payload_bytes(frame) <= target_chunk_bytes for frame in output
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
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
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

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=TimeWindow("2099-01-01", "2099-01-02"),
    ) as empty_session:
        before_empty = calls
        assert (
            len(
                list(
                    dataset_reading.iter_dataset_chunks(
                        empty_session,
                        target_chunk_bytes=1024**2,
                    )
                )
            )
            == 1
        )
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
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as error_session:
        with pytest.raises(RuntimeError, match="injected_read_failure"):
            list(
                dataset_reading.iter_dataset_chunks(
                    error_session,
                    target_chunk_bytes=1024**2,
                )
            )
    assert calls > before_error


@pytest.mark.parametrize(
    "arrays",
    [
        [pa.nulls(1) for _ in range(500)],
        [pa.array([b"x"], type=pa.binary(1)) for _ in range(500)],
        [pa.array([Decimal("1.23")], type=pa.decimal128(10, 2)) for _ in range(500)],
        [pa.array([date(2026, 1, 1)], type=pa.date32()) for _ in range(500)],
    ],
    ids=["null", "fixed-binary", "decimal", "date"],
)
def test_wide_single_row_can_exceed_the_soft_chunk_target(
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

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=4_096,
        )
    )

    assert calls["count"] == 1
    assert len(chunks) == 1
    pd.testing.assert_frame_equal(chunks[0], batch.to_pandas(use_threads=False))


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
    assert (
        max(pandas_frame_payload_bytes(chunk) for chunk in chunks) <= (256 * 1024) // 4
    )


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
    assert (
        max(pandas_frame_payload_bytes(chunk) for chunk in chunks) <= (64 * 1024) // 4
    )


def test_materialized_context_fails_before_second_pass_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yielded = 0
    concat_called = False
    sample = pd.DataFrame({"value": pd.Series(["x"], dtype="string"), "part": [0]})

    def frames(*_args, **_kwargs):
        nonlocal yielded
        for index in range(4):
            yielded += 1
            yield pd.DataFrame(
                {
                    "value": pd.Series(["x" * 1_024], dtype="string"),
                    "part": [index],
                }
            )

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
            SimpleNamespace(
                resource_policy=_issued_test_policy(),
                logical_columns=tuple(sample.columns),
                logical_dtypes=logical_dtype_payload_from_frame(sample),
                inventory=SimpleNamespace(retained_memory_bytes=1024),
                retained_memory_bytes=1024,
            ),
            target_chunk_bytes=64 * 1024,
            materialized_budget_bytes=2_048,
        )

    assert yielded == 4
    assert concat_called is False


def test_materialized_session_accounts_for_concat_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame({"value": ["payload"]})
    budget = pandas_frame_payload_bytes(expected) * 2 + 1024

    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((expected,)),
    )
    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        logical_columns=tuple(expected.columns),
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
    )

    observed = dataset_reading.materialize_dataset_session(
        context,
        target_chunk_bytes=budget,
        materialized_budget_bytes=budget,
    )

    pd.testing.assert_frame_equal(observed, expected)


def test_materialized_nullable_and_string_fill_is_vectorized_at_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 1_000_000
    nullable = pd.Series(pd.array(np.arange(row_count), dtype="Int64"))
    nullable.iloc[::10] = pd.NA
    strings = pd.Series(
        pd.array(
            np.where(np.arange(row_count) % 10, "value", None),
            dtype="string",
        )
    )
    expected = pd.DataFrame({"nullable": nullable, "string": strings})
    dtype_plan = {
        "nullable": ("nullable_integer", nullable.dtype),
        "string": ("string", strings.dtype),
    }
    buffers = dataset_reading._allocate_materialized_buffers(
        dtype_plan,
        row_count=row_count,
        string_bytes={"string": dataset_reading._arrow_string_payload_bytes(strings)},
    )
    monkeypatch.setattr(
        dataset_reading.pd,
        "isna",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("materialized_fill_scalarized")
        ),
    )

    started = perf_counter()
    dataset_reading._fill_materialized_column(buffers["nullable"], nullable, start=0)
    dataset_reading._fill_materialized_column(buffers["string"], strings, start=0)
    duration = perf_counter() - started
    observed = dataset_reading._materialized_frame(buffers)

    pd.testing.assert_frame_equal(observed, expected)
    assert duration < 2.0


def test_materialized_second_pass_failure_closes_both_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame({"value": pd.Series([1, 2], dtype="int64")})
    opened = 0
    closed = 0

    def chunks(*_args, **_kwargs):
        nonlocal opened, closed
        opened += 1
        try:
            yield expected.iloc[:1]
            yield expected.iloc[1:]
        finally:
            closed += 1

    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        logical_columns=tuple(expected.columns),
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
    )
    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(
        dataset_reading,
        "_fill_materialized_column",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("second_pass_failed")
        ),
    )

    with pytest.raises(RuntimeError, match="second_pass_failed"):
        dataset_reading.materialize_dataset_session(
            context,
            target_chunk_bytes=1024,
            materialized_budget_bytes=4096,
        )

    assert opened == 2
    assert closed == 2


def test_materialized_passes_existing_headroom_to_each_conversion_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": pd.Series([1], dtype="int64")})
    observed_headroom: list[int | None] = []

    def chunks(
        _context: object,
        *,
        target_chunk_bytes: int,
        operation_headroom_bytes: int | None = None,
    ):
        del target_chunk_bytes
        observed_headroom.append(operation_headroom_bytes)
        yield frame

    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frame),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
    )
    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(
        dataset_reading,
        "_materialized_buffer_payload_bytes",
        lambda *_args, **_kwargs: 1024,
    )

    result = dataset_reading.materialize_dataset_session(
        context,
        target_chunk_bytes=1024,
        materialized_budget_bytes=4096,
    )

    assert result["value"].tolist() == [1]
    assert observed_headroom == [4096, 3072]


def test_materialized_budget_deducts_inventory_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("decoder_activated_without_inventory_headroom")
        ),
    )

    with pytest.raises(
        ValueError,
        match="dataset_materialized_result_exceeds_target_bytes",
    ):
        dataset_reading.materialize_dataset_session(
            SimpleNamespace(
                resource_policy=_issued_test_policy(),
                inventory=SimpleNamespace(retained_memory_bytes=1024),
                retained_memory_bytes=1024,
            ),
            target_chunk_bytes=1024,
            materialized_budget_bytes=1024,
        )


def test_materialized_zero_transition_headroom_skips_second_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": pd.Series([1], dtype="int64")})
    calls = 0

    def chunks(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        yield frame

    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frame),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
    )
    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(
        dataset_reading,
        "_materialized_buffer_payload_bytes",
        lambda *_args, **_kwargs: 3072,
    )
    monkeypatch.setattr(
        dataset_reading,
        "_allocate_materialized_buffers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("buffers_allocated_without_transition_headroom")
        ),
    )

    with pytest.raises(
        ValueError,
        match="dataset_materialized_result_exceeds_target_bytes",
    ):
        dataset_reading.materialize_dataset_session(
            context,
            target_chunk_bytes=4096,
            materialized_budget_bytes=4096,
        )

    assert calls == 1


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


@pytest.mark.parametrize(
    "reverse",
    (False, True),
    ids=("native-first", "nullable-first"),
)
@pytest.mark.parametrize(
    ("native_dtype", "nullable_dtype"),
    (("float32", "Float32"), ("float64", "Float64")),
)
def test_nullable_float_plan_is_chunk_order_independent(
    tmp_path: Path,
    reverse: bool,
    native_dtype: str,
    nullable_dtype: str,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path,
        f"nullable_{nullable_dtype.lower()}_{reverse}",
    )
    base = make_default_dataset_frame().iloc[:2].copy()
    native = base.iloc[:1].copy()
    nullable = base.iloc[1:].copy()
    native["value"] = pd.Series([1.5], index=native.index, dtype=native_dtype)
    nullable["value"] = pd.Series(
        [pd.NA],
        index=nullable.index,
        dtype=nullable_dtype,
    )
    chunks = (nullable, native) if reverse else (native, nullable)

    write_dataset(
        root,
        chunks,
        spec=make_processed_dataset_spec(
            f"nullable_{nullable_dtype.lower()}_{reverse}"
        ),
    )

    materialized = read_dataset(root)
    chunked = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    assert str(materialized["value"].dtype) == nullable_dtype
    assert {str(chunk["value"].dtype) for chunk in chunked} == {nullable_dtype}


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
        result = read_dataset(root, columns=[], mode=mode)
        if mode is DatasetReadMode.CHUNKED:
            list(result)


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
        chunk["asset_id"].dtype == materialized["asset_id"].dtype for chunk in chunks
    )


@pytest.mark.parametrize(
    "mode",
    (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED),
)
def test_reads_reject_nonnull_category_outside_declared_domain(
    tmp_path: Path,
    mode: DatasetReadMode,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"forged_read_domain_{mode.value}")
    frame = make_default_dataset_frame().iloc[:1].copy()
    known = str(frame["asset_id"].iloc[0])
    frame["asset_id"] = pd.Categorical([known], categories=[known])
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(f"forged_read_domain_{mode.value}"),
    )
    part = next((root / "data").rglob("part-*.parquet"))
    original = pq.ParquetFile(part).read()
    asset_index = original.schema.get_field_index("asset_id")
    asset_type = original.schema.field(asset_index).type
    forged_asset = pa.DictionaryArray.from_arrays(
        pa.array([0], type=asset_type.index_type),
        pa.array(["unknown"], type=asset_type.value_type),
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                forged_asset if index == asset_index else original.column(index)
                for index in range(original.num_columns)
            ],
            schema=original.schema.remove_metadata(),
        ).replace_schema_metadata(original.schema.metadata),
        part,
        compression="zstd",
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_size_bytes"] = sum(
        path.stat().st_size for path in (root / "data").rglob("part-*.parquet")
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        result = read_dataset(root, mode=mode)
        if mode is DatasetReadMode.CHUNKED:
            tuple(result)


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
            "row_hash_xor": int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0))),
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
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:01:00"]),
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


def test_exact_key_index_spills_sqlite_temp_state_to_disk(
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
        temp_store = int(index._connection.execute("PRAGMA temp_store").fetchone()[0])
        index.add_frame(pd.DataFrame({"key": ["a", "b"]}), namespace="left")
        index.add_frame(pd.DataFrame({"key": ["b", "c"]}), namespace="right")

        assert temp_store == 1
        assert index.difference_counts(
            expected_namespace="left",
            actual_namespace="right",
        ) == (1, 1)


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
        assert timestamps.add_frame(pd.DataFrame({"key": [datetime(2026, 1, 1)]})) == 1

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=pa.schema([pa.field("key", pa.string())]),
    ) as strings:
        assert strings.add_frame(pd.DataFrame({"key": ["é"]})) == 0
        assert strings.add_frame(pd.DataFrame({"key": ["e\u0301"]})) == 0


def test_exact_primary_key_accepts_timestamp_outside_nanosecond_range(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "quant_research_platform.storage._dataset_key_tracking"
    )
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
    for name in ("k.db", "k.db-journal", "k.db-wal", "k.db-shm"):
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
    (reused / "k.db").write_bytes(b"scratch")
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
        "k.db",
        "k.db-journal",
        "k.db-wal",
        "k.db-shm",
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
    (live_scratch / "k.db").write_bytes(b"live")

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
    (reused_scratch / "k.db").write_bytes(b"scratch")
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        rss_bytes=0,
    )
    monkeypatch.setattr(
        ownership.psutil,
        "Process",
        lambda _pid: SimpleNamespace(
            create_time=lambda: reused_stage.stat().st_ctime + 60,
            is_running=lambda: True,
        ),
    )

    dataset_writing.write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("reused_pid_stage"),
        resource_policy=policy,
    )

    assert not reused_scratch.exists()


def test_dataset_write_recovers_dead_stage_when_final_exists(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "published_with_dead_stage")
    spec = make_processed_dataset_spec("published_with_dead_stage")
    write_dataset(root, make_default_dataset_frame(), spec=spec)
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(root)
    stale_stage = root.with_name(f"{scratch_prefix}qzzzzzz-stale")
    Path(f"{stale_stage}.target").write_text(
        storage_atomic._atomic_target_identity(root),
        encoding="utf-8",
    )
    stale_scratch = stale_stage / ".k-qzzzzzz-stale"
    stale_scratch.mkdir(parents=True)
    (stale_scratch / "k.db").write_bytes(b"scratch")

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(root, make_default_dataset_frame(), spec=spec)

    assert not stale_stage.exists()


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
        assert index.retained_memory_bytes == 12 * 1024

    assert cache_size == -12
    assert (
        signature(module.ExactPrimaryKeyIndex).parameters["sqlite_cache_bytes"].default
        == module.EXACT_KEY_SQLITE_PAGE_BYTES
    )


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
    canonical_frame = frame.copy()
    canonical_frame["nested"] = canonical_frame["nested"].map(
        dataset_fingerprinting._canonical_nested_value
    )
    target_bytes = (
        pandas_frame_payload_bytes(frame)
        + pandas_frame_payload_bytes(canonical_frame)
        - 1
    )
    calls = {"count": 0}
    canonicalize = dataset_fingerprinting._canonical_nested_value

    def record_canonicalization(
        value: object,
        *,
        max_bytes: int | None = None,
        data_type: pa.DataType | None = None,
    ) -> bytes:
        calls["count"] += 1
        return canonicalize(value, max_bytes=max_bytes, data_type=data_type)

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_canonical_nested_value",
        record_canonicalization,
    )

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        tuple(
            dataset_fingerprinting._iter_fingerprint_hashes(
                frame,
                target_bytes=target_bytes,
            )
        )
    assert calls["count"] == 1


def test_nested_fingerprint_stops_writer_traversal_before_oversized_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"nested": [[None] * 5_000]})
    write_identity = dataset_fingerprinting._write_nested_identity
    calls = 0

    def counted_write(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls > 64:
            raise AssertionError("fingerprint_writer_traversed_past_target")
        write_identity(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_write_nested_identity",
        counted_write,
    )
    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        tuple(
            dataset_fingerprinting._iter_fingerprint_hashes(
                frame,
                target_bytes=pandas_frame_payload_bytes(frame) + 1024,
            )
        )
    assert calls <= 64


def test_ndarray_fingerprint_does_not_make_a_full_object_list_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = bytes(bytearray(b"ndarray-object-sentinel"))
    value = np.empty((100, 1_000), dtype=object)
    value.fill(sentinel)
    baseline_references = sys.getrefcount(sentinel)
    observed_reference_deltas: list[int] = []
    write_identity = dataset_fingerprinting._write_nested_identity

    def record_references(
        writer: object, item: object, *args: object, **kwargs: object
    ) -> None:
        if item is sentinel:
            observed_reference_deltas.append(
                sys.getrefcount(sentinel) - baseline_references
            )
        write_identity(writer, item, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_write_nested_identity",
        record_references,
    )
    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._canonical_nested_value(
            value,
            max_bytes=100_000,
        )
    assert observed_reference_deltas
    assert max(observed_reference_deltas) < 10_000


def test_ndarray_fingerprint_preserves_shape_dtype_order_and_nested_identity() -> None:
    canonicalize = dataset_fingerprinting._canonical_nested_value
    row = np.array([[1, 2]], dtype=np.int32)

    assert canonicalize(row) != canonicalize(row.astype(np.int64))
    assert canonicalize(row) != canonicalize(row.reshape(2, 1))
    assert canonicalize(row) != canonicalize(np.array([[2, 1]], dtype=np.int32))
    nested = np.empty((2,), dtype=object)
    nested[:] = [[1, 2], (1, 2)]
    reversed_nested = nested[::-1].copy()
    assert canonicalize(nested) != canonicalize(reversed_nested)


def test_fingerprint_writer_preserves_canonical_model_golden_bytes() -> None:
    assert dataset_fingerprinting._canonical_nested_value([1, "x"]) == (
        canonical_identity_bytes(("list", (("int", 1), ("string", "x"))))
    )
    assert dataset_fingerprinting._canonical_nested_value(
        {"b": 2, "a": 1}
    ) == canonical_identity_bytes(
        (
            "dict",
            (
                (("string", "a"), ("int", 1)),
                (("string", "b"), ("int", 2)),
            ),
        )
    )


@pytest.mark.parametrize(
    "value",
    (
        [Decimal("7" * 4_097)],
        (Decimal("7" * 4_097),),
        {Decimal("7" * 4_097): "key"},
        {"value": Decimal("7" * 4_097)},
    ),
    ids=("list", "tuple", "mapping-key", "mapping-value"),
)
def test_fingerprint_decimal_ceiling_maps_to_public_capacity_error(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._canonical_nested_value(value, max_bytes=500_000)


def test_fingerprint_decimal_ceiling_applies_to_arrow_map() -> None:
    value = [(Decimal("7" * 4_097), "value")]
    data_type = pa.map_(pa.decimal256(76, 0), pa.string())

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._canonical_nested_value(
            value,
            max_bytes=500_000,
            data_type=data_type,
        )


def test_fingerprint_multi_column_stops_at_oversized_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = [Decimal("7" * 4_097)]
    later = ["later"]
    frame = pd.DataFrame({"first": [oversized], "second": [later]})
    canonicalize = dataset_fingerprinting._canonical_nested_value
    observed: list[object] = []

    def record_value(
        value: object,
        *,
        max_bytes: int | None = None,
        data_type: pa.DataType | None = None,
    ) -> bytes:
        observed.append(value)
        return canonicalize(value, max_bytes=max_bytes, data_type=data_type)

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_canonical_nested_value",
        record_value,
    )

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._normalized_fingerprint_slice(
            frame,
            arrow_schema=None,
            primary_key=(),
            identity_target_bytes=500_000,
        )
    assert observed == [oversized]


def test_fingerprint_python_allocation_diagnostic_rejects_full_snapshot() -> None:
    value = {f"key-{index:05d}": index for index in range(20_000)}

    tracemalloc.start()
    try:
        with pytest.raises(
            ValueError,
            match="dataset_fingerprint_value_exceeds_target_bytes",
        ):
            dataset_fingerprinting._canonical_nested_value(value, max_bytes=1024)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak_bytes < 64 * 1024


def test_fingerprint_depth_uses_contract_owner_and_stable_error() -> None:
    value: object = None
    for _ in range(300):
        value = [value]

    with pytest.raises(ValueError, match="dataset_fingerprint_value_too_deep"):
        dataset_fingerprinting._canonical_nested_value(value)


@pytest.mark.parametrize("family", ("list", "tuple", "dict", "arrow-map", "ndarray"))
def test_fingerprint_nested_families_share_contract_depth_error(family: str) -> None:
    value: object = None
    for _ in range(40):
        if family == "tuple":
            value = (value,)
        elif family == "dict":
            value = {"key": value}
        else:
            value = [value]

    data_type: pa.DataType | None = None
    if family == "arrow-map":
        item_type: pa.DataType = pa.null()
        for _ in range(40):
            item_type = pa.list_(item_type)
        data_type = pa.map_(pa.string(), item_type)
        value = [("key", value)]
    elif family == "ndarray":
        array = np.empty((1,) * 40, dtype=object)
        array[...] = None
        value = array

    with pytest.raises(ValueError, match="dataset_fingerprint_value_too_deep"):
        dataset_fingerprinting._canonical_nested_value(value, data_type=data_type)


def test_bytes_fingerprint_rejects_in_bounded_contract_writer() -> None:
    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting._canonical_nested_value(
            b"x" * (8 * 1024**2),
            max_bytes=1024,
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


def test_fingerprint_nested_normalization_checks_actual_payloads() -> None:
    frame = pd.DataFrame({"nested": [["x" * 50] * 4]})
    canonical = frame.copy()
    canonical["nested"] = canonical["nested"].map(
        dataset_fingerprinting._canonical_nested_value
    )
    input_payload_bytes = pandas_frame_payload_bytes(frame)
    canonical_value_bytes = len(canonical.loc[0, "nested"])
    actual_payload_bytes = (
        input_payload_bytes
        + max(pandas_frame_payload_bytes(canonical), canonical_value_bytes * 2)
        + 8
    )

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        tuple(
            dataset_fingerprinting._iter_fingerprint_hashes(
                frame,
                target_bytes=input_payload_bytes,
            )
        )

    observed = tuple(
        dataset_fingerprinting._iter_fingerprint_hashes(
            frame,
            target_bytes=actual_payload_bytes,
        )
    )
    expected = pd.util.hash_pandas_object(canonical, index=False).to_numpy("uint64")
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], expected)


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
    ("headroom_delta", "should_fail"),
    ((0, False), (-1, True)),
)
def test_fingerprint_accounts_for_live_arrow_and_nested_pandas_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headroom_delta: int,
    should_fail: bool,
) -> None:
    frame = pd.DataFrame({"nested": [["x" * 50] * 1_000]})
    arrow_batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    canonical = frame.copy()
    canonical["nested"] = canonical["nested"].map(
        dataset_fingerprinting._canonical_nested_value
    )
    canonical_value_bytes = len(canonical.loc[0, "nested"])
    aggregate_bytes = (
        arrow_batch.get_total_buffer_size()
        + pandas_frame_payload_bytes(frame)
        + max(pandas_frame_payload_bytes(canonical), canonical_value_bytes * 2)
        + 8
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    path = tmp_path / "data" / "year=2026" / "part-000.parquet"
    observer_calls = 0
    observed_headroom: list[int | None] = []

    def batches(*_args: object, **kwargs: object):
        nonlocal observer_calls
        observed_headroom.append(kwargs.get("operation_headroom_bytes"))
        observer = kwargs["batch_observer"]
        assert callable(observer)
        observer_calls += 1
        observer(arrow_batch, frame)
        yield path, frame

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        batches,
    )
    context = SimpleNamespace(
        resource_policy=policy,
        spec={"primary_key": ["nested"]},
        logical_columns=("nested",),
        arrow_schema=arrow_batch.schema,
        time_window=None,
        data_root=tmp_path / "data",
        schema_hash="schema",
        read_paths=(path,),
        inventory=SimpleNamespace(retained_memory_bytes=0),
        retained_memory_bytes=0,
    )
    partition_state_bytes = dataset_reading.dataset_partition_state_reserve_bytes(
        context
    )
    local_state_bytes = 3 * (struct.calcsize("P") + len(b"nested"))
    operation_headroom = aggregate_bytes + headroom_delta
    retained_memory_bytes = policy.task_memory_budget_bytes - (
        operation_headroom + partition_state_bytes + local_state_bytes
    )
    assert retained_memory_bytes > 0
    context.inventory.retained_memory_bytes = retained_memory_bytes
    context.retained_memory_bytes = retained_memory_bytes

    if should_fail:
        with pytest.raises(
            ValueError,
            match="dataset_fingerprint_value_exceeds_target_bytes",
        ):
            dataset_fingerprinting.fingerprint_dataset_session(
                context,
                resource_policy=policy,
            )
    else:
        fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
            context,
            resource_policy=policy,
        )
        hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy("uint64")
        expected_sum = int(hashes.sum(dtype=np.uint64))
        expected_xor = int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
        assert fingerprint.content_hash == canonical_json_hash(
            {
                "row_hash_sum": expected_sum,
                "row_hash_xor": expected_xor,
                "row_count": 1,
                "columns": ["nested"],
                "hash_columns": ["nested"],
            }
        )
        assert fingerprint.partition_fingerprints == (
            {
                "partition": "year=2026",
                "row_count": 1,
                "content_hash": canonical_json_hash(
                    {
                        "row_hash_sum": expected_sum,
                        "row_hash_xor": expected_xor,
                        "row_count": 1,
                        "hash_columns": ["nested"],
                    }
                ),
            },
        )

    assert observed_headroom == [operation_headroom + retained_memory_bytes]
    assert observer_calls == 1


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
        time_window=None,
        data_root=tmp_path / "data",
        schema_hash="schema",
        read_paths=paths,
        inventory=SimpleNamespace(retained_memory_bytes=0),
        retained_memory_bytes=0,
    )

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context,
        resource_policy=policy,
    )

    assert fingerprint.row_count == 2 * len(frame)
    assert released_before_next_batch == [True]


def test_fingerprint_accounts_for_multislice_parent_arrow_buffers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"nested": [["x" * 50] * 4 for _ in range(20_000)]})
    arrow_batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=16 * 1024 * 1024,
    )
    measured: list[tuple[int, int, int]] = []

    def measure_batch(batch: pa.RecordBatch, observed: pd.DataFrame) -> None:
        canonical = observed.copy()
        canonical["nested"] = canonical["nested"].map(
            dataset_fingerprinting._canonical_nested_value
        )
        measured.append(
            (
                batch.get_total_buffer_size(),
                pandas_frame_payload_bytes(observed),
                pandas_frame_payload_bytes(canonical),
            )
        )

    tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            arrow_batch,
            target_chunk_bytes=policy.remaining_memory_bytes,
            batch_observer=measure_batch,
        )
    )
    assert {item[0] for item in measured} == {arrow_batch.get_total_buffer_size()}
    canonical_row = frame.iloc[:1].copy()
    canonical_row["nested"] = canonical_row["nested"].map(
        dataset_fingerprinting._canonical_nested_value
    )
    exact_headroom = max(
        arrow_bytes + frame_bytes + pandas_frame_payload_bytes(canonical_row) + 8
        for arrow_bytes, frame_bytes, _canonical_bytes in measured
    )
    assert exact_headroom < max(sum(item) for item in measured)
    assert exact_headroom < policy.task_memory_budget_bytes
    path = tmp_path / "data" / "year=2026" / "part-000.parquet"
    observed_slices: list[int] = []

    def batches(*_args: object, **kwargs: object):
        observer = kwargs["batch_observer"]

        def observe_batch(batch: pa.RecordBatch, observed: pd.DataFrame) -> None:
            observed_slices.append(batch.get_total_buffer_size())
            observer(batch, observed)

        for observed in dataset_reading._arrow_batch_to_bounded_pandas(
            arrow_batch,
            target_chunk_bytes=kwargs["target_chunk_bytes"],
            operation_headroom_bytes=kwargs["operation_headroom_bytes"],
            batch_observer=observe_batch,
        ):
            yield path, observed

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        batches,
    )

    def context_for(headroom_bytes: int) -> SimpleNamespace:
        context = SimpleNamespace(
            resource_policy=policy,
            spec={"primary_key": ["nested"]},
            logical_columns=("nested",),
            arrow_schema=arrow_batch.schema,
            time_window=None,
            data_root=tmp_path / "data",
            schema_hash="schema",
            read_paths=(path,),
            inventory=SimpleNamespace(retained_memory_bytes=0),
            retained_memory_bytes=0,
        )
        reserve = dataset_reading.dataset_partition_state_reserve_bytes(context)
        reserve += 3 * (struct.calcsize("P") + len(b"nested"))
        retained = policy.task_memory_budget_bytes - headroom_bytes - reserve
        context.inventory.retained_memory_bytes = retained
        context.retained_memory_bytes = retained
        return context

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context_for(exact_headroom + 4096),
        resource_policy=policy,
    )
    assert fingerprint.row_count == len(frame)
    assert len(observed_slices) > 1
    assert set(observed_slices) == {arrow_batch.get_total_buffer_size()}


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
        inventory=SimpleNamespace(
            retained_memory_bytes=policy.task_memory_budget_bytes
        ),
        retained_memory_bytes=policy.task_memory_budget_bytes,
    )

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_fingerprinting.fingerprint_dataset_session(
            context,
            resource_policy=policy,
        )

    assert traversals == 0


def test_fingerprint_reserves_many_partition_counters_before_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024,
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
        retained_memory_bytes=0,
        data_root=data_root,
        read_paths=paths,
        spec={"primary_key": ["id"]},
        logical_columns=("id",),
        arrow_schema=batch.schema,
        time_window=None,
        schema_hash="schema",
    )
    partition_state_bytes = dataset_reading.dataset_partition_state_reserve_bytes(
        context
    )
    local_state_bytes = 3 * (struct.calcsize("P") + len(b"id"))
    context.retained_memory_bytes = policy.task_memory_budget_bytes - (
        partition_state_bytes + local_state_bytes + 1024
    )

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_fingerprinting.fingerprint_dataset_session(
            context,
            resource_policy=policy,
        )

    assert traversals == len(paths)


def test_iter_bounded_frame_chunks_reuses_bounded_frame_storage() -> None:
    frame = pd.DataFrame({"value": np.arange(20_000, dtype="int64")})

    chunks = tuple(
        dataset_reading.iter_bounded_frame_chunks(
            frame,
            target_chunk_bytes=pandas_frame_payload_bytes(frame) // 2,
        )
    )

    assert len(chunks) >= 2
    assert all(
        np.shares_memory(
            chunk["value"].to_numpy(copy=False),
            frame["value"].to_numpy(copy=False),
        )
        for chunk in chunks
    )


def test_numeric_summary_observer_skips_non_float_columns_and_preserves_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_dtypes: list[np.dtype[object]] = []
    original_isinf = dataset_observation.np.isinf

    def record_isinf(values: np.ndarray) -> np.ndarray:
        observed_dtypes.append(values.dtype)
        return original_isinf(values)

    monkeypatch.setattr(dataset_observation.np, "isinf", record_isinf)
    observer = dataset_observation.DatasetSummaryObserver(
        primary_key=("key",),
        timestamp_column=None,
        column_roles={"key": "key", "integer": "factor", "floating": "label"},
        exact_key_index=SimpleNamespace(add_frame=lambda _frame: 0),
    )
    frame = pd.DataFrame(
        {
            "key": [1, 2],
            "integer": pd.Series([1, 2], dtype="int64"),
            "floating": pd.Series([1.0, np.inf], dtype="float32"),
        }
    )

    observer.observe_frame(frame)

    assert observed_dtypes == [np.dtype("float32")]
    assert observer.non_finite_count_summary == {"floating": 1}


def test_public_storage_operations_delegate_admission_to_storage_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated_policies: list[RuntimeResourcePolicy | None] = []
    sentinel = object()

    def delegate(*_args, **kwargs):
        delegated_policies.append(kwargs["resource_policy"])
        return sentinel

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
    assert delegated_policies == [None] * len(calls)


def test_storage_operations_propagate_the_captured_payload_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    root = make_dataset_artifact_root(tmp_path, "policy_propagation")
    frame = make_default_dataset_frame()
    writer_targets: list[int] = []
    read_targets: list[int] = []
    inspection_targets: list[int] = []
    fingerprint_physical_targets: list[int] = []
    fingerprint_operation_headrooms: list[int] = []
    fingerprint_arrow_bytes: list[int] = []
    fingerprint_hash_targets: list[int] = []

    writer_decode = dataset_writing.iter_declared_parquet_member_batches
    read_chunks = dataset_reading.iter_dataset_chunks
    inspect_batches = dataset_inspection.iter_dataset_file_batches
    fingerprint_batches = dataset_fingerprinting.iter_dataset_file_batches
    fingerprint_hashes = dataset_fingerprinting._iter_fingerprint_hashes

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
        fingerprint_operation_headrooms.append(kwargs["operation_headroom_bytes"])
        observer = kwargs["batch_observer"]

        def record_arrow_bytes(batch: pa.RecordBatch, observed: pd.DataFrame) -> None:
            fingerprint_arrow_bytes.append(batch.get_total_buffer_size())
            observer(batch, observed)

        kwargs["batch_observer"] = record_arrow_bytes
        yield from fingerprint_batches(*args, **kwargs)

    def record_fingerprint_hash_target(*args, **kwargs):
        fingerprint_hash_targets.append(kwargs["target_bytes"])
        yield from fingerprint_hashes(*args, **kwargs)

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
        "_iter_fingerprint_hashes",
        record_fingerprint_hash_target,
    )

    with runtime_resource_policy_scope(policy):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("policy_propagation"),
        )
        dataset_reading.read_dataset(
            root,
            mode=DatasetReadMode.MATERIALIZED,
            resource_policy=policy,
        )
        list(
            dataset_reading.read_dataset(
                root,
                mode=DatasetReadMode.CHUNKED,
                resource_policy=policy,
            )
        )
        inspect_dataset(
            root,
            time_window=TimeWindow("2014-01-01", "2015-12-31"),
        )
        fingerprint_dataset(root)

    assert writer_targets and all(
        0 < target <= policy.task_memory_budget_bytes for target in writer_targets
    )
    assert read_targets and all(
        0 < target <= policy.task_memory_budget_bytes for target in read_targets
    )
    assert inspection_targets and all(
        0 < target <= policy.task_memory_budget_bytes
        for target in inspection_targets
    )
    assert fingerprint_physical_targets and all(
        0 < target <= policy.task_memory_budget_bytes
        for target in fingerprint_physical_targets
    )
    assert fingerprint_hash_targets
    assert len(fingerprint_operation_headrooms) == 1
    assert all(
        0 < target <= fingerprint_operation_headrooms[0] - arrow_bytes
        for target, arrow_bytes in zip(
            fingerprint_hash_targets,
            fingerprint_arrow_bytes,
            strict=True,
        )
    )


def test_file_batch_default_derives_converter_headroom_from_session_owner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": [1]})
    path = Path("part-000.parquet")
    observed_headroom: list[int | None] = []

    def declared_batches(*_args, **kwargs):  # noqa: ANN003
        observed_headroom.append(kwargs.get("operation_headroom_bytes"))
        yield frame

    context = SimpleNamespace(
        logical_dtypes=logical_dtype_payload_from_frame(frame),
        resource_policy=_issued_test_policy(4096),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
        read_paths=(path,),
        parquet_filters=None,
        metadata_for=lambda _path: SimpleNamespace(identity=object()),
        verify=lambda: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "iter_declared_parquet_member_batches",
        declared_batches,
    )

    batches = tuple(
        dataset_reading.iter_dataset_file_batches(
            context,
            columns=("value",),
            target_chunk_bytes=2048,
        )
    )

    assert len(batches) == 1
    assert batches[0][0] == path
    pd.testing.assert_frame_equal(batches[0][1], frame)
    assert observed_headroom == [3072]


def test_compare_reuses_one_policy_for_both_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    observed: list[RuntimeResourcePolicy] = []
    active_contexts = 0
    maximum_active_contexts = 0

    def fake_fingerprint(*_args, resource_policy, **_kwargs):
        assert active_contexts == 1
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

    class Context:
        def __init__(self, root: Path) -> None:
            self.root = root
            self.retained_memory_bytes = 0

        def __enter__(self):  # noqa: ANN204
            nonlocal active_contexts, maximum_active_contexts
            active_contexts += 1
            maximum_active_contexts = max(maximum_active_contexts, active_contexts)
            return self

        def __exit__(self, *_args):  # noqa: ANN204
            nonlocal active_contexts
            active_contexts -= 1
            return None

    monkeypatch.setattr(
        dataset_comparison,
        "open_dataset_session",
        lambda root, **_kwargs: Context(root),
    )
    monkeypatch.setattr(
        dataset_comparison,
        "fingerprint_dataset_session",
        fake_fingerprint,
    )
    monkeypatch.setattr(dataset_comparison, "_session_identity_token", lambda _: b"id")
    monkeypatch.setattr(
        dataset_comparison,
        "_exact_comparison_counts",
        lambda *_args, **_kwargs: {
            "missing_row_count": 0,
            "extra_row_count": 0,
            "changed_row_count": 0,
        },
    )

    comparison = dataset_comparison.compare_datasets(
        left,
        right,
        resource_policy=policy,
    )

    assert comparison.status == "MATCH"
    assert observed == [policy, policy]
    assert maximum_active_contexts == 1
    assert active_contexts == 0


def test_compare_requires_exact_proof_after_forced_fingerprint_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "collision_left")
    right = make_dataset_artifact_root(tmp_path, "collision_right")
    write_dataset(
        left,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("collision_left"),
    )
    changed = make_default_dataset_frame()
    changed.loc[changed.index[-1], "value"] = 99.0
    write_dataset(
        right,
        changed,
        spec=make_processed_dataset_spec("collision_right"),
    )
    real_open = dataset_comparison.open_dataset_session
    real_index = dataset_comparison.ExactPrimaryKeyIndex
    active_indexes: list[dataset_comparison.ExactPrimaryKeyIndex] = []
    acquisitions: list[tuple[int, int | None]] = []

    class RecordingIndex(real_index):
        def __enter__(self):  # noqa: ANN204
            opened = super().__enter__()
            active_indexes.append(self)
            return opened

        def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001, ANN204
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                active_indexes.remove(self)

    @contextmanager
    def record_open(*args, retained_payload_bytes=0, **kwargs):  # noqa: ANN002, ANN003
        expected = active_indexes[-1].retained_memory_bytes if active_indexes else None
        acquisitions.append((retained_payload_bytes, expected))
        with real_open(
            *args,
            retained_payload_bytes=retained_payload_bytes,
            **kwargs,
        ) as session:
            yield session

    monkeypatch.setattr(dataset_comparison, "ExactPrimaryKeyIndex", RecordingIndex)
    monkeypatch.setattr(dataset_comparison, "open_dataset_session", record_open)
    collision = fingerprint_dataset(left)
    monkeypatch.setattr(
        dataset_comparison,
        "fingerprint_dataset_session",
        lambda *_args, **_kwargs: collision,
    )

    comparison = dataset_comparison.compare_datasets(
        left,
        right,
        resource_policy=None,
    )

    assert comparison.status == "CONTENT_MISMATCH"
    assert comparison.summary["exact"] == {
        "missing_row_count": 0,
        "extra_row_count": 0,
        "changed_row_count": 1,
    }
    exact_acquisitions = [pair for pair in acquisitions if pair[1] is not None]
    assert len(exact_acquisitions) == 4
    assert all(retained > expected for retained, expected in exact_acquisitions)
    assert len({retained - expected for retained, expected in exact_acquisitions}) == 1


def test_compare_treats_partition_and_file_boundaries_as_layout_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "layout_left")
    right = make_dataset_artifact_root(tmp_path, "layout_right")
    frame = make_default_dataset_frame()
    write_dataset(
        left,
        iter((frame.iloc[:1], frame.iloc[1:])),
        spec=make_processed_dataset_spec("layout_left"),
    )
    write_dataset(
        right,
        frame,
        spec=make_processed_dataset_spec("layout_right"),
    )
    fingerprint = fingerprint_dataset(left)
    right_fingerprint = replace(
        fingerprint,
        partition_fingerprints=(
            {
                "partition": "layout-only",
                "row_count": fingerprint.row_count,
                "content_hash": fingerprint.content_hash,
            },
        ),
        file_count=1,
    )
    fingerprints = iter((fingerprint, right_fingerprint))
    monkeypatch.setattr(
        dataset_comparison,
        "fingerprint_dataset_session",
        lambda *_args, **_kwargs: next(fingerprints),
    )

    comparison = compare_datasets(left, right)

    assert comparison.status == "MATCH"
    assert comparison.matches
    assert comparison.layout_delta["file_count"] == {"left": 3, "right": 1}
    assert comparison.layout_delta["partitions"]["left"]
    assert comparison.layout_delta["partitions"]["right"]
    assert all(
        set(partition) == {"partition", "row_count"}
        for side in ("left", "right")
        for partition in comparison.layout_delta["partitions"][side]
    )
    assert (
        comparison.layout_delta["partitions"]["left"]
        != comparison.layout_delta["partitions"]["right"]
    )


def test_nested_fingerprint_identity_is_typed_ordered_and_bounded() -> None:
    canonicalize = dataset_fingerprinting._canonical_nested_value

    assert canonicalize({1: "x"}) != canonicalize({"1": "x"})
    assert canonicalize({"a": 1, "b": 2}) == canonicalize({"b": 2, "a": 1})
    assert canonicalize(np.int64(1)) != canonicalize(1)
    assert canonicalize([1]) != canonicalize((1,))
    assert canonicalize([1]) != canonicalize(np.array([1], dtype="int64"))
    assert canonicalize(-0.0) != canonicalize(0.0)
    assert canonicalize(pd.Timestamp("2026-01-01T00:00:00Z")) != canonicalize(
        pd.Timestamp("2026-01-01T08:00:00+08:00")
    )
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="dataset_fingerprint_value_cycle"):
        canonicalize(cyclic)
    first_nan = float("nan")
    second_nan = float("nan")
    colliding_keys = {first_nan: "first", second_nan: "second"}
    assert len(colliding_keys) == 2
    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_mapping_key_collision",
    ):
        canonicalize(colliding_keys)


def test_arrow_map_fingerprint_uses_physical_type_and_preserves_multiplicity() -> None:
    map_type = pa.map_(pa.string(), pa.float64())
    left_batch = pa.record_batch(
        [pa.array([[("b", math.nan), ("a", 1.0)]], type=map_type)],
        names=["value"],
    )
    right_batch = pa.record_batch(
        [pa.array([[("a", 1.0), ("b", math.nan)]], type=map_type)],
        names=["value"],
    )
    duplicate_batch = pa.record_batch(
        [pa.array([[("a", 1.0), ("a", 1.0)]], type=map_type)],
        names=["value"],
    )
    single_batch = pa.record_batch(
        [pa.array([[("a", 1.0)]], type=map_type)],
        names=["value"],
    )

    def hashes(batch: pa.RecordBatch) -> np.ndarray:
        frame = batch.to_pandas()
        return next(
            dataset_fingerprinting._iter_fingerprint_hashes(
                frame,
                target_bytes=1024**2,
                arrow_schema=batch.schema,
            )
        )

    np.testing.assert_array_equal(hashes(left_batch), hashes(right_batch))
    assert not np.array_equal(hashes(duplicate_batch), hashes(single_batch))
    assert dataset_fingerprinting._canonical_nested_value(
        [("b", 2), ("a", 1)]
    ) != dataset_fingerprinting._canonical_nested_value([("a", 1), ("b", 2)])


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit),
)
def test_dataset_session_close_preserves_active_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_primary")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_primary"),
    )
    close = dataset_reading.DatasetSession.close

    def close_then_fail(current: dataset_reading.DatasetSession) -> None:
        close(current)
        raise OSError("cleanup_failure")

    monkeypatch.setattr(
        dataset_reading.DatasetSession,
        "close",
        close_then_fail,
    )

    with pytest.raises(primary_type, match="primary_failure"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            raise primary_type("primary_failure")


def test_dataset_session_normal_close_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_normal_close")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_normal_close"),
    )
    close = dataset_reading.DatasetSession.close

    def close_then_fail(current: dataset_reading.DatasetSession) -> None:
        close(current)
        raise OSError("cleanup_failure")

    monkeypatch.setattr(
        dataset_reading.DatasetSession,
        "close",
        close_then_fail,
    )

    with pytest.raises(OSError, match="cleanup_failure"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            pass


@pytest.mark.parametrize("stage", ("fingerprint", "exact"))
def test_compare_cleanup_does_not_replace_programmer_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"compare_primary_{stage}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"compare_primary_{stage}"),
    )
    close = dataset_reading.DatasetSession.close

    def close_with_active_failure(session: dataset_reading.DatasetSession) -> None:
        close(session)
        if sys.exc_info()[0] is not None:
            raise OSError("cleanup_failure")

    monkeypatch.setattr(
        dataset_reading.DatasetSession,
        "close",
        close_with_active_failure,
    )
    if stage == "fingerprint":
        monkeypatch.setattr(
            dataset_comparison,
            "fingerprint_dataset_session",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("programmer_primary")
            ),
        )
    else:
        monkeypatch.setattr(
            dataset_comparison.ExactPrimaryKeyIndex,
            "add_comparison_table",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("programmer_primary")
            ),
        )

    with pytest.raises(RuntimeError, match="programmer_primary"):
        dataset_comparison.compare_datasets(root, root, resource_policy=None)


def test_compare_propagates_programmer_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "left_programmer_error"
    right = tmp_path / "right_programmer_error"
    left.mkdir()
    right.mkdir()
    monkeypatch.setattr(
        dataset_comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("programmer_error")
        ),
    )

    with pytest.raises(RuntimeError, match="programmer_error"):
        dataset_comparison.compare_datasets(left, right, resource_policy=None)


def test_compare_rejects_mutation_between_fingerprint_and_exact_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "mutation_left")
    right = make_dataset_artifact_root(tmp_path, "mutation_right")
    frame = make_default_dataset_frame()
    write_dataset(left, frame, spec=make_processed_dataset_spec("mutation_left"))
    write_dataset(right, frame, spec=make_processed_dataset_spec("mutation_right"))
    identity_digest = dataset_comparison._session_identity_token
    calls = 0

    def changing_identity(context: object) -> bytes:
        nonlocal calls
        calls += 1
        digest = identity_digest(context)  # type: ignore[arg-type]
        return digest if calls <= 2 else b"changed"

    monkeypatch.setattr(
        dataset_comparison,
        "_session_identity_token",
        changing_identity,
    )

    comparison = compare_datasets(left, right)

    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("mutated_side", ("left", "right"))
def test_compare_best_effort_observed_stage_drift_rejects_post_index_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_side: str,
) -> None:
    roots = {
        side: make_dataset_artifact_root(tmp_path, f"final_barrier_{side}")
        for side in ("left", "right")
    }
    frame = make_default_dataset_frame()
    for side, root in roots.items():
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec(f"final_barrier_{side}"),
        )
    index_rows = dataset_comparison._index_dataset_rows

    def index_then_mutate(root: Path, **kwargs: object) -> None:
        index_rows(root, **kwargs)  # type: ignore[arg-type]
        if kwargs["namespace"] != mutated_side:
            return
        part = next((root / "data").rglob("*.parquet"))
        stat = part.stat()
        os.utime(
            part,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000),
        )

    monkeypatch.setattr(
        dataset_comparison,
        "_index_dataset_rows",
        index_then_mutate,
    )

    comparison = compare_datasets(roots["left"], roots["right"])

    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize(
    ("primary_type", "verify_type"),
    (
        (RuntimeError, OSError),
        (MemoryError, ValueError),
        (KeyboardInterrupt, OSError),
        (SystemExit, ValueError),
    ),
)
@pytest.mark.parametrize("iterator_name", ("file_batches", "chunks"))
def test_reader_final_verify_does_not_replace_active_primary(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
    verify_type: type[BaseException],
    iterator_name: str,
) -> None:
    class Context:
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {"value": {"kind": "numpy_int", "dtype": "int64"}}
        parquet_filters = None
        resource_policy = _issued_test_policy(1024**2)
        inventory = SimpleNamespace(retained_memory_bytes=0)
        retained_memory_bytes = 0

        def __init__(self) -> None:
            self.verify_calls = 0
            self.verify_fail_after = 1 if iterator_name == "file_batches" else 0

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise verify_type("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

    context = Context()

    def fail_batches(*_args: object, **_kwargs: object):
        raise primary_type("primary_failure")
        yield  # pragma: no cover

    if iterator_name == "file_batches":
        monkeypatch.setattr(
            dataset_reading,
            "pandas_dtype_overrides_from_logical_payload",
            lambda *_args, **_kwargs: {},
        )
        monkeypatch.setattr(
            dataset_reading,
            "iter_declared_parquet_member_batches",
            fail_batches,
        )
        iterator = dataset_reading.iter_dataset_file_batches(
            context,  # type: ignore[arg-type]
            columns=("value",),
            target_chunk_bytes=1024,
        )
    else:
        monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", fail_batches)
        iterator = dataset_reading.iter_dataset_chunks(
            context,  # type: ignore[arg-type]
            target_chunk_bytes=1024,
        )

    with pytest.raises(primary_type, match="primary_failure"):
        next(iterator)


@pytest.mark.parametrize("iterator_name", ("file_batches", "chunks"))
@pytest.mark.parametrize("completion", ("exhausted", "early_close"))
def test_reader_final_verify_failure_propagates_without_external_primary(
    monkeypatch: pytest.MonkeyPatch,
    iterator_name: str,
    completion: str,
) -> None:
    frame = pd.DataFrame({"value": [1]})

    class Context:
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {"value": {"kind": "numpy_int", "dtype": "int64"}}
        parquet_filters = None
        resource_policy = _issued_test_policy(1024**2)
        inventory = SimpleNamespace(retained_memory_bytes=0)
        retained_memory_bytes = 0

        def __init__(self) -> None:
            self.verify_calls = 0
            self.verify_fail_after = 1 if iterator_name == "file_batches" else 0

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise OSError("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

    context = Context()
    if iterator_name == "file_batches":
        monkeypatch.setattr(
            dataset_reading,
            "pandas_dtype_overrides_from_logical_payload",
            lambda *_args, **_kwargs: {},
        )

        def declared(*_args: object, **_kwargs: object):
            yield frame

        monkeypatch.setattr(
            dataset_reading,
            "iter_declared_parquet_member_batches",
            declared,
        )
        iterator = dataset_reading.iter_dataset_file_batches(
            context,  # type: ignore[arg-type]
            columns=("value",),
            target_chunk_bytes=1024,
        )
    else:

        def batches(*_args: object, **_kwargs: object):
            yield Path("part.parquet"), frame

        monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", batches)
        iterator = dataset_reading.iter_dataset_chunks(
            context,  # type: ignore[arg-type]
            target_chunk_bytes=1024,
        )

    if completion == "exhausted":
        with pytest.raises(OSError, match="verify_failure"):
            list(iterator)
    else:
        next(iterator)
        iterator.close()  # type: ignore[attr-defined]
        assert context.verify_calls > context.verify_fail_after


@pytest.mark.parametrize("iterator_name", ("file_batches", "chunks"))
@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit),
)
def test_dataset_iterator_context_preserves_caller_primary(
    monkeypatch: pytest.MonkeyPatch,
    iterator_name: str,
    primary_type: type[BaseException],
) -> None:
    frame = pd.DataFrame({"value": [1]})

    class Context:
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {"value": {"kind": "numpy_int", "dtype": "int64"}}
        parquet_filters = None
        resource_policy = _issued_test_policy(1024**2)
        inventory = SimpleNamespace(retained_memory_bytes=0)
        retained_memory_bytes = 0

        def __init__(self) -> None:
            self.verify_calls = 0
            self.verify_fail_after = 1 if iterator_name == "file_batches" else 0

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise OSError("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

    context = Context()
    if iterator_name == "file_batches":
        monkeypatch.setattr(
            dataset_reading,
            "pandas_dtype_overrides_from_logical_payload",
            lambda *_args, **_kwargs: {},
        )

        def declared(*_args: object, **_kwargs: object):
            yield frame

        monkeypatch.setattr(
            dataset_reading,
            "iter_declared_parquet_member_batches",
            declared,
        )
        iterator = dataset_reading.iter_dataset_file_batches(
            context,  # type: ignore[arg-type]
            columns=("value",),
            target_chunk_bytes=1024,
        )
    else:

        def batches(*_args: object, **_kwargs: object):
            yield Path("part.parquet"), frame

        monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", batches)
        iterator = dataset_reading.iter_dataset_chunks(
            context,  # type: ignore[arg-type]
            target_chunk_bytes=1024,
        )

    with pytest.raises(primary_type, match="primary_failure"):
        with dataset_reading.closing_dataset_iterator(iterator):
            next(iterator)
            raise primary_type("primary_failure")


def test_dataset_iterator_context_propagates_close_failure_without_primary() -> None:
    class IteratorWithFailingClose:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self) -> None:
            raise OSError("close_failure")

    with pytest.raises(OSError, match="close_failure"):
        with dataset_reading.closing_dataset_iterator(IteratorWithFailingClose()):
            pass


@pytest.mark.parametrize("seam", ("shape", "fill"))
@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit),
)
def test_materialized_iterator_close_does_not_replace_caller_primary(
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    primary_type: type[BaseException],
) -> None:
    class FailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary_type("primary_failure")

        def close(self) -> None:
            raise OSError("close_failure")

    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: FailingIterator(),
    )
    with pytest.raises(primary_type, match="primary_failure"):
        if seam == "shape":
            dataset_reading._scan_materialized_shape(
                SimpleNamespace(),  # type: ignore[arg-type]
                target_chunk_bytes=1024,
                operation_headroom_bytes=1024,
                string_columns=(),
                object_columns=(),
            )
        else:
            dataset_reading._fill_materialized_buffers(
                SimpleNamespace(),  # type: ignore[arg-type]
                buffers={},
                row_count=1,
                target_chunk_bytes=1024,
                operation_headroom_bytes=1024,
            )


@pytest.mark.parametrize(
    ("left_zero", "right_zero"),
    ((-0.0, 0.0), (0.0, -0.0)),
)
@pytest.mark.parametrize("columns", (None, ["value"]))
def test_compare_float_primary_key_zero_uses_exact_key_semantics(
    tmp_path: Path,
    left_zero: float,
    right_zero: float,
    columns: list[str] | None,
) -> None:
    roots = {
        side: make_dataset_artifact_root(tmp_path, f"float_zero_{side}")
        for side in ("left", "right")
    }
    for side, zero in (("left", left_zero), ("right", right_zero)):
        frame = make_default_dataset_frame(
            asset_ids=(zero,),
            dates=("2014-01-02",),
            values=(1.0,),
        )
        write_dataset(
            roots[side],
            frame,
            spec=make_processed_dataset_spec(f"float_zero_{side}"),
        )

    assert (
        fingerprint_dataset(
            roots["left"],
            columns=columns,
        ).content_hash
        == fingerprint_dataset(
            roots["right"],
            columns=columns,
        ).content_hash
    )
    assert (
        compare_datasets(
            roots["left"],
            roots["right"],
            columns=columns,
        ).status
        == "MATCH"
    )


def test_compare_float_value_zero_keeps_bit_identity(tmp_path: Path) -> None:
    left = make_dataset_artifact_root(tmp_path, "float_value_negative_zero")
    right = make_dataset_artifact_root(tmp_path, "float_value_positive_zero")
    for root, zero in ((left, -0.0), (right, 0.0)):
        write_dataset(
            root,
            make_default_dataset_frame(
                asset_ids=("asset_001",),
                dates=("2014-01-02",),
                values=(zero,),
            ),
            spec=make_processed_dataset_spec(root.name),
        )

    assert compare_datasets(left, right).status == "CONTENT_MISMATCH"


@pytest.mark.parametrize("nested_case", ("ordinary", "map", "multiple"))
def test_fingerprint_normalizes_bounded_slices_without_full_frame_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_case: str,
) -> None:
    row_count = 10_000
    if nested_case == "map":
        map_type = pa.map_(pa.string(), pa.int64())
        batch = pa.record_batch(
            [
                pa.array(range(row_count), type=pa.int64()),
                pa.array(
                    [[("key", value)] for value in range(row_count)],
                    type=map_type,
                ),
            ],
            names=("key", "nested"),
        )
        frame = batch.to_pandas()
    else:
        frame = pd.DataFrame(
            {
                "key": np.arange(row_count, dtype="int64"),
                "nested": [[value] for value in range(row_count)],
            }
        )
        if nested_case == "multiple":
            frame["second_nested"] = [(value, value + 1) for value in range(row_count)]
        batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    input_bytes = pandas_frame_payload_bytes(frame)
    operation_headroom = batch.get_total_buffer_size() + input_bytes + 64 * 1024
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    assert operation_headroom < policy.task_memory_budget_bytes
    path = tmp_path / "data" / "year=2026" / "part-000.parquet"
    data_root = tmp_path / "data"

    def batches(*_args: object, **kwargs: object):
        observer = kwargs["batch_observer"]
        observer(batch, frame)
        yield path, frame

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        batches,
    )
    copy_frame = pd.DataFrame.copy
    map_series = pd.Series.map

    def bounded_copy(current: pd.DataFrame, *args: object, **kwargs: object):
        if kwargs.get("deep", True) is not False and len(current) >= row_count // 2:
            raise AssertionError("fingerprint_copied_the_full_frame")
        return copy_frame(current, *args, **kwargs)

    def bounded_map(current: pd.Series, *args: object, **kwargs: object):
        if len(current) >= row_count // 2:
            raise AssertionError("fingerprint_mapped_the_full_column")
        return map_series(current, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", bounded_copy)
    monkeypatch.setattr(pd.Series, "map", bounded_map)
    logical_columns = tuple(frame.columns)
    context = SimpleNamespace(
        resource_policy=policy,
        spec={"primary_key": ["key"]},
        logical_columns=logical_columns,
        arrow_schema=batch.schema,
        time_window=None,
        data_root=data_root,
        read_paths=(path,),
        inventory=SimpleNamespace(retained_memory_bytes=0),
        retained_memory_bytes=0,
    )
    partition_reserve = dataset_reading.dataset_partition_state_reserve_bytes(context)
    local_reserve = struct.calcsize("P") * (1 + 2 * len(logical_columns))
    local_reserve += len(b"key")
    local_reserve += 2 * sum(len(column.encode("utf-8")) for column in logical_columns)
    retained = (
        policy.task_memory_budget_bytes
        - operation_headroom
        - partition_reserve
        - local_reserve
    )
    context.inventory.retained_memory_bytes = retained
    context.retained_memory_bytes = retained

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context,
        resource_policy=policy,
    )

    assert fingerprint.row_count == row_count


def test_exact_compare_reserves_payload_before_reader_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "exact_payload_reserve")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("exact_payload_reserve"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1024**2,
    )
    reader_headrooms: list[int] = []
    payload_targets: list[int] = []
    comparison_reservations: list[int] = []
    value_columns: list[tuple[str, ...]] = []
    iter_batches = dataset_comparison.iter_dataset_file_batches
    add_table = dataset_comparison.ExactPrimaryKeyIndex.add_comparison_table

    def observed_batches(*args: object, **kwargs: object):
        reader_headrooms.append(kwargs["operation_headroom_bytes"])  # type: ignore[arg-type]
        yield from iter_batches(*args, **kwargs)

    def observed_add(self: object, *args: object, **kwargs: object) -> int:
        payload_targets.append(kwargs["payload_target_bytes"])  # type: ignore[arg-type]
        comparison_reservations.append(policy.reserved_memory_bytes)
        value_columns.append(tuple(kwargs["value_columns"]))  # type: ignore[arg-type]
        return add_table(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        dataset_comparison,
        "iter_dataset_file_batches",
        observed_batches,
    )
    monkeypatch.setattr(
        dataset_comparison.ExactPrimaryKeyIndex,
        "add_comparison_table",
        observed_add,
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        retained_payload_bytes=dataset_comparison.EXACT_KEY_SQLITE_PAGE_BYTES,
    ) as session:
        session_retained_bytes = session.retained_memory_bytes

    comparison = dataset_comparison.compare_datasets(
        root,
        root,
        resource_policy=policy,
    )

    assert comparison.status == "MATCH"
    assert payload_targets
    assert len(reader_headrooms) == 2
    assert all(
        0 < target < policy.task_memory_budget_bytes for target in payload_targets
    )
    assert all(
        reserved >= target
        for reserved, target in zip(
            comparison_reservations,
            payload_targets,
            strict=True,
        )
    )
    assert all(reader > session_retained_bytes for reader in reader_headrooms)
    assert value_columns and set(value_columns) == {("value",)}


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
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
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

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as raw_session:
        spec_path = root / "spec.json"
        spec_path.write_bytes(raw_session.spec_bytes + b" ")
        with pytest.raises(ValueError, match="dataset_spec_identity_changed"):
            raw_session.verify()
        spec_path.write_bytes(raw_session.spec_bytes)

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as inventory_session:
        added = root / "data" / "year=2016" / "part-000.parquet"
        dataset_writing.write_parquet_file(
            _persisted_table(frame),
            added,
        )
        with pytest.raises(ValueError, match="dataset_inventory_changed"):
            inventory_session.verify()
        added.unlink()

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as removal_session:
        removal_session.all_paths[0].unlink()
        with pytest.raises(ValueError, match="dataset_inventory_changed"):
            removal_session.verify()


@pytest.mark.parametrize(
    ("relative_path", "error"),
    (
        (Path("spec.json"), "dataset_spec_identity_changed"),
        (Path("_manifest/manifest.json"), "dataset_manifest_identity_changed"),
    ),
)
def test_dataset_session_rejects_same_bytes_metadata_replacement(
    tmp_path: Path,
    relative_path: Path,
    error: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"session_{relative_path.name}_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"session_{relative_path.name}_drift"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        path = root / relative_path
        replacement = path.with_name(f"{path.name}.replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)

        with pytest.raises(ValueError, match=error):
            session.verify()


def test_physical_decoder_rechecks_the_same_handle_after_decode(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "handle_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("handle_drift"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
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
        receipt.session.close()
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
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
        ),
    )

    assert receipt_checks == [True]
    assert len(observed_paths) == len(set(observed_paths))
    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(observed_paths) == manifest["file_count"]


def test_open_dataset_session_owns_and_closes_inventory(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_inventory_owner")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_inventory_owner"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        scratch = session.inventory.scratch_path
        assert scratch.exists()

    assert not scratch.exists()


def test_open_dataset_session_counts_caller_retained_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_nested_retained")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_nested_retained"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    retained = 1234
    spec_bytes, _ = storage_manifests.read_bounded_qrp_json(root / "spec.json")
    manifest_bytes, _ = storage_manifests.read_bounded_qrp_json(
        root / "_manifest" / "manifest.json"
    )
    observed: list[int] = []

    class StopAfterInventoryBudget:
        def __init__(self, *, task_memory_budget_bytes, **_kwargs):  # noqa: ANN001, ANN003
            observed.append(task_memory_budget_bytes)
            raise RuntimeError("inventory_budget_observed")

    monkeypatch.setattr(
        dataset_reading, "DatasetSessionInventory", StopAfterInventoryBudget
    )

    with pytest.raises(RuntimeError, match="inventory_budget_observed"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            resource_policy=policy,
            retained_payload_bytes=retained,
        ):
            pass

    assert observed == [
        policy.task_memory_budget_bytes
        - retained
        - len(spec_bytes)
        - len(manifest_bytes)
        - len(
            dataset_reading._dataset_file_identity_bytes(
                dataset_physical.dataset_file_identity(root / "spec.json")
            )
        )
        - len(
            dataset_reading._dataset_file_identity_bytes(
                dataset_physical.dataset_file_identity(
                    root / "_manifest" / "manifest.json"
                )
            )
        )
    ]


@pytest.mark.parametrize("remaining_bytes", (0, 1))
def test_session_from_inventory_rejects_live_caller_overlap_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remaining_bytes: int,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "writer_session_overlap")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("writer_session_overlap"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        resource_policy=policy,
    ) as source:
        monkeypatch.setattr(
            dataset_reading,
            "_build_dataset_session",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("session build ran before overlap admission")
            ),
        )
        retained = (
            policy.task_memory_budget_bytes
            - source.inventory.retained_memory_bytes
            - remaining_bytes
        )
        metadata_reads = 0

        def reject_metadata_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal metadata_reads
            metadata_reads += 1
            raise AssertionError("metadata read ran before overlap admission")

        with monkeypatch.context() as admission_patch:
            admission_patch.setattr(
                dataset_reading,
                "read_bounded_qrp_json",
                reject_metadata_read,
            )
            with pytest.raises(
                storage_errors.DatasetReadCapacityExceeded,
                match="dataset_session_acquisition_exceeds_task_memory_budget",
            ):
                dataset_reading.dataset_session_from_inventory(
                    root,
                    inventory=source.inventory,
                    columns=None,
                    time_window=None,
                    resource_policy=policy,
                    retained_payload_bytes=retained,
                )

        assert metadata_reads == 0


@pytest.mark.parametrize("remaining_bytes", (0, 1))
def test_open_dataset_session_rejects_live_caller_overlap_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remaining_bytes: int,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "owned_session_overlap")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("owned_session_overlap"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    monkeypatch.setattr(
        dataset_reading,
        "_build_dataset_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("session build ran before overlap admission")
        ),
    )
    metadata_reads = 0

    def reject_metadata_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran before overlap admission")

    monkeypatch.setattr(
        dataset_reading,
        "read_bounded_qrp_json",
        reject_metadata_read,
    )
    retained = policy.task_memory_budget_bytes - 1024 - remaining_bytes

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_session_acquisition_exceeds_task_memory_budget",
    ):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            resource_policy=policy,
            retained_payload_bytes=retained,
        ):
            pass

    assert metadata_reads == 0


def test_open_dataset_session_preserves_missing_manifest_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "missing_manifest_preflight")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("missing_manifest_preflight"),
    )
    (root / "_manifest" / "manifest.json").unlink()
    original_stat = Path.stat
    metadata_stats = 0
    metadata_reads = 0

    def observe_owner_metadata_stat(
        self: Path,
        *args: object,
        **kwargs: object,
    ):  # noqa: ANN202
        nonlocal metadata_stats
        caller = currentframe().f_back
        if caller is not None and caller.f_code.co_filename.endswith(
            "_dataset_reading.py"
        ):
            metadata_stats += 1
        return original_stat(self, *args, **kwargs)

    def reject_metadata_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran for missing manifest")

    monkeypatch.setattr(Path, "stat", observe_owner_metadata_stat)
    monkeypatch.setattr(
        dataset_reading,
        "read_bounded_qrp_json",
        reject_metadata_read,
    )

    with pytest.raises(ValueError, match="dataset_manifest_missing"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            pass

    assert metadata_stats == 0
    assert metadata_reads == 0


@pytest.mark.parametrize("entrypoint", ("owned", "borrowed"))
def test_dataset_session_metadata_stat_failure_precedes_metadata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"metadata_stat_{entrypoint}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"metadata_stat_{entrypoint}"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    metadata_reads = 0

    def fail_owner_preflight_identity(_path: Path):  # noqa: ANN202
        raise OSError("metadata_stat_failed")

    def reject_metadata_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran after stat failure")

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        resource_policy=policy,
    ) as source:
        with monkeypatch.context() as admission_patch:
            admission_patch.setattr(
                dataset_reading,
                "dataset_file_identity",
                fail_owner_preflight_identity,
            )
            admission_patch.setattr(
                dataset_reading,
                "read_bounded_qrp_json",
                reject_metadata_read,
            )
            with pytest.raises(ValueError, match="dataset_metadata_file_invalid"):
                if entrypoint == "owned":
                    with dataset_reading.open_dataset_session(
                        root,
                        columns=None,
                        time_window=None,
                        resource_policy=policy,
                        retained_payload_bytes=policy.task_memory_budget_bytes - 1,
                    ):
                        pass
                else:
                    dataset_reading.dataset_session_from_inventory(
                        root,
                        inventory=source.inventory,
                        columns=None,
                        time_window=None,
                        resource_policy=policy,
                        retained_payload_bytes=(
                            policy.task_memory_budget_bytes
                            - source.inventory.retained_memory_bytes
                            - 1
                        ),
                    )

    assert metadata_reads == 0


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
    input_bytes = pandas_frame_payload_bytes(frame)
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
        resource_policy=capture_test_policy(
            available_memory_bytes=16 * 1024**2,
            explicit_cap_bytes=8 * 1024 * 1024,
        ),
    )

    assert result.root == root
    assert result.data_root == root / "data"
    assert result.row_count == row_count
    assert len(written_table_bytes) > 1
    assert len(converted_frame_bytes) > 1
    assert max(written_table_bytes) < input_bytes
    assert max(converted_frame_bytes) < input_bytes


def test_private_writer_returns_final_inspection_and_public_writer_returns_none(
    tmp_path: Path,
) -> None:
    private_root = make_dataset_artifact_root(tmp_path, "private_return")
    public_root = make_dataset_artifact_root(tmp_path, "public_return")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
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


def test_dataset_global_nullable_integer_plan_is_chunk_order_independent(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame()
    numpy_chunk = base.iloc[:1].copy()
    nullable_chunk = base.iloc[1:2].copy()
    numpy_chunk["value"] = pd.Series([1], index=numpy_chunk.index, dtype="int64")
    nullable_chunk["value"] = pd.Series([2], index=nullable_chunk.index, dtype="Int64")
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
                (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
            )
        )
    assert manifests[0]["logical_dtypes"] == manifests[1]["logical_dtypes"]
    assert manifests[0]["logical_dtype_hash"] == manifests[1]["logical_dtype_hash"]


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
                (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
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
    with pytest.raises(ValueError, match="manifest_partition_date_mismatch"):
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
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=(("asset_id", "==", ["A"]),),
        ):
            pass


def test_dataset_session_tuple_in_filter_matches_materialized_and_chunked_reads(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter")
    frame = make_default_dataset_frame(
        asset_ids=("A", "B", "C"),
        dates=("2015-01-05", "2014-01-02", "2015-01-05"),
        values=(2.0, 1.0, float("nan")),
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("tuple_in_filter"),
    )

    expected = frame.loc[
        frame["asset_id"].isin(("A", "C")), ["timestamp", "value"]
    ].reset_index(drop=True)
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("A", "A", "C")),),
    ) as session:
        chunked = pd.concat(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=512,
            ),
            ignore_index=True,
        )
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("A", "A", "C")),),
    ) as session:
        materialized = pd.concat(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=1024**2,
            ),
            ignore_index=True,
        )

    pd.testing.assert_frame_equal(materialized, expected)
    pd.testing.assert_frame_equal(chunked, expected)
    with dataset_reading.open_dataset_session(
        root,
        columns=("asset_id",),
        time_window=None,
        row_filters=(("asset_id", "==", "B"),),
    ) as session:
        scalar = pd.concat(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=1024**2,
            ),
            ignore_index=True,
        )
    assert scalar["asset_id"].tolist() == ["B"]


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("asset_id", []),
        ("asset_id", ()),
        ("asset_id", ["A", "C"]),
        ("asset_id", ("A", None)),
        ("value", (1.0, float("nan"))),
        ("value", (1.0, float("inf"))),
    ),
)
def test_dataset_session_tuple_in_filter_has_exact_immutable_nonempty_grammar(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_grammar")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tuple_in_filter_grammar"),
    )

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=((column, "in", value),),
        ):
            pass


def test_dataset_session_tuple_in_filter_rejects_tuple_subclass(tmp_path: Path) -> None:
    class TupleSubclass(tuple):
        pass

    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_subclass")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tuple_in_filter_subclass"),
    )

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=(("asset_id", "in", TupleSubclass(("A",))),),
        ):
            pass


def test_dataset_session_tuple_in_filter_empty_result_preserves_projected_dtypes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_empty")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tuple_in_filter_empty"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    ) as session:
        chunks = list(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=1024 * 1024,
            )
        )

    assert len(chunks) == 1
    assert chunks[0].empty
    assert chunks[0].dtypes.to_dict() == {
        "timestamp": np.dtype("datetime64[ns]"),
        "value": make_default_dataset_frame()["value"].dtype,
    }


def test_categorical_empty_result_can_exceed_the_soft_chunk_target(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "categorical_empty_capacity")
    frame = make_default_dataset_frame()
    frame["value"] = pd.Categorical(frame["value"].astype(str))
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("categorical_empty_capacity"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    ) as session:
        chunks = list(
            dataset_reading.iter_dataset_chunks(session, target_chunk_bytes=1)
        )
    assert len(chunks) == 1
    assert chunks[0].empty
    assert isinstance(chunks[0]["value"].dtype, pd.CategoricalDtype)


def test_dataset_session_tuple_in_filter_prunes_nonmatching_row_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_prune")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tuple_in_filter_prune"),
    )
    observed: list[bool] = []
    original = dataset_physical._row_group_may_match

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(dataset_physical, "_row_group_may_match", record)
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp",),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    ) as session:
        list(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=1024**2,
            )
        )

    assert observed
    assert not any(observed)


def test_session_filter_reuses_prepared_values(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_prepared_once")
    frame = make_default_dataset_frame(
        asset_ids=("A", "B", "C"),
        dates=("2014-01-02", "2015-01-05", "2016-01-04"),
        values=(1.0, 2.0, 3.0),
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("tuple_in_filter_prepared_once"),
    )
    values = ("A", "C", *(f"missing_{position:08d}" for position in range(40_000)))
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024 * 1024,
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(
            ("asset_id", "in", values),
            ("asset_id", "in", values),
        ),
        resource_policy=policy,
        retained_payload_bytes=0,
    ) as session:
        chunks = list(
            dataset_reading.iter_dataset_chunks(session, target_chunk_bytes=512)
        )

    assert sum(map(len, chunks)) == 2


def test_dataset_session_retained_memory_includes_inventory_and_prepared_filters(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_retained_filter")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_retained_filter"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp",),
        time_window=None,
        row_filters=(("asset_id", "in", ("A", "B")),),
    ) as session:
        assert session.prepared_filter_bytes > 0
        metadata_bytes = (
            session.inventory.retained_memory_bytes
            + len(session.spec_bytes)
            + len(session.spec_identity_bytes)
            + len(session.manifest_bytes)
            + len(session.manifest_identity_bytes)
            + len(session.logical_dtype_bytes)
            + session.prepared_filter_bytes
        )
        assert session.plan_base_retained_bytes >= (
            len(session.arrow_schema.serialize())
            + sum(len(column.encode("utf-8")) for column in session.logical_columns)
            + sum(len(column.encode("utf-8")) for column in session.physical_columns)
        )
        assert session.retained_memory_bytes > (
            metadata_bytes + session.plan_base_retained_bytes
        )


def test_dataset_session_plan_retained_memory_scales_with_wide_columns(
    tmp_path: Path,
) -> None:
    narrow = ("column",)
    wide = tuple(f"column_{index:05d}_payload" for index in range(10_000))

    narrow_bytes = dataset_reading._dataset_session_plan_base_retained_memory_bytes(
        root=tmp_path,
        data_root=tmp_path / "data",
        logical_columns=narrow,
        physical_columns=narrow,
        timestamp_column=None,
        schema_hash="hash",
        arrow_schema_span_bytes=1,
    )
    wide_bytes = dataset_reading._dataset_session_plan_base_retained_memory_bytes(
        root=tmp_path,
        data_root=tmp_path / "data",
        logical_columns=wide,
        physical_columns=wide,
        timestamp_column=None,
        schema_hash="hash",
        arrow_schema_span_bytes=1,
    )

    assert wide_bytes - narrow_bytes >= 2 * sum(
        len(column.encode("utf-8")) for column in wide
    )


def test_dataset_filter_capacity_uses_the_typed_storage_signal(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "typed_filter_capacity")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("typed_filter_capacity"),
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    capacity_error = getattr(storage_errors, "DatasetReadCapacityExceeded", None)
    assert capacity_error is not None

    with pytest.raises(
        capacity_error,
        match="dataset_filter_values_exceed_task_memory_budget",
    ):
        with dataset_reading.open_dataset_session(
            root,
            columns=("timestamp",),
            time_window=None,
            row_filters=(("asset_id", "in", ("x" * (2 * 1024**2),)),),
            resource_policy=policy,
        ):
            pass


@pytest.mark.parametrize(
    "shape",
    ("one_wide", "many_small", "duplicates"),
)
def test_dataset_filter_preflight_rejects_before_bulk_arrow_array(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    bulk_array_called = False
    scalar_called = False

    if shape == "one_wide":
        values = ("x" * (policy.task_memory_budget_bytes + 1),)
    elif shape == "many_small":
        values = ("x",) * (policy.task_memory_budget_bytes // 4 + 1)
    else:
        value = "duplicate"
        per_value_bytes = 4 + len(value)
        values = (value,) * (policy.task_memory_budget_bytes // per_value_bytes + 1)

    def reject_bulk_array(*args, **kwargs):
        nonlocal bulk_array_called
        bulk_array_called = True
        raise AssertionError("bulk pa.array must not run after a failed preflight")

    def reject_scalar(*args, **kwargs):
        nonlocal scalar_called
        scalar_called = True
        raise AssertionError("pa.scalar must not run before string preflight")

    monkeypatch.setattr(dataset_reading.pa, "array", reject_bulk_array)
    monkeypatch.setattr(dataset_reading.pa, "scalar", reject_scalar)

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_filter_values_exceed_(payload_target|task_memory_budget)",
    ):
        dataset_reading._prepare_in_filter(
            pa.field("asset_id", pa.string()),
            values,
            resource_policy=policy,
            retained_filter_bytes=0,
        )
    assert bulk_array_called is False
    assert scalar_called is False


def test_dataset_filter_preparation_subtracts_shared_reservations_before_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    values = (1, 2)

    def reject_late_work(*_args, **_kwargs):
        raise AssertionError("late filter preparation work ran before task admission")

    monkeypatch.setattr(builtins, "sorted", reject_late_work)
    monkeypatch.setattr(dataset_reading.pa, "array", reject_late_work)

    with (
        runtime_resource_policy_scope(policy),
        policy.reserve("outer", policy.task_memory_budget_bytes - 1),
        pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="dataset_filter_values_exceed_task_memory_budget",
        ),
    ):
        dataset_reading._prepare_in_filter(
            pa.field("value", pa.int64()),
            values,
            resource_policy=policy,
            retained_filter_bytes=0,
        )


def test_membership_internal_grammar_rejects_raw_arrow_values() -> None:
    raw = pa.array([1, 2], type=pa.int64())

    with pytest.raises(TypeError):
        dataset_physical._membership_max_scalar_bytes(raw)
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._freeze_filter_value(raw)


def test_dataset_prepared_in_filter_compacts_sorted_duplicates_without_arrow_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    def reject_unique(*_args, **_kwargs):
        raise AssertionError("prepared in filters must not allocate a unique copy")

    monkeypatch.setattr(dataset_reading.pa.compute, "unique", reject_unique)

    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", pa.int64()),
        (2, 1, 1),
        resource_policy=policy,
        retained_filter_bytes=0,
    )

    assert prepared.values.to_pylist() == [1, 2]


@pytest.mark.parametrize(
    ("data_type", "value"),
    ((pa.int64(), 7), (pa.string(), "duplicate")),
    ids=("fixed", "string"),
)
def test_dataset_prepared_in_filter_payload_gate_uses_compact_values(
    data_type: pa.DataType,
    value: object,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    values = (value,) * 140_000

    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", data_type),
        values,
        resource_policy=policy,
        retained_filter_bytes=0,
    )

    assert prepared.values.to_pylist() == [value]
    assert prepared.retained_memory_bytes <= policy.task_memory_budget_bytes


def test_dataset_prepared_in_filter_rejects_distinct_equal_string_payloads() -> None:
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=16 * 1024 * 1024,
    )
    values = tuple(("x" * 1000 + str(index))[:1000] for index in range(20_000))
    assert len({id(value) for value in values}) == len(values)

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_filter_values_exceed_task_memory_budget",
    ):
        dataset_reading._prepare_in_filter(
            pa.field("value", pa.string()),
            values,
            resource_policy=policy,
            retained_filter_bytes=0,
        )


@pytest.mark.parametrize(
    ("data_type", "values"),
    (
        (pa.time32("s"), (time(0, 0), time(23, 59, 59))),
        (pa.time64("us"), (time(0, 0, 0, 1), time(23, 59, 59, 999999))),
    ),
    ids=("time32", "time64"),
)
def test_dataset_prepared_in_filter_accepts_arrow_time_values(
    data_type: pa.DataType,
    values: tuple[time, ...],
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", data_type),
        values,
        resource_policy=policy,
        retained_filter_bytes=0,
    )

    assert prepared.values.to_pylist() == list(values)


@pytest.mark.parametrize(
    ("data_type", "values"),
    (
        (pa.time32("s"), (time(0, 0), time(12, 34, 56), time(23, 59, 59))),
        (
            pa.time64("us"),
            (time(0, 0, 0, 1), time(12, 34, 56, 7), time(23, 59, 59, 999999)),
        ),
    ),
    ids=("time32", "time64"),
)
def test_persisted_arrow_time_filters_support_equal_and_membership(
    tmp_path: Path,
    data_type: pa.DataType,
    values: tuple[time, ...],
) -> None:
    path = tmp_path / "time.parquet"
    pq.write_table(pa.table({"value": pa.array(values, type=data_type)}), path)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", data_type),
        (values[0], values[-1]),
        resource_policy=policy,
        retained_filter_bytes=0,
    )

    equal = list(
        dataset_physical.iter_dataset_part_batches(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            target_chunk_bytes=1024,
            operation_headroom_bytes=1024**2,
            parquet_filters=[("value", "==", values[1])],
        )
    )
    membership = list(
        dataset_physical.iter_dataset_part_batches(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            target_chunk_bytes=1024,
            operation_headroom_bytes=1024**2,
            parquet_filters=[("value", "in", prepared)],
        )
    )

    assert [value for batch in equal for value in batch.column(0).to_pylist()] == [
        values[1]
    ]
    assert [value for batch in membership for value in batch.column(0).to_pylist()] == [
        values[0],
        values[-1],
    ]


def test_dataset_prepared_in_filter_rejects_unsupported_arrow_family() -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._prepare_in_filter(
            pa.field("value", pa.list_(pa.int64())),
            ((1,),),
            resource_policy=policy,
            retained_filter_bytes=0,
        )


def test_dataset_prepared_in_filter_accepts_fixed_size_binary() -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", pa.binary(2)),
        (b"ab", b"cd"),
        resource_policy=policy,
        retained_filter_bytes=0,
    )

    assert prepared.values.to_pylist() == [b"ab", b"cd"]


def test_dataset_filter_preflight_rejects_surrogate_before_arrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    def reject_arrow(*_args, **_kwargs):
        raise AssertionError("Arrow must not receive an invalid surrogate")

    monkeypatch.setattr(dataset_reading.pa, "array", reject_arrow)
    monkeypatch.setattr(dataset_reading.pa, "scalar", reject_arrow)

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._prepare_in_filter(
            pa.field("asset_id", pa.string()),
            ("valid\ud800suffix",),
            resource_policy=policy,
            retained_filter_bytes=0,
        )


@pytest.mark.parametrize(
    ("data_type", "offset_bytes"),
    ((pa.binary(), 4), (pa.large_binary(), 8)),
    ids=("binary", "large-binary"),
)
@pytest.mark.parametrize("shape", ("one_wide", "many_small"))
def test_dataset_binary_filter_preflight_rejects_before_arrow(
    monkeypatch: pytest.MonkeyPatch,
    data_type: pa.DataType,
    offset_bytes: int,
    shape: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    if shape == "one_wide":
        values = (b"x" * (policy.task_memory_budget_bytes + 1),)
        expected = "dataset_filter_values_exceed_(payload_target|task_memory_budget)"
    else:
        count = policy.task_memory_budget_bytes // (offset_bytes + 1) + 1
        values = (b"x",) * count
        expected = "dataset_filter_values_exceed_(payload_target|task_memory_budget)"

    def reject_arrow(*_args, **_kwargs):
        raise AssertionError("Arrow must not run before binary capacity preflight")

    monkeypatch.setattr(dataset_reading.pa, "array", reject_arrow)
    monkeypatch.setattr(dataset_reading.pa, "scalar", reject_arrow)

    with pytest.raises(storage_errors.DatasetReadCapacityExceeded, match=expected):
        dataset_reading._prepare_in_filter(
            pa.field("payload", data_type),
            values,
            resource_policy=policy,
            retained_filter_bytes=0,
        )


def test_dataset_filter_capacity_includes_inventory_during_preparation(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "filter_inventory_overlap")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("filter_inventory_overlap"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp",),
        time_window=None,
    ) as session:
        inventory_bytes = session.inventory.retained_memory_bytes
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    value = "x" * (policy.task_memory_budget_bytes - inventory_bytes - 3)
    acquired = pa.array((value,), type=pa.string())
    assert acquired.nbytes <= policy.task_memory_budget_bytes
    assert inventory_bytes + acquired.nbytes > policy.task_memory_budget_bytes

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_filter_values_exceed_task_memory_budget",
    ):
        with dataset_reading.open_dataset_session(
            root,
            columns=("timestamp",),
            time_window=None,
            row_filters=(("asset_id", "in", (value,)),),
            resource_policy=policy,
        ):
            pass


@pytest.mark.parametrize(
    "dtype",
    (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "boolean",
        "Float32",
        "Float64",
        "string",
    ),
)
def test_dataset_global_extension_dtype_survives_read_scopes(
    tmp_path: Path,
    dtype: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"extension_{dtype}")
    frame = make_default_dataset_frame()
    if dtype == "boolean":
        values = [True, pd.NA, False]
    elif dtype == "string":
        values = ["x", pd.NA, "z"]
    elif dtype.startswith("Float"):
        values = [1.5, pd.NA, 3.5]
    else:
        values = [1, pd.NA, 3]
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
    "value",
    (
        np.int64(7),
        True,
        1.25,
        "text",
        b"\x00\xff",
        Decimal("1.25"),
        date(2026, 1, 2),
        pd.Timestamp("2026-01-02T03:04:05+00:00"),
    ),
)
def test_logical_category_scalar_codec_round_trips_supported_values(
    value: object,
) -> None:
    payload = dataset_schema._typed_category_value_payload(value)

    assert (
        dataset_schema._typed_category_value_payload(
            dataset_schema._typed_category_value(payload)
        )
        == payload
    )
    assert dataset_schema._typed_category_identity(payload) == canonical_identity_bytes(
        payload
    )


def test_logical_category_scalar_codec_rejects_invalid_values() -> None:
    for value in (float("nan"), Decimal("NaN"), pd.Timestamp("NaT"), object()):
        with pytest.raises(ValueError, match="dataset_categorical_value_invalid"):
            dataset_schema._typed_category_value_payload(value)

    for payload in (
        {"type": "float", "value": float("nan")},
        {"type": "bytes", "value": "0"},
        {"type": "decimal", "value": "not-a-decimal"},
        {"type": "decimal", "value": "NaN"},
        {"type": "date", "value": "not-a-date"},
        {"type": "timestamp", "value": "not-a-timestamp"},
        {"type": "unknown", "value": "value"},
    ):
        with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
            dataset_schema._typed_category_value(payload)


def test_logical_dtype_payload_rejects_invalid_contract_shapes() -> None:
    schema_id = "qrp-dataset-part-logical-dtypes"
    valid = {"name": "value", "kind": "numpy_integer", "dtype": "int64"}

    def payload(*columns: object) -> dict[str, object]:
        return {"schema_id": schema_id, "columns": list(columns)}

    invalid_payloads = (
        None,
        {"schema_id": "unknown", "columns": [valid]},
        {"schema_id": schema_id, "columns": []},
        payload(None),
        payload({"name": "", "kind": "numpy_integer", "dtype": "int64"}),
        payload({"name": "value", "kind": "", "dtype": "int64"}),
        payload({"name": "value", "kind": "categorical", "ordered": False}),
        payload(
            {
                "name": "value",
                "kind": "categorical",
                "ordered": 0,
                "categories": [],
            }
        ),
        payload(
            {
                "name": "value",
                "kind": "categorical",
                "ordered": False,
                "categories": (),
            }
        ),
        payload(
            {
                "name": "value",
                "kind": "categorical",
                "ordered": False,
                "categories": [
                    {"type": "int", "value": 1},
                    {"type": "int", "value": 1},
                ],
            }
        ),
        payload({**valid, "extra": True}),
        payload({"name": "value", "kind": "numpy_integer", "dtype": ""}),
        payload(
            {
                "name": "value",
                "kind": "categorical",
                "ordered": False,
                "categories": [{"type": "int", "value": 1, "extra": True}],
            }
        ),
        payload({"name": "value", "kind": "numpy_integer", "dtype": "invalid"}),
        payload(valid, valid),
    )

    for invalid in invalid_payloads:
        with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
            dataset_schema.validate_logical_dtype_payload(invalid)


def test_logical_dtype_merge_preserves_only_compatible_semantics() -> None:
    def payload(series: pd.Series, name: str = "value") -> dict[str, object]:
        return logical_dtype_payload_from_frame(pd.DataFrame({name: series}))

    numpy_integer = payload(pd.Series([1], dtype="int64"))
    nullable_integer = payload(pd.Series([1], dtype="Int64"))
    numpy_boolean = payload(pd.Series([True], dtype="bool"))
    nullable_boolean = payload(pd.Series([True], dtype="boolean"))
    numpy_float = payload(pd.Series([1.0], dtype="float64"))
    nullable_float = payload(pd.Series([1.0], dtype="Float64"))

    assert (
        dataset_schema.merge_logical_dtype_payloads([numpy_integer, nullable_integer])
        == nullable_integer
    )
    assert (
        dataset_schema.merge_logical_dtype_payloads([nullable_boolean, numpy_boolean])
        == nullable_boolean
    )
    assert (
        dataset_schema.merge_logical_dtype_payloads([numpy_float, nullable_float])
        == nullable_float
    )

    ordered_a = payload(
        pd.Series(pd.Categorical(["a"], categories=["a"], ordered=True))
    )
    ordered_b = payload(
        pd.Series(pd.Categorical(["b"], categories=["b"], ordered=True))
    )
    ordered_a_copy = payload(
        pd.Series(pd.Categorical(["a"], categories=["a"], ordered=True))
    )
    assert (
        dataset_schema.merge_logical_dtype_payloads([ordered_a, ordered_a_copy])
        == ordered_a
    )

    incompatible = (
        ([], "dataset_logical_dtypes_missing"),
        (
            [numpy_integer, payload(pd.Series([1], dtype="int64"), "other")],
            "dataset_logical_dtype_mismatch",
        ),
        (
            [numpy_integer, payload(pd.Series(["x"], dtype="string"))],
            "dataset_logical_dtype_mismatch",
        ),
        (
            [numpy_integer, payload(pd.Series([1], dtype="int32"))],
            "dataset_logical_dtype_mismatch",
        ),
        (
            [
                ordered_a,
                payload(
                    pd.Series(pd.Categorical(["a"], categories=["a"], ordered=False))
                ),
            ],
            "dataset_logical_dtype_mismatch",
        ),
        ([ordered_a, ordered_b], "dataset_ordered_categorical_mismatch"),
    )
    for payloads, error in incompatible:
        with pytest.raises(ValueError, match=error):
            dataset_schema.merge_logical_dtype_payloads(payloads)


def test_dataset_schema_entrypoints_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="dataset_arrow_schema_required"):
        dataset_schema.physical_schema_from_schema([])
    with pytest.raises(TypeError, match="dataset_frame_required"):
        logical_dtype_payload_from_frame(None)
    with pytest.raises(ValueError, match="dataset_column_not_found"):
        logical_dtype_payload_from_frame(pd.DataFrame({"value": [1]}), columns=("x",))
    payload = logical_dtype_payload_from_frame(pd.DataFrame({"value": [1]}))
    with pytest.raises(ValueError, match="invalid_target_chunk_bytes"):
        dataset_schema.require_logical_dtype_metadata_within_target(
            payload,
            category_payloads=lambda _descriptor: (),
            target_chunk_bytes=0,
        )
    with pytest.raises(TypeError, match="dataset_frame_required"):
        dataset_schema.require_frame_logical_dtype_metadata_within_target(
            None,
            columns=("value",),
            target_chunk_bytes=1024,
        )
    with pytest.raises(ValueError, match="dataset_column_not_found"):
        dataset_schema.require_frame_logical_dtype_metadata_within_target(
            pd.DataFrame({"value": [1]}),
            columns=("x",),
            target_chunk_bytes=1024,
        )
    for series in (
        pd.Series(pd.array([b"x"], dtype=pd.ArrowDtype(pa.binary()))),
        pd.Series(pd.period_range("2026-01", periods=1, freq="M")),
    ):
        with pytest.raises(ValueError, match="unsupported_dataset_logical_dtype"):
            logical_dtype_payload_from_frame(pd.DataFrame({"value": series}))


def test_physical_schema_payload_rejects_invalid_contract_shapes() -> None:
    field = {"name": "value", "nullable": False, "type": {"kind": "string"}}
    invalid_types = (
        None,
        {"kind": "integer", "signed": 1, "bit_width": 8},
        {"kind": "fixed_size_binary", "byte_width": 0},
        {"kind": "timestamp", "unit": "ns", "timezone": 1},
        {"kind": "struct", "fields": []},
        {
            "kind": "map",
            "keys_sorted": 1,
            "key_field": field,
            "item_field": field,
        },
        {"kind": "unknown"},
    )
    invalid_schemas = [
        None,
        [],
        [{}],
        [{**field, "nullable": 0}],
        [field, dict(field)],
        *[[{**field, "type": invalid_type}] for invalid_type in invalid_types],
    ]

    for invalid in invalid_schemas:
        with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
            dataset_schema.validate_physical_schema_payload(invalid)


def test_dataset_schema_validators_reject_mutated_runtime_contracts() -> None:
    with pytest.raises(TypeError, match="invalid_dataset_instance_metadata"):
        dataset_schema.merge_dataset_instance_metadata({}, [])
    with pytest.raises(ValueError, match="unsupported_dataset_instance_metadata_key"):
        dataset_schema.merge_dataset_instance_metadata({}, {1: "value"})
    with pytest.raises(ValueError, match="invalid_dataset_identity"):
        dataset_schema.dataset_identity_payload({}, {})
    with pytest.raises(ValueError, match="unsupported_dataset_instance_metadata_key"):
        dataset_schema.validate_dataset_manifest_instance_metadata({"unknown": True})
    with pytest.raises(ValueError, match="dataset_spec_type_invalid"):
        dataset_schema.validate_dataset_spec(object())
    spec = make_processed_dataset_spec("mutated_schema_version")
    object.__setattr__(spec, "schema_version", "0.2")
    with pytest.raises(ValueError, match="unsupported_schema_version"):
        dataset_schema.validate_dataset_spec(spec)

    for manifest in (
        {},
        {"primary_key": ["asset_id"], "null_count_summary": {}},
        {
            "primary_key": ["asset_id"],
            "null_count_summary": {"asset_id": True},
        },
    ):
        with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
            dataset_schema.validate_dataset_primary_key_null_summary(manifest)


@pytest.mark.parametrize(
    ("case", "error"),
    (
        ("grain", "dataset_grain_invalid"),
        ("missing_primary_key", "dataset_primary_key_invalid"),
        ("partition", "unsupported_partitioning"),
        ("key_role", "invalid_primary_key_roles"),
        ("unpartitioned_timestamp", "unpartitioned_data_must_not_use_timestamp_key"),
        ("missing_timestamp", "invalid_timestamp_primary_key"),
    ),
)
def test_dataset_schema_rejects_mutated_spec_invariants(
    case: str,
    error: str,
) -> None:
    spec = make_processed_dataset_spec(f"mutated_{case}")
    if case == "grain":
        object.__setattr__(spec, "grain", "asset_timestamp")
    elif case == "missing_primary_key":
        object.__setattr__(spec, "primary_key", ("missing",))
    elif case == "partition":
        object.__setattr__(spec, "partition_columns", ("month",))
    elif case == "key_role":
        roles = dict(spec.column_roles)
        roles["asset_id"] = ColumnRole.FACTOR
        object.__setattr__(spec, "column_roles", roles)
    elif case == "unpartitioned_timestamp":
        object.__setattr__(spec, "partition_columns", ())
    else:
        object.__setattr__(spec, "primary_key", ("market_id", "asset_id"))

    with pytest.raises(ValueError, match=error):
        dataset_schema.validate_dataset_spec(spec)


@pytest.mark.parametrize(
    ("_case_id", "arrow_type", "expected_kind", "expected_dtype", "sample"),
    _ARROW_SCALAR_LOGICAL_DTYPE_CASES,
    ids=[case[0] for case in _ARROW_SCALAR_LOGICAL_DTYPE_CASES],
)
@pytest.mark.parametrize("shape", ("empty", "null", "value"))
def test_arrow_scalar_logical_dtypes_use_canonical_pandas_semantic_family(
    _case_id: str,
    arrow_type: pa.DataType,
    expected_kind: str,
    expected_dtype: str,
    sample: object,
    shape: str,
) -> None:
    values = {"empty": [], "null": [None], "value": [sample]}[shape]
    frame = pd.DataFrame({"value": pd.array(values, dtype=pd.ArrowDtype(arrow_type))})

    payload = logical_dtype_payload_from_frame(frame)
    descriptor = payload["columns"][0]
    overrides = dataset_schema.pandas_dtype_overrides_from_logical_payload(
        payload,
        columns=("value",),
    )

    assert descriptor == {
        "name": "value",
        "kind": expected_kind,
        "dtype": expected_dtype,
    }
    parsed = overrides["value"]
    assert not isinstance(parsed, str)
    assert parsed == pd.api.types.pandas_dtype(expected_dtype)


@pytest.mark.parametrize(
    ("_case_id", "arrow_type", "_expected_kind", "expected_dtype", "sample"),
    _ARROW_SCALAR_LOGICAL_DTYPE_CASES,
    ids=[case[0] for case in _ARROW_SCALAR_LOGICAL_DTYPE_CASES],
)
def test_arrow_scalar_dtypes_are_stable_across_storage_read_boundaries(
    tmp_path: Path,
    _case_id: str,
    arrow_type: pa.DataType,
    _expected_kind: str,
    expected_dtype: str,
    sample: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"arrow_{_case_id}")
    frame = make_default_dataset_frame()
    frame["value"] = pd.Series(
        pd.array([sample, None, sample], dtype=pd.ArrowDtype(arrow_type)),
        index=frame.index,
    )
    write_dataset(
        root,
        (frame.iloc[1:].copy(), frame.iloc[:1].copy()),
        spec=make_processed_dataset_spec(
            f"arrow_{_case_id}",
            value_role=ColumnRole.CONTEXT,
        ),
    )

    materialized = read_dataset(root, columns=("timestamp", "value"))
    chunks = list(
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

    assert str(materialized["value"].dtype) == expected_dtype
    assert {str(chunk["value"].dtype) for chunk in chunks} == {expected_dtype}
    assert str(windowed["value"].dtype) == expected_dtype


@pytest.mark.parametrize(
    ("_case_id", "arrow_type", "_expected_kind", "expected_dtype", "sample"),
    _ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES,
    ids=[case[0] for case in _ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES],
)
def test_arrow_datetime_dtype_is_stable_for_filter_empty_direct_session(
    tmp_path: Path,
    _case_id: str,
    arrow_type: pa.DataType,
    _expected_kind: str,
    expected_dtype: str,
    sample: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"arrow_filter_empty_{_case_id}")
    frame = make_default_dataset_frame()
    frame["value"] = pd.Series(
        pd.array([sample, None, sample], dtype=pd.ArrowDtype(arrow_type)),
        index=frame.index,
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(
            f"arrow_filter_empty_{_case_id}",
            value_role=ColumnRole.CONTEXT,
        ),
    )
    policy = _issued_test_policy()

    with runtime_resource_policy_scope(policy):
        with dataset_reading.open_dataset_session(
            root,
            columns=("value",),
            time_window=None,
            row_filters=(("asset_id", "==", "missing"),),
            resource_policy=policy,
        ) as session:
            chunks = list(
                dataset_reading.iter_dataset_chunks(
                    session,
                    target_chunk_bytes=policy.remaining_memory_bytes,
                )
            )
            materialized = dataset_reading.materialize_dataset_session(
                session,
                target_chunk_bytes=policy.remaining_memory_bytes,
                materialized_budget_bytes=policy.task_memory_budget_bytes,
            )

    assert len(chunks) == 1
    assert chunks[0].empty
    assert materialized.empty
    assert str(chunks[0]["value"].dtype) == expected_dtype
    assert str(materialized["value"].dtype) == expected_dtype


@pytest.mark.parametrize(
    "descriptor",
    (
        {"name": "value", "kind": "numpy_boolean", "dtype": "bool[pyarrow]"},
        {"name": "value", "kind": "numpy_integer", "dtype": "int16[pyarrow]"},
        {"name": "value", "kind": "numpy_float", "dtype": "float[pyarrow]"},
        {
            "name": "value",
            "kind": "datetime",
            "dtype": "timestamp[ms][pyarrow]",
        },
        {"name": "value", "kind": "string", "dtype": "string[pyarrow]"},
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


@pytest.mark.parametrize(
    "series",
    (
        pd.Series(pd.Categorical(["b"], categories=["b", "a"], ordered=False)),
        pd.Series(pd.Categorical(["b"], categories=["b", "a"], ordered=True)),
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
                **({"categories": []} if descriptor["kind"] == "categorical" else {}),
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
            (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
        )
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ) as session:
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
        assert read_dataset(root)["asset_id"].astype("string").tolist() == ["live-only"]
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


def test_writer_does_not_retain_prepared_empty_frame_across_nonempty_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "empty_frame_lifecycle"
    empty = make_default_dataset_frame().iloc[:0].copy()
    nonempty = make_default_dataset_frame().iloc[:1].copy()
    empty_refs: list[weakref.ReferenceType[pd.DataFrame]] = []
    alive_at_nonempty: list[bool] = []
    original = dataset_writing._WriteStats.prepare_frame

    def record_prepare(stats, frame):  # noqa: ANN001
        if not frame.empty and empty_refs:
            alive_at_nonempty.append(empty_refs[-1]() is not None)
        prepared, logical_dtypes = original(stats, frame)
        if frame.empty:
            empty_refs.append(weakref.ref(prepared))
        return prepared, logical_dtypes

    monkeypatch.setattr(dataset_writing._WriteStats, "prepare_frame", record_prepare)
    write_dataset(
        make_dataset_artifact_root(tmp_path, name),
        (empty, nonempty),
        spec=make_processed_dataset_spec(name),
    )

    assert alive_at_nonempty == [False]


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
            current["timestamp"] = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        chunks.append(current)
    retained_column_counts: list[int] = []
    original = dataset_writing._WriteStats.observe_logical_dtype_payload

    def record_plan(self, *args, **kwargs):  # noqa: ANN001
        result = original(self, *args, **kwargs)
        assert self._logical_dtype_plan is not None
        retained_column_counts.append(len(self._logical_dtype_plan["columns"]))
        return result

    monkeypatch.setattr(
        dataset_writing._WriteStats,
        "observe_logical_dtype_payload",
        record_plan,
    )
    write_dataset(root, chunks, spec=make_processed_dataset_spec(name))

    assert retained_column_counts
    assert set(retained_column_counts) == {len(base.columns)}


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
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
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
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        manifest_path = root / "_manifest" / "manifest.json"
        manifest_path.write_bytes(session.manifest_bytes + b" ")
        with pytest.raises(ValueError, match="dataset_manifest_identity_changed"):
            session.verify()
        manifest_path.write_bytes(session.manifest_bytes)

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
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
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
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
            resource_policy=capture_test_policy(
                available_memory_bytes=64 * 1024**2,
                explicit_cap_bytes=32 * 1024 * 1024,
            ),
        )
    assert not root.exists()
    assert not list(root.parent.glob(".a-*"))


def test_noncanonical_year_partition_is_rejected_with_and_without_window(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "canonical_year")
    write_dataset(
        root,
        make_default_dataset_frame().iloc[:1],
        spec=make_processed_dataset_spec("canonical_year"),
    )
    canonical = next((root / "data").glob("year=*"))
    canonical.rename(canonical.with_name("year=02014"))

    for window in (None, TimeWindow("2014-01-01", "2014-12-31")):
        with pytest.raises(ValueError, match="invalid_dataset_part_files"):
            with dataset_reading.open_dataset_session(
                root,
                columns=None,
                time_window=window,
            ):
                pass


@pytest.mark.parametrize(
    "columns",
    (
        ["market_id", "asset_id", "timestamp", "value", "value"],
        ["market_id", "asset_id", "timestamp", 1],
        ["market_id", "asset_id", "timestamp", ""],
    ),
)
def test_writer_rejects_noncanonical_dataframe_column_labels_before_arrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    columns: list[object],
) -> None:
    frame = pd.DataFrame([[0] * len(columns)], columns=columns)
    monkeypatch.setattr(
        dataset_writing._WriteStats,
        "table_for_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid_columns_reached_arrow")
        ),
    )
    with pytest.raises(ValueError, match="invalid_dataset_column_labels"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "invalid_labels"),
            frame,
            spec=make_processed_dataset_spec("invalid_labels"),
            resource_policy=active_runtime_resource_policy(),
        )


@pytest.mark.parametrize("columns", ([1], ["value", "value"], [""]))
def test_projection_requires_exact_unique_nonempty_string_labels(
    tmp_path: Path,
    columns: list[object],
) -> None:
    root = make_dataset_artifact_root(tmp_path, "projection_grammar")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("projection_grammar"),
    )
    with pytest.raises(ValueError, match="dataset_columns_invalid"):
        read_dataset(root, columns=columns)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), pd.NaT, True))
def test_physical_filter_rejects_nonfinite_or_schema_incompatible_scalar(
    tmp_path: Path,
    value: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "filter_grammar")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("filter_grammar"),
    )
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=(("value", "==", value),),
        ):
            pass


@pytest.mark.parametrize(
    ("data_type", "value"),
    (
        (pa.float64(), 1),
        (pa.decimal128(12, 2), 1),
        (pa.decimal128(12, 2), 1.0),
        (pa.int64(), 1.0),
    ),
)
def test_physical_filter_requires_exact_scalar_family(
    data_type: pa.DataType,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._validated_physical_filter_value(
            pa.field("value", data_type),
            value,
        )


def test_physical_in_filter_uses_bounded_columnar_equality_not_arrow_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 2], type=pa.int64())],
        names=["value"],
    )

    def reject_hash(*_args, **_kwargs):
        raise AssertionError("physical membership must not allocate native hash state")

    monkeypatch.setattr(dataset_physical.pc, "is_in", reject_hash)

    values = pa.array([2, 2, 3], type=pa.int64())
    prepared = dataset_physical.PreparedMembershipValues(values, 8, values.nbytes)
    filtered = dataset_physical._filter_arrow_batch(
        batch,
        [("value", "in", prepared)],
        operation_headroom_bytes=1024,
    )

    assert filtered.column(0).to_pylist() == [2, 3, 2]


def test_physical_membership_compute_scales_with_domain_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(32), type=pa.int64())],
        names=["value"],
    )
    values = pa.array(range(1024), type=pa.int64())
    prepared = dataset_physical.PreparedMembershipValues(values, 8, values.nbytes)
    equality_calls = 0
    original_equal = dataset_physical.pc.equal

    def count_equal(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal equality_calls
        equality_calls += 1
        return original_equal(*args, **kwargs)

    monkeypatch.setattr(dataset_physical.pc, "equal", count_equal)
    filtered = dataset_physical._filter_arrow_batch(
        batch,
        [("value", "in", prepared)],
        operation_headroom_bytes=16 * 1024,
    )

    assert filtered.column(0).to_pylist() == list(range(32))
    assert equality_calls <= math.ceil(math.log2(len(values) + 1)) + 1


@pytest.mark.parametrize(
    ("data_type", "source", "members", "expected"),
    (
        (
            pa.timestamp("ns"),
            (
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                None,
            ),
            (pd.Timestamp("2026-01-02"), datetime(2026, 1, 1)),
            2,
        ),
        (
            pa.duration("ns"),
            (timedelta(days=1), timedelta(days=2), None),
            (pd.Timedelta(days=2), timedelta(days=1)),
            2,
        ),
        (
            pa.float64(),
            (0.0, -0.0, 2.0, None),
            (2.0, -0.0, 0.0, 2.0),
            3,
        ),
    ),
    ids=("timestamp", "duration", "float-negative-zero"),
)
def test_physical_membership_preserves_generic_exact_scalar_semantics(
    data_type: pa.DataType,
    source: tuple[object, ...],
    members: tuple[object, ...],
    expected: int,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", data_type),
        members,
        resource_policy=policy,
        retained_filter_bytes=0,
    )
    batch = pa.RecordBatch.from_arrays(
        [pa.array(source, type=data_type)],
        names=["value"],
    )

    filtered = dataset_physical._filter_arrow_batch(
        batch,
        [("value", "in", prepared)],
        operation_headroom_bytes=16 * 1024,
    )

    assert filtered.num_rows == expected
    assert filtered.column(0).null_count == 0


def test_physical_membership_rejects_null_filter_member_before_sort() -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._prepare_in_filter(
            pa.field("value", pa.int64()),
            (1, None),
            resource_policy=policy,
            retained_filter_bytes=0,
        )


def test_physical_in_filter_rejects_unadmitted_batch_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(128), type=pa.int64())],
        names=["value"],
    )

    def reject_compute(*_args, **_kwargs):
        raise AssertionError("filter compute must not run before working-set admission")

    monkeypatch.setattr(dataset_physical.pc, "equal", reject_compute)

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_filter_working_set_exceeds_task_memory_budget",
    ):
        dataset_physical._filter_arrow_batch(
            batch,
            [
                (
                    "value",
                    "in",
                    dataset_physical.PreparedMembershipValues(
                        pa.array([1], type=pa.int64()),
                        8,
                        8,
                    ),
                )
            ],
            operation_headroom_bytes=batch.get_total_buffer_size(),
        )


@pytest.mark.parametrize(
    ("operator", "compute_name"),
    (("==", "equal"), (">=", "greater_equal")),
)
def test_physical_scalar_filter_rejects_unadmitted_batch_before_compute(
    monkeypatch: pytest.MonkeyPatch,
    operator: str,
    compute_name: str,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(128), type=pa.int64())],
        names=["value"],
    )

    def reject_compute(*_args, **_kwargs):
        raise AssertionError("filter compute must not run before working-set admission")

    monkeypatch.setattr(dataset_physical.pc, compute_name, reject_compute)

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_filter_working_set_exceeds_task_memory_budget",
    ):
        dataset_physical._filter_arrow_batch(
            batch,
            [("value", operator, 1)],
            operation_headroom_bytes=batch.get_total_buffer_size(),
        )


def test_physical_filter_requires_explicit_operation_headroom() -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1], type=pa.int64())],
        names=["value"],
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_filter_operation_headroom_required",
    ):
        dataset_physical._filter_arrow_batch(
            batch,
            [("value", "==", 1)],
        )


def test_physical_membership_requires_prepared_arrow_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], type=pa.int64())],
        names=["value"],
    )
    unprepared = pa.array([1, 2], type=pa.int64())

    def reject_array(*_args, **_kwargs):
        raise AssertionError("physical filtering must not prepare retained values")

    monkeypatch.setattr(dataset_physical.pa, "array", reject_array)

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_physical._filter_arrow_batch(
            batch,
            [("value", "in", (1, 2))],
            operation_headroom_bytes=1024,
        )

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_physical._filter_arrow_batch(
            batch,
            [("value", "in", unprepared)],
            operation_headroom_bytes=1024,
        )


@pytest.mark.parametrize(
    ("operator", "value", "error"),
    (
        ("in", (1, 2), "invalid_dataset_physical_filter_value"),
        ("contains", 1, "unsupported_dataset_physical_filter"),
    ),
)
def test_physical_filter_grammar_precedes_capacity_admission(
    operator: str,
    value: object,
    error: str,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], type=pa.int64())],
        names=["value"],
    )

    with pytest.raises(ValueError, match=error):
        dataset_physical._filter_arrow_batch(
            batch,
            [("value", operator, value)],
            operation_headroom_bytes=1,
        )


def test_physical_membership_releases_search_temporaries_after_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], type=pa.int64())], names=["value"]
    )
    references: list[weakref.ReferenceType[pa.Array]] = []
    original_equal = dataset_physical.pc.equal

    def observe_equal(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = original_equal(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(dataset_physical.pc, "equal", observe_equal)
    filtered = dataset_physical._filter_arrow_batch(
        batch,
        [
            (
                "value",
                "in",
                dataset_physical.PreparedMembershipValues(
                    pa.array([1, 3], type=pa.int64()), 8, 16
                ),
            )
        ],
        operation_headroom_bytes=4096,
    )

    assert filtered.column(0).to_pylist() == [1, 3]
    assert references and all(reference() is None for reference in references)


def test_physical_multi_filter_admits_each_current_filtered_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 4], type=pa.int64())],
        names=["value"],
    )
    observed: list[weakref.ReferenceType[pa.RecordBatch]] = []
    original = dataset_physical._require_filter_batch_capacity

    def observe(current: pa.RecordBatch, **kwargs) -> None:  # noqa: ANN003
        observed.append(weakref.ref(current))
        original(current, **kwargs)

    monkeypatch.setattr(
        dataset_physical,
        "_require_filter_batch_capacity",
        observe,
    )

    filtered = dataset_physical._filter_arrow_batch(
        batch,
        [("value", ">=", 2), ("value", "<=", 3)],
        operation_headroom_bytes=1024,
    )

    assert observed[0]() is batch
    assert observed[1]() is None
    assert filtered.column(0).to_pylist() == [2, 3]


def test_physical_owner_releases_raw_batch_before_bounded_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), path)
    raw_reference: weakref.ReferenceType[pa.RecordBatch] | None = None

    def replace_raw(raw_batch, *_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal raw_reference
        raw_reference = weakref.ref(raw_batch)
        return pa.RecordBatch.from_arrays(
            [pa.array(raw_batch.column(0).to_pylist())], names=["value"]
        )

    monkeypatch.setattr(dataset_physical, "_filter_arrow_batch", replace_raw)

    batches = dataset_physical.iter_dataset_part_batches(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            target_chunk_bytes=1024,
        )
    try:
        first = next(batches)
        assert raw_reference is not None and raw_reference() is None
        assert first.column(0).to_pylist() == [1, 2]
    finally:
        batches.close()


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ({"duplicate_primary_key_count": 999}, "manifest_summary_count_invalid"),
        ({"date_min": None, "date_max": None}, "manifest_date_range_invalid"),
        (
            {"date_min": "2099-01-01", "date_max": "2099-12-31"},
            "manifest_partition_date_mismatch",
        ),
        (
            {
                "partitions": ["year=2014", "year=02015"],
                "partition_count": 2,
                "partition_row_counts": {"year=2014": 3, "year=02015": 0},
            },
            "invalid_dataset_part_files",
        ),
    ),
)
def test_manifest_structural_truth_rejects_impossible_counts_and_dates(
    tmp_path: Path,
    mutation: dict[str, object],
    error: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "manifest_truth")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("manifest_truth"),
    )
    spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(mutation)
    with pytest.raises(ValueError, match=error):
        dataset_schema.validate_dataset_manifest_payload(manifest, spec=spec)


def test_unpartitioned_non_primary_timestamp_round_trips_without_date_range(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "unpartitioned_non_key_timestamp")
    spec = make_processed_dataset_spec(
        "unpartitioned_non_key_timestamp",
        primary_key=("market_id", "asset_id"),
        partition_columns=(),
    )
    frame = make_default_dataset_frame().iloc[:2].reset_index(drop=True)

    write_dataset(root, frame, spec=spec)
    inspection = inspect_dataset(root)

    assert inspection.manifest["date_min"] is None
    assert inspection.manifest["date_max"] is None
    pd.testing.assert_frame_equal(read_dataset(root), frame)
    forged_manifest = dict(inspection.manifest)
    forged_manifest.update(date_min="2014-01-02", date_max="2014-01-02")
    with pytest.raises(ValueError, match="manifest_date_range_invalid"):
        dataset_schema.validate_dataset_manifest_payload(
            forged_manifest,
            spec=inspection.spec,
        )


def test_session_metadata_reader_rejects_oversize_before_buffering(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "bounded_metadata")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("bounded_metadata"),
    )
    (root / "_manifest" / "manifest.json").write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="dataset_metadata_file_too_large"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            pass


@pytest.mark.parametrize("materialized_budget_bytes", (1025, 1026))
def test_materialize_rejects_exact_buffers_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    materialized_budget_bytes: int,
) -> None:
    frames = (
        pd.DataFrame({"value": pd.Series(["a" * 128], dtype="string")}),
        pd.DataFrame({"value": pd.Series(["b" * 128], dtype="string")}),
    )
    calls = 0

    def chunks(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        yield from frames

    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        chunks,
    )
    monkeypatch.setattr(
        dataset_reading,
        "_allocate_materialized_buffers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("buffers_allocated_before_headroom_check")
        ),
    )
    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frames[0]),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
    )
    with pytest.raises(
        ValueError,
        match="dataset_materialized_result_exceeds_target_bytes",
    ):
        dataset_reading.materialize_dataset_session(
            context,
            target_chunk_bytes=1024,
            materialized_budget_bytes=materialized_budget_bytes,
        )
    assert calls == 1


def test_writer_checks_actual_arrow_headroom_after_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _writer_test_policy()
    monkeypatch.setattr(
        dataset_writing,
        "pandas_frame_payload_bytes",
        lambda _: 100_000,
    )
    conversions = 0

    def oversized_table(*_args, **_kwargs):
        nonlocal conversions
        conversions += 1
        return SimpleNamespace(nbytes=900_000)

    monkeypatch.setattr(dataset_writing._WriteStats, "table_for_frame", oversized_table)
    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "writer_arrow_admission"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("writer_arrow_admission"),
            resource_policy=policy,
        )

    assert conversions == 1


def test_writer_deducts_other_live_root_reservations_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _writer_test_policy()
    allocations = 0

    def forbid_prepare(*_args, **_kwargs):  # noqa: ANN002, ANN003
        nonlocal allocations
        allocations += 1
        raise AssertionError("writer_allocated_without_shared_headroom")

    monkeypatch.setattr(dataset_writing._WriteStats, "prepare_frame", forbid_prepare)
    other_owner_bytes = (
        policy.task_memory_budget_bytes
        - dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
    )

    with runtime_resource_policy_scope(policy):
        with policy.reserve("test.outer.owner", other_owner_bytes):
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                dataset_writing.write_dataset(
                    make_dataset_artifact_root(tmp_path, "writer_shared_headroom"),
                    make_default_dataset_frame(),
                    spec=make_processed_dataset_spec("writer_shared_headroom"),
                    resource_policy=policy,
                )

    assert allocations == 0


def test_writer_rejects_before_pandas_conversion_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _writer_test_policy(working_headroom_bytes=200_000)
    monkeypatch.setattr(
        dataset_writing,
        "pandas_frame_payload_bytes",
        lambda _: 100_000,
    )
    conversions = 0

    def forbid_prepare(*_args, **_kwargs):  # noqa: ANN002, ANN003
        nonlocal conversions
        conversions += 1
        raise AssertionError("pandas_conversion_started_before_admission")

    monkeypatch.setattr(dataset_writing._WriteStats, "prepare_frame", forbid_prepare)

    with pytest.raises(
        (MemoryError, ValueError),
        match=(
            "runtime_memory_budget_exceeded|"
            "dataset_working_set_exceeds_task_memory_budget"
        ),
    ):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "writer_conversion_admission"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("writer_conversion_admission"),
            resource_policy=policy,
        )

    assert conversions == 0


def test_writer_rejects_timestamp_prepare_before_large_temporary_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2014-01-01",
                periods=200_000,
                freq="min",
            )
        }
    )
    policy = _writer_test_policy(working_headroom_bytes=200_000)
    spec = replace(
        make_processed_dataset_spec("writer_timestamp_admission"),
        column_roles={"timestamp": ColumnRole.TIMESTAMP},
        primary_key=("timestamp",),
    )
    conversions = 0

    def forbid_timestamp_conversion(*_args, **_kwargs):
        nonlocal conversions
        conversions += 1
        raise AssertionError("timestamp_conversion_started_before_admission")

    monkeypatch.setattr(
        dataset_writing,
        "strict_timestamp_series",
        forbid_timestamp_conversion,
    )

    with pytest.raises(
        (MemoryError, ValueError),
        match=(
            "runtime_memory_budget_exceeded|"
            "dataset_working_set_exceeds_task_memory_budget"
        ),
    ):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(
                tmp_path,
                "writer_timestamp_admission",
            ),
            frame,
            spec=spec,
            resource_policy=policy,
        )

    assert conversions == 0


def test_writer_rejects_before_native_parquet_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_table = pa.table({"payload": [b"x" * 600_000]})
    policy = _writer_test_policy(working_headroom_bytes=native_table.nbytes + 7)
    monkeypatch.setattr(
        dataset_writing,
        "pandas_frame_payload_bytes",
        lambda _: 1,
    )
    conversions = 0
    native_writes = 0

    def convert_to_table(*_args, **_kwargs):  # noqa: ANN002, ANN003
        nonlocal conversions
        conversions += 1
        return native_table

    def forbid_native_write(*_args, **_kwargs):  # noqa: ANN002, ANN003
        nonlocal native_writes
        native_writes += 1
        raise AssertionError("native_write_started_before_admission")

    monkeypatch.setattr(
        dataset_writing._WriteStats,
        "table_for_frame",
        convert_to_table,
    )
    monkeypatch.setattr(dataset_writing.pq, "write_table", forbid_native_write)

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "writer_native_admission"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("writer_native_admission"),
            resource_policy=policy,
        )

    assert conversions == 1
    assert native_writes == 0


def test_writer_bounded_completes_under_task_wide_copy_and_arrow_headroom(
    tmp_path: Path,
) -> None:
    row_count = 400
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": [f"asset-{index:04d}" for index in range(row_count)],
            "timestamp": pd.date_range("2014-01-02", periods=row_count),
            "value": ["x" * 1_500] * row_count,
        }
    )
    policy = _writer_test_policy()
    root = make_dataset_artifact_root(tmp_path, "writer_task_headroom")
    result = dataset_writing.write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("writer_task_headroom"),
        resource_policy=policy,
    )

    assert result.row_count == row_count
    assert len(tuple(root.glob("data/**/*.parquet"))) > 1
    assert policy.reserved_memory_bytes == 0


@pytest.mark.parametrize(
    "variant",
    ("normal", "empty", "override"),
)
def test_writer_persisted_readback_preserves_schema_and_rows(
    tmp_path: Path,
    variant: str,
) -> None:
    frame = make_default_dataset_frame()
    if variant == "empty":
        frame = frame.iloc[:0].copy()
    elif variant == "override":
        frame["value"] = pd.Series(frame["value"], dtype="Int64")
    policy = _writer_test_policy()
    root = make_dataset_artifact_root(tmp_path, f"writer_readback_{variant}")

    dataset_writing.write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(f"writer_readback_{variant}"),
        resource_policy=policy,
    )

    persisted = read_dataset(root)
    assert tuple(persisted.columns) == tuple(frame.columns)
    assert len(persisted) == len(frame)
    if variant == "override":
        assert str(persisted["value"].dtype) == "Int64"
    if variant != "empty":
        assert sorted(
            zip(persisted["asset_id"], persisted["timestamp"], strict=True)
        ) == sorted(zip(frame["asset_id"], frame["timestamp"], strict=True))


@pytest.mark.parametrize(
    "mode",
    (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED),
)
def test_read_admission_occurs_after_metadata_preflight_and_at_lazy_activation(
    monkeypatch: pytest.MonkeyPatch,
    mode: DatasetReadMode,
) -> None:
    events: list[str] = []
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    class Context:
        resource_policy = policy

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args):  # noqa: ANN204
            return None

        def verify(self) -> None:
            return None

    context = Context()

    def preflight(*_args, **_kwargs):
        events.append("preflight")
        admit()
        return context

    def admit() -> RuntimeResourcePolicy:
        events.append("admission")
        return policy

    monkeypatch.setattr(dataset_reading, "open_dataset_session", preflight)
    monkeypatch.setattr(dataset_reading, "active_runtime_resource_policy", admit)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((pd.DataFrame({"value": [1]}),)),
    )
    monkeypatch.setattr(
        dataset_reading,
        "materialize_dataset_session",
        lambda *_args, **_kwargs: pd.DataFrame({"value": [1]}),
    )

    result = dataset_reading.read_dataset(
        Path("ignored"),
        mode=mode,
        resource_policy=None,
    )
    expected_events = (
        [] if mode is DatasetReadMode.CHUNKED else ["preflight", "admission"]
    )
    assert events == expected_events
    if mode is DatasetReadMode.CHUNKED:
        list(result)
        assert events == ["preflight", "admission"]


def test_public_materialized_read_rejects_competing_active_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _issued_test_policy()
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("competing root reached dataset preflight")
        ),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            RuntimeError,
            match="^runtime_resource_policy_competing_root$",
        ):
            read_dataset(Path("unused"), mode=DatasetReadMode.MATERIALIZED)


def test_chunked_read_defers_session_inventory_until_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Session:
        resource_policy = active_runtime_resource_policy()

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args):  # noqa: ANN204
            return None

    def open_session(*_args, **_kwargs):
        events.append("inventory")
        return Session()

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(()),
    )
    chunks = dataset_reading.read_dataset(
        Path("unused"),
        mode=DatasetReadMode.CHUNKED,
        resource_policy=active_runtime_resource_policy(),
    )
    assert events == []
    next(chunks, None)
    assert events == ["inventory"]


def test_chunked_explicit_inactive_policy_preflights_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    class Session:
        resource_policy = policy

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args):  # noqa: ANN204
            events.append("cleanup")

    @contextmanager
    def preflight(*_args, **_kwargs):  # noqa: ANN202
        events.append("preflight")
        with scope(policy):
            with Session() as session:
                yield session

    @contextmanager
    def scope(_policy):  # noqa: ANN001, ANN202
        events.append("admission")
        try:
            yield policy
        finally:
            events.append("release")

    monkeypatch.setattr(dataset_reading, "open_dataset_session", preflight)
    monkeypatch.setattr(dataset_reading, "runtime_resource_policy_scope", scope)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(()),
    )

    chunks = dataset_reading.read_dataset(
        Path("dataset"),
        mode=DatasetReadMode.CHUNKED,
        resource_policy=policy,
    )
    assert events == []
    list(chunks)

    assert events == ["preflight", "admission", "cleanup", "release"]


def test_chunked_session_inventory_is_removed_on_early_close(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_early_close")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inventory_early_close"),
    )
    chunks = read_dataset(root, mode=DatasetReadMode.CHUNKED)
    next(chunks)
    assert list(root.parent.glob(".i-*"))
    chunks.close()  # type: ignore[attr-defined]
    assert not list(root.parent.glob(".i-*"))


def test_dataset_inventory_cleanup_removes_sqlite_sidecars(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    inventory = dataset_inventory.DatasetSessionInventory(
        data_root=data_root,
        scratch_parent=tmp_path,
        task_memory_budget_bytes=1024,
    )
    scratch = inventory.scratch_path
    assert (scratch / "i.db").is_file()
    assert inventory.retained_memory_bytes == 1024
    assert inventory._connection is not None
    assert inventory._connection.execute("PRAGMA cache_size").fetchone() == (-1,)
    inventory.close()
    assert not scratch.exists()
    assert not list(tmp_path.glob(".i-*"))
    assert not list(tmp_path.rglob("i.db-wal"))
    assert not list(tmp_path.rglob("i.db-shm"))


def test_dataset_inventory_stores_null_columns_as_exact_relational_rows(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_null_columns")
    frame = make_default_dataset_frame(
        asset_ids=("asset_001", "asset_001"),
        dates=("2014-01-02", "2014-01-03"),
        values=(1.0, float("nan")),
    )
    write_dataset(
        root,
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_processed_dataset_spec("inventory_null_columns"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        scratch = session.inventory.scratch_path
        entries = tuple(session.inventory.iter_entries())
        connection = session.inventory._require_connection()
        rows = connection.execute(
            "SELECT relative_path, column_name FROM part_null_columns "
            "ORDER BY relative_path, column_name"
        ).fetchall()
        assert [metadata.null_columns for _path, metadata in entries] == [
            frozenset(),
            frozenset({"value"}),
        ]
        assert rows == [
            (entries[1][0].relative_to(session.inventory.data_root).as_posix(), "value")
        ]

    assert not scratch.exists()


def test_dataset_inventory_low_budget_and_inspection_failure_clean_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(ValueError, match="dataset_inventory_budget_too_small"):
        dataset_inventory.DatasetSessionInventory(
            data_root=data_root,
            scratch_parent=tmp_path,
            task_memory_budget_bytes=1023,
        )
    assert not list(tmp_path.glob(".i-*"))

    root = make_dataset_artifact_root(tmp_path, "inventory_inspection_failure")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inventory_inspection_failure"),
    )
    monkeypatch.setattr(
        dataset_reading,
        "inspect_dataset_part",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("inspect_failed")),
    )
    with pytest.raises(RuntimeError, match="inspect_failed"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            pass
    assert not list(root.parent.glob(".i-*"))


def test_dataset_session_inventory_preserves_numeric_part_order(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_numeric_order")
    base = make_default_dataset_frame()
    later = base.copy()
    later["timestamp"] = later["timestamp"] + pd.Timedelta(days=2)
    frame = pd.concat((base, later), ignore_index=True)
    write_dataset(
        root,
        tuple(frame.iloc[index : index + 1].copy() for index in range(len(frame))),
        spec=make_processed_dataset_spec("inventory_numeric_order"),
    )
    parts = sorted((root / "data").rglob("part-*.parquet"))
    part_indices = tuple(range(998, 998 + len(parts)))
    for path, part_index in zip(parts, part_indices, strict=True):
        path.rename(path.with_name(f"part-{part_index}.parquet"))

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        expected = [
            f"part-{part_index}.parquet" for part_index in part_indices
        ]
        assert [path.name for path in session.all_paths] == expected
        selection = session.inventory.new_path_selection()
        for path in reversed(tuple(session.all_paths)):
            session.inventory.select_path(selection, path)
        connection = session.inventory._require_connection()
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        try:
            first_selected = session.inventory.path_at(
                0,
                selected=True,
                path_selection=selection,
            )
            selected = list(
                session.inventory.iter_paths(
                    selected=True,
                    path_selection=selection,
                )
            )
        finally:
            connection.set_trace_callback(None)
        assert first_selected.name == expected[0]
        assert [path.name for path in selected] == expected
        ordered_selection = [
            statement
            for statement in statements
            if "path_selections" in statement.casefold()
            and "order by" in statement.casefold()
        ]
        assert ordered_selection
        plan = [
            detail
            for statement in ordered_selection
            for *_prefix, detail in connection.execute(
                "EXPLAIN QUERY PLAN " + statement
            )
        ]
        assert not any("TEMP B-TREE" in detail for detail in plan)
        session.inventory.release_path_selection(selection)


def test_dataset_inventory_selected_count_is_selection_index_driven(
    tmp_path: Path,
) -> None:
    with dataset_inventory.DatasetSessionInventory(
        data_root=tmp_path / "data",
        scratch_parent=tmp_path,
        task_memory_budget_bytes=1024,
    ) as inventory:
        connection = inventory._require_connection()
        with connection:
            connection.executemany(
                "INSERT INTO parts(relative_path, partition_kind, partition_year, "
                "part_index, selected, device, inode, size, modified_ns) "
                "VALUES (?, 0, 0, ?, 1, ?, ?, '1', '1')",
                (
                    (
                        f"part-{index}.parquet",
                        index,
                        f"device-{index}",
                        f"inode-{index}",
                    )
                    for index in range(5_000)
                ),
            )
            selection = inventory.new_path_selection()
            connection.execute(
                "INSERT INTO path_selections(selection_id, relative_path) "
                "VALUES (?, ?)",
                (selection, "part-4999.parquet"),
            )

        statements: list[str] = []
        vm_steps = 0

        def count_vm_step() -> int:
            nonlocal vm_steps
            vm_steps += 1
            return 0

        connection.set_trace_callback(statements.append)
        connection.set_progress_handler(count_vm_step, 1)
        try:
            assert inventory.path_count(
                selected=True,
                path_selection=selection,
            ) == 1
        finally:
            connection.set_progress_handler(None, 0)
            connection.set_trace_callback(None)

        count_query = next(
            statement
            for statement in statements
            if statement.lstrip().upper().startswith("SELECT COUNT(*)")
            and "path_selections" in statement.casefold()
        )
        plan = [
            detail
            for *_prefix, detail in connection.execute(
                "EXPLAIN QUERY PLAN " + count_query
            )
        ]
        assert not any("TEMP B-TREE" in detail for detail in plan)
        assert not any("SCAN PARTS" in detail.upper() for detail in plan)
        assert vm_steps < 200


def test_dataset_inventory_disk_gates_native_page_and_encoded_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_calls: list[int] = []
    monkeypatch.setattr(
        dataset_inventory,
        "require_free_disk_headroom",
        lambda _path, *, reserve_bytes: reserve_calls.append(reserve_bytes),
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    root = make_dataset_artifact_root(tmp_path, "inventory_disk_gate")
    dataset_writing.write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inventory_disk_gate"),
        resource_policy=policy,
    )

    assert reserve_calls[0] >= 32 * dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
    assert len(reserve_calls) > 1
    assert policy.task_memory_budget_bytes not in reserve_calls
    assert all(value > 0 for value in reserve_calls)
    physical_floor = 16 * dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
    assert min(reserve_calls[1:]) >= physical_floor


def test_dataset_inventory_release_gates_before_selection_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_selection_release")
    frame = make_default_dataset_frame()
    write_dataset(
        root,
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_processed_dataset_spec("inventory_selection_release"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        paths = tuple(session.all_paths)
        selection = session.inventory.new_path_selection()
        for path in paths:
            session.inventory.select_path(selection, path)
        monkeypatch.setattr(
            dataset_inventory,
            "require_free_disk_headroom",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("selection_release_disk_full")
            ),
        )
        with pytest.raises(OSError, match="selection_release_disk_full"):
            session.inventory.release_path_selection(selection)
        assert tuple(
            session.inventory.iter_paths(selected=True, path_selection=selection)
        ) == paths
        monkeypatch.setattr(
            dataset_inventory,
            "require_free_disk_headroom",
            lambda *_args, **_kwargs: None,
        )
        session.inventory.release_path_selection(selection)
        assert session.inventory.path_count(
            selected=True,
            path_selection=selection,
        ) == 0


def test_dataset_inventory_insert_disk_failure_cleans_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_disk_failure")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inventory_disk_failure"),
    )
    reserve_calls = 0

    def fail_first_row(_path: Path, *, reserve_bytes: int) -> None:
        nonlocal reserve_calls
        reserve_calls += 1
        assert reserve_bytes > 0
        if reserve_calls > 1:
            raise OSError("inventory_disk_full")

    monkeypatch.setattr(
        dataset_inventory,
        "require_free_disk_headroom",
        fail_first_row,
    )
    with pytest.raises(OSError, match="inventory_disk_full"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            pass

    assert reserve_calls == 2
    assert not list(root.parent.glob(".i-*"))


def test_dataset_inventory_init_failure_closes_local_connection_and_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingConnection:
        closed = False

        def execute(self, _statement: str) -> None:
            raise RuntimeError("inventory_init_failed")

        def close(self) -> None:
            self.closed = True

    connection = FailingConnection()
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(dataset_inventory.sqlite3, "connect", lambda _path: connection)

    with pytest.raises(RuntimeError, match="inventory_init_failed"):
        dataset_inventory.DatasetSessionInventory(
            data_root=data_root,
            scratch_parent=tmp_path,
            task_memory_budget_bytes=1024,
        )

    assert connection.closed
    assert not list(tmp_path.glob(".i-*"))


def test_inspection_window_file_count_counts_consecutive_file_batches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data" / "year=2026" / "part-000.parquet"
    empty_path = path.with_name("part-001.parquet")
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]})

    class KeyIndex:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Observer:
        row_count = 0
        date_min = None
        date_max = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        def observe_batch(
            self,
            _batch: pa.RecordBatch,
            *,
            frame: pd.DataFrame,
        ) -> None:
            observed = frame
            self.row_count += len(observed)

    monkeypatch.setattr(dataset_inspection, "ExactPrimaryKeyIndex", KeyIndex)
    monkeypatch.setattr(dataset_inspection, "DatasetSummaryObserver", Observer)
    monkeypatch.setattr(
        dataset_inspection,
        "dataset_primary_key_schema",
        lambda *_args, **_kwargs: pa.schema([]),
    )
    observed_headroom: list[int | None] = []

    def batches(*_args, **kwargs):  # noqa: ANN003
        observed_headroom.append(kwargs.get("operation_headroom_bytes"))
        observer = kwargs["batch_observer"]
        for _, observed in ((path, frame), (path, frame)):
            observer(pa.RecordBatch.from_pandas(observed), observed)
        return iter(((path, frame), (path, frame), (empty_path, frame.iloc[:0])))

    monkeypatch.setattr(
        dataset_inspection,
        "iter_dataset_file_batches",
        batches,
    )
    context = SimpleNamespace(
        timestamp_column="timestamp",
        time_window=TimeWindow("2026-01-01", "2026-01-02"),
        spec={
            "primary_key": ["timestamp"],
            "column_roles": {"timestamp": "timestamp"},
        },
        root=tmp_path,
        data_root=tmp_path / "data",
        arrow_schema=SimpleNamespace(names=["timestamp"]),
        schema=dataset_schema.physical_schema_from_schema(
            pa.schema((pa.field("timestamp", pa.timestamp("ns")),))
        ),
        all_paths=(),
        manifest={},
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        retained_memory_bytes=1024,
        resource_policy=capture_test_policy(
            available_memory_bytes=16 * 1024**2,
        ),
        read_paths=(path, empty_path),
    )

    runtime = dataset_inspection._inspection_runtime_metadata(
        context,
        target_chunk_bytes=1024,
        retained_payload_bytes=0,
    )

    assert runtime["file_count"] == 1
    assert runtime["row_count"] == 2
    observer_bytes = 2 * len(b"timestamp") + 24
    assert observed_headroom == [
        context.resource_policy.task_memory_budget_bytes
        - dataset_inspection.EXACT_KEY_SQLITE_PAGE_BYTES
        - dataset_reading.dataset_partition_state_reserve_bytes(context)
        - observer_bytes
    ]


def test_inspection_rejects_output_overlap_after_partition_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    data_root = tmp_path / "data"
    paths = tuple(
        data_root / f"partition={index:03d}" / "part-000.parquet"
        for index in range(100)
    )
    schema = pa.schema((pa.field("id", pa.int64()),))
    traversals = 0

    def batches(*_args: object, **kwargs: object):
        nonlocal traversals
        observer = kwargs["batch_observer"]
        for index, path in enumerate(paths):
            frame = pd.DataFrame({"id": [index]})
            batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
            traversals += 1
            observer(batch, frame)
            yield path, frame

    monkeypatch.setattr(dataset_inspection, "iter_dataset_file_batches", batches)
    context = SimpleNamespace(
        resource_policy=policy,
        retained_memory_bytes=0,
        root=tmp_path,
        data_root=data_root,
        read_paths=paths,
        all_paths=paths,
        spec={"primary_key": ["id"], "column_roles": {"id": "key"}},
        manifest={},
        schema=dataset_schema.physical_schema_from_schema(schema),
        arrow_schema=schema,
        schema_hash="schema",
        timestamp_column=None,
        time_window=None,
    )
    accumulated_state_bytes = (
        dataset_reading.dataset_partition_state_reserve_bytes(context)
        + len(b"id")
        + len(b"key")
        + 24
    )
    retained_payload_bytes = policy.task_memory_budget_bytes - (
        dataset_inspection.EXACT_KEY_SQLITE_PAGE_BYTES + accumulated_state_bytes + 256
    )

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_inspection.inspect_dataset_session(
            context,
            resource_policy=policy,
            retained_payload_bytes=retained_payload_bytes,
        )

    assert traversals == len(paths)


def test_bounded_metadata_reader_rejects_hardlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = tmp_path / "external.json"
    external.write_text('{"schema_id":"outside"}', encoding="utf-8")
    candidate = tmp_path / "manifest.json"
    try:
        os.link(external, candidate)
    except OSError:
        pytest.skip("hardlink_not_available")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardlinked_metadata_opened")
        ),
    )

    with pytest.raises(ValueError, match="dataset_metadata_file_invalid"):
        storage_manifests.read_bounded_qrp_json(candidate)


def test_private_writer_envelope_reserves_upstream_retained_payload(
    tmp_path: Path,
) -> None:
    policy = _writer_test_policy()
    frame = make_default_dataset_frame()
    envelope = dataset_writing.DatasetWriteChunk(
        frame,
        policy.task_memory_budget_bytes,
    )

    with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "retained_publication"),
            iter((envelope,)),
            spec=make_processed_dataset_spec("retained_publication"),
            resource_policy=policy,
        )


def test_writer_uses_current_declared_credit_across_equal_and_unequal_yields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _writer_test_policy(8 * 1024**2)
    frame = make_default_dataset_frame()
    parts = tuple(frame.iloc[index : index + 1].copy() for index in range(3))
    credits = (4096, 4096, 8192)
    observed_input_reserves: list[int] = []
    original_reserve = RuntimeResourcePolicy.reserve

    @contextmanager
    def record_reserve(self, owner: str, reserve_bytes: int):  # noqa: ANN001
        if owner == "storage.dataset.input":
            observed_input_reserves.append(reserve_bytes)
        with original_reserve(self, owner, reserve_bytes):
            yield

    monkeypatch.setattr(RuntimeResourcePolicy, "reserve", record_reserve)

    def publications():
        for index, (part, credit) in enumerate(
            zip(parts, credits, strict=True)
        ):
            with policy.reserve(f"test.producer.{index}", credit):
                yield dataset_writing.DatasetWriteChunk(part, credit)

    dataset_writing.write_dataset(
        make_dataset_artifact_root(tmp_path, "producer_credit"),
        publications(),
        spec=make_processed_dataset_spec("producer_credit"),
        resource_policy=policy,
    )

    assert observed_input_reserves == [
        pandas_frame_payload_bytes(part) for part in parts
    ]
    assert policy.reserved_memory_bytes == 0


def test_private_writer_baseline_rejects_before_input_generation(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    generated = False

    def chunks():
        nonlocal generated
        generated = True
        yield make_default_dataset_frame()

    with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "retained_baseline"),
            chunks(),
            spec=make_processed_dataset_spec("retained_baseline"),
            resource_policy=policy,
            retained_payload_bytes=policy.task_memory_budget_bytes,
        )

    assert not generated


def test_writer_final_session_receives_key_index_and_last_external_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_default_dataset_frame()
    external_envelopes = (123, 456)
    observed: list[int] = []
    key_index_bytes: list[int] = []
    real_session = dataset_writing.dataset_session_from_inventory

    def record_session(*args, retained_payload_bytes, **kwargs):  # noqa: ANN002, ANN003
        observed.append(retained_payload_bytes)
        return real_session(
            *args,
            retained_payload_bytes=retained_payload_bytes,
            **kwargs,
        )

    monkeypatch.setattr(
        dataset_writing,
        "dataset_session_from_inventory",
        record_session,
    )
    dataset_writing.write_dataset(
        make_dataset_artifact_root(tmp_path, "final_session_envelope"),
        (
            dataset_writing.DatasetWriteChunk(part.copy(), retained)
            for part, retained in zip(
                (frame.iloc[:1], frame.iloc[1:]),
                external_envelopes,
                strict=True,
            )
        ),
        spec=make_processed_dataset_spec("final_session_envelope"),
        staged_validator=lambda receipt: key_index_bytes.append(
            receipt.exact_primary_key_index.retained_memory_bytes
        ),
        resource_policy=None,
    )

    assert observed == [
        dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
        - dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
        + external_envelopes[-1]
    ]


def test_string_filter_grammar_precedes_capacity_for_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        dataset_reading,
        "_require_filter_preparation_overlap",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("capacity ran before string grammar")
        ),
    )

    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._prepare_in_filter(
            pa.field("value", pa.large_string()),
            ("\ud800",),
            resource_policy=policy,
            retained_filter_bytes=0,
        )


@pytest.mark.parametrize(
    ("field", "values"),
    (
        (pa.field("value", pa.string()), ("BMP-汉字", "emoji-😀", "BMP-汉字")),
        (pa.field("value", pa.large_string()), ("emoji-😀", "emoji-😀")),
    ),
)
def test_string_filter_compact_utf8_sizing_matches_arrow(
    field: pa.Field,
    values: tuple[str, ...],
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    prepared = dataset_reading._prepare_in_filter(
        field,
        values,
        resource_policy=policy,
        retained_filter_bytes=0,
    )
    assert prepared.values.to_pylist() == sorted(set(values))


@pytest.mark.parametrize("enveloped", (False, True))
def test_writer_releases_previous_publication_before_requesting_next(
    tmp_path: Path,
    enveloped: bool,
) -> None:
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first_reference = weakref.ref(first)

    def publications():
        nonlocal first
        current = first
        first = None
        yield (dataset_writing.DatasetWriteChunk(current, 0) if enveloped else current)
        del current
        assert first_reference() is None
        yield second

    name = f"publication_liveness_{enveloped}"
    write_dataset(
        make_dataset_artifact_root(tmp_path, name),
        publications(),
        spec=make_processed_dataset_spec(name),
    )
