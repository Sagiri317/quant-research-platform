from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import weakref
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from inspect import currentframe, getsource, signature
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
import quant_research_platform.storage._dataset_paths as dataset_paths
import quant_research_platform.storage._dataset_physical as dataset_physical
import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_schema as dataset_schema
import quant_research_platform.storage._dataset_writing as dataset_writing
import quant_research_platform.storage.atomic as storage_atomic
import quant_research_platform.storage.datasets as datasets
import quant_research_platform.storage.manifests as storage_manifests
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


def test_writer_releases_previous_chunk_locals_before_requesting_next_source(
    tmp_path: Path,
) -> None:
    name = "writer_chunk_lifecycle"
    frame = make_default_dataset_frame().reset_index(drop=True)
    first = frame.iloc[:1].copy()
    second = frame.iloc[1:].copy()
    wrapper_locals: list[set[str]] = []
    writer_locals: list[set[str]] = []

    def chunks():  # noqa: ANN202
        yield first
        wrapper = currentframe().f_back
        assert wrapper is not None
        writer = wrapper.f_back
        assert writer is not None
        wrapper_locals.append({"chunk"}.intersection(wrapper.f_locals))
        writer_locals.append(
            {
                "chunk",
                "declared",
                "bounded_frame",
                "prepared_frame",
                "partition",
                "partition_frame",
                "bounded_partition",
                "table",
                "persisted_table",
                "output_path",
            }.intersection(writer.f_locals)
        )
        yield second

    root = make_dataset_artifact_root(tmp_path, name)
    write_dataset(root, chunks(), spec=make_processed_dataset_spec(name))

    assert wrapper_locals == [set()]
    assert writer_locals == [set()]
    expected = frame.copy()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(
        read_dataset(root)
        .sort_values(list(make_processed_dataset_spec().primary_key))
        .reset_index(drop=True),
        expected.sort_values(
            list(make_processed_dataset_spec().primary_key)
        ).reset_index(drop=True),
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


def test_converter_enforces_actual_arrow_and_pandas_operation_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.record_batch([pa.array([1])], names=["value"])
    monkeypatch.setattr(
        dataset_reading,
        "pandas_frame_payload_bytes",
        lambda _frame: 512,
    )

    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                target_chunk_bytes=4_096,
                operation_headroom_bytes=batch.nbytes + 511,
            )
        )


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
    operation_headroom_bytes = (
        batch.get_total_buffer_size() + sum(output_bytes) + headroom_delta
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


def test_categorical_converter_releases_temporary_codes_before_yield() -> None:
    batch = pa.record_batch(
        [pa.array(["known"] * 10_000)],
        names=["value"],
    )
    converted = dataset_reading._arrow_batch_to_bounded_pandas(
        batch,
        target_chunk_bytes=1024**2,
        operation_headroom_bytes=4 * 1024**2,
        dtype_overrides={
            "value": pd.CategoricalDtype(categories=["known"]),
        },
    )

    next(converted)
    assert converted.gi_frame is not None
    assert "codes" not in converted.gi_frame.f_locals
    converted.close()


@pytest.mark.parametrize(
    ("headroom_delta", "should_fail"),
    ((0, False), (-1, True)),
)
def test_declared_member_accounts_for_previous_output_across_parent_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headroom_delta: int,
    should_fail: bool,
) -> None:
    path = tmp_path / "part.parquet"
    batches = (
        pa.record_batch([pa.array(["x" * 4_096])], names=["value"]),
        pa.record_batch([pa.array(["y" * 4_096])], names=["value"]),
    )
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches)
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    target_chunk_bytes = max(output_bytes)
    exact_headroom = max(
        batch.get_total_buffer_size()
        + output_bytes[index]
        + (output_bytes[index - 1] if index else 0)
        for index, batch in enumerate(batches)
    )
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter(batches),
    )

    converted = dataset_reading.iter_declared_parquet_member_batches(
        path,
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=exact_headroom + headroom_delta,
        expected_identity=SimpleNamespace(),
    )
    if should_fail:
        with pytest.raises(
            ValueError,
            match="dataset_working_set_exceeds_task_memory_budget",
        ):
            tuple(converted)
        return

    output = tuple(converted)
    for observed, expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)


@pytest.mark.parametrize(
    ("headroom_delta", "should_fail"),
    ((0, False), (-1, True)),
)
def test_dataset_file_batches_account_for_previous_output_across_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headroom_delta: int,
    should_fail: bool,
) -> None:
    paths = (tmp_path / "part-0.parquet", tmp_path / "part-1.parquet")
    batches = {
        paths[0]: pa.record_batch([pa.array([1], type=pa.int64())], names=["value"]),
        paths[1]: pa.record_batch([pa.array([2], type=pa.int64())], names=["value"]),
    }
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches.values())
    output_bytes = tuple(pandas_frame_payload_bytes(frame) for frame in expected)
    target_chunk_bytes = max(output_bytes)
    exact_headroom = max(
        batches[path].get_total_buffer_size()
        + output_bytes[index]
        + (output_bytes[index - 1] if index else 0)
        for index, path in enumerate(paths)
    )
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda path, **_kwargs: iter((batches[Path(path)],)),
    )
    context = SimpleNamespace(
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
        operation_headroom_bytes=exact_headroom + headroom_delta,
    )
    if should_fail:
        with pytest.raises(
            ValueError,
            match="dataset_working_set_exceeds_task_memory_budget",
        ):
            tuple(converted)
        return

    output = tuple(converted)
    assert tuple(path for path, _ in output) == paths
    for (_, observed), expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)


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


@pytest.mark.parametrize(
    ("values", "target_chunk_bytes"),
    (
        (tuple("x" * 32 for _ in range(512)), 16 * 1024),
        (tuple([*("x" * 32 for _ in range(511)), "y" * 8_000]), 16 * 1024),
    ),
    ids=("uniform", "skewed"),
)
def test_arrow_batch_converter_calibrates_without_duplicate_observer_effects(
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


@pytest.mark.parametrize(
    ("oversized_stage", "retry_conversion_count"),
    (("raw", 3), ("override", 4)),
)
def test_arrow_batch_converter_releases_rejected_frames_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    oversized_stage: str,
    retry_conversion_count: int,
) -> None:
    row_count = 512
    batch = pa.record_batch(
        [pa.array(range(row_count)), pa.array(["x"] * row_count)],
        names=["row_id", "value"],
    )
    raw_refs: list[weakref.ReferenceType[pd.DataFrame]] = []
    output_refs: list[weakref.ReferenceType[pd.DataFrame]] = []
    raw_alive_before_conversion: list[int] = []
    output_alive_before_conversion: list[int] = []
    original_conversion = dataset_reading._arrow_slice_to_pandas
    original_astype = pd.DataFrame.astype

    def record_conversion(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
        raw_alive_before_conversion.append(
            sum(reference() is not None for reference in raw_refs)
        )
        output_alive_before_conversion.append(
            sum(reference() is not None for reference in output_refs)
        )
        frame = original_conversion(arrow_slice)
        raw_refs.append(weakref.ref(frame))
        return frame

    def record_astype(
        frame: pd.DataFrame,
        *args: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        output = original_astype(frame, *args, **kwargs)
        output_refs.append(weakref.ref(output))
        return output

    def staged_payload_bytes(frame: pd.DataFrame) -> int:
        per_row = 100
        if len(frame) > 1 and (
            oversized_stage == "raw"
            or any(reference() is frame for reference in output_refs)
        ):
            per_row = 200
        return len(frame) * per_row

    monkeypatch.setattr(
        dataset_reading,
        "_arrow_slice_to_pandas",
        record_conversion,
    )
    monkeypatch.setattr(
        dataset_reading,
        "pandas_frame_payload_bytes",
        staged_payload_bytes,
    )
    if oversized_stage == "override":
        monkeypatch.setattr(pd.DataFrame, "astype", record_astype)

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=10_000,
            dtype_overrides={"value": "string"}
            if oversized_stage == "override"
            else None,
        )
    )

    assert (
        raw_alive_before_conversion[:retry_conversion_count]
        == [0] * retry_conversion_count
    )
    if oversized_stage == "override":
        assert (
            output_alive_before_conversion[:retry_conversion_count]
            == [0] * retry_conversion_count
        )
    assert len(raw_alive_before_conversion) < row_count // 10
    assert len(chunks) < row_count // 10
    assert sum(len(frame) for frame in chunks) == row_count


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
    with pytest.raises(RuntimeError, match="injected_read_failure"):
        list(
            dataset_reading.iter_dataset_chunks(
                session,
                target_chunk_bytes=1024**2,
            )
        )
    assert calls > before_error
    session.close()


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
                logical_columns=tuple(sample.columns),
                logical_dtypes=logical_dtype_payload_from_frame(sample),
                inventory=SimpleNamespace(retained_memory_bytes=1024),
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
        logical_columns=tuple(expected.columns),
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
    )

    observed = dataset_reading.materialize_dataset_session(
        context,
        target_chunk_bytes=budget,
        materialized_budget_bytes=budget,
    )

    pd.testing.assert_frame_equal(observed, expected)


def test_materialized_session_uses_two_pass_exact_buffers_without_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame(
        {
            "numpy_integer": pd.Series([1, 2], dtype="int64"),
            "numpy_boolean": pd.Series([True, False], dtype="bool"),
            "numpy_float": pd.Series([1.5, 2.5], dtype="float64"),
            "nullable_float32": pd.Series([1.5, pd.NA], dtype="Float32"),
            "nullable_float64": pd.Series([pd.NA, 2.5], dtype="Float64"),
            "nullable_integer": pd.Series([1, pd.NA], dtype="Int64"),
            "nullable_boolean": pd.Series([True, pd.NA], dtype="boolean"),
            "string": pd.Series(["é", pd.NA], dtype="string"),
            "categorical": pd.Categorical(
                ["a", "b"], categories=["a", "b"], ordered=True
            ),
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "datetime_tz": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "timedelta": pd.to_timedelta([1, 2], unit="s"),
            "object": pd.Series([{"value": 1}, None], dtype="object"),
        }
    )
    calls: list[object] = []
    allocated: dict[str, dataset_reading._MaterializedColumnBuffer] = {}
    context = SimpleNamespace(
        logical_columns=tuple(expected.columns),
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        verify=lambda: None,
    )

    def chunks(observed_context, **_kwargs):  # noqa: ANN001
        calls.append(observed_context)
        yield expected.iloc[:1].copy()
        yield expected.iloc[1:].copy()

    allocate = dataset_reading._allocate_materialized_buffers

    def record_buffers(*args, **kwargs):  # noqa: ANN002, ANN003
        result = allocate(*args, **kwargs)
        allocated.update(result)
        return result

    monkeypatch.setattr(dataset_reading, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(
        dataset_reading,
        "_allocate_materialized_buffers",
        record_buffers,
    )
    monkeypatch.setattr(
        dataset_reading.pd,
        "concat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("materialized_concat_must_not_exist")
        ),
    )

    observed = dataset_reading.materialize_dataset_session(
        context,
        target_chunk_bytes=1024 * 1024,
        materialized_budget_bytes=4 * 1024 * 1024,
    )

    assert calls == [context, context]
    pd.testing.assert_frame_equal(observed, expected)
    assert "pd.concat(" not in getsource(dataset_reading)
    for private_surface in (
        "array._pa_array",
        "array._data",
        "array._mask",
        "DatetimeArray._simple_new",
    ):
        assert private_surface not in getsource(dataset_reading)
    assert isinstance(allocated["numpy_integer"].values, np.ndarray)
    allocated["numpy_integer"].values[0] = 7
    assert observed.loc[0, "numpy_integer"] == 7
    assert isinstance(allocated["nullable_integer"].values, np.ndarray)
    allocated["nullable_integer"].values[0] = 8
    assert observed.loc[0, "nullable_integer"] == 8
    assert isinstance(allocated["nullable_float32"].values, np.ndarray)
    allocated["nullable_float32"].values[0] = 3.5
    assert observed.loc[0, "nullable_float32"] == 3.5
    assert isinstance(allocated["string"].values, bytearray)
    allocated["string"].values[:2] = "ø".encode()
    assert observed.loc[0, "string"] == "ø"

    empty = pd.DataFrame(
        {
            "nullable_float32": pd.Series(dtype="Float32"),
            "nullable_float64": pd.Series(dtype="Float64"),
        }
    )
    empty_context = SimpleNamespace(
        logical_columns=tuple(empty.columns),
        logical_dtypes=logical_dtype_payload_from_frame(empty),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        verify=lambda: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((empty,)),
    )
    empty_observed = dataset_reading.materialize_dataset_session(
        empty_context,
        target_chunk_bytes=1024 * 1024,
        materialized_budget_bytes=4 * 1024 * 1024,
    )
    pd.testing.assert_frame_equal(empty_observed, empty)
    assert tuple(map(str, empty_observed.dtypes)) == ("Float32", "Float64")


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
        logical_columns=tuple(expected.columns),
        logical_dtypes=logical_dtype_payload_from_frame(expected),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
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
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frame),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
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
    assert observed_headroom == [3072, 2048]


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
            SimpleNamespace(inventory=SimpleNamespace(retained_memory_bytes=1024)),
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
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frame),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
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
        temp_store = int(index._connection.execute("PRAGMA temp_store").fetchone()[0])
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
        assert timestamps.add_frame(pd.DataFrame({"key": [datetime(2026, 1, 1)]})) == 1

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
    stale_stage = root.with_name(f"{scratch_prefix}qzzzzzz-stale")
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
    key_source = (
        Path(dataset_reading.__file__)
        .with_name("_dataset_key_tracking.py")
        .read_text(encoding="utf-8")
    )
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
        assert index.retained_memory_bytes == 12 * 1024

    assert cache_size == -12
    assert (
        signature(module.ExactPrimaryKeyIndex).parameters["sqlite_cache_bytes"].default
        == module.EXACT_KEY_SQLITE_PAGE_BYTES
    )


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


def test_hashable_fingerprint_frame_reuses_ordinary_frame_storage() -> None:
    frame = pd.DataFrame({"value": np.arange(20_000, dtype="int64")})

    observed = dataset_fingerprinting._hashable_fingerprint_frame(
        frame,
        target_bytes=pandas_frame_payload_bytes(frame),
    )

    assert observed is frame


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
    aggregate_bytes = (
        arrow_batch.get_total_buffer_size()
        + pandas_frame_payload_bytes(frame)
        + pandas_frame_payload_bytes(canonical)
    )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "8",
            "QRP_PAYLOAD_TARGET_MB": "4",
        },
    )
    operation_headroom = aggregate_bytes + headroom_delta
    retained_memory_bytes = policy.task_memory_budget_bytes - operation_headroom
    assert retained_memory_bytes > 0
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
        spec={"primary_key": ["nested"]},
        logical_columns=("nested",),
        time_window=None,
        data_root=tmp_path / "data",
        schema_hash="schema",
        inventory=SimpleNamespace(retained_memory_bytes=retained_memory_bytes),
    )

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

    assert observed_headroom == [operation_headroom]
    assert observer_calls == 1


def test_fingerprint_releases_hash_array_before_requesting_next_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": np.arange(100_000, dtype="int64")})
    arrow_batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    paths = (
        tmp_path / "data" / "part-0.parquet",
        tmp_path / "data" / "part-1.parquet",
    )
    hashes_live_at_next_batch: list[bool] = []

    def batches(*_args: object, **kwargs: object):
        observer = kwargs["batch_observer"]
        observer(arrow_batch, frame)
        yield paths[0], frame
        caller = currentframe().f_back
        assert caller is not None
        hashes_live_at_next_batch.append("hashes" in caller.f_locals)
        observer(arrow_batch, frame)
        yield paths[1], frame

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        batches,
    )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
    )
    context = SimpleNamespace(
        spec={"primary_key": ["value"]},
        logical_columns=("value",),
        time_window=None,
        data_root=tmp_path / "data",
        schema_hash="schema",
        inventory=SimpleNamespace(retained_memory_bytes=0),
    )

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context,
        resource_policy=policy,
    )

    assert fingerprint.row_count == 2 * len(frame)
    assert hashes_live_at_next_batch == [False]


def test_fingerprint_accounts_for_multislice_parent_arrow_buffers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"nested": [["x" * 50] * 4 for _ in range(20_000)]})
    arrow_batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=32 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "16",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
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
            target_chunk_bytes=policy.payload_target_bytes,
            batch_observer=measure_batch,
        )
    )
    assert len(measured) > 1
    assert {item[0] for item in measured} == {arrow_batch.get_total_buffer_size()}
    exact_headroom = max(sum(item) for item in measured)
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
        return SimpleNamespace(
            spec={"primary_key": ["nested"]},
            logical_columns=("nested",),
            time_window=None,
            data_root=tmp_path / "data",
            schema_hash="schema",
            inventory=SimpleNamespace(
                retained_memory_bytes=(policy.task_memory_budget_bytes - headroom_bytes)
            ),
        )

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context_for(exact_headroom),
        resource_policy=policy,
    )
    assert fingerprint.row_count == len(frame)
    assert len(observed_slices) > 1
    assert set(observed_slices) == {arrow_batch.get_total_buffer_size()}

    with pytest.raises(
        ValueError,
        match="dataset_fingerprint_value_exceeds_target_bytes",
    ):
        dataset_fingerprinting.fingerprint_dataset_session(
            context_for(exact_headroom - 1),
            resource_policy=policy,
        )


def test_fingerprint_rejects_exhausted_operation_headroom_before_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "8"},
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
        inventory=SimpleNamespace(
            retained_memory_bytes=policy.task_memory_budget_bytes
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


def test_bounded_frame_chunks_reuse_bounded_frame_storage() -> None:
    frame = pd.DataFrame({"value": np.arange(20_000, dtype="int64")})

    chunks = tuple(
        dataset_reading._bounded_frame_chunks(
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
    fingerprint_operation_headrooms: list[int] = []
    fingerprint_arrow_bytes: list[int] = []
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
        fingerprint_operation_headrooms.append(kwargs["operation_headroom_bytes"])
        observer = kwargs["batch_observer"]

        def record_arrow_bytes(batch: pa.RecordBatch, observed: pd.DataFrame) -> None:
            fingerprint_arrow_bytes.append(batch.get_total_buffer_size())
            observer(batch, observed)

        kwargs["batch_observer"] = record_arrow_bytes
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

    assert writer_targets and all(
        0 < target <= policy.payload_target_bytes for target in writer_targets
    )
    assert read_targets == [policy.payload_target_bytes] * 3
    assert inspection_targets and set(inspection_targets) == {
        policy.payload_target_bytes
    }
    assert fingerprint_physical_targets and set(fingerprint_physical_targets) == {
        policy.payload_target_bytes
    }
    assert fingerprint_hash_targets
    assert len(fingerprint_operation_headrooms) == 1
    assert fingerprint_hash_targets == [
        fingerprint_operation_headrooms[0] - arrow_bytes
        for arrow_bytes in fingerprint_arrow_bytes
    ]


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
        resource_policy=SimpleNamespace(task_memory_budget_bytes=4096),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
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
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
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

    comparison = dataset_comparison.compare_datasets(
        left,
        right,
        resource_policy=policy,
    )

    assert comparison.status == "MATCH"
    assert observed == [policy, policy]
    assert maximum_active_contexts == 1
    assert active_contexts == 0


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


def test_open_dataset_session_owns_and_closes_inventory(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_inventory_owner")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_inventory_owner"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    )
    scratch = session.inventory.scratch_path

    session.close()

    assert not scratch.exists()


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
            available_memory_bytes=16 * 1024**2,
            environ={
                "QRP_TASK_MEMORY_BUDGET_MB": "8",
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
    categories = [f"category-{index:05d}-" + "x" * 512 for index in range(5_000)]
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
            dataset_reading.open_dataset_session(
                root,
                columns=None,
                time_window=window,
            )


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
        dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=(("value", "==", value),),
        )


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
        dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        )


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
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frames[0]),
        inventory=SimpleNamespace(retained_memory_bytes=1024),
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
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=4 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )
    monkeypatch.setattr(
        dataset_writing,
        "pandas_frame_payload_bytes",
        lambda _: 300_000,
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


def test_writer_enforces_task_wide_copy_and_arrow_headroom(tmp_path: Path) -> None:
    row_count = 400
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": [f"asset-{index:04d}" for index in range(row_count)],
            "timestamp": pd.date_range("2014-01-02", periods=row_count),
            "value": ["x" * 1_500] * row_count,
        }
    )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=4 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )
    with pytest.raises(
        ValueError,
        match="dataset_working_set_exceeds_task_memory_budget",
    ):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "writer_task_headroom"),
            frame,
            spec=make_processed_dataset_spec("writer_task_headroom"),
            resource_policy=policy,
        )


@pytest.mark.parametrize(
    "variant",
    ("normal", "empty", "override"),
)
def test_writer_passes_payload_target_and_actual_headroom_to_persisted_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    frame = make_default_dataset_frame()
    if variant == "empty":
        frame = frame.iloc[:0].copy()
    elif variant == "override":
        frame["value"] = pd.Series(frame["value"], dtype="Int64")
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=4 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )
    observed_targets: list[int] = []
    observed_operation_headroom: list[int] = []
    observed_headroom: list[int] = []
    original = dataset_writing.iter_declared_parquet_member_batches
    require_headroom = dataset_writing._require_writer_headroom

    def record_target(  # noqa: ANN002, ANN003
        *args,
        target_chunk_bytes: int,
        operation_headroom_bytes: int,
        **kwargs,
    ):
        observed_targets.append(target_chunk_bytes)
        observed_operation_headroom.append(operation_headroom_bytes)
        yield from original(
            *args,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            **kwargs,
        )

    monkeypatch.setattr(
        dataset_writing,
        "iter_declared_parquet_member_batches",
        record_target,
    )

    def record_headroom(*args, **kwargs):  # noqa: ANN002, ANN003
        result = require_headroom(*args, **kwargs)
        if "arrow_bytes" in kwargs:
            observed_headroom.append(result)
        return result

    monkeypatch.setattr(
        dataset_writing,
        "_require_writer_headroom",
        record_headroom,
    )
    root = make_dataset_artifact_root(tmp_path, f"writer_readback_{variant}")

    dataset_writing.write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(f"writer_readback_{variant}"),
        resource_policy=policy,
    )

    assert observed_targets
    assert observed_headroom
    assert observed_targets[-1] == policy.payload_target_bytes
    assert observed_operation_headroom[-1] == observed_headroom[-1]
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
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
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


def test_writer_and_session_do_not_retain_full_inventory_containers() -> None:
    assert "_observed_parts" not in dataset_writing._WriteStats.__dataclass_fields__
    fields = dataset_reading.DatasetSession.__dataclass_fields__
    assert "part_metadata" not in fields
    assert "all_paths" not in fields
    assert "read_paths" not in fields


def test_chunked_session_inventory_is_removed_on_early_close(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_early_close")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inventory_early_close"),
    )
    chunks = read_dataset(root, mode=DatasetReadMode.CHUNKED)
    next(chunks)
    assert list(root.parent.glob(".dataset-inventory-*"))
    chunks.close()  # type: ignore[attr-defined]
    assert not list(root.parent.glob(".dataset-inventory-*"))


def test_dataset_inventory_cleanup_removes_sqlite_sidecars(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    inventory = dataset_inventory.DatasetSessionInventory(
        data_root=data_root,
        scratch_parent=tmp_path,
        task_memory_budget_bytes=1024,
    )
    scratch = inventory.scratch_path
    assert (scratch / "inventory.sqlite3").is_file()
    assert inventory.retained_memory_bytes == 1024
    assert inventory._connection is not None
    assert inventory._connection.execute("PRAGMA cache_size").fetchone() == (-1,)
    inventory.close()
    assert not scratch.exists()
    assert not list(tmp_path.glob(".dataset-inventory-*"))
    assert not list(tmp_path.rglob("inventory.sqlite3-wal"))
    assert not list(tmp_path.rglob("inventory.sqlite3-shm"))


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
    assert not list(tmp_path.glob(".dataset-inventory-*"))

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
        dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        )
    assert not list(root.parent.glob(".dataset-inventory-*"))


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
        assert [path.name for path in session.all_paths] == [
            f"part-{part_index}.parquet" for part_index in part_indices
        ]


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
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "32"},
    )
    root = make_dataset_artifact_root(tmp_path, "inventory_disk_gate")
    dataset_writing.write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inventory_disk_gate"),
        resource_policy=policy,
    )

    assert reserve_calls[0] == 1024
    assert len(reserve_calls) > 1
    assert policy.task_memory_budget_bytes not in reserve_calls
    assert all(value > 0 for value in reserve_calls)


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
        dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        )

    assert reserve_calls == 2
    assert not list(root.parent.glob(".dataset-inventory-*"))


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
    assert not list(tmp_path.glob(".dataset-inventory-*"))


def test_dataset_inventory_helpers_are_owned_in_their_only_caller_module() -> None:
    paths_source = getsource(dataset_paths)
    inventory_source = getsource(dataset_inventory)

    for symbol in (
        "_PART_FILE_PATTERN",
        "_path_or_relative_component_is_alias",
        "_physical_file_identity",
    ):
        assert symbol not in paths_source
        assert symbol in inventory_source


def test_dataset_inventory_row_disk_reserve_is_exact_without_payload_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_calls: list[int] = []
    monkeypatch.setattr(
        dataset_inventory,
        "require_free_disk_headroom",
        lambda _path, *, reserve_bytes: reserve_calls.append(reserve_bytes),
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    with dataset_inventory.DatasetSessionInventory(
        data_root=data_root,
        scratch_parent=tmp_path,
        task_memory_budget_bytes=1024,
    ) as inventory:
        inventory._gate_row_write(("é", b"12", -12, None))

    assert "bytearray" not in getsource(
        dataset_inventory.DatasetSessionInventory._gate_row_write
    )
    assert reserve_calls[-1] == 43


def test_file_count_tracking_is_scalar_and_counts_consecutive_file_batches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part-000.parquet"
    empty_path = tmp_path / "part-001.parquet"
    frame = pd.DataFrame({"value": [1]})
    batches = ((path, frame), (path, frame), (empty_path, frame.iloc[:0]))

    def observed_batches(*_args: object, **kwargs: object):
        observer = kwargs["batch_observer"]
        for observed_path, observed_frame in batches:
            observer(
                pa.RecordBatch.from_pandas(observed_frame, preserve_index=False),
                observed_frame,
            )
            yield observed_path, observed_frame

    monkeypatch.setattr(
        dataset_fingerprinting,
        "iter_dataset_file_batches",
        observed_batches,
    )
    context = SimpleNamespace(
        spec={"primary_key": ["value"]},
        logical_columns=("value",),
        time_window=None,
        data_root=tmp_path,
        schema_hash="schema",
        inventory=SimpleNamespace(retained_memory_bytes=0),
    )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
    )

    fingerprint = dataset_fingerprinting.fingerprint_dataset_session(
        context,
        resource_policy=policy,
    )

    assert fingerprint.file_count == 1
    assert fingerprint.row_count == 2
    assert "observed_paths" not in getsource(dataset_fingerprinting)
    assert "observed_paths" not in getsource(dataset_inspection)


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
        all_paths=(),
        manifest={},
        inventory=SimpleNamespace(retained_memory_bytes=1024),
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=16 * 1024**2,
            environ={"QRP_PAYLOAD_TARGET_MB": "4"},
        ),
    )

    runtime = dataset_inspection._inspection_runtime_metadata(
        context,
        target_chunk_bytes=1024,
    )

    assert runtime["file_count"] == 1
    assert runtime["row_count"] == 2
    assert observed_headroom == [
        context.resource_policy.task_memory_budget_bytes - 2048
    ]


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
