from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_is_file,
    filesystem_read_text,
    filesystem_write_text,
)
from tests.storage.dataset_test_support import _input_dataset_identity_payload
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_write_dataset_writes_complete_artifact_root(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    result = write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
    )
    spec_payload = json.loads(
        filesystem_read_text(root / "spec.json", encoding="utf-8")
    )
    assert result.root == root
    assert result.row_count == 3
    assert filesystem_is_file(root / "data" / "year=2014" / "part-000.parquet")
    assert filesystem_is_file(root / "data" / "year=2015" / "part-000.parquet")
    assert manifest["columns"] == ["market_id", "asset_id", "timestamp", "value"]
    assert manifest["schema_id"] == "qrp-dataset-manifest"
    assert manifest["partition_row_counts"] == {"year=2014": 1, "year=2015": 2}
    assert manifest["file_count"] == 2
    assert manifest["row_count"] == 3
    assert spec_payload["data_id"] == "x"
    assert spec_payload["schema_id"] == "qrp-dataset-spec"
    assert spec_payload["kind"] == "factor"


def test_write_dataset_partitions_without_groupby_sort_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def fail_groupby(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("partition_write_must_not_groupby_full_frame")

    monkeypatch.setattr(pd.DataFrame, "groupby", fail_groupby)
    root = make_dataset_artifact_root(tmp_path, "no_groupby_partition")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("no_groupby_partition"),
    )
    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
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


def test_write_dataset_merges_exact_source_artifact_identity(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "source_artifact_identity")
    source_identity = {
        "schema_id": "qrp-binance-recorder",
        "artifact_id": "run-1",
        "manifest_hash": "a" * 64,
    }
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("source_artifact_identity"),
        instance_metadata={
            "input_dataset_identities": {
                "demo_input": _input_dataset_identity_payload()
            },
            "source_artifact_identity": source_identity,
        },
    )
    inspection = inspect_dataset(root)
    assert inspection.manifest["source_artifact_identity"] == source_identity


@pytest.mark.parametrize(
    "source_identity",
    (
        {},
        {
            "schema_id": "qrp-binance-recorder",
            "artifact_id": "run-1",
            "manifest_hash": "bad",
        },
        {
            "schema_id": "qrp-binance-recorder",
            "artifact_id": "run-1",
            "manifest_hash": "a" * 64,
            "extra": "x",
        },
    ),
)
def test_write_dataset_rejects_malformed_source_artifact_identity(
    tmp_path: Path, source_identity: object
) -> None:
    with pytest.raises(ValueError, match="invalid_source_artifact_identity"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "bad_source_artifact_identity"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("bad_source_artifact_identity"),
            instance_metadata={"source_artifact_identity": source_identity},
        )


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
    tmp_path: Path, identities: object
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
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["input_dataset_identities"]["demo_input"]["unknown"] = "x"
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
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
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["input_fingerprints"] = {}
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_manifest_fields_invalid"):
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
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
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
    pd.testing.assert_frame_equal(
        materialized.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        frame.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        check_dtype=False,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_mode: str
) -> None:
    frame = make_default_dataset_frame().iloc[:1].copy()

    class CloseableChunks:
        def __init__(self) -> None:
            self._chunks = iter((frame,))
            self.close_calls = 0

        def __iter__(self):
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

        def fail_write(*_args, **_kwargs):
            raise failure_type("writer_exit")

        monkeypatch.setattr(dataset_writing, "write_parquet_file", fail_write)
    root = make_dataset_artifact_root(tmp_path, f"writer_close_{exit_mode}")
    if exit_mode == "target_exists":
        target_source = source
        dataset_writing.write_dataset(
            root,
            (frame,),
            spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
        )
        with pytest.raises(ValueError, match="dataset_exists"):
            dataset_writing.write_dataset(
                root,
                target_source,
                spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
            )
    elif failure_type is None:
        dataset_writing.write_dataset(
            root, source, spec=make_processed_dataset_spec(f"writer_close_{exit_mode}")
        )
    else:
        with pytest.raises(failure_type, match="writer_exit"):
            dataset_writing.write_dataset(
                root,
                source,
                spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
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

        def __iter__(self):
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

        def __iter__(self):
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

        def fail_write(*_args, **_kwargs):
            raise failure_type("writer_exit")

        monkeypatch.setattr(dataset_writing, "write_parquet_file", fail_write)
    root = make_dataset_artifact_root(tmp_path, f"writer_close_failure_{exit_mode}")
    if exit_mode == "target_exists":
        target_source = facade
        dataset_writing.write_dataset(
            root,
            (frame,),
            spec=make_processed_dataset_spec(f"writer_close_failure_{exit_mode}"),
        )
        input_data = target_source
    else:
        input_data = facade
    with pytest.raises(expected_error, match=expected_message):
        dataset_writing.write_dataset(
            root,
            input_data,
            spec=make_processed_dataset_spec(f"writer_close_failure_{exit_mode}"),
        )
    assert source.close_calls == 1
    assert facade.close_calls == 0
    assert filesystem_exists(root) is (exit_mode == "target_exists")


def test_write_dataset_preserves_schema_for_empty_chunks(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_schema")
    empty_chunk = make_default_dataset_frame().iloc[:0].copy()
    write_dataset(root, [empty_chunk], spec=make_processed_dataset_spec("empty_schema"))
    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
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
            [make_default_dataset_frame().iloc[:1], {"not": "a frame"}],
            spec=make_processed_dataset_spec("invalid"),
        )
    with pytest.raises(ValueError, match="empty_dataset_schema_missing"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "empty_iterable"),
            (),
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
    assert not filesystem_exists(missing_root)


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
    tmp_path: Path, equivalent_asset_ids: pd.Series
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
            root, frame, spec=make_processed_dataset_spec("arrow_key_identity")
        )


@pytest.mark.parametrize("dtype", ("float16", "float32", "float64"))
def test_float_primary_key_zero_identity_publication(tmp_path, dtype):
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["asset_id"] = np.array([-0.0, 0.0], dtype=dtype)
    frame["timestamp"] = pd.Timestamp("2020-01-01")
    root = make_dataset_artifact_root(tmp_path, "float_key")
    spec = make_processed_dataset_spec("float_key")
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(root, frame, spec=spec)
    assert not filesystem_exists(root)
    assert not list(filesystem_glob(root.parent, ".a-*"))
    frame["asset_id"] = np.array([1.0, 2.0], dtype=dtype)
    write_dataset(root, frame, spec=spec)
    restored = read_dataset(root)
    assert restored["asset_id"].dtype == frame["asset_id"].dtype
    assert sorted(restored["asset_id"]) == [1.0, 2.0]
    with pytest.raises(ValueError, match="unsupported_dataset_fingerprint_value"):
        fingerprint_dataset(root)
    comparison = compare_datasets(root, root)
    assert comparison.status == "INSPECTION_FAILED"


@pytest.mark.parametrize(
    "mutation",
    (
        "reverse",
        "duplicate_date",
        "date_format",
        "empty",
        "missing_hash",
        "schema",
        "duplicate_identity",
        "nested",
        "hash",
    ),
)
def test_archive_source_identity_is_validated_on_write_and_read(tmp_path, mutation):
    from copy import deepcopy

    from quant_research_platform.sources.tardis.raw import SCHEMA_ID

    identity = {
        "schema_id": SCHEMA_ID,
        "archives": [
            {
                "date": day,
                "schema_id": SCHEMA_ID,
                "artifact_id": "native_archive_" + token * 24,
                "manifest_hash": token * 64,
            }
            for day, token in (("2026-07-01", "a"), ("2026-08-01", "b"))
        ],
    }
    spec = make_processed_dataset_spec("archive_identity")
    root = make_dataset_artifact_root(tmp_path, spec.data_id)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=spec,
        instance_metadata={"source_artifact_identity": identity},
    )
    assert inspect_dataset(root).manifest["source_artifact_identity"] == identity
    assert len(read_dataset(root)) == 3
    invalid = deepcopy(identity)
    archives = invalid["archives"]
    if mutation == "reverse":
        archives.reverse()
    elif mutation == "duplicate_date":
        archives[1]["date"] = archives[0]["date"]
    elif mutation == "date_format":
        archives[0]["date"] = "20260701"
    elif mutation == "empty":
        archives.clear()
    elif mutation == "missing_hash":
        del archives[0]["manifest_hash"]
    elif mutation == "schema":
        archives[0]["schema_id"] = "other"
    elif mutation == "duplicate_identity":
        archives[1]["artifact_id"] = archives[0]["artifact_id"]
        archives[1]["manifest_hash"] = archives[0]["manifest_hash"]
    elif mutation == "nested":
        archives[0] = identity
    else:
        archives[0]["manifest_hash"] = "bad"
    candidate = make_dataset_artifact_root(tmp_path / "candidate", spec.data_id)
    with pytest.raises(ValueError, match="invalid_source_artifact_identity"):
        write_dataset(
            candidate,
            make_default_dataset_frame(),
            spec=spec,
            instance_metadata={"source_artifact_identity": invalid},
        )
    assert not candidate.exists()
    assert inspect_dataset(root).manifest["source_artifact_identity"] == identity
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["source_artifact_identity"] = invalid
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_source_artifact_identity"):
        read_dataset(root)


def _footer_boundary_frame_and_spec(data_id, *, partitioned):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole

    rows = 600
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["a_share"] * rows, dtype=object),
            "asset_id": pd.Series(
                ["asset_a", "asset_b", "asset_c"] * 200
                if partitioned
                else [f"a{index:06d}" for index in range(rows)],
                dtype=object,
            ),
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="B")
            .repeat(3)
            .astype("datetime64[ns]"),
            **{
                f"factor_{index:02d}": np.arange(rows, dtype=float)
                for index in range(34)
            },
            "note": pd.Series(
                [None] * 220 + [f"value_{index:04d}" for index in range(220, rows)],
                dtype=object,
            ),
            "category": pd.Categorical(
                ["shared", "observed"] * 300,
                categories=["shared", "observed", "unused_" + "汉" * 128],
                ordered=True,
            ),
        }
    )
    frame.loc[rows - 1, "note"] = "汉" * 128
    frame = pd.concat(
        (frame.iloc[220:320], frame.iloc[:220], frame.iloc[320:]), ignore_index=True
    )
    spec = replace(
        make_processed_dataset_spec(data_id),
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            **{f"factor_{index:02d}": ColumnRole.FACTOR for index in range(34)},
            "note": ColumnRole.CONTEXT,
            "category": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp")
        if partitioned
        else ("market_id", "asset_id"),
        partition_columns=("year",) if partitioned else (),
    )
    return (frame, spec)


def _schema_name_frame_and_spec(data_id, *, schema_kind, width):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole

    name = "k" * width
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["a_share"] * 4, dtype=object),
            "asset_id": pd.Series(
                [f"{index:06d}" for index in range(1, 5)], dtype=object
            ),
            "timestamp": pd.Series(
                pd.Timestamp("2024-01-02"), index=range(4), dtype="datetime64[ns]"
            ),
        }
    )
    payload_column = name if schema_kind == "top_level" else "payload"
    frame[payload_column] = (
        pd.Series([1, 2, 3, 4], dtype="int64")
        if schema_kind == "top_level"
        else pd.Series([{name: {"items": [1, 2]}} for _ in range(4)], dtype=object)
    )
    spec = replace(
        make_processed_dataset_spec(data_id),
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            payload_column: ColumnRole.CONTEXT,
        },
    )
    return (frame, spec, name)


def test_public_writer_omits_caller_attrs_before_copy_and_arrow_serialization(
    tmp_path, monkeypatch
):

    class CallerOnly:
        def __deepcopy__(self, _memo):
            raise AssertionError("caller_attrs_were_copied")

    expected = make_default_dataset_frame()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    frames = (expected.iloc[:1].copy(), expected.iloc[1:].copy())
    caller_only = CallerOnly()
    payload = ["x" * 200000, caller_only]
    for frame in frames:
        frame.attrs["caller_only"] = payload
    original_attrs = [frame.attrs for frame in frames]
    root = make_dataset_artifact_root(tmp_path, "caller_attrs")
    write_dataset(
        root,
        frames,
        spec=make_processed_dataset_spec("caller_attrs"),
        instance_metadata={"producer_diagnostics": {"label": "preserved"}},
    )
    primary_key = list(make_processed_dataset_spec("caller_attrs").primary_key)
    pd.testing.assert_frame_equal(
        read_dataset(root).sort_values(primary_key).reset_index(drop=True),
        expected.sort_values(primary_key).reset_index(drop=True),
    )
    for frame, attrs in zip(frames, original_attrs, strict=True):
        assert frame.attrs is attrs
        assert frame.attrs["caller_only"] is payload
    assert payload[0] == "x" * 200000
    assert payload[1] is caller_only
    manifest = json.loads((root / "_manifest" / "manifest.json").read_text())
    assert manifest["producer_diagnostics"] == {"label": "preserved"}


@pytest.mark.parametrize("drift_kind", ("numeric_type", "nested_field"))
def test_public_writer_rejects_nonnull_object_schema_drift(tmp_path, drift_kind):
    from quant_research_platform.contracts.columns import ColumnRole

    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    initial = 1 if drift_kind == "numeric_type" else {"initial": 1}
    changed = 2.0 if drift_kind == "numeric_type" else {"initial": 2, "added": 3}
    first["value"] = pd.Series([initial], index=first.index, dtype=object)
    second["value"] = pd.Series([changed], index=second.index, dtype=object)
    root = make_dataset_artifact_root(tmp_path, "object_numeric_drift")
    spec = make_processed_dataset_spec(
        "object_numeric_drift", value_role=ColumnRole.CONTEXT
    )
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        write_dataset(root, (first, second), spec=spec)
    assert not root.exists()
    assert list(root.parent.iterdir()) == []


@pytest.mark.parametrize("nonnull_first", (False, True))
def test_public_writer_uses_complete_object_string_facts_before_bounded_schema(
    tmp_path, monkeypatch, nonnull_first
):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.contracts.data import ProcessedDataKind

    rows = 6000
    frame = pd.DataFrame(
        {
            "asset_id": pd.Series(
                [f"synthetic_{index:04d}" for index in range(rows)], dtype=object
            ),
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D").astype(
                "datetime64[ns]"
            ),
            **{
                f"factor_{index:02d}": np.arange(rows, dtype=np.float64)
                for index in range(34)
            },
            "note": pd.Series(
                [None] * 5500 + [f"note_{index:04d}" for index in range(5500, rows)],
                dtype=object,
            ),
        }
    )
    expected = frame.copy(deep=True)
    if nonnull_first:
        frame = pd.concat((frame.iloc[5500:], frame.iloc[:5500]), ignore_index=True)
    before = frame.copy(deep=True)
    spec = replace(
        make_processed_dataset_spec(
            "null_prefix_string",
            kind=ProcessedDataKind.PANEL,
            primary_key=("asset_id", "timestamp"),
        ),
        column_roles={
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            **{f"factor_{index:02d}": ColumnRole.FACTOR for index in range(34)},
            "note": ColumnRole.CONTEXT,
        },
    )
    root = make_dataset_artifact_root(
        tmp_path, spec.data_id, kind=ProcessedDataKind.PANEL
    )
    with monkeypatch.context():
        write_dataset(root, frame, spec=spec)
    actual = (
        read_dataset(root).sort_values(list(spec.primary_key)).reset_index(drop=True)
    )
    expected = expected.sort_values(list(spec.primary_key)).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual.drop(columns="note"), expected.drop(columns="note")
    )
    assert actual["note"].dtype == expected["note"].dtype
    pd.testing.assert_series_equal(actual["note"].isna(), expected["note"].isna())
    pd.testing.assert_series_equal(actual["note"].dropna(), expected["note"].dropna())
    assert len(actual) == 6000
    assert actual["note"].isna().sum() == 5500
    pd.testing.assert_frame_equal(frame, before)
