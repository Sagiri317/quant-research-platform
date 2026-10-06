from __future__ import annotations

import errno
import importlib
import importlib.util
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.schema as dataset_schema
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.reading import _prepare_in_filter
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_rename,
    filesystem_replace,
    filesystem_write_bytes,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_writer_rejects_mixed_incompatible_category_values(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "mixed_category")
    frame = make_default_dataset_frame()
    frame["asset_id"] = pd.Categorical([1, "1", 1], categories=[1, "1"])
    with pytest.raises(ValueError, match="unsupported_dataset_logical_dtype"):
        write_dataset(root, frame, spec=make_processed_dataset_spec("mixed_category"))


def test_session_detects_part_manifest_and_same_size_identity_changes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "identity_changes")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("identity_changes"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    manifest_path = root / "_manifest" / "manifest.json"
    filesystem_write_bytes(manifest_path, session.manifest_bytes + b" ")
    with pytest.raises(ValueError, match="dataset_manifest_identity_changed"):
        session.verify()
    filesystem_write_bytes(manifest_path, session.manifest_bytes)
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    part = session.all_paths[0]
    raw = bytearray(filesystem_read_bytes(part))
    raw[len(raw) // 2] ^= 1
    filesystem_write_bytes(part, raw)
    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        session.verify()


def test_session_detects_part_replacement_after_issue(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "part_replacement")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("part_replacement"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    part = session.all_paths[0]
    replacement = tmp_path / "replacement.parquet"
    filesystem_write_bytes(replacement, filesystem_read_bytes(part))
    filesystem_replace(replacement, part)
    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        session.verify()


def test_exact_key_decimal_zero_and_numpy_scalar_identity() -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    decimals = module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.decimal128(18, 4))]),
    )
    assert (
        decimals.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": [Decimal("-0.0000")]}), preserve_index=False
            )
        )
        == 0
    )
    assert (
        decimals.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": [Decimal("0.0000")]}), preserve_index=False
            )
        )
        == 1
    )
    integers = module.ExactPrimaryKeyIndex(
        primary_key=("key",), key_schema=pa.schema([pa.field("key", pa.int64())])
    )
    assert (
        integers.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": [np.int64(7)]}), preserve_index=False
            )
        )
        == 0
    )
    assert (
        integers.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": [7]}), preserve_index=False)
        )
        == 1
    )


def test_staged_validator_wrong_keys_rejects_and_cleans_atomic_state(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "wrong_staged_keys")
    frame = make_default_dataset_frame()

    def reject_wrong_keys(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        wrong = frame.copy()
        wrong["asset_id"] = ["WRONG-A", "WRONG-B", "WRONG-C"]
        receipt.exact_primary_key_index.add_table(
            pa.Table.from_pandas(wrong, preserve_index=False), namespace="expected"
        )
        missing, extra = receipt.exact_primary_key_index.difference_counts(
            expected_namespace="expected", actual_namespace="dataset"
        )
        assert (missing, extra) == (len(frame), len(frame))
        raise ValueError("staged_dataset_primary_key_mismatch")

    with pytest.raises(ValueError, match="staged_dataset_primary_key_mismatch"):
        dataset_writing.write_dataset(
            root,
            (frame,),
            spec=make_processed_dataset_spec("wrong_staged_keys"),
            staged_validator=reject_wrong_keys,
        )
    assert not filesystem_exists(root)
    assert not list(filesystem_glob(root.parent, ".a-*"))


def test_noncanonical_year_partition_is_rejected_with_and_without_window(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "canonical_year")
    write_dataset(
        root,
        make_default_dataset_frame().iloc[:1],
        spec=make_processed_dataset_spec("canonical_year"),
    )
    canonical = next(filesystem_glob(root / "data", "year=*"))
    filesystem_rename(canonical, canonical.with_name("year=02014"))
    for window in (None, TimeWindow("2014-01-01", "2014-12-31")):
        with pytest.raises(ValueError, match="invalid_dataset_part_files"):
            dataset_reading.open_dataset_session(root, columns=None, time_window=window)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, columns: list[object]
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
            (frame,),
            spec=make_processed_dataset_spec("invalid_labels"),
        )


@pytest.mark.parametrize("columns", ([1], ["value", "value"], [""]))
def test_projection_requires_exact_unique_nonempty_string_labels(
    tmp_path: Path, columns: list[object]
) -> None:
    root = make_dataset_artifact_root(tmp_path, "projection_grammar")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("projection_grammar"),
    )
    with pytest.raises(ValueError, match="dataset_columns_invalid"):
        read_dataset(root, columns=columns)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), pd.NaT, True))
def test_physical_filter_rejects_nonfinite_or_schema_incompatible_scalar(
    tmp_path: Path, value: object
) -> None:
    root = make_dataset_artifact_root(tmp_path, "filter_grammar")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("filter_grammar"),
    )
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading.open_dataset_session(
            root, columns=None, time_window=None, row_filters=(("value", "==", value),)
        )
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
    data_type: pa.DataType, value: object
) -> None:
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._validated_physical_filter_value(
            pa.field("value", data_type), value
        )


@pytest.mark.parametrize(
    ("data_type", "source", "members", "expected"),
    (
        (
            pa.timestamp("ns"),
            (datetime(2026, 1, 1), datetime(2026, 1, 2), None),
            (pd.Timestamp("2026-01-02"), datetime(2026, 1, 1)),
            2,
        ),
        (
            pa.duration("ns"),
            (timedelta(days=1), timedelta(days=2), None),
            (pd.Timedelta(days=2), timedelta(days=1)),
            2,
        ),
        (pa.float64(), (0.0, -0.0, 2.0, None), (2.0, -0.0, 0.0, 2.0), 3),
    ),
    ids=("timestamp", "duration", "float-negative-zero"),
)
def test_physical_membership_preserves_generic_exact_scalar_semantics(
    data_type: pa.DataType,
    source: tuple[object, ...],
    members: tuple[object, ...],
    expected: int,
) -> None:
    prepared = _prepare_in_filter(pa.field("value", data_type), members)
    batch = pa.RecordBatch.from_arrays(
        [pa.array(source, type=data_type)], names=["value"]
    )
    filtered = dataset_physical._filter_arrow_batch(batch, [("value", "in", prepared)])
    assert filtered.num_rows == expected
    assert filtered.column(0).null_count == 0


def test_physical_membership_rejects_null_filter_member_before_sort() -> None:
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        _prepare_in_filter(pa.field("value", pa.int64()), (1, None))
        pass


@pytest.mark.parametrize("operator", ("==", ">="))
def test_physical_scalar_filter_preserves_wide_values(
    operator: str,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(128), type=pa.int64())], names=["value"]
    )
    filtered = dataset_physical._filter_arrow_batch(batch, [("value", operator, 1)])
    assert filtered.num_rows == (1 if operator == "==" else 127)


def test_physical_membership_requires_prepared_arrow_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], type=pa.int64())], names=["value"]
    )
    unprepared = pa.array([1, 2], type=pa.int64())

    def reject_array(*_args, **_kwargs):
        raise AssertionError("physical filtering must not prepare retained values")

    monkeypatch.setattr(dataset_physical.pa, "array", reject_array)
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_physical._filter_arrow_batch(batch, [("value", "in", (1, 2))])
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_physical._filter_arrow_batch(batch, [("value", "in", unprepared)])


@pytest.mark.parametrize(
    ("operator", "value", "error"),
    (
        ("in", (1, 2), "invalid_dataset_physical_filter_value"),
        ("contains", 1, "unsupported_dataset_physical_filter"),
    ),
)
def test_physical_filter_rejects_invalid_operator_and_values(
    operator: str, value: object, error: str
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], type=pa.int64())], names=["value"]
    )
    with pytest.raises(ValueError, match=error):
        dataset_physical._filter_arrow_batch(batch, [("value", operator, value)])


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
    tmp_path: Path, mutation: dict[str, object], error: str
) -> None:
    root = make_dataset_artifact_root(tmp_path, "manifest_truth")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("manifest_truth"),
    )
    spec = json.loads(filesystem_read_text(root / "spec.json", encoding="utf-8"))
    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
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
            forged_manifest, spec=inspection.spec
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
    filesystem_write_bytes(
        root / "_manifest" / "manifest.json", b" " * (1024 * 1024 + 1)
    )
    with pytest.raises(ValueError, match="dataset_metadata_file_too_large"):
        dataset_reading.open_dataset_session(root, columns=None, time_window=None)
        pass


def test_dataset_native_disk_failure_removes_partial_candidate_and_preserves_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = make_dataset_artifact_root(tmp_path, "published")
    write_dataset(
        published,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("published"),
    )
    original = {
        path: filesystem_read_bytes(path)
        for path in filesystem_glob(published, "**/*")
        if path.is_file()
    }
    candidate = make_dataset_artifact_root(tmp_path, "candidate")
    primary = OSError(errno.ENOSPC, "synthetic_native_disk_full")
    partial_paths = []

    def fail_native_write(_table, path, **_kwargs):
        partial = Path(path)
        partial.write_bytes(b"partial parquet")
        partial_paths.append(partial)
        raise primary

    monkeypatch.setattr(dataset_writing.pq, "write_table", fail_native_write)
    with pytest.raises(OSError) as caught:
        write_dataset(
            candidate,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("candidate"),
        )
    assert caught.value is primary
    assert partial_paths and all(not path.exists() for path in partial_paths)
    assert not filesystem_exists(candidate)
    assert tuple(candidate.parent.glob(".a-*")) == ()
    assert {path: filesystem_read_bytes(path) for path in original} == original


@pytest.mark.parametrize("variant", ("normal", "empty", "override"))
def test_writer_persisted_readback_preserves_schema_and_rows(
    tmp_path: Path, variant: str
) -> None:
    frame = make_default_dataset_frame()
    if variant == "empty":
        frame = frame.iloc[:0].copy()
    elif variant == "override":
        frame["value"] = pd.Series(frame["value"], dtype="Int64")
    root = make_dataset_artifact_root(tmp_path, f"writer_readback_{variant}")
    dataset_writing.write_dataset(
        root, (frame,), spec=make_processed_dataset_spec(f"writer_readback_{variant}")
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
    pd.testing.assert_frame_equal(
        persisted.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        frame.astype({"timestamp": "datetime64[ns]"})
        .sort_values(["market_id", "asset_id", "timestamp"])
        .reset_index(drop=True),
        check_exact=True,
    )
