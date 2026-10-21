from __future__ import annotations

import importlib
import math
import struct
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    native_filesystem_path,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


@pytest.mark.parametrize("selected", (None, [True, True, False]))
def test_float_lookup_partitions_keys_without_consuming_values(selected):
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index = module.ExactPrimaryKeyIndex(primary_key=("key",))
    source = pa.table({"key": ["a", "b", "c"], "value": [-0.0, 1.25, 2.5]})
    assert (
        index.add_float_value_table(
            source,
            namespace="source",
            value_column="value",
            selected=None if selected is None else pa.array(selected),
        )
        == 0
    )
    payloads = tuple(index._namespaces["source"].values())
    expected_count = 3 if selected is None else 2
    query = pa.table({"key": ["a", "a", "absent"]})
    for mark, remaining in (
        (False, expected_count),
        (True, expected_count - 1),
        (False, expected_count - 1),
        (True, expected_count - 1),
    ):
        values, found = index.lookup_float_values(
            query,
            source_namespace="source",
            matched_namespace="matched",
            mark_matched=mark,
        )
        assert list(found) == [1, 1, 0]
        assert struct.pack(">d", values[0]) == struct.pack(">d", -0.0)
        assert math.isnan(values[2])
        assert index.float_value_count(namespace="source") == remaining
    for mark in (True, False, True):
        values, found = index.lookup_float_values(
            source,
            source_namespace="source",
            matched_namespace="matched",
            mark_matched=mark,
        )
        assert list(found) == [1, 1, int(selected is None)]
        assert values[1] == 1.25
        assert index.float_value_count(namespace="source") == 0
    unmatched, matched = index._namespaces["source"], index._namespaces["matched"]
    assert not unmatched.keys() & matched.keys()
    assert len(unmatched) + len(matched) == 3
    assert all(
        any(value is original for original in payloads) for value in matched.values()
    )
    assert all(value is None for value in unmatched.values())
    if selected is None:
        assert unmatched.__sizeof__() == {}.__sizeof__()


@pytest.mark.parametrize("invalid", ("key", "payload"))
def test_float_lookup_bad_late_row_keeps_match_batch_atomic(invalid):
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index = module.ExactPrimaryKeyIndex(primary_key=("key",))
    index.add_float_value_table(
        pa.table({"key": [[1.0], [2.0]], "value": [2.0, 3.0]}),
        namespace="source",
        value_column="value",
    )
    if invalid == "payload":
        source = index._namespaces["source"]
        source[next(reversed(source))] = b"invalid"
    error = (
        "invalid_dataset_primary_key_value"
        if invalid == "key"
        else "invalid_exact_float_value"
    )
    with pytest.raises(ValueError, match=f"^{error}$"):
        index.lookup_float_values(
            pa.table({"key": [[1.0], [float("inf") if invalid == "key" else 2.0]]}),
            source_namespace="source",
            matched_namespace="matched",
            mark_matched=True,
        )
    assert index.float_value_count(namespace="source") == 2
    assert index.identity_payload_count(namespace="matched") == 0


@pytest.mark.parametrize(
    "source,matched", (("", "matched"), ("source", ""), ("same", "same"))
)
def test_float_lookup_requires_valid_distinct_namespaces(source, matched):
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index = module.ExactPrimaryKeyIndex(primary_key=("key",))
    with pytest.raises(ValueError, match="^invalid_exact_key_namespace$"):
        index.lookup_float_values(
            pa.table({"key": ["a"]}),
            source_namespace=source,
            matched_namespace=matched,
            mark_matched=False,
        )


def test_writer_rejects_inconsistent_ordered_categorical_domains(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "ordered_categorical_parts")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["asset_id"] = pd.Categorical(
        first["asset_id"], categories=[first["asset_id"].iloc[0]], ordered=True
    )
    second["asset_id"] = pd.Categorical(
        second["asset_id"], categories=[second["asset_id"].iloc[0]], ordered=True
    )
    with pytest.raises(ValueError, match="dataset_ordered_categorical_mismatch"):
        write_dataset(
            root,
            (first, second),
            spec=make_processed_dataset_spec("ordered_categorical_parts"),
        )
    assert not filesystem_exists(root)


def test_exact_primary_key_index_retains_exact_native_semantics():
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index = module.ExactPrimaryKeyIndex(primary_key=("asset_id", "timestamp"))
    first = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "asset_id": ["a", "b"],
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            }
        ),
        preserve_index=False,
    )
    same = pa.Table.from_pandas(
        pd.DataFrame(
            {"asset_id": ["a"], "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]}
        ),
        preserve_index=False,
    )
    assert index.add_table(first) == 0
    assert index.add_table(same) == 1
    numeric = module.ExactPrimaryKeyIndex(primary_key=("value",))
    assert numeric.add_table(pa.table({"value": [-0.0]})) == 0
    assert numeric.add_table(pa.table({"value": [0.0]})) == 1


@pytest.mark.parametrize("string_type", (pa.string(), pa.large_string()))
def test_chunked_repeated_keys_preserve_canonical_bytes_and_batch_identity(string_type):
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    timestamp_type = pa.timestamp("ns", tz="UTC")
    table = pa.table(
        {
            "asset_id": pa.chunked_array([["甲", "b"], ["甲", "甲"]], type=string_type),
            "timestamp": pa.chunked_array([[1], [2, 1, 2]], type=timestamp_type),
        }
    )
    index = module.ExactPrimaryKeyIndex(primary_key=table.column_names)
    assert index.add_table(table) == 1

    def frame(kind, payload):
        tag = str(kind).encode("utf-8")
        value = b"1" + payload
        return struct.pack(">Q", len(tag)) + tag + struct.pack(">Q", len(value)) + value

    expected = {
        frame(string_type, asset.encode("utf-8"))
        + frame(timestamp_type, struct.pack(">q", timestamp))
        for asset, timestamp in (("甲", 1), ("b", 2), ("甲", 2))
    }
    assert set(index._namespaces["dataset"]) == expected
    reversed_table = table.take(pa.array([3, 2, 1, 0]))
    assert index.add_table(reversed_table) == 4
    assert (
        index.add_comparison_table(
            reversed_table, namespace="comparison", value_columns=()
        )
        == 1
    )
    assert set(index._namespaces["comparison"]) == expected


def test_exact_key_index_inserts_unique_rows_from_duplicate_chunk() -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index = module.ExactPrimaryKeyIndex(primary_key=("key",))
    assert (
        index.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": ["a", "a", "b"]}), preserve_index=False
            )
        )
        == 1
    )
    assert (
        index.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": ["b"]}), preserve_index=False)
        )
        == 1
    )


def test_exact_primary_key_index_uses_target_arrow_scalar_semantics() -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    numeric = module.ExactPrimaryKeyIndex(
        primary_key=("key",), key_schema=pa.schema([pa.field("key", pa.float64())])
    )
    assert (
        numeric.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": [Decimal("1.0")]}), preserve_index=False
            )
        )
        == 0
    )
    assert (
        numeric.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": [1.0]}), preserve_index=False)
        )
        == 1
    )
    assert (
        numeric.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": [-0.0]}), preserve_index=False)
        )
        == 0
    )
    assert (
        numeric.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": [0.0]}), preserve_index=False)
        )
        == 1
    )
    for infinite_value in (math.inf, -math.inf):
        with pytest.raises(ValueError, match="invalid_dataset_primary_key_value"):
            numeric.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": [infinite_value]}), preserve_index=False
                )
            )
    for null_value in (None, math.nan):
        with pytest.raises(ValueError, match="dataset_primary_key_null"):
            numeric.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": [null_value]}), preserve_index=False
                )
            )
    timestamps = module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.timestamp("ns"))]),
    )
    assert (
        timestamps.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": [date(2026, 1, 1)]}), preserve_index=False
            )
        )
        == 0
    )
    assert (
        timestamps.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": [datetime(2026, 1, 1)]}), preserve_index=False
            )
        )
        == 1
    )
    strings = module.ExactPrimaryKeyIndex(
        primary_key=("key",), key_schema=pa.schema([pa.field("key", pa.string())])
    )
    assert (
        strings.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": ["é"]}), preserve_index=False)
        )
        == 0
    )
    assert (
        strings.add_table(
            pa.Table.from_pandas(pd.DataFrame({"key": ["é"]}), preserve_index=False)
        )
        == 0
    )


def test_exact_primary_key_accepts_timestamp_outside_nanosecond_range() -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    out_of_nanosecond_range = 32503680000
    timestamp_type = pa.timestamp("s", tz="UTC")
    table = pa.table({"key": pa.array([out_of_nanosecond_range], type=timestamp_type)})
    timestamps = module.ExactPrimaryKeyIndex(
        primary_key=("key",), key_schema=pa.schema([pa.field("key", timestamp_type)])
    )
    assert timestamps.add_table(table) == 0
    assert timestamps.add_table(table) == 1


def test_dataset_writer_accepts_dictionary_encoded_primary_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_tracking = importlib.import_module("quant_research_platform.storage.exact_keys")
    tracked_schemas: list[pa.Schema] = []
    original_add_table = key_tracking.ExactPrimaryKeyIndex.add_table

    def record_schema(index, table, **kwargs):
        tracked_schemas.append(table.schema.remove_metadata())
        return original_add_table(index, table, **kwargs)

    monkeypatch.setattr(key_tracking.ExactPrimaryKeyIndex, "add_table", record_schema)
    root = make_dataset_artifact_root(tmp_path, "categorical_keys")
    frame = make_default_dataset_frame()
    frame["asset_id"] = frame["asset_id"].astype("category")
    write_dataset(root, frame, spec=make_processed_dataset_spec("categorical_keys"))
    assert sorted(read_dataset(root)["asset_id"].astype(str).tolist()) == sorted(
        frame["asset_id"].astype(str).tolist()
    )
    part = next(filesystem_glob(root / "data", "**/part-*.parquet"))
    with pq.ParquetFile(native_filesystem_path(part)) as parquet:
        persisted_key_schema = pa.schema([parquet.schema_arrow.field("asset_id")])
    assert tracked_schemas
    assert all(
        pa.schema([schema.field("asset_id")]) == persisted_key_schema
        for schema in tracked_schemas
    )


@pytest.mark.parametrize(
    "data_type,negative,positive",
    [
        (pa.float64(), -0.0, 0.0),
        (pa.float32(), -0.0, 0.0),
        (pa.dictionary(pa.int8(), pa.float64()), -0.0, 0.0),
        (pa.list_(pa.float64()), [-0.0], [0.0]),
        (pa.large_list(pa.float64()), [-0.0], [0.0]),
        (pa.list_(pa.float64(), 1), [-0.0], [0.0]),
        (
            pa.struct([("items", pa.list_(pa.float64()))]),
            {"items": [-0.0]},
            {"items": [0.0]},
        ),
        (pa.map_(pa.float64(), pa.float64()), [(-0.0, -0.0)], [(0.0, 0.0)]),
    ],
)
def test_nested_signed_zero_key_identity_preserves_exact_value_payloads(
    data_type, negative, positive
):
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    left = pa.table(
        {
            "key": pa.array([negative], type=data_type),
            "value": pa.array([negative], type=data_type),
        }
    )
    right = pa.table(
        {
            "key": pa.array([positive], type=data_type),
            "value": pa.array([negative], type=data_type),
        }
    )
    original = left.to_batches()[0].serialize().to_pybytes()
    index = module.ExactPrimaryKeyIndex(primary_key=("key",))
    assert index.add_table(left) == 0
    assert index.add_table(right) == 1
    assert index.classify_comparison_table(
        left, namespace="values", value_columns=("value",)
    ) == ([0], 0)
    assert index.classify_comparison_table(
        right, namespace="values", value_columns=("value",)
    ) == ([], 1)
    changed = right.set_column(1, "value", pa.array([positive], type=data_type))
    with pytest.raises(ValueError, match="^conflicting_exact_primary_key_payload$"):
        index.classify_comparison_table(
            changed, namespace="values", value_columns=("value",)
        )
    assert left.to_batches()[0].serialize().to_pybytes() == original


@pytest.mark.parametrize("split", (False, True))
def test_nested_signed_zero_keys_reject_publication_without_changing_values(
    tmp_path, split
):
    frame = make_default_dataset_frame().iloc[:2].copy().reset_index(drop=True)
    frame["asset_id"] = [[-0.0], [0.0]]
    frame["timestamp"] = frame.timestamp.iloc[0]
    root = make_dataset_artifact_root(tmp_path, "nested_zero")
    with pytest.raises(ValueError, match="^duplicate_dataset_primary_key$"):
        write_dataset(
            root,
            (frame.iloc[:1], frame.iloc[1:]) if split else frame,
            spec=make_processed_dataset_spec("nested_zero"),
        )
    assert not filesystem_exists(root)
    frame.loc[1, "timestamp"] += pd.Timedelta(days=1)
    frame["value"] = [-0.0, 0.0]
    write_dataset(root, frame, spec=make_processed_dataset_spec("nested_zero"))
    restored = read_dataset(root).sort_values("timestamp")
    for actual, expected in zip(restored.asset_id, frame.asset_id, strict=True):
        assert struct.pack(">d", actual[0]) == struct.pack(">d", expected[0])
    for actual, expected in zip(restored.value, frame.value, strict=True):
        assert struct.pack(">d", actual) == struct.pack(">d", expected)
