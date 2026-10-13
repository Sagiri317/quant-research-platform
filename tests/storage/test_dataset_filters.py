from __future__ import annotations

import os
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import _prepare_in_filter
from quant_research_platform.storage.identity import (
    filesystem_write_bytes,
    native_filesystem_path,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


@pytest.mark.parametrize("parent_has_time", [False, True])
@pytest.mark.parametrize("operator", ("==", "in"))
def test_filtered_views_only_narrow_parent_rows_and_preserve_projection(
    tmp_path: Path, parent_has_time: bool, operator: str
) -> None:
    from quant_research_platform.contracts.time import TimeWindow

    root = make_dataset_artifact_root(tmp_path, "views")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b"] * 3,
            "timestamp": pd.date_range("2020-01-01", periods=3).repeat(2),
            "value": [1.0, 101.0, 2.0, 102.0, 3.0, 103.0],
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("views"))
    day = TimeWindow("2020-01-02", "2020-01-02")
    wide = TimeWindow("2020-01-01", "2020-01-03")

    def values(session):
        result = pd.concat(
            tuple(dataset_reading.iter_dataset_chunks(session, batch_rows=2048)),
            ignore_index=True,
        )
        assert tuple(result.columns) == ("value",)
        return result.value.tolist()

    parent = dataset_reading.open_dataset_session(
        root,
        columns=("value",),
        time_window=day if parent_has_time else None,
        row_filters=(("asset_id", operator, "a" if operator == "==" else ("a",)),),
    )
    widened = parent.filtered_view(wide)
    assert values(widened) == ([2.0] if parent_has_time else [1.0, 2.0, 3.0])
    narrowed = widened.filtered_view(day)
    descendant = narrowed.filtered_view(wide)
    assert values(descendant) == [2.0]
    disjoint = narrowed.filtered_view(TimeWindow("2020-01-03", "2020-01-03"))
    assert values(disjoint) == []
    assert values(parent) == ([2.0] if parent_has_time else [1.0, 2.0, 3.0])


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
    write_dataset(root, frame, spec=make_processed_dataset_spec("tuple_in_filter"))
    expected = frame.loc[
        frame["asset_id"].isin(("A", "C")), ["timestamp", "value"]
    ].reset_index(drop=True)
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    session = dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("A", "A", "C")),),
    )
    chunked = pd.concat(
        dataset_reading.iter_dataset_chunks(session, batch_rows=1), ignore_index=True
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("A", "A", "C")),),
    )
    materialized = dataset_reading.materialize_dataset_session(session, batch_rows=1)
    pd.testing.assert_frame_equal(materialized, expected)
    pd.testing.assert_frame_equal(chunked, expected)
    session = dataset_reading.open_dataset_session(
        root,
        columns=("asset_id",),
        time_window=None,
        row_filters=(("asset_id", "==", "B"),),
    )
    scalar = pd.concat(
        dataset_reading.iter_dataset_chunks(session, batch_rows=2048), ignore_index=True
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
    tmp_path: Path, column: str, value: object
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_grammar")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tuple_in_filter_grammar"),
    )
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading.open_dataset_session(
            root, columns=None, time_window=None, row_filters=((column, "in", value),)
        )
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
        dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
            row_filters=(("asset_id", "in", TupleSubclass(("A",))),),
        )
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
    session = dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    )
    chunks = list(dataset_reading.iter_dataset_chunks(session, batch_rows=2048))
    assert len(chunks) == 1
    assert chunks[0].empty
    assert chunks[0].dtypes.to_dict() == {
        "timestamp": np.dtype("datetime64[ns]"),
        "value": make_default_dataset_frame()["value"].dtype,
    }


def test_filtered_empty_chunk_final_verify_preserves_caller_primary(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "filtered_empty_primary")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("filtered_empty_primary"),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    )
    iterator = dataset_reading.iter_dataset_chunks(session, batch_rows=2048)
    with pytest.raises(RuntimeError, match="primary_failure"):
        with closing_iterator(iterator):
            assert next(iterator).empty
            part = session.all_paths[0]
            replacement = tmp_path / "replacement-primary.parquet"
            filesystem_write_bytes(
                replacement, Path(native_filesystem_path(part)).read_bytes()
            )
            os.replace(
                native_filesystem_path(replacement), native_filesystem_path(part)
            )
            raise RuntimeError("primary_failure")


@pytest.mark.parametrize("ordered", (False, True))
def test_categorical_empty_result_preserves_complete_domain(
    tmp_path: Path, ordered
) -> None:
    root = make_dataset_artifact_root(tmp_path, "categorical_empty_domain")
    frame = make_default_dataset_frame()
    categories = ["unused", *reversed(frame["value"].astype(str).dropna().unique())]
    frame["value"] = pd.Categorical(
        frame["value"].astype(str), categories=categories, ordered=ordered
    )
    write_dataset(
        root, frame, spec=make_processed_dataset_spec("categorical_empty_domain")
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    )
    chunks = list(dataset_reading.iter_dataset_chunks(session, batch_rows=1))
    assert len(chunks) == 1
    assert chunks[0].empty
    assert isinstance(chunks[0]["value"].dtype, pd.CategoricalDtype)
    assert chunks[0]["value"].cat.ordered is ordered
    assert chunks[0]["value"].cat.categories.tolist() == (
        categories if ordered else sorted(categories)
    )


def test_session_filter_reuses_prepared_values(tmp_path: Path, monkeypatch) -> None:
    root = make_dataset_artifact_root(tmp_path, "tuple_in_filter_prepared_once")
    frame = make_default_dataset_frame(
        asset_ids=("A", "B", "C"),
        dates=("2014-01-02", "2015-01-05", "2016-01-04"),
        values=(1.0, 2.0, 3.0),
    )
    write_dataset(
        root, frame, spec=make_processed_dataset_spec("tuple_in_filter_prepared_once")
    )
    values = ("A", "C", *(f"missing_{position:08d}" for position in range(40000)))
    calls = []
    prepare = dataset_reading._prepare_in_filter

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(dataset_reading, "_prepare_in_filter", counted)
    session = dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", values), ("asset_id", "in", values)),
    )
    chunks = list(dataset_reading.iter_dataset_chunks(session, batch_rows=1))
    assert sum(map(len, chunks)) == 2
    assert len(calls) == 2


def test_membership_internal_grammar_rejects_raw_arrow_values() -> None:
    raw = pa.array([1, 2], type=pa.int64())
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        dataset_reading._freeze_filter_value(raw)


def test_dataset_prepared_in_filter_compacts_sorted_duplicates() -> None:
    prepared = _prepare_in_filter(pa.field("value", pa.int64()), (2, 1, 1))
    assert prepared.values.to_pylist() == [1, 2]


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
    tmp_path: Path, data_type: pa.DataType, values: tuple[time, ...]
) -> None:
    path = tmp_path / "time.parquet"
    pq.write_table(
        pa.table({"value": pa.array(values, type=data_type)}),
        native_filesystem_path(path),
    )
    prepared = _prepare_in_filter(pa.field("value", data_type), (values[0], values[-1]))
    equal = list(
        dataset_physical.iter_dataset_part_batches(
            path,
            parquet_filters=[("value", "==", values[1])],
            batch_rows=2,
        )
    )
    membership = list(
        dataset_physical.iter_dataset_part_batches(
            path,
            parquet_filters=[("value", "in", prepared)],
            batch_rows=2,
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
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        _prepare_in_filter(pa.field("value", pa.list_(pa.int64())), ((1,),))
        pass


def test_dataset_prepared_in_filter_accepts_fixed_size_binary() -> None:
    prepared = _prepare_in_filter(pa.field("value", pa.binary(2)), (b"ab", b"cd"))
    assert prepared.values.to_pylist() == [b"ab", b"cd"]


def test_dataset_filter_preflight_rejects_surrogate_before_arrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def reject_arrow(*_args, **_kwargs):
        raise AssertionError("Arrow must not receive an invalid surrogate")

    monkeypatch.setattr(dataset_reading.pa, "array", reject_arrow)
    monkeypatch.setattr(dataset_reading.pa, "scalar", reject_arrow)
    with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
        _prepare_in_filter(pa.field("asset_id", pa.string()), ("valid\ud800suffix",))
        pass


def test_timestamp_membership_normalizes_naive_and_aware_values():
    field = pa.field("timestamp", pa.timestamp("ns", tz="UTC"))
    values = (
        pd.Timestamp("2020-01-02", tz="UTC"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-01", tz="UTC"),
    )
    prepared = _prepare_in_filter(field, values)
    assert prepared.values.equals(
        pa.array(
            [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-01-02", tz="UTC"),
            ],
            type=field.type,
        )
    )


@pytest.mark.parametrize("operator", ("==", "in"))
@pytest.mark.parametrize("selected", (1, 2**63, 2**63 + 1, 2**63 + 2))
def test_native_uint64_filters_preserve_unsigned_field_type(
    tmp_path, operator, selected
):
    root = make_dataset_artifact_root(tmp_path, "unsigned_filter")
    values = [1, 2**63, 2**63 + 1, 2**63 + 2, None]
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"a{i}" for i in range(len(values))],
            "timestamp": pd.Timestamp("2020-01-01"),
            "value": pd.array(values, dtype="UInt64"),
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("unsigned_filter"))
    value = selected if operator == "==" else (selected,)
    session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=None, row_filters=(("value", operator, value),)
    )
    actual = dataset_reading.materialize_dataset_session(session)
    assert actual.value.tolist() == [selected]
    for invalid in (-1, 2**64, float(selected), True, None):
        value = invalid if operator == "==" else (invalid,)
        with pytest.raises(ValueError, match="invalid_dataset_physical_filter_value"):
            dataset_reading.open_dataset_session(
                root,
                columns=None,
                time_window=None,
                row_filters=(("value", operator, value),),
            )
