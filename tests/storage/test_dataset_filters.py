from __future__ import annotations

import builtins
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.errors as storage_errors
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    write_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_read_bytes,
    filesystem_replace,
    filesystem_write_bytes,
    native_filesystem_path,
)
from tests.helpers.runtime_resources import capture_test_policy
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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    def values(session):
        result = pd.concat(
            tuple(
                dataset_reading.iter_dataset_chunks(session, target_chunk_bytes=1024**2)
            ),
            ignore_index=True,
        )
        assert tuple(result.columns) == ("value",)
        return result.value.tolist()

    with runtime_resource_policy_scope(policy):
        with dataset_reading.open_dataset_session(
            root,
            columns=("value",),
            time_window=day if parent_has_time else None,
            row_filters=(("asset_id", operator, "a" if operator == "==" else ("a",)),),
            resource_policy=policy,
        ) as parent:
            baseline = policy.used_memory_bytes
            with parent.filtered_view(wide) as widened:
                assert values(widened) == (
                    [2.0] if parent_has_time else [1.0, 2.0, 3.0]
                )
                with widened.filtered_view(day) as narrowed:
                    with narrowed.filtered_view(wide) as descendant:
                        assert values(descendant) == [2.0]
                    with narrowed.filtered_view(
                        TimeWindow("2020-01-03", "2020-01-03")
                    ) as disjoint:
                        assert values(disjoint) == []
            assert values(parent) == ([2.0] if parent_has_time else [1.0, 2.0, 3.0])
            assert policy.used_memory_bytes == baseline
        assert policy.used_memory_bytes == 0


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
        materialized = dataset_reading.materialize_dataset_session(
            session,
            target_chunk_bytes=512,
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


@pytest.mark.parametrize("completion", ("close", "exhaust"))
def test_filtered_empty_chunk_rechecks_identity_after_yield(
    tmp_path: Path,
    completion: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"filtered_empty_{completion}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"filtered_empty_{completion}"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    ) as session:
        iterator = dataset_reading.iter_dataset_chunks(
            session,
            target_chunk_bytes=1024**2,
        )
        assert next(iterator).empty
        part = session.all_paths[0]
        replacement = tmp_path / f"replacement-{completion}.parquet"
        filesystem_write_bytes(replacement, filesystem_read_bytes(part))
        filesystem_replace(replacement, part)

        if completion == "close":
            with pytest.raises(ValueError, match="dataset_file_identity_changed"):
                iterator.close()  # type: ignore[attr-defined]
        else:
            with pytest.raises(ValueError, match="dataset_file_identity_changed"):
                next(iterator)


def test_filtered_empty_chunk_final_verify_preserves_caller_primary(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "filtered_empty_primary")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("filtered_empty_primary"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(("asset_id", "in", ("missing",)),),
    ) as session:
        iterator = dataset_reading.iter_dataset_chunks(
            session,
            target_chunk_bytes=1024**2,
        )
        with pytest.raises(RuntimeError, match="primary_failure"):
            with closing_iterator(iterator):
                assert next(iterator).empty
                part = session.all_paths[0]
                replacement = tmp_path / "replacement-primary.parquet"
                filesystem_write_bytes(replacement, filesystem_read_bytes(part))
                filesystem_replace(replacement, part)
                raise RuntimeError("primary_failure")


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


def test_session_filter_reuses_prepared_values(
    tmp_path: Path,
    monkeypatch,
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
    calls = []
    prepare = dataset_reading._prepare_in_filter

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(dataset_reading, "_prepare_in_filter", counted)
    with dataset_reading.open_dataset_session(
        root,
        columns=("timestamp", "value"),
        time_window=None,
        row_filters=(
            ("asset_id", "in", values),
            ("asset_id", "in", values),
        ),
        resource_policy=policy,
    ) as session:
        chunks = list(
            dataset_reading.iter_dataset_chunks(session, target_chunk_bytes=512)
        )

    assert sum(map(len, chunks)) == 2
    assert len(calls) == 2  # Each predicate is prepared once, independent of chunks.


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


@pytest.mark.parametrize("explicit_cap_bytes", (None, 4 * 1024**2))
def test_dataset_filter_capacity_uses_the_typed_storage_signal(
    tmp_path: Path, explicit_cap_bytes: int | None
) -> None:
    root = make_dataset_artifact_root(tmp_path, "typed_filter_capacity")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("typed_filter_capacity"),
    )
    width = (2 if explicit_cap_bytes is not None else 4) * 1024**2
    values = ("x" * width,)
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=explicit_cap_bytes,
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
            row_filters=(("asset_id", "in", values),),
            resource_policy=policy,
        ):
            pass
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


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
        values = ("x" * (policy.memory_budget_bytes + 1),)
    elif shape == "many_small":
        values = ("x",) * (policy.memory_budget_bytes // 4 + 1)
    else:
        value = "duplicate"
        per_value_bytes = 4 + len(value)
        values = (value,) * (policy.memory_budget_bytes // per_value_bytes + 1)

    def reject_bulk_array(*_args, **_kwargs):
        nonlocal bulk_array_called
        bulk_array_called = True
        raise AssertionError("bulk pa.array must not run after a failed preflight")

    def reject_scalar(*_args, **_kwargs):
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
        )
    assert bulk_array_called is False
    assert scalar_called is False


@pytest.mark.parametrize("explicit_cap_bytes", (None, 1024**2))
def test_dataset_filter_preparation_subtracts_shared_reservations_before_sort(
    monkeypatch: pytest.MonkeyPatch,
    explicit_cap_bytes: int | None,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=explicit_cap_bytes,
    )
    values = (1, 2)

    def reject_late_work(*_args, **_kwargs):
        raise AssertionError("late filter preparation work ran before task admission")

    monkeypatch.setattr(dataset_reading, "sorted", reject_late_work, raising=False)
    monkeypatch.setattr(dataset_reading.pa, "array", reject_late_work)

    with (
        runtime_resource_policy_scope(policy),
        policy.lease("outer", policy.working_set_headroom_bytes - 1),
        pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="dataset_filter_values_exceed_task_memory_budget",
        ),
    ):
        dataset_reading._prepare_in_filter(
            pa.field("value", pa.int64()),
            values,
            resource_policy=policy,
        )
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


@pytest.mark.parametrize("stage", ("sort", "arrow"))
def test_dataset_filter_preparation_checks_physical_overlap_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    values = (
        ("x",) * 400
        if stage == "sort"
        else tuple(f"{index:04d}" + "x" * 48 for index in range(64))
    )
    policy = capture_test_policy(available_memory_bytes=4096)
    sort_calls = []

    def guarded_sort(items):
        assert stage == "arrow", "sort allocated beyond physical capacity"
        sort_calls.append(1)
        return builtins.sorted(items)

    def reject_arrow(*_args, **_kwargs):
        raise AssertionError("Arrow allocated beyond the remaining physical capacity")

    monkeypatch.setattr(dataset_reading, "sorted", guarded_sort, raising=False)
    monkeypatch.setattr(dataset_reading.pa, "array", reject_arrow)

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="^dataset_filter_values_exceed_task_memory_budget$",
    ):
        dataset_reading._prepare_in_filter(
            pa.field("asset_id", pa.string()),
            values,
            resource_policy=policy,
        )

    assert len(sort_calls) == (stage == "arrow")
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


@pytest.mark.parametrize("shape", ("small", "sort_above_soft", "arrow_above_soft"))
def test_dataset_filter_preparation_accepts_physical_capacity_above_soft_target(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    if shape == "small":
        values = ("b", "a")
    elif shape == "sort_above_soft":
        values = ("duplicate",) * 256
    else:
        values = ("x" * 3000,)
    policy = capture_test_policy(available_memory_bytes=4096)
    reservations = []
    array = dataset_reading.pa.array

    def observed_sort(items):
        reservations.append(policy.used_memory_bytes)
        return builtins.sorted(items)

    def observed_array(items, **kwargs):
        reservations.append(policy.used_memory_bytes)
        return array(items, **kwargs)

    monkeypatch.setattr(dataset_reading, "sorted", observed_sort, raising=False)
    monkeypatch.setattr(dataset_reading.pa, "array", observed_array)

    prepared = dataset_reading._prepare_in_filter(
        pa.field("asset_id", pa.string()),
        values,
        resource_policy=policy,
    )

    assert prepared.values.to_pylist() == sorted(set(values))
    assert len(reservations) == 2
    assert max(reservations) < policy.captured_available_memory_bytes
    if shape != "small":
        assert max(reservations) > policy.memory_budget_bytes
        borrowed_bytes = sys.getsizeof(values) + sum(map(len, values))
        assert (
            borrowed_bytes + max(reservations) > policy.captured_available_memory_bytes
        )
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


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
    )

    assert prepared.values.to_pylist() == [value]
    assert prepared.retained_memory_bytes <= policy.memory_budget_bytes


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
    pq.write_table(
        pa.table({"value": pa.array(values, type=data_type)}),
        native_filesystem_path(path),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", data_type),
        (values[0], values[-1]),
        resource_policy=policy,
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
        )


def test_dataset_prepared_in_filter_accepts_fixed_size_binary() -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )

    prepared = dataset_reading._prepare_in_filter(
        pa.field("value", pa.binary(2)),
        (b"ab", b"cd"),
        resource_policy=policy,
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
        values = (b"x" * (policy.memory_budget_bytes + 1),)
        expected = "dataset_filter_values_exceed_(payload_target|task_memory_budget)"
    else:
        count = policy.memory_budget_bytes // (offset_bytes + 1) + 1
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
        )
