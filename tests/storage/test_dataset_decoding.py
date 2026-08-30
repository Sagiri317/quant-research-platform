from __future__ import annotations

import gc
import json
import weakref
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.errors as storage_errors
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_payload_from_frame,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.dataset_test_support import (
    _issued_test_policy,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


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
    row_bytes, fixed_bytes = dataset_physical.record_batch_decoded_working_set_bounds(
        batch
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
    row_bytes, fixed_bytes = dataset_physical.record_batch_decoded_working_set_bounds(
        batch
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
        with policy.lease("research.time.pending", policy.memory_budget_bytes):
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
        with policy.lease(
            "storage.dataset.write",
            policy.memory_budget_bytes - composed_headroom,
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
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=exact_headroom + retained_bytes,
    )
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
        with policy.lease("research.model.fitted_state", retained_bytes):
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
    with runtime_resource_policy_scope(policy):
        converted = dataset_reading.iter_dataset_file_batches(
            context,
            columns=("value",),
            target_chunk_bytes=max(output_bytes),
        )

        with closing_iterator(converted):
            next(converted)
            remaining = policy.memory_budget_bytes - policy.used_memory_bytes
            assert remaining > 0
            with policy.lease(
                "research.time.pending",
                remaining,
            ):
                with pytest.raises(
                    ValueError,
                    match="dataset_working_set_exceeds_task_memory_budget",
                ):
                    next(converted)

    assert conversion_count == 1


def test_dataset_file_batches_do_not_double_count_reserved_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    batches = tuple(
        pa.record_batch([pa.array([index])], names=["value"]) for index in range(2)
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

    with closing_iterator(converted):
        first_path, first = next(converted)
        with runtime_resource_policy_scope(policy):
            with policy.lease("research.time.pending", output_bytes[0]):
                second_path, second = next(converted)

    assert first_path == path
    assert second_path == path
    pd.testing.assert_frame_equal(first, expected[0])
    pd.testing.assert_frame_equal(second, expected[1])


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


def test_metadata_free_oversized_row_group_rejects_before_first_arrow_pull(
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

    dataset_reading._fill_materialized_column(buffers["nullable"], nullable, start=0)
    dataset_reading._fill_materialized_column(buffers["string"], strings, start=0)
    observed = dataset_reading._materialized_frame(buffers)

    pd.testing.assert_frame_equal(observed, expected)


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
    reference_readback = pd.concat(
        tuple(
            pd.read_parquet(path)
            for path in sorted((root / "data").rglob("part-*.parquet"))
        ),
        ignore_index=True,
    )
    hashes = pd.util.hash_pandas_object(
        reference_readback.loc[:, list(hash_columns)],
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
