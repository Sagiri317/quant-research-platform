from __future__ import annotations

import importlib
import importlib.util
import json
import math
import weakref
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.schema as dataset_schema
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.errors as storage_errors
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_payload_from_frame,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.dataset_test_support import (
    _bound_first_frames,
    _issued_test_policy,
    _writer_test_policy,
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
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.decimal128(18, 4))]),
    ) as decimals:
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
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.int64())]),
    ) as integers:
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
            expected_namespace="expected",
            actual_namespace="dataset",
        )
        assert (missing, extra) == (len(frame), len(frame))
        raise ValueError("staged_dataset_primary_key_mismatch")

    with pytest.raises(ValueError, match="staged_dataset_primary_key_mismatch"):
        dataset_writing.write_dataset(
            root,
            _bound_first_frames((frame,)),
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
            _bound_first_frames((frame,)),
            spec=make_processed_dataset_spec("invalid_labels"),
            resource_policy=current_runtime_resource_policy(),
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
        )


def test_physical_in_filter_does_not_treat_soft_headroom_as_hard_cap() -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(128), type=pa.int64())],
        names=["value"],
    )

    filtered = dataset_physical._filter_arrow_batch(
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
    assert filtered.num_rows == 1


@pytest.mark.parametrize("operator", ("==", ">="))
def test_physical_scalar_filter_does_not_treat_soft_headroom_as_hard_cap(
    operator: str,
) -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(range(128), type=pa.int64())],
        names=["value"],
    )

    filtered = dataset_physical._filter_arrow_batch(
        batch,
        [("value", operator, 1)],
        operation_headroom_bytes=batch.get_total_buffer_size(),
    )
    assert filtered.num_rows == (1 if operator == "==" else 127)


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


@pytest.mark.parametrize("budget_slack", (0, 1))
def test_materialize_rejects_exact_buffers_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    budget_slack: int,
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
    required_payload_bound = (
        sum(pandas_frame_payload_bytes(frame) for frame in frames) + 1
    )
    monkeypatch.setattr(
        dataset_reading,
        "_materialized_buffer_payload_bytes",
        lambda *_args, **_kwargs: required_payload_bound,
    )
    materialized_budget_bytes = max(
        1,
        required_payload_bound - budget_slack,
    )
    context = SimpleNamespace(
        resource_policy=_issued_test_policy(),
        logical_columns=("value",),
        logical_dtypes=logical_dtype_payload_from_frame(frames[0]),
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
    policy = _writer_test_policy(working_headroom_bytes=5 * 256 * 1024)
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
            _bound_first_frames((make_default_dataset_frame(),)),
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
        policy.memory_budget_bytes
        - dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
    )

    with runtime_resource_policy_scope(policy):
        with policy.lease("test.outer.owner", other_owner_bytes):
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                dataset_writing.write_dataset(
                    make_dataset_artifact_root(tmp_path, "writer_shared_headroom"),
                    _bound_first_frames((make_default_dataset_frame(),)),
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
            _bound_first_frames((make_default_dataset_frame(),)),
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
            _bound_first_frames(
                (frame,), bound=pandas_frame_payload_bytes(frame) + 128 * 1024
            ),
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
            _bound_first_frames((make_default_dataset_frame(),)),
            spec=make_processed_dataset_spec("writer_native_admission"),
            resource_policy=policy,
        )

    assert conversions == 1
    assert native_writes == 0


def test_dataset_parquet_disk_preflight_precedes_native_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table({"value": [1]})
    path = tmp_path / "part.parquet"
    policy = _writer_test_policy()
    captured: dict[str, int] = {}
    native_writes = 0

    def reject_disk(path: Path, *, reserve_bytes: int) -> None:
        captured["reserve_bytes"] = reserve_bytes
        assert path == tmp_path
        raise OSError("dataset_disk_preflight")

    def forbid_native_write(*_args, **_kwargs):  # noqa: ANN002, ANN003
        nonlocal native_writes
        native_writes += 1
        raise AssertionError("native_write_started_before_disk_preflight")

    monkeypatch.setattr(dataset_writing, "require_free_disk_headroom", reject_disk)
    monkeypatch.setattr(dataset_writing.pq, "write_table", forbid_native_write)

    with pytest.raises(OSError, match="dataset_disk_preflight") as raised:
        with runtime_resource_policy_scope(policy):
            dataset_writing.write_parquet_file(table, path)

    persisted = dataset_writing.table_with_decoded_row_bounds(table)
    expected = dataset_writing._table_parquet_page_working_set_bytes(
        persisted,
        rows=table.num_rows,
    ) + dataset_writing._parquet_row_group_footer_upper_bound(
        persisted.schema,
        rows=table.num_rows,
    )
    assert captured["reserve_bytes"] == expected
    assert native_writes == 0
    assert not path.exists()
    assert str(raised.value) == "dataset_disk_preflight"
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


def test_qrp_parquet_reader_bounded_completes_with_physical_page_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "row-groups.parquet"
    table = pa.table({"value": ["x" * 4096] * 100})
    page_headroom = (
        table.nbytes + 2 * dataset_physical._PARQUET_NATIVE_METADATA_PAGE_BYTES
    )
    policy = _writer_test_policy(working_headroom_bytes=page_headroom)
    with runtime_resource_policy_scope(policy):
        dataset_writing.write_parquet_file(table, path)
        batches = tuple(
            dataset_physical.iter_dataset_part_batches(
                path,
                expected_identity=None,
                target_chunk_bytes=32 * 1024,
                operation_headroom_bytes=4 * page_headroom,
                columns=("value",),
                source_batch_rows=1,
            )
        )

    assert sum(batch.num_rows for batch in batches) == 100
    assert max(batch.num_rows for batch in batches) == 1


def test_dataset_parquet_writer_bounds_native_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    table = pa.table({"value": pa.nulls(64 * 1024 * 1024 + 1)})
    captured: dict[str, object] = {}
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=8 * 1024**2,
    )

    def capture_write(*_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        captured["table"] = _args[0]
        captured["used_memory_bytes"] = policy.used_memory_bytes

    monkeypatch.setattr(dataset_writing.pq, "write_table", capture_write)
    with runtime_resource_policy_scope(policy):
        dataset_writing.write_parquet_file(table, tmp_path / "part.parquet")

    assert captured["use_dictionary"] is False
    assert captured["write_page_index"] is False
    assert captured["write_page_checksum"] is False
    assert captured["write_statistics"] is False
    assert captured["max_rows_per_page"] == captured["write_batch_size"]
    assert captured["write_batch_size"] < table.num_rows
    assert captured["row_group_size"] == 64 * 1024 * 1024
    persisted = captured["table"]
    assert isinstance(persisted, pa.Table)
    batch_rows = captured["write_batch_size"]
    assert type(batch_rows) is int
    page_bytes = dataset_physical._table_parquet_page_working_set_bytes(
        persisted,
        rows=batch_rows,
    )
    footer_bytes = dataset_physical._parquet_row_group_footer_upper_bound(
        persisted.schema,
        rows=1,
    )
    assert captured["used_memory_bytes"] >= page_bytes + footer_bytes


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
    policy = _writer_test_policy(working_headroom_bytes=2 * 1024**2)
    root = make_dataset_artifact_root(tmp_path, "writer_task_headroom")
    result = dataset_writing.write_dataset(
        root,
        _bound_first_frames(
            (frame,), bound=pandas_frame_payload_bytes(frame) + 128 * 1024
        ),
        spec=make_processed_dataset_spec("writer_task_headroom"),
        resource_policy=policy,
    )

    assert result.row_count == row_count
    assert len(tuple(root.glob("data/**/*.parquet"))) > 1
    assert policy.used_memory_bytes == 0


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
        _bound_first_frames((frame,)),
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


@pytest.mark.parametrize("offset", (-1, 0, 1))
@pytest.mark.parametrize("writer", ("dataset", "artifact"))
def test_parquet_writers_admit_one_page_above_soft_target(tmp_path, offset, writer):
    table = pa.table({"value": ["x" * 4096] * 3})
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with policy.lease("test.retained", policy.memory_budget_bytes + offset):
            if writer == "dataset":
                path = tmp_path / "over-soft.parquet"
                dataset_writing.write_parquet_file(table, path)
                assert pq.read_table(path).to_pydict() == table.to_pydict()
            else:
                persisted, resources, batch_rows, _ = (
                    dataset_physical._parquet_table_write_state(table)
                )
                with resources:
                    assert persisted.to_pydict() == table.to_pydict()
                    assert batch_rows == 1
        assert policy.used_memory_bytes == 0
