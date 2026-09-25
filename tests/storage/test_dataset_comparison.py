from __future__ import annotations

import os
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.runtime.resources as runtime_resources
import quant_research_platform.storage.datasets.comparison as dataset_comparison
import quant_research_platform.storage.datasets.fingerprinting as dataset_fingerprinting
import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.errors as storage_errors
import quant_research_platform.storage.exact_keys as dataset_key_tracking
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_payload_from_frame,
)
from quant_research_platform.storage.datasets.types import DatasetFingerprint
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_mkdir,
    filesystem_stat,
    native_filesystem_path,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.dataset_test_support import (
    _bound_first_frames,
    _issued_test_policy,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


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
        dataset_writing.write_dataset(
            root,
            _bound_first_frames((frame,)),
            spec=make_processed_dataset_spec("policy_propagation"),
            resource_policy=policy,
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
        dataset_inspection.inspect_dataset(
            root,
            time_window=TimeWindow("2014-01-01", "2015-12-31"),
            resource_policy=policy,
        )
        dataset_fingerprinting.fingerprint_dataset(root, resource_policy=policy)

    assert writer_targets and all(
        0 < target <= policy.memory_budget_bytes for target in writer_targets
    )
    assert read_targets and all(
        0 < target <= policy.memory_budget_bytes for target in read_targets
    )
    assert inspection_targets and all(
        0 < target <= policy.memory_budget_bytes for target in inspection_targets
    )
    assert fingerprint_physical_targets and all(
        0 < target <= policy.memory_budget_bytes
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


def test_file_batch_default_uses_live_policy_headroom(
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
    assert observed_headroom == [context.resource_policy.memory_budget_bytes]


def test_compare_reuses_one_policy_for_both_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    filesystem_mkdir(left)
    filesystem_mkdir(right)
    observed: list[RuntimeResourcePolicy] = []
    fingerprint_refs: list[weakref.ReferenceType[DatasetFingerprint]] = []
    active_contexts = 0
    maximum_active_contexts = 0

    def fake_fingerprint(*_args, resource_policy, **_kwargs):
        assert active_contexts == 2
        observed.append(resource_policy)
        fingerprint = DatasetFingerprint(
            schema_hash="schema",
            column_names=("value",),
            row_count=0,
            key_columns=("value",),
            content_hash="content",
            partition_fingerprints=(),
            file_count=1,
        )
        fingerprint_refs.append(weakref.ref(fingerprint))
        return runtime_resources.OwnedValue(
            fingerprint,
            resource_policy.lease("test.compare.fingerprint", 0),
        )

    class Context:
        def __init__(self, root: Path) -> None:
            self.root = root
            self.retained_memory_bytes = 0

        def verify(self):
            pass

        def __enter__(self):  # noqa: ANN204
            nonlocal active_contexts, maximum_active_contexts
            active_contexts += 1
            maximum_active_contexts = max(maximum_active_contexts, active_contexts)
            return self

        def __exit__(self, *_args):  # noqa: ANN204
            nonlocal active_contexts
            assert len(fingerprint_refs) == 2
            assert all(reference() is None for reference in fingerprint_refs)
            active_contexts -= 1
            return None

    monkeypatch.setattr(
        dataset_comparison,
        "open_dataset_session",
        lambda root, **_kwargs: Context(root),
    )
    monkeypatch.setattr(
        dataset_comparison,
        "_fingerprint_dataset_session_owned",
        fake_fingerprint,
    )
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
    assert maximum_active_contexts == 2
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
    collision = fingerprint_dataset(left)

    def owned_collision(*_args, resource_policy, **_kwargs):  # noqa: ANN001, ANN003
        return runtime_resources.OwnedValue(
            collision,
            resource_policy.lease(
                "test.compare.collision",
                dataset_fingerprinting.dataset_fingerprint_retained_memory_bytes(
                    collision
                ),
            ),
        )

    monkeypatch.setattr(
        dataset_comparison,
        "_fingerprint_dataset_session_owned",
        owned_collision,
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


def test_compare_treats_partition_and_file_boundaries_as_layout_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "layout_left")
    right = make_dataset_artifact_root(tmp_path, "layout_right")
    frame = make_default_dataset_frame()
    write_dataset(
        left,
        (frame.iloc[:1], frame.iloc[1:]),
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

    def owned_fingerprint(*_args, resource_policy, **_kwargs):  # noqa: ANN001, ANN003
        return runtime_resources.OwnedValue(
            next(fingerprints),
            resource_policy.lease("test.compare.fingerprint", 0),
        )

    monkeypatch.setattr(
        dataset_comparison,
        "_fingerprint_dataset_session_owned",
        owned_fingerprint,
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


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit),
)
def test_dataset_session_close_preserves_active_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
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

    policy: RuntimeResourcePolicy | None = None
    try:
        with pytest.raises(primary_type, match="primary_failure"):
            with dataset_reading.open_dataset_session(
                root,
                columns=None,
                time_window=None,
            ) as session:
                policy = session.resource_policy
                raise primary_type("primary_failure")
    finally:
        if policy is not None and policy.state == "POISONED":
            policy._lock.close()  # noqa: SLF001


def test_dataset_session_normal_close_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
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

    policy: RuntimeResourcePolicy | None = None
    try:
        with pytest.raises(OSError, match="cleanup_failure"):
            with dataset_reading.open_dataset_session(
                root,
                columns=None,
                time_window=None,
            ) as session:
                policy = session.resource_policy
    finally:
        if policy is not None and policy.state == "POISONED":
            policy._lock.close()  # noqa: SLF001


@pytest.mark.parametrize("stage", ("fingerprint", "exact"))
def test_compare_cleanup_does_not_replace_programmer_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
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
    policy: RuntimeResourcePolicy | None = None
    real_open = dataset_comparison.open_dataset_session

    def record_open(*args: object, **kwargs: object):  # noqa: ANN202
        nonlocal policy
        current = kwargs.get("resource_policy")
        assert isinstance(current, RuntimeResourcePolicy)
        policy = current
        return real_open(*args, **kwargs)

    monkeypatch.setattr(dataset_comparison, "open_dataset_session", record_open)
    if stage == "fingerprint":
        monkeypatch.setattr(
            dataset_comparison,
            "_fingerprint_dataset_session_owned",
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

    try:
        with pytest.raises(RuntimeError, match="programmer_primary"):
            dataset_comparison.compare_datasets(root, root, resource_policy=None)
    finally:
        if policy is not None and policy.state == "POISONED":
            policy._lock.close()  # noqa: SLF001


def test_owned_result_release_clears_payload_before_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    original_close = runtime_resources.MemoryLease.close
    observed: list[object | None] = []
    frame = pd.DataFrame({"value": [1, 2, 3]})
    frame_ref = weakref.ref(frame)

    with runtime_resource_policy_scope(policy):
        lease = policy.lease("storage.dataset.inspection.result", 4096)
        owned = runtime_resources.OwnedValue(frame, lease)
        del frame

        def close_with_failure(current: runtime_resources.MemoryLease) -> None:
            if current is lease:
                observed.append(frame_ref())
                raise OSError("result_lease_close_failure")
            original_close(current)

        monkeypatch.setattr(
            runtime_resources.MemoryLease,
            "close",
            close_with_failure,
        )
        primary = RuntimeError("result_primary")
        try:
            raise primary
        except RuntimeError as active:
            owned.release(owner="test.result")
            assert active is primary

        assert owned.value is None
        assert observed == [None]
        assert any(
            "OSError" in note and "result_lease_close_failure" not in note
            for note in primary.__notes__
        )
        monkeypatch.setattr(runtime_resources.MemoryLease, "close", original_close)
        lease.close()

    assert policy.used_memory_bytes == 0


def test_compare_result_cleanup_handles_primary_before_payload_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "compare_result_primary_left")
    right = make_dataset_artifact_root(tmp_path, "compare_result_primary_right")
    frame = make_default_dataset_frame()
    write_dataset(
        left,
        frame,
        spec=make_processed_dataset_spec("compare_result_primary_left"),
    )
    write_dataset(
        right,
        frame,
        spec=make_processed_dataset_spec("compare_result_primary_right"),
    )
    policy = _issued_test_policy()
    primary = RuntimeError("comparison_result_primary")

    def raising_partitions():
        raise primary
        yield {}

    def fake_owned(*_args: object, resource_policy: RuntimeResourcePolicy):
        fingerprint = SimpleNamespace(
            schema_hash="schema",
            row_count=1,
            key_columns=("key",),
            content_hash="content",
            partition_fingerprints=raising_partitions(),
            file_count=1,
            column_names=("key",),
        )
        lease = resource_policy.lease(
            "test.comparison.fingerprint",
            128,
        )
        return runtime_resources.OwnedValue(fingerprint, lease)

    monkeypatch.setattr(
        dataset_comparison,
        "_fingerprint_dataset_session_owned",
        fake_owned,
    )
    monkeypatch.setattr(
        dataset_comparison,
        "_comparison_result_bound",
        lambda *_args: 128,
    )

    with pytest.raises(RuntimeError) as raised:
        dataset_comparison.compare_datasets(
            left,
            right,
            resource_policy=policy,
        )

    assert raised.value is primary
    assert policy.used_memory_bytes == 0


def test_compare_propagates_programmer_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "left_programmer_error"
    right = tmp_path / "right_programmer_error"
    filesystem_mkdir(left)
    filesystem_mkdir(right)
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
    fingerprint = dataset_comparison._fingerprint_dataset_session_owned
    calls = 0

    def changing_identity(context, **kwargs):
        nonlocal calls
        result = fingerprint(context, **kwargs)
        calls += 1
        if calls == 2:
            part = next(filesystem_glob(left / "data", "**/*.parquet"))
            stat = filesystem_stat(part)
            os.utime(
                native_filesystem_path(part),
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000),
            )
        return result

    monkeypatch.setattr(
        dataset_comparison, "_fingerprint_dataset_session_owned", changing_identity
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

    def index_then_mutate(context, **kwargs: object) -> None:
        index_rows(context, **kwargs)  # type: ignore[arg-type]
        if kwargs["namespace"] != mutated_side:
            return
        part = next(filesystem_glob(context.root / "data", "**/*.parquet"))
        stat = filesystem_stat(part)
        os.utime(
            native_filesystem_path(part),
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


@pytest.mark.parametrize("completion", ("exhausted", "early_close"))
def test_reader_final_verify_failure_propagates_without_external_primary(
    monkeypatch: pytest.MonkeyPatch,
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
            self.verify_fail_after = 1

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise OSError("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

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
    context = Context()
    iterator = dataset_reading.iter_dataset_file_batches(
        context,  # type: ignore[arg-type]
        columns=("value",),
        target_chunk_bytes=1024,
    )

    if completion == "exhausted":
        with pytest.raises(OSError, match="verify_failure"):
            list(iterator)
    else:
        next(iterator)
        with pytest.raises(OSError, match="verify_failure"):
            iterator.close()  # type: ignore[attr-defined]


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
        with closing_iterator(iterator):
            next(iterator)
            raise primary_type("primary_failure")


def test_exact_key_cursor_close_preserves_generator_primary() -> None:
    primary = RuntimeError("cursor_primary")

    class Cursor:
        def __iter__(self):
            raise primary

        def close(self) -> None:
            raise OSError("cursor_close_failure")

    class Connection:
        def execute(self, *_args):
            return Cursor()

    index = object.__new__(dataset_key_tracking.ExactPrimaryKeyIndex)
    index._connection = Connection()

    with pytest.raises(RuntimeError) as raised:
        next(
            dataset_key_tracking.ExactPrimaryKeyIndex.identity_payload_values(
                index,
                namespace="test",
            )
        )

    assert raised.value is primary
    assert raised.value.__notes__ == [
        "exact key cursor close also failed: OSError",
    ]


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
        explicit_cap_bytes=4 * 1024**2,
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
        comparison_reservations.append(policy.used_memory_bytes)
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

    comparison = dataset_comparison.compare_datasets(
        root,
        root,
        resource_policy=policy,
    )

    assert comparison.status == "MATCH"
    assert payload_targets
    assert len(reader_headrooms) == 2
    assert all(0 < target < policy.memory_budget_bytes for target in payload_targets)
    assert all(
        reserved >= target
        for reserved, target in zip(
            comparison_reservations,
            payload_targets,
            strict=True,
        )
    )
    assert all(reader > 0 for reader in reader_headrooms)
    assert value_columns and set(value_columns) == {("value",)}


def test_aggregate_chunks_stream_under_budget_without_terminal_materialization(
    tmp_path,
):
    _aggregate_streaming_case(tmp_path, tmpfs_failure=False)


def test_aggregate_chunks_tmpfs_growth_obeys_physical_capacity(tmp_path):
    _aggregate_streaming_case(tmp_path, tmpfs_failure=True)


def _aggregate_streaming_case(tmp_path: Path, *, tmpfs_failure: bool) -> None:
    from quant_research_platform.storage._scratch_ownership import _memory_filesystem

    if _memory_filesystem(tmp_path) != tmpfs_failure:
        pytest.skip(
            "requires actual tmpfs"
            if tmpfs_failure
            else "disk spill requires non-memory TMPDIR"
        )
    chunk_rows = 2_000
    chunk_count = 16
    total_rows = chunk_rows * chunk_count
    asset_suffix = "x" * 512
    chunks: list[pd.DataFrame] = []
    for start in range(0, total_rows, chunk_rows):
        positions = np.arange(start, start + chunk_rows, dtype=np.int64)
        chunks.append(
            pd.DataFrame(
                {
                    "market_id": ["a_share"] * chunk_rows,
                    "asset_id": [
                        f"asset_{value % 8:02d}_{asset_suffix}" for value in positions
                    ],
                    "timestamp": pd.date_range(
                        pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=int(start)),
                        periods=chunk_rows,
                        freq="min",
                    ),
                    "value": positions.astype("float64"),
                }
            )
        )
    explicit_cap_bytes = 16 * 1024**2

    def fresh_policy() -> RuntimeResourcePolicy:
        return capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=explicit_cap_bytes,
        )

    payloads = tuple(pandas_frame_payload_bytes(chunk) for chunk in chunks)
    assert all(payload < explicit_cap_bytes // 10 for payload in payloads)
    assert sum(payloads) > explicit_cap_bytes
    left = make_dataset_artifact_root(tmp_path / "left_parent", "aggregate_streaming")
    right = make_dataset_artifact_root(tmp_path / "right_parent", "aggregate_streaming")
    spec = make_processed_dataset_spec("aggregate_streaming")

    for root in (left, right):
        policy = fresh_policy()
        if tmpfs_failure:
            with pytest.raises(
                ValueError, match="storage_scratch_exceeds_memory_capacity"
            ):
                dataset_writing.write_dataset(
                    root, chunks, spec=spec, resource_policy=policy, borrowed_input=True
                )
            assert policy.used_memory_bytes == 0
            assert not filesystem_exists(root)
            return
        inspection = dataset_writing.write_dataset(
            root,
            chunks,
            spec=spec,
            resource_policy=policy,
            borrowed_input=True,
        )
        assert inspection.row_count == total_rows
        assert policy.used_memory_bytes == 0

    materialized_policy = fresh_policy()
    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="dataset_materialized_result_exceeds_target_bytes",
    ):
        dataset_reading.read_dataset(
            left,
            mode=DatasetReadMode.MATERIALIZED,
            resource_policy=materialized_policy,
        )
    assert materialized_policy.used_memory_bytes == 0

    streamed_policy = fresh_policy()
    streamed_rows = 0
    streamed = dataset_reading.read_dataset(
        left,
        mode=DatasetReadMode.CHUNKED,
        resource_policy=streamed_policy,
    )
    try:
        for chunk in streamed:
            streamed_rows += len(chunk)
            del chunk
    finally:
        close = getattr(streamed, "close", None)
        if close is not None:
            close()
    assert streamed_rows == total_rows
    assert streamed_policy.used_memory_bytes == 0

    inspections = []
    for root in (left, right):
        policy = fresh_policy()
        inspections.append(
            dataset_inspection.inspect_dataset(
                root,
                resource_policy=policy,
            )
        )
        assert policy.used_memory_bytes == 0
    left_inspection, right_inspection = inspections
    assert left_inspection.row_count == right_inspection.row_count == total_rows

    fingerprints = []
    for root in (left, right):
        policy = fresh_policy()
        fingerprints.append(
            dataset_fingerprinting.fingerprint_dataset(
                root,
                resource_policy=policy,
            )
        )
        assert policy.used_memory_bytes == 0
    left_fingerprint, right_fingerprint = fingerprints
    assert left_fingerprint == right_fingerprint

    comparison_policy = fresh_policy()
    comparison = dataset_comparison.compare_datasets(
        left,
        right,
        resource_policy=comparison_policy,
    )
    assert comparison.status == "MATCH"
    assert comparison_policy.used_memory_bytes == 0
