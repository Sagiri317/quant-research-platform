from __future__ import annotations

import os
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.runtime.resources as runtime_resources
import quant_research_platform.storage.datasets.comparison as dataset_comparison
import quant_research_platform.storage.datasets.fingerprinting as dataset_fingerprinting
import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.inventory as dataset_inventory
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.schema as dataset_schema
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.exact_keys as dataset_key_tracking
import quant_research_platform.storage.manifests as storage_manifests
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    read_dataset,
    write_dataset,
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
    monkeypatch.setattr(
        dataset_reading,
        "current_runtime_resource_policy",
        lambda: None,
    )
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


def test_public_chunked_read_rejects_root_appearing_after_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _issued_test_policy()
    chunks = read_dataset(Path("unused"), mode=DatasetReadMode.CHUNKED)
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
            next(chunks)


def test_chunked_read_defers_session_inventory_until_activation(
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
        resource_policy=policy,
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
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with runtime_resource_policy_scope(policy):
        inventory = dataset_inventory.DatasetSessionInventory(
            data_root=data_root,
            scratch_parent=tmp_path,
            resource_policy=policy,
        )
        scratch = inventory.scratch_path
        assert (scratch / "i.db").is_file()
        assert inventory._connection is not None
        assert inventory._connection.execute("PRAGMA cache_size").fetchone() == (-1,)
        inventory.close()
        assert policy.used_memory_bytes == 0
    assert not scratch.exists()
    assert not list(tmp_path.glob(".i-*"))
    assert not list(tmp_path.rglob("i.db-wal"))
    assert not list(tmp_path.rglob("i.db-shm"))


def test_dataset_inventory_low_budget_and_inspection_failure_clean_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    low_policy = capture_test_policy(
        available_memory_bytes=64 * 1024,
        explicit_cap_bytes=1024,
    )
    with (
        runtime_resource_policy_scope(low_policy),
        pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"),
    ):
        dataset_inventory.DatasetSessionInventory(
            data_root=data_root,
            scratch_parent=tmp_path,
            resource_policy=low_policy,
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
        expected = [f"part-{part_index}.parquet" for part_index in part_indices]
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
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with (
        runtime_resource_policy_scope(policy),
        dataset_inventory.DatasetSessionInventory(
            data_root=tmp_path / "data",
            scratch_parent=tmp_path,
            resource_policy=policy,
        ) as inventory,
    ):
        connection = inventory._require_connection()
        with connection:
            inventory._gate_physical_write(
                logical_bytes=500_000,
                row_count=5_000,
            )
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
            assert (
                inventory.path_count(
                    selected=True,
                    path_selection=selection,
                )
                == 1
            )
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


def test_dataset_inventory_does_not_claim_all_free_disk_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = 64 * dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
    reserve_calls: list[int] = []

    def free_bytes(_path: Path, *, reserve_bytes: int) -> int:
        reserve_calls.append(reserve_bytes)
        return available

    monkeypatch.setattr(
        dataset_inventory,
        "require_free_disk_headroom",
        free_bytes,
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with (
        runtime_resource_policy_scope(policy),
        dataset_inventory.DatasetSessionInventory(
            data_root=tmp_path / "data",
            scratch_parent=tmp_path,
            resource_policy=policy,
        ) as inventory,
    ):
        assert inventory._connection is not None
        page_limit = inventory._connection.execute(
            "PRAGMA main.max_page_count"
        ).fetchone()[0]
        assert page_limit == (
            dataset_inventory._DATASET_INVENTORY_SQLITE_STARTUP_RESERVE_BYTES
            // dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
        )
        assert (
            page_limit < available // dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
        )

    assert reserve_calls[0] >= 32 * dataset_inventory.DATASET_INVENTORY_RESERVE_BYTES
    assert len(reserve_calls) == 1


def test_exact_key_index_does_not_claim_all_free_disk_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = 64 * dataset_key_tracking.EXACT_KEY_SQLITE_PAGE_BYTES
    monkeypatch.setattr(
        dataset_key_tracking,
        "require_free_disk_headroom",
        lambda *_args, **_kwargs: available,
    )

    with dataset_key_tracking.ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        assert index._connection is not None
        page_limit = index._connection.execute("PRAGMA main.max_page_count").fetchone()[
            0
        ]
        assert page_limit == (
            dataset_key_tracking._EXACT_KEY_SQLITE_STARTUP_RESERVE_BYTES
            // dataset_key_tracking.EXACT_KEY_SQLITE_PAGE_BYTES
        )
        assert (
            page_limit < available // dataset_key_tracking.EXACT_KEY_SQLITE_PAGE_BYTES
        )


def test_exact_key_index_admits_combined_main_and_temp_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_calls: list[int] = []
    monkeypatch.setattr(
        dataset_key_tracking,
        "require_free_disk_headroom",
        lambda _path, *, reserve_bytes: reserve_calls.append(reserve_bytes),
    )

    with dataset_key_tracking.ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ) as index:
        index._admit_physical_write(4096)

    assert reserve_calls[-1] == 8192


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
        assert (
            tuple(session.inventory.iter_paths(selected=True, path_selection=selection))
            == paths
        )
        monkeypatch.setattr(
            dataset_inventory,
            "require_free_disk_headroom",
            lambda *_args, **_kwargs: None,
        )
        session.inventory.release_path_selection(selection)
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=selection,
            )
            == 0
        )


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

    def fail_first_row(_path: Path, *, reserve_bytes: int) -> int:
        nonlocal reserve_calls
        reserve_calls += 1
        assert reserve_bytes > 0
        if reserve_calls > 1:
            raise OSError("inventory_disk_full")
        return 1024**3

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
    policy = capture_test_policy(available_memory_bytes=64 * 1024)

    with (
        runtime_resource_policy_scope(policy),
        pytest.raises(RuntimeError, match="inventory_init_failed"),
    ):
        dataset_inventory.DatasetSessionInventory(
            data_root=data_root,
            scratch_parent=tmp_path,
            resource_policy=policy,
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
        resource_policy=capture_test_policy(
            available_memory_bytes=16 * 1024**2,
        ),
        read_paths=(path, empty_path),
    )

    runtime = dataset_inspection._inspection_runtime_metadata(
        context,
        target_chunk_bytes=1024,
    )

    assert runtime["file_count"] == 1
    assert runtime["row_count"] == 2
    assert observed_headroom == [
        context.resource_policy.memory_budget_bytes
        - context.resource_policy.used_memory_bytes
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
    with runtime_resource_policy_scope(policy):
        retained_lease = policy.lease(
            "test.inspection.retained",
            policy.memory_budget_bytes
            - dataset_inspection.EXACT_KEY_SQLITE_CACHE_BYTES
            - accumulated_state_bytes
            - 256,
        )
        try:
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                dataset_inspection.inspect_dataset_session(
                    context,
                    resource_policy=policy,
                )
        finally:
            retained_lease.close()

    assert traversals == 0


def test_inspection_admits_result_before_result_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "inspection_result_admission")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("inspection_result_admission"),
    )
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    before_runtime: list[int] = []
    after_runtime: list[int] = []
    at_construction: list[int] = []
    original_runtime = dataset_inspection._inspection_runtime_metadata
    original_type = dataset_inspection.DatasetInspection

    def observe_runtime(*args: object, **kwargs: object) -> dict[str, object]:
        runtime = original_runtime(*args, **kwargs)
        after_runtime.append(policy.used_memory_bytes)
        return runtime

    def observe_construction(*args: object, **kwargs: object):  # noqa: ANN202
        at_construction.append(policy.used_memory_bytes)
        return original_type(*args, **kwargs)

    monkeypatch.setattr(
        dataset_inspection,
        "_inspection_runtime_metadata",
        observe_runtime,
    )
    monkeypatch.setattr(dataset_inspection, "DatasetInspection", observe_construction)

    with runtime_resource_policy_scope(policy):
        before_runtime.append(policy.used_memory_bytes)
        result = dataset_inspection.inspect_dataset(root, resource_policy=policy)

    assert result.row_count == len(make_default_dataset_frame())
    assert len(after_runtime) == len(at_construction) == 1
    assert after_runtime[0] > before_runtime[0]
    assert (
        after_runtime[0] - before_runtime[0]
        >= dataset_inspection.EXACT_KEY_SQLITE_CACHE_BYTES
    )
    assert at_construction[0] == after_runtime[0]
    assert policy.used_memory_bytes == 0


def test_fingerprint_admits_result_before_result_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint_result_admission")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("fingerprint_result_admission"),
    )
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    before_bound: list[int] = []
    after_bound: list[int] = []
    at_construction: list[int] = []
    original_bound = dataset_fingerprinting._fingerprint_retained_memory_bytes
    original_type = dataset_fingerprinting.DatasetFingerprint
    calls = 0

    def observe_bound(**kwargs: object) -> int:
        nonlocal calls
        calls += 1
        bound = original_bound(**kwargs)  # type: ignore[arg-type]
        if calls == 1:
            after_bound.append(policy.used_memory_bytes)
        return bound

    def observe_construction(*args: object, **kwargs: object):  # noqa: ANN202
        at_construction.append(policy.used_memory_bytes)
        return original_type(*args, **kwargs)

    monkeypatch.setattr(
        dataset_fingerprinting,
        "_fingerprint_retained_memory_bytes",
        observe_bound,
    )
    monkeypatch.setattr(
        dataset_fingerprinting,
        "DatasetFingerprint",
        observe_construction,
    )

    with runtime_resource_policy_scope(policy):
        before_bound.append(policy.used_memory_bytes)
        result = dataset_fingerprinting.fingerprint_dataset(
            root,
            resource_policy=policy,
        )

    assert result.row_count == len(make_default_dataset_frame())
    assert len(after_bound) == len(at_construction) == 1
    assert after_bound[0] > before_bound[0]
    assert at_construction[0] == after_bound[0]
    assert policy.used_memory_bytes == 0


def test_private_writer_owned_result_retains_one_lease(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "writer_owned_result")
    policy = _writer_test_policy()

    with runtime_resource_policy_scope(policy):
        owned = dataset_writing._write_dataset_owned(
            root,
            _bound_first_frames((make_default_dataset_frame(),)),
            spec=make_processed_dataset_spec("writer_owned_result"),
        )
        assert isinstance(owned, runtime_resources.OwnedValue)
        assert owned.lease.active
        assert owned.value.root == root
        assert policy.used_memory_bytes >= owned.lease.reserved_bytes
        owned.value = None
        owned.lease.close()

    assert policy.used_memory_bytes == 0


def test_private_writer_releases_result_when_publish_loses_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "publish_race")
    policy = _writer_test_policy()
    real_atomic_write_directory = dataset_writing.atomic_write_directory

    def publish_after_writer(final_root, writer):  # noqa: ANN001
        def race_writer(temp_root):  # noqa: ANN001
            result = writer(temp_root)
            final_root.mkdir()
            (final_root / "winner").write_text("winner", encoding="utf-8")
            return result

        return real_atomic_write_directory(final_root, race_writer)

    monkeypatch.setattr(
        dataset_writing,
        "atomic_write_directory",
        publish_after_writer,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="dataset_exists") as raised:
            dataset_writing.write_dataset(
                root,
                _bound_first_frames((make_default_dataset_frame(),)),
                spec=make_processed_dataset_spec("publish_race"),
                resource_policy=policy,
            )
        assert policy.used_memory_bytes == 0
        assert policy.state == "ACTIVE"
        assert "runtime memory lease leaked" not in getattr(
            raised.value,
            "__notes__",
            (),
        )

    assert (root / "winner").read_text(encoding="utf-8") == "winner"
    assert not list(tmp_path.glob(".a-*"))


def test_compare_uses_one_fingerprint_lease_per_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "compare_lease_left")
    right = make_dataset_artifact_root(tmp_path, "compare_lease_right")
    frame = make_default_dataset_frame()
    write_dataset(left, frame, spec=make_processed_dataset_spec("compare_lease_left"))
    write_dataset(
        right,
        frame,
        spec=make_processed_dataset_spec("compare_lease_right"),
    )
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    observed: list[int] = []
    identity_reserved: list[int] = []
    original_exact = dataset_comparison._exact_comparison_counts
    original_identity = dataset_comparison._session_identity_token

    def observe_exact(*args: object, **kwargs: object) -> dict[str, int]:
        assert "retained_payload_bytes" not in kwargs
        observed.append(policy.used_memory_bytes)
        return original_exact(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dataset_comparison, "_exact_comparison_counts", observe_exact)

    def observe_identity(context: object) -> bytes:
        identity_reserved.append(policy.used_memory_bytes)
        return original_identity(context)  # type: ignore[arg-type]

    monkeypatch.setattr(
        dataset_comparison,
        "_session_identity_token",
        observe_identity,
    )

    with runtime_resource_policy_scope(policy):
        comparison = dataset_comparison.compare_datasets(
            left,
            right,
            resource_policy=policy,
        )

    assert comparison.status == "MATCH"
    assert observed and observed[0] > 0
    assert len(identity_reserved) >= 4
    assert all(value > 0 for value in identity_reserved)
    assert policy.used_memory_bytes == 0


def test_bounded_metadata_reader_accepts_single_external_hardlink(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.json"
    external.write_text('{"schema_id":"outside"}', encoding="utf-8")
    candidate = tmp_path / "manifest.json"
    try:
        os.link(external, candidate)
    except OSError:
        pytest.skip("hardlink_not_available")
    assert storage_manifests.read_bounded_qrp_json(candidate)[1] == {
        "schema_id": "outside"
    }


def test_private_writer_baseline_rejects_before_input_generation(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=(
            2 * dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
        ),
    )
    generated = False

    def chunks():
        nonlocal generated
        generated = True
        yield make_default_dataset_frame()

    with runtime_resource_policy_scope(policy):
        retained_lease = policy.lease(
            "test.writer.retained",
            policy.memory_budget_bytes
            - dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES,
        )
        try:
            with pytest.raises(
                ValueError,
                match="dataset_working_set_exceeds_task_memory_budget",
            ):
                dataset_writing.write_dataset(
                    make_dataset_artifact_root(tmp_path, "retained_baseline"),
                    chunks(),
                    spec=make_processed_dataset_spec("retained_baseline"),
                    resource_policy=policy,
                )
        finally:
            retained_lease.close()

    assert not generated


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
    )
    assert prepared.values.to_pylist() == sorted(set(values))


def test_writer_releases_previous_publication_before_requesting_next(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first_reference = weakref.ref(first)

    def publications():
        nonlocal first
        lease = yield 256 * 1024
        current = first
        first = None
        yield runtime_resources.OwnedValue(current, lease)
        del current
        assert first_reference() is None
        lease = yield 256 * 1024
        yield runtime_resources.OwnedValue(second, lease)

    name = "publication_liveness"
    dataset_writing.write_dataset(
        make_dataset_artifact_root(tmp_path, name),
        publications(),
        spec=make_processed_dataset_spec(name),
        resource_policy=policy,
    )
