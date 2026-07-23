from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.storage._dataset_comparison as comparison
import quant_research_platform.storage._dataset_fingerprinting as fingerprinting
import quant_research_platform.storage._dataset_inspection as inspection
import quant_research_platform.storage._dataset_key_tracking as key_owner
import quant_research_platform.storage._dataset_physical as physical
import quant_research_platform.storage._dataset_reading as dataset_reading
import quant_research_platform.storage._dataset_writing as dataset_writing
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage import (
    _dataset_inventory as inventory_owner,
)
from quant_research_platform.storage import (
    atomic,
    manifests,
)
from quant_research_platform.storage._dataset_types import DatasetReadMode

_PRIMARY_EXCEPTION_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)

_STALE_SWEEP_FAILURE_TYPES = (
    OSError,
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
)


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": str(megabytes)},
    )


def test_chunked_read_rejects_different_explicit_policy_at_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _policy(1)
    different = _policy(2)
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_after_policy_conflict"),
    )

    with runtime_resource_policy_scope(active):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            dataset_reading.read_dataset(
                Path("unused"),
                mode=DatasetReadMode.CHUNKED,
                resource_policy=different,
            )


@pytest.mark.parametrize("explicit_mode", ("none", "same"))
def test_chunked_read_binds_outer_for_active_consumption(
    explicit_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    observed: list[RuntimeResourcePolicy] = []

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        observed.append(resource_policy)
        yield SimpleNamespace(resource_policy=resource_policy)

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter(()),
    )

    with runtime_resource_policy_scope(policy):
        chunks = dataset_reading.read_dataset(
            Path("unused"),
            mode=DatasetReadMode.CHUNKED,
            resource_policy=policy if explicit_mode == "same" else None,
        )
        assert list(chunks) == []

    assert observed == [policy]


@pytest.mark.parametrize("explicit_mode", ("none", "same"))
def test_chunked_read_rejects_delayed_consumption_after_outer_exit(
    explicit_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_for_retired_policy"),
    )

    with runtime_resource_policy_scope(policy):
        chunks = dataset_reading.read_dataset(
            Path("unused"),
            mode=DatasetReadMode.CHUNKED,
            resource_policy=policy if explicit_mode == "same" else None,
        )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        next(chunks)


@pytest.mark.parametrize(
    ("module", "owner", "body_name"),
    (
        (inspection, inspection.inspect_dataset, "inspect_dataset_session"),
        (
            fingerprinting,
            fingerprinting.fingerprint_dataset,
            "fingerprint_dataset_session",
        ),
    ),
)
def test_standalone_storage_root_preflights_before_admission_and_cleans_before_release(
    module,
    owner,
    body_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )

    class Session:
        resource_policy = policy

        def __enter__(self) -> Session:
            events.append("session-enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("cleanup")

    @contextmanager
    def scope(_policy):  # noqa: ANN001, ANN202
        events.append("admission")
        try:
            yield policy
        finally:
            events.append("release")

    def preflight(*_args, **_kwargs):  # noqa: ANN202
        events.append("preflight")
        return Session()

    monkeypatch.setattr(
        module,
        "active_runtime_resource_policy",
        lambda: pytest.fail("implicit_policy_issued_before_metadata_preflight"),
        raising=False,
    )

    @contextmanager
    def preflight_without_early_snapshot(*_args, **kwargs):  # noqa: ANN202
        assert kwargs["resource_policy"] is None
        session = preflight()
        with scope(policy):
            with session:
                yield session

    monkeypatch.setattr(
        module,
        "open_dataset_session",
        preflight_without_early_snapshot,
    )
    monkeypatch.setattr(
        module,
        body_name,
        lambda *_args, **_kwargs: events.append("body"),
    )

    owner(Path("dataset"), resource_policy=None)

    assert events == [
        "preflight",
        "admission",
        "session-enter",
        "body",
        "cleanup",
        "release",
    ]


@pytest.mark.parametrize(
    "mode",
    (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED),
)
def test_read_dataset_keeps_discovered_policy_active_through_session_exit(
    mode: DatasetReadMode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )
    competing = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )

    class Session:
        resource_policy = policy

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            assert current_runtime_resource_policy() is policy
            with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
                with runtime_resource_policy_scope(competing):
                    pass

    @contextmanager
    def open_session(*_args, **_kwargs):  # noqa: ANN202
        with runtime_resource_policy_scope(policy):
            with Session() as session:
                yield session

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "materialize_dataset_session",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(()),
    )

    result = dataset_reading.read_dataset(
        Path("dataset"),
        mode=mode,
        resource_policy=None,
    )
    if mode is DatasetReadMode.CHUNKED:
        list(result)


class _FailureBetweenValues:
    def __init__(
        self,
        first: Path,
        failure: BaseException,
        later: Path,
    ) -> None:
        self._values = iter((first, failure, later))

    def __iter__(self) -> _FailureBetweenValues:
        return self

    def __next__(self) -> Path:
        value = next(self._values)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.mark.parametrize(
    ("owner", "later_name"),
    (
        (inspection.inspect_dataset_session, "_inspection_runtime_metadata"),
        (fingerprinting.fingerprint_dataset_session, "iter_dataset_file_batches"),
    ),
)
def test_session_owner_rejects_policy_mismatch_before_storage_work(
    monkeypatch: pytest.MonkeyPatch,
    owner,
    later_name: str,
) -> None:
    session_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    explicit_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )
    module = (
        inspection if owner is inspection.inspect_dataset_session else fingerprinting
    )
    monkeypatch.setattr(
        module,
        later_name,
        lambda *_args, **_kwargs: pytest.fail("storage work reached"),
    )
    context = SimpleNamespace(resource_policy=session_policy)

    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        owner(context, resource_policy=explicit_policy)


class _BodyAndCloseFailingHandle:
    def __init__(self, primary: BaseException) -> None:
        self.primary = primary

    def __enter__(self) -> _BodyAndCloseFailingHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def fileno(self) -> int:
        return 1

    def read(self, *_args: object) -> bytes:
        raise self.primary

    def write(self, *_args: object) -> int:
        raise self.primary

    def flush(self) -> None:
        pass

    def close(self) -> None:
        raise OSError("close_secondary")


def _inventory_with_failing_connection(
    tmp_path: Path,
) -> inventory_owner.DatasetSessionInventory:
    class FailingConnection:
        def close(self) -> None:
            raise RuntimeError("close_failed")

    inventory = inventory_owner.DatasetSessionInventory(
        data_root=tmp_path / "data",
        scratch_parent=tmp_path / "scratch",
        task_memory_budget_bytes=1024,
    )
    assert inventory._connection is not None
    inventory._connection.close()
    inventory._connection = FailingConnection()  # type: ignore[assignment]
    return inventory


def test_inventory_exit_preserves_generator_exit_primary(tmp_path: Path) -> None:
    inventory = _inventory_with_failing_connection(tmp_path)

    with pytest.raises(GeneratorExit):
        with inventory:
            raise GeneratorExit()


def test_inventory_exit_propagates_cleanup_failure_without_primary(
    tmp_path: Path,
) -> None:
    inventory = _inventory_with_failing_connection(tmp_path)

    with pytest.raises(RuntimeError, match="close_failed"):
        with inventory:
            pass


def _patch_open_session_metadata_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, str], dict[str, str]]:
    spec = {"schema_id": "qrp-dataset-spec"}
    manifest = {"schema_id": "qrp-dataset-manifest"}
    monkeypatch.setattr(dataset_reading, "require_artifact_root", Path)
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    monkeypatch.setattr(Path, "is_dir", lambda _self: True)
    monkeypatch.setattr(
        dataset_reading,
        "read_bounded_qrp_json",
        lambda path: (b"{}", spec if Path(path).name == "spec.json" else manifest),
    )
    monkeypatch.setattr(
        dataset_reading,
        "validate_dataset_spec_payload",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "validate_dataset_manifest_payload",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "validate_dataset_manifest_consumable",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        dataset_reading,
        "manifest_has_runtime_metadata",
        lambda *_a, **_k: True,
    )
    return spec, manifest


def test_open_session_construction_cleanup_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )

    class Inventory:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs: object) -> None:
            assert current_runtime_resource_policy() is policy

        def discover(self, _spec: object) -> None:
            raise GeneratorExit()

        def close(self) -> None:
            raise RuntimeError("close_failed")

    _patch_open_session_metadata_preflight(monkeypatch)
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)

    with pytest.raises(GeneratorExit):
        with dataset_reading.open_dataset_session(
            Path("dataset"),
            columns=None,
            time_window=None,
            resource_policy=policy,
        ):
            pass


def test_open_session_keeps_exact_policy_active_through_owned_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )
    events: list[str] = []
    candidate = Path("dataset/data/year=2026/part-000.parquet")

    def record(event: str) -> None:
        assert current_runtime_resource_policy() is policy
        events.append(event)

    class Inventory:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs: object) -> None:
            record("inventory-init")

        def discover(self, _spec: object) -> None:
            record("discover")

        def path_count(self, *, selected: bool) -> int:
            assert selected is False
            assert current_runtime_resource_policy() is policy
            return 1

        def iter_candidate_paths(self):  # noqa: ANN202
            record("candidate-iteration")
            yield candidate

        def observe(self, path: Path, _metadata: object, *, selected: bool) -> None:
            assert path == candidate
            assert selected
            record("observe")

        def close(self) -> None:
            record("inventory-close")

    class Session:
        def __init__(self, inventory: Inventory) -> None:
            self.inventory = inventory

        def __enter__(self) -> Session:
            record("session-enter")
            return self

        def __exit__(self, *_args: object) -> None:
            record("session-exit")
            self.inventory.close()

    def inspect(path: Path) -> object:
        assert path == candidate
        record("inspect")
        return object()

    def prune(
        _data_root: Path,
        paths: tuple[Path, ...],
        **_kwargs: object,
    ) -> tuple[Path, ...]:
        assert paths == (candidate,)
        assert current_runtime_resource_policy() is policy
        return paths

    def build(**kwargs: object) -> Session:
        record("build")
        return Session(kwargs["inventory"])  # type: ignore[arg-type]

    _patch_open_session_metadata_preflight(monkeypatch)
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(dataset_reading, "inspect_dataset_part", inspect)
    monkeypatch.setattr(dataset_reading, "prune_paths_for_time_window", prune)
    monkeypatch.setattr(dataset_reading, "_build_dataset_session", build)

    with dataset_reading.open_dataset_session(
        Path("dataset"),
        columns=None,
        time_window=None,
        resource_policy=policy,
    ) as session:
        assert isinstance(session, Session)
        record("body")

    assert events == [
        "inventory-init",
        "discover",
        "candidate-iteration",
        "inspect",
        "observe",
        "build",
        "session-enter",
        "body",
        "session-exit",
        "inventory-close",
    ]


def test_open_session_policy_entry_failure_precedes_inventory_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )
    primary = RuntimeError("runtime_resource_policy_concurrent_root")
    inventory_init_calls = 0

    class Inventory:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal inventory_init_calls
            inventory_init_calls += 1

    @contextmanager
    def failing_scope(current: RuntimeResourcePolicy):
        assert current is policy
        raise primary
        yield

    _patch_open_session_metadata_preflight(monkeypatch)
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(dataset_reading, "runtime_resource_policy_scope", failing_scope)

    with pytest.raises(RuntimeError) as raised:
        with dataset_reading.open_dataset_session(
            Path("dataset"),
            columns=None,
            time_window=None,
            resource_policy=policy,
        ):
            pytest.fail("session body reached after policy entry failure")

    assert raised.value is primary
    assert inventory_init_calls == 0


def test_opened_parquet_close_preserves_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def fileno(self) -> int:
            return 1

    class Parquet:
        def close(self) -> None:
            raise RuntimeError("close_failed")

    metadata = type(
        "Stat",
        (),
        {"st_dev": 1, "st_ino": 2, "st_size": 3, "st_mtime_ns": 4},
    )()
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with pytest.raises(GeneratorExit):
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            raise GeneratorExit()


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_write_dataset_activates_and_closes_caller_iterator_under_policy(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )
    events: list[tuple[str, RuntimeResourcePolicy | None]] = []
    primary = primary_type("body_primary")

    class InputIterator:
        def __iter__(self) -> InputIterator:
            return self

        def __next__(self) -> pd.DataFrame:
            raise StopIteration

        def close(self) -> None:
            events.append(("close", current_runtime_resource_policy()))

    class CallerIterable:
        def __iter__(self) -> InputIterator:
            events.append(("iter", current_runtime_resource_policy()))
            return InputIterator()

    monkeypatch.setattr(dataset_writing, "validate_dataset_spec", lambda _spec: None)
    monkeypatch.setattr(
        dataset_writing,
        "validate_dataset_artifact_root",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        dataset_writing,
        "atomic_write_directory",
        lambda *_args: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(primary_type) as raised:
        dataset_writing.write_dataset(
            Path("dataset"),
            CallerIterable(),
            spec=object(),  # type: ignore[arg-type]
            resource_policy=policy,
        )

    assert raised.value is primary
    assert events == [("iter", policy), ("close", policy)]


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_opened_parquet_raw_handle_close_preserves_active_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("body_primary")

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return 1

        def close(self) -> None:
            raise OSError("raw_close_secondary")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with pytest.raises(primary_type) as raised:
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            raise primary

    assert raised.value is primary


def test_opened_parquet_raw_handle_close_propagates_without_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return 1

        def close(self) -> None:
            raise OSError("raw_close_failed")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with pytest.raises(OSError, match="raw_close_failed"):
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            pass


@pytest.mark.parametrize(
    "primary",
    (MemoryError(), KeyboardInterrupt(), SystemExit(), GeneratorExit()),
)
def test_atomic_directory_cleanup_preserves_active_primary(
    primary: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        atomic,
        "_remove_path",
        lambda _path: (_ for _ in ()).throw(RuntimeError("cleanup_failed")),
    )

    def writer(_path: Path) -> None:
        raise primary

    with pytest.raises(type(primary)):
        atomic.atomic_write_directory(tmp_path / "result", writer)


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_atomic_directory_attempts_all_cleanup_and_preserves_body_primary(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("body_primary")
    events: list[str] = []

    def remove_path(_path: Path) -> None:
        events.append("stage")
        raise OSError("stage_cleanup_secondary")

    original_unlink = Path.unlink

    def unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.endswith(".target"):
            events.append("marker")
            raise OSError("marker_cleanup_secondary")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(atomic, "_remove_path", remove_path)
    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(primary_type) as raised:
        atomic.atomic_write_directory(
            tmp_path / "result",
            lambda _path: (_ for _ in ()).throw(primary),
        )

    assert raised.value is primary
    assert events == ["stage", "marker"]


@pytest.mark.parametrize("orphan", (False, True), ids=("stale-stage", "orphan"))
def test_atomic_stale_marker_cleanup_failure_propagates_without_primary(
    orphan: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    stage = final_root.with_name(f"{prefix}owner-stale")
    marker = Path(f"{stage}.target")
    if not orphan:
        stage.mkdir()
    marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    original_unlink = Path.unlink

    def unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == marker:
            raise OSError("marker_cleanup_failed")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(OSError, match="marker_cleanup_failed"):
        atomic._cleanup_stale_atomic_directories(
            final_root,
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert not stage.exists()
    assert marker.exists()


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_atomic_stale_sweep_attempts_independent_cleanup_and_preserves_first_failure(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    first_stage = final_root.with_name(f"{prefix}first-stage")
    later_stage = final_root.with_name(f"{prefix}later-stage")
    first_marker = Path(f"{first_stage}.target")
    later_marker = Path(f"{later_stage}.target")
    first_orphan = final_root.with_name(f"{prefix}first-orphan.target")
    later_orphan = final_root.with_name(f"{prefix}later-orphan.target")
    for stage in (first_stage, later_stage):
        stage.mkdir()
    for marker in (first_marker, later_marker, first_orphan, later_orphan):
        marker.write_text(atomic._atomic_target_identity(final_root), encoding="utf-8")
    primary = primary_type("first_cleanup_failure")
    events: list[tuple[str, Path]] = []

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter((first_stage, later_stage))
        assert pattern == f"{prefix}*.target"
        return iter((first_orphan, later_orphan))

    def remove_path(path: Path) -> None:
        events.append(("stage", path))
        if path == first_stage:
            raise primary

    def remove_marker(path: Path) -> None:
        events.append(("marker", path))
        if path == first_orphan:
            raise OSError("later_cleanup_failure")

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(atomic, "_remove_path", remove_path)
    monkeypatch.setattr(atomic, "_remove_atomic_target_marker", remove_marker)

    with pytest.raises(primary_type) as raised:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert raised.value is primary
    assert events == [
        ("stage", first_stage),
        ("stage", later_stage),
        ("marker", later_marker),
        ("marker", first_orphan),
        ("marker", later_orphan),
    ]
    assert first_stage.exists()
    assert first_marker.exists()


@pytest.mark.parametrize("primary_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize(
    "later_failure_site",
    ("glob", "is_dir", "ownership", "iterdir", "is_file", "exists"),
)
def test_atomic_stale_sweep_preserves_first_failure_across_later_probes(
    primary_type: type[BaseException],
    later_failure_site: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    first_stage = final_root.with_name(f"{prefix}first")
    probe_stage = final_root.with_name(f"{prefix}probe")
    later_stage = final_root.with_name(f"{prefix}later")
    first_marker = Path(f"{first_stage}.target")
    probe_marker = Path(f"{probe_stage}.target")
    later_marker = Path(f"{later_stage}.target")
    probe_orphan = final_root.with_name(f"{prefix}probe-orphan.target")
    later_orphan = final_root.with_name(f"{prefix}later-orphan.target")
    primary = primary_type("first_failure")
    later_failure = RuntimeError("later_probe_failure")
    removed_stages: list[Path] = []
    removed_markers: list[Path] = []

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            if later_failure_site == "glob":
                return _FailureBetweenValues(first_stage, later_failure, later_stage)
            return iter((first_stage, probe_stage, later_stage))
        assert pattern == f"{prefix}*.target"
        if later_failure_site == "exists":
            return iter((probe_orphan, later_orphan))
        return iter(())

    def is_dir(path: Path) -> bool:
        if later_failure_site == "is_dir" and path == probe_stage:
            raise later_failure
        return path in {first_stage, probe_stage, later_stage}

    def is_stale(path: Path, *, prefix: str) -> bool:
        assert prefix
        if later_failure_site == "ownership" and path == probe_stage:
            raise later_failure
        return True

    def iterdir(path: Path):
        if later_failure_site == "iterdir" and path == probe_stage:
            raise later_failure
        return iter(())

    def is_file(path: Path) -> bool:
        if later_failure_site == "is_file" and path == probe_marker:
            raise later_failure
        return path in {first_marker, probe_marker, later_marker}

    def exists(path: Path) -> bool:
        if later_failure_site == "exists" and path == probe_orphan:
            raise later_failure
        return False

    def remove_path(path: Path) -> None:
        removed_stages.append(path)
        if path == first_stage:
            raise primary

    def remove_marker(path: Path) -> None:
        removed_markers.append(path)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", is_stale)
    monkeypatch.setattr(
        atomic,
        "_read_atomic_target_marker",
        lambda _path: atomic._atomic_target_identity(final_root),
    )
    monkeypatch.setattr(atomic, "_remove_path", remove_path)
    monkeypatch.setattr(atomic, "_remove_atomic_target_marker", remove_marker)

    with pytest.raises(primary_type) as raised:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert raised.value is primary
    assert later_stage in removed_stages
    if later_failure_site not in {"glob", "exists"}:
        assert probe_stage not in removed_stages
    if later_failure_site == "exists":
        assert later_orphan in removed_markers


@pytest.mark.parametrize("primary_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize(
    ("owner", "prefix"),
    (
        (inventory_owner._cleanup_stale_inventory, ".dataset-inventory-"),
        (key_owner._cleanup_stale_key_scratch, ".k-"),
    ),
)
@pytest.mark.parametrize("later_failure_site", ("glob", "is_dir", "ownership"))
def test_sqlite_stale_sweep_attempts_later_owned_candidates_after_failure(
    owner,
    prefix: str,
    later_failure_site: str,
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / f"{prefix}first"
    probe = tmp_path / f"{prefix}probe"
    later = tmp_path / f"{prefix}later"
    primary = primary_type("first_failure")
    later_failure = RuntimeError("later_probe_failure")
    removed: list[Path] = []

    def glob(_path: Path, pattern: str):
        assert pattern == f"{prefix}*"
        if later_failure_site == "glob":
            return _FailureBetweenValues(first, later_failure, later)
        return iter((first, probe, later))

    def is_dir(path: Path) -> bool:
        if later_failure_site == "is_dir" and path == probe:
            raise later_failure
        return True

    def is_stale(path: Path, *, prefix: str) -> bool:
        assert prefix
        if later_failure_site == "ownership" and path == probe:
            raise later_failure
        return True

    def rmtree(path: Path) -> None:
        removed.append(path)
        if path == first:
            raise primary

    module = (
        inventory_owner
        if owner is inventory_owner._cleanup_stale_inventory
        else key_owner
    )
    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr(module, "scratch_owner_is_stale", is_stale)
    monkeypatch.setattr(module.shutil, "rmtree", rmtree)

    with pytest.raises(primary_type) as raised:
        owner(tmp_path)

    assert raised.value is primary
    assert removed == [first, later]


@pytest.mark.parametrize("failure_type", _STALE_SWEEP_FAILURE_TYPES)
def test_atomic_stale_marker_probe_failure_propagates_after_attempting_later_stage(
    failure_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    unverified = final_root.with_name(f"{prefix}unverified")
    later = final_root.with_name(f"{prefix}later")
    unverified_marker = Path(f"{unverified}.target")
    later_marker = Path(f"{later}.target")
    failure = failure_type("marker_probe_failure")
    removed: list[Path] = []

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter((unverified, later))
        return iter(())

    def is_file(path: Path) -> bool:
        if path == unverified_marker:
            raise failure
        return path == later_marker

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "iterdir", lambda _path: iter(()))
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(
        atomic,
        "_read_atomic_target_marker",
        lambda _path: atomic._atomic_target_identity(final_root),
    )
    monkeypatch.setattr(atomic, "_remove_path", removed.append)
    monkeypatch.setattr(atomic, "_remove_atomic_target_marker", lambda _path: None)

    with pytest.raises(failure_type) as raised:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )

    assert raised.value is failure
    assert removed == [later]


@pytest.mark.parametrize("marker_failure_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize("earlier_failure", (False, True))
def test_atomic_stale_marker_read_failure_preserves_first_and_attempts_later_stage(
    marker_failure_type: type[BaseException],
    earlier_failure: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_root = tmp_path / "result"
    prefix = atomic._atomic_directory_scratch_prefix(final_root)
    first = final_root.with_name(f"{prefix}first")
    unverified = final_root.with_name(f"{prefix}unverified")
    later = final_root.with_name(f"{prefix}later")
    first_marker = Path(f"{first}.target")
    unverified_marker = Path(f"{unverified}.target")
    later_marker = Path(f"{later}.target")
    primary = OSError("first_removal_failure")
    marker_failure = marker_failure_type("marker_read_failure")
    attempts: list[Path] = []
    stages = (first, unverified, later) if earlier_failure else (unverified, later)

    def glob(_path: Path, pattern: str):
        if pattern == f"{prefix}*":
            return iter(stages)
        return iter(())

    def read_marker(path: Path) -> str:
        if path == unverified_marker:
            raise marker_failure
        assert path in {first_marker, later_marker}
        return atomic._atomic_target_identity(final_root)

    def remove_path(path: Path) -> None:
        attempts.append(path)
        if earlier_failure and path == first:
            raise primary

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "iterdir", lambda _path: iter(()))
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(atomic, "scratch_owner_is_stale", lambda *_a, **_k: True)
    monkeypatch.setattr(atomic, "_read_atomic_target_marker", read_marker)
    monkeypatch.setattr(atomic, "_remove_path", remove_path)
    monkeypatch.setattr(atomic, "_remove_atomic_target_marker", lambda _path: None)

    expected = primary if earlier_failure else marker_failure
    observed: BaseException | None = None
    try:
        atomic._cleanup_stale_atomic_directories(
            final_root,
            prefix=prefix,
            target_identity=atomic._atomic_target_identity(final_root),
        )
    except BaseException as raised:
        observed = raised

    assert observed is expected
    assert unverified not in attempts
    assert later in attempts


@pytest.mark.parametrize("failure_type", _STALE_SWEEP_FAILURE_TYPES)
@pytest.mark.parametrize(
    ("owner", "prefix"),
    (
        (inventory_owner._cleanup_stale_inventory, ".dataset-inventory-"),
        (key_owner._cleanup_stale_key_scratch, ".k-"),
    ),
)
def test_sqlite_stale_ownership_probe_failure_propagates_after_later_cleanup(
    owner,
    prefix: str,
    failure_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unverified = tmp_path / f"{prefix}unverified"
    later = tmp_path / f"{prefix}later"
    failure = failure_type("ownership_probe_failure")
    removed: list[Path] = []

    def is_stale(path: Path, *, prefix: str) -> bool:
        assert prefix
        if path == unverified:
            raise failure
        return True

    module = (
        inventory_owner
        if owner is inventory_owner._cleanup_stale_inventory
        else key_owner
    )
    monkeypatch.setattr(Path, "glob", lambda *_a, **_k: iter((unverified, later)))
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(module, "scratch_owner_is_stale", is_stale)
    monkeypatch.setattr(module.shutil, "rmtree", removed.append)

    with pytest.raises(failure_type) as raised:
        owner(tmp_path)

    assert raised.value is failure
    assert removed == [later]


def test_manifest_replace_does_not_check_exists_before_guarded_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "manifest.json"
    temp_path = final_path.with_name(f".tmp-{'c' * 12}")
    events: list[str] = []

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def write(self, _text: str) -> int:
            raise RuntimeError("body_primary")

    original_exists = Path.exists

    def exists(path: Path) -> bool:
        if path == temp_path:
            raise OSError("exists_secondary")
        return original_exists(path)

    def unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == temp_path:
            assert missing_ok is True
            events.append("unlink")
            raise OSError("unlink_secondary")
        raise AssertionError(f"unexpected_unlink: {path}")

    monkeypatch.setattr(
        manifests,
        "uuid4",
        lambda: type("Uuid", (), {"hex": "c" * 32})(),
    )
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Handle())
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "unlink", unlink)

    with pytest.raises(RuntimeError, match="body_primary"):
        manifests.replace_qrp_json(final_path, {"schema_id": "qrp-test"})

    assert events == ["unlink"]


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_bounded_manifest_read_preserves_body_primary_over_handle_close(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_id":"qrp-test"}', encoding="utf-8")
    metadata = path.stat()
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )
    monkeypatch.setattr(manifests.os, "fstat", lambda _fd: metadata)

    with pytest.raises(primary_type, match="body_primary"):
        manifests.read_bounded_qrp_json(path)


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_manifest_replace_preserves_body_primary_over_handle_close(
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )

    with pytest.raises(primary_type, match="body_primary"):
        manifests.replace_qrp_json(
            tmp_path / "manifest.json",
            {"schema_id": "qrp-test"},
        )


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
@pytest.mark.parametrize(
    "operation_name",
    ("text-write", "marker-write", "marker-read"),
)
def test_atomic_file_handles_preserve_body_primary_over_close(
    operation_name: str,
    primary_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker.target"
    if operation_name == "marker-read":
        marker.write_text("target", encoding="utf-8")
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )

    if operation_name == "text-write":
        def operation() -> object:
            return atomic.atomic_write_text(tmp_path / "output.txt", "value")
    elif operation_name == "marker-write":
        def operation() -> object:
            return atomic.atomic_write_directory(
                tmp_path / "output",
                lambda _path: None,
            )
    else:
        def operation() -> object:
            return atomic._read_atomic_target_marker(marker)

    with pytest.raises(primary_type, match="body_primary"):
        operation()


def test_compare_runtime_admission_failure_is_not_mapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    monkeypatch.setattr(
        comparison,
        "active_runtime_resource_policy",
        lambda: (_ for _ in ()).throw(ValueError("runtime_admission_failed")),
    )

    with pytest.raises(ValueError, match="runtime_admission_failed"):
        comparison.compare_datasets(left, right, resource_policy=None)


def test_compare_expected_storage_validation_still_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "1"},
    )
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )

    assert comparison.compare_datasets(
        left, right, resource_policy=policy
    ).status == "INSPECTION_FAILED"
