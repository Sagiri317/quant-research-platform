from __future__ import annotations

import io
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.runtime.resources as runtime_resources
import quant_research_platform.storage.datasets.comparison as comparison
import quant_research_platform.storage.datasets.fingerprinting as fingerprinting
import quant_research_platform.storage.datasets.inspection as inspection
import quant_research_platform.storage.datasets.inventory as inventory_owner
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.exact_keys as key_owner
import quant_research_platform.storage.identity as identity
import quant_research_platform.storage.parquet as physical
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage import _scratch_ownership as scratch_owner
from quant_research_platform.storage import (
    atomic,
    manifests,
)
from quant_research_platform.storage.datasets.types import DatasetReadMode
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)

_PRIMARY_EXCEPTION_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


def _open_inventory_for_failure(data_root: Path, scratch_parent: Path):
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with runtime_resource_policy_scope(policy):
        return inventory_owner.DatasetSessionInventory(
            data_root=data_root,
            scratch_parent=scratch_parent,
            resource_policy=policy,
        )


@pytest.mark.parametrize("long_path", (False, True))
def test_shared_path_move_is_fail_if_destination_exists(
    tmp_path: Path, long_path
) -> None:
    shared = __import__(
        "quant_research_platform.storage._scratch_ownership", fromlist=["*"]
    )
    parent = tmp_path

    def native(path):
        return Path(shared._windows_extended_path(path)) if os.name == "nt" else path

    if long_path:
        parent = parent / ("nested" * 20) / ("nested" * 20)
        native(parent).mkdir(parents=True)
    source = parent / "source"
    destination = parent / "destination"
    native(source).write_text("source", encoding="utf-8")
    native(destination).write_text("destination", encoding="utf-8")

    with pytest.raises(FileExistsError):
        shared.move_path_no_replace(source, destination)

    assert native(source).read_text(encoding="utf-8") == "source"
    assert native(destination).read_text(encoding="utf-8") == "destination"
    moved = parent / "moved"
    shared.move_path_no_replace(source, moved)
    assert native(moved).read_text(encoding="utf-8") == "source"
    assert not native(source).exists()


def test_write_and_scratch_owners_reject_aliased_parent_before_access(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    operations = (
        lambda: atomic.atomic_write_text(alias / "text.txt", "payload"),
        lambda: atomic.atomic_write_directory(alias / "tree", lambda _root: None),
        lambda: manifests.replace_qrp_json(
            alias / "manifest.json", {"schema_id": "qrp-test"}
        ),
        lambda: key_owner.ExactPrimaryKeyIndex(
            primary_key=("value",),
            timestamp_columns=(),
            scratch_parent=alias,
        ).__enter__(),
        lambda: _open_inventory_for_failure(tmp_path / "data", alias),
    )

    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    assert not tuple(target.iterdir())


@pytest.mark.parametrize(
    ("owner_module", "open_owner"),
    (
        (
            key_owner,
            lambda scratch: key_owner.ExactPrimaryKeyIndex(
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=scratch,
            ).__enter__(),
        ),
        (
            inventory_owner,
            lambda scratch: _open_inventory_for_failure(
                scratch.parent / "data",
                scratch,
            ),
        ),
    ),
    ids=["exact-primary-key-index", "dataset-session-inventory"],
)
def test_scratch_owner_cleans_retained_temp_after_post_create_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_module: object,
    open_owner,
) -> None:
    scratch = tmp_path / "scratch"
    created: list[Path] = []
    gate_failure = KeyboardInterrupt("post_create_gate")

    post_create_calls = 0
    real_gate = owner_module.require_unaliased_path

    def gate(path: Path, *, error: str) -> Path:
        nonlocal post_create_calls
        candidate = Path(path)
        if candidate.parent == scratch and candidate.name.startswith("."):
            if not created:
                created.append(candidate)
            post_create_calls += 1
            if post_create_calls == 1:
                raise gate_failure
        return real_gate(candidate, error=error)

    monkeypatch.setattr(owner_module, "require_unaliased_path", gate)

    with pytest.raises(KeyboardInterrupt) as caught:
        open_owner(scratch)

    assert caught.value is gate_failure
    assert len(created) == 1
    assert not created[0].exists()


@pytest.mark.parametrize(
    ("owner_module", "open_owner"),
    (
        (
            key_owner,
            lambda scratch: key_owner.ExactPrimaryKeyIndex(
                primary_key=("value",),
                timestamp_columns=(),
                scratch_parent=scratch,
            ).__enter__(),
        ),
        (
            inventory_owner,
            lambda scratch: _open_inventory_for_failure(
                scratch.parent / "data",
                scratch,
            ),
        ),
    ),
    ids=("exact-primary-key-index", "dataset-session-inventory"),
)
@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt))
def test_scratch_owner_cleans_when_database_constructor_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_module: object,
    open_owner,
    primary_type: type[BaseException],
) -> None:
    scratch = tmp_path / "scratch"
    primary = primary_type("constructor_primary")
    database_paths: list[Path] = []

    def fail_connect(database_path: Path, *_args, **_kwargs):
        database_paths.append(Path(database_path))
        raise primary

    monkeypatch.setattr(owner_module.sqlite3, "connect", fail_connect)

    with pytest.raises(primary_type) as caught:
        open_owner(scratch)

    assert caught.value is primary
    assert len(database_paths) == 1
    assert not database_paths[0].parent.exists()


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=megabytes * 1024 * 1024,
    )


def test_unaliased_path_checks_lexical_component_before_parent_traversal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    alias = tmp_path / "alias"
    visited: list[Path] = []

    def is_alias(path: Path) -> bool:
        visited.append(path)
        return path == alias

    monkeypatch.setattr(identity, "path_is_filesystem_alias", is_alias)

    with pytest.raises(ValueError, match="invalid_path"):
        identity.require_unaliased_path(
            alias / ".." / "artifact",
            error="invalid_path",
        )

    assert alias in visited


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


def test_chunked_read_binds_explicit_outer_for_active_consumption(
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
            resource_policy=policy,
        )
        assert list(chunks) == []

    assert observed == [policy]


def test_chunked_read_rejects_delayed_consumption_after_outer_exit(
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
            resource_policy=policy,
        )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        next(chunks)


def test_chunked_read_rejects_implicit_active_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_reading,
        "open_dataset_session",
        lambda *_args, **_kwargs: pytest.fail("session_opened_after_competing_root"),
    )

    with runtime_resource_policy_scope(_policy(1)):
        with pytest.raises(
            RuntimeError,
            match="runtime_resource_policy_competing_root",
        ):
            dataset_reading.read_dataset(
                Path("unused"),
                mode=DatasetReadMode.CHUNKED,
                resource_policy=None,
            )


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
    policy = capture_storage_test_policy(
        working_headroom_bytes=1024**2,
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
        "current_runtime_resource_policy",
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
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    competing = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
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
    session_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    explicit_policy = capture_test_policy(
        available_memory_bytes=1024**3,
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


@contextmanager
def _inventory_with_failing_connection(
    tmp_path: Path,
):
    class FailingConnection:
        def close(self) -> None:
            raise RuntimeError("close_failed")

    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with runtime_resource_policy_scope(policy):
        with inventory_owner.DatasetSessionInventory(
            data_root=tmp_path / "data",
            scratch_parent=tmp_path / "scratch",
            resource_policy=policy,
        ) as inventory:
            connection = inventory._connection
            assert connection is not None
            connection.close()
            inventory._connection = FailingConnection()  # type: ignore[assignment]
            yield inventory


def test_inventory_exit_preserves_generator_exit_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
    with pytest.raises(GeneratorExit):
        with _inventory_with_failing_connection(tmp_path):
            raise GeneratorExit()


def test_inventory_exit_propagates_cleanup_failure_without_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
    with pytest.raises(RuntimeError, match="close_failed"):
        with _inventory_with_failing_connection(tmp_path):
            pass


def _patch_open_session_metadata_preflight(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    spec = {"schema_id": "qrp-dataset-spec"}
    manifest = {"schema_id": "qrp-dataset-manifest"}
    (root / "data").mkdir(parents=True)
    (root / "_manifest").mkdir()
    (root / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (root / "_manifest" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
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
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    class Inventory:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs: object) -> None:
            assert current_runtime_resource_policy() is policy

        def discover(self, _spec: object) -> None:
            raise GeneratorExit()

        def close(self) -> None:
            raise RuntimeError("close_failed")

    _patch_open_session_metadata_preflight(monkeypatch, tmp_path / "dataset")
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)

    with pytest.raises(GeneratorExit):
        with dataset_reading.open_dataset_session(
            tmp_path / "dataset",
            columns=None,
            time_window=None,
            resource_policy=policy,
        ):
            pass


def test_open_session_build_rejection_closes_lease_inside_same_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    candidate = tmp_path / "dataset/data/part-000.parquet"

    class Inventory:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs: object) -> None:
            return None

        def discover(self, _spec: object) -> None:
            return None

        def path_count(self, *, selected: bool) -> int:
            assert selected is False
            return 1

        def iter_candidate_paths(self):  # noqa: ANN202
            yield candidate

        def observe(self, *_args: object, **_kwargs: object) -> None:
            return None

        def close(self) -> None:
            return None

    _patch_open_session_metadata_preflight(monkeypatch, tmp_path / "dataset")
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(
        dataset_reading,
        "inspect_dataset_part",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        dataset_reading,
        "prune_paths_for_time_window",
        lambda *_args, **_kwargs: (candidate,),
    )
    monkeypatch.setattr(
        dataset_reading,
        "_build_dataset_session",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("session_build_failed")),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="session_build_failed"):
            with dataset_reading.open_dataset_session(
                tmp_path / "dataset",
                columns=None,
                time_window=None,
                resource_policy=policy,
            ):
                pass
        assert policy.used_memory_bytes == 0


def test_session_from_inventory_build_rejection_closes_lease_inside_same_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    class Inventory:
        def verify_filesystem(self, _spec: object) -> None:
            return None

    _patch_open_session_metadata_preflight(monkeypatch, tmp_path / "dataset")
    monkeypatch.setattr(
        dataset_reading,
        "_build_dataset_session",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("session_build_failed")),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="session_build_failed"):
            dataset_reading.dataset_session_from_inventory(
                tmp_path / "dataset",
                inventory=Inventory(),  # type: ignore[arg-type]
                columns=None,
                time_window=None,
                resource_policy=policy,
            )
        assert policy.used_memory_bytes == 0


def test_open_session_keeps_exact_policy_active_through_owned_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    events: list[str] = []
    candidate = tmp_path / "dataset/data/year=2026/part-000.parquet"

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
        def __init__(self, inventory: Inventory, lease) -> None:  # noqa: ANN001
            self.inventory = inventory
            self.lease = lease
            self.closed = False

        def __enter__(self) -> Session:
            record("session-enter")
            return self

        def __exit__(self, *_args: object) -> None:
            record("session-exit")
            self.close()

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            self.inventory.close()
            self.lease.close()

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
        return Session(kwargs["inventory"], kwargs["session_lease"])  # type: ignore[arg-type]

    _patch_open_session_metadata_preflight(monkeypatch, tmp_path / "dataset")
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(dataset_reading, "inspect_dataset_part", inspect)
    monkeypatch.setattr(dataset_reading, "prune_paths_for_time_window", prune)
    monkeypatch.setattr(dataset_reading, "_build_dataset_session", build)

    with dataset_reading.open_dataset_session(
        tmp_path / "dataset",
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
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
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

    _patch_open_session_metadata_preflight(monkeypatch, tmp_path / "dataset")
    monkeypatch.setattr(dataset_reading, "DatasetSessionInventory", Inventory)
    monkeypatch.setattr(dataset_reading, "runtime_resource_policy_scope", failing_scope)

    with pytest.raises(RuntimeError) as raised:
        with dataset_reading.open_dataset_session(
            tmp_path / "dataset",
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
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def fileno(self) -> int:
            return 1

        def tell(self) -> int:
            return getattr(self, "position", 0)

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = 8 + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return b"\0\0\0\0PAR1"

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
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle: Parquet())

    with runtime_resource_policy_scope(policy):
        with pytest.raises(GeneratorExit):
            with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
                raise GeneratorExit()
    assert policy.used_memory_bytes == 0


def test_opened_parquet_reserves_metadata_for_lifetime_and_scales_with_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def admitted_bound(footer_length: int) -> int:
        policy = capture_test_policy(
            available_memory_bytes=16 * 1024**2,
            explicit_cap_bytes=8 * 1024 * 1024,
        )
        events: list[tuple[str, int]] = []
        trailer = footer_length.to_bytes(4, "little") + b"PAR1"

        class Handle:
            size = footer_length + 8

            def __init__(self) -> None:
                self.position = 0
                self.closed = False

            def fileno(self) -> int:
                return 1

            def tell(self) -> int:
                return self.position

            def seek(self, offset: int, whence: int = 0) -> int:
                if whence == 2:
                    self.position = self.size + offset
                elif whence == 0:
                    self.position = offset
                else:
                    self.position += offset
                return self.position

            def read(self, _size: int = -1) -> bytes:
                return trailer

            def close(self) -> None:
                self.closed = True

        handle = Handle()
        metadata = SimpleNamespace(
            st_dev=1,
            st_ino=2,
            st_size=handle.size,
            st_mtime_ns=4,
        )
        monkeypatch.setattr(Path, "open", lambda *_a, **_k: handle)
        monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
        monkeypatch.setattr(
            physical,
            "dataset_file_identity",
            lambda _path: identity.dataset_file_identity_from_stat(metadata),
        )

        class Parquet:
            def close(self) -> None:
                events.append(("close", policy.used_memory_bytes))

        def construct(_handle: object) -> Parquet:
            events.append(("construct", policy.used_memory_bytes))
            return Parquet()

        monkeypatch.setattr(physical.pq, "ParquetFile", construct)

        with runtime_resource_policy_scope(policy):
            with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
                events.append(("body", policy.used_memory_bytes))

        assert [event[1] for event in events] == [events[0][1]] * 3
        assert events[0][1] > 0
        assert handle.closed
        assert policy.used_memory_bytes == 0
        return events[0][1]

    assert admitted_bound(2048) > admitted_bound(1024)


def test_opened_parquet_rejects_metadata_before_constructor_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    footer_length = 9 * 1024 * 1024
    trailer = footer_length.to_bytes(4, "little") + b"PAR1"
    constructor_calls = 0

    class Handle:
        size = footer_length + 8

        def __init__(self) -> None:
            self.position = 0
            self.closed = False

        def fileno(self) -> int:
            return 1

        def tell(self) -> int:
            return self.position

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = self.size + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return trailer

        def close(self) -> None:
            self.closed = True

    handle = Handle()
    metadata = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_size=handle.size,
        st_mtime_ns=4,
    )
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: handle)
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )

    def construct(_handle: object) -> object:
        nonlocal constructor_calls
        constructor_calls += 1
        return object()

    monkeypatch.setattr(physical.pq, "ParquetFile", construct)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
                pass

    assert constructor_calls == 0
    assert handle.closed
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "payload",
    (b"\0\0\0\0NOPE", (9).to_bytes(4, "little") + b"PAR1"),
)
def test_parquet_footer_rejects_invalid_trailer_without_constructor(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError, match="dataset_parquet_footer_invalid"):
        physical._parquet_footer_length(io.BytesIO(payload))


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_write_dataset_activates_and_closes_caller_iterator_under_policy(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_storage_test_policy(
        working_headroom_bytes=1024**2,
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

        def tell(self) -> int:
            return getattr(self, "position", 0)

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = 8 + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return b"\0\0\0\0PAR1"

        def close(self) -> None:
            raise OSError("raw_close_secondary")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )
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

        def tell(self) -> int:
            return getattr(self, "position", 0)

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = 8 + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return b"\0\0\0\0PAR1"

        def close(self) -> None:
            raise OSError("raw_close_failed")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )
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
        "remove_owned_scratch",
        lambda _path: (_ for _ in ()).throw(RuntimeError("cleanup_failed")),
    )

    def writer(_path: Path) -> None:
        raise primary

    with pytest.raises(type(primary)):
        atomic.atomic_write_directory(tmp_path / "result", writer)


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
        scratch_owner,
        "open_owned_scratch_file",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )

    with pytest.raises(primary_type, match="body_primary"):
        manifests.replace_qrp_json(
            tmp_path / "manifest.json",
            {"schema_id": "qrp-test"},
        )


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
        "current_runtime_resource_policy",
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
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )

    assert (
        comparison.compare_datasets(left, right, resource_policy=policy).status
        == "INSPECTION_FAILED"
    )


def test_artifact_root_alias_is_rejected_by_unique_root_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("dataset")
    absolute_root = Path.cwd() / root
    monkeypatch.setattr(
        identity,
        "path_is_filesystem_alias",
        lambda candidate: candidate == absolute_root,
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        identity.require_artifact_root(root)


@pytest.mark.parametrize("position", ("root", "immediate", "deep"))
@pytest.mark.parametrize("alias_kind", ("symlink", "junction"))
def test_artifact_root_rejects_exact_or_ancestor_alias(
    tmp_path: Path,
    position: str,
    alias_kind: str,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    if alias_kind == "symlink":
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink_not_available: {exc}")
    else:
        if os.name != "nt" or not hasattr(Path, "is_junction"):
            pytest.skip("junction_capability_required")
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip("junction_not_available")
    root = (
        alias
        if position == "root"
        else alias / "artifact"
        if position == "immediate"
        else alias / "nested" / "artifact"
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        identity.require_artifact_root(root)


def test_artifact_root_ordinary_chain_preserves_candidate_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("relative") / "dataset"
    monkeypatch.setattr(
        identity,
        "path_is_filesystem_alias",
        lambda _candidate: False,
    )

    assert identity.require_artifact_root(root) == root


def test_dataset_file_identity_accepts_single_external_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"payload")
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    assert identity.dataset_file_identity(alias) == identity.dataset_file_identity(
        source
    )


def test_dataset_file_identity_rejects_ancestor_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "part.bin").write_bytes(b"payload")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        identity.dataset_file_identity(alias / "part.bin")


def test_bounded_metadata_reader_rejects_ancestor_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text('{"schema_id":"test"}', encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="^dataset_metadata_file_invalid$"):
        manifests.read_bounded_qrp_json(alias / "manifest.json")


def test_write_dataset_requires_artifact_root_before_writer_or_data_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    monkeypatch.setattr(dataset_writing, "validate_dataset_spec", lambda _spec: None)
    monkeypatch.setattr(
        dataset_writing,
        "validate_dataset_artifact_root",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        dataset_writing,
        "require_artifact_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid_dataset_part_files")),
        raising=False,
    )
    monkeypatch.setattr(
        dataset_writing,
        "_write_dataset_body",
        lambda *_args, **_kwargs: pytest.fail("writer_before_artifact_root_gate"),
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        dataset_writing.write_dataset(
            Path("dataset"),
            (_ for _ in ()),
            spec=object(),  # type: ignore[arg-type]
            resource_policy=policy,
        )


def test_compare_rechecks_artifact_roots_before_each_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = Path("left")
    right = Path("right")
    validations = {left: 0, right: 0}
    probes = {left: 0, right: 0}
    original_exists = Path.exists
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    def require_root(path: Path) -> Path:
        candidate = Path(path)
        validations[candidate] += 1
        return candidate

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate not in probes:
            return original_exists(path)
        probes[candidate] += 1
        assert validations[candidate] == probes[candidate]
        return True

    monkeypatch.setattr(
        comparison,
        "require_artifact_root",
        require_root,
        raising=False,
    )
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )

    result = comparison.compare_datasets(left, right, resource_policy=policy)

    assert result.status == "INSPECTION_FAILED"
    assert validations == {left: 2, right: 2}
    assert probes == {left: 2, right: 2}


def test_compare_maps_aliased_root_before_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = Path("left")
    right = Path("right")
    original_exists = Path.exists

    def reject_alias(path: Path) -> Path:
        if Path(path) == left:
            raise ValueError("invalid_dataset_part_files")
        return Path(path)

    monkeypatch.setattr(
        comparison,
        "require_artifact_root",
        reject_alias,
        raising=False,
    )

    def exists(path: Path) -> bool:
        if Path(path) in (left, right):
            raise AssertionError("dataset_exists_before_artifact_root_alias_gate")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)

    result = comparison.compare_datasets(left, right, resource_policy=None)

    assert result.status == "INSPECTION_FAILED"
    assert result.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("alias_side", ("left", "right"))
def test_compare_maps_ancestor_alias_before_public_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
    alias_side: str,
) -> None:
    left = Path("left-parent") / "left"
    right = Path("right-parent") / "right"
    aliased_parent = Path.cwd() / (left if alias_side == "left" else right).parent

    monkeypatch.setattr(
        identity,
        "path_is_filesystem_alias",
        lambda candidate: candidate == aliased_parent,
    )
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate == (left if alias_side == "left" else right):
            raise AssertionError("dataset_exists_before_ancestor_alias_gate")
        if candidate in (left, right):
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)

    result = comparison.compare_datasets(left, right, resource_policy=None)

    assert result.status == "INSPECTION_FAILED"
    assert result.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("alias_side", ("left", "right"))
def test_compare_internal_recheck_blocks_new_ancestor_alias_before_exists(
    monkeypatch: pytest.MonkeyPatch,
    alias_side: str,
) -> None:
    left = Path("left-parent") / "left"
    right = Path("right-parent") / "right"
    target = left if alias_side == "left" else right
    aliased_parent = Path.cwd() / target.parent
    gate_count = 0
    probe_count = {left: 0, right: 0}
    original_require = comparison.require_artifact_root
    original_exists = Path.exists

    def require_root(path: Path) -> Path:
        nonlocal gate_count
        gate_count += 1
        return original_require(path)

    def is_alias(candidate: Path) -> bool:
        internal_gate = gate_count >= (3 if alias_side == "left" else 4)
        return internal_gate and candidate == aliased_parent

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate not in probe_count:
            return original_exists(path)
        probe_count[candidate] += 1
        if candidate == target:
            assert probe_count[candidate] == 1
        return True

    monkeypatch.setattr(comparison, "require_artifact_root", require_root)
    monkeypatch.setattr(identity, "path_is_filesystem_alias", is_alias)
    monkeypatch.setattr(Path, "exists", exists)

    result = comparison.compare_datasets(
        left,
        right,
        resource_policy=capture_test_policy(
            available_memory_bytes=16 * 1024 * 1024,
            explicit_cap_bytes=1 * 1024 * 1024,
        ),
    )

    assert result.status == "INSPECTION_FAILED"
    assert probe_count[target] == 1


def test_shared_scratch_removes_only_exact_owned_paths(tmp_path: Path) -> None:
    owned_file = tmp_path / "owned-file"
    owned_file.write_bytes(b"x")
    owned_directory = tmp_path / "owned-directory"
    owned_directory.mkdir()
    (owned_directory / "part").write_bytes(b"x")

    scratch_owner.remove_owned_scratch(owned_file)
    scratch_owner.remove_owned_scratch(owned_directory)

    assert not owned_file.exists()
    assert not owned_directory.exists()
