from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import pytest

import quant_research_platform.research.datasets.assembler as assembler
import quant_research_platform.runtime.resources as resources_owner
import quant_research_platform.storage._dataset_reading as dataset_reading
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as a_share_materializer,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materializer as crypto_materializer,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_types import DatasetReadMode


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": str(megabytes)},
    )


def test_bind_runtime_resource_policy_truth_table_without_active_scope() -> None:
    explicit = _policy(1)

    assert bind_runtime_resource_policy(None) is None
    assert bind_runtime_resource_policy(explicit) is explicit


def test_bind_runtime_resource_policy_truth_table_with_active_scope() -> None:
    active = _policy(1)
    different = _policy(2)

    with runtime_resource_policy_scope(active):
        assert bind_runtime_resource_policy(None) is active
        assert bind_runtime_resource_policy(active) is active
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            bind_runtime_resource_policy(different)


def test_resource_scope_is_same_object_reentrant_and_rejects_different_policy() -> None:
    outer = _policy(1)
    other = _policy(2)

    with runtime_resource_policy_scope(outer):
        with runtime_resource_policy_scope(outer):
            assert active_runtime_resource_policy() is outer
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            with runtime_resource_policy_scope(other):
                pass

    assert active_runtime_resource_policy() is not outer


def test_retired_policy_cannot_start_a_second_root_but_fresh_policy_can() -> None:
    retired = _policy(1)

    with runtime_resource_policy_scope(retired):
        assert current_runtime_resource_policy() is retired

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(retired):
            pass

    fresh = _policy(1)
    with runtime_resource_policy_scope(fresh):
        assert current_runtime_resource_policy() is fresh


@pytest.mark.parametrize("root", ("a-share", "crypto", "research"))
def test_invalid_bounded_preflight_leaves_explicit_policy_available(
    root: str,
) -> None:
    policy = _policy(1)

    with pytest.raises(ValueError):
        if root == "a-share":
            a_share_materializer.materialize_processed_data(
                layout=SimpleNamespace(),
                source_id=SimpleNamespace(),
                time_window=SimpleNamespace(),
                resource_policy=policy,
            )
        elif root == "crypto":
            crypto_materializer.materialize_processed_data(
                layout=SimpleNamespace(),
                source_id=SimpleNamespace(),
                time_window=SimpleNamespace(),
                resource_policy=policy,
            )
        else:
            dataset_catalog = SimpleNamespace(
                get=lambda _data_id: (_ for _ in ()).throw(
                    ValueError("unknown_research_dataset")
                )
            )
            assembler.assemble_research_dataset(
                layout=SimpleNamespace(),
                dataset_catalog=dataset_catalog,
                processed_catalog=SimpleNamespace(),
                dataset_id="missing",
                time_window=SimpleNamespace(),
                resource_policy=policy,
            )

    with runtime_resource_policy_scope(policy):
        assert current_runtime_resource_policy() is policy


def test_copied_context_is_valid_only_while_issuance_is_active() -> None:
    policy = _policy(1)

    with runtime_resource_policy_scope(policy):
        copied = copy_context()
        assert copied.run(current_runtime_resource_policy) is policy

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        copied.run(current_runtime_resource_policy)


def test_independent_thread_root_fails_closed_without_blocking() -> None:
    outer = _policy(1)
    other = _policy(2)
    entered = Event()
    release = Event()
    result: Queue[str] = Queue()

    def hold_root() -> None:
        with runtime_resource_policy_scope(outer):
            entered.set()
            release.wait(timeout=5)

    def try_root() -> None:
        entered.wait(timeout=5)
        try:
            with runtime_resource_policy_scope(other):
                result.put("entered")
        except RuntimeError as exc:
            result.put(str(exc))

    first = Thread(target=hold_root)
    second = Thread(target=try_root)
    first.start()
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)

    assert result.get_nowait() == "runtime_resource_operation_already_active"


@pytest.mark.parametrize(
    "primary",
    (
        RuntimeError("body"),
        MemoryError(),
        KeyboardInterrupt(),
        SystemExit(),
        GeneratorExit(),
    ),
)
def test_resource_scope_releases_after_every_primary(primary: BaseException) -> None:
    first = _policy(1)
    second = _policy(2)

    with pytest.raises(type(primary)):
        with runtime_resource_policy_scope(first):
            raise primary

    with runtime_resource_policy_scope(second):
        assert active_runtime_resource_policy() is second


@pytest.mark.parametrize(
    "body_primary",
    (False, True),
    ids=("normal", "generator-exit"),
)
@pytest.mark.parametrize(
    "failure_sites",
    (
        frozenset({"retire"}),
        frozenset({"reset"}),
        frozenset({"release"}),
        frozenset({"retire", "reset"}),
        frozenset({"retire", "release"}),
        frozenset({"reset", "release"}),
        frozenset({"retire", "reset", "release"}),
    ),
)
def test_resource_scope_attempts_retire_reset_and_release_preserving_first_failure(
    body_primary: bool,
    failure_sites: frozenset[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    events: list[str] = []

    class Context:
        active: RuntimeResourcePolicy | None = None

        def get(self) -> RuntimeResourcePolicy | None:
            return self.active

        def set(self, value: RuntimeResourcePolicy) -> object:
            self.active = value
            return object()

        def reset(self, _token: object) -> None:
            events.append("reset")
            if "reset" in failure_sites:
                raise RuntimeError("reset_failed")
            self.active = None

    class OperationLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return True

        def release(self) -> None:
            events.append("release")
            if "release" in failure_sites:
                raise OSError("release_failed")

    def retire(_policy: RuntimeResourcePolicy) -> None:
        events.append("retire")
        if "retire" in failure_sites:
            raise ValueError("retire_failed")

    monkeypatch.setattr(resources_owner, "_ACTIVE_RUNTIME_RESOURCE_POLICY", Context())
    monkeypatch.setattr(
        resources_owner,
        "_RUNTIME_RESOURCE_OPERATION_LOCK",
        OperationLock(),
    )
    monkeypatch.setattr(resources_owner, "_retire_issued_policy", retire)

    if body_primary:
        expected = GeneratorExit
        match = None
    else:
        first_failure = next(
            site for site in ("retire", "reset", "release") if site in failure_sites
        )
        expected = {"retire": ValueError, "reset": RuntimeError, "release": OSError}[
            first_failure
        ]
        match = f"{first_failure}_failed"
    with pytest.raises(expected, match=match):
        with runtime_resource_policy_scope(policy):
            if body_primary:
                raise GeneratorExit()

    assert events == ["retire", "reset", "release"]


def test_resource_scope_releases_lock_when_context_set_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    events: list[str] = []

    class Context:
        def get(self) -> None:
            return None

        def set(self, _value: RuntimeResourcePolicy) -> object:
            raise RuntimeError("set_failed")

    class OperationLock:
        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return True

        def release(self) -> None:
            events.append("release")
            raise OSError("release_failed")

    def retire(_policy: RuntimeResourcePolicy) -> None:
        events.append("retire")
        raise ValueError("retire_failed")

    monkeypatch.setattr(resources_owner, "_ACTIVE_RUNTIME_RESOURCE_POLICY", Context())
    monkeypatch.setattr(
        resources_owner,
        "_RUNTIME_RESOURCE_OPERATION_LOCK",
        OperationLock(),
    )
    monkeypatch.setattr(resources_owner, "_retire_issued_policy", retire)

    with pytest.raises(RuntimeError, match="set_failed"):
        with runtime_resource_policy_scope(policy):
            pass

    assert events == ["retire", "release"]


def test_lazy_read_binds_creation_policy_and_rejects_different_consumer_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_under = _policy(1)
    consumed_under = _policy(2)
    observed: list[RuntimeResourcePolicy | None] = []

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

    with runtime_resource_policy_scope(created_under):
        lazy = dataset_reading.read_dataset(
            Path("unused"),
            mode=DatasetReadMode.CHUNKED,
            resource_policy=None,
        )
    with runtime_resource_policy_scope(consumed_under):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            list(lazy)

    assert observed == []


def test_lazy_read_bound_to_outer_consumes_only_while_outer_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(1)
    observed: list[RuntimeResourcePolicy] = []

    class Session:
        resource_policy = policy

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        observed.append(resource_policy)
        return Session()

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter(()),
    )

    with runtime_resource_policy_scope(policy):
        active_lazy = dataset_reading.read_dataset(
            Path("unused"), mode=DatasetReadMode.CHUNKED, resource_policy=None
        )
        assert list(active_lazy) == []
        stale_lazy = dataset_reading.read_dataset(
            Path("unused"), mode=DatasetReadMode.CHUNKED, resource_policy=None
        )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        list(stale_lazy)
    assert observed == [policy]


def test_lazy_read_without_outer_admits_fresh_policy_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[RuntimeResourcePolicy] = []

    class Session:
        def __init__(self, resource_policy: RuntimeResourcePolicy) -> None:
            self.resource_policy = resource_policy

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        issued_policy = resource_policy or active_runtime_resource_policy()
        observed.append(issued_policy)
        with runtime_resource_policy_scope(issued_policy):
            with Session(issued_policy) as session:
                yield session

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter(()),
    )

    lazy = dataset_reading.read_dataset(
        Path("unused"),
        mode=DatasetReadMode.CHUNKED,
        resource_policy=None,
    )

    assert list(lazy) == []
    assert len(observed) == 1
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(observed[0]):
            pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork is unavailable")
def test_fork_child_rejects_inherited_policy_and_admits_fresh_policy() -> None:
    inherited = _policy(1)
    read_fd, write_fd = os.pipe()

    with runtime_resource_policy_scope(inherited):
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                assert current_runtime_resource_policy() is None
                try:
                    with runtime_resource_policy_scope(inherited):
                        pass
                except RuntimeError as exc:
                    assert str(exc) == "runtime_resource_policy_wrong_process"
                else:
                    raise AssertionError("fork_child_reused_inherited_policy")
                fresh = _policy(1)
                with runtime_resource_policy_scope(fresh):
                    assert current_runtime_resource_policy() is fresh
                os.write(write_fd, b"PASS")
            except BaseException as exc:
                os.write(write_fd, f"FAIL:{type(exc).__name__}:{exc}".encode())
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        child_result = os.read(read_fd, 4096)
        os.close(read_fd)
        _, child_status = os.waitpid(child_pid, 0)

    assert child_status == 0
    assert child_result == b"PASS"
