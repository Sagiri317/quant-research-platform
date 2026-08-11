from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

import quant_research_platform.storage._dataset_reading as dataset_reading
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_types import DatasetReadMode
from tests.helpers.runtime_resources import capture_test_policy


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        rss_bytes=1024,
        explicit_cap_bytes=megabytes * 1024 * 1024,
    )


def test_bind_runtime_resource_policy_truth_table() -> None:
    active = _policy(1)
    different = _policy(2)

    assert bind_runtime_resource_policy(None) is None
    assert bind_runtime_resource_policy(active) is active
    with runtime_resource_policy_scope(active):
        assert bind_runtime_resource_policy(None) is active
        assert bind_runtime_resource_policy(active) is active
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            bind_runtime_resource_policy(different)


def test_scope_is_same_object_reentrant_and_rejects_other_root() -> None:
    outer = _policy(1)
    other = _policy(2)

    with runtime_resource_policy_scope(outer):
        with runtime_resource_policy_scope(outer):
            assert active_runtime_resource_policy() is outer
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            with runtime_resource_policy_scope(other):
                pass

    assert current_runtime_resource_policy() is None


def test_retired_policy_cannot_start_a_second_root() -> None:
    retired = _policy(1)
    with runtime_resource_policy_scope(retired):
        assert current_runtime_resource_policy() is retired

    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(retired):
            pass


def test_same_policy_cannot_reenter_from_another_thread() -> None:
    policy = _policy(1)
    observed: list[str] = []

    def enter() -> None:
        try:
            with runtime_resource_policy_scope(policy):
                observed.append("entered")
        except RuntimeError as exc:
            observed.append(str(exc))

    with runtime_resource_policy_scope(policy):
        worker = Thread(target=enter)
        worker.start()
        worker.join(timeout=5)

    assert observed == ["runtime_resource_policy_wrong_owner"]


@pytest.mark.parametrize(
    "primary",
    [RuntimeError("body"), MemoryError(), KeyboardInterrupt(), SystemExit()],
)
def test_scope_releases_after_every_primary(primary: BaseException) -> None:
    first = _policy(1)
    second = _policy(2)

    with pytest.raises(type(primary)):
        with runtime_resource_policy_scope(first):
            raise primary

    with runtime_resource_policy_scope(second):
        assert current_runtime_resource_policy() is second


def test_lazy_read_binds_creation_policy_and_rejects_other_consumer(
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


def test_lazy_read_without_outer_admits_one_fresh_policy(
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
        issued = resource_policy or active_runtime_resource_policy()
        observed.append(issued)
        with runtime_resource_policy_scope(issued):
            with Session(issued) as session:
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
