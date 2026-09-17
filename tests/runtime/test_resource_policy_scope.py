from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import quant_research_platform.runtime.resources as resources
import quant_research_platform.storage.datasets.reading as dataset_reading
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.types import DatasetReadMode
from tests.helpers.runtime_resources import capture_test_policy


def _policy(megabytes: int) -> RuntimeResourcePolicy:
    return capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=megabytes * 1024 * 1024,
    )


def test_scope_reuses_exact_context_and_rejects_other_root() -> None:
    outer = _policy(1)
    other = _policy(2)

    with runtime_resource_policy_scope(outer):
        with runtime_resource_policy_scope() as nested:
            assert nested is outer
            assert current_runtime_resource_policy() is outer
        with runtime_resource_policy_scope(outer) as same:
            assert same is outer
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            with runtime_resource_policy_scope(other):
                pass

    assert current_runtime_resource_policy() is None


def test_scope_none_is_the_only_new_root_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []
    original = resources._capture_runtime_resource_policy

    def capture(*, explicit_cap_bytes=None):  # type: ignore[no-untyped-def]
        calls.append(explicit_cap_bytes)
        return original(explicit_cap_bytes=explicit_cap_bytes)

    monkeypatch.setattr(resources, "_capture_runtime_resource_policy", capture)
    with runtime_resource_policy_scope(explicit_cap_bytes=1024 * 1024) as policy:
        assert current_runtime_resource_policy() is policy
        with runtime_resource_policy_scope() as nested:
            assert nested is policy

    assert calls == [1024 * 1024]


def test_active_scope_does_not_capture_again(monkeypatch: pytest.MonkeyPatch) -> None:
    active = _policy(1)

    def fail_capture(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("nested scope must reuse the active policy")

    monkeypatch.setattr(resources, "_capture_runtime_resource_policy", fail_capture)
    with runtime_resource_policy_scope(active):
        with runtime_resource_policy_scope() as nested:
            assert nested is active


def test_explicit_cap_is_validated_before_scope_body() -> None:
    policy = _policy(1)
    with pytest.raises(ValueError, match="invalid_runtime_resource_explicit_cap"):
        with runtime_resource_policy_scope(policy, explicit_cap_bytes=1):
            pass


def test_retired_policy_cannot_start_a_second_root() -> None:
    retired = _policy(1)
    with runtime_resource_policy_scope(retired):
        pass

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        with runtime_resource_policy_scope(retired):
            pass


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
        with runtime_resource_policy_scope(resource_policy) as issued:
            observed.append(issued)
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


def test_suspended_public_read_rejects_a_second_independent_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issued = _policy(1)

    @contextmanager
    def open_session(*_args: object, resource_policy=None, **_kwargs: object):  # type: ignore[no-untyped-def]
        policy = resource_policy or issued
        with runtime_resource_policy_scope(policy):
            yield SimpleNamespace(resource_policy=policy)

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_a, **_k: iter((SimpleNamespace(),)),
    )

    first = dataset_reading.read_dataset(
        Path("first"),
        mode=DatasetReadMode.CHUNKED,
        resource_policy=None,
    )
    next(first)
    try:
        with pytest.raises(
            RuntimeError,
            match="runtime_resource_policy_competing_root",
        ):
            dataset_reading.read_dataset(
                Path("second"),
                mode=DatasetReadMode.CHUNKED,
                resource_policy=None,
            )
    finally:
        first.close()  # type: ignore[union-attr]
