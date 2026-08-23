from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.runtime_resources import capture_test_policy

GIB = 1024**3
MIB = 1024**2


@pytest.fixture(autouse=True)
def _isolated_runtime_root() -> Iterator[None]:
    resources._ACTIVE_RUNTIME_RESOURCE_POLICY = None
    resources._PROCESS_POISONED = False
    yield
    active = resources._ACTIVE_RUNTIME_RESOURCE_POLICY
    if active is not None:
        active._lock.close()  # noqa: SLF001
    resources._ACTIVE_RUNTIME_RESOURCE_POLICY = None
    resources._PROCESS_POISONED = False


class _FakeMachineLock:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_memory_bytes: int = 10 * GIB,
) -> tuple[list[str], list[_FakeMachineLock]]:
    events: list[str] = []
    locks: list[_FakeMachineLock] = []

    def acquire() -> _FakeMachineLock:
        events.append("lock")
        lock = _FakeMachineLock()
        locks.append(lock)
        return lock

    def available() -> int:
        events.append("available")
        return available_memory_bytes

    monkeypatch.setattr(resources, "_acquire_machine_lock", acquire)
    monkeypatch.setattr(resources, "_available_memory_bytes", available)
    return events, locks


def test_root_opener_locks_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    events, locks = _patch_capture(monkeypatch)

    with runtime_resource_policy_scope() as policy:
        assert policy.state == "ACTIVE"

    assert events == ["lock", "available"]
    assert locks[0].closed


def test_budget_is_exact_seventy_percent_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capture(monkeypatch, available_memory_bytes=10 * GIB + 1)

    with runtime_resource_policy_scope() as policy:
        assert policy.captured_available_memory_bytes == 10 * GIB + 1
        assert policy.memory_budget_bytes == 7 * GIB
        assert policy.used_memory_bytes == 0
        assert policy.available_memory_bytes == 7 * GIB
        assert policy.to_payload() == {
            "captured_available_memory_bytes": 10 * GIB + 1,
            "memory_budget_bytes": 7 * GIB,
            "used_memory_bytes": 0,
            "available_memory_bytes": 7 * GIB,
        }


def test_explicit_cap_only_lowers_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_capture(monkeypatch)

    with runtime_resource_policy_scope(explicit_cap_bytes=20 * GIB) as high:
        assert high.memory_budget_bytes == 7 * GIB
    with runtime_resource_policy_scope(explicit_cap_bytes=2 * GIB) as low:
        assert low.memory_budget_bytes == 2 * GIB


@pytest.mark.parametrize("explicit_cap_bytes", (True, 0, -1, 1.0, "1"))
def test_invalid_cap_fails_before_provider_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
    explicit_cap_bytes: object,
) -> None:
    events, locks = _patch_capture(monkeypatch)

    with pytest.raises(ValueError, match="invalid_runtime_resource_explicit_cap"):
        with runtime_resource_policy_scope(
            explicit_cap_bytes=explicit_cap_bytes,  # type: ignore[arg-type]
        ):
            pass

    assert events == ["lock"]
    assert locks[0].closed


@pytest.mark.parametrize("available", (0, True, 1.0, -1))
def test_invalid_provider_fails_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
    available: object,
) -> None:
    events, locks = _patch_capture(monkeypatch)

    def provider() -> object:
        events.append("available")
        return available

    monkeypatch.setattr(resources, "_available_memory_bytes", provider)
    with pytest.raises(ValueError, match="invalid_runtime_resource_available_memory"):
        with runtime_resource_policy_scope():
            pass

    assert events == ["lock", "available"]
    assert locks[0].closed


def test_provider_failure_releases_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    events, locks = _patch_capture(monkeypatch)

    def provider() -> int:
        events.append("available")
        raise RuntimeError("runtime_available_memory_discovery_failed")

    monkeypatch.setattr(resources, "_available_memory_bytes", provider)
    with pytest.raises(RuntimeError, match="runtime_available_memory_discovery_failed"):
        with runtime_resource_policy_scope():
            pass
    assert events == ["lock", "available"]
    assert locks[0].closed


def test_nested_scope_reuses_exact_context_and_rejects_conflict() -> None:
    outer = capture_test_policy(available_memory_bytes=16 * MIB)
    other = capture_test_policy(available_memory_bytes=16 * MIB)

    with runtime_resource_policy_scope(outer):
        with runtime_resource_policy_scope() as nested:
            assert nested is outer
        with runtime_resource_policy_scope(outer) as same:
            assert same is outer
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            with runtime_resource_policy_scope(other):
                pass


def test_lease_overflow_release_and_leak_poison() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB, explicit_cap_bytes=733)
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("producer", 600)
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            lease.grow(134)
        lease.grow(133)
        lease.shrink_to(500)
        assert policy.used_memory_bytes == 500
        lease.close()
        assert policy.used_memory_bytes == 0

    leaked = capture_test_policy(available_memory_bytes=MIB, explicit_cap_bytes=733)
    with pytest.raises(RuntimeError, match="runtime_memory_lease_leaked"):
        with runtime_resource_policy_scope(leaked):
            leaked.lease("producer", 1)
    assert leaked.state == "POISONED"
    assert not leaked._lock.closed  # noqa: SLF001


def test_lease_context_manager_closes_and_preserves_body_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=MIB, explicit_cap_bytes=733)
    real_close = resources.MemoryLease.close

    def fail_close(lease: resources.MemoryLease) -> None:
        real_close(lease)
        raise OSError("lease_release_secondary")

    monkeypatch.setattr(resources.MemoryLease, "close", fail_close)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyboardInterrupt, match="lease_body_primary") as raised:
            with policy.lease("primary", 17) as lease:
                assert lease.active
                assert policy.used_memory_bytes == 17
                raise KeyboardInterrupt("lease_body_primary")
        assert any(
            "lease release also failed: OSError" in note
            for note in raised.value.__notes__
        )
        assert policy.used_memory_bytes == 0


def test_leaked_child_keeps_machine_lock_until_process_exit() -> None:
    child = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "from quant_research_platform.runtime.resources import "
            "runtime_resource_policy_scope\n"
            "try:\n"
            "    with runtime_resource_policy_scope(\n"
            "        explicit_cap_bytes=1048576\n"
            "    ) as policy:\n"
            "        policy.lease('leak', 1)\n"
            "except RuntimeError as exc:\n"
            "    assert str(exc) == 'runtime_memory_lease_leaked'\n"
            "    print('leak-handled', flush=True)\n"
            "input()\n",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_environment(),
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == "leak-handled\n"
        with pytest.raises(RuntimeError, match="runtime_data_scale_lock_busy"):
            with runtime_resource_policy_scope(explicit_cap_bytes=MIB):
                pass
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

    with runtime_resource_policy_scope(explicit_cap_bytes=MIB) as policy:
        assert policy.state == "ACTIVE"


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return environment


def test_two_process_busy_then_crash_reentry() -> None:
    child = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "from quant_research_platform.runtime.resources import "
            "runtime_resource_policy_scope\n"
            "with runtime_resource_policy_scope(explicit_cap_bytes=1048576):\n"
            " print('held', flush=True)\n"
            " input()\n",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_environment(),
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == "held\n"
        with pytest.raises(RuntimeError, match="runtime_data_scale_lock_busy"):
            with runtime_resource_policy_scope(explicit_cap_bytes=MIB):
                pass
    finally:
        child.kill()
        child.wait(timeout=5)

    with runtime_resource_policy_scope(explicit_cap_bytes=MIB) as policy:
        assert policy.state == "ACTIVE"


def test_retired_root_cannot_reenter() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        pass
    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        with runtime_resource_policy_scope(policy):
            pass
