from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
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
    closed = False

    def close(self) -> None:
        self.closed = True


def _patch_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_memory_bytes: int = 10 * GIB,
    explicit_cap_bytes: int | None = None,
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


def test_capture_acquires_machine_lock_before_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, locks = _patch_capture(monkeypatch)

    policy = RuntimeResourcePolicy.capture()

    assert events == ["lock", "available"]
    assert len(locks) == 1
    assert not locks[0].closed
    assert policy.state == "ACTIVE"


def test_budget_is_exact_seventy_percent_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_capture(monkeypatch, available_memory_bytes=10 * GIB + 1)

    policy = RuntimeResourcePolicy.capture()

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


def test_explicit_cap_only_lowers_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_capture(monkeypatch)

    uncapped = RuntimeResourcePolicy.capture()
    high = RuntimeResourcePolicy.capture(explicit_cap_bytes=20 * GIB)
    low = RuntimeResourcePolicy.capture(explicit_cap_bytes=2 * GIB)

    assert uncapped.memory_budget_bytes == 7 * GIB
    assert high.memory_budget_bytes == 7 * GIB
    assert low.memory_budget_bytes == 2 * GIB


@pytest.mark.parametrize("explicit_cap_bytes", (True, 0, -1, 1.0, "1"))
def test_invalid_cap_fails_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
    explicit_cap_bytes: object,
) -> None:
    events, locks = _patch_capture(monkeypatch)

    with pytest.raises(ValueError, match="invalid_runtime_resource_explicit_cap"):
        RuntimeResourcePolicy.capture(
            explicit_cap_bytes=explicit_cap_bytes,  # type: ignore[arg-type]
        )

    assert events == ["lock"]
    assert locks[0].closed


@pytest.mark.parametrize(
    ("available_memory_bytes", "error"),
    [
        (0, "invalid_runtime_resource_available_memory"),
        (True, "invalid_runtime_resource_available_memory"),
        (1.0, "invalid_runtime_resource_available_memory"),
        (-1, "invalid_runtime_resource_available_memory"),
    ],
)
def test_invalid_available_memory_fails_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
    available_memory_bytes: object,
    error: str,
) -> None:
    events, locks = _patch_capture(
        monkeypatch,
        available_memory_bytes=10 * GIB,  # type: ignore[arg-type]
    )

    def available() -> object:
        events.append("available")
        return available_memory_bytes

    monkeypatch.setattr(
        resources,
        "_available_memory_bytes",
        available,
    )

    with pytest.raises(ValueError, match=error):
        RuntimeResourcePolicy.capture()

    assert events == ["lock", "available"]
    assert locks[0].closed


def test_provider_failure_fails_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, locks = _patch_capture(monkeypatch)

    def fail() -> int:
        events.append("available")
        raise RuntimeError("runtime_available_memory_discovery_failed")

    monkeypatch.setattr(resources, "_available_memory_bytes", fail)
    with pytest.raises(RuntimeError, match="runtime_available_memory_discovery_failed"):
        RuntimeResourcePolicy.capture()

    assert events == ["lock", "available"]
    assert locks[0].closed


def test_held_machine_lock_fails_before_available_memory_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def acquire() -> _FakeMachineLock:
        calls.append("lock")
        if len(calls) == 1:
            return _FakeMachineLock()
        raise RuntimeError("runtime_data_scale_lock_busy")

    monkeypatch.setattr(resources, "_acquire_machine_lock", acquire)
    monkeypatch.setattr(
        resources,
        "_available_memory_bytes",
        lambda: calls.append("available") or 10 * GIB,
    )

    first = RuntimeResourcePolicy.capture()
    with pytest.raises(RuntimeError, match="runtime_data_scale_lock_busy"):
        RuntimeResourcePolicy.capture()

    assert calls == ["lock", "available", "lock"]
    with runtime_resource_policy_scope(first):
        pass


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return environment


def test_second_process_is_busy_and_crash_releases_machine_lock() -> None:
    child = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "from quant_research_platform.runtime.resources import "
            "RuntimeResourcePolicy\n"
            "RuntimeResourcePolicy.capture()\n"
            "print('held', flush=True)\n"
            "input()\n",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_environment(),
    )
    try:
        assert child.stdout is not None and child.stdin is not None
        assert child.stdout.readline() == "held\n"
        with pytest.raises(RuntimeError, match="runtime_data_scale_lock_busy"):
            RuntimeResourcePolicy.capture()
    finally:
        child.kill()
        child.wait(timeout=5)

    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        pass


def test_transfer_keeps_root_total_constant() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("producer", 600)
        assert (lease.owner, lease.reserved_bytes, lease.active) == (
            "producer",
            600,
            True,
        )
        lease.transfer("writer")
        assert policy.used_memory_bytes == 600
        lease.close()
        assert policy.used_memory_bytes == 0
        assert lease.active is False
        lease.close()
        assert policy.used_memory_bytes == 0


def test_grow_pre_admits_atomically_and_shrink_only_decreases() -> None:
    policy = capture_test_policy(
        available_memory_bytes=MIB,
        explicit_cap_bytes=733,
    )
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("producer", 600)
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            lease.grow(134)
        assert (lease.reserved_bytes, policy.used_memory_bytes) == (600, 600)

        lease.grow(133)
        assert (lease.reserved_bytes, policy.used_memory_bytes) == (733, 733)
        with pytest.raises(ValueError, match="invalid_runtime_memory_lease_bytes"):
            lease.shrink_to(734)
        assert lease.reserved_bytes == 733

        lease.shrink_to(500)
        assert (lease.reserved_bytes, policy.used_memory_bytes) == (500, 500)
        lease.close()


def test_closed_lease_cannot_be_reused() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("producer", 10)
        lease.close()
        with pytest.raises(RuntimeError, match="runtime_memory_lease_not_active"):
            lease.transfer("writer")
        with pytest.raises(RuntimeError, match="runtime_memory_lease_not_active"):
            lease.grow(1)
        with pytest.raises(RuntimeError, match="runtime_memory_lease_not_active"):
            lease.shrink_to(0)


@pytest.mark.parametrize("owner", (None, "", 1))
def test_lease_rejects_invalid_owner(owner: object) -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="invalid_runtime_memory_lease_owner"):
            policy.lease(owner, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("reserved_bytes", (True, -1, 1.0, "1"))
def test_lease_rejects_invalid_bytes(reserved_bytes: object) -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="invalid_runtime_memory_lease_bytes"):
            policy.lease("owner", reserved_bytes)  # type: ignore[arg-type]


def test_owned_value_is_one_generic_wrapper() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("producer", 1)
        owned = resources._OwnedValue(1, lease)
        assert owned.value == 1
        assert owned.lease is lease
        lease.close()


def test_cleanup_registry_is_lifo() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    events: list[str] = []
    with runtime_resource_policy_scope(policy):
        policy.register_cleanup(lambda: events.append("first"))
        policy.register_cleanup(lambda: events.append("second"))

    assert events == ["second", "first"]
    assert policy.state == "CLOSED"


def test_clean_keyboard_interrupt_preserves_primary_and_releases_lock() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with pytest.raises(KeyboardInterrupt):
        with runtime_resource_policy_scope(policy):
            policy.register_cleanup(lambda: None)
            raise KeyboardInterrupt()

    assert policy.state == "CLOSED"
    assert policy._lock.closed  # type: ignore[union-attr]  # noqa: SLF001


def test_leaked_lease_poisons_root_and_blocks_second_root(
) -> None:
    code = """
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=1024)
lease = None
try:
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("storage", 10)
        raise KeyboardInterrupt
except KeyboardInterrupt:
    pass
print(policy.state, policy.used_memory_bytes, policy._lock.closed)
lease.close()
print(policy.state, policy.used_memory_bytes, policy._lock.closed)
try:
    RuntimeResourcePolicy.capture()
except RuntimeError as exc:
    print(exc)
"""
    output = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=30,
    ).stdout.splitlines()

    assert output == [
        "POISONED 10 False",
        "POISONED 0 False",
        "runtime_data_scale_process_poisoned",
    ]


def test_cleanup_failure_adds_note_poisons_and_preserves_primary() -> None:
    code = """
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=1024)
try:
    with runtime_resource_policy_scope(policy):
        def fail():
            raise RuntimeError("cleanup failed")
        policy.register_cleanup(fail)
        raise KeyboardInterrupt
except KeyboardInterrupt as exc:
    print(policy.state, policy._lock.closed)
    print(exc.__notes__)
"""
    output = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=30,
    ).stdout.splitlines()

    assert output[0] == "POISONED False"
    assert "runtime cleanup also failed: RuntimeError: cleanup failed" in output[1]


def test_old_rss_surface_has_no_safety_authority() -> None:
    policy = capture_test_policy(available_memory_bytes=MIB)
    with runtime_resource_policy_scope(policy):
        evidence = policy.memory_evidence()
        policy.assert_memory_within_budget()
        assert policy.finish_memory_measurement() == evidence

    assert evidence == {"evidence_scope": "external_release_only"}
    assert "status" not in evidence
    assert not hasattr(resources, "_OperationRssMeasurement")
    assert not hasattr(resources, "_process_rss_bytes")
