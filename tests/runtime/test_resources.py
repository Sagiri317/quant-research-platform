from __future__ import annotations

import os
import subprocess
import sys
import threading
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


def test_owned_value_release_clears_value_and_lease() -> None:
    policy = capture_test_policy(available_memory_bytes=16 * MIB)
    with runtime_resource_policy_scope(policy):
        owned = resources.OwnedValue(
            {"payload": True},
            policy.lease("test.owned", 128),
        )
        assert policy.used_memory_bytes == 128

        owned.release(owner="test.owned")

        assert owned.value is None
        assert policy.used_memory_bytes == 0


def test_soft_target_is_not_an_exact_hard_cap_but_explicit_cap_is() -> None:
    soft = capture_test_policy(available_memory_bytes=1_000_000)
    with runtime_resource_policy_scope(soft):
        assert soft.operation_headroom_bytes == soft.memory_budget_bytes
        lease = soft.lease("test.soft.target", soft.memory_budget_bytes + 1)
        assert soft.used_memory_bytes == 700_001
        assert soft.planning_headroom_bytes == 0
        assert soft.operation_headroom_bytes == 299_999
        lease.grow(300_000)
        assert soft.operation_headroom_bytes == 0
        lease.close()

    high_cap = capture_test_policy(
        available_memory_bytes=1_000,
        explicit_cap_bytes=900,
    )
    with runtime_resource_policy_scope(high_cap):
        assert high_cap.memory_budget_bytes == 700
        assert high_cap.operation_headroom_bytes == 700
        lease = high_cap.lease("test.high.cap", 701)
        assert high_cap.planning_headroom_bytes == 0
        assert high_cap.operation_headroom_bytes == 199
        lease.close()

    capped = capture_test_policy(
        available_memory_bytes=1_000_000,
        explicit_cap_bytes=600_000,
    )
    with runtime_resource_policy_scope(capped):
        assert capped.operation_headroom_bytes == capped.explicit_memory_cap_bytes
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            capped.lease("test.explicit.cap", 600_001)
        lease = capped.lease("test.explicit.cap", 600_000)
        assert capped.operation_headroom_bytes == 0
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            lease.grow(1)
        lease.close()


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

    monkeypatch.setattr(resources._MachineLock, "acquire", acquire)
    monkeypatch.setattr(resources, "_available_memory_bytes", available)
    return events, locks


def test_root_opener_locks_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    events, locks = _patch_capture(monkeypatch)

    with runtime_resource_policy_scope() as policy:
        assert policy.state == "ACTIVE"

    assert events == ["lock", "available"]
    assert locks[0].closed


def test_budget_captures_one_available_and_uses_one_planning_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = 10 * GIB + 1
    _patch_capture(monkeypatch, available_memory_bytes=available)

    with runtime_resource_policy_scope() as policy:
        assert policy.captured_available_memory_bytes == available
        target = policy.memory_budget_bytes
        assert 0 < target < available
        assert policy.used_memory_bytes == 0
        assert policy.planning_headroom_bytes == target
        assert policy.to_payload() == {
            "captured_available_memory_bytes": available,
            "memory_budget_bytes": target,
            "explicit_memory_cap_bytes": None,
            "used_memory_bytes": 0,
            "planning_headroom_bytes": target,
        }


def test_explicit_cap_only_lowers_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    available = 10 * GIB
    _patch_capture(monkeypatch, available_memory_bytes=available)
    with runtime_resource_policy_scope() as baseline:
        expected_budget = baseline.memory_budget_bytes

    with runtime_resource_policy_scope(
        explicit_cap_bytes=expected_budget + MIB
    ) as high:
        assert high.memory_budget_bytes == expected_budget
    with runtime_resource_policy_scope(explicit_cap_bytes=expected_budget - MIB) as low:
        assert low.memory_budget_bytes == expected_budget - MIB


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


@pytest.mark.parametrize("explicit_policy", (False, True))
def test_independent_thread_scope_is_busy_then_can_reenter(
    explicit_policy: bool,
) -> None:
    entered = threading.Event()
    attempted = threading.Event()
    finished = threading.Event()
    outcomes: list[str] = []

    def contender() -> None:
        assert entered.wait(timeout=5)
        assert resources.current_runtime_resource_policy() is None
        try:
            with runtime_resource_policy_scope(
                root if explicit_policy else resources.current_runtime_resource_policy()
            ):
                outcomes.append("incorrectly_shared")
        except RuntimeError as exc:
            outcomes.append(str(exc))
        finally:
            attempted.set()
        assert finished.wait(timeout=5)
        with runtime_resource_policy_scope() as subsequent:
            with subsequent.lease("subsequent", 1):
                outcomes.append("reentered")
        assert subsequent.state == "CLOSED"

    thread = threading.Thread(target=contender)
    thread.start()
    try:
        with runtime_resource_policy_scope() as root:
            with root.lease("first", 1):
                entered.set()
                assert attempted.wait(timeout=5)
                assert root.state == "ACTIVE"
                assert root.used_memory_bytes == 1
        assert root.state == "CLOSED"
        assert root.used_memory_bytes == 0
    finally:
        finished.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcomes == ["runtime_data_scale_lock_busy", "reentered"]


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


def test_concurrent_lease_admission_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=MIB,
        explicit_cap_bytes=100,
    )
    checks = threading.Barrier(2)
    original_used = resources.RuntimeResourcePolicy.used_memory_bytes.fget

    def observed_used(root: resources.RuntimeResourcePolicy) -> int:
        ledger_lock = getattr(root, "_ledger_lock", None)
        assert original_used is not None
        value = original_used(root)
        if (
            threading.current_thread() is not threading.main_thread()
            and (ledger_lock is None or not ledger_lock._is_owned())  # noqa: SLF001
        ):
            checks.wait(timeout=2)
        return value

    monkeypatch.setattr(
        resources.RuntimeResourcePolicy,
        "used_memory_bytes",
        property(observed_used),
    )
    outcomes: list[str] = []
    leases: list[resources.MemoryLease] = []

    def admit() -> None:
        try:
            leases.append(policy.lease("concurrent", 80))
        except MemoryError:
            outcomes.append("rejected")
        else:
            outcomes.append("accepted")

    with runtime_resource_policy_scope(policy):
        threads = [threading.Thread(target=admit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)
        for lease in leases:
            lease.close()

    assert sorted(outcomes) == ["accepted", "rejected"]
    assert policy.used_memory_bytes == 0


def test_concurrent_lease_growth_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=MIB,
        explicit_cap_bytes=100,
    )
    checks = threading.Barrier(2)
    original_used = resources.RuntimeResourcePolicy.used_memory_bytes.fget

    def observed_used(root: resources.RuntimeResourcePolicy) -> int:
        ledger_lock = getattr(root, "_ledger_lock", None)
        assert original_used is not None
        value = original_used(root)
        if (
            threading.current_thread() is not threading.main_thread()
            and (ledger_lock is None or not ledger_lock._is_owned())  # noqa: SLF001
        ):
            checks.wait(timeout=2)
        return value

    monkeypatch.setattr(
        resources.RuntimeResourcePolicy,
        "used_memory_bytes",
        property(observed_used),
    )
    outcomes: list[str] = []

    def grow(lease: resources.MemoryLease) -> None:
        try:
            lease.grow(50)
        except MemoryError:
            outcomes.append("rejected")
        else:
            outcomes.append("accepted")

    with runtime_resource_policy_scope(policy):
        first = policy.lease("first", 20)
        second = policy.lease("second", 20)
        threads = [
            threading.Thread(target=grow, args=(first,)),
            threading.Thread(target=grow, args=(second,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)
        first.close()
        second.close()

    assert sorted(outcomes) == ["accepted", "rejected"]
    assert policy.used_memory_bytes == 0


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


@pytest.mark.parametrize(
    "child_limit,parent_limit,current,expected",
    [
        ("900", "800", "200", 600),
        ("max", "800", "200", 600),
        (None, None, "0", 1000),
        ("2000", "max", "0", 1000),
        ("200", "max", "200", 0),
        ("100", "max", "200", 0),
        ("invalid", "max", "0", None),
    ],
)
def test_cgroup_memory_capture_respects_visible_parent_and_unlimited_bounds(
    tmp_path, monkeypatch, child_limit, parent_limit, current, expected
):
    from types import SimpleNamespace

    import psutil

    root = tmp_path / "cgroup"
    child = root / "team" / "job"
    child.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("memory cpu")
    membership = tmp_path / "membership"
    membership.write_text("0::/team/job\n")
    for directory, limit in ((child, child_limit), (child.parent, parent_limit)):
        if limit is not None:
            (directory / "memory.max").write_text(limit)
            (directory / "memory.current").write_text(current)
    original_path = Path

    def paths(value):
        return {
            "/sys/fs/cgroup": root,
            "/proc/self/cgroup": membership,
        }.get(str(value), original_path(value))

    monkeypatch.setattr(resources, "Path", paths)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=1000)
    )
    if expected is None:
        with pytest.raises(
            RuntimeError, match="runtime_available_memory_discovery_failed"
        ):
            resources._available_memory_bytes()
    elif expected == 0:
        with pytest.raises(
            ValueError, match="invalid_runtime_resource_available_memory"
        ):
            resources._available_memory_bytes()
    else:
        assert resources._available_memory_bytes() == expected


@pytest.mark.parametrize(
    "quota,parent_quota,affinity,expected",
    [
        ("1400000 100000", "max 100000", 32, 14),
        ("max 100000", "250000 100000", 32, 2),
        ("1400000 100000", "max 100000", 3, 3),
        ("50000 100000", "max 100000", 32, 1),
        ("max 100000", "max 100000", 32, 16),
        ("invalid", "max 100000", 32, None),
        ("100000 0", "max 100000", 32, None),
    ],
)
def test_model_threads_bind_affinity_and_cgroup_bandwidth(
    tmp_path, monkeypatch, quota, parent_quota, affinity, expected
):
    import psutil

    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.supervised import (
        resolved_model_hyperparameters,
    )

    root = tmp_path / "cgroup"
    child = root / "job"
    child.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("memory cpu")
    (root / "cpu.max").write_text(parent_quota)
    (child / "cpu.max").write_text(quota)
    membership = tmp_path / "membership"
    membership.write_text("0::/job\n")
    original_path = Path
    monkeypatch.setattr(
        resources,
        "Path",
        lambda value: {"/sys/fs/cgroup": root, "/proc/self/cgroup": membership}.get(
            str(value), original_path(value)
        ),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _: set(range(affinity)), raising=False
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda **_: 16)
    spec = ModelSpec(
        model_id="capacity_model",
        model_algorithm="random_forest_regressor",
        input_feature_columns=("feature",),
        label_column="label",
        hyperparameters={"n_jobs": 16},
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="runtime_cpu_capacity_discovery_failed"):
            resolved_model_hyperparameters(spec)
    else:
        assert resources.model_cpu_capacity() == (16, expected)
        assert resolved_model_hyperparameters(spec)["n_jobs"] == expected


def test_non_linux_resource_discovery_keeps_host_capacity(monkeypatch):
    from types import SimpleNamespace

    import psutil

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=1000)
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda **_: 8)
    assert resources._available_memory_bytes() == 1000
    assert resources.model_cpu_capacity() == (8, 8)


@pytest.mark.parametrize(
    "membership", ["0::/\n", "0::/missing\n", "0::/../escape\n", "1:cpu:/\n"]
)
def test_cgroup_membership_is_bound_to_visible_hierarchy(
    tmp_path, monkeypatch, membership
):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("memory cpu")
    member_file = tmp_path / "membership"
    member_file.write_text(membership)
    original_path = Path
    monkeypatch.setattr(
        resources,
        "Path",
        lambda value: {"/sys/fs/cgroup": root, "/proc/self/cgroup": member_file}.get(
            str(value), original_path(value)
        ),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    if membership == "0::/\n":
        assert list(resources._cgroup_v2_directories()) == [root]
        (root / "cgroup.controllers").unlink()
        assert list(resources._cgroup_v2_directories()) == []
    else:
        with pytest.raises((ValueError, RuntimeError), match="cgroup_membership"):
            list(resources._cgroup_v2_directories())


@pytest.mark.parametrize(
    "requested,expected",
    [(-1, 4), (-2, 3), (1, 1), (8, 4), (0, None), (-5, None), (17, None), (True, None)],
)
def test_model_thread_rule_uses_capacity_without_rewriting_request(requested, expected):
    from quant_research_platform.contracts.models import _resolved_model_thread_count

    if expected is None:
        with pytest.raises(ValueError):
            _resolved_model_thread_count(
                requested, physical_cores=16, available_cores=4
            )
    else:
        assert (
            _resolved_model_thread_count(
                requested, physical_cores=16, available_cores=4
            )
            == expected
        )
