"""Runtime-owned local resource admission and lifetime ownership."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if os.name == "nt":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    import pandas as pd

MIB = 1024 * 1024
_MEMORY_BUDGET_NUMERATOR = 70
_MEMORY_BUDGET_DENOMINATOR = 100


@dataclass(slots=True)
class _MachineLock:
    fd: int
    closed: bool = False

    @classmethod
    def acquire(cls) -> _MachineLock:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(_machine_lock_path(), flags, 0o600)
        except OSError:
            raise RuntimeError("runtime_data_scale_lock_failed") from None
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError("runtime_data_scale_lock_busy") from None
        return cls(fd=fd)

    def close(self) -> None:
        if self.closed:
            return
        unlock_error: BaseException | None = None
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            if os.name == "nt":
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        except BaseException as exc:
            unlock_error = exc
        finally:
            try:
                os.close(self.fd)
            except BaseException as close_error:
                if unlock_error is None:
                    unlock_error = close_error
            self.closed = True
        if unlock_error is not None:
            raise RuntimeError(
                "runtime_data_scale_lock_release_failed"
            ) from unlock_error


@dataclass(slots=True, eq=False)
class MemoryLease:
    owner: str
    reserved_bytes: int
    active: bool
    _root: RuntimeResourcePolicy = field(repr=False)

    def __enter__(self) -> MemoryLease:
        self._root._require_lease_owner(self)
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: object,
    ) -> bool:
        try:
            self.close()
        except BaseException as release_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                "runtime memory lease release also failed: "
                f"{type(release_error).__name__}"
            )
        return False

    def transfer(self, new_owner: str) -> None:
        self._root._transfer_lease(self, new_owner)

    def grow(self, additional_bytes: int) -> None:
        self._root._require_lease_owner(self)
        additional = _exact_nonnegative_int(
            additional_bytes,
            error="invalid_runtime_memory_lease_bytes",
        )
        self._root._grow_lease(self, additional)

    def shrink_to(self, actual_bytes: int) -> None:
        self._root._shrink_lease(self, actual_bytes)

    def close(self) -> None:
        if not self.active:
            return
        self._root._close_lease(self)


@dataclass(slots=True)
class OwnedValue[T]:
    value: T
    lease: MemoryLease

    def release(self, *, owner: str) -> None:
        self.value = None
        _close_lease_preserving_primary(self.lease, owner=owner)


def _close_lease_preserving_primary(lease: MemoryLease, *, owner: str) -> None:
    primary = sys.exception()
    try:
        lease.close()
    except BaseException as cleanup_error:
        if primary is None:
            raise
        primary.add_note(
            f"{owner} lease close also failed: {type(cleanup_error).__name__}"
        )


@dataclass(frozen=True, init=False, eq=False, slots=True)
class RuntimeResourcePolicy:
    """One soft host planning target and approximate owned-memory ledger."""

    captured_available_memory_bytes: int
    memory_budget_bytes: int
    explicit_memory_cap_bytes: int | None
    state: str
    _lock: _MachineLock
    _ledger_lock: threading.RLock
    _leases: set[MemoryLease] = field(default_factory=set, repr=False)

    def __new__(cls) -> RuntimeResourcePolicy:
        raise TypeError("runtime_resource_policy_capture_required")

    @property
    def used_memory_bytes(self) -> int:
        with self._ledger_lock:
            return self._used_memory_bytes()

    @property
    def planning_headroom_bytes(self) -> int:
        with self._ledger_lock:
            return max(0, self.memory_budget_bytes - self._used_memory_bytes())

    @property
    def operation_headroom_bytes(self) -> int:
        """Return the bounded headroom for one managed operation."""

        with self._ledger_lock:
            used = self._used_memory_bytes()
            planning = self.memory_budget_bytes - used
            if planning > 0:
                return planning
            if self.explicit_memory_cap_bytes is None:
                return self.memory_budget_bytes
            return max(
                0,
                min(
                    self.memory_budget_bytes,
                    self.explicit_memory_cap_bytes - used,
                ),
            )

    def lease(self, owner: str, reserved_bytes: int) -> MemoryLease:
        with self._ledger_lock:
            self._require_active_root()
            valid_owner = _required_owner(owner)
            reserved = _exact_nonnegative_int(
                reserved_bytes,
                error="invalid_runtime_memory_lease_bytes",
            )
            if (
                self.explicit_memory_cap_bytes is not None
                and self._used_memory_bytes() + reserved
                > self.explicit_memory_cap_bytes
            ):
                raise MemoryError("runtime_memory_budget_exceeded")
            leased = MemoryLease(valid_owner, reserved, True, self)
            self._leases.add(leased)
            return leased

    def to_payload(self) -> dict[str, int | None]:
        with self._ledger_lock:
            used = self._used_memory_bytes()
            return {
                "captured_available_memory_bytes": self.captured_available_memory_bytes,
                "memory_budget_bytes": self.memory_budget_bytes,
                "explicit_memory_cap_bytes": self.explicit_memory_cap_bytes,
                "used_memory_bytes": used,
                "planning_headroom_bytes": max(
                    0,
                    self.memory_budget_bytes - used,
                ),
            }

    def _require_active_root(self) -> None:
        if self.state != "ACTIVE" or _ACTIVE_RUNTIME_RESOURCE_POLICY is not self:
            raise RuntimeError("runtime_resource_policy_not_active")

    def _require_lease_owner(self, lease: MemoryLease) -> None:
        with self._ledger_lock:
            if lease._root is not self or not lease.active:
                raise RuntimeError("runtime_memory_lease_not_active")

    def _grow_lease(self, lease: MemoryLease, additional_bytes: int) -> None:
        with self._ledger_lock:
            self._require_active_root()
            self._require_lease_owner(lease)
            if self.explicit_memory_cap_bytes is not None and (
                self._used_memory_bytes() + additional_bytes
                > self.explicit_memory_cap_bytes
            ):
                raise MemoryError("runtime_memory_budget_exceeded")
            lease.reserved_bytes += additional_bytes

    def _shrink_lease(self, lease: MemoryLease, actual_bytes: int) -> None:
        with self._ledger_lock:
            self._require_lease_owner(lease)
            actual = _exact_nonnegative_int(
                actual_bytes,
                error="invalid_runtime_memory_lease_bytes",
            )
            if actual > lease.reserved_bytes:
                raise ValueError("invalid_runtime_memory_lease_bytes")
            lease.reserved_bytes = actual

    def _transfer_lease(self, lease: MemoryLease, new_owner: str) -> None:
        with self._ledger_lock:
            self._require_lease_owner(lease)
            lease.owner = _required_owner(new_owner)

    def _close_lease(self, lease: MemoryLease) -> None:
        with self._ledger_lock:
            if not lease.active:
                return
            if self.state == "CLOSED":
                raise RuntimeError("runtime_memory_lease_not_active")
            if self.state == "ACTIVE":
                self._require_active_root()
            self._leases.discard(lease)
            lease.active = False

    def _used_memory_bytes(self) -> int:
        return sum(lease.reserved_bytes for lease in self._leases)


_ACTIVE_RUNTIME_RESOURCE_POLICY: RuntimeResourcePolicy | None = None
_PROCESS_POISONED = False


def current_runtime_resource_policy() -> RuntimeResourcePolicy | None:
    return _ACTIVE_RUNTIME_RESOURCE_POLICY


@contextmanager
def runtime_resource_policy_scope(
    policy: RuntimeResourcePolicy | None = None,
    *,
    explicit_cap_bytes: int | None = None,
) -> Iterator[RuntimeResourcePolicy]:
    global _ACTIVE_RUNTIME_RESOURCE_POLICY
    active = current_runtime_resource_policy()
    owns_root = active is None
    if policy is not None:
        if explicit_cap_bytes is not None:
            raise ValueError("invalid_runtime_resource_explicit_cap")
        if type(policy) is not RuntimeResourcePolicy:
            raise TypeError("invalid_runtime_resource_policy")
        if active is not None and active is not policy:
            raise RuntimeError("runtime_resource_policy_conflict")
    elif active is not None:
        policy = active
        owns_root = False
        if explicit_cap_bytes is not None:
            raise RuntimeError("runtime_resource_policy_conflict")
    else:
        policy = _capture_runtime_resource_policy(
            explicit_cap_bytes=explicit_cap_bytes,
        )
    assert policy is not None
    if active is None and policy.state != "ACTIVE":
        raise RuntimeError("runtime_resource_policy_not_active")
    if active is not None and active is not policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    if policy.state != "ACTIVE":
        raise RuntimeError("runtime_resource_policy_not_active")
    if active is None:
        _ACTIVE_RUNTIME_RESOURCE_POLICY = policy
    try:
        yield policy
    finally:
        primary = sys.exception()
        if owns_root:
            _close_runtime_resource_policy(policy, primary=primary)


def _close_runtime_resource_policy(
    policy: RuntimeResourcePolicy,
    *,
    primary: BaseException | None,
) -> None:
    global _ACTIVE_RUNTIME_RESOURCE_POLICY, _PROCESS_POISONED

    outstanding = policy.used_memory_bytes
    if outstanding:
        object.__setattr__(policy, "state", "POISONED")
        _PROCESS_POISONED = True
        if _ACTIVE_RUNTIME_RESOURCE_POLICY is policy:
            _ACTIVE_RUNTIME_RESOURCE_POLICY = None
        if primary is None:
            raise RuntimeError("runtime_memory_lease_leaked")
        primary.add_note("runtime memory lease leaked")
        return

    try:
        policy._lock.close()
    except BaseException as exc:
        object.__setattr__(policy, "state", "POISONED")
        _PROCESS_POISONED = True
        if _ACTIVE_RUNTIME_RESOURCE_POLICY is policy:
            _ACTIVE_RUNTIME_RESOURCE_POLICY = None
        if primary is not None:
            primary.add_note(
                f"runtime machine-lock release also failed: {type(exc).__name__}"
            )
            return
        raise RuntimeError("runtime_machine_lock_release_failed") from exc
    object.__setattr__(policy, "state", "CLOSED")
    if _ACTIVE_RUNTIME_RESOURCE_POLICY is policy:
        _ACTIVE_RUNTIME_RESOURCE_POLICY = None


def pandas_frame_payload_bytes(frame: pd.DataFrame) -> int:
    """Return the Runtime-approved Pandas payload diagnostic."""

    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("invalid_pandas_frame_payload")
    return int(frame.memory_usage(index=True, deep=True).sum())


def require_free_disk_headroom(path: Path, *, reserve_bytes: int) -> int:
    """Require owner-specified free-space headroom on the target volume."""

    reserve = _exact_nonnegative_int(
        reserve_bytes,
        error="invalid_runtime_resource_disk_reserve",
    )
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    free = shutil.disk_usage(candidate.resolve()).free
    if free < reserve:
        raise RuntimeError("insufficient_free_disk")
    return free


def _acquire_machine_lock() -> _MachineLock:
    return _MachineLock.acquire()


def _capture_runtime_resource_policy(
    *,
    explicit_cap_bytes: int | None = None,
) -> RuntimeResourcePolicy:
    """Create the one root owned by ``runtime_resource_policy_scope``."""

    if _PROCESS_POISONED:
        raise RuntimeError("runtime_data_scale_process_poisoned")
    lock = _acquire_machine_lock()
    try:
        explicit_cap = (
            None
            if explicit_cap_bytes is None
            else _exact_positive_int(
                explicit_cap_bytes,
                error="invalid_runtime_resource_explicit_cap",
            )
        )
        available = _exact_positive_int(
            _available_memory_bytes(),
            error="invalid_runtime_resource_available_memory",
        )
        budget = available * _MEMORY_BUDGET_NUMERATOR // _MEMORY_BUDGET_DENOMINATOR
        if explicit_cap is not None:
            budget = min(budget, explicit_cap)
        if budget <= 0:
            raise ValueError("invalid_runtime_resource_memory_budget")
        policy = object.__new__(RuntimeResourcePolicy)
        object.__setattr__(policy, "captured_available_memory_bytes", available)
        object.__setattr__(policy, "memory_budget_bytes", budget)
        object.__setattr__(policy, "explicit_memory_cap_bytes", explicit_cap)
        object.__setattr__(policy, "state", "ACTIVE")
        object.__setattr__(policy, "_lock", lock)
        object.__setattr__(policy, "_ledger_lock", threading.RLock())
        object.__setattr__(policy, "_leases", set())
        return policy
    except BaseException as primary:
        try:
            lock.close()
        except BaseException as release_error:
            primary.add_note(
                "runtime machine-lock release also failed: "
                f"{type(release_error).__name__}"
            )
        raise


def _machine_lock_path() -> Path:
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    user_hash = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f".qrp-data-scale-{user_hash}.lock"


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        available = psutil.virtual_memory().available
    except Exception:
        raise RuntimeError("runtime_available_memory_discovery_failed") from None
    return _exact_positive_int(
        available,
        error="invalid_runtime_resource_available_memory",
    )


def _required_owner(owner: object) -> str:
    if not isinstance(owner, str) or not owner:
        raise ValueError("invalid_runtime_memory_lease_owner")
    return owner


def _exact_positive_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(error)
    return value


def _exact_nonnegative_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(error)
    return value
