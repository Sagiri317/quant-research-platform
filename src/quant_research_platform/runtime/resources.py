"""Runtime-owned local resource admission and lifetime ownership."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
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

    def transfer(self, new_owner: str) -> None:
        self._root._require_lease_owner(self)
        self.owner = _required_owner(new_owner)

    def grow(self, additional_bytes: int) -> None:
        self._root._require_lease_owner(self)
        additional = _exact_nonnegative_int(
            additional_bytes,
            error="invalid_runtime_memory_lease_bytes",
        )
        self._root._grow_lease(self, additional)

    def shrink_to(self, actual_bytes: int) -> None:
        self._root._require_lease_owner(self)
        actual = _exact_nonnegative_int(
            actual_bytes,
            error="invalid_runtime_memory_lease_bytes",
        )
        if actual > self.reserved_bytes:
            raise ValueError("invalid_runtime_memory_lease_bytes")
        self.reserved_bytes = actual

    def close(self) -> None:
        if not self.active:
            return
        self._root._close_lease(self)


@dataclass(slots=True)
class _OwnedValue[T]:
    value: T
    lease: MemoryLease


type _OwnedFrame = _OwnedValue[pd.DataFrame]


@dataclass(frozen=True, init=False, eq=False, slots=True)
class RuntimeResourcePolicy:
    """One immutable data-scale root budget and mutable lease ledger."""

    captured_available_memory_bytes: int
    memory_budget_bytes: int
    state: str
    _lock: _MachineLock
    _leases: set[MemoryLease] = field(default_factory=set, repr=False)
    _cleanups: list[Callable[[], None]] = field(default_factory=list, repr=False)
    _scope_depth: int = field(default=0, repr=False)

    def __new__(cls) -> RuntimeResourcePolicy:
        raise TypeError("runtime_resource_policy_capture_required")

    @classmethod
    def capture(
        cls,
        *,
        explicit_cap_bytes: int | None = None,
    ) -> RuntimeResourcePolicy:
        """Acquire the data-scale lock, then capture available memory once."""

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
            budget = (
                available * _MEMORY_BUDGET_NUMERATOR
                // _MEMORY_BUDGET_DENOMINATOR
            )
            if explicit_cap is not None:
                budget = min(budget, explicit_cap)
            if budget <= 0:
                raise ValueError("invalid_runtime_resource_memory_budget")
            policy = object.__new__(cls)
            object.__setattr__(
                policy,
                "captured_available_memory_bytes",
                available,
            )
            object.__setattr__(policy, "memory_budget_bytes", budget)
            object.__setattr__(policy, "state", "ACTIVE")
            object.__setattr__(policy, "_lock", lock)
            object.__setattr__(policy, "_leases", set())
            object.__setattr__(policy, "_cleanups", [])
            object.__setattr__(policy, "_scope_depth", 0)
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

    @property
    def used_memory_bytes(self) -> int:
        return sum(lease.reserved_bytes for lease in self._leases)

    @property
    def available_memory_bytes(self) -> int:
        return self.memory_budget_bytes - self.used_memory_bytes

    @property
    def task_memory_budget_bytes(self) -> int:
        return self.memory_budget_bytes

    @property
    def reserved_memory_bytes(self) -> int:
        return self.used_memory_bytes

    @property
    def remaining_memory_bytes(self) -> int:
        return self.available_memory_bytes

    def lease(self, owner: str, reserved_bytes: int) -> MemoryLease:
        self._require_active_root()
        valid_owner = _required_owner(owner)
        reserved = _exact_nonnegative_int(
            reserved_bytes,
            error="invalid_runtime_memory_lease_bytes",
        )
        if self.used_memory_bytes + reserved > self.memory_budget_bytes:
            raise MemoryError("runtime_memory_budget_exceeded")
        leased = MemoryLease(valid_owner, reserved, True, self)
        self._leases.add(leased)
        return leased

    @contextmanager
    def reserve(self, owner: str, reserve_bytes: int) -> Iterator[None]:
        """Temporary bridge for callers not yet migrated to MemoryLease."""

        leased = self.lease(owner, reserve_bytes)
        try:
            yield
        finally:
            primary = sys.exception()
            try:
                leased.close()
            except BaseException as release_error:
                if primary is None:
                    raise
                primary.add_note(
                    "runtime memory reservation release also failed: "
                    f"{type(release_error).__name__}"
                )

    def register_cleanup(self, close: Callable[[], None]) -> None:
        self._require_active_root()
        self._cleanups.append(close)

    def to_payload(self) -> dict[str, int]:
        return {
            "captured_available_memory_bytes": self.captured_available_memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "used_memory_bytes": self.used_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
        }

    def memory_evidence(self) -> dict[str, str]:
        """Temporary bridge; release evidence is collected externally."""

        return {"evidence_scope": "external_release_only"}

    def assert_memory_within_budget(self) -> None:
        """Temporary bridge; production safety is structural admission."""

    def finish_memory_measurement(
        self,
        *,
        on_reject: Callable[[], None] | None = None,
    ) -> dict[str, str]:
        """Temporary bridge; external evidence has no publish authority."""

        del on_reject
        return self.memory_evidence()

    def _require_active_root(self) -> None:
        if self.state != "ACTIVE" or _ACTIVE_RUNTIME_RESOURCE_POLICY is not self:
            raise RuntimeError("runtime_resource_policy_not_active")

    def _require_lease_owner(self, lease: MemoryLease) -> None:
        if lease._root is not self or not lease.active:
            raise RuntimeError("runtime_memory_lease_not_active")

    def _grow_lease(self, lease: MemoryLease, additional_bytes: int) -> None:
        self._require_active_root()
        if self.used_memory_bytes + additional_bytes > self.memory_budget_bytes:
            raise MemoryError("runtime_memory_budget_exceeded")
        lease.reserved_bytes += additional_bytes

    def _close_lease(self, lease: MemoryLease) -> None:
        if self.state == "CLOSED":
            raise RuntimeError("runtime_memory_lease_not_active")
        if self.state == "ACTIVE":
            self._require_active_root()
        self._leases.discard(lease)
        lease.active = False


_ACTIVE_RUNTIME_RESOURCE_POLICY: RuntimeResourcePolicy | None = None
_PROCESS_POISONED = False


def current_runtime_resource_policy() -> RuntimeResourcePolicy | None:
    """Temporary bridge for callers not yet migrated to explicit roots."""

    return _ACTIVE_RUNTIME_RESOURCE_POLICY


def bind_runtime_resource_policy(
    explicit: RuntimeResourcePolicy | None,
    *,
    allow_implicit_active: bool = False,
) -> RuntimeResourcePolicy | None:
    """Temporary bridge for creation-time explicit policy binding."""

    if explicit is not None and type(explicit) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    if type(allow_implicit_active) is not bool:
        raise TypeError("invalid_runtime_resource_policy_binding")
    active = current_runtime_resource_policy()
    if active is not None and explicit is None and not allow_implicit_active:
        raise RuntimeError("runtime_resource_policy_competing_root")
    if active is not None and explicit is not None and explicit is not active:
        raise RuntimeError("runtime_resource_policy_conflict")
    return active if active is not None else explicit


def active_runtime_resource_policy() -> RuntimeResourcePolicy:
    """Temporary bridge that captures a root only when none is active."""

    policy = current_runtime_resource_policy()
    if policy is None:
        return RuntimeResourcePolicy.capture()
    return policy


@contextmanager
def runtime_resource_policy_scope(
    policy: RuntimeResourcePolicy,
) -> Iterator[RuntimeResourcePolicy]:
    global _ACTIVE_RUNTIME_RESOURCE_POLICY
    if type(policy) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    active = current_runtime_resource_policy()
    if active is not None and active is not policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    if policy.state != "ACTIVE":
        raise RuntimeError("runtime_resource_policy_not_active")
    object.__setattr__(policy, "_scope_depth", policy._scope_depth + 1)
    if active is None:
        _ACTIVE_RUNTIME_RESOURCE_POLICY = policy
    try:
        yield policy
    finally:
        primary = sys.exception()
        object.__setattr__(policy, "_scope_depth", policy._scope_depth - 1)
        if policy._scope_depth == 0:
            _close_runtime_resource_policy(policy, primary=primary)


def _close_runtime_resource_policy(
    policy: RuntimeResourcePolicy,
    *,
    primary: BaseException | None,
) -> None:
    global _ACTIVE_RUNTIME_RESOURCE_POLICY, _PROCESS_POISONED
    if _ACTIVE_RUNTIME_RESOURCE_POLICY is policy:
        _ACTIVE_RUNTIME_RESOURCE_POLICY = None

    cleanup_error: BaseException | None = None
    while policy._cleanups:
        close = policy._cleanups.pop()
        try:
            close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            if primary is not None:
                primary.add_note(
                    "runtime cleanup also failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    outstanding = policy.used_memory_bytes
    if cleanup_error is not None or outstanding:
        object.__setattr__(policy, "state", "POISONED")
        _PROCESS_POISONED = True
        if primary is None:
            if cleanup_error is not None:
                raise cleanup_error
            raise RuntimeError("runtime_memory_lease_leaked")
        return

    try:
        policy._lock.close()
    except BaseException as exc:
        object.__setattr__(policy, "state", "POISONED")
        _PROCESS_POISONED = True
        if primary is not None:
            primary.add_note(
                "runtime machine-lock release also failed: "
                f"{type(exc).__name__}"
            )
            return
        raise RuntimeError("runtime_machine_lock_release_failed") from exc
    object.__setattr__(policy, "state", "CLOSED")


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


__all__ = [
    "MemoryLease",
    "RuntimeResourcePolicy",
    "active_runtime_resource_policy",
    "pandas_frame_payload_bytes",
    "require_free_disk_headroom",
    "runtime_resource_policy_scope",
]
