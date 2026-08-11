"""Runtime-owned resource policy and payload diagnostics."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import get_ident

import pandas as pd

MIB = 1024 * 1024
_TASK_MEMORY_CEILING_NUMERATOR = 4
_TASK_MEMORY_CEILING_DENOMINATOR = 5
_MAX_PAYLOAD_TARGET_BYTES = 256 * MIB


@dataclass(slots=True)
class _ReservationLedger:
    state: str = "READY"
    owner_thread_id: int | None = None
    by_owner: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(self.by_owner.values())

    def reserve(self, owner: str, reserve_bytes: int, *, budget_bytes: int) -> None:
        if self.total_bytes + reserve_bytes > budget_bytes:
            raise MemoryError("runtime_memory_budget_exceeded")
        self.by_owner[owner] = self.by_owner.get(owner, 0) + reserve_bytes

    def release(self, owner: str, reserve_bytes: int) -> None:
        remaining = self.by_owner.get(owner, 0) - reserve_bytes
        if remaining < 0:
            raise RuntimeError("runtime_memory_reservation_underflow")
        if remaining:
            self.by_owner[owner] = remaining
        else:
            self.by_owner.pop(owner, None)


@dataclass(frozen=True, init=False, eq=False, slots=True)
class RuntimeResourcePolicy:
    """Immutable one-shot resource admission shared by one root operation."""

    captured_available_memory_bytes: int
    rss0_bytes: int
    task_memory_budget_bytes: int
    payload_target_bytes: int
    _ledger: _ReservationLedger

    @classmethod
    def capture(
        cls,
        *,
        available_memory_bytes: int | None = None,
        rss_bytes: int | None = None,
        explicit_cap_bytes: int | None = None,
    ) -> RuntimeResourcePolicy:
        """Capture available memory and RSS exactly once before root allocation."""

        available = (
            _available_memory_bytes()
            if available_memory_bytes is None
            else _exact_positive_int(
                available_memory_bytes,
                error="invalid_runtime_resource_available_memory",
            )
        )
        rss0 = (
            _process_rss_bytes()
            if rss_bytes is None
            else _exact_nonnegative_int(
                rss_bytes,
                error="invalid_runtime_resource_rss",
            )
        )
        hard_budget = (
            available
            * _TASK_MEMORY_CEILING_NUMERATOR
            // _TASK_MEMORY_CEILING_DENOMINATOR
        )
        if explicit_cap_bytes is not None:
            hard_budget = min(
                hard_budget,
                _exact_positive_int(
                    explicit_cap_bytes,
                    error="invalid_runtime_resource_explicit_cap",
                ),
            )
        if hard_budget <= 0:
            raise ValueError("invalid_runtime_resource_available_memory")
        payload_target = min(
            hard_budget,
            max(MIB, min(_MAX_PAYLOAD_TARGET_BYTES, hard_budget // 8)),
        )
        policy = object.__new__(cls)
        object.__setattr__(policy, "captured_available_memory_bytes", available)
        object.__setattr__(policy, "rss0_bytes", rss0)
        object.__setattr__(policy, "task_memory_budget_bytes", hard_budget)
        object.__setattr__(policy, "payload_target_bytes", payload_target)
        object.__setattr__(policy, "_ledger", _ReservationLedger())
        return policy

    @property
    def reserved_memory_bytes(self) -> int:
        return self._ledger.total_bytes

    @contextmanager
    def reserve(self, owner: str, reserve_bytes: int) -> Iterator[None]:
        """Reserve one owner-local upper bound against this root budget."""

        if not isinstance(owner, str) or not owner:
            raise ValueError("invalid_runtime_memory_reservation_owner")
        reserve = _exact_nonnegative_int(
            reserve_bytes,
            error="invalid_runtime_memory_reservation",
        )
        if (
            _ACTIVE_RUNTIME_RESOURCE_POLICY is not self
            or self._ledger.state != "ACTIVE"
        ):
            raise RuntimeError("runtime_resource_policy_not_active")
        if self._ledger.owner_thread_id != get_ident():
            raise RuntimeError("runtime_resource_policy_wrong_owner")
        self._ledger.reserve(
            owner,
            reserve,
            budget_bytes=self.task_memory_budget_bytes,
        )
        try:
            yield
        finally:
            primary = sys.exception()
            try:
                self._ledger.release(owner, reserve)
            except BaseException:
                if primary is None:
                    raise

    def to_payload(self) -> dict[str, int]:
        """Return the exact private runtime resource payload."""

        return {
            "captured_available_memory_bytes": self.captured_available_memory_bytes,
            "rss0_bytes": self.rss0_bytes,
            "task_memory_budget_bytes": self.task_memory_budget_bytes,
            "payload_target_bytes": self.payload_target_bytes,
        }


_ACTIVE_RUNTIME_RESOURCE_POLICY: RuntimeResourcePolicy | None = None


def current_runtime_resource_policy() -> RuntimeResourcePolicy | None:
    """Return the active policy without admitting a new root operation."""

    return _ACTIVE_RUNTIME_RESOURCE_POLICY


def bind_runtime_resource_policy(
    explicit: RuntimeResourcePolicy | None,
) -> RuntimeResourcePolicy | None:
    """Bind a lazy operation to its exact creation-time policy."""

    if explicit is not None and type(explicit) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    active = current_runtime_resource_policy()
    if active is not None and explicit is not None and explicit is not active:
        raise RuntimeError("runtime_resource_policy_conflict")
    return active if active is not None else explicit


def active_runtime_resource_policy() -> RuntimeResourcePolicy:
    """Return the active snapshot or discover one for the caller to capture."""

    policy = current_runtime_resource_policy()
    if policy is None:
        return RuntimeResourcePolicy.capture()
    return policy


@contextmanager
def runtime_resource_policy_scope(
    policy: RuntimeResourcePolicy,
) -> Iterator[RuntimeResourcePolicy]:
    """Expose one explicit process-local root policy to nested synchronous stages."""

    global _ACTIVE_RUNTIME_RESOURCE_POLICY
    if type(policy) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    active = current_runtime_resource_policy()
    if active is not None:
        if active is not policy:
            raise RuntimeError("runtime_resource_policy_conflict")
        if policy._ledger.owner_thread_id != get_ident():
            raise RuntimeError("runtime_resource_policy_wrong_owner")
        yield policy
        return
    if policy._ledger.state == "RETIRED":
        raise RuntimeError("runtime_resource_policy_retired")
    if policy._ledger.state != "READY":
        raise RuntimeError("runtime_resource_policy_already_active")
    policy._ledger.state = "ACTIVE"
    policy._ledger.owner_thread_id = get_ident()
    _ACTIVE_RUNTIME_RESOURCE_POLICY = policy
    try:
        yield policy
    finally:
        primary = sys.exception()
        leaked = policy.reserved_memory_bytes
        policy._ledger.state = "RETIRED"
        policy._ledger.owner_thread_id = None
        _ACTIVE_RUNTIME_RESOURCE_POLICY = None
        if primary is None and leaked:
            raise RuntimeError("runtime_memory_reservation_leaked")


def pandas_frame_payload_bytes(frame: pd.DataFrame) -> int:
    """Return the sole Runtime-approved Pandas payload diagnostic."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("invalid_pandas_frame_payload")
    return int(frame.memory_usage(index=True, deep=True).sum())


def require_free_disk_headroom(path: Path, *, reserve_bytes: int) -> None:
    """Require owner-specified free-space headroom on the target volume."""

    reserve = _exact_nonnegative_int(
        reserve_bytes,
        error="invalid_runtime_resource_disk_reserve",
    )
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if shutil.disk_usage(candidate.resolve()).free < reserve:
        raise RuntimeError("insufficient_free_disk")


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


def _process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        rss = psutil.Process().memory_info().rss
    except Exception:
        raise RuntimeError("runtime_rss_discovery_failed") from None
    return _exact_nonnegative_int(rss, error="invalid_runtime_resource_rss")


def _exact_positive_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(error)
    return value


def _exact_nonnegative_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(error)
    return value


__all__ = [
    "RuntimeResourcePolicy",
    "active_runtime_resource_policy",
    "pandas_frame_payload_bytes",
    "require_free_disk_headroom",
    "runtime_resource_policy_scope",
]
