"""Runtime-owned resource policy and payload diagnostics."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from weakref import WeakKeyDictionary

import pandas as pd

MIB = 1024 * 1024
_TASK_MEMORY_CEILING_NUMERATOR = 4
_TASK_MEMORY_CEILING_DENOMINATOR = 5
_RETIRED_RESOURCE_SETTINGS = frozenset(
    {
        "QRP_MEMORY_BUDGET_FRACTION",
        "QRP_MEMORY_BUDGET_MB",
        "QRP_CHUNK_TARGET_MB",
        "QRP_MEMORY_SAFETY_FACTOR",
        "QRP_MIN_FREE_DISK_MB",
    }
)


@dataclass(frozen=True, init=False, eq=False)
class RuntimeResourcePolicy:
    """Factory-issued immutable resource snapshot for one runtime task."""

    __slots__ = ("__weakref__",)

    task_memory_budget_bytes: int
    payload_target_bytes: int

    def __new__(cls, *_args: object, **_kwargs: object) -> RuntimeResourcePolicy:
        raise TypeError("runtime_resource_policy_factory_required")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("runtime_resource_policy_subclass_forbidden")

    @property
    def task_memory_budget_bytes(self) -> int:
        return _issued_policy_values(self)[0]

    @property
    def payload_target_bytes(self) -> int:
        return _issued_policy_values(self)[1]

    @classmethod
    def from_environment(
        cls,
        *,
        available_memory_bytes: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeResourcePolicy:
        """Resolve current settings through the sole policy factory."""

        if cls is not RuntimeResourcePolicy:
            raise TypeError("runtime_resource_policy_subclass_forbidden")
        return _runtime_resource_policy_factory(
            available_memory_bytes=available_memory_bytes,
            environ=environ,
        )

    def to_payload(self) -> dict[str, int]:
        """Return the exact private runtime resource payload."""

        task_memory_budget_bytes, payload_target_bytes = _issued_policy_values(self)
        return {
            "task_memory_budget_bytes": task_memory_budget_bytes,
            "payload_target_bytes": payload_target_bytes,
        }


@dataclass(slots=True)
class _PolicyIssuance:
    task_memory_budget_bytes: int
    payload_target_bytes: int
    state: str
    issuer_pid: int


def _build_policy_factory():
    issued_policies: WeakKeyDictionary[
        RuntimeResourcePolicy,
        _PolicyIssuance,
    ] = WeakKeyDictionary()

    def factory(
        *,
        available_memory_bytes: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeResourcePolicy:
        env = os.environ if environ is None else environ
        if any(name in env for name in _RETIRED_RESOURCE_SETTINGS):
            raise ValueError("retired_runtime_resource_setting")
        available = (
            _available_memory_bytes()
            if available_memory_bytes is None
            else _exact_positive_int(
                available_memory_bytes,
                error="invalid_runtime_resource_available_memory",
            )
        )
        ceiling = (
            available
            * _TASK_MEMORY_CEILING_NUMERATOR
            // _TASK_MEMORY_CEILING_DENOMINATOR
        )
        if ceiling <= 0:
            raise ValueError("invalid_runtime_resource_available_memory")
        task_override = _megabytes_env(env, "QRP_TASK_MEMORY_BUDGET_MB")
        task_budget = ceiling if task_override is None else min(task_override, ceiling)
        payload_override = _megabytes_env(env, "QRP_PAYLOAD_TARGET_MB")
        payload_target = task_budget if payload_override is None else payload_override
        if payload_target > task_budget:
            raise ValueError(
                "runtime_payload_target_exceeds_task_memory_budget"
            )
        policy = object.__new__(RuntimeResourcePolicy)
        issued_policies[policy] = _PolicyIssuance(
            task_memory_budget_bytes=task_budget,
            payload_target_bytes=payload_target,
            state="ISSUED_UNUSED",
            issuer_pid=os.getpid(),
        )
        return policy

    def issuance(policy: object) -> _PolicyIssuance:
        if type(policy) is not RuntimeResourcePolicy:
            raise TypeError("invalid_runtime_resource_policy")
        record = issued_policies.get(policy)
        if record is None:
            raise TypeError("invalid_runtime_resource_policy")
        if record.issuer_pid != os.getpid():
            raise RuntimeError("runtime_resource_policy_wrong_process")
        return record

    def issued_values(policy: object) -> tuple[int, int]:
        record = issuance(policy)
        return record.task_memory_budget_bytes, record.payload_target_bytes

    def activate(policy: object) -> None:
        record = issuance(policy)
        if record.state == "RETIRED":
            raise RuntimeError("runtime_resource_policy_retired")
        if record.state != "ISSUED_UNUSED":
            raise RuntimeError("runtime_resource_policy_already_active")
        record.state = "ACTIVE"

    def require_active(policy: object) -> None:
        record = issuance(policy)
        if record.state == "RETIRED":
            raise RuntimeError("runtime_resource_policy_retired")
        if record.state != "ACTIVE":
            raise RuntimeError("runtime_resource_policy_not_active")

    def retire(policy: object) -> None:
        record = issuance(policy)
        if record.state != "ACTIVE":
            raise RuntimeError("runtime_resource_policy_not_active")
        record.state = "RETIRED"

    return factory, issued_values, activate, require_active, retire


(
    _runtime_resource_policy_factory,
    _issued_policy_values,
    _activate_issued_policy,
    _require_active_issued_policy,
    _retire_issued_policy,
) = _build_policy_factory()
_ACTIVE_RUNTIME_RESOURCE_POLICY: ContextVar[RuntimeResourcePolicy | None] = (
    ContextVar("qrp_active_runtime_resource_policy", default=None)
)
_RUNTIME_RESOURCE_OPERATION_LOCK = Lock()


def _reset_runtime_resource_state_after_fork() -> None:
    global _ACTIVE_RUNTIME_RESOURCE_POLICY, _RUNTIME_RESOURCE_OPERATION_LOCK
    _ACTIVE_RUNTIME_RESOURCE_POLICY = ContextVar(
        "qrp_active_runtime_resource_policy",
        default=None,
    )
    _RUNTIME_RESOURCE_OPERATION_LOCK = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_runtime_resource_state_after_fork)


def current_runtime_resource_policy() -> RuntimeResourcePolicy | None:
    """Return the active policy without admitting a new root operation."""

    policy = _ACTIVE_RUNTIME_RESOURCE_POLICY.get()
    if policy is not None:
        _require_active_issued_policy(policy)
    return policy


def bind_runtime_resource_policy(
    explicit: RuntimeResourcePolicy | None,
) -> RuntimeResourcePolicy | None:
    """Bind a lazy operation to its exact creation-time policy."""

    if explicit is not None:
        _issued_policy_values(explicit)
    active = current_runtime_resource_policy()
    if active is not None and explicit is not None and explicit is not active:
        raise RuntimeError("runtime_resource_policy_conflict")
    return active if active is not None else explicit


def active_runtime_resource_policy() -> RuntimeResourcePolicy:
    """Return the active snapshot or discover one for the caller to capture."""

    policy = current_runtime_resource_policy()
    if policy is None:
        return RuntimeResourcePolicy.from_environment()
    _issued_policy_values(policy)
    return policy


@contextmanager
def runtime_resource_policy_scope(
    policy: RuntimeResourcePolicy,
) -> Iterator[RuntimeResourcePolicy]:
    """Temporarily expose one factory-issued exact policy snapshot."""

    _issued_policy_values(policy)
    active = current_runtime_resource_policy()
    if active is not None:
        if active is not policy:
            raise RuntimeError("runtime_resource_policy_conflict")
        yield policy
        return
    if not _RUNTIME_RESOURCE_OPERATION_LOCK.acquire(blocking=False):
        raise RuntimeError("runtime_resource_operation_already_active")
    try:
        _activate_issued_policy(policy)
    except BaseException:
        try:
            _RUNTIME_RESOURCE_OPERATION_LOCK.release()
        except BaseException:
            pass
        raise
    try:
        policy_context_handle = _ACTIVE_RUNTIME_RESOURCE_POLICY.set(policy)
    except BaseException:
        try:
            _retire_issued_policy(policy)
        except BaseException:
            pass
        try:
            _RUNTIME_RESOURCE_OPERATION_LOCK.release()
        except BaseException:
            pass
        raise
    try:
        yield policy
    finally:
        primary = sys.exception()
        cleanup_failure: BaseException | None = None
        try:
            _retire_issued_policy(policy)
        except BaseException as exc:
            cleanup_failure = exc
        try:
            _ACTIVE_RUNTIME_RESOURCE_POLICY.reset(policy_context_handle)
        except BaseException as exc:
            if cleanup_failure is None:
                cleanup_failure = exc
        try:
            _RUNTIME_RESOURCE_OPERATION_LOCK.release()
        except BaseException as exc:
            if cleanup_failure is None:
                cleanup_failure = exc
        if primary is None and cleanup_failure is not None:
            raise cleanup_failure


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


def _megabytes_env(env: Mapping[str, str], name: str) -> int | None:
    value = env.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"invalid_runtime_resource_env: {name}")
    megabytes = int(value)
    if megabytes <= 0:
        raise ValueError(f"invalid_runtime_resource_env: {name}")
    return megabytes * MIB


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
