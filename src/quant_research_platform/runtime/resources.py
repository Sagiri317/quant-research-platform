"""Runtime-owned resource policy and payload diagnostics."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
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


def _build_policy_factory():
    issued_policies: WeakKeyDictionary[
        RuntimeResourcePolicy,
        tuple[int, int],
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
        issued_policies[policy] = (task_budget, payload_target)
        return policy

    def issued_values(policy: object) -> tuple[int, int]:
        if type(policy) is not RuntimeResourcePolicy:
            raise TypeError("invalid_runtime_resource_policy")
        expected = issued_policies.get(policy)
        if expected is None:
            raise TypeError("invalid_runtime_resource_policy")
        return expected

    return factory, issued_values


_runtime_resource_policy_factory, _issued_policy_values = _build_policy_factory()
_ACTIVE_RUNTIME_RESOURCE_POLICY: ContextVar[RuntimeResourcePolicy | None] = (
    ContextVar("qrp_active_runtime_resource_policy", default=None)
)


def active_runtime_resource_policy() -> RuntimeResourcePolicy:
    """Return the active snapshot or discover one for the caller to capture."""

    policy = _ACTIVE_RUNTIME_RESOURCE_POLICY.get()
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
    policy_context_handle = _ACTIVE_RUNTIME_RESOURCE_POLICY.set(policy)
    try:
        yield policy
    finally:
        try:
            _issued_policy_values(policy)
        finally:
            _ACTIVE_RUNTIME_RESOURCE_POLICY.reset(policy_context_handle)


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
