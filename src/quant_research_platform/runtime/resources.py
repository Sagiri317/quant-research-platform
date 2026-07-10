"""Runtime resource policy and chunk planning helpers."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from math import ceil, isfinite
from pathlib import Path

DEFAULT_AVAILABLE_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MEMORY_BUDGET_FRACTION = 0.80
DEFAULT_CHUNK_TARGET_FRACTION = 0.30
DEFAULT_CHUNK_TARGET_CAP_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MEMORY_SAFETY_FACTOR = 0.60
DEFAULT_MIN_FREE_DISK_BYTES = 0
GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024


class ChunkAxis(StrEnum):
    """Semantic axis used to split a dataset or build operation."""

    ASSET = "asset"
    TIME = "time"
    PARTITION = "partition"
    ROW_GROUP = "row_group"


@dataclass(frozen=True)
class RuntimeResourcePolicy:
    """Memory and disk limits for one local QRP runtime task."""

    available_memory_bytes: int
    memory_budget_fraction: float = DEFAULT_MEMORY_BUDGET_FRACTION
    chunk_target_fraction: float = DEFAULT_CHUNK_TARGET_FRACTION
    chunk_target_cap_bytes: int = DEFAULT_CHUNK_TARGET_CAP_BYTES
    chunk_target_bytes: int | None = None
    memory_safety_factor: float = DEFAULT_MEMORY_SAFETY_FACTOR
    min_free_disk_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES
    memory_budget_bytes_override: int | None = None

    def __post_init__(self) -> None:
        if int(self.available_memory_bytes) <= 0:
            raise ValueError("invalid_runtime_resource_available_memory")
        _require_fraction("memory_budget_fraction", self.memory_budget_fraction)
        _require_fraction("chunk_target_fraction", self.chunk_target_fraction)
        _require_fraction("memory_safety_factor", self.memory_safety_factor)
        if int(self.chunk_target_cap_bytes) <= 0:
            raise ValueError("invalid_runtime_resource_chunk_target_cap")
        if self.chunk_target_bytes is not None and int(self.chunk_target_bytes) <= 0:
            raise ValueError("invalid_runtime_resource_chunk_target")
        if int(self.min_free_disk_bytes) < 0:
            raise ValueError("invalid_runtime_resource_min_free_disk")
        if (
            self.memory_budget_bytes_override is not None
            and int(self.memory_budget_bytes_override) <= 0
        ):
            raise ValueError("invalid_runtime_resource_memory_budget")

    @classmethod
    def from_environment(
        cls,
        *,
        available_memory_bytes: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeResourcePolicy:
        env = os.environ if environ is None else environ
        available = (
            int(available_memory_bytes)
            if available_memory_bytes is not None
            else _available_memory_bytes()
        )
        fraction = _fraction_env(
            env,
            "QRP_MEMORY_BUDGET_FRACTION",
            DEFAULT_MEMORY_BUDGET_FRACTION,
        )
        budget = _bytes_env(env, "QRP_MEMORY_BUDGET_MB")
        memory_budget_bytes = (
            int(budget) if budget is not None else max(1, int(available * fraction))
        )
        target = _bytes_env(env, "QRP_CHUNK_TARGET_MB")
        if target is None:
            target = min(
                int(memory_budget_bytes * DEFAULT_CHUNK_TARGET_FRACTION),
                DEFAULT_CHUNK_TARGET_CAP_BYTES,
            )
        safety = _fraction_env(
            env,
            "QRP_MEMORY_SAFETY_FACTOR",
            DEFAULT_MEMORY_SAFETY_FACTOR,
        )
        min_disk = _bytes_env(env, "QRP_MIN_FREE_DISK_MB")
        policy = cls(
            available_memory_bytes=available,
            memory_budget_fraction=fraction,
            chunk_target_fraction=DEFAULT_CHUNK_TARGET_FRACTION,
            chunk_target_cap_bytes=DEFAULT_CHUNK_TARGET_CAP_BYTES,
            chunk_target_bytes=target,
            memory_safety_factor=safety,
            min_free_disk_bytes=(
                DEFAULT_MIN_FREE_DISK_BYTES if min_disk is None else min_disk
            ),
            memory_budget_bytes_override=budget,
        )
        policy.require_valid_chunk_budget()
        return policy

    @property
    def memory_budget_bytes(self) -> int:
        if self.memory_budget_bytes_override is not None:
            return self.memory_budget_bytes_override
        return max(1, int(self.available_memory_bytes * self.memory_budget_fraction))

    @property
    def effective_chunk_bytes(self) -> int:
        target = self._resolved_chunk_target_bytes()
        effective = max(1, int(target * self.memory_safety_factor))
        if effective > self.memory_budget_bytes:
            raise ValueError("chunk_target_exceeds_memory_budget")
        return effective

    def require_valid_chunk_budget(self) -> None:
        _ = self.effective_chunk_bytes

    def to_payload(self) -> dict[str, int | float | None]:
        """Return exact runtime resource policy metadata."""

        return {
            "available_memory_bytes": self.available_memory_bytes,
            "memory_budget_fraction": self.memory_budget_fraction,
            "memory_budget_bytes": self.memory_budget_bytes,
            "memory_budget_bytes_override": self.memory_budget_bytes_override,
            "chunk_target_fraction": self.chunk_target_fraction,
            "chunk_target_cap_bytes": self.chunk_target_cap_bytes,
            "chunk_target_bytes": self._resolved_chunk_target_bytes(),
            "effective_chunk_bytes": self.effective_chunk_bytes,
            "memory_safety_factor": self.memory_safety_factor,
            "min_free_disk_bytes": self.min_free_disk_bytes,
        }

    def to_review_safe_summary_payload(self) -> dict[str, int | float | str | bool]:
        """Return coarse resource policy metadata for review-safe summaries."""

        return {
            "memory_budget_fraction": self.memory_budget_fraction,
            "memory_safety_factor": self.memory_safety_factor,
            "available_memory_bucket": _byte_bucket(self.available_memory_bytes),
            "memory_budget_bucket": _byte_bucket(self.memory_budget_bytes),
            "chunk_target_bucket": _byte_bucket(self._resolved_chunk_target_bytes()),
            "effective_chunk_bucket": _byte_bucket(self.effective_chunk_bytes),
            "has_memory_budget_override": (
                self.memory_budget_bytes_override is not None
            ),
            "min_free_disk_bucket": _byte_bucket(self.min_free_disk_bytes),
        }

    def has_minimum_free_disk(self, path: Path) -> bool:
        candidate = Path(path)
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        free = shutil.disk_usage(candidate.resolve()).free
        return free >= self.min_free_disk_bytes

    def require_minimum_free_disk(self, path: Path) -> None:
        if not self.has_minimum_free_disk(path):
            raise RuntimeError("insufficient_free_disk")

    def _resolved_chunk_target_bytes(self) -> int:
        if self.chunk_target_bytes is not None:
            return self.chunk_target_bytes
        return min(
            int(self.memory_budget_bytes * self.chunk_target_fraction),
            self.chunk_target_cap_bytes,
        )


_ACTIVE_RUNTIME_RESOURCE_POLICY: ContextVar[RuntimeResourcePolicy | None] = (
    ContextVar("qrp_active_runtime_resource_policy", default=None)
)


def active_runtime_resource_policy() -> RuntimeResourcePolicy:
    """Return the active runtime resource snapshot or create one."""

    policy = _ACTIVE_RUNTIME_RESOURCE_POLICY.get()
    if policy is not None:
        return policy
    return RuntimeResourcePolicy.from_environment()


@contextmanager
def runtime_resource_policy_scope(
    policy: RuntimeResourcePolicy,
) -> Iterator[RuntimeResourcePolicy]:
    """Temporarily make one resource policy snapshot active."""

    policy_context_handle = _ACTIVE_RUNTIME_RESOURCE_POLICY.set(policy)
    try:
        yield policy
    finally:
        _ACTIVE_RUNTIME_RESOURCE_POLICY.reset(policy_context_handle)


def require_free_disk_headroom(
    policy: RuntimeResourcePolicy,
    path: Path,
    *,
    reserve_bytes: int,
) -> None:
    """Require the policy floor plus owner-estimated staged-write headroom."""

    if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int):
        raise ValueError("invalid_runtime_resource_disk_reserve")
    if reserve_bytes < 0:
        raise ValueError("invalid_runtime_resource_disk_reserve")
    policy.require_minimum_free_disk(path)
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    free = shutil.disk_usage(candidate.resolve()).free
    if free < policy.min_free_disk_bytes + reserve_bytes:
        raise RuntimeError("insufficient_free_disk")


@dataclass(frozen=True)
class ResourceChunkPlan:
    """Resource-derived chunk target for one semantic axis."""

    axis: ChunkAxis
    session_count: int
    target_core_session_count: int
    chunk_count: int
    estimated_bytes_per_session: int
    estimated_chunk_bytes: int


class ResourceChunkPlanner:
    """Plan resource chunk sizes from runtime policy and data shape."""

    def __init__(self, policy: RuntimeResourcePolicy | None = None) -> None:
        self.policy = policy or active_runtime_resource_policy()

    def plan_sessions(
        self,
        *,
        axis: ChunkAxis,
        session_count: int,
        estimated_bytes_per_session: int,
    ) -> ResourceChunkPlan:
        self.policy.require_valid_chunk_budget()
        if session_count < 0:
            raise ValueError("invalid_session_count")
        if estimated_bytes_per_session <= 0:
            raise ValueError("invalid_estimated_bytes_per_session")
        if (
            session_count > 0
            and estimated_bytes_per_session > self.policy.effective_chunk_bytes
        ):
            raise ValueError("session_estimate_exceeds_chunk_budget")
        if session_count == 0:
            target_core_session_count = 1
            chunk_count = 0
        else:
            target_core_session_count = max(
                1,
                min(
                    session_count,
                    self.policy.effective_chunk_bytes
                    // estimated_bytes_per_session,
                ),
            )
            chunk_count = ceil(session_count / target_core_session_count)
        return ResourceChunkPlan(
            axis=axis,
            session_count=session_count,
            target_core_session_count=target_core_session_count,
            chunk_count=chunk_count,
            estimated_bytes_per_session=estimated_bytes_per_session,
            estimated_chunk_bytes=(
                target_core_session_count * estimated_bytes_per_session
            ),
        )


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().available)
    except Exception:
        return DEFAULT_AVAILABLE_MEMORY_BYTES


def _bytes_env(env: Mapping[str, str], name: str) -> int | None:
    value = env.get(name)
    if value in {None, ""}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"invalid_runtime_resource_env: {name}")
    return max(1, int(parsed * MIB))


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if value in {None, ""}:
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"invalid_runtime_resource_env: {name}")
    return parsed


def _fraction_env(env: Mapping[str, str], name: str, default: float) -> float:
    parsed = _float_env(env, name, default)
    if not isfinite(parsed) or parsed > 1:
        raise ValueError(f"invalid_runtime_resource_env: {name}")
    return parsed


def _require_fraction(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0 or value > 1:
        raise ValueError(f"invalid_runtime_resource_fraction: {name}")


def _byte_bucket(value: int) -> str:
    size = int(value)
    if size <= 0:
        return "0"
    units = (("GiB", GIB), ("MiB", MIB))
    for suffix, unit in units:
        if size >= unit:
            lower = 1
            scaled = size / unit
            while lower * 2 <= scaled:
                lower *= 2
            return f"{lower}-{lower * 2}{suffix}"
    return "<1MiB"


__all__ = [
    "ChunkAxis",
    "ResourceChunkPlan",
    "ResourceChunkPlanner",
    "RuntimeResourcePolicy",
]
