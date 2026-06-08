"""Runtime resource policy and chunk planning helpers."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path

DEFAULT_AVAILABLE_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MEMORY_BUDGET_FRACTION = 0.70
DEFAULT_MEMORY_SAFETY_FACTOR = 0.60
DEFAULT_MIN_FREE_DISK_BYTES = 0
GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024


class ChunkAxis(StrEnum):
    """Semantic axis used to split a dataset or build operation."""

    ASSET = "asset"
    YEAR = "year"
    PARTITION = "partition"
    ROW_GROUP = "row_group"


@dataclass(frozen=True)
class RuntimeResourcePolicy:
    """Memory and disk limits for one local QRP runtime task."""

    available_memory_bytes: int
    memory_budget_fraction: float = DEFAULT_MEMORY_BUDGET_FRACTION
    chunk_target_bytes: int | None = None
    memory_safety_factor: float = DEFAULT_MEMORY_SAFETY_FACTOR
    min_free_disk_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES
    memory_budget_bytes_override: int | None = None

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
        fraction = _float_env(
            env,
            "QRP_MEMORY_BUDGET_FRACTION",
            DEFAULT_MEMORY_BUDGET_FRACTION,
        )
        budget = _bytes_env(env, "QRP_MEMORY_BUDGET_MB")
        target = _bytes_env(env, "QRP_CHUNK_TARGET_MB")
        if target is None:
            target = min(int(available * fraction * 0.30), 2 * GIB)
        safety = _float_env(
            env,
            "QRP_MEMORY_SAFETY_FACTOR",
            DEFAULT_MEMORY_SAFETY_FACTOR,
        )
        min_disk = _bytes_env(env, "QRP_MIN_FREE_DISK_MB")
        policy = cls(
            available_memory_bytes=available,
            memory_budget_fraction=fraction,
            chunk_target_bytes=target,
            memory_safety_factor=safety,
            min_free_disk_bytes=(
                DEFAULT_MIN_FREE_DISK_BYTES if min_disk is None else min_disk
            ),
        )
        if budget is None:
            return policy
        return cls(
            available_memory_bytes=available,
            memory_budget_fraction=fraction,
            chunk_target_bytes=target,
            memory_safety_factor=safety,
            min_free_disk_bytes=policy.min_free_disk_bytes,
            memory_budget_bytes_override=budget,
        )

    @property
    def memory_budget_bytes(self) -> int:
        if self.memory_budget_bytes_override is not None:
            return self.memory_budget_bytes_override
        return max(1, int(self.available_memory_bytes * self.memory_budget_fraction))

    @property
    def effective_chunk_bytes(self) -> int:
        target = self.chunk_target_bytes
        if target is None:
            target = min(int(self.memory_budget_bytes * 0.30), 2 * GIB)
        return max(1, int(target * self.memory_safety_factor))

    def has_minimum_free_disk(self, path: Path) -> bool:
        candidate = Path(path)
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        free = shutil.disk_usage(candidate.resolve()).free
        return free >= self.min_free_disk_bytes

    def require_minimum_free_disk(self, path: Path) -> None:
        if not self.has_minimum_free_disk(path):
            raise RuntimeError("insufficient_free_disk")


@dataclass(frozen=True)
class ChunkPlan:
    """Concrete chunk plan for one semantic axis."""

    axis: ChunkAxis
    item_count: int
    chunk_size: int
    chunk_count: int
    estimated_bytes_per_item: int
    estimated_chunk_bytes: int
    lookback_window: int = 0
    lookahead_window: int = 0
    output_width: int = 0
    disjoint: bool = True
    boundary_carry_required: bool = False


class ChunkPlanner:
    """Plan chunk sizes from runtime policy and data shape."""

    def __init__(self, policy: RuntimeResourcePolicy | None = None) -> None:
        self.policy = policy or RuntimeResourcePolicy.from_environment()

    def plan(
        self,
        *,
        axis: ChunkAxis,
        item_count: int,
        estimated_bytes_per_item: int,
        lookback_window: int = 0,
        lookahead_window: int = 0,
        output_width: int = 0,
        disjoint: bool = True,
    ) -> ChunkPlan:
        if item_count < 0:
            raise ValueError("invalid_item_count")
        if estimated_bytes_per_item <= 0:
            raise ValueError("invalid_estimated_bytes_per_item")
        if item_count == 0:
            chunk_size = 1
            chunk_count = 0
        else:
            chunk_size = max(
                1,
                min(
                    item_count,
                    self.policy.effective_chunk_bytes // estimated_bytes_per_item,
                ),
            )
            chunk_count = ceil(item_count / chunk_size)
        return ChunkPlan(
            axis=axis,
            item_count=item_count,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            estimated_bytes_per_item=estimated_bytes_per_item,
            estimated_chunk_bytes=chunk_size * estimated_bytes_per_item,
            lookback_window=lookback_window,
            lookahead_window=lookahead_window,
            output_width=output_width,
            disjoint=disjoint,
            boundary_carry_required=lookback_window > 0 or lookahead_window > 0,
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
    return max(1, int(float(value) * MIB))


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if value in {None, ""}:
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"invalid_runtime_resource_env: {name}")
    return parsed


__all__ = [
    "ChunkAxis",
    "ChunkPlan",
    "ChunkPlanner",
    "RuntimeResourcePolicy",
]
