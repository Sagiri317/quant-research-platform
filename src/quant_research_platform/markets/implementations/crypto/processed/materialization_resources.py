"""Owner-local total working-set allocation for crypto materialization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    require_free_disk_headroom,
)

_ALLOCATION_WEIGHTS = {
    "physical_read": 3,
    "core_window": 1,
    "carry_or_label_buffer": 1,
    "dependency_key_index": 4,
    "reducer_state": 2,
    "trade_output": 1,
    "depth_output": 2,
    "materialized_output": 2,
    "sqlite_cache": 1,
    "allocator_headroom": 1,
}
_MIN_ALLOCATION_UNIT_BYTES = 1024


@dataclass(frozen=True)
class MaterializationBudget:
    """Allocate one effective chunk budget across simultaneously live state."""

    allocations: dict[str, int]

    @classmethod
    def from_policy(
        cls,
        policy: RuntimeResourcePolicy,
    ) -> MaterializationBudget:
        policy.require_valid_chunk_budget()
        effective = policy.effective_chunk_bytes
        total_weight = sum(_ALLOCATION_WEIGHTS.values())
        if effective < total_weight * _MIN_ALLOCATION_UNIT_BYTES:
            raise ValueError("crypto_materialization_budget_too_small")
        unit = effective // total_weight
        allocations = {
            name: unit * weight
            for name, weight in _ALLOCATION_WEIGHTS.items()
        }
        allocations["allocator_headroom"] += effective - sum(
            allocations.values()
        )
        return cls(allocations=allocations)

    @property
    def physical_read_bytes(self) -> int:
        return self.allocations["physical_read"]

    @property
    def reducer_state_bytes(self) -> int:
        return self.allocations["reducer_state"]

    @property
    def trade_output_bytes(self) -> int:
        return self.allocations["trade_output"]

    @property
    def depth_output_bytes(self) -> int:
        return self.allocations["depth_output"]

    @property
    def materialized_output_bytes(self) -> int:
        return self.allocations["materialized_output"]

    @property
    def carry_bytes(self) -> int:
        return self.allocations["carry_or_label_buffer"]

    @property
    def panel_key_index_bytes(self) -> int:
        return sum(
            self.allocations[name]
            for name in (
                "carry_or_label_buffer",
                "dependency_key_index",
                "sqlite_cache",
            )
        )

    @property
    def dependent_key_index_bytes(self) -> int:
        return sum(
            self.allocations[name]
            for name in (
                "core_window",
                "dependency_key_index",
                "reducer_state",
                "trade_output",
                "depth_output",
                "sqlite_cache",
            )
        )

    @property
    def validation_key_index_bytes(self) -> int:
        return sum(
            self.allocations[name]
            for name in (
                "core_window",
                "carry_or_label_buffer",
                "dependency_key_index",
                "reducer_state",
                "trade_output",
                "depth_output",
                "materialized_output",
            )
        )

    @property
    def sqlite_cache_bytes(self) -> int:
        return self.allocations["sqlite_cache"]


def require_materialization_disk_headroom(
    policy: RuntimeResourcePolicy,
    scratch_parent: Path,
) -> None:
    require_free_disk_headroom(
        policy,
        scratch_parent,
        reserve_bytes=2 * policy.effective_chunk_bytes,
    )


def iter_disk_guarded_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    policy: RuntimeResourcePolicy,
    scratch_parent: Path,
) -> Iterator[pd.DataFrame]:
    for chunk in chunks:
        chunk_bytes = materialization_frame_bytes(chunk)
        require_free_disk_headroom(
            policy,
            scratch_parent,
            reserve_bytes=(
                2 * policy.effective_chunk_bytes
                + 4 * chunk_bytes
            ),
        )
        yield chunk
    require_materialization_disk_headroom(policy, scratch_parent)


def require_frame_within_budget(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
    error: str,
) -> None:
    if materialization_frame_bytes(frame) > target_bytes:
        raise ValueError(error)


def materialization_frame_bytes(frame: pd.DataFrame) -> int:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("invalid_crypto_materialization_chunk")
    return int(frame.memory_usage(index=True, deep=True).sum())
