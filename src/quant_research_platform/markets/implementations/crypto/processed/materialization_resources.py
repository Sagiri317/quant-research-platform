"""Owner-local total working-set allocation for crypto materialization."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.runtime.resources import RuntimeResourcePolicy

_ALLOCATION_WEIGHTS = {
    "arrow_batch": 1,
    "pandas_frame": 1,
    "normalization_copy": 1,
    "core_window": 1,
    "carry_or_label_buffer": 1,
    "reducer_state": 1,
    "output_rows": 1,
    "output_frame": 1,
    "sqlite_cache": 1,
    "allocator_headroom": 3,
}


@dataclass(frozen=True)
class MaterializationBudget:
    """Allocate one effective chunk budget across simultaneously live state."""

    effective_bytes: int
    allocations: dict[str, int]

    @classmethod
    def from_policy(
        cls,
        policy: RuntimeResourcePolicy,
    ) -> MaterializationBudget:
        policy.require_valid_chunk_budget()
        effective = policy.effective_chunk_bytes
        total_weight = sum(_ALLOCATION_WEIGHTS.values())
        if effective < total_weight:
            raise ValueError("crypto_materialization_budget_too_small")
        unit = effective // total_weight
        allocations = {
            name: unit * weight
            for name, weight in _ALLOCATION_WEIGHTS.items()
        }
        allocations["allocator_headroom"] += effective - sum(
            allocations.values()
        )
        return cls(effective_bytes=effective, allocations=allocations)

    @property
    def physical_batch_bytes(self) -> int:
        return self.allocations["arrow_batch"]

    @property
    def reducer_state_bytes(self) -> int:
        return self.allocations["reducer_state"]

    @property
    def output_bytes(self) -> int:
        return (
            self.allocations["output_rows"]
            + self.allocations["output_frame"]
        )

    @property
    def carry_bytes(self) -> int:
        return self.allocations["carry_or_label_buffer"]

    def require_observed(self, **component_bytes: int) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in component_bytes.values()
        ):
            raise ValueError("invalid_crypto_materialization_component_bytes")
        if sum(component_bytes.values()) > self.effective_bytes:
            raise ValueError("crypto_materialization_working_set_exceeds_budget")
