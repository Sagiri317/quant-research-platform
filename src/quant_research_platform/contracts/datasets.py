"""Research dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True, kw_only=True)
class ResearchDatasetSpec(DataSpec):
    market_id: MarketId
    input_data_ids: tuple[str, ...]
    base_input_data_id: str
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    filter_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()
    key_coverage: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        invalid_inputs = sorted(set(self.key_coverage).difference(self.input_data_ids))
        if invalid_inputs:
            raise ValueError(
                "unknown_research_key_coverage_input: " + ", ".join(invalid_inputs)
            )
        invalid_policies = sorted(
            f"{data_id}={policy}"
            for data_id, policy in self.key_coverage.items()
            if policy not in {"strict", "sparse"}
        )
        if invalid_policies:
            raise ValueError(
                "invalid_research_key_coverage: " + ", ".join(invalid_policies)
            )
        if self.key_coverage.get(self.base_input_data_id, "strict") != "strict":
            raise ValueError("base_input_key_coverage_must_be_strict")
        normalized = {
            data_id: self.key_coverage.get(data_id, "strict")
            for data_id in self.input_data_ids
        }
        object.__setattr__(self, "key_coverage", normalized)


__all__ = ["ResearchDatasetSpec"]
