"""Output contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.policy.identifiers import validate_output_identifier


class OutputScope(StrEnum):
    LOCAL_ONLY = "LOCAL_ONLY"
    REVIEW_SAFE = "REVIEW_SAFE"


@dataclass(frozen=True)
class OutputDefinition:
    output_definition_id: str
    display_name: str
    run_type: str
    scope: OutputScope

    def __post_init__(self) -> None:
        OutputScope(self.scope)


@dataclass(frozen=True)
class OutputSpec:
    output_id: str
    output_definition_id: str
    run_id: str
    market_id: MarketId
    run_type: str
    scope: OutputScope

    def __post_init__(self) -> None:
        OutputScope(self.scope)
        validate_output_identifier(self.output_id)
        if self.output_id == self.output_definition_id:
            raise ValueError("output_id_must_identify_instance")
        validate_run_id(self.run_id)
