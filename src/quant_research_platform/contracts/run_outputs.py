"""Run output contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id


class OutputScope(StrEnum):
    LOCAL_ONLY = "LOCAL_ONLY"
    REVIEW_SAFE = "REVIEW_SAFE"


@dataclass(frozen=True)
class RunOutputDefinition:
    output_definition_id: str
    display_name: str
    run_type: str
    scope: OutputScope
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_run_output_contract_version")
        OutputScope(self.scope)


@dataclass(frozen=True)
class RunOutputSpec:
    output_id: str
    output_definition_id: str
    run_id: str
    market_id: MarketId
    run_type: str
    scope: OutputScope
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_run_output_contract_version")
        OutputScope(self.scope)
        validate_run_id(self.run_id)


__all__ = ["OutputScope", "RunOutputDefinition", "RunOutputSpec"]
