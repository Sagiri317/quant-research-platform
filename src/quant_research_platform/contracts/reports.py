"""Report contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id


@dataclass(frozen=True)
class ReportDefinition:
    report_definition_id: str
    display_name: str
    run_type: str
    contract_version: str = "0.1"
    private_derived: bool = True

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_report_contract_version")


@dataclass(frozen=True)
class ReportSpec:
    report_id: str
    report_definition_id: str
    run_id: str
    market_id: MarketId
    run_type: str
    contract_version: str = "0.1"
    private_boundary_required: bool = True

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_report_contract_version")
        validate_run_id(self.run_id)


__all__ = ["ReportDefinition", "ReportSpec"]
