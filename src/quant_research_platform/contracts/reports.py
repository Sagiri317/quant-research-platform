"""Report contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class ReportDefinition:
    report_definition_id: str
    display_name: str
    run_type: str
    contract_version: str = "0.1"
    private_derived: bool = True


@dataclass(frozen=True)
class ReportSpec:
    report_id: str
    report_definition_id: str
    run_id: str
    market_id: MarketId
    run_type: str
    contract_version: str = "0.1"
    private_boundary_required: bool = True
    allowed_output_root: str = "reports/private_real"


__all__ = ["ReportDefinition", "ReportSpec"]
