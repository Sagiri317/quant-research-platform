"""Report manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.run_id import validate_run_id


@dataclass(frozen=True)
class ReportManifest:
    run_id: str
    report_version: str = "0.1"

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.report_version != "0.1":
            raise ValueError("unsupported_report_manifest_version")


__all__ = ["ReportManifest"]
