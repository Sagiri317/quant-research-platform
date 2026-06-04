"""Structured report summary objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from quant_research_platform.runtime.privacy import sanitize_public_object


@dataclass(frozen=True)
class ManifestSummary:
    artifact_id: str
    market_id: str
    status: str
    contract_version: str = "0.1"
    aggregate_metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return sanitize_public_object(asdict(self))


@dataclass(frozen=True)
class ReportSummary:
    run_id: str
    market_id: str
    run_type: str
    status: str
    contract_version: str = "0.1"
    aggregate_metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return sanitize_public_object(asdict(self))


@dataclass(frozen=True)
class ParityEvidenceSummary:
    evidence_id: str
    market_id: str
    data_product_id: str
    privacy_level: str
    status: str
    contract_version: str = "0.1"
    aggregate_metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return sanitize_public_object(asdict(self))


@dataclass(frozen=True)
class ReportRunIndexRecord:
    run_id: str
    market_id: str
    run_type: str
    status: str
    report_definition_id: str
    contract_version: str = "0.1"

    def to_record(self) -> dict[str, Any]:
        return sanitize_public_object(asdict(self))


def summarize_status(status: str) -> dict[str, str]:
    return {"status": sanitize_public_object(status)}


__all__ = [
    "ManifestSummary",
    "ParityEvidenceSummary",
    "ReportRunIndexRecord",
    "ReportSummary",
    "summarize_status",
]
