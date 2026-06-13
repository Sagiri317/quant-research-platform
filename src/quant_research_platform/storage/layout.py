"""Portable disk layout for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId, SourceId


@dataclass(frozen=True)
class StorageLayout:
    project_root: Path = Path(".")

    def raw_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/raw" / str(source_id)

    def interim_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/interim" / str(source_id)

    def interim_data_root(self, source_id: SourceId, data_id: str) -> Path:
        return self.interim_source_root(source_id) / data_id

    def processed_market_root(self, market_id: MarketId) -> Path:
        return self.project_root / "data/processed" / str(market_id)

    def processed_data_root(
        self,
        market_id: MarketId,
        kind: ProcessedDataKind | str,
        data_id: str,
    ) -> Path:
        return self.processed_market_root(market_id) / str(kind) / data_id

    def research_dataset_root(self, market_id: MarketId, data_id: str) -> Path:
        return self.project_root / "data/research" / str(market_id) / data_id

    def private_report_root(self, market_id: MarketId, run_type: str) -> Path:
        return self.project_root / "reports/private_real" / str(market_id) / run_type


@dataclass(frozen=True)
class VerificationEvidenceLayout:
    output_root: Path

    @classmethod
    def from_output_root(cls, output_root: Path) -> VerificationEvidenceLayout:
        return cls(output_root=Path(output_root))

    @property
    def config_root(self) -> Path:
        return self.output_root / "config"

    @property
    def data_root(self) -> Path:
        return self.output_root / "data"

    @property
    def interim_root(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def research_root(self) -> Path:
        return self.data_root / "research"

    @property
    def evidence_root(self) -> Path:
        return self.data_root / "evidence"

    @property
    def manifests_root(self) -> Path:
        return self.output_root / "manifests"

    @property
    def validations_root(self) -> Path:
        return self.output_root / "validations"

    @property
    def timings_root(self) -> Path:
        return self.output_root / "timings"

    @property
    def logs_root(self) -> Path:
        return self.output_root / "logs"

    @property
    def summary_json_path(self) -> Path:
        return self.output_root / "summary.json"

    @property
    def summary_md_path(self) -> Path:
        return self.output_root / "summary.md"


def layout_from_processed_market_root(processed_root: Path) -> StorageLayout:
    """Resolve ``data/processed/<market>`` back to a project storage layout."""

    root = Path(processed_root)
    parts = root.parts
    if len(parts) < 3 or parts[-3:-1] != ("data", "processed"):
        raise ValueError("processed_root_must_be_data_processed_market")
    project_root = Path(*parts[:-3]) if len(parts) > 3 else Path(".")
    return StorageLayout(project_root=project_root)


__all__ = [
    "VerificationEvidenceLayout",
    "StorageLayout",
    "layout_from_processed_market_root",
]
