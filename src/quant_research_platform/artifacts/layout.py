"""Artifact output layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class LocalVerificationLayout:
    output_root: Path

    @classmethod
    def from_output_root(cls, output_root: Path) -> LocalVerificationLayout:
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


def local_run_output_root(
    project_root: Path,
    market_id: MarketId | str,
    run_type: str,
) -> Path:
    return Path(project_root) / "run_outputs" / "local" / str(market_id) / run_type


def review_evidence_root(project_root: Path) -> Path:
    return Path(project_root) / "handoff_evidence"


__all__ = [
    "LocalVerificationLayout",
    "local_run_output_root",
    "review_evidence_root",
]
