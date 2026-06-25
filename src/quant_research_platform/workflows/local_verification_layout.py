"""Local verification workflow layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


__all__ = ["LocalVerificationLayout"]
