"""Local verification workflow layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalVerificationLayout:
    run_root: Path

    @classmethod
    def from_run_root(cls, run_root: Path) -> LocalVerificationLayout:
        return cls(run_root=Path(run_root))

    @property
    def run_json_path(self) -> Path:
        return self.run_root / "run.json"

    @property
    def data_root(self) -> Path:
        return self.run_root / "data"

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
    def outputs_root(self) -> Path:
        return self.run_root / "outputs"

    @property
    def logs_root(self) -> Path:
        return self.run_root / "logs"

    @property
    def summary_md_path(self) -> Path:
        return self.run_root / "summary.md"


__all__ = ["LocalVerificationLayout"]
