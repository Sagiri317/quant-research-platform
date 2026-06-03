"""Report manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportManifest:
    run_id: str
    report_version: str = "0.1"


__all__ = ["ReportManifest"]
