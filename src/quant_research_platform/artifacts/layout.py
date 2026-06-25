"""Artifact output layout helpers."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.contracts.ids import MarketId


def local_run_output_root(
    project_root: Path,
    market_id: MarketId | str,
    run_type: str,
) -> Path:
    return Path(project_root) / "run_outputs" / "local" / str(market_id) / run_type


def review_evidence_root(project_root: Path) -> Path:
    return Path(project_root) / "handoff_evidence"


__all__ = [
    "local_run_output_root",
    "review_evidence_root",
]
