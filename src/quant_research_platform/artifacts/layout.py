"""Artifact layout helpers."""

from __future__ import annotations

from pathlib import Path


def output_bundle_root(run_root: Path, output_id: str) -> Path:
    return Path(run_root) / "outputs" / output_id


def handoff_evidence_root(project_root: Path) -> Path:
    return Path(project_root) / "handoff_evidence"


__all__ = [
    "handoff_evidence_root",
    "output_bundle_root",
]
