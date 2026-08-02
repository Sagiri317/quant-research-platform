"""Artifact layout helpers."""

from __future__ import annotations

from pathlib import Path


def output_bundle_root(run_root: Path, output_id: str) -> Path:
    return Path(run_root) / "outputs" / output_id


__all__ = [
    "output_bundle_root",
]
