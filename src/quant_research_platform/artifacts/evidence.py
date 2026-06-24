"""Writers for already-built handoff-safe evidence payloads and text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text


def write_json_evidence(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def write_markdown_evidence(path: Path, text: str) -> Path:
    return atomic_write_text(path, text, encoding="utf-8")


def write_validation_summary(path: Path, payload: dict[str, Any]) -> Path:
    return write_json_evidence(path, payload)


def write_verification_evidence_json(path: Path, payload: dict[str, Any]) -> Path:
    return write_json_evidence(path, payload)


def write_verification_evidence_markdown(path: Path, text: str) -> Path:
    return write_markdown_evidence(path, text)


def write_run_summary_json(path: Path, payload: dict[str, Any]) -> Path:
    return write_json_evidence(path, payload)


def write_run_summary_markdown(path: Path, text: str) -> Path:
    return write_markdown_evidence(path, text)


__all__ = [
    "write_json_evidence",
    "write_markdown_evidence",
    "write_run_summary_json",
    "write_run_summary_markdown",
    "write_validation_summary",
    "write_verification_evidence_json",
    "write_verification_evidence_markdown",
]
