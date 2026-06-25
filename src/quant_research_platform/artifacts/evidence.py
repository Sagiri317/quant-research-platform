"""Writers for already-built handoff-safe evidence payloads and text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.policy.share_safety import qrp_json_identity_reason
from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text


def write_json_evidence(path: Path, payload: dict[str, Any]) -> Path:
    reason = qrp_json_identity_reason(payload)
    if reason is not None:
        raise ValueError(f"evidence_json_identity_invalid: {reason}")
    return atomic_write_json(path, payload)


def write_markdown_evidence(path: Path, text: str) -> Path:
    return atomic_write_text(path, text, encoding="utf-8")


def write_run_summary_markdown(path: Path, text: str) -> Path:
    return write_markdown_evidence(path, text)


__all__ = [
    "write_json_evidence",
    "write_markdown_evidence",
    "write_run_summary_markdown",
]
