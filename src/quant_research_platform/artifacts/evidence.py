"""Writers for already-built review-safe evidence payloads and text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
    evidence_text_unsafe_reason,
)
from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text
from quant_research_platform.storage.manifests import (
    read_qrp_json_file,
)


def write_json_evidence(path: Path, payload: dict[str, Any]) -> Path:
    schema_id = payload.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id or "/" in schema_id:
        raise ValueError("evidence_json_identity_invalid")
    unsafe = evidence_json_value_unsafe_reason(payload)
    if unsafe is not None:
        raise ValueError(f"unsafe_evidence: {unsafe}")
    return atomic_write_json(path, payload)


def read_json_evidence(path: Path, *, expected_schema_id: str, error: str):
    """Read declared evidence once and check its schema."""
    _raw, payload = read_qrp_json_file(path)
    if payload.get("schema_id") != expected_schema_id:
        raise ValueError(error)
    return payload


def write_markdown_evidence(path: Path, text: str) -> Path:
    unsafe = evidence_text_unsafe_reason(text)
    if unsafe is not None:
        raise ValueError(f"unsafe_evidence: {unsafe}")
    return atomic_write_text(path, text, encoding="utf-8")
