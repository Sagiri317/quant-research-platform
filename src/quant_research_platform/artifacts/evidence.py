"""Writers for already-built review-safe evidence payloads and text."""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    bounded_rendered_qrp_json_object_size,
    json_record_working_set_bytes,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
    evidence_text_unsafe_reason,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text
from quant_research_platform.storage.manifests import (
    _qrp_json_file_identity,
    read_bounded_qrp_json,
)


def write_json_evidence(path: Path, payload: dict[str, Any]) -> Path:
    schema_id = payload.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id or "/" in schema_id:
        raise ValueError("evidence_json_identity_invalid")
    unsafe = evidence_json_value_unsafe_reason(payload)
    if unsafe is not None:
        raise ValueError(f"unsafe_evidence: {unsafe}")
    encoded = (
        bounded_rendered_qrp_json_object_size(
            payload, lazy_arrays={}, max_bytes=sys.maxsize
        )
        + 1
    )
    with runtime_resource_policy_scope() as policy:
        with policy.lease(
            "evidence.json.write", json_record_working_set_bytes(encoded)
        ):
            return atomic_write_json(path, payload)


@contextmanager
def read_json_evidence(path: Path, *, expected_schema_id: str, error: str, lease=None):
    """Keep decoded evidence admitted until its consumer exits or releases its lease."""
    identity = _qrp_json_file_identity(path)
    with runtime_resource_policy_scope() as policy:
        with (
            policy.lease("evidence.json.read", 0)
            if lease is None
            else nullcontext(lease)
        ) as owned:
            owned.grow(json_record_working_set_bytes(identity.size))
            payload = read_bounded_qrp_json(
                path, expected_identity=identity, max_bytes=identity.size
            )[1]
            if payload.get("schema_id") != expected_schema_id:
                raise ValueError(error)
            yield payload


def write_markdown_evidence(path: Path, text: str) -> Path:
    unsafe = evidence_text_unsafe_reason(text)
    if unsafe is not None:
        raise ValueError(f"unsafe_evidence: {unsafe}")
    return atomic_write_text(path, text, encoding="utf-8")
