"""Identifier safety checks for run and output ids."""

from __future__ import annotations

import re

from quant_research_platform.policy.share_safety import public_text_unsafe_reason

_RUN_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SOURCE_PROVIDER_TOKEN_RE = re.compile(
    r"(?i)(?:^|[._-])(?:source|provider|vendor)(?:$|[._-])"
)


def unsafe_identifier_reason(value: object) -> str | None:
    """Return why a run/output identifier is unsafe for sharing, if applicable."""

    text = str(value)
    if _SOURCE_PROVIDER_TOKEN_RE.search(text):
        return "private_marker"
    tokenized = text.replace("_", " ").replace("-", " ")
    reason = public_text_unsafe_reason(tokenized)
    if reason in {"stock_code_sample", "private_marker", "local_path", "private_path"}:
        return reason
    return None


def validate_run_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid_run_id")
    if not _RUN_IDENTIFIER_RE.fullmatch(value):
        raise ValueError("invalid_run_id")
    if unsafe_identifier_reason(value) is not None:
        raise ValueError("invalid_run_id")
    return value


__all__ = ["unsafe_identifier_reason", "validate_run_identifier"]
