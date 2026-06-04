"""Safety checks for proposal-only agent boundaries."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any


class ProposalSafetyError(ValueError):
    """Raised when a proposal contains public-unsafe content."""


_UNSAFE_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_path",
        re.compile(
            r"(?i)(?:[a-z]:\\|/(?:mnt|home|Users|tmp)/|"
            r"data[/\\](?:raw|interim|processed)|reports[/\\]private_real)"
        ),
    ),
    ("row_sample", re.compile(r"(?i)\brow\s+sample\b|market_id\s*=.*asset_id\s*=")),
    ("stock_code", re.compile(r"(?i)\bstock\s+code\b|\b(?:00|30|60|68)\d{4}\b")),
    (
        "source_marker",
        re.compile(r"(?i)\bsource\s+marker\b|\bvendor\s+marker\b"),
    ),
)


def require_human_approval(proposal_or_approved: Any) -> None:
    approved = (
        proposal_or_approved
        if isinstance(proposal_or_approved, bool)
        else bool(getattr(proposal_or_approved, "approved", False))
    )
    if not approved:
        raise PermissionError("proposal_requires_human_approval")


def assert_proposal_starts_unapproved(proposal: Any) -> None:
    if bool(getattr(proposal, "approved", False)):
        raise ProposalSafetyError("proposal_must_start_unapproved")


def assert_public_safe_text(text: str) -> None:
    for marker, pattern in _UNSAFE_TEXT_PATTERNS:
        if pattern.search(text):
            raise ProposalSafetyError(f"proposal_contains_public_unsafe_text:{marker}")


def assert_proposal_is_public_safe(proposal: Any) -> None:
    assert_proposal_starts_unapproved(proposal)
    for text in _iter_text_values(proposal):
        assert_public_safe_text(text)


def _iter_text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        texts: list[str] = []
        for key, item in value.items():
            texts.extend(_iter_text_values(key))
            texts.extend(_iter_text_values(item))
        return tuple(texts)
    if isinstance(value, (tuple, list, set, frozenset)):
        texts = []
        for item in value:
            texts.extend(_iter_text_values(item))
        return tuple(texts)
    if is_dataclass(value) and not isinstance(value, type):
        texts = []
        for item in fields(value):
            texts.extend(_iter_text_values(getattr(value, item.name)))
        return tuple(texts)
    return ()


__all__ = [
    "ProposalSafetyError",
    "assert_proposal_is_public_safe",
    "assert_proposal_starts_unapproved",
    "assert_public_safe_text",
    "require_human_approval",
]
