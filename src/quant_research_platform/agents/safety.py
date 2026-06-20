"""Safety checks for proposal-only agent boundaries."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from quant_research_platform.runtime.privacy import public_text_unsafe_reason


class ProposalSafetyError(ValueError):
    """Raised when a proposal contains public-unsafe content."""


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
    reason = public_text_unsafe_reason(text)
    if reason is not None:
        raise ProposalSafetyError(f"proposal_contains_public_unsafe_text:{reason}")


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
    if isinstance(value, tuple | list | set | frozenset):
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
