"""Safety checks for agent-assisted research payloads."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any

from quant_research_platform.policy.share_safety import unsafe_for_share_reason

_PROPOSAL_DATA_FILE_PATH_RE = re.compile(
    r"(?i)(?:^|[\s`'\"=:\[{(])[^`'\"\s\]}),;]+"
    r"\.(?:parquet|csv|pkl|pickle|feather|arrow)(?:$|[\s`'\"\]}),;])"
)
_SOURCE_ASSET_ID_TERM = "source_asset" + "_id"
_SOURCE_ID_TERM = "source" + "_id"
_PROPOSAL_FORBIDDEN_TEXT_RE = re.compile(
    "(?i)(?:"
    + "|".join(
        (
            _SOURCE_ASSET_ID_TERM,
            _SOURCE_ID_TERM,
            r"api[_-]?key",
            r"access[_-]?token",
            r"refresh[_-]?token",
        )
    )
    + ")"
)


class ProposalSafetyError(ValueError):
    """Raised when a proposal contains share-unsafe content."""


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


def assert_share_safe_text(text: str) -> None:
    if _PROPOSAL_FORBIDDEN_TEXT_RE.search(text):
        raise ProposalSafetyError("proposal_contains_share_unsafe_text:private_marker")
    if _PROPOSAL_DATA_FILE_PATH_RE.search(text):
        raise ProposalSafetyError("proposal_contains_share_unsafe_text:data_file_path")
    reason = unsafe_for_share_reason(text)
    if reason is not None:
        raise ProposalSafetyError(f"proposal_contains_share_unsafe_text:{reason}")


def assert_proposal_is_share_safe(proposal: Any) -> None:
    assert_proposal_starts_unapproved(proposal)
    assert_proposal_text_is_share_safe(proposal)


def assert_proposal_text_is_share_safe(proposal: Any) -> None:
    for text in _iter_text_values(proposal):
        assert_share_safe_text(text)


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
    "assert_proposal_is_share_safe",
    "assert_proposal_starts_unapproved",
    "assert_proposal_text_is_share_safe",
    "assert_share_safe_text",
    "require_human_approval",
]
