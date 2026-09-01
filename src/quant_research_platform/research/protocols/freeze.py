"""Explicit development/holdout protocol gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolFreeze:
    protocol_hash: str
    git_commit: str
    git_tree: str
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.frozen or any(
            type(value) is not str or not value
            for value in (self.protocol_hash, self.git_commit, self.git_tree)
        ):
            raise ValueError("protocol_freeze_invalid")


def require_phase_access(*, phase: str, protocol_frozen: bool) -> None:
    if phase == "holdout" and protocol_frozen is not True:
        raise PermissionError("holdout_protocol_freeze_required")
    if phase not in {"development", "holdout"}:
        raise ValueError("research_phase_invalid")


def invalidate_holdout_evidence(*, bugfix_changes_results: bool) -> str:
    return "INVALIDATED_BY_BUGFIX" if bugfix_changes_results else "VALID"
