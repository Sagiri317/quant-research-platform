"""Shared run id contract."""

from __future__ import annotations

from quant_research_platform.policy.identifiers import validate_run_identifier


class RunIdValidationError(ValueError):
    """Raised when a run id is not public-safe for contracts or paths."""


def validate_run_id(value: str) -> str:
    """Return a validated run id or raise ``RunIdValidationError``."""

    try:
        return validate_run_identifier(value)
    except ValueError as exc:
        raise RunIdValidationError("invalid_run_id") from exc


__all__ = ["RunIdValidationError", "validate_run_id"]
