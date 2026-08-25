"""Shared run id contract."""

from __future__ import annotations

import re

_RUN_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class RunIdValidationError(ValueError):
    """Raised when a run id violates its contract grammar."""


def validate_run_id(value: str) -> str:
    """Return a validated run id or raise ``RunIdValidationError``."""

    if not isinstance(value, str) or _RUN_IDENTIFIER_RE.fullmatch(value) is None:
        raise RunIdValidationError("invalid_run_id")
    return value
