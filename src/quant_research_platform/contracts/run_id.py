"""Shared public-safe run id contract."""

from __future__ import annotations

import re

_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_STOCK_CODE_RE = re.compile(r"(?<!\d)\d{6}\.(?:sh|sz|bj)\b")
_BARE_A_SHARE_CODE_RE = re.compile(
    r"(?:(?<=^)|(?<=[._-]))"
    r"(?:000|001|002|003|300|301|600|601|603|605|688|689)\d{3}"
    r"(?=$|[._-])"
)
_FORBIDDEN_FRAGMENTS = (
    "csmar",
    "provider_id",
    "source_id",
    "implementation_package",
    "data/raw",
    "data\\raw",
    "data/interim",
    "data\\interim",
    "data/processed",
    "data\\processed",
    "data/research",
    "data\\research",
    "reports/private_real",
    "reports\\private_real",
)


class RunIdValidationError(ValueError):
    """Raised when a run id is not public-safe for contracts or paths."""


def validate_run_id(value: str) -> str:
    """Return a validated run id or raise ``RunIdValidationError``."""

    if not isinstance(value, str):
        raise RunIdValidationError("invalid_run_id")
    if not _RUN_ID_RE.fullmatch(value):
        raise RunIdValidationError("invalid_run_id")
    lowered = value.lower()
    if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
        raise RunIdValidationError("invalid_run_id")
    if _STOCK_CODE_RE.search(lowered):
        raise RunIdValidationError("invalid_run_id")
    if _BARE_A_SHARE_CODE_RE.search(lowered):
        raise RunIdValidationError("invalid_run_id")
    return value


__all__ = ["RunIdValidationError", "validate_run_id"]
