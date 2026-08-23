"""Identifier safety checks for run and output ids."""

from __future__ import annotations

import re

_RUN_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_OUTPUT_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
def validate_run_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid_run_id")
    if not _RUN_IDENTIFIER_RE.fullmatch(value):
        raise ValueError("invalid_run_id")
    return value


def validate_output_identifier(value: str) -> str:
    if not isinstance(value, str) or not _OUTPUT_IDENTIFIER_RE.fullmatch(value):
        raise ValueError("invalid_output_id")
    if value.casefold() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("invalid_output_id")
    return value
