"""Public-safe metadata sanitizers for downloadable artifact JSON."""

from __future__ import annotations

import os
import re
from typing import Final

REDACTED_PATH: Final = "[redacted-path]"

_DROP_KEYS: Final = frozenset({"input_file"})
_PATH_KEYS: Final = frozenset(
    {
        "file",
        "path",
        "source_file",
        "input_path",
        "output_path",
        "output_dir",
        "run_root",
    }
)
_PRIVATE_MARKERS: Final = (
    "data/raw_private/",
    "data/interim_private/",
    "data/processed_private/",
    "reports/private_real/",
    "raw_private/",
    "interim_private/",
    "processed_private/",
    "private_real/",
)
_WINDOWS_ABSOLUTE_RE: Final = re.compile(r"^[A-Za-z]:[\\/]")


def sanitize_public_metadata(value: object, *, key: str | None = None) -> object:
    """Return a JSON-like payload with local/private paths redacted."""

    if isinstance(value, dict):
        result: dict[str, object] = {}
        for item_key, item_value in value.items():
            if not isinstance(item_key, str):
                result[item_key] = sanitize_public_metadata(item_value)
                continue
            if item_key in _DROP_KEYS:
                continue
            sanitized = sanitize_public_metadata(item_value, key=item_key)
            result[item_key] = sanitized
        return result
    if isinstance(value, list):
        return [sanitize_public_metadata(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_metadata(item, key=key) for item in value]
    if isinstance(value, os.PathLike):
        path_value = os.fspath(value)
        if _should_redact_string(path_value, key=key):
            return REDACTED_PATH
        return path_value
    if isinstance(value, str) and _should_redact_string(value, key=key):
        return REDACTED_PATH
    return value


def _should_redact_string(value: str, *, key: str | None) -> bool:
    normalized = value.replace("\\", "/")
    lower = normalized.lower()
    if any(marker in lower for marker in _PRIVATE_MARKERS):
        return True
    if _WINDOWS_ABSOLUTE_RE.match(value) or normalized.startswith(("/", "~")):
        return True
    if key is None:
        return False
    key_lower = key.lower()
    is_path_key = key_lower in _PATH_KEYS or key_lower.endswith("_path")
    return is_path_key and ("/" in normalized or "\\" in value)


__all__ = ["REDACTED_PATH", "sanitize_public_metadata"]
