"""Private boundary and public text sanitization helpers."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

PRIVATE_BOUNDARY_ROOTS: tuple[str, ...] = ("data", "reports/private_real")
PRIVATE_DATA_LAYERS: tuple[str, ...] = (
    "raw",
    "interim",
    "processed",
    "reference_docs",
)
_LOCAL_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/][^\s,;:]+|\\\\[^\s,;:]+)"
)
_STOCK_CODE_RE = re.compile(r"\b\d{6}(?:\.(?:SZ|SH|BJ))?\b", re.IGNORECASE)
_PRIVATE_MARKER_RE = re.compile(
    r"(?i)\b(?:data/raw/csmar|data/interim/csmar|data/processed|"
    r"reports/private_real|CSMAR|vendor)\b"
)


def _parts(path: Path) -> tuple[str, ...]:
    text = path.as_posix()
    if ":" in text:
        text = text.split(":", 1)[1]
    return tuple(part for part in PurePosixPath(text).parts if part not in {"", "/"})


def is_private_boundary_path(path: Path) -> bool:
    parts = _parts(path)
    if not parts:
        return False
    if "docs" in parts:
        return False
    for index, part in enumerate(parts[:-1]):
        if part == "data" and parts[index + 1] in PRIVATE_DATA_LAYERS:
            return True
    return any(
        parts[index : index + 2] == ("reports", "private_real")
        for index in range(max(len(parts) - 1, 0))
    )


def sanitize_public_text(text: object) -> str:
    result = str(text)
    result = _LOCAL_PATH_RE.sub("[local-path]", result)
    result = _PRIVATE_MARKER_RE.sub("[private]", result)
    result = _STOCK_CODE_RE.sub("[asset-id]", result)
    return result


__all__ = [
    "PRIVATE_DATA_LAYERS",
    "PRIVATE_BOUNDARY_ROOTS",
    "is_private_boundary_path",
    "sanitize_public_text",
]
