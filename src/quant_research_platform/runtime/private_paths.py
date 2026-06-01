"""Shared private-path marker helpers for local artifact boundaries."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

PRIVATE_PATH_MARKERS = (
    "private",
    "raw_private",
    "interim_private",
    "processed_private",
    "private_real",
    "vendor_private",
    "raw_vendor",
)


def text_has_private_marker(value: str) -> bool:
    """Return whether text contains a private data/output marker."""

    normalized = value.replace("\\", "/").casefold()
    return any(marker in normalized for marker in PRIVATE_PATH_MARKERS)


def path_has_private_marker(path: Path) -> bool:
    """Return whether a path is inside a private-marked boundary."""

    return text_has_private_marker(path.as_posix())


def looks_like_private_or_local_path(
    value: str,
    *,
    reject_parent_reference: bool = False,
) -> bool:
    """Return whether text looks like a private marker or machine-local path."""

    normalized = value.replace("\\", "/").casefold()
    if "/" in normalized:
        return True
    if re.match(r"^[a-z]:", normalized):
        return True
    if reject_parent_reference and ".." in normalized.split():
        return True
    return text_has_private_marker(normalized)


def require_private_outputs_for_private_input(
    input_path: Path,
    output_paths: Iterable[Path],
    *,
    error_message: str,
) -> None:
    """Reject private-marker input unless all row-level outputs are private-marked."""

    if not path_has_private_marker(input_path):
        return
    if not all(path_has_private_marker(path) for path in output_paths):
        raise ValueError(error_message)


__all__ = [
    "PRIVATE_PATH_MARKERS",
    "looks_like_private_or_local_path",
    "path_has_private_marker",
    "require_private_outputs_for_private_input",
    "text_has_private_marker",
]
