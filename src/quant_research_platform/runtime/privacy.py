"""Private boundary and public text sanitization helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

PRIVATE_BOUNDARY_ROOTS: tuple[str, ...] = (
    "data/raw",
    "data/interim",
    "data/processed",
    "data/research",
    "reports/private_research",
)
PRIVATE_REPORT_PATH_NAMES: tuple[str, ...] = (
    "private_research",
    "private" + "_real",
    "private" + "_data",
)
PRIVATE_DATA_LAYERS: tuple[str, ...] = (
    "raw",
    "interim",
    "processed",
    "research",
    "reference_docs",
)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?i)\b[a-z]:[\\/][^\r\n\t\"'<>|]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\r\n\t\"'<>|]+")
_POSIX_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?<!\w)/(?:mnt/data|home|Users|workspace|workspaces|tmp|var/tmp|private/tmp)"
    r"/[^\r\n\t\"'<>|]+"
)
_PRIVATE_REPORT_PATH_PATTERN = "|".join(
    re.escape(path_name) for path_name in PRIVATE_REPORT_PATH_NAMES
)
_ABSOLUTE_REPO_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?<!\w)(?:[a-z]:[\\/]|/)(?:[^\s,;\"'<>|]+[/\\])*"
    r"(?:data[/\\](?:raw|interim|processed|research|reference_docs)"
    r"(?:[/\\][^\s,;\"'<>|]+)?|reports[/\\](?:"
    + _PRIVATE_REPORT_PATH_PATTERN
    + r")"
    r"(?:[/\\][^\s,;\"'<>|]+)?)"
)
_REPO_PRIVATE_PATH_RE = re.compile(
    r"(?i)\b(?:data[/\\](?:raw|interim|processed|research|reference_docs)"
    r"(?:[/\\][^\s,;\"'<>|]+)?|reports[/\\](?:"
    + _PRIVATE_REPORT_PATH_PATTERN
    + r")"
    r"(?:[/\\][^\s,;\"'<>|]+)?)"
)
_STOCK_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[03468]\d{5})"
    r"(?:\.(?:SZ|SH|BJ))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ROW_SAMPLE_RE = re.compile(
    r"(?i)\brow[-_ ]?samples?\b|"
    r"\bmarket_id\s*=\s*[^\s,;]+.*\basset_id\s*=\s*[^\s,;]+|"
    r"\basset_id\s*=\s*[^\s,;]+.*\btimestamp\s*="
)
_PRIVATE_MARKER_RE = re.compile(
    r"(?i)\b(?:csmar|source_id|provider_id|implementation_package|"
    r"source[-_ ]?name|provider[-_ ]?name|data[-_ ]?provider|vendor[-_ ]?name|"
    r"provider[-_ ]?marker|source[-_ ]?marker|vendor[-_ ]?marker|"
    r"vendor[-_ ]?payload|raw[-_ ]?vendor|vendor(?:[-_ ]?source)?|"
    r"raw[-_ ]?source[-_ ]?marker)\b"
)
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:token|secret|credential|password|api[_-]?key)\s*[:=]\s*"
    r"[A-Za-z0-9._~+/=-]{8,}"
)
_TRACEBACK_RE = re.compile(r"(?is)Traceback \(most recent call last\):.*")


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
        parts[index] == "reports" and parts[index + 1] in PRIVATE_REPORT_PATH_NAMES
        for index in range(max(len(parts) - 1, 0))
    )


def public_text_unsafe_reason(text: object) -> str | None:
    value = str(text)
    if _TRACEBACK_RE.search(value):
        return "traceback"
    if _CREDENTIAL_RE.search(value):
        return "credential"
    if (
        _ABSOLUTE_REPO_PRIVATE_PATH_RE.search(value)
        or _REPO_PRIVATE_PATH_RE.search(value)
    ):
        return "private_path"
    if (
        _WINDOWS_DRIVE_PATH_RE.search(value)
        or _UNC_PATH_RE.search(value)
        or _POSIX_PRIVATE_PATH_RE.search(value)
    ):
        return "local_path"
    if _ROW_SAMPLE_RE.search(value):
        return "row_sample"
    if _PRIVATE_MARKER_RE.search(value):
        return "private_marker"
    if _STOCK_CODE_RE.search(value):
        return "stock_code_sample"
    return None


def sanitize_public_text(text: object) -> str:
    """Return public-safe text for API, CLI, manifest, and report summaries."""

    result = str(text)
    result = _TRACEBACK_RE.sub("[traceback]", result)
    result = _CREDENTIAL_RE.sub("[credential]", result)
    result = _ABSOLUTE_REPO_PRIVATE_PATH_RE.sub("[private-path]", result)
    result = _WINDOWS_DRIVE_PATH_RE.sub("[local-path]", result)
    result = _UNC_PATH_RE.sub("[local-path]", result)
    result = _POSIX_PRIVATE_PATH_RE.sub("[local-path]", result)
    result = _REPO_PRIVATE_PATH_RE.sub("[private-path]", result)
    result = _ROW_SAMPLE_RE.sub("[row-sample]", result)
    result = _PRIVATE_MARKER_RE.sub("[private]", result)
    result = _STOCK_CODE_RE.sub("[asset-id]", result)
    return result


def sanitize_public_object(value: Any) -> Any:
    """Recursively sanitize mapping/list scalar values for public-safe metadata."""

    if isinstance(value, Mapping):
        return {
            sanitize_public_text(key): sanitize_public_object(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_public_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_public_object(item) for item in value)
    if isinstance(value, set):
        return sorted(sanitize_public_object(item) for item in value)
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value

__all__ = [
    "PRIVATE_DATA_LAYERS",
    "PRIVATE_BOUNDARY_ROOTS",
    "PRIVATE_REPORT_PATH_NAMES",
    "is_private_boundary_path",
    "public_text_unsafe_reason",
    "sanitize_public_object",
    "sanitize_public_text",
]
