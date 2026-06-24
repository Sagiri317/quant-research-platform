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
    "private_real",
    "private_data",
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
    r"(?<![A-Za-z0-9.-])(?:[03468]\d{5})"
    r"(?:\.(?:SZ|SH|BJ))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ROW_SAMPLE_RE = re.compile(
    r"(?is)[\"']?\brow[-_ ]?samples?\b[\"']?\s*[:=]\s*"
    r"(?:\[[^\]]*\]|\{[^}]*\}|[^\r\n;]+)|"
    r"\bmarket_id\s*=\s*[^\s,;]+.*\basset_id\s*=\s*[^\s,;]+|"
    r"\basset_id\s*=\s*[^\s,;]+.*\btimestamp\s*="
)
_ROW_SAMPLE_MARKER_RE = re.compile(r"(?i)\brow[-_ ]?samples?\b")
_PROVIDER_SOURCE_FIELD_RE = re.compile(
    r"""(?ix)
    ["']?
    (?:source_id|provider_id|source[-_ ]?name|provider[-_ ]?name|
       data[-_ ]?provider|vendor[-_ ]?name|provider[-_ ]?marker|
       source[-_ ]?marker|vendor[-_ ]?marker)
    ["']?
    \s*[:=]\s*
    (?:
        ["'][^"']*["']
        |
        [^\s,;}\]]+
    )
    """
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
_RAW_EXPORT_FILENAME_RE = re.compile(
    r"(?i)\b[a-z0-9_]+__[^\\/]+__export-\d{8}\.(?:csv|txt)\b"
)
_SOURCE_LOCAL_KEY_RE = re.compile(
    r"(?i)\b(?:source_asset_id|source_market_id)\b"
)
_PRIVATE_RUNTIME_PAYLOAD_RE = re.compile(
    r"(?i)\b(?:runtime_payload|private_runtime_payload|"
    r"exact_private_runtime_payload)\b"
)
_PRIVATE_RUN_PATH_RE = re.compile(
    r"(?i)(?:^|[\s`'\"])\.run[\\/]+(?:private|smoke|s[0-9]+)"
    r"(?:[\\/]+|[\s`'\"]|$)"
)
_ENVIRONMENT_PATH_RE = re.compile(
    r"(?i)(?:\bquant-workspace\b|\\\\quant-workspace\\\\|"
    r"\bsite-packages\b|\bconda env path\b|"
    r"(?:anaconda3|miniconda3)[\\/](?:envs|lib[\\/]site-packages))"
)
_ROW_SAMPLE_KEYS = {
    "row",
    "rows",
    "sample",
    "samples",
    "row_sample",
    "row_samples",
}


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


def public_text_unsafe_reason(
    text: object,
    *,
    allow_logical_private_path: bool = False,
) -> str | None:
    value = str(text)
    if _TRACEBACK_RE.search(value):
        return "traceback"
    if _CREDENTIAL_RE.search(value):
        return "credential"
    if _PRIVATE_RUN_PATH_RE.search(value):
        return "private_run_path"
    if _ABSOLUTE_REPO_PRIVATE_PATH_RE.search(value):
        return "private_path"
    if allow_logical_private_path:
        value = _REPO_PRIVATE_PATH_RE.sub("[logical-private-path]", value)
    elif _REPO_PRIVATE_PATH_RE.search(value):
        return "private_path"
    if (
        _WINDOWS_DRIVE_PATH_RE.search(value)
        or _UNC_PATH_RE.search(value)
        or _POSIX_PRIVATE_PATH_RE.search(value)
        or _ENVIRONMENT_PATH_RE.search(value)
    ):
        return "local_path"
    if _ROW_SAMPLE_RE.search(value) or _ROW_SAMPLE_MARKER_RE.search(value):
        return "row_sample"
    if _RAW_EXPORT_FILENAME_RE.search(value):
        return "raw_export_filename"
    if _SOURCE_LOCAL_KEY_RE.search(value):
        return "source_local_key"
    if _PRIVATE_RUNTIME_PAYLOAD_RE.search(value):
        return "private_runtime_payload"
    if _PROVIDER_SOURCE_FIELD_RE.search(value):
        return "private_marker"
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
    result = _PRIVATE_RUN_PATH_RE.sub("[private-run-path]", result)
    result = _ABSOLUTE_REPO_PRIVATE_PATH_RE.sub("[private-path]", result)
    result = _WINDOWS_DRIVE_PATH_RE.sub("[local-path]", result)
    result = _UNC_PATH_RE.sub("[local-path]", result)
    result = _POSIX_PRIVATE_PATH_RE.sub("[local-path]", result)
    result = _ENVIRONMENT_PATH_RE.sub("[local-path]", result)
    result = _REPO_PRIVATE_PATH_RE.sub("[private-path]", result)
    result = _ROW_SAMPLE_RE.sub("[row-sample]", result)
    result = _ROW_SAMPLE_MARKER_RE.sub("[row-sample]", result)
    result = _RAW_EXPORT_FILENAME_RE.sub("[raw-export]", result)
    result = _SOURCE_LOCAL_KEY_RE.sub("[source-local-key]", result)
    result = _PRIVATE_RUNTIME_PAYLOAD_RE.sub("[private-runtime-payload]", result)
    result = _PROVIDER_SOURCE_FIELD_RE.sub("[private]", result)
    result = _PRIVATE_MARKER_RE.sub("[private]", result)
    result = _STOCK_CODE_RE.sub("[asset-id]", result)
    return result


def sanitize_public_object(value: Any) -> Any:
    """Recursively sanitize mapping/list scalar values for public-safe metadata."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            safe_key = sanitize_public_text(key_text)
            sanitized[safe_key] = (
                "[row-sample]"
                if key_text.lower() in _ROW_SAMPLE_KEYS and item
                else sanitize_public_object(item)
            )
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_public_object(item) for item in value)
    if isinstance(value, set):
        return sorted(sanitize_public_object(item) for item in value)
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value


def assert_internal_artifact_metadata_safe(value: Any) -> None:
    """Reject path, credential, row-level, or asset-like internal metadata."""

    unsafe = internal_artifact_metadata_unsafe_reason(value)
    if unsafe is not None:
        raise ValueError(f"unsafe_internal_artifact_metadata: {unsafe}")


def internal_artifact_metadata_unsafe_reason(
    value: Any,
    *,
    key: str | None = None,
) -> str | None:
    """Return why internal artifact metadata is unsafe, if applicable."""

    if key is not None and key.lower() in _ROW_SAMPLE_KEYS and value:
        return "row_sample"
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            reason = internal_artifact_metadata_unsafe_reason(
                item,
                key=str(item_key),
            )
            if reason is not None:
                return reason
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            reason = internal_artifact_metadata_unsafe_reason(item, key=key)
            if reason is not None:
                return reason
        return None
    text = str(value) if isinstance(value, (str, Path)) else None
    if text is None:
        return None
    logical_path_key = key in {
        "manifest_path",
        "missing_dataset_manifests",
        "output_root",
    }
    if _ABSOLUTE_REPO_PRIVATE_PATH_RE.search(text):
        return "private_path"
    if not logical_path_key and _REPO_PRIVATE_PATH_RE.search(text):
        return "private_path"
    if (
        _WINDOWS_DRIVE_PATH_RE.search(text)
        or _UNC_PATH_RE.search(text)
        or _POSIX_PRIVATE_PATH_RE.search(text)
    ):
        return "absolute_path"
    if _RAW_EXPORT_FILENAME_RE.search(text):
        return "raw_export_filename"
    if _CREDENTIAL_RE.search(text):
        return "credential"
    if _STOCK_CODE_RE.search(text):
        return "stock_code_sample"
    return None

__all__ = [
    "PRIVATE_DATA_LAYERS",
    "PRIVATE_BOUNDARY_ROOTS",
    "PRIVATE_REPORT_PATH_NAMES",
    "assert_internal_artifact_metadata_safe",
    "internal_artifact_metadata_unsafe_reason",
    "is_private_boundary_path",
    "public_text_unsafe_reason",
    "sanitize_public_object",
    "sanitize_public_text",
]
