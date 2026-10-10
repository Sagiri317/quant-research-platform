"""Protect credentials and local absolute paths in research outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from quant_research_platform.contracts.outputs import validate_output_identifier
from quant_research_platform.contracts.run_id import validate_run_id

_RETIRED_LOCAL_ONLY_REPORT_PATH_NAMES: tuple[str, ...] = (
    "private_real",
    "private_data",
    "private_research",
)


LOCAL_ONLY_DATA_LAYERS: tuple[str, ...] = (
    "raw",
    "interim",
    "processed",
    "research",
    "reference_docs",
)


_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?i)\b[a-z]:[\\/][^\r\n\t\"'<>|]+")


_UNC_PATH_RE = re.compile(r"\\\\[^\r\n\t\"'<>|]+")


_POSIX_LOCAL_PATH_RE = re.compile(
    r"(?i)(?<![\w:/])/(?!/)[^\r\n\t\"'<>|,\s]+"
    r"(?:/[^\r\n\t\"'<>|,\s]+)*|"
    r"(?<![\w:/])//[^\r\n\t\"'<>|,\s]+(?:/[^\r\n\t\"'<>|,\s]+)*|"
    r"(?<!:)//(?:mnt/data|home|Users|workspace|workspaces|tmp|"
    r"var/tmp|private/tmp|etc|opt|srv)(?:/[^\r\n\t\"'<>|,\s]+)*"
)


_CREDENTIAL_RE = re.compile(
    r"(?i)['\"]?\b(?:\w*[_-])?(?:token|secret|credential|password|"
    r"api[_-]?key|private[_-]?key|authorization|auth)['\"]?\s*[:=]\s*"
    r"(?:\"(?:\\[\s\S]|[^\"\\])*\"?|'(?:\\[\s\S]|[^'\\])*'?|"
    r"(?:(?:bearer|basic)\s+)?[^\s,;\}\]\)\"']+)?"
)


_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)^(?:.*[_-])?(?:token|secret|credential|password|api[_-]?key|"
    r"authorization|auth|private[_-]?key)$"
)


_TRACEBACK_RE = re.compile(r"(?is)Traceback \(most recent call last\):.*")


def _parts(path: Path) -> tuple[str, ...]:
    text = path.as_posix()
    if ":" in text:
        text = text.split(":", 1)[1]
    return tuple(part for part in PurePosixPath(text).parts if part not in {"", "/"})


def is_local_only_path(path: Path) -> bool:
    parts = _parts(path)
    if not parts:
        return False
    for index, part in enumerate(parts):
        if part == "outputs":
            return True
        if (
            index < len(parts) - 1
            and part == "data"
            and parts[index + 1] in LOCAL_ONLY_DATA_LAYERS
        ):
            return True
    return any(
        parts[index] == "reports"
        and parts[index + 1] in _RETIRED_LOCAL_ONLY_REPORT_PATH_NAMES
        for index in range(max(len(parts) - 1, 0))
    )


def unsafe_for_share_reason(text: object) -> str | None:
    value = str(text)
    if _TRACEBACK_RE.search(value):
        return "traceback"
    if _CREDENTIAL_RE.search(value):
        return "credential"
    if (
        _WINDOWS_DRIVE_PATH_RE.search(value)
        or _UNC_PATH_RE.search(value)
        or _POSIX_LOCAL_PATH_RE.search(value)
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        return "local_path"
    return None


def evidence_text_unsafe_reason(text: str) -> str | None:
    reason = unsafe_for_share_reason(text)
    return "absolute_path" if reason == "local_path" else reason


def evidence_json_value_unsafe_reason(
    value: object, *, key: str | None = None
) -> str | None:
    if key is not None:
        if _CREDENTIAL_KEY_RE.fullmatch(key):
            return "credential"
        reason = evidence_text_unsafe_reason(key)
        if reason is not None:
            return reason
    if key == "run_id":
        if not isinstance(value, str):
            return "invalid_run_id"
        try:
            validate_run_id(value)
        except ValueError:
            return "invalid_run_id"
    if key in {"output_id", "output_definition_id"}:
        if not isinstance(value, str):
            return "invalid_output_id"
        try:
            validate_output_identifier(value)
        except ValueError:
            return "invalid_output_id"
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            reason = evidence_json_value_unsafe_reason(item, key=str(item_key))
            if reason is not None:
                return reason
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            reason = evidence_json_value_unsafe_reason(item, key=key)
            if reason is not None:
                return reason
    elif isinstance(value, (str, Path)):
        return evidence_text_unsafe_reason(str(value))
    return None


def sanitize_shared_text(text: object) -> str:
    """Redact credentials, tracebacks and absolute paths; preserve research facts."""
    result = _TRACEBACK_RE.sub("[traceback]", str(text))
    result = _CREDENTIAL_RE.sub("[credential]", result)
    result = _WINDOWS_DRIVE_PATH_RE.sub("[local-path]", result)
    result = _UNC_PATH_RE.sub("[local-path]", result)
    return _POSIX_LOCAL_PATH_RE.sub("[local-path]", result)


def assert_internal_artifact_metadata_safe(value: object) -> None:
    reason = evidence_json_value_unsafe_reason(value)
    if reason is not None:
        raise ValueError(f"unsafe_internal_artifact_metadata: {reason}")
