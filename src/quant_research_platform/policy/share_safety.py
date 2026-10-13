"""Protect credentials and local absolute paths in research outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from quant_research_platform.contracts.outputs import validate_output_identifier

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
    """Private native artifacts may name real inputs, but never store credentials."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _CREDENTIAL_KEY_RE.fullmatch(str(key)):
                raise ValueError("unsafe_internal_artifact_metadata: credential")
            assert_internal_artifact_metadata_safe(str(key))
            assert_internal_artifact_metadata_safe(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            assert_internal_artifact_metadata_safe(item)
    elif isinstance(value, (str, Path)):
        text = str(value)
        if _CREDENTIAL_RE.search(text):
            raise ValueError("unsafe_internal_artifact_metadata: credential")
        if _TRACEBACK_RE.search(text):
            raise ValueError("unsafe_internal_artifact_metadata: traceback")
