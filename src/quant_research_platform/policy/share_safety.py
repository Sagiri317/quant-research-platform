"""Share-safety rules and sanitization helpers."""

from __future__ import annotations

import codecs
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from quant_research_platform.contracts.outputs import validate_output_identifier
from quant_research_platform.contracts.run_id import validate_run_id

LOCAL_ONLY_ROOTS: tuple[str, ...] = (
    "data/raw",
    "data/interim",
    "data/processed",
    "data/research",
    "outputs",
)
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
    r"(?<!:)//(?:mnt/data|home|Users|workspace|workspaces|tmp|"
    r"var/tmp|private/tmp|etc|opt|srv)(?:/[^\r\n\t\"'<>|,\s]+)*"
)
_RETIRED_LOCAL_ONLY_REPORT_PATH_PATTERN = "|".join(
    re.escape(path_name) for path_name in _RETIRED_LOCAL_ONLY_REPORT_PATH_NAMES
)
_ABSOLUTE_REPO_LOCAL_ONLY_PATH_RE = re.compile(
    r"(?i)(?<!\w)(?:[a-z]:[\\/]|/)(?:[^\s,;\"'<>|]+[/\\])*"
    r"(?:data[/\\](?:raw|interim|processed|research|reference_docs)"
    r"(?:[/\\][^\s,;\"'<>|]+)?|outputs"
    r"(?:[/\\][^\s,;\"'<>|]+)?|reports[/\\](?:"
    + _RETIRED_LOCAL_ONLY_REPORT_PATH_PATTERN
    + r")"
    r"(?:[/\\][^\s,;\"'<>|]+)?)"
)
_REPO_LOCAL_ONLY_PATH_RE = re.compile(
    r"(?i)\b(?:data[/\\](?:raw|interim|processed|research|reference_docs)"
    r"(?:[/\\][^\s,;\"'<>|]+)?|outputs[/\\][^\s,;\"'<>|]*|reports[/\\](?:"
    + _RETIRED_LOCAL_ONLY_REPORT_PATH_PATTERN
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
_PROVIDER_SOURCE_MARKER_RE = re.compile(
    r"(?i)\b(?:csmar|source_id|provider_id|implementation_package|"
    r"source[-_ ]?name|provider[-_ ]?name|data[-_ ]?provider|vendor[-_ ]?name|"
    r"provider[-_ ]?marker|source[-_ ]?marker|vendor[-_ ]?marker|"
    r"vendor[-_ ]?payload|raw[-_ ]?vendor|vendor(?:[-_ ]?source)?|"
    r"raw[-_ ]?source[-_ ]?marker)\b"
)
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:token|secret|credential|password|api[_-]?key|private[_-]?key)\s*[:=]\s*"
    r"(?:\"[^\r\n\"]+\"|'[^\r\n']+'|[^\s,;\}\]\)\"']+)"
)
_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)^(?:.*[_-])?(?:token|secret|credential|password|api[_-]?key|"
    r"authorization|auth|private[_-]?key)$"
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\b(?:authorization|auth)\s*[:=]\s*"
    r"['\"]?(?:bearer|basic)\s+[^\s,;\}\]\)\"']+['\"]?"
)
_TRACEBACK_RE = re.compile(r"(?is)Traceback \(most recent call last\):.*")
_RAW_EXPORT_FILENAME_RE = re.compile(
    r"(?i)\b[a-z0-9_]+__[^\\/]+__export-\d{8}\.(?:csv|txt)\b"
)
_SOURCE_LOCAL_KEY_RE = re.compile(r"(?i)\b(?:source_asset_id|source_market_id)\b")
SOURCE_LOCAL_KEYS = frozenset({"source_asset_id", "source_market_id"})
JSON_LOGICAL_PATH_VALUE_KEYS = frozenset(
    {
        "manifest_path",
        "path",
        "invalid",
        "invalid_dataset_manifests",
        "missing",
        "missing_dataset_manifests",
        "requested_paths",
        "run_root",
        "run_manifest_path",
        "validation_summary_path",
    }
)
_RUNTIME_PAYLOAD_RE = re.compile(r"(?i)\b(?:runtime_payload|exact_runtime_payload)\b")
_LOCAL_RUN_PATH_RE = re.compile(
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
MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS = 8 * 1024


def evidence_text_stream_unsafe_reason(
    chunks: Iterable[bytes | bytearray | memoryview],
) -> str | None:
    """Scan persisted UTF-8 evidence with bounded retained text state."""

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    pending = ""
    seen_market_id = False
    seen_asset_id = False

    def scan_line(line: str) -> str | None:
        nonlocal seen_asset_id, seen_market_id
        reason = evidence_text_unsafe_reason(line)
        if reason is not None:
            return reason
        has_market_id = re.search(r"\bmarket_id\s*=", line) is not None
        has_asset_id = re.search(r"\basset_id\s*=", line) is not None
        has_timestamp = re.search(r"\btimestamp\s*=", line) is not None
        if (seen_market_id and has_asset_id) or (seen_asset_id and has_timestamp):
            return "row_sample"
        seen_market_id = seen_market_id or has_market_id
        seen_asset_id = seen_asset_id or has_asset_id
        return None

    def consume(decoded: str) -> str | None:
        nonlocal pending
        if not decoded:
            return None
        parts = decoded.split("\n")
        for index, part in enumerate(parts):
            if len(pending) + len(part) > MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS:
                return "evidence_capacity_exceeded"
            if part:
                pending += part
            if index == len(parts) - 1:
                continue
            reason = scan_line(pending)
            if reason is not None:
                return reason
            pending = ""
        return None

    for chunk in chunks:
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("evidence_text_bytes_required")
        view = memoryview(chunk)
        for offset in range(0, len(view), MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS):
            reason = consume(
                decoder.decode(
                    view[offset : offset + MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS],
                    final=False,
                )
            )
            if reason is not None:
                return reason
    reason = consume(decoder.decode(b"", final=True))
    return reason if reason is not None else scan_line(pending)


def _parts(path: Path) -> tuple[str, ...]:
    text = path.as_posix()
    if ":" in text:
        text = text.split(":", 1)[1]
    return tuple(part for part in PurePosixPath(text).parts if part not in {"", "/"})


def is_local_only_path(path: Path) -> bool:
    parts = _parts(path)
    if not parts:
        return False
    if "docs" in parts:
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


def local_run_root_reason(run_root: Path, repo_root: Path) -> str | None:
    """Return why an explicitly selected local run root is unsafe."""

    root = Path(repo_root).resolve(strict=False)
    target = Path(run_root).resolve(strict=False)
    if target == root:
        return "run_root_must_not_be_repo_root"
    return None


def unsafe_for_share_reason(
    text: object,
    *,
    allow_logical_local_only_path: bool = False,
) -> str | None:
    value = str(text)
    if _TRACEBACK_RE.search(value):
        return "traceback"
    if _CREDENTIAL_RE.search(value) or _AUTHORIZATION_RE.search(value):
        return "credential"
    if _LOCAL_RUN_PATH_RE.search(value):
        return "local_run_path"
    if _ABSOLUTE_REPO_LOCAL_ONLY_PATH_RE.search(value):
        return "local_only_path"
    if (
        _WINDOWS_DRIVE_PATH_RE.search(value)
        or _UNC_PATH_RE.search(value)
        or _POSIX_LOCAL_PATH_RE.search(value)
        or _ENVIRONMENT_PATH_RE.search(value)
    ):
        return "local_path"
    if allow_logical_local_only_path:
        value = _REPO_LOCAL_ONLY_PATH_RE.sub("[logical-local-only-path]", value)
    elif _REPO_LOCAL_ONLY_PATH_RE.search(value):
        return "local_only_path"
    if _ROW_SAMPLE_RE.search(value) or _ROW_SAMPLE_MARKER_RE.search(value):
        return "row_sample"
    if _RAW_EXPORT_FILENAME_RE.search(value):
        return "raw_export_filename"
    if _SOURCE_LOCAL_KEY_RE.search(value):
        return "source_local_key"
    if _RUNTIME_PAYLOAD_RE.search(value):
        return "runtime_payload"
    if _PROVIDER_SOURCE_FIELD_RE.search(value):
        return "provider_source_marker"
    if _PROVIDER_SOURCE_MARKER_RE.search(value):
        return "provider_source_marker"
    if _STOCK_CODE_RE.search(value):
        return "stock_code_sample"
    return None


def evidence_text_unsafe_reason(
    text: str,
    *,
    allow_logical_local_only_path: bool = False,
) -> str | None:
    shared_reason = unsafe_for_share_reason(
        text,
        allow_logical_local_only_path=allow_logical_local_only_path,
    )
    if shared_reason in {"local_only_path", "local_path"}:
        return "absolute_path"
    if shared_reason == "credential":
        return "credential"
    if shared_reason == "stock_code_sample":
        return "stock_sample"
    if shared_reason == "row_sample":
        return "row_sample"
    if shared_reason == "provider_source_marker":
        return "vendor_marker"
    if shared_reason == "traceback":
        return "traceback"
    if shared_reason == "raw_export_filename":
        return "vendor_filename"
    if shared_reason == "source_local_key":
        return "source_local_key"
    if shared_reason == "runtime_payload":
        return "runtime_payload"
    if shared_reason == "local_run_path":
        return "absolute_path"
    return None


def evidence_json_value_unsafe_reason(
    value: object,
    *,
    key: str | None = None,
) -> str | None:
    if key == "run_id":
        if not isinstance(value, str):
            return "invalid_run_id"
        try:
            validate_run_id(value)
        except ValueError:
            return "invalid_run_id"
        return None
    if key in {"output_id", "output_definition_id"}:
        if not isinstance(value, str):
            return "invalid_output_id"
        try:
            validate_output_identifier(value)
        except ValueError:
            return "invalid_output_id"
        return None
    if key is not None:
        if _CREDENTIAL_KEY_RE.fullmatch(key):
            return "credential"
        key_reason = evidence_text_unsafe_reason(
            key,
            allow_logical_local_only_path=True,
        )
        if key_reason is not None:
            return key_reason
    if isinstance(value, Mapping):
        for item_key, item in value.items():
            reason = evidence_json_value_unsafe_reason(item, key=str(item_key))
            if reason is not None:
                return reason
        return None
    if isinstance(value, list | tuple):
        for item in value:
            reason = evidence_json_value_unsafe_reason(item, key=key)
            if reason is not None:
                return reason
        return None
    if isinstance(value, str):
        return evidence_text_unsafe_reason(
            value,
            allow_logical_local_only_path=key in JSON_LOGICAL_PATH_VALUE_KEYS,
        )
    return None


def contains_source_local_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in SOURCE_LOCAL_KEYS or contains_source_local_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return any(contains_source_local_key(item) for item in value)
    return isinstance(value, str) and value in SOURCE_LOCAL_KEYS


def sanitize_shared_text(text: object) -> str:
    """Redact secrets and local implementation details from shared text."""

    result = str(text)
    result = _TRACEBACK_RE.sub("[traceback]", result)
    result = _CREDENTIAL_RE.sub("[credential]", result)
    result = _AUTHORIZATION_RE.sub("[credential]", result)
    result = _LOCAL_RUN_PATH_RE.sub("[local-run-path]", result)
    result = _ABSOLUTE_REPO_LOCAL_ONLY_PATH_RE.sub("[local-only-path]", result)
    result = _WINDOWS_DRIVE_PATH_RE.sub("[local-path]", result)
    result = _UNC_PATH_RE.sub("[local-path]", result)
    result = _POSIX_LOCAL_PATH_RE.sub("[local-path]", result)
    result = _ENVIRONMENT_PATH_RE.sub("[local-path]", result)
    result = _REPO_LOCAL_ONLY_PATH_RE.sub("[local-only-path]", result)
    return result


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
        "invalid_dataset_manifests",
        "missing_dataset_manifests",
        "run_root",
    }
    if _ABSOLUTE_REPO_LOCAL_ONLY_PATH_RE.search(text):
        return "local_only_path"
    if not logical_path_key and _REPO_LOCAL_ONLY_PATH_RE.search(text):
        return "local_only_path"
    if (
        _WINDOWS_DRIVE_PATH_RE.search(text)
        or _UNC_PATH_RE.search(text)
        or _POSIX_LOCAL_PATH_RE.search(text)
    ):
        return "absolute_path"
    if _RAW_EXPORT_FILENAME_RE.search(text):
        return "raw_export_filename"
    if _CREDENTIAL_RE.search(text) or _AUTHORIZATION_RE.search(text):
        return "credential"
    if _STOCK_CODE_RE.search(text):
        return "stock_code_sample"
    return None
