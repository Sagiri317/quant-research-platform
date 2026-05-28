"""Shared helpers for manifest.display table and download contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
)

DISPLAY_PREVIEW_ROWS_DEFAULT = 20
DISPLAY_PREVIEW_ROWS_MAX = 1000


@dataclass(frozen=True)
class DisplayTablePreview:
    """Portable preview contract for one manifest.display table."""

    default_limit: int
    max_limit: int
    supports_offset: bool
    supports_combination_filter: bool
    filter_parameter: str | None


def build_display_table_preview_descriptor(
    *,
    table_key: str,
    supports_combination_filter: bool,
    default_limit: int = DISPLAY_PREVIEW_ROWS_DEFAULT,
    max_limit: int = DISPLAY_PREVIEW_ROWS_MAX,
) -> dict[str, object]:
    """Build the portable preview descriptor persisted in manifest.display."""

    return {
        "kind": "table",
        "table_key": table_key,
        "default_limit": default_limit,
        "max_limit": max_limit,
        "supports_offset": True,
        "supports_combination_filter": supports_combination_filter,
        "filter_parameter": ("combination_id" if supports_combination_filter else None),
    }


def display_table_by_key(
    manifest: dict[str, object],
    table_key: str,
) -> dict[str, object]:
    """Return a display table entry by key."""

    if not table_key:
        raise ValueError("display table key must be non-empty.")
    display = _require_supported_display(manifest)
    tables = display.get("tables")
    if not isinstance(tables, list):
        raise ValueError("manifest display.tables must be a list.")
    for table in tables:
        if isinstance(table, dict) and table.get("key") == table_key:
            return table
    raise ValueError(f"manifest display table {table_key!r} is not exposed.")


def display_table_artifact_path(
    manifest: dict[str, object],
    table_key: str,
) -> str:
    """Return a display table's artifact-relative path."""

    table = display_table_by_key(manifest, table_key)
    path = table.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"manifest display table {table_key!r} path is invalid.")
    validate_display_contract_path(path, source=f"display table {table_key}")
    return path


def resolve_display_table_artifact_path(
    manifest: dict[str, object],
    output_dir: Path,
    table_key: str,
) -> Path:
    """Resolve a display table path inside an artifact run directory."""

    relative_path = display_table_artifact_path(manifest, table_key)
    return resolve_artifact_contract_path(output_dir, relative_path, table_key)


def display_table_preview_limits(
    manifest: dict[str, object],
    table_key: str,
) -> DisplayTablePreview:
    """Return validated preview limits for one display table."""

    table = display_table_by_key(manifest, table_key)
    preview = table.get("preview")
    if not isinstance(preview, dict):
        raise ValueError(f"manifest display table {table_key!r} preview is required.")
    return _preview_from_mapping(preview, table_key=table_key)


def display_download_items(manifest: dict[str, object]) -> set[str]:
    """Return compact download refs exposed by manifest.display.downloads.items."""

    display = _require_supported_display(manifest)
    downloads = display.get("downloads")
    if not isinstance(downloads, dict) or not isinstance(downloads.get("items"), list):
        raise ValueError("manifest display.downloads.items must be a list.")
    result: set[str] = set()
    for option in downloads["items"]:
        if not isinstance(option, dict):
            raise ValueError(
                "manifest display.downloads.items entries must be objects."
            )
        kind = option.get("kind")
        key = option.get("key")
        if not isinstance(kind, str) or not isinstance(key, str) or not key:
            raise ValueError("manifest display download item must include kind/key.")
        result.add(f"{kind}:{key}")
    return result


def display_download_table_key(
    manifest: dict[str, object],
    item: str,
) -> str | None:
    """Return the display table key for a download item, when it is a table."""

    display = _require_supported_display(manifest)
    downloads = display.get("downloads")
    if not isinstance(downloads, dict):
        return None
    items = downloads.get("items")
    if not isinstance(items, list):
        return None
    for option in items:
        if not isinstance(option, dict):
            continue
        if f"{option.get('kind')}:{option.get('key')}" != item:
            continue
        if option.get("kind") != "table":
            return None
        table_key = option.get("table") or option.get("key")
        return table_key if isinstance(table_key, str) else None
    return None


def validate_display_contract_path(path: str, *, source: str) -> None:
    """Reject absolute paths, parent traversal, and platform-specific paths."""

    parsed = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"manifest {source} path is unsafe: {path!r}.")


def resolve_artifact_contract_path(
    output_dir: Path,
    relative_path: str,
    source: str,
) -> Path:
    """Resolve an artifact-relative path while keeping it inside output_dir."""

    validate_display_contract_path(relative_path, source=source)
    candidate = (output_dir / relative_path).resolve()
    resolved_output_dir = output_dir.resolve()
    if (
        resolved_output_dir not in candidate.parents
        and candidate != resolved_output_dir
    ):
        raise ValueError(f"artifact path out of bounds: {source}")
    return candidate


def _preview_from_mapping(
    preview: dict[Any, Any],
    *,
    table_key: str,
) -> DisplayTablePreview:
    if preview.get("kind") != "table":
        raise ValueError(f"manifest display table {table_key!r} preview kind invalid.")
    if preview.get("table_key") != table_key:
        raise ValueError(
            f"manifest display table {table_key!r} preview table_key mismatch."
        )
    default_limit = preview.get("default_limit")
    max_limit = preview.get("max_limit")
    supports_offset = preview.get("supports_offset")
    supports_combination_filter = preview.get("supports_combination_filter")
    filter_parameter = preview.get("filter_parameter")
    if not isinstance(default_limit, int) or default_limit <= 0:
        raise ValueError(
            f"manifest display table {table_key!r} preview default_limit invalid."
        )
    if (
        not isinstance(max_limit, int)
        or max_limit < default_limit
        or max_limit > DISPLAY_PREVIEW_ROWS_MAX
    ):
        raise ValueError(
            f"manifest display table {table_key!r} preview max_limit invalid."
        )
    if supports_offset is not True:
        raise ValueError(
            f"manifest display table {table_key!r} preview supports_offset invalid."
        )
    if not isinstance(supports_combination_filter, bool):
        raise ValueError(
            f"manifest display table {table_key!r} preview filter support invalid."
        )
    if supports_combination_filter:
        if filter_parameter != "combination_id":
            raise ValueError(
                f"manifest display table {table_key!r} preview filter_parameter "
                "must be combination_id."
            )
    elif filter_parameter is not None:
        raise ValueError(
            f"manifest display table {table_key!r} preview filter_parameter "
            "must be absent."
        )
    return DisplayTablePreview(
        default_limit=default_limit,
        max_limit=max_limit,
        supports_offset=supports_offset,
        supports_combination_filter=supports_combination_filter,
        filter_parameter=filter_parameter,
    )


def _require_supported_display(manifest: dict[str, object]) -> dict[str, object]:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("manifest.json must contain a display object.")
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("manifest display schema_version must be supported.")
    return display


__all__ = [
    "DISPLAY_PREVIEW_ROWS_DEFAULT",
    "DISPLAY_PREVIEW_ROWS_MAX",
    "DisplayTablePreview",
    "build_display_table_preview_descriptor",
    "display_download_items",
    "display_download_table_key",
    "display_table_artifact_path",
    "display_table_by_key",
    "display_table_preview_limits",
    "resolve_artifact_contract_path",
    "resolve_display_table_artifact_path",
    "validate_display_contract_path",
]
