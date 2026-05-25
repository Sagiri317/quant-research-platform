"""Artifact download package contract and ZIP assembly."""

from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from quant_research_platform.artifacts.artifact_loader import LoadedEvaluationSet
from quant_research_platform.artifacts.artifact_schema import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.public_metadata import sanitize_public_metadata


class ArtifactPackageError(ValueError):
    """Raised when an artifact download package request is invalid."""


_DynamicDownloadResolver = Callable[[], bytes]


@dataclass(frozen=True)
class _ResolvedDownloadItem:
    """Resolved physical or dynamic artifact download content."""

    name: str
    media_type: str
    source_path: Path | None = None
    content: bytes | None = None

    def require_content(self) -> bytes:
        if self.content is not None:
            return self.content
        if self.source_path is None:
            raise ArtifactPackageError(f"download item has no content: {self.name}")
        return self.source_path.read_bytes()


def _require_manifest_display_download_item(
    manifest: dict[str, object],
    item: str,
) -> None:
    """Require an item to be present in manifest.display.downloads.items."""

    _validate_download_item_name(item)
    if item not in _display_download_items(manifest):
        raise ArtifactPackageError(
            f"{item} is not exposed by manifest display downloads"
        )


def require_manifest_display_download_item(
    manifest: dict[str, object],
    item: str,
) -> None:
    """Require an item to be exposed by manifest.display.downloads.items."""

    _require_manifest_display_download_item(manifest, item)


def resolve_manifest_display_artifact_path(
    manifest: dict[str, object],
    output_dir: Path,
    item: str,
) -> Path | None:
    """Resolve a display-exposed manifest artifact path inside a run directory."""

    require_manifest_display_download_item(manifest, item)
    relative_path = _artifact_relative_path_from_manifest(manifest, item)
    if relative_path is None:
        return None
    if item in {"report.md", "manifest.json", "config.json", "summary.json"}:
        relative_path = item
    return _resolve_artifact_path(output_dir, relative_path, item)


def _display_download_items(manifest: dict[str, object]) -> set[str]:
    """Validate display version and return manifest.display download items."""

    display = _require_supported_display_manifest(manifest)
    downloads = display.get("downloads")
    if not isinstance(downloads, dict) or not isinstance(downloads.get("items"), list):
        raise ArtifactPackageError("manifest display downloads invalid")
    items: set[str] = set()
    for option in downloads["items"]:
        if not isinstance(option, dict):
            raise ArtifactPackageError("manifest display downloads invalid")
        item = option.get("item")
        if not isinstance(item, str) or not item:
            raise ArtifactPackageError("manifest display download item invalid")
        _validate_download_item_name(item)
        items.add(item)
    return items


def build_selected_artifact_zip(
    run: LoadedEvaluationSet,
    *,
    source_alias: str,
    run_id: str,
    items: list[str],
    cache_root: Path,
    dynamic_items: dict[str, _DynamicDownloadResolver] | None = None,
) -> Path:
    """Build or reuse a selected artifact ZIP and return its cache path."""

    unique_items = list(dict.fromkeys(items))
    if not unique_items:
        raise ArtifactPackageError("请选择至少一个下载项。")
    allowed_items = _display_download_items(run.manifest)
    for item in unique_items:
        _validate_download_item_name(item)
        if item not in allowed_items:
            raise ArtifactPackageError(f"未知下载项：{item}")

    dynamic_cache: dict[str, bytes] = {}

    def dynamic_resolver(item: str) -> bytes:
        if dynamic_items is None or item not in dynamic_items:
            raise ArtifactPackageError(f"未知下载项：{item}")
        if item not in dynamic_cache:
            dynamic_cache[item] = dynamic_items[item]()
        return dynamic_cache[item]

    resolved_dynamic_items = (
        {item: lambda item=item: dynamic_resolver(item) for item in dynamic_items}
        if dynamic_items
        else None
    )
    zip_path = _selected_zip_cache_path(
        run,
        source_alias=source_alias,
        run_id=run_id,
        items=unique_items,
        cache_root=cache_root,
        dynamic_items=resolved_dynamic_items,
    )
    if not zip_path.is_file():
        _write_selected_zip(run, unique_items, zip_path, resolved_dynamic_items)
    return zip_path


def _selected_zip_cache_path(
    run: LoadedEvaluationSet,
    *,
    source_alias: str,
    run_id: str,
    items: list[str],
    cache_root: Path,
    dynamic_items: dict[str, _DynamicDownloadResolver] | None = None,
) -> Path:
    """Return the cache path for a selected artifact ZIP."""

    _validate_cache_path_segment(source_alias, source="source_alias")
    _validate_cache_path_segment(run_id, source="run_id")
    fingerprint: list[str] = []
    for item in items:
        _validate_download_item_name(item)
        if item in {"config.json", "lineage.json"}:
            digest = hashlib.sha256(
                _public_metadata_bytes(run.output_dir, item)
            ).hexdigest()
            fingerprint.append(f"{item}\0dynamic\0{digest}")
            continue
        source_path = _download_item_path(run, item)
        if source_path is not None:
            stat = source_path.stat()
            fingerprint.append(
                f"{item}\0{source_path.as_posix()}\0{stat.st_size}\0"
                f"{stat.st_mtime_ns}"
            )
            continue
        if dynamic_items and item in dynamic_items:
            digest = hashlib.sha256(dynamic_items[item]()).hexdigest()
            fingerprint.append(f"{item}\0dynamic\0{digest}")
            continue
        fingerprint.append(f"{item}\0dynamic")
    digest = hashlib.sha256("\n".join(fingerprint).encode("utf-8")).hexdigest()[:16]
    resolved_cache_root = cache_root.resolve()
    zip_path = (
        resolved_cache_root
        / _cache_identity_segment(source_alias)
        / _cache_identity_segment(run_id)
        / f"{digest}.zip"
    ).resolve()
    if resolved_cache_root not in zip_path.parents:
        raise ArtifactPackageError("download cache path out of bounds")
    return zip_path


def _resolve_download_item(
    run: LoadedEvaluationSet,
    item: str,
    dynamic_items: dict[str, _DynamicDownloadResolver] | None = None,
) -> _ResolvedDownloadItem:
    """Resolve a physical, metadata, table, or dynamic download item."""

    _validate_download_item_name(item)
    if dynamic_items and item in dynamic_items:
        return _ResolvedDownloadItem(
            name=item,
            media_type=_media_type_for_item(item),
            content=dynamic_items[item](),
        )
    source_path = _download_item_path(run, item)
    if source_path is not None:
        return _ResolvedDownloadItem(
            name=item,
            media_type=_media_type_for_item(item),
            source_path=source_path,
        )
    if item == "report.md":
        return _ResolvedDownloadItem(
            name=item,
            media_type="text/markdown; charset=utf-8",
            content=run.report.encode("utf-8"),
        )
    if item == "manifest.json":
        return _ResolvedDownloadItem(
            name=item,
            media_type="application/json",
            content=_json_bytes(run.manifest),
        )
    if item == "config.json":
        return _ResolvedDownloadItem(
            name=item,
            media_type="application/json",
            content=_public_metadata_bytes(run.output_dir, item, fallback=run.config),
        )
    if item == "lineage.json":
        return _ResolvedDownloadItem(
            name=item,
            media_type="application/json",
            content=public_lineage_bytes(run.output_dir),
        )
    if item == "summary.json":
        return _ResolvedDownloadItem(
            name=item,
            media_type="application/json",
            content=_json_bytes(run.summary),
        )
    display = run.manifest.get("display")
    if isinstance(display, dict):
        downloads = display.get("downloads")
        if isinstance(downloads, dict):
            for option in downloads.get("items", []):
                if not isinstance(option, dict) or option.get("item") != item:
                    continue
                metric_name = option.get("table")
                if not isinstance(metric_name, str):
                    break
                frame = run.tables.get(metric_name)
                if frame is None:
                    break
                return _ResolvedDownloadItem(
                    name=item,
                    media_type="text/csv; charset=utf-8",
                    content=frame.to_csv(index=False).encode("utf-8"),
                )
    raise ArtifactPackageError(f"未知下载项：{item}")


def _download_item_path(run: LoadedEvaluationSet, item: str) -> Path | None:
    """Return an existing artifact file path; dynamic content returns None."""

    _validate_download_item_name(item)
    if item in {"config.json", "lineage.json"}:
        return None
    artifacts = run.manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    path = _artifact_relative_path(run, item, artifacts)
    if path is None:
        return None
    candidate = _resolve_artifact_path(run.output_dir, path, item)
    if not candidate.is_file():
        raise ArtifactPackageError(f"下载项文件不存在：{item}")
    return candidate


def _artifact_relative_path(
    run: LoadedEvaluationSet,
    item: str,
    artifacts: dict[object, object],
) -> str | None:
    _validate_download_item_name(item)
    if item == "report.md":
        value = artifacts.get("report")
        return value if value == item else None
    if item in {"manifest.json", "config.json", "lineage.json", "summary.json"}:
        return item
    display = run.manifest.get("display")
    if isinstance(display, dict):
        downloads = display.get("downloads")
        if isinstance(downloads, dict):
            for option in downloads.get("items", []):
                if not isinstance(option, dict) or option.get("item") != item:
                    continue
                table_name = option.get("table")
                if not isinstance(table_name, str):
                    return None
                return _table_relative_path(artifacts, table_name)
    return None


def _artifact_relative_path_from_manifest(
    manifest: dict[str, object],
    item: str,
) -> str | None:
    _validate_download_item_name(item)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    if item == "report.md":
        value = artifacts.get("report")
        return value if value == item else None
    if item in {"manifest.json", "config.json", "lineage.json", "summary.json"}:
        return item
    display = manifest.get("display")
    if isinstance(display, dict):
        downloads = display.get("downloads")
        if isinstance(downloads, dict):
            for option in downloads.get("items", []):
                if not isinstance(option, dict) or option.get("item") != item:
                    continue
                table_name = option.get("table")
                if not isinstance(table_name, str):
                    return None
                return _table_relative_path(artifacts, table_name)
    return None


def _table_relative_path(
    artifacts: dict[object, object],
    table_name: str,
) -> str | None:
    if table_name == "combinations":
        dimensions = artifacts.get("dimensions")
        value = dimensions.get("combinations") if isinstance(dimensions, dict) else None
        return value if isinstance(value, str) else None
    for group in ("metrics", "portfolio", "diagnostics", "input_quality", "summaries"):
        section = artifacts.get(group)
        if isinstance(section, dict):
            value = section.get(table_name)
            if isinstance(value, str):
                return value
    for period in ("year", "quarter", "month"):
        section = artifacts.get(f"breakdowns/{period}")
        if isinstance(section, dict):
            value = section.get(table_name)
            if isinstance(value, str):
                return value
    return None


def _validate_download_item_name(item: str) -> None:
    parsed = PurePosixPath(item)
    if (
        not item
        or "\\" in item
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ArtifactPackageError(f"下载项路径不安全：{item}")


def _validate_cache_path_segment(value: str, *, source: str) -> None:
    if (
        not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or PurePosixPath(value).is_absolute()
    ):
        raise ArtifactPackageError(f"{source} 路径不安全：{value}")


def _json_bytes(payload: dict[str, object]) -> bytes:
    """Encode a JSON-like payload as UTF-8 JSON bytes."""

    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def public_lineage_bytes(output_dir: Path) -> bytes:
    """Return API-safe lineage metadata without private/local input file fields."""

    return _public_metadata_bytes(output_dir, "lineage.json")


def public_config_bytes(output_dir: Path) -> bytes:
    """Return API-safe config metadata without private/local path fields."""

    return _public_metadata_bytes(output_dir, "config.json")


def _public_metadata_bytes(
    output_dir: Path,
    file_name: str,
    *,
    fallback: dict[str, object] | None = None,
) -> bytes:
    return _json_bytes(
        _public_metadata_payload(output_dir, file_name, fallback=fallback)
    )


def _public_metadata_payload(
    output_dir: Path,
    file_name: str,
    *,
    fallback: dict[str, object] | None = None,
) -> dict[str, object]:
    path = output_dir / file_name
    if not path.is_file() and fallback is not None:
        value = fallback
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactPackageError(f"{file_name} must contain an object")
    sanitized = sanitize_public_metadata(value)
    if not isinstance(sanitized, dict):
        raise ArtifactPackageError(f"{file_name} must contain an object")
    return sanitized


def _write_selected_zip(
    run: LoadedEvaluationSet,
    unique_items: list[str],
    zip_path: Path,
    dynamic_items: dict[str, _DynamicDownloadResolver] | None,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = zip_path.with_name(f"{zip_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for item in unique_items:
                _validate_download_item_name(item)
                source_path = _download_item_path(run, item)
                if source_path is not None:
                    archive.write(source_path, item)
                    continue
                resolved = _resolve_download_item(run, item, dynamic_items)
                archive.writestr(item, resolved.require_content())
        temporary_path.replace(zip_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _require_supported_display_manifest(
    manifest: dict[str, object],
) -> dict[str, object]:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ArtifactPackageError("manifest display missing")
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactPackageError("manifest display schema_version unsupported")
    return display


def _resolve_artifact_path(output_dir: Path, relative_path: str, item: str) -> Path:
    candidate = (output_dir / relative_path).resolve()
    resolved_output_dir = output_dir.resolve()
    if (
        resolved_output_dir not in candidate.parents
        and candidate != resolved_output_dir
    ):
        raise ArtifactPackageError(f"下载项路径越界：{item}")
    return candidate


def _cache_identity_segment(value: str) -> str:
    if len(value) <= 32:
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _media_type_for_item(item: str) -> str:
    if item.endswith(".csv"):
        return "text/csv; charset=utf-8"
    if item.endswith(".parquet"):
        return "application/vnd.apache.parquet"
    if item.endswith(".md"):
        return "text/markdown; charset=utf-8"
    if item.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


__all__ = [
    "ArtifactPackageError",
    "build_selected_artifact_zip",
    "require_manifest_display_download_item",
    "resolve_manifest_display_artifact_path",
]
