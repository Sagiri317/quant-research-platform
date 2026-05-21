"""Download responses for reports, CSV artifacts, and ZIP bundles."""

from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, Response
from fastapi.responses import FileResponse

from quant_research_platform.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    LoadedEvaluationSet,
)


def direct_manifest_download_response(
    manifest: dict[str, object],
    output_dir: Path,
    artifact_item: str,
    comparison_csv: Callable[[], bytes] | None = None,
) -> Response:
    """Return a display-exposed item using only the lightweight manifest."""

    validate_download_item_name(artifact_item)
    ensure_download_item_exposed(manifest, artifact_item)
    if artifact_item == "comparison.csv":
        if comparison_csv is None:
            raise HTTPException(status_code=422, detail="comparison.csv 不可用。")
        return Response(
            content=comparison_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="comparison.csv"'
            },
        )
    relative_path = artifact_relative_path_from_manifest(manifest, artifact_item)
    if relative_path is None:
        raise HTTPException(status_code=422, detail=f"未知下载项：{artifact_item}")
    if artifact_item in {"report.md", "manifest.json", "config.json", "summary.json"}:
        relative_path = artifact_item
    path = _resolve_artifact_path(output_dir, relative_path, artifact_item)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(
        path,
        media_type=_media_type_for_item(artifact_item),
        filename=PurePosixPath(artifact_item).name,
    )


def report_markdown_response(run: LoadedEvaluationSet) -> Response:
    """返回 report.md 的 Markdown download response。"""

    if "report.md" not in display_download_items(run.manifest):
        raise HTTPException(status_code=404, detail="report not found")
    return Response(
        content=run.report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


def ensure_download_item_exposed(manifest: dict[str, object], item: str) -> None:
    """Require one item to be present in manifest.display.downloads.items."""

    validate_download_item_name(item)
    if item not in display_download_items(manifest):
        raise HTTPException(
            status_code=422,
            detail=f"{item} is not exposed by manifest display downloads",
        )


def selected_download_response(
    run: LoadedEvaluationSet,
    source_alias: str,
    run_id: str,
    items: list[str],
    cache_root: Path,
    comparison_csv: Callable[[], bytes],
) -> Response:
    """按前端选择项生成 ZIP download response。"""

    unique_items = list(dict.fromkeys(items))
    if not unique_items:
        raise HTTPException(status_code=422, detail="请选择至少一个下载项。")
    for item in unique_items:
        validate_download_item_name(item)
    allowed_items = display_download_items(run.manifest)
    for item in unique_items:
        if item not in allowed_items:
            raise HTTPException(status_code=422, detail=f"未知下载项：{item}")

    comparison_cache: bytes | None = None

    def comparison_bytes() -> bytes:
        nonlocal comparison_cache
        if comparison_cache is None:
            comparison_cache = comparison_csv()
        return comparison_cache

    zip_path = _selected_zip_cache_path(
        run,
        source_alias,
        run_id,
        unique_items,
        cache_root,
        comparison_bytes,
    )
    if not zip_path.is_file():
        _write_selected_zip(run, unique_items, comparison_bytes, zip_path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{source_alias}_{run_id}_selected_artifacts.zip",
    )


def _write_selected_zip(
    run: LoadedEvaluationSet,
    unique_items: list[str],
    comparison_csv: Callable[[], bytes],
    zip_path: Path,
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
                validate_download_item_name(item)
                source_path = download_item_path(run, item)
                if source_path is not None:
                    archive.write(source_path, item)
                    continue
                content, _, _ = download_item(run, item, comparison_csv)
                archive.writestr(item, content)
        temporary_path.replace(zip_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _selected_zip_cache_path(
    run: LoadedEvaluationSet,
    source_alias: str,
    run_id: str,
    unique_items: list[str],
    cache_root: Path,
    comparison_csv: Callable[[], bytes],
) -> Path:
    validate_cache_path_segment(source_alias, source="source_alias")
    validate_cache_path_segment(run_id, source="run_id")
    fingerprint: list[str] = []
    for item in unique_items:
        validate_download_item_name(item)
        source_path = download_item_path(run, item)
        if source_path is not None:
            stat = source_path.stat()
            fingerprint.append(
                f"{item}\0{source_path.as_posix()}\0{stat.st_size}\0"
                f"{stat.st_mtime_ns}"
            )
            continue
        if item == "comparison.csv":
            digest = hashlib.sha256(comparison_csv()).hexdigest()
            fingerprint.append(f"{item}\0dynamic\0{digest}")
            continue
        fingerprint.append(f"{item}\0dynamic")
    digest = hashlib.sha256("\n".join(fingerprint).encode("utf-8")).hexdigest()[:16]
    resolved_cache_root = cache_root.resolve()
    zip_path = (resolved_cache_root / source_alias / run_id / f"{digest}.zip").resolve()
    if resolved_cache_root not in zip_path.parents:
        raise HTTPException(status_code=422, detail="download cache path out of bounds")
    return zip_path


def download_item(
    run: LoadedEvaluationSet,
    item: str,
    comparison_csv: Callable[[], bytes] | None = None,
) -> tuple[bytes, str, str]:
    """解析一个 download item，返回文件 bytes、media type 和文件名。"""

    validate_download_item_name(item)
    if item == "comparison.csv":
        if comparison_csv is None:
            raise HTTPException(status_code=422, detail="comparison.csv 不可用。")
        return comparison_csv(), "text/csv", item
    source_path = download_item_path(run, item)
    if source_path is not None:
        return source_path.read_bytes(), _media_type_for_item(item), item
    if item == "report.md":
        return run.report.encode("utf-8"), "text/markdown; charset=utf-8", item
    if item == "manifest.json":
        return json_bytes(run.manifest), "application/json", item
    if item == "config.json":
        return json_bytes(run.config), "application/json", item
    if item == "summary.json":
        return json_bytes(run.summary), "application/json", item
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
                return frame.to_csv(index=False).encode("utf-8"), "text/csv", item
    raise HTTPException(status_code=422, detail=f"未知下载项：{item}")


def display_download_items(manifest: dict[str, object]) -> set[str]:
    """Return manifest.display download items after validating the display version."""

    display = _require_supported_display_manifest(manifest)
    downloads = display.get("downloads")
    if not isinstance(downloads, dict) or not isinstance(downloads.get("items"), list):
        raise HTTPException(
            status_code=422,
            detail="manifest display downloads invalid",
        )
    items: set[str] = set()
    for option in downloads["items"]:
        if not isinstance(option, dict):
            raise HTTPException(
                status_code=422,
                detail="manifest display downloads invalid",
            )
        item = option.get("item")
        if not isinstance(item, str) or not item:
            raise HTTPException(
                status_code=422,
                detail="manifest display download item invalid",
            )
        validate_download_item_name(item)
        items.add(item)
    return items


def _require_supported_display_manifest(
    manifest: dict[str, object],
) -> dict[str, object]:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise HTTPException(status_code=422, detail="manifest display missing")
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail="manifest display schema_version unsupported",
        )
    return display


def download_item_path(run: LoadedEvaluationSet, item: str) -> Path | None:
    """返回已存在 artifact 文件路径；动态内容返回 None。"""

    validate_download_item_name(item)
    artifacts = run.manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    path = artifact_relative_path(run, item, artifacts)
    if path is None:
        return None
    candidate = _resolve_artifact_path(run.output_dir, path, item)
    if not candidate.is_file():
        raise HTTPException(status_code=422, detail=f"下载项文件不存在：{item}")
    return candidate


def artifact_relative_path(
    run: LoadedEvaluationSet,
    item: str,
    artifacts: dict[object, object],
) -> str | None:
    validate_download_item_name(item)
    if item in {"report.md", "manifest.json", "config.json", "summary.json"}:
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
                return table_relative_path(artifacts, table_name)
    return None


def artifact_relative_path_from_manifest(
    manifest: dict[str, object],
    item: str,
) -> str | None:
    validate_download_item_name(item)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    if item in {"report.md", "manifest.json", "config.json", "summary.json"}:
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
                return table_relative_path(artifacts, table_name)
    return None


def _resolve_artifact_path(output_dir: Path, relative_path: str, item: str) -> Path:
    candidate = (output_dir / relative_path).resolve()
    resolved_output_dir = output_dir.resolve()
    if (
        resolved_output_dir not in candidate.parents
        and candidate != resolved_output_dir
    ):
        raise HTTPException(status_code=422, detail=f"下载项路径越界：{item}")
    return candidate


def table_relative_path(
    artifacts: dict[object, object],
    table_name: str,
) -> str | None:
    if table_name == "combinations":
        value = artifacts.get("combinations")
        return value if isinstance(value, str) else None
    for group in ("metrics", "portfolio", "diagnostics"):
        section = artifacts.get(group)
        if isinstance(section, dict):
            value = section.get(table_name)
            if isinstance(value, str):
                return value
    for period in ("year", "quarter", "month"):
        section = artifacts.get(f"metrics_by_{period}")
        if isinstance(section, dict):
            value = section.get(table_name)
            if isinstance(value, str):
                return value
    return None


def validate_download_item_name(item: str) -> None:
    parsed = PurePosixPath(item)
    if (
        not item
        or "\\" in item
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise HTTPException(status_code=422, detail=f"下载项路径不安全：{item}")


def validate_cache_path_segment(value: str, *, source: str) -> None:
    if (
        not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or PurePosixPath(value).is_absolute()
    ):
        raise HTTPException(status_code=422, detail=f"{source} 路径不安全：{value}")


def _media_type_for_item(item: str) -> str:
    if item.endswith(".csv"):
        return "text/csv; charset=utf-8"
    if item.endswith(".md"):
        return "text/markdown; charset=utf-8"
    if item.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def json_bytes(payload: dict[str, object]) -> bytes:
    """把 JSON-like payload 编码为 UTF-8 JSON bytes。"""

    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
