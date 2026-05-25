"""Factor Factory run 浏览路由。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from quant_research_platform.interfaces.api.payloads import table_payload
from quant_research_platform.interfaces.api.run_index import (
    validate_manifest_relative_path,
)
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.research.factor_factory.reports import (
    FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION,
)

router = APIRouter()

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
PRIVATE_TEXT_MARKERS = (
    "reports/private_real",
    "data/raw_private",
    "data/interim_private",
    "data/processed_private",
    "raw_private",
    "interim_private",
    "processed_private",
)
LOCAL_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:/|//)[^,\s;)\]]+",
)


@router.get("/api/factor-factory-runs")
def list_factor_factory_runs(request: Request) -> list[dict[str, object]]:
    """列出本地 Factor Factory 产物目录。"""

    root = api_context(request).factor_factory_root
    if not root.is_dir():
        return []
    runs: list[dict[str, object]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            run_dir = _resolve_run_dir(root, child.name)
            manifest = _load_manifest(run_dir / "manifest.json")
        except HTTPException:
            continue
        runs.append(_summary_payload(child.name, manifest))
    return runs


@router.get("/api/factor-factory-runs/{run_id}")
def factor_factory_run_detail(run_id: str, request: Request) -> dict[str, object]:
    """返回一个 Factor Factory run 的 manifest/display 摘要。"""

    run_dir = _resolve_run_dir(api_context(request).factor_factory_root, run_id)
    manifest = _load_manifest(run_dir / "manifest.json")
    return {
        "run": _summary_payload(run_id, manifest),
        "display": _manifest_display(manifest),
    }


@router.get("/api/factor-factory-runs/{run_id}/tables/{table_key}")
def factor_factory_table(
    run_id: str,
    table_key: str,
    request: Request,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, object]:
    """按 manifest.display 暴露 Factor Factory 表格预览。"""

    run_dir = _resolve_run_dir(api_context(request).factor_factory_root, run_id)
    manifest = _load_manifest(run_dir / "manifest.json")
    table = _display_table(manifest, table_key)
    path = _resolve_artifact_path(run_dir, str(table["path"]))
    return _csv_table_payload(path, limit=limit, offset=offset)


@router.get("/api/factor-factory-runs/{run_id}/artifacts/{artifact_item:path}")
def factor_factory_artifact(
    run_id: str,
    artifact_item: str,
    request: Request,
) -> FileResponse:
    """下载 manifest.display.downloads 允许的 Factor Factory 产物。"""

    run_dir = _resolve_run_dir(api_context(request).factor_factory_root, run_id)
    manifest = _load_manifest(run_dir / "manifest.json")
    if artifact_item not in _download_items(manifest):
        raise HTTPException(status_code=422, detail="download item is not exposed")
    path = _resolve_artifact_path(run_dir, artifact_item)
    _validate_public_artifact_content(path)
    return FileResponse(path, filename=Path(artifact_item).name)


def _resolve_run_dir(root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=422, detail="unsafe factor factory run_id")
    root_path = root.resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root_path:
        raise HTTPException(
            status_code=403,
            detail="factor factory run path not allowed",
        )
    if not (run_dir / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="factor factory run not found")
    return run_dir


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="factor factory manifest invalid",
        ) from exc
    if manifest.get("artifact_type") != "factor_factory_run":
        raise HTTPException(
            status_code=422,
            detail="factor factory artifact_type invalid",
        )
    if (
        manifest.get("artifact_schema_version")
        != FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION
    ):
        raise HTTPException(status_code=422, detail="factor factory schema unsupported")
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise HTTPException(status_code=422, detail="factor factory display invalid")
    if display.get("schema_version") != FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail="factor factory display unsupported",
        )
    _validate_private_display_text(display)
    _validate_display_paths(display)
    return manifest


def _summary_payload(run_id: str, manifest: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": run_id,
        "artifact_type": manifest["artifact_type"],
        "artifact_schema_version": manifest["artifact_schema_version"],
        "candidate_count": int(manifest.get("candidate_count", 0)),
        "success_count": int(manifest.get("success_count", 0)),
        "failed_count": int(manifest.get("failed_count", 0)),
    }


def _display_table(
    manifest: dict[str, object],
    table_key: str,
) -> dict[str, object]:
    display = _manifest_display(manifest)
    tables = display.get("tables")
    if not isinstance(tables, list):
        raise HTTPException(status_code=422, detail="factor factory tables invalid")
    for table in tables:
        if isinstance(table, dict) and table.get("key") == table_key:
            return table
    raise HTTPException(status_code=404, detail="factor factory table not found")


def _download_items(manifest: dict[str, object]) -> set[str]:
    display = _manifest_display(manifest)
    downloads = display.get("downloads")
    items = downloads.get("items") if isinstance(downloads, dict) else None
    if not isinstance(items, list):
        raise HTTPException(status_code=422, detail="factor factory downloads invalid")
    result: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("item"), str):
            raise HTTPException(
                status_code=422,
                detail="factor factory download invalid",
            )
        result.add(item["item"])
    return result


def _manifest_display(manifest: dict[str, object]) -> dict[str, object]:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise HTTPException(status_code=422, detail="factor factory display invalid")
    return display


def _validate_display_paths(display: dict[str, object]) -> None:
    tables = display.get("tables")
    if not isinstance(tables, list):
        raise HTTPException(status_code=422, detail="factor factory tables invalid")
    for table in tables:
        if not isinstance(table, dict):
            raise HTTPException(status_code=422, detail="factor factory table invalid")
        if not isinstance(table.get("key"), str) or not isinstance(
            table.get("path"),
            str,
        ):
            raise HTTPException(status_code=422, detail="factor factory table invalid")
        _validate_manifest_path(str(table["path"]))

    downloads = display.get("downloads")
    if not isinstance(downloads, dict):
        raise HTTPException(status_code=422, detail="factor factory downloads invalid")
    items = downloads.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=422, detail="factor factory downloads invalid")
    exposed_items: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("item"), str):
            raise HTTPException(
                status_code=422,
                detail="factor factory download invalid",
            )
        _validate_manifest_path(str(item["item"]))
        exposed_items.add(str(item["item"]))

    presets = downloads.get("presets", [])
    if not isinstance(presets, list):
        raise HTTPException(status_code=422, detail="factor factory preset invalid")
    for preset in presets:
        if not isinstance(preset, dict):
            raise HTTPException(status_code=422, detail="factor factory preset invalid")
        preset_items = preset.get("items")
        if not isinstance(preset_items, list) or not all(
            isinstance(item, str) for item in preset_items
        ):
            raise HTTPException(status_code=422, detail="factor factory preset invalid")
        for item in preset_items:
            _validate_manifest_path(item)
            if item not in exposed_items:
                raise HTTPException(
                    status_code=422,
                    detail="factor factory preset references unknown item",
                )


def _validate_manifest_path(relative_path: str) -> None:
    try:
        validate_manifest_relative_path(relative_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="unsafe factor factory artifact path",
        ) from exc


def _validate_private_display_text(value: object) -> None:
    if isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if any(marker in normalized for marker in PRIVATE_TEXT_MARKERS):
            raise HTTPException(
                status_code=422,
                detail="factor factory display contains private path text",
            )
        if LOCAL_ABSOLUTE_PATH_PATTERN.search(normalized):
            raise HTTPException(
                status_code=422,
                detail="factor factory display contains local path text",
            )
        return
    if isinstance(value, dict):
        for child in value.values():
            _validate_private_display_text(child)
        return
    if isinstance(value, list):
        for child in value:
            _validate_private_display_text(child)


def _validate_public_artifact_content(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                _validate_private_display_text(line)
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="factor factory artifact must be UTF-8 text",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail="factor factory artifact not found",
        ) from exc


def _csv_table_payload(path: Path, *, limit: int, offset: int) -> dict[str, object]:
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be nonnegative")
    if limit < 1:
        raise HTTPException(status_code=422, detail="limit must be positive")
    if limit > 1000:
        raise HTTPException(status_code=422, detail="limit must be at most 1000")
    try:
        read_frame = pd.read_csv(
            path,
            skiprows=lambda row_index: 0 < row_index <= offset,
            nrows=limit + 1,
        )
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="factor factory artifact must be UTF-8 text",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail="factor factory artifact not found",
        ) from exc
    has_more = len(read_frame) > limit
    frame = read_frame.iloc[:limit].copy()
    _validate_public_table_frame(frame)
    payload = table_payload(frame)
    payload["offset"] = offset
    payload["limit"] = limit
    payload["has_more"] = has_more
    if not has_more:
        payload["total_rows"] = offset + len(frame)
    return payload


def _validate_public_table_frame(frame: pd.DataFrame) -> None:
    _validate_private_display_text(list(frame.columns))
    for row in frame.to_dict(orient="records"):
        _validate_private_display_text(row)


def _resolve_artifact_path(run_dir: Path, relative_path: str) -> Path:
    _validate_manifest_path(relative_path)
    path = (run_dir / relative_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="unsafe factor factory artifact path",
        ) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="factor factory artifact not found")
    return path


__all__ = ["router"]
