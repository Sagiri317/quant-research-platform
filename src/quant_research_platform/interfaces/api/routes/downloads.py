"""报告和 ZIP 下载路由。"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from quant_research_platform.interfaces.api.models import DownloadArtifactsPayload
from quant_research_platform.interfaces.api.services import download_service
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.run_repository import (
    resolve_run_index_record,
)

router = APIRouter()
DOWNLOAD_ITEMS_QUERY = Query(...)


@router.get("/api/runs/{source_alias}/{run_id}/report.md")
def run_report_markdown(source_alias: str, run_id: str, request: Request) -> Response:
    """以 Markdown attachment 形式下载 run report。"""

    context = api_context(request)
    record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    return download_service.report_markdown_response_from_manifest(
        record.manifest,
        record.ref.path,
    )


@router.post("/api/runs/{source_alias}/{run_id}/download")
def selected_artifact_download(
    source_alias: str,
    run_id: str,
    payload: DownloadArtifactsPayload,
    request: Request,
) -> Response:
    """按前端选择项打包下载 ZIP。"""

    context = api_context(request)
    record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    return download_service.selected_download_response_from_manifest(
        record.manifest,
        record.ref.path,
        source_alias,
        run_id,
        payload.items,
        context.download_cache_root(source_alias),
        context.comparison_csv_bytes_for_download,
    )


@router.get("/api/runs/{source_alias}/{run_id}/download")
def selected_artifact_download_from_query(
    source_alias: str,
    run_id: str,
    request: Request,
    items: list[str] = DOWNLOAD_ITEMS_QUERY,
) -> Response:
    """按 query items 打包下载 ZIP，供浏览器原生下载流使用。"""

    context = api_context(request)
    record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    return download_service.selected_download_response_from_manifest(
        record.manifest,
        record.ref.path,
        source_alias,
        run_id,
        items,
        context.download_cache_root(source_alias),
        context.comparison_csv_bytes_for_download,
    )


__all__ = ["router"]
