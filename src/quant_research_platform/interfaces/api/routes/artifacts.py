"""单个 artifact table 直接下载路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from quant_research_platform.interfaces.api.services import download_service
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.run_repository import (
    resolve_run_index_record,
)

router = APIRouter()


@router.get("/api/runs/{source_alias}/{run_id}/artifacts/{artifact_item:path}")
def artifact_csv(
    source_alias: str,
    run_id: str,
    artifact_item: str,
    request: Request,
) -> Response:
    """以 CSV attachment 形式下载单个 artifact table。"""

    context = api_context(request)
    record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    return download_service.direct_manifest_download_response(
        record.manifest,
        record.ref.path,
        artifact_item,
        comparison_csv=context.comparison_csv_bytes_for_download,
    )


__all__ = ["router"]
