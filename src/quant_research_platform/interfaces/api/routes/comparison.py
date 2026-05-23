"""多 run comparison 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from quant_research_platform.interfaces.api.payloads import table_payload
from quant_research_platform.interfaces.api.services.comparison_service import (
    comparison_index_frame,
)
from quant_research_platform.interfaces.api.services.context import api_context

router = APIRouter()


@router.get("/api/comparison")
def comparison(request: Request) -> dict[str, object]:
    """返回所有 run 的 comparison table。"""

    records = api_context(request).comparison_index_records()
    if not records:
        return {"columns": [], "rows": []}
    return table_payload(comparison_index_frame(records))


@router.get("/api/comparison.csv")
def comparison_csv(request: Request) -> Response:
    """以 CSV attachment 形式下载 comparison table。"""

    return Response(
        content=api_context(request).comparison_csv_bytes_for_download(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="comparison.csv"'},
    )


__all__ = ["router"]
