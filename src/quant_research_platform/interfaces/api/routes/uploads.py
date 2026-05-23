"""上传辅助路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from quant_research_platform.interfaces.api.catalog import FACTOR_CSV_TEMPLATE
from quant_research_platform.interfaces.api.models import CreateRunPayload
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.upload_service import (
    create_uploaded_run,
)

router = APIRouter()


@router.post("/api/runs", status_code=status.HTTP_201_CREATED)
def create_run(payload: CreateRunPayload, request: Request) -> dict[str, object]:
    """基于用户上传 CSV 创建一个本地 evaluation run。"""

    return create_uploaded_run(api_context(request).run_roots, payload)


@router.get("/api/templates/factor-csv")
def factor_csv_template() -> Response:
    """下载用户上传面板的 CSV template。"""

    return Response(
        content=FACTOR_CSV_TEMPLATE,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="factor_panel_template.csv"'
        },
    )


__all__ = ["router"]
