"""健康检查和本地 run root 配置路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from quant_research_platform.interfaces.api.services.context import api_context

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """返回平台健康状态和当前配置的 artifact run roots。"""

    context = api_context(request)
    return {
        "status": "ok",
        "run_roots": [
            {"source_alias": root.source_alias}
            for root in context.run_roots
        ],
    }


__all__ = ["router"]
