"""因子 template 和 pipeline snapshot 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.pipeline_service import (
    PipelineRunRequestPayload,
    create_pipeline_snapshot_payload,
    list_factor_template_payloads,
    pipeline_checkpoint_payloads,
    pipeline_run_status_payload,
)

router = APIRouter()


@router.get("/api/factor-templates")
def factor_templates(market: str = "a_share") -> list[dict[str, object]]:
    """返回可编辑的 factor expression templates。"""

    return list_factor_template_payloads(market)


@router.post("/api/pipeline-runs", status_code=status.HTTP_201_CREATED)
def create_pipeline_run(
    payload: PipelineRunRequestPayload,
    request: Request,
) -> dict[str, object]:
    """校验 factor 设置并写入本地 pipeline snapshot。"""

    context = api_context(request)
    return create_pipeline_snapshot_payload(
        payload,
        context.pipeline_runs,
        output_root=context.pipeline_output_root,
    )


@router.get("/api/pipeline-runs/{run_id}")
def pipeline_run_status(run_id: str, request: Request) -> dict[str, object]:
    """返回内存中的 pipeline snapshot 状态。"""

    return pipeline_run_status_payload(run_id, api_context(request).pipeline_runs)


@router.get("/api/pipeline-runs/{run_id}/checkpoints")
def pipeline_checkpoints(run_id: str, request: Request) -> list[dict[str, object]]:
    """返回 pipeline run 的 snapshot 阶段 checkpoints。"""

    context = api_context(request)
    return pipeline_checkpoint_payloads(
        run_id,
        context.pipeline_runs,
        output_root=context.pipeline_output_root,
    )


__all__ = ["router"]
