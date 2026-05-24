"""Factor template, pipeline snapshot, and pipeline run routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.pipeline_service import (
    PipelineRunRequestPayload,
    PipelineSnapshotRequestPayload,
    create_pipeline_run_payload,
    create_pipeline_snapshot_payload,
    list_factor_template_payloads,
    pipeline_checkpoint_payloads,
    pipeline_run_status_payload,
)

router = APIRouter()


@router.get("/api/factor-templates")
def factor_templates(market: str = "a_share") -> list[dict[str, object]]:
    """Return editable factor expression templates."""

    return list_factor_template_payloads(market)


@router.post("/api/pipeline-snapshots", status_code=status.HTTP_201_CREATED)
def create_pipeline_snapshot(
    payload: PipelineSnapshotRequestPayload,
    request: Request,
) -> dict[str, object]:
    """Validate factor settings and write a configuration-only snapshot."""

    context = api_context(request)
    return create_pipeline_snapshot_payload(
        payload,
        context.pipeline_runs,
        output_root=context.pipeline_output_root,
    )


@router.post("/api/pipeline-runs", status_code=status.HTTP_201_CREATED)
def create_pipeline_run(
    payload: PipelineRunRequestPayload,
    request: Request,
) -> dict[str, object]:
    """Execute the A-share pipeline from the requested start layer."""

    context = api_context(request)
    return create_pipeline_run_payload(
        payload,
        context.pipeline_runs,
        output_root=context.pipeline_output_root,
    )


@router.get("/api/pipeline-snapshots/{run_id}")
def pipeline_snapshot_status(run_id: str, request: Request) -> dict[str, object]:
    """Return in-memory pipeline snapshot status."""

    return pipeline_run_status_payload(
        run_id,
        api_context(request).pipeline_runs,
        kind="snapshot",
    )


@router.get("/api/pipeline-snapshots/{run_id}/checkpoints")
def pipeline_snapshot_checkpoints(
    run_id: str,
    request: Request,
) -> list[dict[str, object]]:
    """Return pipeline snapshot checkpoints."""

    context = api_context(request)
    return pipeline_checkpoint_payloads(
        run_id,
        context.pipeline_runs,
        kind="snapshot",
        output_root=context.pipeline_output_root,
    )


@router.get("/api/pipeline-runs/{run_id}")
def pipeline_run_status(run_id: str, request: Request) -> dict[str, object]:
    """Return in-memory pipeline run status."""

    return pipeline_run_status_payload(
        run_id,
        api_context(request).pipeline_runs,
        kind="run",
    )


@router.get("/api/pipeline-runs/{run_id}/checkpoints")
def pipeline_checkpoints(run_id: str, request: Request) -> list[dict[str, object]]:
    """Return pipeline run checkpoints."""

    context = api_context(request)
    return pipeline_checkpoint_payloads(
        run_id,
        context.pipeline_runs,
        kind="run",
        output_root=context.pipeline_output_root,
    )


__all__ = ["router"]
