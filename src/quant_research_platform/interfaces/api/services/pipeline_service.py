"""本地 factor templates 和 pipeline run 的 API helper。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from quant_research_platform.pipelines.a_share.config import (
    DEFAULT_OUTPUT_ROOT,
    PipelineConfig,
    validate_pipeline_config,
)
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    run_processed_existing_pipeline,
)
from quant_research_platform.pipelines.a_share.snapshots import (
    PipelineSnapshot,
    PipelineSnapshotError,
    write_run_snapshot,
)
from quant_research_platform.research.factor_factory.factor_registry import (
    FactorTemplateRegistry,
)

TEMPLATE_ROOT = Path("config/a_share/factor_templates")


class FactorTemplatePayload(BaseModel):
    """暴露给 web UI 的 factor expression template。"""

    id: str
    name: str
    version: str
    market: str = "a_share"
    expression: str
    description: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    grid_parameter: str | None = None
    grid_values: list[str] = Field(default_factory=list)


class PipelineRunRequestPayload(BaseModel):
    """前端提交的受控本地 pipeline 请求。"""

    model_config = ConfigDict(extra="forbid")

    market: str = "a_share"
    run_id: str
    date_start: str
    date_end: str
    factor_expression: str
    factor_template_id: str | None = None
    portfolio_mode: str = "long_only"
    cost_bps: float = 0.0
    pnl_return_column: str = "forward_return_1d"
    label_columns: list[str] = Field(default_factory=lambda: ["forward_return_1d"])
    processed_factor_column: str | None = None
    engine: str = "dataframe"
    execution_mode: str = "snapshot"


class PipelineRunSummaryPayload(BaseModel):
    """本地 pipeline run 的轻量状态 payload。"""

    run_id: str
    status: str
    message: str | None = None
    run_ref: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class PipelineCheckpointPayload(BaseModel):
    """前端展示的一条 checkpoint。"""

    key: str
    label: str
    status: str
    artifact: str | None = None
    message: str | None = None
    duration_seconds: float | None = None
    row_count: int | None = None
    failure_code: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    blocking_issues: list[dict[str, Any]] = Field(default_factory=list)


def list_factor_template_payloads(market: str = "a_share") -> list[dict[str, Any]]:
    """返回指定 market 的受控 factor templates。"""

    if market != "a_share":
        return []
    registry = FactorTemplateRegistry.from_directory(TEMPLATE_ROOT, market=market)
    payloads: list[FactorTemplatePayload] = []
    for template in registry.templates():
        payloads.append(
            FactorTemplatePayload(
                id=template.template_id,
                name=template.name,
                version=template.version,
                market=template.market,
                expression=template.expression,
                description=template.description,
                dependencies=list(template.dependencies),
                inputs=list(template.inputs),
                outputs=list(template.outputs),
                capabilities=list(template.capabilities),
                grid_parameter=template.grid_parameter,
                grid_values=list(template.grid_values),
            )
        )
    return [payload.model_dump() for payload in payloads]


def create_pipeline_snapshot_payload(
    payload: PipelineRunRequestPayload,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """校验前端 pipeline 请求，并写出 snapshot 或执行轻量 runner。"""

    config = _pipeline_config_from_payload(payload, output_root=output_root)
    allow_processed_expression = payload.execution_mode == "processed_existing"
    validation = validate_pipeline_config(
        config,
        allow_processed_factor_expression=allow_processed_expression,
    )
    if validation.has_blocking:
        detail = {
            "blocking": [asdict(issue) for issue in validation.blocking],
            "warnings": [asdict(issue) for issue in validation.warnings],
        }
        raise HTTPException(status_code=422, detail=detail)
    if payload.execution_mode == "snapshot":
        try:
            snapshot = write_run_snapshot(config)
        except PipelineSnapshotError as exc:
            raise HTTPException(
                status_code=422,
                detail="pipeline snapshot could not be written",
            ) from exc
        summary = _summary_from_snapshot(config, snapshot)
    elif payload.execution_mode == "processed_existing":
        try:
            result = run_processed_existing_pipeline(config)
        except PipelineRunnerError as exc:
            run_dir = getattr(exc, "run_dir", None)
            if run_dir is None:
                raise HTTPException(
                    status_code=422,
                    detail="pipeline execution failed",
                ) from exc
            summary = PipelineRunSummaryPayload(
                run_id=config.run_id,
                status="failed",
                message=_pipeline_failure_message(exc),
                run_ref=run_dir.name,
                warnings=[],
            )
            _remember_pipeline_summary(pipeline_runs, payload.run_id, summary)
            return summary.model_dump()
        run_ref = result.run_dir.name
        summary = PipelineRunSummaryPayload(
            run_id=config.run_id,
            status=result.status,
            message="processed-existing pipeline 已完成",
            run_ref=run_ref,
            warnings=result.warnings,
        )
    else:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported execution_mode: {payload.execution_mode}",
        )
    _remember_pipeline_summary(pipeline_runs, payload.run_id, summary)
    return summary.model_dump()


def pipeline_run_status_payload(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
) -> dict[str, Any]:
    """返回当前进程内保存的 pipeline run 状态。"""

    summary = pipeline_runs.get(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    return summary.model_dump()


def pipeline_checkpoint_payloads(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[dict[str, Any]]:
    """返回 pipeline run 的 snapshot 或 runner checkpoints。"""

    summary = pipeline_runs.get(run_id)
    if summary is None:
        run_dir = _pipeline_run_dir(output_root, run_id)
        checkpoints_path = run_dir / "checkpoints.json"
        if checkpoints_path.is_file():
            return _checkpoint_payloads_from_json(checkpoints_path)
        raise HTTPException(status_code=404, detail="pipeline run not found")
    if summary.run_ref is None:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    run_dir = _pipeline_run_dir(output_root, summary.run_ref)
    checkpoints_path = run_dir / "checkpoints.json"
    if checkpoints_path.is_file():
        return _checkpoint_payloads_from_json(checkpoints_path)
    checkpoints = [
        ("run_config", "run_config.yaml", "run_config.yaml"),
        ("warnings", "warnings.json", "warnings.json"),
        ("summary", "summary.md", "summary.md"),
        ("evaluation_manifest", "评估清单", "evaluation/evaluation_manifest.json"),
    ]
    payloads: list[PipelineCheckpointPayload] = []
    for key, label, relative_path in checkpoints:
        path = run_dir / relative_path
        ready = path.is_file()
        payloads.append(
            PipelineCheckpointPayload(
                key=key,
                label=label,
                status="ready" if ready else "pending",
                artifact=relative_path,
                message=None if ready else "快照阶段尚未生成",
            )
        )
    return [payload.model_dump() for payload in payloads]


def _pipeline_failure_message(exc: PipelineRunnerError) -> str:
    failure_code = getattr(exc, "failure_code", None)
    if isinstance(failure_code, str) and failure_code:
        return f"processed-existing pipeline 执行失败：{failure_code}"
    return "processed-existing pipeline 执行失败"


def _pipeline_run_dir(output_root: Path, run_ref: str) -> Path:
    path = Path(run_ref)
    if path.is_absolute() or path.name != run_ref or run_ref in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    return output_root / run_ref


def _checkpoint_payloads_from_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise HTTPException(status_code=422, detail="pipeline checkpoints invalid")
    payloads: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="pipeline checkpoints invalid")
        artifact = item.get("artifact")
        if artifact is None and isinstance(item.get("path"), str):
            artifact = Path(item["path"]).name
        payload = {
            "key": item.get("key"),
            "label": item.get("label"),
            "status": item.get("status"),
            "artifact": artifact,
            "message": item.get("message"),
            "duration_seconds": item.get("duration_seconds"),
            "row_count": item.get("row_count"),
            "failure_code": item.get("failure_code"),
            "warnings": item.get("warnings", []),
            "blocking_issues": item.get("blocking_issues", []),
        }
        try:
            payloads.append(PipelineCheckpointPayload(**payload).model_dump())
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="pipeline checkpoints invalid",
            ) from exc
    return payloads


def _pipeline_config_from_payload(
    payload: PipelineRunRequestPayload,
    *,
    output_root: Path,
) -> PipelineConfig:
    return PipelineConfig(
        market=payload.market,
        run_id=payload.run_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        factor_expression=payload.factor_expression,
        factor_template_id=payload.factor_template_id or "custom",
        factor_template_source=(
            f"config/a_share/factor_templates/{payload.factor_template_id}.yaml"
            if payload.factor_template_id
            else None
        ),
        portfolio_mode=payload.portfolio_mode,
        cost_bps=payload.cost_bps,
        pnl_return_column=payload.pnl_return_column,
        label_columns=tuple(payload.label_columns),
        processed_factor_column=payload.processed_factor_column,
        engine=payload.engine,
        output_root=output_root,
    )


def _remember_pipeline_summary(
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    requested_run_id: str,
    summary: PipelineRunSummaryPayload,
) -> None:
    """同时按请求 run_id 和实际 run_ref 保存进程内状态。"""

    pipeline_runs[requested_run_id] = summary
    if summary.run_ref:
        pipeline_runs[summary.run_ref] = summary


def _summary_from_snapshot(
    config: PipelineConfig,
    snapshot: PipelineSnapshot,
) -> PipelineRunSummaryPayload:
    warnings = [asdict(issue) for issue in snapshot.warnings]
    message = "快照已生成"
    if warnings:
        message = f"快照已生成，包含 {len(warnings)} 条警告"
    return PipelineRunSummaryPayload(
        run_id=config.run_id,
        status="ready",
        message=message,
        run_ref=snapshot.run_dir.name,
        warnings=warnings,
    )
