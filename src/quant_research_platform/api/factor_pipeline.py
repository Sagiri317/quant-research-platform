"""API helpers for factor templates and pipeline snapshot runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from quant_research_platform.markets.a_share.factors.expressions import (
    FactorExpressionError,
    parse_factor_expression,
)
from quant_research_platform.markets.a_share.pipeline.config import (
    DEFAULT_OUTPUT_ROOT,
    PipelineConfig,
    validate_pipeline_config,
)
from quant_research_platform.markets.a_share.pipeline.snapshots import (
    PipelineSnapshot,
    PipelineSnapshotError,
    write_run_snapshot,
)

TEMPLATE_ROOT = Path("config/a_share/factor_templates")


class FactorTemplatePayload(BaseModel):
    """Factor expression template exposed to the web UI."""

    id: str
    name: str
    market: str = "a_share"
    expression: str
    description: str | None = None
    dependencies: list[str] = Field(default_factory=list)


class PipelineRunRequestPayload(BaseModel):
    """Frontend request for a controlled local pipeline snapshot."""

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


class PipelineRunSummaryPayload(BaseModel):
    """Small status payload for a local pipeline snapshot."""

    run_id: str
    status: str
    message: str | None = None
    snapshot_path: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class PipelineCheckpointPayload(BaseModel):
    """One checkpoint item displayed by the frontend."""

    key: str
    label: str
    status: str
    path: str | None = None
    message: str | None = None


def list_factor_template_payloads(market: str = "a_share") -> list[dict[str, Any]]:
    """Return configured factor templates for one market."""

    if market != "a_share":
        return []
    templates: list[FactorTemplatePayload] = []
    for path in sorted(TEMPLATE_ROOT.glob("*.yaml")):
        values = _read_simple_yaml_mapping(path)
        if values.get("market", "a_share") != market:
            continue
        expression = str(values.get("expression", "")).strip()
        dependencies: list[str] = []
        if expression:
            try:
                dependencies = sorted(parse_factor_expression(expression).dependencies)
            except FactorExpressionError:
                dependencies = []
        templates.append(
            FactorTemplatePayload(
                id=str(values.get("id", path.stem)),
                name=str(values.get("name", path.stem)),
                market=market,
                expression=expression,
                description=(
                    str(values["description"])
                    if values.get("description") is not None
                    else None
                ),
                dependencies=dependencies,
            )
        )
    return [template.model_dump() for template in templates]


def create_pipeline_snapshot_payload(
    payload: PipelineRunRequestPayload,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate a frontend pipeline request and write a small run snapshot."""

    config = _pipeline_config_from_payload(payload, output_root=output_root)
    validation = validate_pipeline_config(config)
    if validation.has_blocking:
        detail = {
            "blocking": [asdict(issue) for issue in validation.blocking],
            "warnings": [asdict(issue) for issue in validation.warnings],
        }
        raise HTTPException(status_code=422, detail=detail)
    try:
        snapshot = write_run_snapshot(config)
    except PipelineSnapshotError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = _summary_from_snapshot(config, snapshot)
    pipeline_runs[payload.run_id] = summary
    return summary.model_dump()


def pipeline_run_status_payload(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
) -> dict[str, Any]:
    """Return the current in-process status for a pipeline snapshot."""

    summary = pipeline_runs.get(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    return summary.model_dump()


def pipeline_checkpoint_payloads(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
) -> list[dict[str, Any]]:
    """Return expected snapshot checkpoints for a pipeline run."""

    summary = pipeline_runs.get(run_id)
    if summary is None or summary.snapshot_path is None:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    run_dir = Path(summary.snapshot_path)
    checkpoints = [
        ("run_config", "run_config.yaml", "run_config.yaml"),
        ("warnings", "warnings.json", "warnings.json"),
        ("summary", "summary.md", "summary.md"),
        ("evaluation_manifest", "评估清单", "manifest.json"),
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
                path=path.as_posix(),
                message=None if ready else "快照阶段尚未生成",
            )
        )
    return [payload.model_dump() for payload in payloads]


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
        output_root=output_root,
    )


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
        snapshot_path=snapshot.run_dir.as_posix(),
        warnings=warnings,
    )


def _read_simple_yaml_mapping(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
