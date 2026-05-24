"""API helpers for factor templates, pipeline snapshots, and pipeline runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from quant_research_platform.pipelines.a_share.config import (
    DEFAULT_OUTPUT_ROOT,
    PipelineConfig,
    PipelineStartFrom,
    validate_pipeline_config,
)
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    run_a_share_pipeline,
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
FACTOR_TEMPLATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
PipelineResourceKind = Literal["snapshot", "run"]
PRIVATE_PATH_MARKERS = (
    "raw_private",
    "interim_private",
    "processed_private",
    "private_real",
)


class FactorTemplatePayload(BaseModel):
    """Factor expression template exposed to the local web UI."""

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


class PipelineSnapshotRequestPayload(BaseModel):
    """Configuration-only A-share pipeline snapshot request."""

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


class PipelineRunRequestPayload(PipelineSnapshotRequestPayload):
    """Executable A-share pipeline request.

    ``start_from`` is the public contract. Every run continues through evaluation.
    """

    start_from: PipelineStartFrom


class PipelineRunSummaryPayload(BaseModel):
    """Lightweight pipeline snapshot/run status payload."""

    kind: PipelineResourceKind
    run_id: str
    status: str
    message: str | None = None
    run_ref: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class PipelineCheckpointPayload(BaseModel):
    """One checkpoint row exposed to the frontend."""

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
    """Return controlled factor templates for ``market``."""

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
    payload: PipelineSnapshotRequestPayload,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate and persist a configuration-only pipeline snapshot."""

    config = _pipeline_config_from_payload(payload, output_root=output_root)
    validation = validate_pipeline_config(config)
    if validation.has_blocking:
        _raise_validation_error(validation)
    try:
        snapshot = write_run_snapshot(config)
    except PipelineSnapshotError as exc:
        raise HTTPException(
            status_code=422,
            detail="pipeline snapshot could not be written",
        ) from exc
    summary = _summary_from_snapshot(config, snapshot)
    _remember_pipeline_summary(pipeline_runs, payload.run_id, summary)
    return summary.model_dump()


def create_pipeline_run_payload(
    payload: PipelineRunRequestPayload,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate and execute an A-share pipeline run."""

    config = _pipeline_config_from_payload(payload, output_root=output_root)
    validation = validate_pipeline_config(
        config,
        allow_processed_factor_expression=True,
    )
    if validation.has_blocking:
        _raise_validation_error(validation)
    try:
        result = run_a_share_pipeline(config)
    except PipelineRunnerError as exc:
        run_dir = getattr(exc, "run_dir", None)
        if run_dir is None:
            raise HTTPException(
                status_code=422,
                detail=_pipeline_failure_message(exc),
            ) from exc
        summary = PipelineRunSummaryPayload(
            kind="run",
            run_id=config.run_id,
            status="failed",
            message=_pipeline_failure_message(exc),
            run_ref=run_dir.name,
            warnings=[],
        )
        _remember_pipeline_summary(pipeline_runs, payload.run_id, summary)
        return summary.model_dump()
    summary = PipelineRunSummaryPayload(
        kind="run",
        run_id=config.run_id,
        status=result.status,
        message=f"pipeline run completed from {config.start_from}",
        run_ref=result.run_dir.name,
        warnings=result.warnings,
    )
    _remember_pipeline_summary(pipeline_runs, payload.run_id, summary)
    return summary.model_dump()


def pipeline_run_status_payload(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    kind: PipelineResourceKind,
) -> dict[str, Any]:
    """Return in-memory pipeline run or snapshot status for the requested kind."""

    summary = _pipeline_summary_for_kind(run_id, pipeline_runs, kind)
    return summary.model_dump()


def pipeline_checkpoint_payloads(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    *,
    kind: PipelineResourceKind,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[dict[str, Any]]:
    """Return checkpoints for the requested snapshot or executable run kind."""

    summary = pipeline_runs.get(_pipeline_summary_key(kind, run_id))
    if summary is None:
        if kind != "run":
            raise HTTPException(status_code=404, detail=_not_found_detail(kind))
        run_dir = _pipeline_run_dir(output_root, run_id)
        checkpoints_path = run_dir / "checkpoints.json"
        if checkpoints_path.is_file():
            return _checkpoint_payloads_from_json(checkpoints_path)
        raise HTTPException(status_code=404, detail=_not_found_detail(kind))
    if summary.kind != kind:
        raise HTTPException(status_code=404, detail=_not_found_detail(kind))
    if summary.run_ref is None:
        raise HTTPException(status_code=404, detail=_not_found_detail(kind))
    run_dir = _pipeline_run_dir(output_root, summary.run_ref)
    checkpoints_path = run_dir / "checkpoints.json"
    if kind == "run" and checkpoints_path.is_file():
        return _checkpoint_payloads_from_json(checkpoints_path)
    if kind == "run":
        raise HTTPException(status_code=404, detail=_not_found_detail(kind))
    checkpoints = [
        ("run_config", "run_config.yaml", "run_config.yaml"),
        ("warnings", "warnings.json", "warnings.json"),
        ("summary", "summary.md", "summary.md"),
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
                message=None if ready else "snapshot artifact is not generated yet",
            )
        )
    return [payload.model_dump() for payload in payloads]


def _pipeline_summary_for_kind(
    run_id: str,
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    kind: PipelineResourceKind,
) -> PipelineRunSummaryPayload:
    summary = pipeline_runs.get(_pipeline_summary_key(kind, run_id))
    if summary is None or summary.kind != kind:
        raise HTTPException(status_code=404, detail=_not_found_detail(kind))
    return summary


def _not_found_detail(kind: PipelineResourceKind) -> str:
    return f"pipeline {kind} not found"


def _raise_validation_error(validation) -> None:
    detail = {
        "blocking": [asdict(issue) for issue in validation.blocking],
        "warnings": [asdict(issue) for issue in validation.warnings],
    }
    raise HTTPException(status_code=422, detail=detail)


def _pipeline_failure_message(exc: PipelineRunnerError) -> str:
    failure_code = getattr(exc, "failure_code", None)
    if isinstance(failure_code, str) and failure_code:
        return f"pipeline execution failed: {failure_code}"
    return "pipeline execution failed"


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
        artifact = _safe_checkpoint_artifact(item.get("artifact"))
        if artifact is None and isinstance(item.get("path"), str):
            path_value = item["path"]
            artifact = (
                None
                if _has_private_path_marker(path_value)
                else _safe_checkpoint_artifact(Path(path_value).name)
            )
        payload = {
            "key": item.get("key"),
            "label": item.get("label"),
            "status": item.get("status"),
            "artifact": artifact,
            "message": _sanitize_checkpoint_string(item.get("message")),
            "duration_seconds": item.get("duration_seconds"),
            "row_count": item.get("row_count"),
            "failure_code": _sanitize_checkpoint_string(item.get("failure_code")),
            "warnings": _sanitize_checkpoint_records(item.get("warnings", [])),
            "blocking_issues": _sanitize_checkpoint_records(
                item.get("blocking_issues", [])
            ),
        }
        try:
            payloads.append(PipelineCheckpointPayload(**payload).model_dump())
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="pipeline checkpoints invalid",
            ) from exc
    return payloads


def _safe_checkpoint_artifact(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if _has_private_path_marker(value):
        return None
    path = Path(value)
    if path.is_absolute() or path.drive:
        return None
    return value


def _sanitize_checkpoint_string(value: object) -> str | None:
    if not isinstance(value, str) or _has_private_path_marker(value):
        return None
    return value


def _sanitize_checkpoint_records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    records: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            records.append(_sanitize_checkpoint_record(item))
    return records


def _sanitize_checkpoint_record(value: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        sanitized_value = _sanitize_checkpoint_value(item)
        if sanitized_value is not None:
            sanitized[key] = sanitized_value
    return sanitized


def _sanitize_checkpoint_value(value: object) -> object | None:
    if isinstance(value, str):
        return None if _has_private_path_marker(value) else value
    if isinstance(value, dict):
        return _sanitize_checkpoint_record(value)
    if isinstance(value, list):
        return [
            sanitized_item
            for item in value
            if (sanitized_item := _sanitize_checkpoint_value(item)) is not None
        ]
    return value


def _has_private_path_marker(value: str) -> bool:
    lowered = value.replace("\\", "/").lower()
    return any(marker in lowered for marker in PRIVATE_PATH_MARKERS)


def _pipeline_config_from_payload(
    payload: PipelineSnapshotRequestPayload,
    *,
    output_root: Path,
) -> PipelineConfig:
    factor_template_id, factor_template_source = _factor_template_metadata(
        payload.factor_template_id,
        market=payload.market,
    )
    return PipelineConfig(
        market=payload.market,
        run_id=payload.run_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        factor_expression=payload.factor_expression,
        factor_template_id=factor_template_id,
        factor_template_source=factor_template_source,
        portfolio_mode=payload.portfolio_mode,
        cost_bps=payload.cost_bps,
        pnl_return_column=payload.pnl_return_column,
        label_columns=tuple(payload.label_columns),
        processed_factor_column=payload.processed_factor_column,
        start_from=getattr(payload, "start_from", "existing_factors"),
        engine=payload.engine,
        output_root=output_root,
    )


def _factor_template_metadata(
    factor_template_id: str | None,
    *,
    market: str,
) -> tuple[str, str | None]:
    template_id = factor_template_id or "custom"
    if template_id == "custom":
        return template_id, None
    if not FACTOR_TEMPLATE_ID_PATTERN.fullmatch(template_id):
        _raise_payload_validation_error(
            code="unsafe_factor_template_id",
            field="factor_template_id",
            message=(
                "factor_template_id may only contain ASCII letters, numbers, "
                "underscores, and hyphens"
            ),
        )
    if market == "a_share" and template_id not in _controlled_template_ids(market):
        _raise_payload_validation_error(
            code="unknown_factor_template_id",
            field="factor_template_id",
            message=f"unknown factor_template_id: {template_id}",
        )
    return template_id, f"{TEMPLATE_ROOT.as_posix()}/{template_id}.yaml"


def _controlled_template_ids(market: str) -> set[str]:
    registry = FactorTemplateRegistry.from_directory(TEMPLATE_ROOT, market=market)
    return {template.template_id for template in registry.templates()}


def _raise_payload_validation_error(*, code: str, field: str, message: str) -> None:
    detail = {
        "blocking": [
            {
                "severity": "blocking",
                "code": code,
                "message": message,
                "field": field,
            }
        ],
        "warnings": [],
    }
    raise HTTPException(status_code=422, detail=detail)


def _remember_pipeline_summary(
    pipeline_runs: dict[str, PipelineRunSummaryPayload],
    requested_run_id: str,
    summary: PipelineRunSummaryPayload,
) -> None:
    """Save status by requested run id and effective run ref."""

    pipeline_runs[_pipeline_summary_key(summary.kind, requested_run_id)] = summary
    if summary.run_ref:
        pipeline_runs[_pipeline_summary_key(summary.kind, summary.run_ref)] = summary


def _pipeline_summary_key(kind: PipelineResourceKind, run_id: str) -> str:
    return f"{kind}:{run_id}"


def _summary_from_snapshot(
    config: PipelineConfig,
    snapshot: PipelineSnapshot,
) -> PipelineRunSummaryPayload:
    warnings = [asdict(issue) for issue in snapshot.warnings]
    message = "snapshot generated"
    if warnings:
        message = f"snapshot generated with {len(warnings)} warning(s)"
    return PipelineRunSummaryPayload(
        kind="snapshot",
        run_id=config.run_id,
        status="ready",
        message=message,
        run_ref=snapshot.run_dir.name,
        warnings=warnings,
    )
