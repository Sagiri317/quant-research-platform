"""Backend-only A-share research batch routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict

from quant_research_platform.interfaces.api.errors import ApiError
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.markets.implementations.a_share.factor_rebuild import (
    FactorRebuildConfig,
    run_factor_rebuild,
)
from quant_research_platform.research.prediction_handoff import (
    PredictionHandoffConfig,
    run_prediction_handoff,
)

router = APIRouter()


class AShareFactorRebuildRequest(BaseModel):
    """Request payload for one local A-share factor rebuild run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    panel_parquet: Path
    factor_pack: str = "a_share_literature_price_volume_rebuild_v1"
    factor_parquet: Path | None = None
    dry_run: bool = False
    memory_budget_gb: float = 10.0
    overwrite_processed: bool = False
    backup_existing: bool = False


class AShareRunCheckpoint(BaseModel):
    key: str
    label: str
    status: str


class ASharePredictionHandoffRequest(BaseModel):
    """Request payload for one local A-share prediction handoff run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    input_parquet: Path
    timestamp_column: str = "timestamp"
    entity_column: str = "asset"
    feature_columns: list[str]
    label_column: str = "forward_return_20d"
    return_source_column: str = "forward_return_1d"
    label_horizon_days: int = 20
    folds: list[dict[str, str | None]]
    model_ids: list[str] = ["ridge", "elastic_net", "hist_gradient_boosting"]
    memory_budget_gb: float = 10.0
    dry_run: bool = False


@router.post(
    "/api/a-share/factor-rebuild-runs",
    status_code=status.HTTP_201_CREATED,
)
def create_factor_rebuild_run(
    payload: AShareFactorRebuildRequest,
    request: Request,
) -> dict[str, object]:
    """Run or dry-run an A-share literature factor rebuild."""

    context = api_context(request)
    try:
        result = run_factor_rebuild(
            FactorRebuildConfig(
                panel_parquet=payload.panel_parquet,
                output_root=context.factor_rebuild_root,
                run_id=payload.run_id,
                factor_pack=payload.factor_pack,
                memory_budget_gb=payload.memory_budget_gb,
                dry_run=payload.dry_run,
                factor_parquet=payload.factor_parquet,
                overwrite_processed=payload.overwrite_processed,
                backup_existing=payload.backup_existing,
            )
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _raise_a_share_api_error(exc)
    return _factor_rebuild_payload(
        run_id=payload.run_id,
        status=result.status,
        row_count=result.row_count,
        factor_pack=payload.factor_pack,
    )


@router.post("/api/a-share/prediction-handoffs", status_code=status.HTTP_201_CREATED)
def create_prediction_handoff(
    payload: ASharePredictionHandoffRequest,
    request: Request,
) -> dict[str, object]:
    """Run or dry-run an A-share prediction handoff."""

    context = api_context(request)
    try:
        result = run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=payload.input_parquet,
                output_root=context.prediction_handoff_root,
                run_id=payload.run_id,
                timestamp_column=payload.timestamp_column,
                entity_column=payload.entity_column,
                feature_columns=tuple(payload.feature_columns),
                label_column=payload.label_column,
                return_source_column=payload.return_source_column,
                label_horizon_days=payload.label_horizon_days,
                folds=tuple(payload.folds),
                model_ids=tuple(payload.model_ids),
                memory_budget_gb=payload.memory_budget_gb,
                dry_run=payload.dry_run,
            )
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _raise_prediction_api_error(exc)
    return {
        "run_id": payload.run_id,
        "status": result.status,
        "row_count": result.row_count,
        "prediction_columns": list(result.prediction_columns),
        "model_suite_id": "n6_a_share_sklearn_boosting_suite_v1",
    }


@router.get("/api/a-share/factor-rebuild-runs/{run_id}")
def factor_rebuild_status(run_id: str, request: Request) -> dict[str, object]:
    """Return sanitized factor rebuild run status."""

    run_dir = _resolve_run_dir(api_context(request).factor_rebuild_root, run_id)
    summary_path = run_dir / "runtime_summary.json"
    if not summary_path.is_file():
        raise ApiError(
            error_code="missing_factor_rebuild_status",
            message="factor rebuild status not found",
            status_code=404,
        )
    summary = _read_json(summary_path)
    return {
        "run_id": run_id,
        "status": str(summary.get("status", "unknown")),
        "factor_pack": str(
            summary.get("factor_pack", "a_share_literature_price_volume_rebuild_v1")
        ),
        "row_count": int(summary.get("row_count", 0) or 0),
        **(
            {"failure_code": str(summary["failure_code"])}
            if isinstance(summary.get("failure_code"), str)
            else {}
        ),
    }


@router.get("/api/a-share/factor-rebuild-runs/{run_id}/checkpoints")
def factor_rebuild_checkpoints(
    run_id: str,
    request: Request,
) -> list[dict[str, object]]:
    """Return aggregate artifact checkpoints for one factor rebuild run."""

    run_dir = _resolve_run_dir(api_context(request).factor_rebuild_root, run_id)
    checkpoints = (
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("runtime_summary", "runtime_summary.json"),
    )
    return [
        AShareRunCheckpoint(
            key=key,
            label=file_name,
            status="ready" if (run_dir / file_name).is_file() else "pending",
        ).model_dump()
        for key, file_name in checkpoints
    ]


@router.get("/api/a-share/factor-rebuild-runs/{run_id}/eda")
def factor_rebuild_eda(run_id: str, request: Request) -> dict[str, object]:
    """Return EDA status without exposing private paths."""

    run_dir = _resolve_run_dir(api_context(request).factor_rebuild_root, run_id)
    manifest = run_dir / "eda" / "eda_manifest.json"
    if not manifest.is_file():
        return {"run_id": run_id, "status": "not_generated"}
    value = _read_json(manifest)
    return {
        "run_id": run_id,
        "status": str(value.get("status", "ready")),
        "artifact_count": int(value.get("artifact_count", 0) or 0),
    }


@router.get("/api/a-share/prediction-handoffs/{run_id}")
def prediction_handoff_status(run_id: str, request: Request) -> dict[str, object]:
    """Return sanitized prediction handoff status."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    summary = _read_json(run_dir / "runtime_summary.json")
    return {
        "run_id": run_id,
        "status": str(summary.get("status", "unknown")),
        "row_count": int(summary.get("row_count", 0) or 0),
        "model_suite_id": str(
            summary.get("model_suite_id", "n6_a_share_sklearn_boosting_suite_v1")
        ),
    }


@router.get("/api/a-share/prediction-handoffs/{run_id}/checkpoints")
def prediction_handoff_checkpoints(
    run_id: str,
    request: Request,
) -> list[dict[str, object]]:
    """Return aggregate checkpoints for one prediction handoff."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    checkpoints = (
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("predictions", "predictions.parquet"),
        ("evaluation", "evaluation/manifest.json"),
        ("comparison", "comparison/model_leaderboard.csv"),
        ("eda", "eda/eda_manifest.json"),
        ("runtime_summary", "runtime_summary.json"),
    )
    return [
        AShareRunCheckpoint(
            key=key,
            label=label,
            status="ready" if (run_dir / label).is_file() else "pending",
        ).model_dump()
        for key, label in checkpoints
    ]


@router.get("/api/a-share/prediction-handoffs/{run_id}/schema")
def prediction_handoff_schema(run_id: str, request: Request) -> dict[str, object]:
    """Return portable prediction handoff schema metadata."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    metadata = _read_json(run_dir / "model_metadata.json")
    return {
        "prediction_handoff_version": str(
            metadata.get("prediction_handoff_version", "")
        ),
        "model_suite_id": str(metadata.get("model_suite_id", "")),
        "prediction_columns": list(metadata.get("prediction_columns", [])),
        "target_column": str(metadata.get("target_column", "")),
        "label_horizon_days": int(metadata.get("label_horizon_days", 0) or 0),
    }


@router.get("/api/a-share/prediction-handoffs/{run_id}/comparison")
def prediction_handoff_comparison(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Return aggregate comparison status and model count."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    path = run_dir / "comparison" / "model_leaderboard.json"
    if not path.is_file():
        return {"run_id": run_id, "status": "not_generated", "model_count": 0}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ApiError(
            error_code="prediction_comparison_failed",
            message="prediction comparison invalid",
            status_code=422,
        )
    return {"run_id": run_id, "status": "ready", "model_count": len(value)}


@router.get("/api/a-share/prediction-handoffs/{run_id}/eda")
def prediction_handoff_eda(run_id: str, request: Request) -> dict[str, object]:
    """Return prediction handoff EDA status."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    path = run_dir / "eda" / "eda_manifest.json"
    if not path.is_file():
        return {"run_id": run_id, "status": "not_generated"}
    value = _read_json(path)
    return {
        "run_id": run_id,
        "status": str(value.get("status", "ready")),
        "artifact_count": int(value.get("artifact_count", 0) or 0),
    }


def _factor_rebuild_payload(
    *,
    run_id: str,
    status: str,
    row_count: int,
    factor_pack: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": status,
        "row_count": int(row_count),
        "factor_pack": factor_pack,
    }


def _resolve_run_dir(root: Path, run_id: str) -> Path:
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ApiError(
            error_code="invalid_factor_rebuild_config",
            message="factor rebuild run not found",
            status_code=404,
        )
    run_dir = root / run_id
    if not run_dir.is_dir():
        raise ApiError(
            error_code="factor_rebuild_not_found",
            message="factor rebuild run not found",
            status_code=404,
        )
    return run_dir


def _resolve_prediction_dir(root: Path, run_id: str) -> Path:
    return _resolve_run_dir(root, run_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            error_code="factor_rebuild_status_invalid",
            message="factor rebuild status invalid",
            status_code=422,
        ) from exc
    if not isinstance(value, dict):
        raise ApiError(
            error_code="factor_rebuild_status_invalid",
            message="factor rebuild status invalid",
            status_code=422,
        )
    return value


def _raise_a_share_api_error(exc: BaseException) -> None:
    detail = str(exc)
    error_code = detail if detail else "unknown_n6_a_share_goal_error"
    if isinstance(exc, FileNotFoundError):
        error_code = "missing_factor_input"
    status_code = 409 if error_code == "duplicate_run_id" else 422
    raise ApiError(
        error_code=error_code,
        message=_safe_error_message(error_code),
        status_code=status_code,
    ) from exc


def _raise_prediction_api_error(exc: BaseException) -> None:
    detail = str(exc)
    error_code = detail if detail else "unknown_n6_a_share_goal_error"
    if isinstance(exc, FileNotFoundError):
        error_code = "missing_prediction_input"
    status_code = 409 if error_code == "duplicate_run_id" else 422
    message = "prediction handoff request failed"
    if error_code == "duplicate_run_id":
        message = "prediction handoff run_id already exists"
    elif error_code == "private_output_policy_violation":
        message = "private output policy violation"
    elif error_code == "missing_prediction_input":
        message = "prediction input not found"
    raise ApiError(
        error_code=error_code,
        message=message,
        status_code=status_code,
    ) from exc


def _safe_error_message(error_code: str) -> str:
    if error_code == "duplicate_run_id":
        return "factor rebuild run_id already exists"
    if error_code == "private_output_policy_violation":
        return "private output policy violation"
    if error_code == "missing_factor_input":
        return "factor input not found"
    return "factor rebuild request failed"


__all__ = ["router"]
