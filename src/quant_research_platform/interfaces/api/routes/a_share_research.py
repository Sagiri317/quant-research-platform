"""Backend-only A-share research batch routes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field

from quant_research_platform.interfaces.api.errors import ApiError
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.markets.implementations.a_share.factor_rebuild import (
    FactorRebuildConfig,
    run_factor_rebuild,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    literature_price_volume,
)
from quant_research_platform.research.ml_model_suite import (
    A_SHARE_MODEL_SUITE_ID,
    PREDICTION_HANDOFF_VERSION,
)
from quant_research_platform.research.prediction_handoff import (
    DEFAULT_NON_DL_MODEL_IDS,
    PredictionHandoffConfig,
    run_prediction_handoff,
)
from quant_research_platform.runtime.private_paths import text_has_private_marker

router = APIRouter()

_LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID = (
    literature_price_volume.LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID
)
_ALLOWED_FACTOR_PACKS = {_LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID}
_ALLOWED_MODEL_SUITE_IDS = {A_SHARE_MODEL_SUITE_ID}
_ALLOWED_PREDICTION_HANDOFF_VERSIONS = {PREDICTION_HANDOFF_VERSION}
_ALLOWED_RUNTIME_STATUSES = {"ready", "dry_run", "failed"}
_ALLOWED_EDA_STATUSES = {"ready", "failed"}
_ALLOWED_CHART_DEPENDENCY_STATUSES = {
    "ready",
    "not_generated",
    "dependency_unavailable",
}
_ALLOWED_FAILURE_CODES = {
    "dependency_unavailable",
    "duplicate_run_id",
    "factor_rebuild_failed",
    "factor_validation_failed",
    "insufficient_sample",
    "invalid_factor_rebuild_config",
    "invalid_feature_preset",
    "invalid_prediction_config",
    "memory_preflight_failed",
    "missing_factor_field",
    "missing_factor_input",
    "missing_prediction_input",
    "prediction_handoff_failed",
    "private_output_policy_violation",
    "warmup_insufficient",
}
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"(credential|secret|token|api[_-]?key|password)",
    re.IGNORECASE,
)
_SAFE_PREDICTION_COLUMN_PATTERN = re.compile(
    r"^prediction_score_[A-Za-z0-9_]+_(?:\d+d|label)$"
)
_SAFE_TARGET_COLUMN_PATTERN = re.compile(r"^forward_return_\d+d$")
_SAFE_CHART_FILE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.png$")


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
    start_date: str | None = None
    end_date: str | None = None
    calculation_start_date: str | None = None
    evaluation_start_date: str | None = None
    factor_set_tier: str = "full"
    require_sufficient_warmup: bool = False


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
    feature_columns: list[str] = Field(default_factory=list)
    feature_preset: str | None = None
    label_column: str = "forward_return_20d"
    return_source_column: str = "forward_return_1d"
    label_horizon_days: int = 20
    folds: list[dict[str, str | None]]
    model_ids: list[str] = list(DEFAULT_NON_DL_MODEL_IDS)
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
                start_date=payload.start_date,
                end_date=payload.end_date,
                calculation_start_date=payload.calculation_start_date,
                evaluation_start_date=payload.evaluation_start_date,
                factor_set_tier=payload.factor_set_tier,
                require_sufficient_warmup=payload.require_sufficient_warmup,
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
        feature_columns = _resolve_prediction_feature_columns(
            payload.input_parquet,
            explicit_columns=payload.feature_columns,
            feature_preset=payload.feature_preset,
        )
        resolved_feature_preset = (
            None if tuple(column for column in payload.feature_columns if column)
            else payload.feature_preset
        )
        result = run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=payload.input_parquet,
                output_root=context.prediction_handoff_root,
                run_id=payload.run_id,
                timestamp_column=payload.timestamp_column,
                entity_column=payload.entity_column,
                feature_columns=feature_columns,
                feature_preset=resolved_feature_preset,
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
    status_value = (
        _safe_allowed_runtime_string(summary.get("status"), _ALLOWED_RUNTIME_STATUSES)
        or "unknown"
    )
    factor_pack = _safe_allowed_runtime_string(
        summary.get("factor_pack"),
        _ALLOWED_FACTOR_PACKS,
        default=_LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
    )
    failure_code = _safe_allowed_runtime_string(
        summary.get("failure_code"),
        _ALLOWED_FAILURE_CODES,
    )
    payload = {
        "run_id": run_id,
        "status": status_value,
        "factor_pack": factor_pack,
        "row_count": int(summary.get("row_count", 0) or 0),
    }
    if failure_code:
        payload["failure_code"] = failure_code
    return payload


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
        ("warmup_sufficiency", "warmup_sufficiency.json"),
        ("factor_rebuild", "factor_rebuild.parquet"),
        ("feature_set_manifest", "feature_set_manifest.json"),
        ("overwrite_manifest", "overwrite_manifest.json"),
        ("eda", "eda/eda_manifest.json"),
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


@router.get("/api/a-share/factor-rebuild-runs/{run_id}/schema-inventory")
def factor_rebuild_schema_inventory(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Return sanitized aggregate schema inventory for a factor rebuild run."""

    run_dir = _resolve_run_dir(api_context(request).factor_rebuild_root, run_id)
    return _read_schema_inventory_payload(
        run_dir / "schema_inventory.json",
        run_id=run_id,
        invalid_error_code="factor_rebuild_schema_inventory_invalid",
        invalid_message="factor rebuild schema inventory invalid",
        not_found_error_code="factor_rebuild_schema_inventory_not_found",
        not_found_message="factor rebuild schema inventory not found",
    )


@router.get("/api/a-share/factor-rebuild-runs/{run_id}/eda")
def factor_rebuild_eda(run_id: str, request: Request) -> dict[str, object]:
    """Return EDA status without exposing private paths."""

    run_dir = _resolve_run_dir(api_context(request).factor_rebuild_root, run_id)
    manifest = run_dir / "eda" / "eda_manifest.json"
    if not manifest.is_file():
        return {"run_id": run_id, "status": "not_generated"}
    value = _read_json(
        manifest,
        invalid_error_code="factor_rebuild_eda_invalid",
        invalid_message="factor rebuild EDA invalid",
    )
    eda_status = (
        _safe_allowed_runtime_string(value.get("status"), _ALLOWED_EDA_STATUSES)
        or "unknown"
    )
    return _eda_manifest_payload(
        run_id=run_id,
        status=eda_status,
        artifact_count=int(value.get("artifact_count", 0) or 0),
        manifest=value,
    )


@router.get("/api/a-share/prediction-handoffs/{run_id}")
def prediction_handoff_status(run_id: str, request: Request) -> dict[str, object]:
    """Return sanitized prediction handoff status."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    summary = _read_json(
        run_dir / "runtime_summary.json",
        invalid_error_code="prediction_handoff_status_invalid",
        invalid_message="prediction handoff status invalid",
    )
    status_value = (
        _safe_allowed_runtime_string(summary.get("status"), _ALLOWED_RUNTIME_STATUSES)
        or "unknown"
    )
    model_suite_id = _safe_allowed_runtime_string(
        summary.get("model_suite_id"),
        _ALLOWED_MODEL_SUITE_IDS,
        default=A_SHARE_MODEL_SUITE_ID,
    )
    failure_code = _safe_allowed_runtime_string(
        summary.get("failure_code"),
        _ALLOWED_FAILURE_CODES,
    )
    payload = {
        "run_id": run_id,
        "status": status_value,
        "row_count": int(summary.get("row_count", 0) or 0),
        "model_suite_id": model_suite_id,
    }
    if failure_code:
        payload["failure_code"] = failure_code
    return payload


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
        ("feature_set_manifest", "feature_set_manifest.json"),
        ("predictions", "predictions.parquet"),
        ("prediction_records", "prediction_records.parquet"),
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
    metadata = _read_json(
        run_dir / "model_metadata.json",
        invalid_error_code="prediction_handoff_status_invalid",
        invalid_message="prediction handoff status invalid",
    )
    target_column = _safe_schema_string(metadata.get("target_column")) or ""
    if target_column and not _SAFE_TARGET_COLUMN_PATTERN.fullmatch(target_column):
        target_column = ""
    return {
        "prediction_handoff_version": (
            _safe_allowed_runtime_string(
                metadata.get("prediction_handoff_version"),
                _ALLOWED_PREDICTION_HANDOFF_VERSIONS,
            )
            or ""
        ),
        "model_suite_id": (
            _safe_allowed_runtime_string(
                metadata.get("model_suite_id"),
                _ALLOWED_MODEL_SUITE_IDS,
                default=A_SHARE_MODEL_SUITE_ID,
            )
            or A_SHARE_MODEL_SUITE_ID
        ),
        "prediction_columns": _safe_prediction_columns(
            metadata.get("prediction_columns", [])
        ),
        "target_column": target_column,
        "label_horizon_days": int(metadata.get("label_horizon_days", 0) or 0),
    }


@router.get("/api/a-share/prediction-handoffs/{run_id}/schema-inventory")
def prediction_handoff_schema_inventory(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Return sanitized aggregate schema inventory for one prediction handoff."""

    run_dir = _resolve_prediction_dir(
        api_context(request).prediction_handoff_root,
        run_id,
    )
    return _read_schema_inventory_payload(
        run_dir / "schema_inventory.json",
        run_id=run_id,
        invalid_error_code="prediction_schema_inventory_invalid",
        invalid_message="prediction schema inventory invalid",
        not_found_error_code="prediction_schema_inventory_not_found",
        not_found_message="prediction schema inventory not found",
    )


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
    value = _read_json_list(
        path,
        invalid_error_code="prediction_comparison_invalid",
        invalid_message="prediction comparison invalid",
    )
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
    value = _read_json(
        path,
        invalid_error_code="prediction_eda_invalid",
        invalid_message="prediction EDA invalid",
    )
    eda_status = (
        _safe_allowed_runtime_string(value.get("status"), _ALLOWED_EDA_STATUSES)
        or "unknown"
    )
    return _eda_manifest_payload(
        run_id=run_id,
        status=eda_status,
        artifact_count=int(value.get("artifact_count", 0) or 0),
        manifest=value,
    )


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
    return _resolve_named_run_dir(
        root,
        run_id,
        invalid_error_code="invalid_factor_rebuild_config",
        not_found_error_code="factor_rebuild_not_found",
        message="factor rebuild run not found",
    )


def _resolve_prediction_dir(root: Path, run_id: str) -> Path:
    return _resolve_named_run_dir(
        root,
        run_id,
        invalid_error_code="invalid_prediction_config",
        not_found_error_code="prediction_handoff_not_found",
        message="prediction handoff run not found",
    )


def _resolve_named_run_dir(
    root: Path,
    run_id: str,
    *,
    invalid_error_code: str,
    not_found_error_code: str,
    message: str,
) -> Path:
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ApiError(
            error_code=invalid_error_code,
            message=message,
            status_code=404,
        )
    run_dir = root / run_id
    if not run_dir.is_dir():
        raise ApiError(
            error_code=not_found_error_code,
            message=message,
            status_code=404,
        )
    return run_dir


def _read_json(
    path: Path,
    *,
    invalid_error_code: str = "factor_rebuild_status_invalid",
    invalid_message: str = "factor rebuild status invalid",
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            error_code=invalid_error_code,
            message=invalid_message,
            status_code=422,
        ) from exc
    if not isinstance(value, dict):
        raise ApiError(
            error_code=invalid_error_code,
            message=invalid_message,
            status_code=422,
        )
    return value


def _read_json_list(
    path: Path,
    *,
    invalid_error_code: str,
    invalid_message: str,
) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            error_code=invalid_error_code,
            message=invalid_message,
            status_code=422,
        ) from exc
    if not isinstance(value, list):
        raise ApiError(
            error_code=invalid_error_code,
            message=invalid_message,
            status_code=422,
        )
    return value


def _eda_manifest_payload(
    *,
    run_id: str,
    status: str,
    artifact_count: int,
    manifest: dict[str, Any],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "artifact_count": artifact_count,
    }
    chart_count = _optional_int(manifest.get("chart_count"))
    if chart_count is not None:
        payload["chart_count"] = chart_count
    chart_status = _safe_allowed_runtime_string(
        manifest.get("chart_dependency_status"),
        _ALLOWED_CHART_DEPENDENCY_STATUSES,
    )
    if chart_status:
        payload["chart_dependency_status"] = chart_status
    chart_files = _safe_chart_files(manifest.get("chart_files"))
    if chart_files:
        payload["chart_files"] = chart_files
    return payload


def _safe_chart_files(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    files: list[str] = []
    for item in value:
        text = _safe_schema_string(item)
        if text and _SAFE_CHART_FILE_PATTERN.fullmatch(text):
            files.append(text)
    return files


def _read_schema_inventory_payload(
    path: Path,
    *,
    run_id: str,
    invalid_error_code: str,
    invalid_message: str,
    not_found_error_code: str,
    not_found_message: str,
) -> dict[str, object]:
    if not path.is_file():
        raise ApiError(
            error_code=not_found_error_code,
            message=not_found_message,
            status_code=404,
        )
    value = _read_json(
        path,
        invalid_error_code=invalid_error_code,
        invalid_message=invalid_message,
    )
    try:
        return _schema_inventory_payload(run_id, value)
    except (TypeError, ValueError) as exc:
        raise ApiError(
            error_code=invalid_error_code,
            message=invalid_message,
            status_code=422,
        ) from exc


def _schema_inventory_payload(run_id: str, value: dict[str, Any]) -> dict[str, object]:
    return {
        "run_id": run_id,
        "schema_inventory_version": _safe_schema_string(
            value.get("schema_inventory_version")
        )
        or "",
        "row_count": _required_int(value.get("row_count", 0)),
        "row_group_count": _required_int(value.get("row_group_count", 0)),
        "columns": _safe_schema_columns(value.get("columns")),
        "missing_expected_columns": _safe_schema_string_list(
            value.get("missing_expected_columns")
        ),
        "unexpected_columns": _safe_schema_string_list(
            value.get("unexpected_columns")
        ),
        "duplicate_key_count": _optional_int(value.get("duplicate_key_count")),
        "date_range": _safe_date_range(value.get("date_range")),
        "market_scope": _safe_market_scope(value.get("market_scope")),
        "date_filter": _safe_date_range(value.get("date_filter")),
    }


def _safe_schema_columns(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    columns: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _safe_schema_string(item.get("name"))
        dtype = _safe_schema_string(item.get("dtype"))
        role = _safe_schema_string(item.get("role"))
        if name and dtype and role:
            columns.append({"name": name, "dtype": dtype, "role": role})
    return columns


def _safe_schema_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        text
        for item in value
        if (text := _safe_schema_string(item)) is not None
    ]


def _safe_date_range(value: object) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    payload: dict[str, str | None] = {}
    for key in ("min", "max", "start_date", "end_date"):
        raw = value.get(key)
        if raw is None:
            payload[key] = None
            continue
        text = _safe_schema_string(raw)
        if text is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            payload[key] = text
    return payload or None


def _safe_market_scope(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    field = _safe_schema_string(value.get("field"))
    if not field:
        return None
    return {
        "field": field,
        "true_count": _required_int(value.get("true_count", 0)),
        "false_count": _required_int(value.get("false_count", 0)),
        "null_count": _required_int(value.get("null_count", 0)),
    }


def _required_int(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _resolve_prediction_feature_columns(
    input_path: Path,
    *,
    explicit_columns: list[str],
    feature_preset: str | None,
) -> tuple[str, ...]:
    explicit = tuple(column for column in explicit_columns if column)
    if explicit:
        return explicit
    if (
        feature_preset
        != literature_price_volume.LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID
    ):
        raise ValueError("invalid_feature_preset")
    columns = pq.ParquetFile(input_path).schema.names
    selected, _ = (
        literature_price_volume
        .select_literature_price_volume_feature_columns_from_schema(columns)
    )
    return selected


def _safe_allowed_runtime_string(
    value: object,
    allowed_values: set[str],
    *,
    default: str | None = None,
) -> str | None:
    if not isinstance(value, str) or not value:
        return default
    if (
        _SENSITIVE_TEXT_PATTERN.search(value)
        or text_has_private_marker(value)
        or _has_unsafe_path_text(value)
    ):
        return default
    return value if value in allowed_values else default


def _safe_schema_string(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if (
        _SENSITIVE_TEXT_PATTERN.search(value)
        or text_has_private_marker(value)
        or _has_unsafe_path_text(value)
    ):
        return None
    return value


def _safe_prediction_columns(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    columns: list[str] = []
    for item in value:
        column = _safe_schema_string(item)
        if column and _SAFE_PREDICTION_COLUMN_PATTERN.fullmatch(column):
            columns.append(column)
    return columns


def _has_unsafe_path_text(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:/", normalized):
        return True
    if normalized.startswith(("/", "//")):
        return True
    if re.search(r"(?<![A-Za-z0-9_.])\.\.(?:/|$)", normalized):
        return True
    return any(".." in Path(token).parts for token in normalized.split())


def _raise_a_share_api_error(exc: BaseException) -> None:
    error_code = _batch_error_code(exc, default="factor_rebuild_failed")
    status_code = 409 if error_code == "duplicate_run_id" else 422
    raise ApiError(
        error_code=error_code,
        message=_safe_error_message(error_code),
        status_code=status_code,
    ) from exc


def _raise_prediction_api_error(exc: BaseException) -> None:
    error_code = _batch_error_code(exc, default="prediction_handoff_failed")
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


def _batch_error_code(exc: BaseException, *, default: str) -> str:
    if isinstance(exc, FileNotFoundError):
        if "prediction" in default:
            return "missing_prediction_input"
        return "missing_factor_input"
    detail = str(exc)
    known = {
        "duplicate_run_id",
        "private_output_policy_violation",
        "memory_preflight_failed",
        "invalid_feature_preset",
        "invalid_prediction_config",
        "invalid_factor_rebuild_config",
        "insufficient_sample",
        "dependency_unavailable",
        "factor_validation_failed",
        "missing_factor_field",
        "warmup_insufficient",
    }
    if detail in known:
        return detail
    if detail.startswith("invalid_feature_preset:"):
        return "invalid_feature_preset"
    return default


def _safe_error_message(error_code: str) -> str:
    if error_code == "duplicate_run_id":
        return "factor rebuild run_id already exists"
    if error_code == "private_output_policy_violation":
        return "private output policy violation"
    if error_code == "missing_factor_input":
        return "factor input not found"
    return "factor rebuild request failed"


__all__ = ["router"]
