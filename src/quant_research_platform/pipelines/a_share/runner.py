"""运行 A 股 processed-existing 轻量 pipeline。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from quant_research_platform.markets.implementations.a_share import evaluation
from quant_research_platform.markets.implementations.a_share.factors import (
    registry as factor_registry,
)
from quant_research_platform.pipelines.a_share.config import (
    COLUMN_NAME_PATTERN,
    PipelineConfig,
    ValidationIssue,
    validate_pipeline_config,
)
from quant_research_platform.pipelines.a_share.snapshots import (
    PipelineSnapshotError,
    write_run_snapshot,
)

PipelineExecutionMode = Literal["processed_existing"]
run_processed_factor_evaluation = (
    evaluation.evaluation_adapter.run_processed_factor_evaluation
)


class PipelineRunnerError(RuntimeError):
    """表示 pipeline runner 无法继续执行。"""

    def __init__(
        self,
        message: str,
        *,
        run_dir: Path | None = None,
        failure_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.run_dir = run_dir
        self.failure_code = failure_code


@dataclass(frozen=True)
class PipelineCheckpoint:
    """记录一个 pipeline step 的状态和输出。"""

    key: str
    label: str
    status: str
    artifact: str | None = None
    message: str | None = None
    duration_seconds: float | None = None
    row_count: int | None = None
    failure_code: str | None = None
    warnings: list[dict[str, object]] = field(default_factory=list)
    blocking_issues: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineRunResult:
    """记录 pipeline runner 的最终状态。"""

    run_id: str
    status: str
    run_dir: Path
    checkpoints: list[PipelineCheckpoint]
    runtime_summary_path: Path
    evaluation_manifest_path: Path | None = None
    warnings: list[dict[str, object]] = field(default_factory=list)
    failed_step: str | None = None


def run_processed_existing_pipeline(
    config: PipelineConfig,
    *,
    processed_root: Path = Path("data/processed_private/csmar"),
    overwrite: bool = False,
    n_groups: int = 5,
) -> PipelineRunResult:
    """用已存在 processed factor table 运行 A 股 evaluation adapter。"""

    started = time.perf_counter()
    validation = validate_pipeline_config(config)
    if validation.has_blocking:
        raise PipelineRunnerError(_blocking_issue_message(validation.blocking))
    feature_columns = resolve_processed_factor_columns(config)
    try:
        snapshot = write_run_snapshot(config, overwrite=overwrite)
    except PipelineSnapshotError as exc:
        raise PipelineRunnerError(str(exc)) from exc

    warnings = [asdict(issue) for issue in validation.warnings]
    checkpoints = [
        PipelineCheckpoint(
            key="snapshot",
            label="配置快照",
            status="ready",
            artifact="run_config.yaml",
            warnings=warnings,
        )
    ]
    evaluation_root = snapshot.run_dir / "evaluation"
    evaluation_started = time.perf_counter()
    try:
        label_summary, manifest_path, evaluation_runs = run_processed_factor_evaluation(
            processed_root=processed_root,
            report_root=evaluation_root,
            start_year=int(config.date_start[:4]),
            end_year=int(config.date_end[:4]),
            requested_labels=config.label_columns,
            requested_factors=feature_columns,
            all_factors=False,
            n_groups=n_groups,
            start_date=config.date_start,
            end_date=config.date_end,
            pnl_return_column=config.pnl_return_column,
            cost_bps=config.cost_bps,
            portfolio_mode=config.portfolio_mode,
            engine=config.engine,
        )
    except Exception as exc:
        failure_code = _failure_code_from_exception(exc)
        checkpoint = PipelineCheckpoint(
            key="processed_evaluation",
            label="A 股 processed evaluation",
            status="failed",
            message="processed evaluation failed",
            duration_seconds=_elapsed(evaluation_started),
            failure_code=failure_code,
        )
        checkpoints.append(checkpoint)
        summary_path = _write_runtime_summary(
            snapshot.run_dir,
            config=config,
            status="failed",
            checkpoints=checkpoints,
            warnings=warnings,
            failed_step="processed_evaluation",
            failure_code=failure_code,
            started=started,
        )
        _write_checkpoints(snapshot.run_dir, checkpoints)
        raise PipelineRunnerError(
            f"processed evaluation failed: {failure_code}",
            run_dir=snapshot.run_dir,
            failure_code=failure_code,
        ) from exc

    row_count = int(label_summary["rows"].max()) if "rows" in label_summary else None
    checkpoints.append(
        PipelineCheckpoint(
            key="processed_evaluation",
            label="A 股 processed evaluation",
            status="ready",
            artifact=_relative_artifact_path(manifest_path, snapshot.run_dir),
            duration_seconds=_elapsed(evaluation_started),
            row_count=row_count,
        )
    )
    summary_path = _write_runtime_summary(
        snapshot.run_dir,
        config=config,
        status="ready",
        checkpoints=checkpoints,
        warnings=warnings,
        failed_step=None,
        failure_code=None,
        started=started,
    )
    _write_checkpoints(snapshot.run_dir, checkpoints)
    return PipelineRunResult(
        run_id=snapshot.run_dir.name,
        status="ready",
        run_dir=snapshot.run_dir,
        checkpoints=checkpoints,
        runtime_summary_path=summary_path,
        evaluation_manifest_path=manifest_path,
        warnings=warnings,
    )


def resolve_processed_factor_columns(config: PipelineConfig) -> tuple[str, ...]:
    """解析 processed-existing 模式可直接读取的 factor columns。"""

    if config.processed_factor_column is not None:
        return (config.processed_factor_column,)
    template_column = factor_registry.processed_factor_column_for_template(
        config.factor_template_id
    )
    if template_column is not None:
        return (template_column,)
    if COLUMN_NAME_PATTERN.fullmatch(config.factor_expression):
        return (config.factor_expression,)
    raise PipelineRunnerError(
        "processed_existing mode requires a known factor_template_id or an "
        "existing processed factor column name."
    )


def _write_runtime_summary(
    run_dir: Path,
    *,
    config: PipelineConfig,
    status: str,
    checkpoints: list[PipelineCheckpoint],
    warnings: list[dict[str, object]],
    failed_step: str | None,
    failure_code: str | None,
    started: float,
) -> Path:
    payload = {
        "run_id": run_dir.name,
        "requested_run_id": config.run_id,
        "status": status,
        "execution_mode": "processed_existing",
        "date_start": config.date_start,
        "date_end": config.date_end,
        "engine": config.engine,
        "duration_seconds": _elapsed(started),
        "warnings": warnings,
        "failed_step": failed_step,
        "failure_code": failure_code,
        "checkpoints": [_checkpoint_payload(checkpoint) for checkpoint in checkpoints],
    }
    path = run_dir / "runtime_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_checkpoints(run_dir: Path, checkpoints: list[PipelineCheckpoint]) -> Path:
    path = run_dir / "checkpoints.json"
    path.write_text(
        json.dumps(
            [_checkpoint_payload(checkpoint) for checkpoint in checkpoints],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _checkpoint_payload(checkpoint: PipelineCheckpoint) -> dict[str, object]:
    return asdict(checkpoint)


def _relative_artifact_path(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _blocking_issue_message(issues: list[ValidationIssue]) -> str:
    codes = ", ".join(issue.code for issue in issues)
    return f"blocking config issues prevent pipeline execution: {codes}"


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def _failure_code_from_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if (
        "missing" in message
        or "required column" in message
        or "not in index" in message
    ):
        return "missing_processed_field"
    if "empty" in message or "insufficient" in message or "no rows" in message:
        return "insufficient_sample"
    return "cannot_compute"


__all__ = [
    "PipelineCheckpoint",
    "PipelineExecutionMode",
    "PipelineRunResult",
    "PipelineRunnerError",
    "resolve_processed_factor_columns",
    "run_processed_existing_pipeline",
]
