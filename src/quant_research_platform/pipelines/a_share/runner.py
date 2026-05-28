"""Run the local A-share build/evaluate pipeline from an explicit start layer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

from quant_research_platform.evaluations.diagnostics import DiagnosticsSpec
from quant_research_platform.markets.implementations.a_share.csmar import (
    build_interim,
)
from quant_research_platform.markets.implementations.a_share.datasets import (
    build_panel,
)
from quant_research_platform.markets.implementations.a_share.evaluation.service import (
    AShareEvaluationRequest,
    run_a_share_evaluation,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    build_factors,
    price_volume,
)
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

StepPayload = dict[str, Any]
BuildStep = Callable[[PipelineConfig], StepPayload]
EvaluateStep = Callable[[PipelineConfig, Path], StepPayload]
build_interim_main = build_interim.main
build_panel_main = build_panel.main
build_factors_main = build_factors.main


class PipelineRunnerError(RuntimeError):
    """Raised when the A-share pipeline cannot continue."""

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
    """One persisted pipeline step status."""

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
    """Final pipeline runner status."""

    run_id: str
    status: str
    run_dir: Path
    checkpoints: list[PipelineCheckpoint]
    runtime_summary_path: Path
    evaluation_artifact_path: Path | None = None
    warnings: list[dict[str, object]] = field(default_factory=list)
    failed_step: str | None = None


@dataclass(frozen=True)
class _StepSpec:
    key: str
    label: str
    function: BuildStep | EvaluateStep
    artifact: str | None = None


def run_a_share_pipeline(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    n_groups: int = 5,
) -> PipelineRunResult:
    """Run A-share build/evaluate steps from ``config.start_from`` to evaluation."""

    started = time.perf_counter()
    validation = validate_pipeline_config(
        config,
        allow_processed_factor_expression=True,
    )
    if validation.has_blocking:
        raise PipelineRunnerError(_blocking_issue_message(validation.blocking))
    try:
        requested_factors = resolve_processed_factor_columns(config)
    except PipelineRunnerError as exc:
        raise PipelineRunnerError(
            str(exc),
            failure_code="unresolved_processed_factor",
        ) from exc
    _validate_rebuild_outputs(config, requested_factors)
    try:
        snapshot = write_run_snapshot(
            config,
            overwrite=overwrite,
            allow_processed_factor_expression=True,
        )
    except PipelineSnapshotError as exc:
        raise PipelineRunnerError(str(exc)) from exc

    warnings = [asdict(issue) for issue in validation.warnings]
    checkpoints = [
        PipelineCheckpoint(
            key="snapshot",
            label="Config snapshot",
            status="ready",
            artifact="run_config.yaml",
            warnings=warnings,
        )
    ]
    evaluation_artifact_path: Path | None = None

    all_steps = _all_steps()
    start_index = _start_step_index(config.start_from)
    checkpoints.extend(
        _skipped_checkpoints(
            all_steps[:start_index],
            message=f"skipped by start_from={config.start_from}",
        )
    )

    for step_index, step in enumerate(all_steps[start_index:], start=start_index):
        step_started = time.perf_counter()
        try:
            if step.key == "evaluate":
                if n_groups == 5:
                    payload = _run_evaluate_step(config, snapshot.run_dir)
                else:
                    payload = _run_evaluate_step(
                        config,
                        snapshot.run_dir,
                        n_groups=n_groups,
                    )
            else:
                payload = step.function(config)  # type: ignore[misc]
        except Exception as exc:
            failure_code = _failure_code_from_exception(exc)
            message = (
                "pipeline evaluation failed"
                if step.key == "evaluate"
                else f"{step.key} failed"
            )
            checkpoints.append(
                PipelineCheckpoint(
                    key=step.key,
                    label=step.label,
                    status="failed",
                    artifact=step.artifact,
                    message=message,
                    duration_seconds=_elapsed(step_started),
                    failure_code=failure_code,
                )
            )
            checkpoints.extend(
                _skipped_checkpoints(
                    all_steps[step_index + 1 :],
                    message=f"skipped after failed step={step.key}",
                )
            )
            _write_runtime_summary(
                snapshot.run_dir,
                config=config,
                status="failed",
                checkpoints=checkpoints,
                warnings=warnings,
                failed_step=step.key,
                failure_code=failure_code,
                started=started,
            )
            _write_checkpoints(snapshot.run_dir, checkpoints)
            error_message = (
                f"pipeline evaluation failed: {failure_code}"
                if step.key == "evaluate"
                else f"pipeline failed at {step.key}: {failure_code}"
            )
            raise PipelineRunnerError(
                error_message,
                run_dir=snapshot.run_dir,
                failure_code=failure_code,
            ) from exc

        artifact = payload.get("artifact")
        row_count = payload.get("row_count")
        if step.key == "evaluate":
            evaluation_artifact_path = payload.get("evaluation_artifact_path")
        checkpoints.append(
            PipelineCheckpoint(
                key=step.key,
                label=step.label,
                status="ready",
                artifact=str(artifact) if artifact is not None else step.artifact,
                duration_seconds=_elapsed(step_started),
                row_count=int(row_count) if row_count is not None else None,
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
        evaluation_artifact_path=evaluation_artifact_path,
        warnings=warnings,
    )


def resolve_processed_factor_columns(config: PipelineConfig) -> tuple[str, ...]:
    """Resolve factor columns that should already exist in processed factor tables."""

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
        "existing factor evaluation requires a known factor_template_id or an "
        "existing processed factor column name."
    )


def _all_steps() -> tuple[_StepSpec, ...]:
    return (
        _StepSpec(
            key="build_interim",
            label="A-share interim build",
            function=_run_build_interim_step,
            artifact="interim_dataset",
        ),
        _StepSpec(
            key="build_panel",
            label="A-share research panel build",
            function=_run_build_panel_step,
            artifact="research_panel",
        ),
        _StepSpec(
            key="build_factors",
            label="A-share factor table build",
            function=_run_build_factors_step,
            artifact="factor_table",
        ),
        _StepSpec(
            key="evaluate",
            label="A-share evaluation",
            function=_run_evaluate_step,
            artifact="evaluation/manifest.json",
        ),
    )


def _start_step_index(start_from: str) -> int:
    start_index = {
        "rebuild_interim": 0,
        "rebuild_panel": 1,
        "rebuild_factors": 2,
        "existing_factors": 3,
    }[start_from]
    return start_index


def _skipped_checkpoints(
    steps: tuple[_StepSpec, ...],
    *,
    message: str,
) -> list[PipelineCheckpoint]:
    return [
        PipelineCheckpoint(
            key=step.key,
            label=step.label,
            status="skipped",
            artifact=step.artifact,
            message=message,
        )
        for step in steps
    ]


def _validate_rebuild_outputs(
    config: PipelineConfig,
    requested_factors: tuple[str, ...],
) -> None:
    if config.start_from == "existing_factors":
        return
    expected_factors = set(
        price_volume.expected_price_volume_factor_columns(windows=config.factor_windows)
    )
    missing_factors = [
        factor for factor in requested_factors if factor not in expected_factors
    ]
    if missing_factors:
        raise PipelineRunnerError(
            f"unavailable_factor_column: {', '.join(missing_factors)}",
            failure_code="unavailable_factor_column",
        )

    expected_labels = {
        f"forward_return_{horizon}d" for horizon in config.label_horizons
    }
    requested_labels = set(config.label_columns)
    requested_labels.add(config.return_source_column)
    missing_labels = sorted(
        label for label in requested_labels if label not in expected_labels
    )
    if missing_labels:
        raise PipelineRunnerError(
            f"unavailable_label_column: {', '.join(missing_labels)}",
            failure_code="unavailable_label_column",
        )


def _run_build_interim_step(config: PipelineConfig) -> dict[str, Any]:
    argv = [
        "--raw-root",
        config.raw_root.as_posix(),
        "--output-root",
        config.interim_root.as_posix(),
        "--start-date",
        config.date_start,
        "--end-date",
        config.date_end,
    ]
    _require_zero_exit("build_interim", _call_main_sanitized(build_interim_main, argv))
    return {"artifact": "interim_dataset", "row_count": None}


def _run_build_panel_step(config: PipelineConfig) -> dict[str, Any]:
    argv = [
        "--interim-root",
        config.interim_root.as_posix(),
        "--processed-root",
        config.processed_root.as_posix(),
        "--start-year",
        str(_start_year(config)),
        "--end-year",
        str(_end_year(config)),
        "--end-date",
        config.date_end,
        "--min-listing-trading-days",
        str(config.min_listing_trading_days),
    ]
    _require_zero_exit("build_panel", _call_main_sanitized(build_panel_main, argv))
    return {"artifact": "research_panel", "row_count": None}


def _run_build_factors_step(config: PipelineConfig) -> dict[str, Any]:
    argv = [
        "--processed-root",
        config.processed_root.as_posix(),
        "--start-year",
        str(_start_year(config)),
        "--end-year",
        str(_end_year(config)),
        "--factor-set",
        "price_volume",
        "--windows",
        ",".join(str(value) for value in config.factor_windows),
        "--horizons",
        ",".join(str(value) for value in config.label_horizons),
    ]
    _require_zero_exit("build_factors", _call_main_sanitized(build_factors_main, argv))
    return {
        "artifact": "factor_table",
        "row_count": None,
    }


def _run_evaluate_step(
    config: PipelineConfig,
    run_dir: Path,
    *,
    n_groups: int = 5,
) -> dict[str, Any]:
    result = run_a_share_evaluation(
        AShareEvaluationRequest(
            processed_root=config.processed_root,
            report_root=run_dir / "evaluation",
            start_year=_start_year(config),
            end_year=_end_year(config),
            requested_labels=config.label_columns,
            requested_factors=resolve_processed_factor_columns(config),
            all_factors=False,
            n_groups=n_groups,
            start_date=config.date_start,
            end_date=config.date_end,
            return_source_column=config.return_source_column,
            cost_bps=config.cost_bps,
            portfolio_modes=(config.portfolio_mode,),
            kernels=(config.kernel,),
            source_alias="pipeline",
            manifest_run_id=run_dir.name,
            diagnostics_spec=DiagnosticsSpec.from_request(
                {
                    "enabled": bool(config.diagnostics),
                    "materialized_views": list(config.diagnostics),
                }
            ),
        )
    )
    label_summary = result.label_summary
    row_count = int(label_summary["rows"].max()) if "rows" in label_summary else None
    return {
        "artifact": "evaluation/manifest.json",
        "row_count": row_count,
        "evaluation_artifact_path": result.manifest_path,
    }


def _require_zero_exit(step: str, code: int | None) -> None:
    if code not in (0, None):
        raise RuntimeError(f"{step} returned nonzero exit code")


def _call_main_sanitized(
    function: Callable[[list[str]], int | None],
    argv: list[str],
) -> int | None:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return function(argv)


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
        "start_from": config.start_from,
        "date_start": config.date_start,
        "date_end": config.date_end,
        "kernel": config.kernel,
        "duration_seconds": _elapsed(started),
        "warnings": warnings,
        "failed_step": failed_step,
        "failure_code": failure_code,
        "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
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
            [asdict(checkpoint) for checkpoint in checkpoints],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _blocking_issue_message(issues: list[ValidationIssue]) -> str:
    codes = ", ".join(issue.code for issue in issues)
    return f"blocking config issues prevent pipeline execution: {codes}"


def _start_year(config: PipelineConfig) -> int:
    return int(config.date_start[:4])


def _end_year(config: PipelineConfig) -> int:
    return int(config.date_end[:4])


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
    "PipelineRunResult",
    "PipelineRunnerError",
    "resolve_processed_factor_columns",
    "run_a_share_pipeline",
]
