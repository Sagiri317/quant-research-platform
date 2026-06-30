"""Local verification workflow orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.artifacts.evidence import (
    write_json_evidence,
    write_run_summary_markdown,
)
from quant_research_platform.artifacts.outputs import (
    write_evaluation_outputs,
    write_portfolio_outputs,
    write_prediction_outputs,
)
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.check_status import (
    overall_verification_status,
    validate_verification_statuses,
)
from quant_research_platform.policy.share_safety import local_run_root_reason
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.reports.verification_summary import (
    build_validation_summary,
    render_run_summary_markdown,
)
from quant_research_platform.research.datasets import (
    assemble_research_dataset,
)
from quant_research_platform.research.evaluations.runner import (
    build_evaluation_result_from_research_dataset,
)
from quant_research_platform.research.portfolios.runner import (
    build_portfolio_result_from_research_dataset,
)
from quant_research_platform.research.predictions.outputs import PredictionResult
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.execution_metrics import ExecutionMetricsRecorder
from quant_research_platform.runtime.logical_paths import logical_path
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    build_layer_summary,
    build_run_data_summary,
    read_qrp_json,
    replace_qrp_json,
)
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)


@dataclass(frozen=True)
class LocalVerificationRunConfig:
    run_root: Path
    start_date: str
    end_date: str
    profile: ResearchProfile
    prediction_train_end: str
    prediction_test_start: str
    portfolio_frequency: str
    portfolio_execution_lag_steps: int
    evaluation_start_date: str | None = None
    evaluation_end_date: str | None = None
    prediction_test_end: str | None = None
    replace_output: bool = False
    run_id: str | None = None


@dataclass(frozen=True)
class LocalVerificationRunResult:
    status: str
    run_id: str
    run_root: str
    summary: dict[str, Any]


_RUN_SCHEMA_ID = "qrp-run"
_WORKFLOW_ID = "local_verification"


def run_local_verification(
    config: LocalVerificationRunConfig,
) -> LocalVerificationRunResult:
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    if start > end:
        raise ValueError("start_date_after_end_date")
    effective_windows = _resolve_effective_windows(config)
    run_root = Path(config.run_root).resolve(strict=False)
    run_id = resolve_local_verification_run_id(config.run_id, run_root)
    _prepare_run_root(
        run_root,
        run_id=run_id,
        replace_output=config.replace_output,
    )
    layout = LocalVerificationLayout.from_run_root(run_root)
    _create_layout_directories(layout)
    resource_policy = RuntimeResourcePolicy.from_environment()

    with runtime_resource_policy_scope(resource_policy):
        return _run_local_verification_with_policy(
            config=config,
            effective_windows=effective_windows,
            run_root=run_root,
            run_id=run_id,
            layout=layout,
            resource_policy=resource_policy,
        )


def _run_local_verification_with_policy(
    *,
    config: LocalVerificationRunConfig,
    effective_windows: dict[str, str | None],
    run_root: Path,
    run_id: str,
    layout: LocalVerificationLayout,
    resource_policy: RuntimeResourcePolicy,
) -> LocalVerificationRunResult:
    recorder = ExecutionMetricsRecorder()
    storage_layout = StorageLayout(project_root=run_root)
    time_window = TimeWindow(config.start_date, config.end_date)
    evaluation_time_window = TimeWindow(
        effective_windows["evaluation_start_date"],
        effective_windows["evaluation_end_date"],
    )
    catalogs = config.profile
    prediction_catalog = config.profile.build_prediction_catalog(
        train_end=effective_windows["prediction_train_end"],
        test_start=effective_windows["prediction_test_start"],
    )
    portfolio_catalog = config.profile.build_portfolio_catalog(
        rebalance_frequency=config.portfolio_frequency,
        execution_lag_steps=config.portfolio_execution_lag_steps,
    )
    source_id = config.profile.source_id
    with recorder.record("interim_generation", logical_layer="interim"):
        interim_result = config.profile.materialize_interim_data(
            storage_layout,
            Path("data") / "raw" / str(source_id),
            config.start_date,
            config.end_date,
        )
    with recorder.record("processed_generation", logical_layer="processed") as record:
        processed_result = config.profile.materialize_processed_data(
            storage_layout,
            source_id,
            time_window,
            frozenset(
                {
                    ProcessedDataKind.PANEL,
                    ProcessedDataKind.ANNOTATION,
                    ProcessedDataKind.FACTOR,
                    ProcessedDataKind.LABEL,
                }
            ),
        )
        record.output_rows = sum(item.row_count for item in processed_result.values())
    with recorder.record("research_generation", logical_layer="research"):
        research_root = assemble_research_dataset(
            layout=storage_layout,
            dataset_catalog=catalogs.research_datasets,
            processed_catalog=catalogs.processed,
            dataset_id="price_volume_forward_returns",
            time_window=time_window,
            resource_policy=resource_policy,
        )
    with recorder.record("evaluation_generation"):
        evaluation_result = build_evaluation_result_from_research_dataset(
            layout=storage_layout,
            dataset_catalog=catalogs.research_datasets,
            evaluation_catalog=catalogs.evaluations,
            run_id="signal_forward_return_20d",
            time_window=evaluation_time_window,
        )
        evaluation_root = write_evaluation_outputs(
            evaluation_result.metrics,
            evaluation_result.grouped_label_values,
            ic_summary=evaluation_result.ic_summary,
            group_spread_summary=evaluation_result.group_spread_summary,
            evaluation_diagnostics=evaluation_result.diagnostics,
            spec=evaluation_result.spec,
            output_spec=_output_spec(
                output_id=evaluation_result.spec.run_id,
                output_definition_id="evaluation_outputs",
                run_id=run_id,
                market_id=evaluation_result.spec.market_id,
                run_type=evaluation_result.spec.run_type,
            ),
            layout=storage_layout,
        )
    with recorder.record("prediction_generation"):
        prediction_roots = tuple(
            _write_prediction_result(
                build_prediction_result_from_research_dataset(
                    layout=storage_layout,
                    dataset_catalog=catalogs.research_datasets,
                    model_catalog=catalogs.models,
                    prediction_catalog=prediction_catalog,
                    prediction_id=prediction_spec.prediction_id,
                    time_window=time_window,
                    prediction_train_end=effective_windows["prediction_train_end"],
                    prediction_test_start=effective_windows["prediction_test_start"],
                    prediction_test_end=effective_windows["prediction_test_end"],
                ),
                layout=storage_layout,
                run_id=run_id,
            )
            for prediction_spec in prediction_catalog.list()
        )
    with recorder.record("portfolio_generation"):
        portfolio_result = build_portfolio_result_from_research_dataset(
            layout=storage_layout,
            dataset_catalog=catalogs.research_datasets,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="signal_forward_return_20d_portfolio",
            time_window=evaluation_time_window,
        )
        portfolio_root = write_portfolio_outputs(
            output_chunks=portfolio_result.output_chunks,
            summary_factory=portfolio_result.summary_factory,
            spec=portfolio_result.spec,
            output_spec=_output_spec(
                output_id=portfolio_result.spec.portfolio_id,
                output_definition_id="portfolio_outputs",
                run_id=run_id,
                market_id=portfolio_result.spec.market_id,
                run_type=portfolio_result.spec.run_type,
            ),
            layout=storage_layout,
        )

    dataset_roots = _dataset_roots(
        layout,
        interim_roots=tuple(interim_result.artifact_roots.values()),
        processed_roots=tuple(item.root for item in processed_result.values()),
        research_roots=(Path(research_root),),
    )
    output_bundles = (
        Path(evaluation_root),
        *(Path(prediction_root) for prediction_root in prediction_roots),
        Path(portfolio_root),
    )
    layer_summaries = _build_layer_summaries(
        layout=layout,
        dataset_roots=dataset_roots,
    )
    with recorder.record("validation_summary"):
        validation_summary = build_validation_summary(
            run_id=run_id,
            dataset_roots=dataset_roots,
            layer_summaries=layer_summaries,
            generated_text_paths=(),
        )
    statuses = validate_verification_statuses(
        smoke_status=str(validation_summary.get("status", "FAIL")),
        compare_status="NOT_RUN",
        full_horizon_status="NOT_RUN",
        compare_evidence=None,
        full_horizon_evidence=None,
    )
    status_payload = {
        "overall": overall_verification_status(
            validation_summary.get("status", "FAIL"),
            statuses,
        ),
        "validation": str(validation_summary.get("status", "FAIL")),
        "smoke": statuses["smoke_status"],
        "compare": statuses["compare_status"],
        "share_safety": str(
            validation_summary.get("share_safety_scan", {}).get("status", "FAIL")
        ),
        "full_horizon": statuses["full_horizon_status"],
        "validation_matrix": statuses["validation_matrix_status"],
    }
    seed_payload = _read_run_json(run_root)
    run_payload = {
        **seed_payload,
        "completed_at": _utc_now(),
        "status": status_payload,
        "code": {},
        "config": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "requested_windows": {
                "evaluation_start_date": config.evaluation_start_date,
                "evaluation_end_date": config.evaluation_end_date,
                "prediction_train_end": config.prediction_train_end,
                "prediction_test_start": config.prediction_test_start,
                "prediction_test_end": config.prediction_test_end,
            },
            "effective_windows": effective_windows,
            "replace_output": config.replace_output,
            "runtime_resource_policy": resource_policy.to_payload(),
        },
        "inputs": {
            "date_range": {
                "start_date": config.start_date,
                "end_date": config.end_date,
            }
        },
        "data": build_run_data_summary(layer_summaries=layer_summaries),
        "outputs": [_output_index(root, run_root) for root in output_bundles],
        "checks": {"validation": validation_summary},
        "execution_metrics": recorder.payload(run_id=run_id),
        "logs": {},
        "summary": {"markdown_path": "summary.md"},
    }
    summary_markdown = render_run_summary_markdown(run_payload)
    replace_qrp_json(layout.run_json_path, run_payload)
    write_run_summary_markdown(layout.summary_md_path, summary_markdown)
    return LocalVerificationRunResult(
        status=status_payload["overall"],
        run_id=run_id,
        run_root=_logical_run_root(run_root),
        summary=run_payload,
    )


def _write_prediction_result(
    result: PredictionResult,
    *,
    layout: StorageLayout,
    run_id: str,
) -> Path:
    return write_prediction_outputs(
        spec=result.spec,
        output_spec=_output_spec(
            output_id=result.spec.prediction_id,
            output_definition_id="prediction_outputs",
            run_id=run_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )


def _output_spec(
    *,
    output_id: str,
    output_definition_id: str,
    run_id: str,
    run_type: str,
    market_id: str = "a_share",
) -> OutputSpec:
    return OutputSpec(
        output_id=output_id,
        output_definition_id=output_definition_id,
        run_id=run_id,
        market_id=market_id,
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def resolve_local_verification_run_id(
    config_run_id: str | None,
    run_root: Path,
) -> str:
    run_id = config_run_id if config_run_id is not None else Path(run_root).name
    return validate_run_id(run_id)


def _logical_run_root(run_root: Path) -> str:
    return Path(run_root).name


def _resolve_local_output_repo_root(
    run_root: Path,
    explicit_repo_root: Path | None,
) -> Path:
    if explicit_repo_root is not None:
        return Path(explicit_repo_root)
    for candidate in (
        _find_git_root(Path.cwd().resolve(strict=False)),
        _find_git_root(Path(__file__).resolve(strict=False)),
    ):
        if candidate is not None:
            return candidate
    return Path.cwd()


def _find_git_root(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _prepare_run_root(
    path: Path,
    *,
    run_id: str,
    replace_output: bool,
    repo_root: Path | None = None,
) -> None:
    run_root = Path(path)
    reason = local_run_root_reason(
        run_root,
        _resolve_local_output_repo_root(run_root, repo_root),
    )
    if reason is not None:
        raise ValueError(reason)
    if run_root.exists():
        entries = tuple(run_root.iterdir())
        if entries and not replace_output:
            raise ValueError("run_root_already_exists")
        if entries:
            _assert_owned_local_verification_run_root(run_root)
            shutil.rmtree(run_root)
            run_root.mkdir(parents=True, exist_ok=False)
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    _write_seed_run_json(run_root, run_id=run_id)


def _resolve_effective_windows(
    config: LocalVerificationRunConfig,
) -> dict[str, str | None]:
    evaluation_start_date = config.evaluation_start_date or config.start_date
    evaluation_end_date = config.evaluation_end_date or config.end_date
    run_start = pd.Timestamp(config.start_date)
    run_end = pd.Timestamp(config.end_date)
    evaluation_start = pd.Timestamp(evaluation_start_date)
    evaluation_end = pd.Timestamp(evaluation_end_date)
    if evaluation_start > evaluation_end:
        raise ValueError("evaluation_start_date_after_end_date")
    if evaluation_start < run_start or evaluation_end > run_end:
        raise ValueError("evaluation_window_out_of_run_range")

    prediction_train_end = config.prediction_train_end
    prediction_test_start = config.prediction_test_start
    prediction_test_end = config.prediction_test_end
    train_end = pd.Timestamp(prediction_train_end)
    test_start = pd.Timestamp(prediction_test_start)
    test_end = (
        pd.Timestamp(prediction_test_end) if prediction_test_end is not None else None
    )
    if (
        train_end < run_start
        or train_end > run_end
        or test_start < run_start
        or test_start > run_end
        or (test_end is not None and test_end > run_end)
    ):
        raise ValueError("prediction_window_out_of_run_range")
    if train_end >= test_start:
        raise ValueError("prediction_train_end_must_precede_test_start")
    if test_end is not None and test_end < test_start:
        raise ValueError("prediction_test_end_must_not_precede_test_start")
    return {
        "evaluation_start_date": evaluation_start_date,
        "evaluation_end_date": evaluation_end_date,
        "prediction_train_end": prediction_train_end,
        "prediction_test_start": prediction_test_start,
        "prediction_test_end": prediction_test_end,
    }


def _run_json_path(run_root: Path) -> Path:
    return Path(run_root) / "run.json"


def _write_seed_run_json(run_root: Path, *, run_id: str) -> Path:
    created_at = _utc_now()
    return write_json_evidence(
        _run_json_path(run_root),
        {
            "schema_id": _RUN_SCHEMA_ID,
            "workflow_id": _WORKFLOW_ID,
            "run_id": run_id,
            "created_at": created_at,
            "root_ownership": {
                "owner": _WORKFLOW_ID,
                "workflow_id": _WORKFLOW_ID,
                "created_by": "quant_research_platform",
                "created_at": created_at,
                "replace_safe": True,
            },
        },
    )


def _read_run_json(run_root: Path) -> dict[str, Any]:
    path = _run_json_path(run_root)
    if not path.is_file():
        raise ValueError("run_root_not_owned_by_local_verification_workflow")
    try:
        payload = read_qrp_json(
            path,
            expected_schema_id=_RUN_SCHEMA_ID,
            error="run_root_not_owned_by_local_verification_workflow",
        )
    except ValueError:
        raise ValueError("run_root_not_owned_by_local_verification_workflow") from None
    return payload


def _assert_owned_local_verification_run_root(run_root: Path) -> None:
    payload = _read_run_json(run_root)
    ownership = payload.get("root_ownership")
    if (
        payload.get("schema_id") != _RUN_SCHEMA_ID
        or not isinstance(ownership, dict)
        or ownership.get("owner") != _WORKFLOW_ID
        or ownership.get("workflow_id") != _WORKFLOW_ID
        or ownership.get("replace_safe") is not True
    ):
        raise ValueError("run_root_not_owned_by_local_verification_workflow")


def _create_layout_directories(layout: LocalVerificationLayout) -> None:
    for root in (
        layout.interim_root,
        layout.processed_root,
        layout.research_root,
        layout.outputs_root,
        layout.logs_root,
    ):
        root.mkdir(parents=True, exist_ok=True)


def _dataset_roots(
    layout: LocalVerificationLayout,
    *,
    interim_roots: tuple[Path, ...] = (),
    processed_roots: tuple[Path, ...] = (),
    research_roots: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    del layout
    roots = {
        Path(root)
        for root in (*interim_roots, *processed_roots, *research_roots)
        if root is not None
    }
    return tuple(sorted(roots, key=lambda item: item.as_posix()))


def _logical_layer(dataset_root: Path, run_root: Path) -> str:
    relative = logical_path(dataset_root, run_root=run_root)
    if relative.startswith("data/interim/"):
        return "interim"
    if relative.startswith("data/research/"):
        return "research"
    return "processed"


def _layer_for_root(dataset_root: Path, layout: LocalVerificationLayout) -> str:
    return _logical_layer(dataset_root, layout.run_root)


def _build_layer_summaries(
    *,
    layout: LocalVerificationLayout,
    dataset_roots: tuple[Path, ...],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Path]] = {}
    for root in dataset_roots:
        grouped.setdefault(_layer_for_root(root, layout), []).append(root)
    return tuple(
        build_layer_summary(
            layer_root=layout.data_root / logical_layer,
            logical_layer=logical_layer,
            dataset_roots=tuple(sorted(roots, key=lambda item: item.as_posix())),
        )
        for logical_layer, roots in sorted(grouped.items())
    )


def _output_index(output_bundle: Path, run_root: Path) -> dict[str, Any]:
    manifest_path = Path(output_bundle) / "manifest.json"
    payload = read_qrp_json(
        manifest_path,
        expected_schema_id="qrp-output-manifest",
        error="output_manifest_schema_id_invalid",
    )
    return {
        **payload,
        "manifest_path": manifest_path.relative_to(run_root).as_posix(),
    }


def _dataset_id(dataset_root: Path) -> str:
    return dataset_root.name


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "LocalVerificationRunConfig",
    "LocalVerificationRunResult",
    "run_local_verification",
]
