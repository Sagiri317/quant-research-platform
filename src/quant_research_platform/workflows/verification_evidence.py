"""Minimal removable verification evidence verification harness."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.catalog.profiles import ResearchProfile
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.reports.verification_evidence import (
    VerificationEvidenceLayout,
    build_validation_summary,
    build_verification_evidence_summary,
    write_run_summary_json,
    write_run_summary_md,
    write_validation_summary,
    write_verification_evidence_summary_json,
    write_verification_evidence_summary_md,
)
from quant_research_platform.research.datasets import (
    assemble_research_dataset,
)
from quant_research_platform.research.evaluations.runner import (
    run_evaluation_from_research_dataset,
)
from quant_research_platform.research.predictions.handoff import (
    run_prediction_from_research_dataset,
)
from quant_research_platform.runtime.execution_metrics import (
    ExecutionMetricsRecorder,
    write_execution_metrics_csv,
    write_execution_metrics_json,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.atomic import atomic_write_json
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_layer_manifest,
    write_run_manifest,
)


@dataclass(frozen=True)
class VerificationEvidenceRunConfig:
    output_root: Path
    start_date: str
    end_date: str
    profile: ResearchProfile
    evaluation_start_date: str | None = None
    evaluation_end_date: str | None = None
    prediction_train_end: str | None = None
    prediction_test_start: str | None = None
    prediction_test_end: str | None = None
    replace_output: bool = False
    run_id: str | None = None


@dataclass(frozen=True)
class VerificationEvidenceRunResult:
    status: str
    run_id: str
    output_root: str
    summary: dict[str, Any]


_OWNERSHIP_MARKER_FILENAME = ".qrp-verification-evidence.json"
_OWNERSHIP_MARKER = "qrp_verification_evidence_workflow"
_OWNERSHIP_CONTRACT_VERSION = "0.1"


def run_verification_evidence(
    config: VerificationEvidenceRunConfig,
) -> VerificationEvidenceRunResult:
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    if start > end:
        raise ValueError("start_date_after_end_date")
    effective_windows = _resolve_effective_windows(config)
    output_root = Path(config.output_root)
    _prepare_output_root(output_root, replace_output=config.replace_output)
    run_id = config.run_id or output_root.name
    layout = VerificationEvidenceLayout.from_output_root(output_root)
    _create_layout_directories(layout)
    resource_policy = RuntimeResourcePolicy.from_environment()
    atomic_write_json(
        layout.config_root / "run_config.json",
        {
            "run_id": run_id,
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
            "output_root": _logical_path(output_root),
            "replace_output": config.replace_output,
            "runtime_resource_policy": resource_policy.to_payload(),
            "workflow_id": _OWNERSHIP_MARKER,
            "contract_version": _OWNERSHIP_CONTRACT_VERSION,
            "created_at": _utc_now(),
        },
    )

    policy_scope = runtime_resource_policy_scope(resource_policy)
    policy_scope.__enter__()
    try:
        return _run_verification_evidence_with_policy(
            config=config,
            effective_windows=effective_windows,
            output_root=output_root,
            run_id=run_id,
            layout=layout,
            resource_policy=resource_policy,
        )
    finally:
        policy_scope.__exit__(None, None, None)


def _run_verification_evidence_with_policy(
    *,
    config: VerificationEvidenceRunConfig,
    effective_windows: dict[str, str | None],
    output_root: Path,
    run_id: str,
    layout: VerificationEvidenceLayout,
    resource_policy: RuntimeResourcePolicy,
) -> VerificationEvidenceRunResult:
    recorder = ExecutionMetricsRecorder()
    storage_layout = StorageLayout(project_root=output_root)
    time_window = TimeWindow(config.start_date, config.end_date)
    evaluation_time_window = TimeWindow(
        effective_windows["evaluation_start_date"],
        effective_windows["evaluation_end_date"],
    )
    catalogs = config.profile
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
            dataset_id="a_share_price_volume_forward_returns",
            time_window=time_window,
            resource_policy=resource_policy,
        )
    with recorder.record("evaluation_generation", logical_layer="evidence"):
        evaluation_root = run_evaluation_from_research_dataset(
            layout=storage_layout,
            dataset_catalog=catalogs.research_datasets,
            evaluation_catalog=catalogs.evaluations,
            run_id="a_share_mom_20d_forward_return_20d_2014",
            time_window=evaluation_time_window,
        )
    with recorder.record("prediction_generation", logical_layer="evidence"):
        prediction_root = run_prediction_from_research_dataset(
            layout=storage_layout,
            dataset_catalog=catalogs.research_datasets,
            model_catalog=catalogs.models,
            prediction_catalog=catalogs.predictions,
            prediction_id="a_share_linear_mom_20d_prediction_2014",
            time_window=time_window,
            prediction_train_end=effective_windows["prediction_train_end"],
            prediction_test_start=effective_windows["prediction_test_start"],
            prediction_test_end=effective_windows["prediction_test_end"],
        )

    dataset_roots = _dataset_roots(
        layout,
        interim_roots=tuple(interim_result.artifact_roots.values()),
        processed_roots=tuple(item.root for item in processed_result.values()),
        research_roots=(Path(research_root),),
    )
    report_roots = (Path(evaluation_root), Path(prediction_root))
    layer_manifest_paths = (
        write_layer_manifest(
            layer_root=layout.interim_root,
            logical_layer="interim",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "interim"
            ),
        ),
        write_layer_manifest(
            layer_root=layout.processed_root,
            logical_layer="processed",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "processed"
            ),
        ),
        write_layer_manifest(
            layer_root=layout.research_root,
            logical_layer="research",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "research"
            ),
        ),
        write_layer_manifest(
            layer_root=layout.evidence_root,
            logical_layer="evidence",
            run_id=run_id,
            dataset_roots=(),
        ),
    )
    with recorder.record("manifest_generation") as record:
        run_manifest_path = write_run_manifest(
            manifests_root=layout.manifests_root,
            run_id=run_id,
            output_root=output_root,
            layer_manifest_paths=layer_manifest_paths,
        )
        record.file_count = 1
    run_manifest = read_manifest(run_manifest_path)
    timings_json = layout.timings_root / "timings.json"
    generated_paths = (
        layout.config_root / "run_config.json",
        run_manifest_path,
        *layer_manifest_paths,
        timings_json,
    )
    with recorder.record("validation_summary"):
        validation_summary = build_validation_summary(
            run_id=run_id,
            output_root=output_root,
            dataset_roots=dataset_roots,
            layer_manifest_paths=layer_manifest_paths,
            generated_text_paths=generated_paths,
        )
    write_validation_summary(
        layout.validations_root / "validation_summary.json",
        validation_summary,
    )
    timings_json = write_execution_metrics_json(
        timings_json,
        recorder.records,
        run_id=run_id,
        memory_sampler_unavailable=recorder.memory_sampler.unavailable,
    )
    write_execution_metrics_csv(layout.timings_root / "timings.csv", recorder.records)
    timings_payload = read_manifest(timings_json)
    evidence_summary = build_verification_evidence_summary(
        run_id=run_id,
        date_range={"start_date": config.start_date, "end_date": config.end_date},
        manifests={
            "dataset_count": len(dataset_roots),
            "layer_count": len(layer_manifest_paths),
            "run_validation_status": run_manifest["run_validation_status"],
            "report_count": len(report_roots),
        },
        timings=timings_payload,
        validation_summary=validation_summary,
        output_root=output_root,
        runtime_resource_policy=resource_policy.to_payload(),
        runtime_execution_plan=_runtime_execution_plan_payload(),
    )
    write_verification_evidence_summary_json(
        layout.evidence_root / "evidence_summary.json",
        evidence_summary,
    )
    write_verification_evidence_summary_md(
        layout.evidence_root / "evidence_summary.md",
        evidence_summary,
    )
    summary = {
        **evidence_summary,
        "validation_summary_path": _logical_path(
            layout.validations_root / "validation_summary.json"
        ),
        "run_manifest_path": _logical_path(run_manifest_path),
    }
    write_run_summary_json(layout.summary_json_path, summary)
    write_run_summary_md(layout.summary_md_path, summary)
    return VerificationEvidenceRunResult(
        status=str(validation_summary["status"]),
        run_id=run_id,
        output_root=output_root.as_posix(),
        summary=summary,
    )


def _prepare_output_root(path: Path, *, replace_output: bool) -> None:
    output_root = Path(path)
    if output_root.exists() and not replace_output:
        raise ValueError("output_root_already_exists")
    if output_root.exists() and replace_output:
        _assert_owned_verification_evidence_output(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_ownership_marker(output_root)


def _resolve_effective_windows(
    config: VerificationEvidenceRunConfig,
) -> dict[str, str | None]:
    evaluation_start_date = config.evaluation_start_date or config.start_date
    evaluation_end_date = config.evaluation_end_date or config.end_date
    evidence_start = pd.Timestamp(config.start_date)
    evidence_end = pd.Timestamp(config.end_date)
    evaluation_start = pd.Timestamp(evaluation_start_date)
    evaluation_end = pd.Timestamp(evaluation_end_date)
    if evaluation_start > evaluation_end:
        raise ValueError("evaluation_start_date_after_end_date")
    if evaluation_start < evidence_start or evaluation_end > evidence_end:
        raise ValueError("evaluation_window_out_of_evidence_range")

    prediction_spec = config.profile.predictions.get(
        "a_share_linear_mom_20d_prediction_2014"
    )
    prediction_train_end = config.prediction_train_end or prediction_spec.train_end
    prediction_test_start = (
        config.prediction_test_start or prediction_spec.test_start
    )
    prediction_test_end = (
        config.prediction_test_end
        if config.prediction_test_end is not None
        else prediction_spec.test_end
    )
    train_end = pd.Timestamp(prediction_train_end)
    test_start = pd.Timestamp(prediction_test_start)
    test_end = (
        pd.Timestamp(prediction_test_end)
        if prediction_test_end is not None
        else None
    )
    if (
        train_end < evidence_start
        or train_end > evidence_end
        or test_start < evidence_start
        or test_start > evidence_end
        or (test_end is not None and test_end > evidence_end)
    ):
        raise ValueError("prediction_window_out_of_evidence_range")
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


def _runtime_execution_plan_payload() -> dict[str, object]:
    return {
        "storage_read": {
            "read_context": "_DatasetReadContext",
            "time_window_year_partition_pruning": True,
            "chunked_read_granularity": "parquet_file",
            "row_group_streaming": "NOT_LANDED",
        },
        "timestamp_sessions": {
            "evaluation": "complete_timestamp_sessions",
            "model": "complete_timestamp_sessions",
            "prediction": "complete_timestamp_sessions",
        },
        "report_writing": {
            "prediction_parquet": "chunked_internal",
            "evaluation_parquet": "materialized",
        },
        "full_horizon_tiling": {
            "status": "NOT_LANDED",
            "reason": "no_external_time_parameters_added",
        },
    }


def _ownership_marker_path(output_root: Path) -> Path:
    return Path(output_root) / _OWNERSHIP_MARKER_FILENAME


def _write_ownership_marker(output_root: Path) -> Path:
    return atomic_write_json(
        _ownership_marker_path(output_root),
        {
            "workflow_id": _OWNERSHIP_MARKER,
            "contract_version": _OWNERSHIP_CONTRACT_VERSION,
            "created_at": _utc_now(),
        },
    )


def _assert_owned_verification_evidence_output(output_root: Path) -> None:
    marker_path = _ownership_marker_path(output_root)
    if not marker_path.is_file():
        raise ValueError("output_root_not_owned_by_verification_evidence_workflow")
    marker = read_manifest(marker_path)
    if (
        marker.get("workflow_id") != _OWNERSHIP_MARKER
        or marker.get("contract_version") != _OWNERSHIP_CONTRACT_VERSION
    ):
        raise ValueError("output_root_not_owned_by_verification_evidence_workflow")


def _create_layout_directories(layout: VerificationEvidenceLayout) -> None:
    for root in (
        layout.config_root,
        layout.interim_root,
        layout.processed_root,
        layout.research_root,
        layout.evidence_root,
        layout.manifests_root,
        layout.validations_root,
        layout.timings_root,
        layout.logs_root,
    ):
        root.mkdir(parents=True, exist_ok=True)


def _dataset_roots(
    layout: VerificationEvidenceLayout,
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


def _logical_layer(dataset_root: Path, output_root: Path) -> str:
    relative = dataset_root.relative_to(output_root).as_posix()
    if relative.startswith("data/interim/"):
        return "interim"
    if relative.startswith("data/research/"):
        return "research"
    if relative.startswith("reports/") or relative.startswith("data/evidence/"):
        return "evidence"
    return "processed"


def _layer_for_root(dataset_root: Path, layout: VerificationEvidenceLayout) -> str:
    return _logical_layer(dataset_root, layout.output_root)


def _dataset_id(dataset_root: Path) -> str:
    return dataset_root.name


def _logical_path(path: Path) -> str:
    item = Path(path)
    parts = item.parts
    for marker in ("data", "reports", "manifests", "validations", "timings"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :]).as_posix()
    if item.is_absolute():
        return item.name
    return item.as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "VerificationEvidenceRunConfig",
    "VerificationEvidenceRunResult",
    "run_verification_evidence",
]
