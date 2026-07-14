"""Local verification workflow orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from quant_research_platform.artifacts._output_bundle_validation import (
    validate_output_bundle,
)
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
    require_status,
    validate_verification_statuses,
)
from quant_research_platform.policy.share_safety import local_run_root_reason
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.reports.verification_summary import (
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
from quant_research_platform.runtime.logical_paths import (
    RUN_RESULT_DATASET_LAYERS,
    dataset_layer_from_path,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    build_layer_summary,
    build_run_data_summary,
    read_qrp_json,
    replace_qrp_json,
)
from quant_research_platform.workflows.diagnostics.schema_inventory import (
    build_schema_inventory_payload,
    summarize_schema_inventory_payload,
)
from quant_research_platform.workflows.diagnostics.validation_summary import (
    build_validation_summary,
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
_SEED_RUN_KEYS = frozenset(
    {"schema_id", "workflow_id", "run_id", "created_at", "root_ownership"}
)
_ROOT_OWNERSHIP_KEYS = frozenset(
    {"owner", "workflow_id", "created_by", "created_at", "replace_safe"}
)
_PROFILE_PREDICTION_PREVIEW_TRAIN_END = "2000-01-01"
_PROFILE_PREDICTION_PREVIEW_TEST_START = "2000-01-02"


def run_local_verification(
    config: LocalVerificationRunConfig,
) -> LocalVerificationRunResult:
    start = _local_verification_bound(config.start_date)
    end = _local_verification_bound(config.end_date)
    if start > end:
        raise ValueError("start_date_after_end_date")
    effective_windows = _resolve_effective_windows(config)
    run_root = Path(config.run_root).resolve(strict=False)
    run_id = resolve_local_verification_run_id(config.run_id, run_root)
    resource_policy = RuntimeResourcePolicy.from_environment()
    require_free_disk_headroom(
        resource_policy,
        run_root,
        reserve_bytes=resource_policy.effective_chunk_bytes * 2,
    )
    _prepare_run_root(
        run_root,
        run_id=run_id,
        replace_output=config.replace_output,
    )
    layout = LocalVerificationLayout.from_run_root(run_root)
    _create_layout_directories(layout)

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
    research_dataset_id = _only_research_dataset_id(config.profile)
    evaluation_run_id = _only_evaluation_run_id(config.profile)
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
            dataset_id=research_dataset_id,
            time_window=time_window,
            resource_policy=resource_policy,
        )
    with recorder.record("evaluation_generation"):
        evaluation_result = build_evaluation_result_from_research_dataset(
            layout=storage_layout,
            dataset_catalog=catalogs.research_datasets,
            evaluation_catalog=catalogs.evaluations,
            run_id=evaluation_run_id,
            time_window=evaluation_time_window,
        )
        _require_output_disk_headroom(resource_policy, run_root)
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
        prediction_roots_list: list[Path] = []
        for prediction_spec in _optional_prediction_specs(prediction_catalog):
            prediction_result = build_prediction_result_from_research_dataset(
                layout=storage_layout,
                dataset_catalog=catalogs.research_datasets,
                model_catalog=catalogs.models,
                prediction_catalog=prediction_catalog,
                prediction_id=prediction_spec.prediction_id,
                time_window=time_window,
                prediction_train_end=effective_windows["prediction_train_end"],
                prediction_test_start=effective_windows["prediction_test_start"],
                prediction_test_end=effective_windows["prediction_test_end"],
            )
            _require_output_disk_headroom(resource_policy, run_root)
            prediction_roots_list.append(
                _write_prediction_result(
                    prediction_result,
                    layout=storage_layout,
                    run_id=run_id,
                    market_id=catalogs.research_datasets.get(
                        prediction_spec.research_dataset_id
                    ).market_id,
                )
            )
        prediction_roots = tuple(prediction_roots_list)
    with recorder.record("portfolio_generation"):
        portfolio_roots_list: list[Path] = []
        for portfolio_id in _optional_portfolio_ids(portfolio_catalog):
            portfolio_result = build_portfolio_result_from_research_dataset(
                layout=storage_layout,
                dataset_catalog=catalogs.research_datasets,
                portfolio_catalog=portfolio_catalog,
                portfolio_id=portfolio_id,
                time_window=evaluation_time_window,
            )
            _require_output_disk_headroom(resource_policy, run_root)
            portfolio_roots_list.append(
                write_portfolio_outputs(
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
            )
        portfolio_roots = tuple(portfolio_roots_list)

    dataset_roots = _dataset_roots(
        layout,
        interim_roots=tuple(interim_result.artifact_roots.values()),
        processed_roots=tuple(item.root for item in processed_result.values()),
        research_roots=(Path(research_root),),
    )
    output_bundles = (
        Path(evaluation_root),
        *(Path(prediction_root) for prediction_root in prediction_roots),
        *(Path(portfolio_root) for portfolio_root in portfolio_roots),
    )
    output_indexes = tuple(
        _output_index(root, run_root, expected_run_id=run_id)
        for root in output_bundles
    )
    layer_summaries = _build_layer_summaries(
        layout=layout,
        dataset_roots=dataset_roots,
    )
    with recorder.record("validation_summary"):
        schema_inventory = build_schema_inventory_payload(
            run_root,
            important_null_count_columns=(
                config.profile.schema_inventory_important_columns()
            ),
        )
        schema_inventory_summary = summarize_schema_inventory_payload(schema_inventory)
        validation_summary = build_validation_summary(
            run_id=run_id,
            dataset_roots=dataset_roots,
            layer_summaries=layer_summaries,
            generated_text_paths=_generated_text_paths(
                output_bundles,
                output_indexes=output_indexes,
            ),
        )
    validation_status = require_status(validation_summary.get("status"))
    share_safety_scan = validation_summary.get("share_safety_scan")
    if type(share_safety_scan) is not dict:
        raise ValueError("validation_summary_share_safety_invalid")
    share_safety_status = require_status(share_safety_scan.get("status"))
    statuses = validate_verification_statuses(
        smoke_status=validation_status,
        compare_status="NOT_RUN",
        full_horizon_status="NOT_RUN",
        compare_evidence=None,
        full_horizon_evidence=None,
    )
    status_payload = {
        "overall": overall_verification_status(
            validation_status,
            statuses,
        ),
        "validation": validation_status,
        "smoke": statuses["smoke_status"],
        "compare": statuses["compare_status"],
        "share_safety": share_safety_status,
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
        "outputs": list(output_indexes),
        "checks": {
            "schema_inventory": schema_inventory_summary,
            "validation": validation_summary,
        },
        "execution_metrics": recorder.payload(run_id=run_id),
        "logs": {},
        "summary": {"markdown_path": "summary.md"},
    }
    summary_markdown = render_run_summary_markdown(run_payload)
    _require_output_disk_headroom(resource_policy, run_root)
    write_run_summary_markdown(layout.summary_md_path, summary_markdown)
    replace_qrp_json(layout.run_json_path, run_payload)
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
    market_id: str,
) -> Path:
    return write_prediction_outputs(
        spec=result.spec,
        output_spec=_output_spec(
            output_id=result.spec.prediction_id,
            output_definition_id="prediction_outputs",
            run_id=run_id,
            market_id=market_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )


def _require_output_disk_headroom(
    resource_policy: RuntimeResourcePolicy,
    run_root: Path,
) -> None:
    require_free_disk_headroom(
        resource_policy,
        run_root,
        reserve_bytes=resource_policy.effective_chunk_bytes * 2,
    )


def _output_spec(
    *,
    output_id: str,
    output_definition_id: str,
    run_id: str,
    run_type: str,
    market_id: str,
) -> OutputSpec:
    return OutputSpec(
        output_id=output_id,
        output_definition_id=output_definition_id,
        run_id=run_id,
        market_id=market_id,
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _only_research_dataset_id(profile: ResearchProfile) -> str:
    specs = profile.research_datasets.list()
    if len(specs) != 1:
        raise ValueError("ambiguous_research_dataset")
    return specs[0].data_id


def _only_evaluation_run_id(profile: ResearchProfile) -> str:
    specs = profile.evaluations.list()
    if len(specs) != 1:
        raise ValueError("ambiguous_evaluation_run")
    return specs[0].run_id


def _optional_prediction_specs(prediction_catalog) -> tuple[object, ...]:
    return prediction_catalog.list()


def _optional_portfolio_ids(portfolio_catalog) -> tuple[str, ...]:
    specs = portfolio_catalog.list()
    if not specs:
        return ()
    if len(specs) != 1:
        raise ValueError("ambiguous_portfolio")
    return (specs[0].portfolio_id,)


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
    run_start = _local_verification_bound(config.start_date)
    run_end = _local_verification_bound(config.end_date)
    evaluation_start = _local_verification_bound(evaluation_start_date)
    evaluation_end = _local_verification_bound(evaluation_end_date)
    if evaluation_start > evaluation_end:
        raise ValueError("evaluation_start_date_after_end_date")
    if evaluation_start < run_start or evaluation_end > run_end:
        raise ValueError("evaluation_window_out_of_run_range")
    if not _profile_builds_prediction_specs(config.profile):
        return {
            "evaluation_start_date": evaluation_start_date,
            "evaluation_end_date": evaluation_end_date,
            "prediction_train_end": None,
            "prediction_test_start": None,
            "prediction_test_end": None,
        }

    prediction_train_end = config.prediction_train_end
    prediction_test_start = config.prediction_test_start
    prediction_test_end = config.prediction_test_end
    train_end = _local_verification_bound(prediction_train_end)
    test_start = _local_verification_bound(prediction_test_start)
    test_end = (
        _local_verification_bound(prediction_test_end)
        if prediction_test_end is not None
        else None
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


def _profile_builds_prediction_specs(profile: ResearchProfile) -> bool:
    preview = profile.build_prediction_catalog(
        train_end=_PROFILE_PREDICTION_PREVIEW_TRAIN_END,
        test_start=_PROFILE_PREDICTION_PREVIEW_TEST_START,
        test_end=None,
    )
    return bool(preview.list())


def _local_verification_bound(value: str) -> date | datetime:
    return TimeWindow(value, value).start


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
        ownership = payload.get("root_ownership")
        run_id = payload.get("run_id")
        created_at = payload.get("created_at")
        if (
            set(payload) != _SEED_RUN_KEYS
            or payload.get("workflow_id") != _WORKFLOW_ID
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(created_at, str)
            or not created_at
            or not isinstance(ownership, dict)
            or set(ownership) != _ROOT_OWNERSHIP_KEYS
            or ownership.get("owner") != _WORKFLOW_ID
            or ownership.get("workflow_id") != _WORKFLOW_ID
            or ownership.get("created_by") != "quant_research_platform"
            or ownership.get("created_at") != created_at
            or ownership.get("replace_safe") is not True
        ):
            raise ValueError("run_root_not_owned_by_local_verification_workflow")
        validate_run_id(run_id)
    except (TypeError, ValueError):
        raise ValueError("run_root_not_owned_by_local_verification_workflow") from None
    return payload


def _assert_owned_local_verification_run_root(run_root: Path) -> None:
    _read_run_json(run_root)


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
    try:
        relative_root = Path(dataset_root).resolve(strict=False).relative_to(
            Path(run_root).resolve(strict=False)
        )
    except ValueError:
        raise ValueError("local_verification_dataset_layer_invalid") from None
    logical_layer = dataset_layer_from_path(relative_root)
    if (
        logical_layer not in RUN_RESULT_DATASET_LAYERS
        or relative_root.parts[:2] != ("data", logical_layer)
    ):
        raise ValueError("local_verification_dataset_layer_invalid")
    return logical_layer


def _build_layer_summaries(
    *,
    layout: LocalVerificationLayout,
    dataset_roots: tuple[Path, ...],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Path]] = {}
    for root in dataset_roots:
        logical_layer = _logical_layer(root, layout.run_root)
        grouped.setdefault(logical_layer, []).append(root)
    return tuple(
        build_layer_summary(
            layer_root=layout.data_root / logical_layer,
            logical_layer=logical_layer,
            dataset_roots=tuple(sorted(roots, key=lambda item: item.as_posix())),
        )
        for logical_layer, roots in sorted(grouped.items())
    )


def _output_index(
    output_bundle: Path,
    run_root: Path,
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    try:
        resolved_run_root = Path(run_root).resolve(strict=False)
        resolved_output_bundle = Path(output_bundle).resolve(strict=True)
        if not resolved_output_bundle.is_relative_to(resolved_run_root):
            raise ValueError("output_manifest_schema_id_invalid")
        payload, _declared_paths = validate_output_bundle(
            resolved_output_bundle,
            expected_run_id=expected_run_id,
            expected_output_id=resolved_output_bundle.name,
        )
    except (KeyError, OSError, TypeError, ValueError):
        raise ValueError("output_manifest_schema_id_invalid") from None
    manifest_path = resolved_output_bundle / "manifest.json"
    output_id = payload["output_id"]
    try:
        manifest_relative_path = manifest_path.relative_to(
            resolved_run_root
        ).as_posix()
    except ValueError:
        raise ValueError("output_manifest_schema_id_invalid") from None
    if manifest_relative_path != f"outputs/{output_id}/manifest.json":
        raise ValueError("output_manifest_schema_id_invalid")
    return {
        **payload,
        "manifest_path": manifest_relative_path,
    }


def _generated_text_paths(
    output_bundles: tuple[Path, ...],
    *,
    output_indexes: tuple[dict[str, Any], ...],
) -> tuple[Path, ...]:
    if len(output_bundles) != len(output_indexes):
        raise ValueError("output_evidence_declaration_mismatch")
    paths: list[Path] = []
    for root, index in zip(output_bundles, output_indexes, strict=True):
        files = index["files"]
        if type(files) is not list:
            raise ValueError("output_evidence_declaration_mismatch")
        paths.append(Path(root) / "manifest.json")
        paths.extend(
            Path(root) / name
            for name in files
            if Path(name).suffix.casefold() != ".parquet"
        )
    if not paths:
        raise ValueError("output_evidence_declaration_empty")
    return tuple(paths)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "LocalVerificationRunConfig",
    "LocalVerificationRunResult",
    "run_local_verification",
]
