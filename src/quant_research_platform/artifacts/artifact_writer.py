"""写入 evaluation-set artifact 的入口。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    EVALUATION_SET_ARTIFACT_TYPE,
    EVALUATION_SET_REQUIRED_TABLES,
    build_evaluation_set_artifact_paths,
)
from quant_research_platform.artifacts.display_manifest import build_display_manifest
from quant_research_platform.artifacts.manifest_validation import (
    evaluation_set_csv_paths,
    validate_display_chart_columns,
    validate_display_manifest,
    validate_evaluation_set_table_columns,
    validate_run_context_payload,
    validate_summary_payload,
)
from quant_research_platform.artifacts.public_metadata import sanitize_public_metadata
from quant_research_platform.artifacts.schema_registry import get_table_spec
from quant_research_platform.runtime.profiling import measure_stage


class ArtifactOutputs(BaseModel):
    """记录 research run 写出的路径集合。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output_dir: Path
    summary_path: Path
    report_path: Path
    metric_paths: dict[str, Path]
    manifest_path: Path | None = None
    config_path: Path | None = None
    lineage_path: Path | None = None
    run_context_path: Path | None = None


def write_evaluation_set_artifacts(
    output_dir: str | Path,
    *,
    artifact_paths: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
    run_context: dict[str, object] | None = None,
    lineage: dict[str, object] | None = None,
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
    report: str | None,
) -> ArtifactOutputs:
    """写入 batch-native evaluation-set artifact contract。"""

    persisted_summary = _summary_with_artifact_contract(summary)
    persisted_run_context = _run_context_payload(
        run_context, config, lineage, persisted_summary
    )
    if manifest is None:
        if artifact_paths is None:
            artifact_paths = build_evaluation_set_artifact_paths(
                summary=persisted_summary,
                output_dir=output_dir,
                include_report=report is not None,
                materialized_tables=set(tables),
            )
        manifest = build_evaluation_set_manifest(
            summary=persisted_summary,
            artifact_paths=artifact_paths,
            default_selection=_default_selection(tables),
        )
    elif artifact_paths is not None:
        raise ValueError("provide either manifest or artifact_paths, not both.")

    validate_summary_payload(persisted_summary)
    validate_run_context_payload(persisted_run_context, summary=persisted_summary)
    _validate_required_evaluation_set_tables(tables)
    validate_evaluation_set_table_columns(tables)
    validate_display_manifest(manifest)
    _validate_manifest_tables_available(manifest, tables)
    _validate_no_extra_payload_tables(manifest, tables)
    validate_display_chart_columns(manifest, tables)

    destination = Path(output_dir)
    dimensions_dir = destination / "dimensions"
    metrics_dir = destination / "metrics"
    portfolio_dir = destination / "portfolio"
    diagnostics_dir = destination / "diagnostics"
    input_quality_dir = destination / "input_quality"
    for directory in (
        destination,
        dimensions_dir,
        metrics_dir,
        portfolio_dir,
        diagnostics_dir,
        input_quality_dir,
    ):
        _timed_artifact_operation(
            "directory_creation",
            lambda directory=directory: directory.mkdir(parents=True, exist_ok=True),
        )

    manifest_path = destination / "manifest.json"
    run_context_path = destination / "run_context.json"
    summary_path = destination / "summary.json"
    report_path = destination / "report.md"

    for path, payload in (
        (manifest_path, manifest),
        (run_context_path, persisted_run_context),
        (summary_path, persisted_summary),
    ):
        _timed_artifact_operation(
            "json_writing",
            lambda path=path, payload=payload: path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            ),
        )

    metric_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"table {name!r} must be a pandas DataFrame.")
        path = _evaluation_set_table_path(
            name=name,
            output_dir=destination,
            dimensions_dir=dimensions_dir,
            metrics_dir=metrics_dir,
            portfolio_dir=portfolio_dir,
            diagnostics_dir=diagnostics_dir,
            input_quality_dir=input_quality_dir,
        )
        _timed_artifact_operation(
            "directory_creation",
            lambda path=path: path.parent.mkdir(parents=True, exist_ok=True),
        )
        _timed_artifact_operation(
            "parquet_writing" if path.suffix == ".parquet" else "csv_writing",
            lambda frame=frame, path=path: _write_table(frame, path),
        )
        metric_paths[name] = path

    if report is not None:
        _timed_artifact_operation(
            "report_writing",
            lambda: report_path.write_text(report, encoding="utf-8"),
        )
    return ArtifactOutputs(
        output_dir=destination,
        summary_path=summary_path,
        report_path=report_path,
        metric_paths=metric_paths,
        manifest_path=manifest_path,
        config_path=None,
        lineage_path=None,
        run_context_path=run_context_path,
    )


def _evaluation_set_table_path(
    *,
    name: str,
    output_dir: Path,
    dimensions_dir: Path,
    metrics_dir: Path,
    portfolio_dir: Path,
    diagnostics_dir: Path,
    input_quality_dir: Path,
) -> Path:
    del dimensions_dir, metrics_dir, portfolio_dir, diagnostics_dir, input_quality_dir
    return output_dir / get_table_spec(name).path


def _validate_required_evaluation_set_tables(tables: dict[str, pd.DataFrame]) -> None:
    missing = sorted(set(EVALUATION_SET_REQUIRED_TABLES).difference(tables))
    if missing:
        raise ValueError(f"missing required evaluation-set tables: {missing}.")


def _validate_manifest_tables_available(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    missing = sorted(set(evaluation_set_csv_paths(artifacts)).difference(tables))
    if missing:
        raise ValueError(f"display artifact tables missing from payload: {missing}.")


def _validate_no_extra_payload_tables(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be an object.")
    declared = set(evaluation_set_csv_paths(artifacts))
    extra = sorted(set(tables).difference(declared))
    if extra:
        raise ValueError(f"payload tables not declared by manifest: {extra}.")


def build_evaluation_set_manifest(
    *,
    summary: dict[str, object],
    artifact_paths: dict[str, object],
    default_selection: dict[str, str] | None = None,
) -> dict[str, object]:
    """构造 evaluation set 持久化 manifest 和 display contract。"""

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": EVALUATION_SET_ARTIFACT_TYPE,
        "run_id": summary["run_id"],
        "project": summary["project"],
        "kind": summary["kind"],
        "data_stage": summary["data_stage"],
        "base_evaluation": summary["base_evaluation"],
        "diagnostics": summary["diagnostics"],
        "n_combinations": summary["n_combinations"],
        "artifacts": artifact_paths,
        "display": build_display_manifest(
            artifact_paths,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            default_selection=default_selection,
        ),
    }


def _summary_with_artifact_contract(summary: dict[str, object]) -> dict[str, object]:
    result = dict(summary)
    result["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    result["artifact_type"] = EVALUATION_SET_ARTIFACT_TYPE
    return result


def _run_context_payload(
    run_context: dict[str, object] | None,
    config: dict[str, object] | None,
    lineage: dict[str, object] | None,
    summary: dict[str, object],
) -> dict[str, object]:
    if run_context is not None:
        payload = run_context
    else:
        payload = {
            "config": {
                "evaluation_spec": config or {},
                "diagnostics_spec": summary.get("diagnostics", {"enabled": False}),
                "a_share_assumptions": {"status": "not_applicable"},
                "config_fingerprint": "unavailable",
            },
            "provenance": {
                "run_id": summary.get("run_id"),
                "input_panel_fingerprint": None,
                "input_data_version": "unavailable",
                "input_artifact_refs": [],
                "output_artifact_index": {},
                "diagnostic_policy_fingerprint": _diagnostic_policy_fingerprint(
                    summary
                ),
                "generated_at": "unavailable",
                "generator": "quant_research_platform",
                "code_version": "unavailable",
                "environment_summary": {},
                "staging_payload_fingerprint": None,
                "factor_definition_version": "unavailable",
                "random_seed": "not_applicable",
                **(lineage or {}),
            },
        }
    sanitized = sanitize_public_metadata(payload)
    if not isinstance(sanitized, dict):
        return {}
    return sanitized


def _diagnostic_policy_fingerprint(summary: dict[str, object]) -> str:
    diagnostics = summary.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("enabled") is True:
        return "n5_v1_descriptive_proxy"
    return "disabled"


def _public_lineage(lineage: dict[str, object] | None) -> dict[str, object]:
    return _public_metadata(lineage or {})


def _public_metadata(value: dict[str, object]) -> dict[str, object]:
    sanitized = sanitize_public_metadata(value)
    if not isinstance(sanitized, dict):
        return {}
    return sanitized


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    frame.to_csv(path, index=False)


def _default_selection(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    combinations = tables.get("combinations")
    if combinations is None or combinations.empty:
        return {}
    row = combinations.iloc[0]
    selection: dict[str, str] = {}
    for column in ("feature", "label", "kernel", "portfolio_mode"):
        value = row.get(column)
        if isinstance(value, str):
            selection[column] = value
    return selection


def _timed_artifact_operation(category: str, operation: Callable[[], object]) -> object:
    return measure_stage("artifact", category, operation)


__all__ = [
    "ArtifactOutputs",
    "build_evaluation_set_manifest",
    "write_evaluation_set_artifacts",
]
