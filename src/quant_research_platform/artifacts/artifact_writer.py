"""写入 evaluation-set artifact 的入口。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    BREAKDOWN_PERIODS,
    EVALUATION_SET_ARTIFACT_TYPE,
    EVALUATION_SET_DIAGNOSTIC_TABLES,
    EVALUATION_SET_METRIC_TABLES,
    EVALUATION_SET_PORTFOLIO_TABLES,
    EVALUATION_SET_REQUIRED_TABLES,
    build_evaluation_set_artifact_paths,
)
from quant_research_platform.artifacts.display_manifest import build_display_manifest
from quant_research_platform.artifacts.manifest_validation import (
    evaluation_set_csv_paths,
    validate_display_chart_columns,
    validate_display_manifest,
    validate_evaluation_set_table_columns,
)
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


def write_evaluation_set_artifacts(
    output_dir: str | Path,
    *,
    artifact_paths: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    config: dict[str, object],
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
    report: str,
) -> ArtifactOutputs:
    """写入 batch-native evaluation-set artifact contract。"""

    persisted_summary = _summary_with_artifact_contract(summary)
    if manifest is None:
        if artifact_paths is None:
            artifact_paths = build_evaluation_set_artifact_paths(
                summary=persisted_summary,
                output_dir=output_dir,
            )
        manifest = build_evaluation_set_manifest(
            summary=persisted_summary,
            artifact_paths=artifact_paths,
        )
    elif artifact_paths is not None:
        raise ValueError("provide either manifest or artifact_paths, not both.")

    _validate_required_evaluation_set_tables(tables)
    validate_evaluation_set_table_columns(tables)
    validate_display_manifest(manifest)
    _validate_manifest_tables_available(manifest, tables)
    validate_display_chart_columns(manifest, tables)

    destination = Path(output_dir)
    metrics_dir = destination / "metrics"
    portfolio_dir = destination / "portfolio"
    diagnostics_dir = destination / "diagnostics"
    for directory in (destination, metrics_dir, portfolio_dir, diagnostics_dir):
        _timed_artifact_operation(
            "directory_creation",
            lambda directory=directory: directory.mkdir(parents=True, exist_ok=True),
        )

    manifest_path = destination / "manifest.json"
    config_path = destination / "config.json"
    summary_path = destination / "summary.json"
    report_path = destination / "report.md"

    for path, payload in (
        (manifest_path, manifest),
        (config_path, config),
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
            metrics_dir=metrics_dir,
            portfolio_dir=portfolio_dir,
            diagnostics_dir=diagnostics_dir,
        )
        _timed_artifact_operation(
            "directory_creation",
            lambda path=path: path.parent.mkdir(parents=True, exist_ok=True),
        )
        _timed_artifact_operation(
            "csv_writing",
            lambda frame=frame, path=path: frame.to_csv(path, index=False),
        )
        metric_paths[name] = path

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
        config_path=config_path,
    )


def _evaluation_set_table_path(
    *,
    name: str,
    output_dir: Path,
    metrics_dir: Path,
    portfolio_dir: Path,
    diagnostics_dir: Path,
) -> Path:
    if name == "combinations":
        return output_dir / "combinations.csv"
    if name in EVALUATION_SET_METRIC_TABLES:
        return metrics_dir / f"{name}.csv"
    if name in EVALUATION_SET_PORTFOLIO_TABLES:
        return portfolio_dir / f"{name}.csv"
    if name in EVALUATION_SET_DIAGNOSTIC_TABLES:
        return diagnostics_dir / f"{name}.csv"
    for period in BREAKDOWN_PERIODS:
        if name.endswith(f"_by_{period}"):
            return output_dir / f"metrics_by_{period}" / f"{name}.csv"
    return metrics_dir / f"{name}.csv"


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


def build_evaluation_set_manifest(
    *,
    summary: dict[str, object],
    artifact_paths: dict[str, object],
) -> dict[str, object]:
    """构造 evaluation set 持久化 manifest 和 display contract。"""

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": EVALUATION_SET_ARTIFACT_TYPE,
        "evaluation_id": summary["evaluation_id"],
        "project": summary["project"],
        "kind": summary["kind"],
        "domain": summary["domain"],
        "data_stage": summary["data_stage"],
        "uses_external_data": summary["uses_external_data"],
        "schema": summary["schema"],
        "n_combinations": summary["n_combinations"],
        "segment_columns": summary["segment_columns"],
        "breakdown_enabled": summary["breakdown_enabled"],
        "breakdown_columns": summary["breakdown_columns"],
        "period_breakdown_policy": summary["period_breakdown_policy"],
        "engine": summary["engine"],
        "position_artifact_policy": summary["position_artifact_policy"],
        "artifacts": artifact_paths,
        "display": build_display_manifest(
            artifact_paths,
            schema_version=ARTIFACT_SCHEMA_VERSION,
        ),
    }


def _summary_with_artifact_contract(summary: dict[str, object]) -> dict[str, object]:
    result = dict(summary)
    result["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    result["artifact_type"] = EVALUATION_SET_ARTIFACT_TYPE
    return result


def _timed_artifact_operation(category: str, operation: Callable[[], object]) -> object:
    return measure_stage("artifact", category, operation)


__all__ = [
    "ArtifactOutputs",
    "build_evaluation_set_manifest",
    "write_evaluation_set_artifacts",
]
