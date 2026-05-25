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
    EVALUATION_SET_INPUT_QUALITY_TABLES,
    EVALUATION_SET_METRIC_TABLES,
    EVALUATION_SET_PORTFOLIO_TABLES,
    EVALUATION_SET_REQUIRED_TABLES,
    EVALUATION_SET_SUMMARY_TABLES,
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
    lineage_path: Path | None = None


def write_evaluation_set_artifacts(
    output_dir: str | Path,
    *,
    artifact_paths: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    config: dict[str, object],
    lineage: dict[str, object] | None = None,
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
            default_selection=_default_selection(tables),
        )
    elif artifact_paths is not None:
        raise ValueError("provide either manifest or artifact_paths, not both.")

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
    summaries_dir = destination / "summaries"
    for directory in (
        destination,
        dimensions_dir,
        metrics_dir,
        portfolio_dir,
        diagnostics_dir,
        input_quality_dir,
        summaries_dir,
    ):
        _timed_artifact_operation(
            "directory_creation",
            lambda directory=directory: directory.mkdir(parents=True, exist_ok=True),
        )

    manifest_path = destination / "manifest.json"
    config_path = destination / "config.json"
    lineage_path = destination / "lineage.json"
    summary_path = destination / "summary.json"
    report_path = destination / "report.md"

    persisted_lineage = _public_lineage(lineage)
    persisted_lineage.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
    for path, payload in (
        (manifest_path, manifest),
        (config_path, config),
        (lineage_path, persisted_lineage),
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
            summaries_dir=summaries_dir,
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
        lineage_path=lineage_path,
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
    summaries_dir: Path,
) -> Path:
    if name == "combinations":
        return dimensions_dir / "combinations.csv"
    if name in EVALUATION_SET_METRIC_TABLES:
        return metrics_dir / f"{name}.csv"
    if name in EVALUATION_SET_PORTFOLIO_TABLES:
        suffix = ".parquet" if name == "weights" else ".csv"
        return portfolio_dir / f"{name}{suffix}"
    if name in EVALUATION_SET_DIAGNOSTIC_TABLES:
        return diagnostics_dir / f"{name}.csv"
    if name in EVALUATION_SET_INPUT_QUALITY_TABLES:
        return input_quality_dir / f"{name}.csv"
    if name in EVALUATION_SET_SUMMARY_TABLES:
        return summaries_dir / f"{name}.csv"
    for period in BREAKDOWN_PERIODS:
        if name.endswith(f"_by_{period}"):
            table_name = name.removesuffix(f"_by_{period}")
            return output_dir / "breakdowns" / period / f"{table_name}.csv"
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
        "kernel": summary.get("kernel"),
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


def _public_lineage(lineage: dict[str, object] | None) -> dict[str, object]:
    result = dict(lineage or {})
    result.pop("input_file", None)
    return result


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
