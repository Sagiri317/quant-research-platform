"""加载 evaluation-set artifact 的入口。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from quant_research_platform.artifacts.manifest_validation import (
    display_chart_table_names,
    evaluation_set_csv_paths,
    validate_artifact_relative_path,
    validate_artifact_schema_version,
    validate_display_chart_columns,
    validate_display_manifest,
    validate_evaluation_set_payload,
    validate_evaluation_set_table_columns,
    validate_portable_run_identity,
    validate_run_context_payload,
    validate_summary_payload,
)


class LoadedEvaluationSet(BaseModel):
    """已校验的 evaluation-set artifact 内容。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output_dir: Path
    manifest: dict[str, object]
    config: dict[str, object]
    run_context: dict[str, object]
    summary: dict[str, object]
    tables: dict[str, pd.DataFrame]
    report: str | None


def load_evaluation_set(
    output_dir: str | Path,
    *,
    table_names: set[str] | None = None,
) -> LoadedEvaluationSet:
    """读取并校验 batch-native evaluation-set artifact 目录。"""

    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    run_context_path = destination / "run_context.json"
    summary_path = destination / "summary.json"
    for path in (manifest_path, run_context_path, summary_path):
        _require_file(path, base=destination)

    manifest = _read_json(manifest_path)
    run_context = _read_json(run_context_path)
    summary = _read_json(summary_path)
    validate_artifact_schema_version(manifest, source="manifest.json")
    validate_evaluation_set_payload(manifest, source="manifest.json")
    validate_summary_payload(summary)
    validate_run_context_payload(run_context, summary=summary)
    validate_portable_run_identity(manifest, summary, run_context)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    table_paths = evaluation_set_csv_paths(artifacts)
    validate_display_manifest(manifest)
    if table_names is not None:
        missing_tables = sorted(table_names.difference(table_paths))
        if missing_tables:
            raise ValueError(f"manifest tables are missing: {missing_tables}.")
        table_paths = {
            name: relative_path
            for name, relative_path in table_paths.items()
            if name in table_names
        }
    tables = {
        name: _read_table(_require_file(destination / relative_path, base=destination))
        for name, relative_path in table_paths.items()
    }
    validate_evaluation_set_table_columns(tables)
    if display_chart_table_names(manifest).issubset(tables):
        validate_display_chart_columns(manifest, tables)

    report: str | None = None
    report_relative = artifacts.get("report")
    if report_relative is not None:
        if not isinstance(report_relative, str) or not report_relative:
            raise ValueError("manifest artifacts.report must be a non-empty string.")
        validate_artifact_relative_path(report_relative, source="artifacts.report")
        report_path = _require_file(destination / report_relative, base=destination)
        report = report_path.read_text(encoding="utf-8")
    return LoadedEvaluationSet(
        output_dir=destination,
        manifest=manifest,
        config=run_context.get("config", {})
        if isinstance(run_context.get("config"), dict)
        else {},
        run_context=run_context,
        summary=summary,
        tables=tables,
        report=report,
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_file(path: Path, *, base: Path) -> Path:
    base_path = base.resolve()
    candidate = path.resolve()
    if candidate != base_path and base_path not in candidate.parents:
        raise ValueError(f"artifact path is unsafe: {path}.")
    if not candidate.is_file():
        try:
            missing = candidate.relative_to(base_path).as_posix()
        except ValueError:
            missing = str(path)
        raise FileNotFoundError(missing)
    return candidate


__all__ = ["LoadedEvaluationSet", "load_evaluation_set"]
