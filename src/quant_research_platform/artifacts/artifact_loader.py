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
)
from quant_research_platform.runtime.run_metadata import validate_run_metadata


class LoadedEvaluationSet(BaseModel):
    """已校验的 evaluation-set artifact 内容。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output_dir: Path
    manifest: dict[str, object]
    config: dict[str, object]
    summary: dict[str, object]
    tables: dict[str, pd.DataFrame]
    report: str


def load_evaluation_set(
    output_dir: str | Path,
    *,
    table_names: set[str] | None = None,
) -> LoadedEvaluationSet:
    """读取并校验 batch-native evaluation-set artifact 目录。"""

    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    config_path = destination / "config.json"
    lineage_path = destination / "lineage.json"
    summary_path = destination / "summary.json"
    for path in (manifest_path, config_path, lineage_path, summary_path):
        _require_file(path, base=destination)

    manifest = _read_json(manifest_path)
    config = _read_json(config_path)
    summary = _read_json(summary_path)
    validate_artifact_schema_version(manifest, source="manifest.json")
    validate_artifact_schema_version(summary, source="summary.json")
    validate_evaluation_set_payload(manifest, source="manifest.json")
    validate_evaluation_set_payload(summary, source="summary.json")
    config_metadata = config.get("metadata")
    if not isinstance(config_metadata, dict):
        raise ValueError("config.json must contain a metadata object.")
    validate_run_metadata(config_metadata, source="config.json metadata")

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
        name: _read_table(
            _require_file(destination / relative_path, base=destination)
        )
        for name, relative_path in table_paths.items()
    }
    validate_evaluation_set_table_columns(tables)
    if display_chart_table_names(manifest).issubset(tables):
        validate_display_chart_columns(manifest, tables)

    report_relative = artifacts.get("report")
    if not isinstance(report_relative, str) or not report_relative:
        raise ValueError("manifest artifacts must include report.")
    validate_artifact_relative_path(report_relative, source="artifacts.report")
    report_path = _require_file(destination / report_relative, base=destination)
    return LoadedEvaluationSet(
        output_dir=destination,
        manifest=manifest,
        config=config,
        summary=summary,
        tables=tables,
        report=report_path.read_text(encoding="utf-8"),
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
