"""本地 API run list 使用的轻量 evaluation-set artifact index loader。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.artifacts.manifest_validation import (
    evaluation_set_csv_paths,
    validate_artifact_schema_version,
    validate_display_chart_columns,
    validate_display_manifest,
    validate_evaluation_set_payload,
    validate_evaluation_set_table_columns,
    validate_portable_run_identity,
    validate_run_context_payload,
    validate_summary_payload,
)
from quant_research_platform.artifacts.schema_registry import (
    get_table_spec,
    validate_table_contract,
)

from .models import RunIndexRecord, RunRef


def load_run_index(source_alias: str, path: Path) -> RunIndexRecord:
    """读取一个 artifact run 的轻量 index，不加载全部 metrics。"""

    manifest = read_json(path / "manifest.json")
    run_context = read_json(path / "run_context.json")
    summary = read_json(path / "summary.json")
    validate_artifact_schema_version(manifest, source="manifest.json")
    validate_evaluation_set_payload(manifest, source="manifest.json")
    validate_summary_payload(summary)
    validate_run_context_payload(run_context, summary=summary)
    validate_portable_run_identity(manifest, summary, run_context)
    validate_display_manifest(manifest)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    table_paths = evaluation_set_csv_paths(artifacts)
    table_files = {
        name: require_index_file(path, relative_path)
        for name, relative_path in table_paths.items()
    }
    table_headers = {
        name: read_manifest_table_header(table_file)
        for name, table_file in table_files.items()
    }
    validate_run_index_table_contracts(table_headers, table_files)
    validate_display_chart_columns(manifest, table_headers)
    report_path = artifacts.get("report")
    if report_path is not None:
        if not isinstance(report_path, str) or not report_path:
            raise ValueError("manifest artifacts.report must be a non-empty string.")
        require_index_file(path, report_path)
    metrics = artifacts.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("manifest artifacts must contain a metrics object.")

    evaluation_metrics_summary = read_manifest_metric(
        path,
        metrics,
        "evaluation_metrics_summary",
        required_columns={
            "combination_id",
            "metric_family",
            "metric_name",
            "metric_value",
        },
    )
    run_id = str(summary["run_id"])
    return RunIndexRecord(
        ref=RunRef(source_alias, run_id, path),
        manifest=manifest,
        config=run_context.get("config", {})
        if isinstance(run_context.get("config"), dict)
        else {},
        run_context=run_context,
        summary=summary,
        evaluation_metrics_summary=evaluation_metrics_summary,
    )


def read_manifest_metric(
    base_path: Path,
    metrics: dict[object, object],
    name: str,
    *,
    required_columns: set[str],
) -> pd.DataFrame:
    """按 manifest 中的相对路径读取一个 summary metric 并校验列。"""

    relative_path = metrics.get(name)
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"manifest metrics must include {name}.")
    frame = pd.read_csv(require_index_file(base_path, relative_path))
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")
    return frame


def read_manifest_table_header(path: Path) -> pd.DataFrame:
    """Read only table column names for lightweight schema validation."""

    if path.suffix == ".parquet":
        return pd.DataFrame(columns=pq.ParquetFile(path).schema_arrow.names)
    return pd.read_csv(path, nrows=0)


def validate_run_index_table_contracts(
    table_headers: dict[str, pd.DataFrame],
    table_files: dict[str, Path],
) -> None:
    """Run schema-registry row checks while reading only validation columns."""

    validate_evaluation_set_table_columns(table_headers)
    for name, header in table_headers.items():
        spec = get_table_spec(name)
        rows = read_manifest_table_contract_rows(
            table_files[name],
            spec_key=name,
            columns=tuple(str(column) for column in header.columns),
        )
        validate_table_contract(
            spec,
            rows=rows,
            columns=tuple(str(column) for column in header.columns),
        )


def read_manifest_table_contract_rows(
    path: Path,
    *,
    spec_key: str,
    columns: tuple[str, ...],
) -> list[dict[str, object]]:
    """Read only registry columns needed for run-index row validation."""

    spec = get_table_spec(spec_key)
    validation_columns = tuple(
        dict.fromkeys(
            column
            for column in (
                *spec.unique_key,
                *(spec.required_if),
                *(spec.sentinel_values),
                *((spec.status_column,) if spec.status_column is not None else ()),
            )
            if column in columns
        )
    )
    if not validation_columns:
        return []
    if path.suffix == ".parquet":
        frame = pq.read_table(path, columns=list(validation_columns)).to_pandas()
    else:
        frame = pd.read_csv(path, usecols=list(validation_columns))
    return frame.to_dict(orient="records")


def read_json(path: Path) -> dict[str, object]:
    """读取 JSON object 文件；非 object 顶层结构会被拒绝。"""

    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def require_index_file(base_path: Path, relative_path: str) -> Path:
    """把 manifest 相对路径解析为必须存在的 artifact 文件。"""

    validate_manifest_relative_path(relative_path)
    resolved_base = base_path.resolve()
    path = (base_path / relative_path).resolve()
    if path != resolved_base and resolved_base not in path.parents:
        raise ValueError(f"manifest artifact path is unsafe: {relative_path}.")
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    return path


def validate_manifest_relative_path(relative_path: str) -> None:
    """拒绝绝对路径、父目录穿越和平台特定 artifact 路径。"""

    parsed = PurePosixPath(relative_path)
    if (
        "\\" in relative_path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"manifest artifact path is unsafe: {relative_path}.")


def require_column(frame: pd.DataFrame, name: str, column: str) -> None:
    """要求当前 schema artifact column 存在。"""

    if column not in frame.columns:
        raise ValueError(f"{name} is missing column: {column}.")
