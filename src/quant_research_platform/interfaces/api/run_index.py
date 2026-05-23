"""本地 API run list 使用的轻量 evaluation-set artifact index loader。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from quant_research_platform.artifacts.manifest_validation import (
    validate_artifact_schema_version,
    validate_display_manifest,
    validate_evaluation_set_payload,
)

from .models import RunIndexRecord, RunRef


def load_run_index(source_alias: str, path: Path) -> RunIndexRecord:
    """读取一个 artifact run 的轻量 index，不加载全部 metrics。"""

    manifest = read_json(path / "manifest.json")
    summary = read_json(path / "summary.json")
    validate_artifact_schema_version(manifest, source="manifest.json")
    validate_artifact_schema_version(summary, source="summary.json")
    validate_evaluation_set_payload(manifest, source="manifest.json")
    validate_evaluation_set_payload(summary, source="summary.json")
    validate_display_manifest(manifest)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    metrics = artifacts.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("manifest artifacts must contain a metrics object.")

    ic_summary = read_manifest_metric(
        path,
        metrics,
        "ic_summary",
        required_columns={
            "mean_ic",
            "mean_rankic",
            "icir",
            "rankicir",
        },
    )
    performance_summary = read_manifest_metric(
        path,
        metrics,
        "performance_summary",
        required_columns={
            "mean_turnover",
            "total_cost",
            "sharpe_like",
            "sortino_like",
            "max_drawdown",
            "win_rate",
        },
    )
    require_column(
        performance_summary,
        "performance_summary",
        "final_cumulative_net_return",
    )
    run_id = str(summary["evaluation_id"])
    return RunIndexRecord(
        ref=RunRef(source_alias, run_id, path),
        manifest=manifest,
        summary=summary,
        ic_summary=ic_summary,
        performance_summary=performance_summary,
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
