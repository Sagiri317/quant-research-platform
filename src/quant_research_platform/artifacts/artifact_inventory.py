"""已完成 evaluation-set artifacts 的只读 inventory helper。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.artifact_loader import (
    LoadedEvaluationSet,
    load_evaluation_set,
)


def index_evaluation_sets(runs_root: str | Path) -> pd.DataFrame:
    """返回便于 API 消费的 completed evaluation set index。"""

    root = Path(runs_root)
    run_dirs = _discover_run_dirs(root)
    if not run_dirs:
        raise ValueError(f"runs_root contains no evaluation-set artifacts: {root}")
    records = [
        _run_index_record(load_evaluation_set(run_dir)) for run_dir in run_dirs
    ]
    return pd.DataFrame.from_records(records).sort_values(
        "evaluation_id"
    ).reset_index(drop=True)


def _discover_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)
    return sorted(
        child
        for child in runs_root.iterdir()
        if child.is_dir() and (child / "manifest.json").is_file()
    )


def _run_index_record(run: LoadedEvaluationSet) -> dict[str, object]:
    summary = run.summary
    config = run.config
    performance = run.tables["performance_summary"]
    return {
        "evaluation_id": str(summary["evaluation_id"]),
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "domain": str(summary["domain"]),
        "data_stage": str(summary["data_stage"]),
        "uses_external_data": bool(summary["uses_external_data"]),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "n_combinations": int(summary["n_combinations"]),
        "weight_method": _weight_method(config),
        "portfolio_mode": _portfolio_mode(config),
        "mean_final_return": _mean_numeric(
            performance,
            "final_cumulative_net_return",
        ),
        "mean_turnover": _mean_numeric(performance, "mean_turnover"),
        "total_cost": _sum_numeric(performance, "total_cost"),
    }


def _weight_method(config: dict[str, object]) -> str:
    configs = config.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and first.get("method") is not None:
            return str(first["method"])
    return "unknown"


def _portfolio_mode(config: dict[str, object]) -> str:
    modes = config.get("portfolio_modes")
    if isinstance(modes, list) and modes:
        normalized = [str(mode) for mode in modes if mode]
        if len(set(normalized)) > 1:
            return "multiple"
        if normalized:
            return normalized[0]
    configs = config.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and first.get("portfolio_mode") is not None:
            return str(first["portfolio_mode"])
    return "unknown"


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


__all__ = ["index_evaluation_sets"]
