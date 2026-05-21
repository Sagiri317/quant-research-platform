"""Read-only inventory helpers for completed evaluation-set artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts import (
    LoadedEvaluationSet,
    load_evaluation_set,
)


def index_evaluation_sets(runs_root: str | Path) -> pd.DataFrame:
    """Return a future-API-friendly index of completed evaluation sets."""

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
        "weight_method": _weight_method(summary),
        "portfolio_mode": _portfolio_mode(summary),
        "mean_final_return": _mean_numeric(
            performance,
            "final_cumulative_net_return",
        ),
        "mean_turnover": _mean_numeric(performance, "mean_turnover"),
        "total_cost": _sum_numeric(performance, "total_cost"),
    }


def _weight_method(summary: dict[str, object]) -> str:
    configs = summary.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and first.get("method") is not None:
            return str(first["method"])
    return "unknown"


def _portfolio_mode(summary: dict[str, object]) -> str:
    configs = summary.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and first.get("portfolio_mode") is not None:
            return str(first["portfolio_mode"])
    return "unknown"


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum())
