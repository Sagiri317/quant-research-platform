"""Comparison helpers for evaluation-set artifact directories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.writer import (
    LoadedEvaluationSet,
    load_evaluation_set,
)
from quant_research_platform.core.results import EvaluationSetResult


def compare_evaluation_sets(output_dirs: list[str | Path]) -> pd.DataFrame:
    """Summarize completed evaluation sets into one comparison table."""

    if not output_dirs:
        raise ValueError("output_dirs must contain at least one run directory.")
    runs = [load_evaluation_set(output_dir) for output_dir in output_dirs]
    records = [_summarize_run(run) for run in runs]
    comparison = pd.DataFrame.from_records(records).sort_values(
        "evaluation_id"
    ).reset_index(drop=True)
    return _add_metric_ranks(comparison)


def compare_evaluation_results(results: list[EvaluationSetResult]) -> pd.DataFrame:
    """Summarize in-memory evaluation-set results without rereading artifacts."""

    if not results:
        raise ValueError("results must contain at least one evaluation result.")
    records = [_summarize_result(result) for result in results]
    comparison = pd.DataFrame.from_records(records).sort_values(
        "evaluation_id"
    ).reset_index(drop=True)
    return _add_metric_ranks(comparison)


def write_run_comparison(
    output_dirs: list[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """Write a CSV comparison table and return the in-memory DataFrame."""

    comparison = compare_evaluation_sets(output_dirs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    return comparison


def _summarize_run(run: LoadedEvaluationSet) -> dict[str, object]:
    return _comparison_record(
        summary=run.summary,
        metrics=run.tables,
        fallback_run_id=run.output_dir.name,
    )


def _summarize_result(result: EvaluationSetResult) -> dict[str, object]:
    return _comparison_record(
        summary=result.summary,
        metrics=result.metrics,
        fallback_run_id=result.artifacts.output_dir.name,
    )


def _comparison_record(
    *,
    summary: dict[str, object],
    metrics: dict[str, pd.DataFrame],
    fallback_run_id: str,
) -> dict[str, object]:
    combinations = metrics.get("combinations")
    if combinations is None:
        combinations = pd.DataFrame()
    performance = metrics["performance_summary"]
    ic_summary = metrics["ic_summary"]
    return {
        "evaluation_id": str(summary.get("evaluation_id", fallback_run_id)),
        "project": str(summary.get("project", "unknown")),
        "kind": str(summary.get("kind", "unknown")),
        "n_rows_evaluated": int(summary.get("n_rows_evaluated", 0)),
        "n_combinations": int(summary.get("n_combinations", len(combinations))),
        "weight_method": _weight_method(summary),
        "cost_bps": _weight_cost_bps(summary),
        "mean_ic": _mean_numeric(ic_summary, "mean_ic"),
        "mean_rankic": _mean_numeric(ic_summary, "mean_rankic"),
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


def _weight_cost_bps(summary: dict[str, object]) -> float:
    cost_configs = summary.get("cost_configs")
    cost_config = (
        cost_configs[0]
        if isinstance(cost_configs, list) and cost_configs
        else None
    )
    if isinstance(cost_config, dict):
        detailed_cost = (
            float(cost_config.get("buy_cost_bps") or 0.0)
            + float(cost_config.get("sell_cost_bps") or 0.0)
            + float(cost_config.get("slippage_bps") or 0.0)
            + float(cost_config.get("spread_bps") or 0.0)
        )
        if detailed_cost:
            return detailed_cost
    configs = summary.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and first.get("cost_bps") is not None:
            return float(first["cost_bps"])
    return 0.0


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def _add_metric_ranks(comparison: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison.copy()
    rank_specs = {
        "mean_ic": False,
        "mean_rankic": False,
        "mean_final_return": False,
        "mean_turnover": True,
        "total_cost": True,
    }
    for column, ascending in rank_specs.items():
        ranked[f"rank_{column}"] = (
            ranked[column].rank(method="min", ascending=ascending).astype("int64")
        )
    return ranked
