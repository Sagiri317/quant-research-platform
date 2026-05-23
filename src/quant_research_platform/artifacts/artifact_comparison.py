"""提供 evaluation-set artifact 目录的 comparison helper。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.artifacts.artifact_loader import (
    LoadedEvaluationSet,
    load_evaluation_set,
)


def compare_evaluation_sets(output_dirs: list[str | Path]) -> pd.DataFrame:
    """把已完成的 evaluation sets 汇总成一张 comparison table。"""

    if not output_dirs:
        raise ValueError("output_dirs must contain at least one run directory.")
    runs = [load_evaluation_set(output_dir) for output_dir in output_dirs]
    records = [_summarize_run(run) for run in runs]
    comparison = pd.DataFrame.from_records(records).sort_values(
        "evaluation_id"
    ).reset_index(drop=True)
    return _add_metric_ranks(comparison)


def compare_evaluation_results(results: list[Any]) -> pd.DataFrame:
    """在不重读 artifacts 的情况下汇总内存中的 evaluation-set results。"""

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
    """写入 CSV comparison table，并返回内存中的 DataFrame。"""

    comparison = compare_evaluation_sets(output_dirs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    return comparison


def _summarize_run(run: LoadedEvaluationSet) -> dict[str, object]:
    return _comparison_record(
        summary=run.summary,
        metrics=run.tables,
    )


def _summarize_result(result: Any) -> dict[str, object]:
    if not hasattr(result, "summary") or not hasattr(result, "metrics"):
        raise TypeError("result must expose summary and metrics attributes.")
    return _comparison_record(
        summary=result.summary,
        metrics=result.metrics,
    )


def _comparison_record(
    *,
    summary: dict[str, object],
    metrics: dict[str, pd.DataFrame],
) -> dict[str, object]:
    evaluation_id = summary.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise ValueError("summary must contain a non-empty evaluation_id.")
    performance = metrics["performance_summary"]
    ic_summary = metrics["ic_summary"]
    return {
        "evaluation_id": evaluation_id,
        "project": _required_string(summary, "project"),
        "kind": _required_string(summary, "kind"),
        "n_rows_evaluated": _required_int(summary, "n_rows_evaluated"),
        "n_combinations": _required_int(summary, "n_combinations"),
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


def _required_string(summary: dict[str, object], field: str) -> str:
    value = summary.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"summary must contain a non-empty {field}.")
    return value


def _required_int(summary: dict[str, object], field: str) -> int:
    value = summary.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"summary must contain integer {field}.")
    return value


def _weight_method(summary: dict[str, object]) -> str:
    configs = summary.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and isinstance(first.get("method"), str):
            return str(first["method"])
    raise ValueError("summary weight_configs must include a non-empty method.")


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


__all__ = [
    "compare_evaluation_results",
    "compare_evaluation_sets",
    "write_run_comparison",
]
