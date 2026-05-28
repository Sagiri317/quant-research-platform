"""提供 evaluation-set artifact 目录的 comparison helper。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.artifacts.artifact_loader import (
    LoadedEvaluationSet,
)
from quant_research_platform.artifacts.manifest_validation import (
    evaluation_set_csv_paths,
    validate_artifact_schema_version,
    validate_display_manifest,
    validate_evaluation_set_payload,
    validate_run_context_payload,
    validate_summary_payload,
)


def compare_evaluation_sets(output_dirs: list[str | Path]) -> pd.DataFrame:
    """把已完成的 evaluation sets 汇总成一张 low-memory comparison table。"""

    if not output_dirs:
        raise ValueError("output_dirs must contain at least one run directory.")
    records = [_summarize_run_path(Path(output_dir)) for output_dir in output_dirs]
    comparison = (
        pd.DataFrame.from_records(records).sort_values("run_id").reset_index(drop=True)
    )
    return _add_metric_ranks(comparison)


def compare_evaluation_results(results: list[Any]) -> pd.DataFrame:
    """在不重读 artifacts 的情况下汇总内存中的 evaluation-set results。"""

    if not results:
        raise ValueError("results must contain at least one evaluation result.")
    records = [_summarize_result(result) for result in results]
    comparison = (
        pd.DataFrame.from_records(records).sort_values("run_id").reset_index(drop=True)
    )
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
        config=run.config,
        metrics=run.tables,
    )


def _summarize_run_path(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    manifest_path = output_dir / "manifest.json"
    run_context_path = output_dir / "run_context.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not run_context_path.is_file():
        raise FileNotFoundError(run_context_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object.")
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    if not isinstance(run_context, dict):
        raise ValueError("run_context.json must contain a JSON object.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain a JSON object.")
    validate_artifact_schema_version(manifest, source="manifest.json")
    validate_evaluation_set_payload(manifest, source="manifest.json")
    validate_summary_payload(summary)
    validate_run_context_payload(run_context, summary=summary)
    validate_display_manifest(manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    table_paths = evaluation_set_csv_paths(artifacts)
    metrics = {
        "evaluation_metrics_summary": pd.read_csv(
            output_dir / table_paths["evaluation_metrics_summary"]
        ),
    }
    config = run_context.get("config", {})
    if not isinstance(config, dict):
        config = {}
    return _comparison_record(summary=summary, config=config, metrics=metrics)


def _summarize_result(result: Any) -> dict[str, object]:
    if not hasattr(result, "summary") or not hasattr(result, "metrics"):
        raise TypeError("result must expose summary and metrics attributes.")
    return _comparison_record(
        summary=result.summary,
        config=getattr(result, "config", result.summary),
        metrics=result.metrics,
    )


def _comparison_record(
    *,
    summary: dict[str, object],
    config: dict[str, object],
    metrics: dict[str, pd.DataFrame],
) -> dict[str, object]:
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("summary must contain a non-empty run_id.")
    evaluation_metrics = metrics["evaluation_metrics_summary"]
    return {
        "run_id": run_id,
        "project": _required_string(summary, "project"),
        "kind": _required_string(summary, "kind"),
        "n_rows_evaluated": _required_int(summary, "n_rows_evaluated"),
        "n_combinations": _required_int(summary, "n_combinations"),
        "weight_method": _weight_method(config),
        "cost_bps": _weight_cost_bps(config),
        "mean_ic": _metric_value(evaluation_metrics, "mean_ic"),
        "mean_rankic": _metric_value(evaluation_metrics, "mean_rankic"),
        "mean_final_return": _metric_value(
            evaluation_metrics,
            "final_cumulative_net_return",
        ),
        "mean_turnover": _metric_value(evaluation_metrics, "mean_turnover"),
        "total_cost": _metric_value(evaluation_metrics, "total_cost"),
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


def _weight_method(config: dict[str, object]) -> str:
    configs = config.get("weight_configs")
    if not isinstance(configs, list):
        evaluation_spec = config.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            configs = evaluation_spec.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and isinstance(first.get("method"), str):
            return str(first["method"])
    raise ValueError("config weight_configs must include a non-empty method.")


def _weight_cost_bps(config: dict[str, object]) -> float:
    cost_configs = config.get("cost_configs")
    if not isinstance(cost_configs, list):
        evaluation_spec = config.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            cost_configs = evaluation_spec.get("cost_configs")
    cost_config = (
        cost_configs[0] if isinstance(cost_configs, list) and cost_configs else None
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
    cost_config = config.get("cost_config")
    if isinstance(cost_config, dict) and cost_config.get("cost_bps") is not None:
        return float(cost_config.get("cost_bps") or 0.0)
    weight_config = config.get("weight_config")
    if isinstance(weight_config, dict) and weight_config.get("cost_bps") is not None:
        return float(weight_config.get("cost_bps") or 0.0)
    weight_configs = config.get("weight_configs")
    if not isinstance(weight_configs, list):
        evaluation_spec = config.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            weight_configs = evaluation_spec.get("weight_configs")
    weight_config = (
        weight_configs[0]
        if isinstance(weight_configs, list) and weight_configs
        else None
    )
    if isinstance(weight_config, dict) and weight_config.get("cost_bps") is not None:
        return float(weight_config.get("cost_bps") or 0.0)
    return 0.0


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def _metric_value(frame: pd.DataFrame, metric_name: str) -> float:
    rows = frame[frame["metric_name"] == metric_name]
    if rows.empty:
        return 0.0
    return float(pd.to_numeric(rows["metric_value"], errors="coerce").mean())


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
