"""提供本地 API 使用的动态 run comparison tables。"""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from quant_research_platform.artifacts import LoadedEvaluationSet
from quant_research_platform.interfaces.api.models import (
    RunIndexRecord,
    RunRecord,
    RunRef,
)
from quant_research_platform.interfaces.api.payloads import (
    metric_value,
    weight_method,
)


def comparison_frame(records: list[RunRecord]) -> pd.DataFrame:
    """从完整 artifact records 构造带排名的 comparison DataFrame。"""

    rows = []
    for record in records:
        try:
            rows.append(comparison_record(record.ref, record.artifacts))
        except HTTPException:
            continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(rows).sort_values("public_run_id")
    return add_metric_ranks(frame.reset_index(drop=True))


def comparison_csv_bytes(records: list[RunRecord]) -> bytes:
    """把完整 artifact comparison table 编码为 UTF-8 CSV bytes。"""

    frame = comparison_frame(records) if records else pd.DataFrame()
    return frame.to_csv(index=False).encode("utf-8")


def comparison_index_frame(records: list[RunIndexRecord]) -> pd.DataFrame:
    """从轻量 run index 构造带排名的 comparison DataFrame。"""

    rows = []
    for record in records:
        try:
            rows.append(comparison_index_record(record))
        except HTTPException:
            continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(rows).sort_values("public_run_id")
    return add_metric_ranks(frame.reset_index(drop=True))


def comparison_index_csv_bytes(records: list[RunIndexRecord]) -> bytes:
    """把轻量 run index comparison table 编码为 UTF-8 CSV bytes。"""

    frame = comparison_index_frame(records) if records else pd.DataFrame()
    return frame.to_csv(index=False).encode("utf-8")


def comparison_record(
    ref: RunRef,
    run: LoadedEvaluationSet,
) -> dict[str, object]:
    """提取一个已加载 run 的 comparison row。"""

    summary = run.summary
    metrics = run.tables["evaluation_metrics_summary"]
    return {
        "source_alias": ref.source_alias,
        "public_run_id": ref.public_run_id,
        "run_id": ref.run_id,
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "weight_method": weight_method(run.config),
        "mean_ic": metric_value(metrics, "mean_ic"),
        "mean_rankic": metric_value(metrics, "mean_rankic"),
        "mean_final_return": metric_value(metrics, "final_cumulative_net_return"),
        "mean_turnover": metric_value(metrics, "mean_turnover"),
        "total_cost": metric_value(metrics, "total_cost"),
        "sharpe_like": metric_value(metrics, "sharpe_like"),
        "sortino_like": metric_value(metrics, "sortino_like"),
        "max_drawdown": metric_value(metrics, "max_drawdown"),
        "win_rate": metric_value(metrics, "win_rate"),
    }


def comparison_index_record(record: RunIndexRecord) -> dict[str, object]:
    """提取一个轻量 run index 的 comparison row。"""

    summary = record.summary
    metrics = record.evaluation_metrics_summary
    return {
        "source_alias": record.ref.source_alias,
        "public_run_id": record.ref.public_run_id,
        "run_id": record.ref.run_id,
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "weight_method": weight_method(record.config),
        "mean_ic": metric_value(metrics, "mean_ic"),
        "mean_rankic": metric_value(metrics, "mean_rankic"),
        "mean_final_return": metric_value(metrics, "final_cumulative_net_return"),
        "mean_turnover": metric_value(metrics, "mean_turnover"),
        "total_cost": metric_value(metrics, "total_cost"),
        "sharpe_like": metric_value(metrics, "sharpe_like"),
        "sortino_like": metric_value(metrics, "sortino_like"),
        "max_drawdown": metric_value(metrics, "max_drawdown"),
        "win_rate": metric_value(metrics, "win_rate"),
    }


def add_metric_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """为核心指标添加 deterministic rank columns，供页面排序和横向比较。"""

    ranked = frame.copy()
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
