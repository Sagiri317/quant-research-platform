"""JSON table and run-summary payload builders for the API."""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from quant_research_platform.artifacts import LoadedEvaluationSet

from .models import RunIndexRecord, RunRef


def run_payload(run: LoadedEvaluationSet, ref: RunRef) -> dict[str, object]:
    """把已加载 run artifacts 压缩成前端 run detail/list 可用摘要。"""

    summary = run.summary
    performance = run.tables["performance_summary"]
    ic_summary = run.tables["ic_summary"]
    combinations = run.tables["combinations"]
    run_id = str(summary["evaluation_id"])
    return {
        "source_alias": ref.source_alias,
        "run_id": run_id,
        "evaluation_id": run_id,
        "public_run_id": ref.public_run_id,
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "domain": str(summary["domain"]),
        "data_stage": str(summary["data_stage"]),
        "uses_external_data": bool(summary["uses_external_data"]),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "label_columns": label_columns(summary),
        "pnl_return_column": pnl_return_column(summary),
        "weight_method": weight_method(summary),
        "portfolio_mode": portfolio_mode(summary),
        "n_combinations": int(summary["n_combinations"]),
        "breakdown_columns": breakdown_columns(summary),
        "combinations": table_payload(combinations),
        "mean_final_return": mean_numeric(
            performance,
            final_cumulative_net_return_column(performance),
        ),
        "mean_turnover": mean_numeric(performance, "mean_turnover"),
        "total_cost": sum_numeric(performance, "total_cost"),
        "sharpe_like": mean_numeric(performance, "sharpe_like"),
        "sortino_like": mean_numeric(performance, "sortino_like"),
        "max_drawdown": min_numeric(performance, "max_drawdown"),
        "win_rate": mean_numeric(performance, "win_rate"),
        "mean_ic": mean_numeric(ic_summary, "mean_ic"),
        "mean_rankic": mean_numeric(ic_summary, "mean_rankic"),
        "icir": mean_numeric(ic_summary, "icir"),
        "rankicir": mean_numeric(ic_summary, "rankicir"),
    }


def run_index_payload(record: RunIndexRecord) -> dict[str, object]:
    """把轻量 run index record 压缩成前端 run list 摘要。"""

    summary = record.summary
    run_id = str(summary["evaluation_id"])
    return {
        "source_alias": record.ref.source_alias,
        "run_id": run_id,
        "evaluation_id": run_id,
        "public_run_id": record.ref.public_run_id,
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "domain": str(summary["domain"]),
        "data_stage": str(summary["data_stage"]),
        "uses_external_data": bool(summary["uses_external_data"]),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "label_columns": label_columns(summary),
        "pnl_return_column": pnl_return_column(summary),
        "weight_method": weight_method(summary),
        "portfolio_mode": portfolio_mode(summary),
        "n_combinations": int(summary["n_combinations"]),
        "breakdown_columns": breakdown_columns(summary),
        "mean_final_return": mean_numeric(
            record.performance_summary,
            final_cumulative_net_return_column(record.performance_summary),
        ),
        "mean_turnover": mean_numeric(record.performance_summary, "mean_turnover"),
        "total_cost": sum_numeric(record.performance_summary, "total_cost"),
        "sharpe_like": mean_numeric(record.performance_summary, "sharpe_like"),
        "sortino_like": mean_numeric(record.performance_summary, "sortino_like"),
        "max_drawdown": min_numeric(record.performance_summary, "max_drawdown"),
        "win_rate": mean_numeric(record.performance_summary, "win_rate"),
        "mean_ic": mean_numeric(record.ic_summary, "mean_ic"),
        "mean_rankic": mean_numeric(record.ic_summary, "mean_rankic"),
        "icir": mean_numeric(record.ic_summary, "icir"),
        "rankicir": mean_numeric(record.ic_summary, "rankicir"),
    }


def run_evidence_payload(
    run: LoadedEvaluationSet,
    comparison_payload: dict[str, object],
    *,
    ref: RunRef,
    eager_table_names: set[str] | None = None,
) -> dict[str, object]:
    """构造 evidence 页面一次加载所需的 report、comparison 和 tables。"""

    table_names = (
        evidence_payload_table_names(run.manifest)
        if eager_table_names is None
        else eager_table_names
    )
    return {
        "summary": evidence_summary_payload(run.summary, ref),
        "report": {"markdown": run.report},
        "comparison": comparison_payload,
        "display": run.manifest["display"],
        "combination_summaries": combination_summaries_payload(run.tables),
        "tables": {
            metric_name: table_payload(frame)
            for metric_name, frame in run.tables.items()
            if metric_name in table_names
        },
    }


def evidence_summary_payload(
    summary: dict[str, object],
    ref: RunRef,
) -> dict[str, object]:
    payload = dict(summary)
    payload["source_alias"] = ref.source_alias
    payload["run_id"] = ref.run_id
    payload["public_run_id"] = ref.public_run_id
    return payload


def evidence_load_table_names(manifest: dict[str, object]) -> set[str]:
    """Return the artifact tables needed to build the evidence payload."""

    return {
        "combinations",
        "ic_summary",
        "performance_summary",
        *display_chart_table_names(manifest),
    }


def evidence_payload_table_names(manifest: dict[str, object]) -> set[str]:
    """Return table payloads that should be shipped with the first view."""

    return {"combinations", *display_chart_table_names(manifest)}


def display_chart_table_names(manifest: dict[str, object]) -> set[str]:
    """Read first-view table names from manifest.display.charts."""

    display = manifest.get("display")
    if not isinstance(display, dict):
        return set()
    charts = display.get("charts")
    if not isinstance(charts, list):
        return set()
    table_names: set[str] = set()
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        table_name = chart.get("table")
        if isinstance(table_name, str) and table_name:
            table_names.add(table_name)
    return table_names


def combination_summaries_payload(
    tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Build the API-facing per-combination summary without changing artifacts."""

    combinations = tables.get("combinations")
    if combinations is None:
        return table_payload(pd.DataFrame())
    summary = combinations.copy()
    for column in COMBINATION_SUMMARY_METRIC_COLUMNS:
        if column in summary.columns:
            summary = summary.drop(columns=column)
    ic_summary = _aggregate_combination_summary(
        tables.get("ic_summary"),
        {
            "mean_ic": "mean",
            "icir": "mean",
            "mean_rankic": "mean",
            "rankicir": "mean",
        },
    )
    performance_summary = _aggregate_combination_summary(
        tables.get("performance_summary"),
        {
            "final_cumulative_net_return": "mean",
            "sharpe_like": "mean",
            "sortino_like": "mean",
            "max_drawdown": "min",
            "win_rate": "mean",
            "mean_turnover": "mean",
            "total_cost": "sum",
        },
    )
    for frame in (performance_summary, ic_summary):
        if frame.empty:
            continue
        summary = summary.merge(frame, on="combination_id", how="left")
    ordered_columns = [
        column for column in COMBINATION_SUMMARY_COLUMNS if column in summary.columns
    ]
    remaining_columns = [
        column for column in summary.columns if column not in ordered_columns
    ]
    return table_payload(summary[ordered_columns + remaining_columns])


COMBINATION_SUMMARY_DIMENSION_COLUMNS = [
    "combination_id",
    "feature",
    "label",
    "weight_id",
    "cost_id",
    "weight_method",
    "direction",
    "portfolio_mode",
    "cost_bps",
]

COMBINATION_SUMMARY_METRIC_COLUMNS = [
    "final_cumulative_net_return",
    "sharpe_like",
    "sortino_like",
    "max_drawdown",
    "win_rate",
    "mean_ic",
    "icir",
    "mean_rankic",
    "rankicir",
    "mean_turnover",
    "total_cost",
]

COMBINATION_SUMMARY_COLUMNS = [
    *COMBINATION_SUMMARY_DIMENSION_COLUMNS,
    *COMBINATION_SUMMARY_METRIC_COLUMNS,
]


def _aggregate_combination_summary(
    frame: pd.DataFrame | None,
    aggregations: dict[str, str],
) -> pd.DataFrame:
    if frame is None or frame.empty or "combination_id" not in frame.columns:
        return pd.DataFrame()
    available_aggregations = {
        column: aggregation
        for column, aggregation in aggregations.items()
        if column in frame.columns
    }
    if not available_aggregations:
        return pd.DataFrame()
    normalized = frame[["combination_id", *available_aggregations]].copy()
    for column in available_aggregations:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return (
        normalized.groupby("combination_id", as_index=False)
        .agg(available_aggregations)
        .reset_index(drop=True)
    )


def metric_payload(
    run: LoadedEvaluationSet,
    metric_name: str,
    *,
    combination_id: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, object]:
    """从 loaded run 中取一个 metric table 并转成分页 JSON payload。"""

    frame = run.tables.get(metric_name)
    if frame is None:
        raise HTTPException(status_code=404, detail="metric not found")
    if combination_id and "combination_id" in frame.columns:
        frame = frame[frame["combination_id"].astype(str) == combination_id]
    return table_payload(frame, limit=limit, offset=offset)


def table_payload(
    frame: pd.DataFrame,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, object]:
    """把 DataFrame 转成前端表格 JSON，统一处理时间列和分页参数。"""

    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be nonnegative")
    if limit is not None and limit < 1:
        raise HTTPException(status_code=422, detail="limit must be positive")
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    total_rows = len(normalized)
    if limit is not None:
        normalized = normalized.iloc[offset : offset + limit]
    elif offset:
        normalized = normalized.iloc[offset:]
    return {
        "columns": list(normalized.columns),
        "rows": normalized.to_dict(orient="records"),
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
    }


def weight_method(summary: dict[str, object]) -> str:
    """从 summary 的 weight_config 中读取 weighting method。"""

    configs = summary.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and isinstance(first.get("method"), str):
            method = first["method"]
            if method:
                return method
    raise HTTPException(status_code=422, detail="summary weight_configs invalid")


def portfolio_mode(summary: dict[str, object]) -> str:
    """从 summary 的 weight_config 中读取 portfolio mode。"""

    configs = summary.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and isinstance(first.get("portfolio_mode"), str):
            mode = first["portfolio_mode"]
            if mode:
                return mode
    raise HTTPException(status_code=422, detail="summary weight_configs invalid")


def label_columns(summary: dict[str, object]) -> list[str]:
    """从 summary schema 中读取 label columns。"""

    schema = summary.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("label_columns"), list):
        return [str(value) for value in schema["label_columns"]]
    return []


def pnl_return_column(summary: dict[str, object]) -> str | None:
    """从 summary schema 中读取 PnL return column。"""

    schema = summary.get("schema")
    if isinstance(schema, dict) and schema.get("pnl_return_column") is not None:
        return str(schema["pnl_return_column"])
    return None


def breakdown_columns(summary: dict[str, object]) -> list[str]:
    """从 summary 中读取已生成的 time breakdown columns。"""

    value = summary.get("breakdown_columns")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def mean_numeric(frame: pd.DataFrame, column: str) -> float:
    """按数值列计算 mean，非数值输入先转成 NaN。"""

    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def final_cumulative_net_return_column(frame: pd.DataFrame) -> str:
    """Return the current final return artifact column."""

    return "final_cumulative_net_return"


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    """按数值列计算 sum，非数值输入先转成 NaN。"""

    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def min_numeric(frame: pd.DataFrame, column: str) -> float:
    """按数值列计算 min，非数值输入先转成 NaN。"""

    return float(pd.to_numeric(frame[column], errors="coerce").min())
