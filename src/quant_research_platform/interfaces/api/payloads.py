"""构造 API 使用的 JSON table 和 run summary payload。"""

from __future__ import annotations

import copy

import pandas as pd
from fastapi import HTTPException

from quant_research_platform.artifacts import LoadedEvaluationSet

from .models import RunIndexRecord, RunRef


def run_payload(run: LoadedEvaluationSet, ref: RunRef) -> dict[str, object]:
    """把已加载 run artifacts 压缩成前端 run detail/list 可用摘要。"""

    summary = run.summary
    config = run.config
    metrics = run.tables["evaluation_metrics_summary"]
    combinations = run.tables["combinations"]
    run_id = str(summary["run_id"])
    base = _base_evaluation(summary)
    return {
        "source_alias": ref.source_alias,
        "run_id": run_id,
        "public_run_id": ref.public_run_id,
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "domain": str(base.get("domain", "evaluation")),
        "data_stage": str(summary["data_stage"]),
        "uses_external_data": bool(base.get("uses_external_data", False)),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "label_columns": label_columns(summary),
        "return_source_column": return_source_column(summary),
        "weight_method": weight_method(config),
        "portfolio_mode": portfolio_mode(config),
        "n_combinations": int(summary["n_combinations"]),
        "breakdown_columns": breakdown_columns(summary),
        "combinations": table_payload(combinations),
        "mean_final_return": metric_value(metrics, "final_cumulative_net_return"),
        "mean_turnover": metric_value(metrics, "mean_turnover"),
        "total_cost": metric_value(metrics, "total_cost"),
        "sharpe_like": metric_value(metrics, "sharpe_like"),
        "sortino_like": metric_value(metrics, "sortino_like"),
        "max_drawdown": metric_value(metrics, "max_drawdown"),
        "win_rate": metric_value(metrics, "win_rate"),
        "mean_ic": metric_value(metrics, "mean_ic"),
        "mean_rankic": metric_value(metrics, "mean_rankic"),
        "icir": metric_value(metrics, "icir"),
        "rankicir": metric_value(metrics, "rankicir"),
    }


def run_index_payload(record: RunIndexRecord) -> dict[str, object]:
    """把轻量 run index record 压缩成前端 run list 摘要。"""

    summary = record.summary
    config = record.config
    metrics = record.evaluation_metrics_summary
    run_id = str(summary["run_id"])
    base = _base_evaluation(summary)
    return {
        "source_alias": record.ref.source_alias,
        "run_id": run_id,
        "public_run_id": record.ref.public_run_id,
        "project": str(summary["project"]),
        "kind": str(summary["kind"]),
        "domain": str(base.get("domain", "evaluation")),
        "data_stage": str(summary["data_stage"]),
        "uses_external_data": bool(base.get("uses_external_data", False)),
        "n_rows_evaluated": int(summary["n_rows_evaluated"]),
        "label_columns": label_columns(summary),
        "return_source_column": return_source_column(summary),
        "weight_method": weight_method(config),
        "portfolio_mode": portfolio_mode(config),
        "n_combinations": int(summary["n_combinations"]),
        "breakdown_columns": breakdown_columns(summary),
        "mean_final_return": metric_value(metrics, "final_cumulative_net_return"),
        "mean_turnover": metric_value(metrics, "mean_turnover"),
        "total_cost": metric_value(metrics, "total_cost"),
        "sharpe_like": metric_value(metrics, "sharpe_like"),
        "sortino_like": metric_value(metrics, "sortino_like"),
        "max_drawdown": metric_value(metrics, "max_drawdown"),
        "win_rate": metric_value(metrics, "win_rate"),
        "mean_ic": metric_value(metrics, "mean_ic"),
        "mean_rankic": metric_value(metrics, "mean_rankic"),
        "icir": metric_value(metrics, "icir"),
        "rankicir": metric_value(metrics, "rankicir"),
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
        "report": None if run.report is None else {"markdown": run.report},
        "comparison": comparison_payload,
        "display": runtime_display_payload(run.manifest["display"], ref),
        "combination_summaries": combination_summaries_payload(run.tables),
        "tables": {
            metric_name: table_payload(frame)
            for metric_name, frame in run.tables.items()
            if metric_name in table_names
        },
    }


def runtime_display_payload(
    display: object,
    ref: RunRef,
) -> dict[str, object]:
    """Add API-runtime preview URLs to a portable artifact display manifest."""

    if not isinstance(display, dict):
        raise HTTPException(status_code=422, detail="manifest display invalid")
    payload = copy.deepcopy(display)
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise HTTPException(status_code=422, detail="manifest display.tables invalid")
    base_path = f"/api/runs/{ref.source_alias}/{ref.run_id}"
    for table in tables:
        if not isinstance(table, dict):
            raise HTTPException(
                status_code=422,
                detail="manifest display.tables entries invalid",
            )
        key = table.get("key")
        if not isinstance(key, str) or not key:
            raise HTTPException(
                status_code=422,
                detail="manifest display table key invalid",
            )
        table["preview_url"] = f"{base_path}/tables/{key}/preview"
    downloads = payload.get("downloads")
    if isinstance(downloads, dict):
        items = downloads.get("items")
        if isinstance(items, list):
            items.append(
                {
                    "kind": "dynamic",
                    "key": "comparison",
                    "label": "Comparison comparison.csv",
                    "path": "comparison.csv",
                    "download_url": f"{base_path}/artifacts/dynamic:comparison",
                }
            )
        presets = downloads.get("presets")
        if isinstance(presets, list):
            for preset in presets:
                if isinstance(preset, dict) and preset.get("key") == "all":
                    preset_items = preset.get("items")
                    if isinstance(preset_items, list):
                        preset_items.append({"kind": "dynamic", "key": "comparison"})
    return payload


def evidence_summary_payload(
    summary: dict[str, object],
    ref: RunRef,
) -> dict[str, object]:
    payload = dict(summary)
    base = _base_evaluation(summary)
    payload["domain"] = str(base.get("domain", "evaluation"))
    payload["uses_external_data"] = bool(base.get("uses_external_data", False))
    payload["source_alias"] = ref.source_alias
    payload["run_id"] = ref.run_id
    payload["public_run_id"] = ref.public_run_id
    return payload


def evidence_load_table_names(manifest: dict[str, object]) -> set[str]:
    """返回构造 evidence payload 所需的 artifact tables。"""

    return {"combinations", "evaluation_metrics_summary", "diagnostic_summary"}


def evidence_payload_table_names(manifest: dict[str, object]) -> set[str]:
    """返回首屏需要随 payload 下发的 table payloads。"""

    return {"combinations"}


def display_chart_table_names(manifest: dict[str, object]) -> set[str]:
    """从 manifest.display.charts 读取首屏 table 名称。"""

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
        table_name = chart.get("source_table_key")
        if isinstance(table_name, str) and table_name:
            table_names.add(table_name)
    return table_names


def combination_summaries_payload(
    tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """在不改 artifact 的前提下构造 API 可见的 combination summary。"""

    combinations = tables.get("combinations")
    if combinations is None:
        return table_payload(pd.DataFrame())
    summary = combinations.copy()
    for column in COMBINATION_SUMMARY_METRIC_COLUMNS:
        if column in summary.columns:
            summary = summary.drop(columns=column)
    metric_summary = _wide_evaluation_metrics(tables.get("evaluation_metrics_summary"))
    if not metric_summary.empty:
        summary = summary.merge(metric_summary, on="combination_id", how="left")
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
    "kernel",
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


def _wide_evaluation_metrics(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {"combination_id", "metric_name", "metric_value"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    values = frame[["combination_id", "metric_name", "metric_value"]].copy()
    values["metric_value"] = pd.to_numeric(values["metric_value"], errors="coerce")
    return (
        values.pivot_table(
            index="combination_id",
            columns="metric_name",
            values="metric_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
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


def weight_method(config: dict[str, object]) -> str:
    """从 summary 的 weight_config 中读取 weighting method。"""

    configs = config.get("weight_configs")
    if not isinstance(configs, list):
        evaluation_spec = config.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            configs = evaluation_spec.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and isinstance(first.get("method"), str):
            method = first["method"]
            if method:
                return method
    raise HTTPException(status_code=422, detail="config weight_configs invalid")


def portfolio_mode(config: dict[str, object]) -> str:
    """从 summary 的 weight_config 中读取 portfolio mode。"""

    modes = config.get("portfolio_modes")
    if not isinstance(modes, list):
        evaluation_spec = config.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            modes = evaluation_spec.get("portfolio_modes")
    if isinstance(modes, list) and modes:
        normalized = [str(mode) for mode in modes if mode]
        if len(set(normalized)) > 1:
            return "multiple"
        if normalized:
            return normalized[0]
    configs = config.get("weight_configs")
    if not isinstance(configs, list):
        evaluation_spec = config.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            configs = evaluation_spec.get("weight_configs")
    if isinstance(configs, list) and configs:
        first = configs[0]
        if isinstance(first, dict) and isinstance(first.get("portfolio_mode"), str):
            mode = first["portfolio_mode"]
            if mode:
                return mode
    raise HTTPException(status_code=422, detail="config weight_configs invalid")


def label_columns(summary: dict[str, object]) -> list[str]:
    """从 summary schema 中读取 label columns。"""

    schema = _base_evaluation(summary)
    if isinstance(schema, dict) and isinstance(schema.get("label_columns"), list):
        return [str(value) for value in schema["label_columns"]]
    return []


def return_source_column(summary: dict[str, object]) -> str | None:
    """从 summary schema 中读取收益来源 column。"""

    schema = _base_evaluation(summary)
    if isinstance(schema, dict) and schema.get("return_source_column") is not None:
        return str(schema["return_source_column"])
    return None


def breakdown_columns(summary: dict[str, object]) -> list[str]:
    """从 summary 中读取已生成的 time breakdown columns。"""

    value = _base_evaluation(summary).get("breakdown_columns")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def mean_numeric(frame: pd.DataFrame, column: str) -> float:
    """按数值列计算 mean，非数值输入先转成 NaN。"""

    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def metric_value(frame: pd.DataFrame, metric_name: str) -> float:
    if frame.empty or "metric_name" not in frame.columns:
        return 0.0
    rows = frame[frame["metric_name"] == metric_name]
    if rows.empty:
        return 0.0
    return float(pd.to_numeric(rows["metric_value"], errors="coerce").mean())


def _base_evaluation(summary: dict[str, object]) -> dict[str, object]:
    value = summary.get("base_evaluation")
    return value if isinstance(value, dict) else {}


def final_cumulative_net_return_column(frame: pd.DataFrame) -> str:
    """返回当前 final return artifact column。"""

    return "final_cumulative_net_return"


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    """按数值列计算 sum，非数值输入先转成 NaN。"""

    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def min_numeric(frame: pd.DataFrame, column: str) -> float:
    """按数值列计算 min，非数值输入先转成 NaN。"""

    return float(pd.to_numeric(frame[column], errors="coerce").min())
