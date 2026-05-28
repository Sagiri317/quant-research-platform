"""从 API payload 创建本地 uploaded-panel evaluation runs。"""

from __future__ import annotations

import re

import pandas as pd
from fastapi import HTTPException

from quant_research_platform.datasets.csv_panel import read_factor_panel_csv_text
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    EvaluationSpec,
    TensorBatchConfig,
    WeightConfig,
)
from quant_research_platform.interfaces.api.models import (
    CreateRunPayload,
    RunRef,
    RunRoot,
)
from quant_research_platform.interfaces.api.payloads import run_payload
from quant_research_platform.interfaces.api.services.run_repository import (
    load_artifacts_or_422,
)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def create_uploaded_run(
    run_roots: list[RunRoot],
    payload: CreateRunPayload,
) -> dict[str, object]:
    """解析用户上传 CSV，运行本地 evaluation，并返回前端 run payload。"""

    if not run_roots:
        raise HTTPException(status_code=422, detail="没有配置 run 输出目录。")
    if not RUN_ID_PATTERN.fullmatch(payload.run_id):
        raise HTTPException(
            status_code=422,
            detail="run_id 只能包含英文字母、数字、下划线或连字符。",
        )

    root = run_roots[0]
    output_dir = root.path / payload.run_id
    if output_dir.exists():
        raise HTTPException(status_code=409, detail="run_id 已存在，请换一个运行编号。")

    try:
        panel = read_factor_panel_csv_text(payload.csv_text)
        result = run_evaluation(
            EvaluationSpec(
                panel=panel,
                timestamp_column=payload.timestamp_column,
                asset_column=payload.asset_column,
                feature_columns=tuple(payload.feature_columns),
                label_columns=tuple(payload.label_columns),
                metadata={
                    "run_id": payload.run_id,
                    "project": payload.project,
                    "kind": payload.kind,
                    "domain": payload.domain,
                    "data_stage": "uploaded_panel",
                    "uses_external_data": payload.uses_external_data,
                },
                domain_notes={
                    "boundary": payload.boundary or default_upload_boundary()
                },
                n_groups=payload.n_groups,
                weight_configs=(
                    WeightConfig(
                        method=payload.weight_method,
                        direction=payload.direction,
                        portfolio_mode=payload.portfolio_mode,
                        top_quantile=payload.top_quantile,
                        bottom_quantile=payload.bottom_quantile,
                        cost_bps=payload.cost_bps,
                        risk_column=payload.risk_column,
                    ),
                ),
                segment_columns=tuple(payload.segment_columns),
                time_breakdowns=tuple(payload.time_breakdowns),
                kernel=payload.kernel,
                return_source_column=payload.return_source_column,
                tensor_config=TensorBatchConfig(
                    max_memory_mb=payload.tensor_max_memory_mb
                ),
                output_dir=output_dir,
            )
        )
    except (pd.errors.ParserError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=user_facing_error(exc)) from exc

    ref = RunRef(root.source_alias, payload.run_id, result.artifacts.output_dir)
    return run_payload(load_artifacts_or_422(ref.path), ref)


def default_upload_boundary() -> str:
    """返回上传面板 evaluation 的默认数据与交易安全边界说明。"""

    return (
        "User-uploaded panel evaluated locally; the platform did not download "
        "data, connect external APIs, create credentials, paper trade, live "
        "trade, or provide investment advice."
    )


def user_facing_error(exc: Exception) -> str:
    """把内部解析或 evaluation 异常改写成前端可读中文错误。"""

    message = str(exc)
    if isinstance(exc, pd.errors.ParserError):
        return "CSV 解析失败，请检查分隔符、引号和表头。"
    if "panel is missing columns" in message or "missing columns" in message:
        missing = message.split(":", 1)[-1].strip().rstrip(".")
        return f"CSV 缺少必需列：{missing}"
    if "Portfolio cumulative returns require return_source_column" in message:
        return (
            "CSV 缺少收益来源列；请选择 return_source_column 或提供 forward_return_1d。"
        )
    if "must be numeric" in message or "could not convert" in message:
        return "因子列、标签列和收益来源列必须是数值。"
    if "csv_text must contain at least one data row" in message:
        return "CSV 至少需要一行数据。"
    return f"回测创建失败：{message}"
