"""定义 API 请求模型和 run identity 模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CreateRunPayload(BaseModel):
    """用户上传 panel 生成本地 evaluation set 的请求。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    project: str
    kind: str = "factor"
    domain: str
    data_stage: Literal["uploaded_panel"] = "uploaded_panel"
    uses_external_data: bool = True
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"
    feature_columns: list[str] = Field(default_factory=lambda: ["factor"])
    label_columns: list[str] = Field(default_factory=lambda: ["forward_return"])
    return_source_column: str | None = None
    segment_columns: list[str] = Field(default_factory=list)
    time_breakdowns: list[str] = Field(default_factory=list)
    kernel: str = "dataframe"
    n_groups: int = 5
    weight_method: str = "equal_weight_quantile"
    direction: str = "long_high"
    portfolio_mode: str = "long_short"
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    cost_bps: float = 0.0
    risk_column: str | None = None
    portfolio_feature: str | None = None
    tensor_max_memory_mb: float | None = None
    boundary: str | None = None
    csv_text: str


class DownloadArtifactsPayload(BaseModel):
    """单个 run 的选中文件下载请求。"""

    items: list[str]


@dataclass(frozen=True)
class RunRoot:
    """一个已配置的本地 evaluation-set artifact 目录。"""

    source_alias: str
    path: Path


@dataclass(frozen=True)
class RunRef:
    """已解析的 run identity 和 artifact 目录。"""

    source_alias: str
    run_id: str
    path: Path

    @property
    def public_run_id(self) -> str:
        """返回前端和下载文件名使用的 source/run 形式标识。"""

        return f"{self.source_alias}/{self.run_id}"


@dataclass(frozen=True)
class RunRecord:
    """已加载 evaluation set 及其公开 identity。"""

    ref: RunRef
    artifacts: Any


@dataclass(frozen=True)
class RunIndexRecord:
    """用于 run 列表和 comparison 表的轻量 index 记录。"""

    ref: RunRef
    manifest: dict[str, object]
    config: dict[str, object]
    summary: dict[str, object]
    ic_summary: pd.DataFrame
    performance_summary: pd.DataFrame
