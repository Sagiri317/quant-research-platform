"""Shared API request and run identity models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class CreateRunPayload(BaseModel):
    """User-uploaded panel and metadata for one local evaluation set."""

    run_id: str
    project: str
    kind: str = "factor"
    domain: str
    data_stage: str = "uploaded_panel"
    uses_external_data: bool = True
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"
    feature_columns: list[str] = Field(default_factory=lambda: ["factor"])
    label_columns: list[str] = Field(default_factory=lambda: ["forward_return"])
    pnl_return_column: str | None = None
    segment_columns: list[str] = Field(default_factory=list)
    time_breakdowns: list[str] = Field(default_factory=list)
    engine: str = "dataframe"
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
    """Selected files to download for one run."""

    items: list[str]


@dataclass(frozen=True)
class RunRoot:
    """One configured directory of local evaluation-set artifacts."""

    source_alias: str
    path: Path


@dataclass(frozen=True)
class RunRef:
    """Resolved run identity and artifact directory."""

    source_alias: str
    run_id: str
    path: Path

    @property
    def public_run_id(self) -> str:
        """返回前端和下载文件名使用的 source/run 形式标识。"""

        return f"{self.source_alias}/{self.run_id}"


@dataclass(frozen=True)
class RunRecord:
    """Loaded evaluation set with its public identity."""

    ref: RunRef
    artifacts: Any


@dataclass(frozen=True)
class RunIndexRecord:
    """Lightweight run index row for run lists and comparison tables."""

    ref: RunRef
    manifest: dict[str, object]
    summary: dict[str, object]
    ic_summary: pd.DataFrame
    performance_summary: pd.DataFrame
