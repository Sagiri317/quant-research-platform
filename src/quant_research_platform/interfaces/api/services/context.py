"""本地 API 共享 context 和 service helper。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from quant_research_platform.interfaces.api.models import RunRoot
from quant_research_platform.interfaces.api.services.comparison_service import (
    comparison_index_csv_bytes,
)
from quant_research_platform.interfaces.api.services.download_service import (
    ensure_download_item_exposed,
)
from quant_research_platform.interfaces.api.services.pipeline_service import (
    PipelineRunSummaryPayload,
)
from quant_research_platform.interfaces.api.services.run_repository import (
    discover_run_indexes,
)


@dataclass
class ApiContext:
    """本地 API routers 共享的运行时状态。"""

    run_roots: list[RunRoot]
    pipeline_runs: dict[str, PipelineRunSummaryPayload]
    pipeline_output_root: Path
    factor_factory_root: Path

    def download_cache_root(self, source_alias: str) -> Path:
        """解析指定 run source 的本地 ZIP download cache root。"""

        for root in self.run_roots:
            if root.source_alias == source_alias:
                return root.path.resolve().parent / "download_cache"
        raise HTTPException(
            status_code=404,
            detail=f"unknown run source: {source_alias}",
        )

    def comparison_index_records(self):
        """返回可参与 comparison 的 run index records。"""

        records = discover_run_indexes(self.run_roots)
        for record in records:
            ensure_download_item_exposed(record.manifest, "comparison.csv")
        return records

    def comparison_csv_bytes_for_download(self) -> bytes:
        """返回 direct 和 ZIP download 共用的 comparison CSV bytes。"""

        return comparison_index_csv_bytes(self.comparison_index_records())


def api_context(request: Request) -> ApiContext:
    """从 FastAPI app state 读取 typed API context。"""

    context = getattr(request.app.state, "qrp_context", None)
    if not isinstance(context, ApiContext):
        raise RuntimeError("QRP API context is not configured.")
    return context


__all__ = ["ApiContext", "api_context"]
