"""本地 evidence API 的 FastAPI 入口。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.interfaces.api.routes import (
    artifacts,
    comparison,
    downloads,
    factor_factory,
    health,
    markets,
    pipeline,
    runs,
    uploads,
)
from quant_research_platform.interfaces.api.services.context import ApiContext
from quant_research_platform.interfaces.api.services.run_repository import (
    DEFAULT_RUN_ROOT_ALIAS as _DEFAULT_RUN_ROOT_ALIAS,
)
from quant_research_platform.interfaces.api.services.run_repository import (
    normalize_run_roots,
    run_root,
    run_roots_from_env,
)

DEFAULT_RUN_ROOT_ALIAS = _DEFAULT_RUN_ROOT_ALIAS


def create_app(
    run_roots: list[tuple[str, str | Path]] | None = None,
    *,
    pipeline_output_root: str | Path | None = None,
    factor_factory_root: str | Path | None = None,
) -> FastAPI:
    """基于配置的 artifact run roots 创建本地 API app。"""

    app = FastAPI(
        title="Quant Research Platform",
        version="0.1.0",
        description=(
            "Local API over approved evaluation-set artifacts and "
            "user-uploaded CSV runs."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["DELETE", "GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    configured_roots = (
        normalize_run_roots(run_roots)
        if run_roots is not None
        else run_roots_from_env()
    )
    pipeline_root = (
        Path(pipeline_output_root)
        if pipeline_output_root is not None
        else Path("reports/private_real/pipeline_runs")
    )
    factor_factory_output_root = (
        Path(factor_factory_root)
        if factor_factory_root is not None
        else Path("reports/private_real/factor_factory")
    )
    if all(
        root.source_alias != "pipeline" for root in configured_roots
    ):
        configured_roots = [*configured_roots, run_root("pipeline", pipeline_root)]
    app.state.qrp_context = ApiContext(
        run_roots=configured_roots,
        pipeline_runs={},
        pipeline_output_root=pipeline_root,
        factor_factory_root=factor_factory_output_root,
    )
    for router in (
        health.router,
        markets.router,
        pipeline.router,
        runs.router,
        uploads.router,
        artifacts.router,
        downloads.router,
        comparison.router,
        factor_factory.router,
    ):
        app.include_router(router)
    return app


app = create_app()
