"""本地 evidence API 的 FastAPI 入口。"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.interfaces.api.errors import register_api_error_handlers
from quant_research_platform.interfaces.api.routes import (
    a_share_research,
    artifacts,
    comparison,
    downloads,
    factor_factory,
    factor_templates,
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
PIPELINE_OUTPUT_ROOT_ENV = "QRP_PIPELINE_OUTPUT_ROOT"
FACTOR_FACTORY_ROOT_ENV = "QRP_FACTOR_FACTORY_ROOT"
FACTOR_REBUILD_ROOT_ENV = "QRP_FACTOR_REBUILD_ROOT"
PREDICTION_HANDOFF_ROOT_ENV = "QRP_PREDICTION_HANDOFF_ROOT"
CORS_ORIGINS_ENV = "QRP_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def _cors_origins_from_env() -> list[str]:
    origins = list(DEFAULT_CORS_ORIGINS)
    env_value = os.environ.get(CORS_ORIGINS_ENV, "")
    for origin in (item.strip() for item in env_value.split(";")):
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def _pipeline_root_from_config(
    configured_roots,
    *,
    pipeline_output_root: str | Path | None,
    run_roots_were_explicit: bool,
) -> Path:
    if pipeline_output_root is not None:
        return Path(pipeline_output_root)
    env_value = os.environ.get(PIPELINE_OUTPUT_ROOT_ENV)
    if env_value:
        return Path(env_value)
    if run_roots_were_explicit:
        bases = [_run_root_project_base(root.path) for root in configured_roots]
        if bases and all(base.is_absolute() for base in bases):
            return Path(os.path.commonpath([str(base) for base in bases])) / (
                "reports/private_real/pipeline_runs"
            )
    return Path("reports/private_real/pipeline_runs")


def _run_root_project_base(path: Path) -> Path:
    parent = path.parent
    if parent.name == "reports":
        return parent.parent
    return parent


def _configured_root_path(configured_roots, source_alias: str) -> Path | None:
    for root in configured_roots:
        if root.source_alias == source_alias:
            return root.path
    return None


def create_app(
    run_roots: list[tuple[str, str | Path]] | None = None,
    *,
    pipeline_output_root: str | Path | None = None,
    factor_factory_root: str | Path | None = None,
    factor_rebuild_root: str | Path | None = None,
    prediction_handoff_root: str | Path | None = None,
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
        allow_origins=_cors_origins_from_env(),
        allow_methods=["DELETE", "GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    register_api_error_handlers(app)
    configured_roots = (
        normalize_run_roots(run_roots)
        if run_roots is not None
        else run_roots_from_env()
    )
    pipeline_root = _pipeline_root_from_config(
        configured_roots,
        pipeline_output_root=pipeline_output_root,
        run_roots_were_explicit=run_roots is not None,
    )
    factor_factory_output_root = (
        Path(factor_factory_root)
        if factor_factory_root is not None
        else Path(
            os.environ.get(
                FACTOR_FACTORY_ROOT_ENV,
                "reports/private_real/factor_factory",
            )
        )
    )
    factor_rebuild_output_root = (
        Path(factor_rebuild_root)
        if factor_rebuild_root is not None
        else Path(
            os.environ.get(
                FACTOR_REBUILD_ROOT_ENV,
                "reports/private_real/a_share_factor_rebuilds",
            )
        )
    )
    prediction_handoff_env = os.environ.get(PREDICTION_HANDOFF_ROOT_ENV)
    configured_prediction_root = _configured_root_path(
        configured_roots,
        "a_share_prediction",
    )
    prediction_handoff_output_root = (
        Path(prediction_handoff_root)
        if prediction_handoff_root is not None
        else configured_prediction_root
        if prediction_handoff_env is None and configured_prediction_root is not None
        else Path(
            prediction_handoff_env
            or "reports/private_real/a_share_prediction_handoffs"
        )
    )
    if all(root.source_alias != "pipeline" for root in configured_roots):
        configured_roots = [*configured_roots, run_root("pipeline", pipeline_root)]
    should_register_prediction_root = (
        run_roots is None
        or prediction_handoff_root is not None
        or prediction_handoff_env is not None
        or configured_prediction_root is not None
    )
    if should_register_prediction_root:
        configured_roots = [
            root
            for root in configured_roots
            if root.source_alias != "a_share_prediction"
        ]
        configured_roots = [
            *configured_roots,
            run_root("a_share_prediction", prediction_handoff_output_root),
        ]
    app.state.qrp_context = ApiContext(
        run_roots=configured_roots,
        pipeline_runs={},
        pipeline_output_root=pipeline_root,
        factor_factory_root=factor_factory_output_root,
        factor_rebuild_root=factor_rebuild_output_root,
        prediction_handoff_root=prediction_handoff_output_root,
    )
    for router in (
        health.router,
        markets.router,
        a_share_research.router,
        factor_templates.router,
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
