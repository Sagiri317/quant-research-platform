"""FastAPI entry point for the QRP 0.1 spec-first API."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.interfaces import services
from quant_research_platform.interfaces.api.errors import (
    ApiError,
    register_api_error_handlers,
)

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


def create_app() -> FastAPI:
    """Create the QRP 0.1 API app without legacy endpoint compatibility."""

    app = FastAPI(
        title="Quant Research Platform",
        version="0.1.0",
        description=(
            "Spec-first local API for catalog queries, source imports, "
            "market processed data, research datasets, workflows, and reports."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    register_api_error_handlers(app)
    _include_api_01_skeleton(app)
    return app


def _include_api_01_skeleton(app: FastAPI) -> None:
    @app.get("/api/0.1/catalog/factors")
    def catalog_factors() -> dict[str, object]:
        return services.list_factors()

    @app.get("/api/0.1/catalog/labels")
    def catalog_labels() -> dict[str, object]:
        return services.list_labels()

    @app.get("/api/0.1/catalog/universes")
    def catalog_universes() -> dict[str, object]:
        return services.list_universes()

    @app.post("/api/0.1/sources/{source_id}/interim")
    def build_source_interim(source_id: str) -> dict[str, object]:
        try:
            return services.validate_source_interim(source_id)
        except KeyError as exc:
            raise ApiError(
                error_code="invalid_source_id",
                message=str(exc),
                field="source_id",
            ) from exc

    @app.post("/api/0.1/markets/{market_id}/processed-data")
    def build_processed_data(market_id: str) -> dict[str, object]:
        try:
            return services.validate_market_processed(market_id)
        except KeyError as exc:
            raise ApiError(
                error_code="invalid_market_id",
                message=str(exc),
                field="market_id",
            ) from exc

    @app.post("/api/0.1/research-datasets")
    def build_research_dataset() -> dict[str, object]:
        return services.validate_research_dataset()

    @app.post("/api/0.1/workflows")
    def run_workflow() -> dict[str, object]:
        return services.validate_workflow_run()

    @app.get("/api/0.1/reports/{run_id}")
    def report_detail(run_id: str) -> dict[str, object]:
        try:
            return services.report_metadata(run_id)
        except ValueError as exc:
            raise ApiError(
                error_code="invalid_run_id",
                message=str(exc),
                field="run_id",
            ) from exc


app = create_app()
