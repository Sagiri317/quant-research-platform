"""FastAPI entry point for the QRP 0.1 spec-first API."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.universes import UniverseDefinition
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
        catalog = FactorCatalog.bootstrap_default()
        return {
            "api_version": "0.1",
            "items": [item.to_manifest() for item in catalog.list()],
        }

    @app.get("/api/0.1/catalog/labels")
    def catalog_labels() -> dict[str, object]:
        catalog = LabelCatalog.bootstrap_default()
        return {
            "api_version": "0.1",
            "items": [item.to_manifest() for item in catalog.list()],
        }

    @app.get("/api/0.1/catalog/universes")
    def catalog_universes() -> dict[str, object]:
        universe = UniverseDefinition(
            universe_id="a_share_default",
            market_id=MarketId("a_share"),
            membership_column="in_universe",
        )
        return {
            "api_version": "0.1",
            "items": [
                {
                    "definition_version": universe.definition_version,
                    "universe_id": universe.universe_id,
                    "market_id": str(universe.market_id),
                    "membership_column": universe.membership_column,
                }
            ],
        }

    @app.post("/api/0.1/sources/{source_id}/interim")
    def build_source_interim(source_id: str) -> dict[str, object]:
        if source_id != "csmar":
            raise ApiError(
                error_code="invalid_source_id",
                message="unsupported source id for API 0.1",
                field="source_id",
            )
        return {"api_version": "0.1", "source": source_id, "status": "planned"}

    @app.post("/api/0.1/markets/{market_id}/processed-data")
    def build_processed_data(market_id: str) -> dict[str, object]:
        if market_id != "a_share":
            raise ApiError(
                error_code="invalid_market_id",
                message="unsupported market id for API 0.1",
                field="market_id",
            )
        return {"api_version": "0.1", "market": market_id, "status": "planned"}

    @app.post("/api/0.1/research-datasets")
    def build_research_dataset() -> dict[str, object]:
        return {"api_version": "0.1", "status": "planned"}

    @app.post("/api/0.1/workflows")
    def run_workflow() -> dict[str, object]:
        return {"api_version": "0.1", "status": "planned"}

    @app.get("/api/0.1/reports/{run_id}")
    def report_detail(run_id: str) -> dict[str, object]:
        return {"api_version": "0.1", "run_id": run_id, "status": "planned"}


app = create_app()
