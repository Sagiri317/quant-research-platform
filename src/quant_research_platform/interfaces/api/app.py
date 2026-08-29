"""FastAPI entry point for the QRP 0.1 spec-first API."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.interfaces import queries
from quant_research_platform.interfaces.api.errors import (
    ApiError,
    register_api_error_handlers,
)
from quant_research_platform.interfaces.models import (
    FactorCatalogResponse,
    LabelCatalogResponse,
    ResearchDatasetRequirementsResponse,
)
from quant_research_platform.profiles.research import research_profile

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
    """Create the QRP 0.1 API app."""

    app = FastAPI(
        title="Quant Research Platform",
        version="0.1.0",
        description=(
            "Spec-first local API for catalog queries and research dataset "
            "requirements."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    register_api_error_handlers(app)
    _include_api_01_skeleton(app)
    return app


def _include_api_01_skeleton(app: FastAPI) -> None:
    @app.get(
        "/api/0.1/catalog/factors",
        operation_id="list_catalog_factors",
    )
    def catalog_factors() -> FactorCatalogResponse:
        return queries.list_factors(research_profile())

    @app.get(
        "/api/0.1/catalog/labels",
        operation_id="list_catalog_labels",
    )
    def catalog_labels() -> LabelCatalogResponse:
        return queries.list_labels(research_profile())

    @app.get(
        "/api/0.1/research-datasets/{dataset_id}/requirements",
        operation_id="describe_research_dataset_requirements",
    )
    def describe_research_dataset_requirements(
        dataset_id: str,
    ) -> ResearchDatasetRequirementsResponse:
        try:
            return queries.describe_research_dataset_requirements(
                dataset_id,
                research_profile(),
            )
        except KeyError as exc:
            raise ApiError(
                error_code="unknown_dataset_id",
                status_code=404,
                field="dataset_id",
            ) from exc


app = create_app()
