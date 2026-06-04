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
from quant_research_platform.interfaces.api.models import DataProductInterimRequest
from quant_research_platform.storage.errors import WriteRejectedError

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
            "Spec-first local API for catalog queries, data product imports, "
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

    @app.get("/api/0.1/catalog/data-products")
    def catalog_data_products() -> dict[str, object]:
        return services.list_data_products()

    @app.post(
        "/api/0.1/data-products/a_share_daily_bar/interim",
        tags=["Data Products"],
        operation_id="create_data_product_interim",
    )
    def build_data_product_interim(
        request: DataProductInterimRequest | None = None,
    ) -> dict[str, object]:
        payload = request or DataProductInterimRequest()
        try:
            return services.validate_data_product_interim(
                "a_share_daily_bar",
                execution_mode=payload.execution_mode,
                run_id=payload.run_id,
            )
        except KeyError as exc:
            raise ApiError(
                error_code="unknown_data_product",
                message=str(exc),
                status_code=404,
                field="data_product_id",
            ) from exc
        except WriteRejectedError as exc:
            raise ApiError(
                error_code=exc.failure_code,
                message=exc.failure_code,
                status_code=409,
                details=exc.to_public_summary(),
            ) from exc
        except ValueError as exc:
            raise ApiError(
                error_code="invalid_execution_mode",
                message=str(exc),
                status_code=400,
                field="execution_mode",
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
