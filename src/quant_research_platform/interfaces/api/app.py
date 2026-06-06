"""FastAPI entry point for the QRP 0.1 spec-first API."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.catalog.registry_validation import CatalogConsistencyError
from quant_research_platform.contracts.run_id import RunIdValidationError
from quant_research_platform.interfaces import services
from quant_research_platform.interfaces.api.errors import (
    ApiError,
    register_api_error_handlers,
)
from quant_research_platform.interfaces.api.models import DataProductInterimRunRequest
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
            "Spec-first local API for catalog queries, data product interim "
            "runs, and research dataset requirements."
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
    @app.get(
        "/api/0.1/catalog/factors",
        operation_id="list_catalog_factors",
    )
    def catalog_factors() -> dict[str, object]:
        return services.list_factors()

    @app.get(
        "/api/0.1/catalog/labels",
        operation_id="list_catalog_labels",
    )
    def catalog_labels() -> dict[str, object]:
        return services.list_labels()

    @app.get(
        "/api/0.1/catalog/universes",
        operation_id="list_catalog_universes",
    )
    def catalog_universes() -> dict[str, object]:
        return services.list_universes()

    @app.get(
        "/api/0.1/catalog/data-products",
        operation_id="list_catalog_data_products",
    )
    def catalog_data_products() -> dict[str, object]:
        return services.list_data_products()

    @app.post(
        "/api/0.1/data-products/{data_product_id}/interim-runs",
        tags=["Data Products"],
        operation_id="run_data_product_interim",
    )
    def run_data_product_interim(
        data_product_id: str,
        request: DataProductInterimRunRequest | None = None,
    ) -> dict[str, object]:
        payload = request or DataProductInterimRunRequest()
        try:
            return services.run_data_product_interim(
                data_product_id,
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
        except RunIdValidationError as exc:
            raise ApiError(
                error_code="invalid_run_id",
                message=str(exc),
                status_code=422,
                field="run_id",
            ) from exc
        except CatalogConsistencyError as exc:
            raise ApiError(
                error_code=exc.error_code,
                message=exc.error_code,
                status_code=500,
            ) from exc
        except NotImplementedError as exc:
            raise ApiError(
                error_code=str(exc),
                message=str(exc),
                status_code=501,
                field="execution_mode",
            ) from exc
        except ValueError as exc:
            raise ApiError(
                error_code="invalid_execution_mode",
                message=str(exc),
                status_code=400,
                field="execution_mode",
            ) from exc

    @app.get(
        "/api/0.1/research-datasets/{dataset_id}/requirements",
        operation_id="describe_research_dataset_requirements",
    )
    def describe_research_dataset_requirements(dataset_id: str) -> dict[str, object]:
        try:
            return services.describe_research_dataset_requirements(dataset_id)
        except KeyError as exc:
            raise ApiError(
                error_code="unknown_dataset_id",
                message=str(exc),
                status_code=404,
                field="dataset_id",
            ) from exc


app = create_app()
