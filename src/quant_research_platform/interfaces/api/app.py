"""FastAPI entry point for the QRP spec-first API."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query
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
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    research_profile,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research

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


def create_app(*, project_root: Path | None = None) -> FastAPI:
    """Create the QRP API app."""

    app = FastAPI(
        title="Quant Research Platform",
        version="0.1.0",
        description=(
            "Read-only local catalog and explicitly selected "
            "published research evidence."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    register_api_error_handlers(app)
    _include_routes(
        app,
        layout=None
        if project_root is None
        else StorageLayout(project_root=Path(project_root).resolve()),
    )
    return app


def _include_routes(app: FastAPI, *, layout: StorageLayout | None) -> None:
    def selected_profile(profile_id):
        try:
            return research_profile(profile_id)
        except KeyError as exc:
            raise ApiError(
                "unknown_profile_id", status_code=404, field="profile"
            ) from exc

    def evidence_query(query, **kwargs):
        if layout is None:
            raise ApiError("project_root_not_configured", status_code=503)
        try:
            return query(layout=layout, **kwargs)
        except KeyError as exc:
            raise ApiError(
                "unknown_target_id", status_code=404, field="target_id"
            ) from exc
        except FileNotFoundError as exc:
            raise ApiError("published_evidence_missing", status_code=404) from exc
        except (ValueError, TypeError) as exc:
            raise ApiError("published_evidence_invalid", status_code=422) from exc
        except OSError as exc:
            raise ApiError("published_evidence_unavailable", status_code=422) from exc
        except RuntimeError as exc:
            if str(exc) == "runtime_data_scale_lock_busy":
                raise ApiError("runtime_data_scale_lock_busy", status_code=503) from exc
            raise

    @app.get(
        "/api/0.1/catalog/factors",
        operation_id="list_catalog_factors",
    )
    def catalog_factors(profile: str = A_SHARE_PROFILE_ID) -> FactorCatalogResponse:
        return queries.list_factors(selected_profile(profile))

    @app.get(
        "/api/0.1/catalog/labels",
        operation_id="list_catalog_labels",
    )
    def catalog_labels(profile: str = A_SHARE_PROFILE_ID) -> LabelCatalogResponse:
        return queries.list_labels(selected_profile(profile))

    @app.get(
        "/api/0.1/research-datasets/{dataset_id}/requirements",
        operation_id="describe_research_dataset_requirements",
    )
    def describe_research_dataset_requirements(
        dataset_id: str,
        profile: str = A_SHARE_PROFILE_ID,
    ) -> ResearchDatasetRequirementsResponse:
        try:
            return queries.describe_research_dataset_requirements(
                dataset_id,
                selected_profile(profile),
            )
        except KeyError as exc:
            raise ApiError(
                error_code="unknown_dataset_id",
                status_code=404,
                field="dataset_id",
            ) from exc

    @app.get("/api/0.1/status", operation_id="read_run_status")
    def run_status(
        profile: str = A_SHARE_PROFILE_ID, run_id: str | None = None
    ) -> dict[str, object]:
        return evidence_query(
            research.status, profile=selected_profile(profile), run_id=run_id
        )

    @app.get("/api/0.1/inspect/{target_id}", operation_id="inspect_published_target")
    def inspect_target(
        target_id: str, profile: str = A_SHARE_PROFILE_ID, run_id: str | None = None
    ) -> dict[str, object]:
        return evidence_query(
            research.inspect,
            profile=selected_profile(profile),
            dataset_id=target_id,
            run_id=run_id,
        )

    @app.get(
        "/api/0.1/runs/{run_id}/reports/{report_identity}",
        operation_id="read_published_report",
    )
    def report(
        run_id: str, report_identity: str, profile: str = A_SHARE_PROFILE_ID
    ) -> dict[str, object]:
        return evidence_query(
            queries.read_report,
            profile=selected_profile(profile),
            run_id=run_id,
            report_identity=report_identity,
        )

    @app.get(
        "/api/0.1/runs/{run_id}/portfolios/{target_id}",
        operation_id="read_portfolio_curve",
    )
    def portfolio(
        run_id: str,
        target_id: str,
        profile: str = A_SHARE_PROFILE_ID,
        stride: int = Query(1, ge=1),
    ) -> dict[str, object]:
        return evidence_query(
            queries.portfolio_curve,
            profile=selected_profile(profile),
            run_id=run_id,
            target_id=target_id,
            stride=stride,
        )


app = create_app()
