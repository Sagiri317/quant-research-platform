"""Read-only HTTP queries for catalogs and explicitly selected local artifacts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import FastAPI, Query

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
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
    sanitize_shared_text,
)
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    research_profile,
)


def _public_evidence(value):
    """Keep private input paths out of the HTTP representation."""
    if isinstance(value, dict):
        return {
            sanitize_shared_text(key): _public_evidence(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_public_evidence(item) for item in value]
    if isinstance(value, (str, Path)):
        return sanitize_shared_text(value)
    return value


def create_app(*, project_root: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Quant Research Platform",
        version="unversioned",
        description=(
            "Read-only catalogs and explicitly selected published numerical evidence."
        ),
    )
    register_api_error_handlers(app)
    root = None if project_root is None else Path(project_root).resolve()

    def selected_profile(profile_id):
        try:
            return research_profile(profile_id)
        except KeyError as exc:
            raise ApiError(
                "unknown_profile_id", status_code=404, field="profile"
            ) from exc

    def artifact_path(relative):
        if root is None:
            raise ApiError("project_root_not_configured", status_code=503)
        value = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or value.is_absolute()
            or PureWindowsPath(relative).drive
            or ".." in value.parts
            or "raw" in value.parts
            or not value.parts
        ):
            raise ApiError("artifact_path_invalid", status_code=422, field="path")
        selected = root.joinpath(*value.parts).resolve()
        if (
            not selected.is_relative_to(root)
            or "raw" in selected.relative_to(root).parts
        ):
            raise ApiError("artifact_path_invalid", status_code=422, field="path")
        return selected

    def evidence(query, relative, **kwargs):
        selected = artifact_path(relative)
        try:
            result = query(selected, **kwargs)
            assert_internal_artifact_metadata_safe(result)
            return _public_evidence(result)
        except FileNotFoundError as exc:
            raise ApiError("published_evidence_missing", status_code=404) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise ApiError("published_evidence_invalid", status_code=422) from exc
        except OSError as exc:
            raise ApiError("published_evidence_unavailable", status_code=422) from exc

    @app.get("/api/catalog/factors", operation_id="list_catalog_factors")
    def factors(profile: str = A_SHARE_PROFILE_ID) -> FactorCatalogResponse:
        return queries.list_factors(selected_profile(profile))

    @app.get("/api/catalog/labels", operation_id="list_catalog_labels")
    def labels(profile: str = A_SHARE_PROFILE_ID) -> LabelCatalogResponse:
        return queries.list_labels(selected_profile(profile))

    @app.get(
        "/api/research-datasets/{dataset_id}/requirements",
        operation_id="describe_research_dataset_requirements",
    )
    def dataset(
        dataset_id: str, profile: str = A_SHARE_PROFILE_ID
    ) -> ResearchDatasetRequirementsResponse:
        try:
            return queries.describe_research_dataset_requirements(
                dataset_id, selected_profile(profile)
            )
        except KeyError as exc:
            raise ApiError(
                "unknown_dataset_id", status_code=404, field="dataset_id"
            ) from exc

    @app.get("/api/inspect", operation_id="inspect_published_artifact")
    def inspect(path: str) -> dict[str, object]:
        return evidence(queries.inspect, path)

    @app.get("/api/portfolio-curve", operation_id="read_portfolio_curve")
    def portfolio_curve(
        path: str, stride: int = Query(1, ge=1), execution_mode: str = "continuous"
    ) -> dict[str, object]:
        return evidence(
            queries.portfolio_curve, path, stride=stride, execution_mode=execution_mode
        )

    return app


app = create_app()
