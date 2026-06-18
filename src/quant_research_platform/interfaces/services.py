"""Thin service layer shared by API and CLI adapters."""

from __future__ import annotations

from quant_research_platform.catalog.profiles import ResearchProfile
from quant_research_platform.runtime.privacy import sanitize_public_object

API_VERSION = "0.1"


def list_factors(profile: ResearchProfile) -> dict[str, object]:
    return _response(
        items=[item.to_manifest() for item in profile.factors.list()]
    )


def list_labels(profile: ResearchProfile) -> dict[str, object]:
    return _response(
        items=[item.to_manifest() for item in profile.labels.list()]
    )


def describe_research_dataset_requirements(
    dataset_id: str,
    profile: ResearchProfile,
) -> dict[str, object]:
    spec = profile.research_datasets.get(dataset_id)
    return _response(
        status="available",
        mode="requirements_query_only",
        dataset_id=spec.data_id,
        market_id=str(spec.market_id),
        input_data_ids=list(spec.input_data_ids),
        factor_count=len(spec.factor_columns),
        feature_count=len(spec.feature_columns),
        label_count=len(spec.label_columns),
    )


def _response(**payload: object) -> dict[str, object]:
    return sanitize_public_object({"api_version": API_VERSION, **payload})


__all__ = [
    "API_VERSION",
    "describe_research_dataset_requirements",
    "list_factors",
    "list_labels",
]
