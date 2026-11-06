"""Read-only catalog responses and native artifact metadata inspection."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.interfaces.models import (
    FactorCatalogResponse,
    LabelCatalogResponse,
    ResearchDatasetRequirementsResponse,
)
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.storage.manifests import read_qrp_json_file


def list_factors(profile: ResearchProfile) -> FactorCatalogResponse:
    return FactorCatalogResponse(
        items=[item.to_manifest() for item in profile.factors.list()],
    )


def list_labels(profile: ResearchProfile) -> LabelCatalogResponse:
    return LabelCatalogResponse(
        items=[item.to_manifest() for item in profile.labels.list()],
    )


def describe_research_dataset_requirements(
    dataset_id: str,
    profile: ResearchProfile,
) -> ResearchDatasetRequirementsResponse:
    spec = profile.research_datasets.get(dataset_id)
    return ResearchDatasetRequirementsResponse(
        status="available",
        mode="requirements_query_only",
        dataset_id=spec.data_id,
        market_id=str(spec.market_id),
        input_data_ids=list(spec.input_data_ids),
        factor_count=len(spec.factor_columns),
        feature_count=len(spec.feature_columns),
        label_count=len(spec.label_columns),
    )


def inspect(path: Path) -> dict[str, object]:
    """Read only the explicitly selected artifact's native metadata."""
    root = Path(path)
    if root.is_file():
        if root.name not in {
            "manifest.json",
            "spec.json",
            "metadata.json",
            "model.json",
        }:
            raise ValueError("inspection_requires_native_json_metadata")
        return read_qrp_json_file(root)[1]
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in (
        "manifest.json",
        "_manifest/manifest.json",
        "model.json",
        "spec.json",
        "metadata.json",
    ):
        candidate = root / name
        if candidate.is_file():
            return read_qrp_json_file(candidate)[1]
    raise ValueError("published_metadata_missing")


def inspect_native(path: Path) -> dict[str, object]:
    """Recognize native metadata shapes, without authenticating their provenance.

    HTTP must not expose arbitrary JSON merely because it has a familiar filename.
    This does not load data members or establish that a trusted writer produced it.
    """
    metadata = inspect(path)
    schema_shapes = {
        "qrp-dataset-spec": {
            "data_id": str,
            "required_columns": list,
            "column_roles": dict,
            "primary_key": list,
        },
        "qrp-dataset-manifest": {
            "row_count": int,
            "columns": list,
            "partitions": list,
        },
        "qrp-output-manifest": {
            "output_definition_id": str,
            "metadata": dict,
            "files": list,
        },
        "qrp-portfolio-ledger": {
            "metadata": dict,
            "capital": dict,
            "members": list,
        },
        "qrp-statistics": {
            "table_group": str,
            "metadata": dict,
            "members": list,
        },
    }
    kind_shapes = {
        "bivariate_exponential_hawkes": {
            "config": dict,
            "parameters": dict,
            "fit_cutoff": str,
        },
        "cost_aware_allocator": {"config": dict, "periods": int, "device": str},
        "MODEL_EXPLANATION": {
            "model": str,
            "model_id": str,
            "methods": list,
            "input_columns": list,
            "members": list,
        },
    }
    shape = schema_shapes.get(metadata.get("schema_id")) or kind_shapes.get(
        metadata.get("kind")
    )
    if shape is None and set(metadata) == {"model_metadata", "facts", "trial_id"}:
        shape = {"model_metadata": dict, "facts": dict, "trial_id": str}
    if shape is None or any(
        type(metadata.get(key)) is not kind for key, kind in shape.items()
    ):
        raise ValueError("inspection_requires_native_artifact_metadata")
    return metadata
