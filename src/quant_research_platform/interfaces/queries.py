"""Read-only catalog responses and native artifact metadata inspection."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.interfaces.models import (
    FactorCatalogResponse,
    LabelCatalogResponse,
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


def inspect(path: Path) -> dict[str, object]:
    """Read only the explicitly selected artifact's native metadata."""
    root = Path(path)
    if root.is_file():
        return read_qrp_json_file(root)[1]
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in (
        "manifest.json",
        "_manifest/manifest.json",
        "model.json",
        "policy.json",
        "spec.json",
        "metadata.json",
        "summary.json",
    ):
        candidate = root / name
        if candidate.is_file():
            return read_qrp_json_file(candidate)[1]
    raise ValueError("published_metadata_missing")
