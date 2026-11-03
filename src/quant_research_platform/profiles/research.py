"""Read-only market catalogs, without operation bindings or scientific overrides."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.ids import MarketId

A_SHARE_PROFILE_ID = "a_share_csmar_choice"
A_SHARE_HFT_PROFILE_ID = "a_share_level2"


@dataclass(frozen=True)
class ResearchProfile:
    profile_id: str
    market_id: MarketId
    factors: FactorCatalog
    labels: LabelCatalog
    research_datasets: ResearchDatasetCatalog


@cache
def research_profile(profile_id: str = A_SHARE_PROFILE_ID) -> ResearchProfile:
    """Select a static catalog. Operations receive their own explicit inputs."""
    from quant_research_platform.research.protocols import a_share, a_share_hft

    if profile_id == A_SHARE_PROFILE_ID:
        return ResearchProfile(
            profile_id,
            MarketId("a_share"),
            a_share.factor_catalog(),
            a_share.label_catalog(),
            a_share.research_dataset_catalog(),
        )
    if profile_id == A_SHARE_HFT_PROFILE_ID:
        return ResearchProfile(
            profile_id,
            MarketId("a_share"),
            a_share_hft.factor_catalog(),
            a_share_hft.label_catalog(),
            a_share_hft.research_dataset_catalog(),
        )
    raise KeyError("unknown_research_profile")
