"""Research profile composition roots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketRegistry
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
)
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import (
    research_profiles as a_share_research_profiles,
)
from quant_research_platform.markets.implementations.a_share.definition import (
    a_share_market_definition,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as a_share_factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.fields import (
    a_share_field_catalog,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as a_share_label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as a_share_processed_catalog_module,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as a_share_processed_materializer,
)
from quant_research_platform.markets.implementations.a_share.research_scenarios import (
    build_a_share_evaluation_catalog,
    build_a_share_model_catalog,
)
from quant_research_platform.reports.catalog import report_catalog
from quant_research_platform.sources.csmar.catalog import csmar_source_definition
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_interim_catalog,
)
from quant_research_platform.sources.csmar.interim import (
    materialize_interim_data as _materialize_csmar_interim_data,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class ResearchProfile:
    fields: FieldCatalog
    factors: FactorCatalog
    labels: LabelCatalog
    processed: ProcessedDataCatalog
    research_datasets: ResearchDatasetCatalog
    evaluations: EvaluationCatalog
    models: ModelCatalog
    predictions: PredictionCatalog
    portfolios: PortfolioCatalog
    sources: SourceRegistry
    markets: MarketRegistry
    reports: ReportCatalog
    source_interim_catalogs: Mapping[str, InterimDataCatalog]
    source_id: SourceId
    materialize_interim_data: Callable[[StorageLayout, Path, str, str], object]
    materialize_processed_data: Callable[
        [StorageLayout, SourceId, TimeWindow, frozenset[ProcessedDataKind]],
        Mapping[str, DatasetInspection],
    ]


def research_profile() -> ResearchProfile:
    """Return the active research profile."""

    return _a_share_csmar_research_profile()


def _a_share_csmar_research_profile() -> ResearchProfile:
    return ResearchProfile(
        fields=a_share_field_catalog(),
        factors=a_share_factor_definitions.a_share_factor_catalog(),
        labels=a_share_label_definitions.a_share_label_catalog(),
        processed=a_share_processed_catalog_module.a_share_processed_catalog(),
        research_datasets=(
            a_share_research_profiles.build_a_share_research_dataset_catalog()
        ),
        evaluations=build_a_share_evaluation_catalog(),
        models=build_a_share_model_catalog(),
        predictions=PredictionCatalog(),
        portfolios=PortfolioCatalog(),
        sources=_a_share_csmar_source_registry(),
        markets=_a_share_market_registry(),
        reports=report_catalog(),
        source_interim_catalogs=_a_share_csmar_source_interim_catalogs(),
        source_id=_a_share_csmar_source_id(),
        materialize_interim_data=_materialize_a_share_csmar_interim_data,
        materialize_processed_data=_materialize_a_share_processed_data,
    )


def _a_share_csmar_source_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(csmar_source_definition())
    return registry


def _a_share_market_registry() -> MarketRegistry:
    registry = MarketRegistry()
    registry.register(a_share_market_definition())
    return registry


def _a_share_csmar_source_interim_catalogs() -> Mapping[str, InterimDataCatalog]:
    catalog = csmar_interim_catalog()
    source_ids = {str(spec.source_id) for spec in catalog.list()}
    if len(source_ids) != 1:
        raise ValueError("unexpected_source_interim_catalog_sources")
    return {next(iter(source_ids)): catalog}


def _a_share_csmar_source_id() -> SourceId:
    return SourceId(next(iter(_a_share_csmar_source_interim_catalogs())))


def _materialize_a_share_csmar_interim_data(
    layout: StorageLayout,
    raw_root: Path,
    start_date: str,
    end_date: str,
) -> object:
    return _materialize_csmar_interim_data(
        layout=layout,
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
    )


def _materialize_a_share_processed_data(
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    kinds: frozenset[ProcessedDataKind],
) -> Mapping[str, DatasetInspection]:
    return a_share_processed_materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        kinds=kinds,
    )


__all__ = ["ResearchProfile", "research_profile"]
