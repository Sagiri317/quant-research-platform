"""Platform default catalog composition root."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketRegistry
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_evaluation_catalog,
    a_share_factor_catalog,
    a_share_field_catalog,
    a_share_label_catalog,
    a_share_market_definition,
    a_share_model_catalog,
    a_share_prediction_catalog,
    a_share_processed_catalog,
    a_share_research_dataset_catalog,
)
from quant_research_platform.reports.catalog import report_catalog
from quant_research_platform.sources.csmar.catalog import csmar_source_definition
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_interim_catalog,
)


@dataclass(frozen=True)
class PlatformCatalogs:
    fields: FieldCatalog
    factors: FactorCatalog
    labels: LabelCatalog
    processed: ProcessedDataCatalog
    research_datasets: ResearchDatasetCatalog
    evaluations: EvaluationCatalog
    models: ModelCatalog
    predictions: PredictionCatalog
    sources: SourceRegistry
    markets: MarketRegistry
    reports: ReportCatalog
    source_interim_catalogs: Mapping[str, InterimDataCatalog]


def default_field_catalog() -> FieldCatalog:
    return a_share_field_catalog()


def default_factor_catalog() -> FactorCatalog:
    return a_share_factor_catalog()


def default_label_catalog() -> LabelCatalog:
    return a_share_label_catalog()


def default_processed_catalog() -> ProcessedDataCatalog:
    return a_share_processed_catalog()


def default_research_dataset_catalog() -> ResearchDatasetCatalog:
    return a_share_research_dataset_catalog()


def default_evaluation_catalog() -> EvaluationCatalog:
    return a_share_evaluation_catalog()


def default_model_catalog() -> ModelCatalog:
    return a_share_model_catalog()


def default_prediction_catalog() -> PredictionCatalog:
    return a_share_prediction_catalog()


def default_source_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(csmar_source_definition())
    return registry


def default_market_registry() -> MarketRegistry:
    registry = MarketRegistry()
    registry.register(a_share_market_definition())
    return registry


def default_report_catalog() -> ReportCatalog:
    return report_catalog()


def default_source_interim_catalogs() -> dict[str, InterimDataCatalog]:
    catalog = csmar_interim_catalog()
    source_ids = {str(spec.source_id) for spec in catalog.list()}
    if len(source_ids) != 1:
        raise ValueError("unexpected_source_interim_catalog_sources")
    return {next(iter(source_ids)): catalog}


def default_catalogs() -> PlatformCatalogs:
    return PlatformCatalogs(
        fields=default_field_catalog(),
        factors=default_factor_catalog(),
        labels=default_label_catalog(),
        processed=default_processed_catalog(),
        research_datasets=default_research_dataset_catalog(),
        evaluations=default_evaluation_catalog(),
        models=default_model_catalog(),
        predictions=default_prediction_catalog(),
        sources=default_source_registry(),
        markets=default_market_registry(),
        reports=default_report_catalog(),
        source_interim_catalogs=default_source_interim_catalogs(),
    )


__all__ = [
    "PlatformCatalogs",
    "default_catalogs",
    "default_evaluation_catalog",
    "default_factor_catalog",
    "default_field_catalog",
    "default_label_catalog",
    "default_market_registry",
    "default_model_catalog",
    "default_prediction_catalog",
    "default_processed_catalog",
    "default_report_catalog",
    "default_research_dataset_catalog",
    "default_source_interim_catalogs",
    "default_source_registry",
]
