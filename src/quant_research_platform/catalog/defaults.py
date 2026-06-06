"""Platform default catalog composition root."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quant_research_platform.catalog.data_products import DataProductRegistry
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketDefinition, MarketRegistry
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.provider_bindings import (
    ProviderBinding,
    ProviderBindingRegistry,
)
from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.catalog.universes import UniverseCatalog
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.data_products import DataProductDefinition
from quant_research_platform.contracts.universes import UniverseDefinition
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_evaluation_catalog,
    a_share_factor_catalog,
    a_share_field_catalog,
    a_share_label_catalog,
    a_share_model_catalog,
    a_share_prediction_catalog,
    a_share_processed_catalog,
    a_share_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_MARKET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.reports.catalog import report_catalog
from quant_research_platform.sources.csmar.catalog import (
    csmar_interim_catalog,
    csmar_source_definition,
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
    data_products: DataProductRegistry
    provider_bindings: ProviderBindingRegistry
    sources: SourceRegistry
    markets: MarketRegistry
    reports: ReportCatalog
    source_interim_catalogs: Mapping[str, InterimDataCatalog]
    universes: UniverseCatalog


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


def default_data_product_registry() -> DataProductRegistry:
    registry = DataProductRegistry()
    _register_default_data_products(registry)
    return registry


def default_provider_binding_registry() -> ProviderBindingRegistry:
    registry = ProviderBindingRegistry()
    _register_default_provider_bindings(registry)
    return registry


def default_source_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(csmar_source_definition())
    return registry


def default_market_registry() -> MarketRegistry:
    registry = MarketRegistry()
    _register_default_markets(registry)
    return registry


def default_report_catalog() -> ReportCatalog:
    return report_catalog()


def default_source_interim_catalogs() -> dict[str, InterimDataCatalog]:
    return {"csmar": csmar_interim_catalog()}


def default_universe_catalog() -> UniverseCatalog:
    catalog = UniverseCatalog()
    catalog.register(
        UniverseDefinition(
            universe_id=A_SHARE_DEFAULT_UNIVERSE_ID,
            market_id=A_SHARE_MARKET_ID,
            membership_column="in_universe",
        )
    )
    return catalog


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
        data_products=default_data_product_registry(),
        provider_bindings=default_provider_binding_registry(),
        sources=default_source_registry(),
        markets=default_market_registry(),
        reports=default_report_catalog(),
        source_interim_catalogs=default_source_interim_catalogs(),
        universes=default_universe_catalog(),
    )


def _register_default_data_products(registry: DataProductRegistry) -> None:
    registry.register(
        DataProductDefinition(
            data_product_id="a_share_daily_bar",
            market_id=A_SHARE_MARKET_ID,
            grain=DataGrain.ASSET_TIMESTAMP,
            interim_spec_ids=("a_share_daily_bar_interim",),
        )
    )


def _register_default_provider_bindings(
    registry: ProviderBindingRegistry,
) -> None:
    registry.register(
        ProviderBinding(
            data_product_id="a_share_daily_bar",
            provider_id="csmar",
            source_id="csmar",
            implementation_package="quant_research_platform.sources.csmar",
            required_source_dataset_ids=(
                "daily_market",
                "stock_master",
                "market_cap",
                "adjustment",
                "index_info",
                "benchmark_index_market",
                "trading_calendar",
            ),
        )
    )


def _register_default_markets(registry: MarketRegistry) -> None:
    registry.register(
        MarketDefinition(
            market_id=A_SHARE_MARKET_ID,
            display_name="A-share market implementation",
            processed_spec_ids=(
                DAILY_PANEL_PROCESSED_DATA_ID,
                DAILY_CONTEXT_PROCESSED_DATA_ID,
                A_SHARE_DEFAULT_UNIVERSE_ID,
                A_SHARE_AS_OF_MASK_ID,
                A_SHARE_EXECUTION_MASK_ID,
                A_SHARE_EVALUATION_MASK_ID,
                PRICE_VOLUME_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
            ),
        )
    )


__all__ = [
    "PlatformCatalogs",
    "default_catalogs",
    "default_data_product_registry",
    "default_evaluation_catalog",
    "default_factor_catalog",
    "default_field_catalog",
    "default_label_catalog",
    "default_market_registry",
    "default_model_catalog",
    "default_prediction_catalog",
    "default_processed_catalog",
    "default_provider_binding_registry",
    "default_report_catalog",
    "default_research_dataset_catalog",
    "default_source_interim_catalogs",
    "default_source_registry",
    "default_universe_catalog",
]
