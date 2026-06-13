"""Local catalog profile composition roots."""

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
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
)
from quant_research_platform.catalog.predictions import (
    default_prediction_column as _prediction_column_name,
)
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
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
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_MARKET_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
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
from quant_research_platform.reports.catalog import report_catalog
from quant_research_platform.sources.csmar.catalog import csmar_source_definition
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_interim_catalog,
)
from quant_research_platform.sources.csmar.interim import (
    materialize_interim_data as _materialize_csmar_interim_data,
)


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
    sources: SourceRegistry
    markets: MarketRegistry
    reports: ReportCatalog
    source_interim_catalogs: Mapping[str, InterimDataCatalog]


def research_profile() -> ResearchProfile:
    """Return the active local research profile."""

    return _a_share_csmar_research_profile()


def _a_share_csmar_research_profile() -> ResearchProfile:
    return ResearchProfile(
        fields=a_share_field_catalog(),
        factors=a_share_factor_definitions.a_share_factor_catalog(),
        labels=a_share_label_definitions.a_share_label_catalog(),
        processed=a_share_processed_catalog_module.a_share_processed_catalog(),
        research_datasets=a_share_research_profiles.a_share_research_dataset_catalog(),
        evaluations=_a_share_evaluation_catalog(),
        models=_a_share_model_catalog(),
        predictions=_a_share_prediction_catalog(),
        sources=_a_share_csmar_source_registry(),
        markets=_a_share_market_registry(),
        reports=report_catalog(),
        source_interim_catalogs=_a_share_csmar_source_interim_catalogs(),
    )


def _a_share_evaluation_catalog() -> EvaluationCatalog:
    catalog = EvaluationCatalog()
    catalog.register(
        EvaluationSpec(
            run_id="a_share_mom_20d_forward_return_20d_2014",
            market_id=A_SHARE_MARKET_ID,
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
        )
    )
    return catalog


def _a_share_model_catalog() -> ModelCatalog:
    catalog = ModelCatalog()
    catalog.register(
        ModelSpec(
            model_id="a_share_linear_mom_20d_forward_return_20d",
            model_family="linear",
            feature_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    return catalog


def _a_share_prediction_catalog() -> PredictionCatalog:
    catalog = PredictionCatalog()
    catalog.register(
        PredictionSpec(
            prediction_id="a_share_linear_mom_20d_prediction_2014",
            model_id="a_share_linear_mom_20d_forward_return_20d",
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            prediction_column=_prediction_column_name(),
            primary_key=("market_id", "asset_id", "timestamp"),
            train_end="2014-06-30",
            test_start="2014-07-01",
        )
    )
    return catalog


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


def _a_share_csmar_source_id() -> str:
    return next(iter(_a_share_csmar_source_interim_catalogs()))


def _materialize_a_share_csmar_interim_data(**kwargs: object) -> object:
    return _materialize_csmar_interim_data(**kwargs)


def _materialize_a_share_processed_data(**kwargs: object) -> object:
    source_id = kwargs.get("source_id")
    if isinstance(source_id, str):
        kwargs = {**kwargs, "source_id": SourceId(source_id)}
    return a_share_processed_materializer.materialize_processed_data(**kwargs)


__all__ = ["ResearchProfile", "research_profile"]
