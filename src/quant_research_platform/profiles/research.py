"""Research profile composition roots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketRegistry
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.outputs import OutputCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
)
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
    data_frequency_slug,
    normalize_data_frequency,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import (
    definition as a_share_definition,
)
from quant_research_platform.markets.implementations.a_share import (
    fields as a_share_fields,
)
from quant_research_platform.markets.implementations.a_share import (
    research_profiles as a_share_research_profiles,
)
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios as a_share_research_scenarios,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog_module,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    diagnostics as processed_diagnostics_module,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as processed_materializer_module,
)
from quant_research_platform.markets.implementations.crypto import (
    definition as crypto_definition,
)
from quant_research_platform.markets.implementations.crypto import (
    fields as crypto_fields,
)
from quant_research_platform.markets.implementations.crypto import (
    research_profiles as crypto_research_profiles,
)
from quant_research_platform.markets.implementations.crypto import (
    research_scenarios as crypto_research_scenarios,
)
from quant_research_platform.markets.implementations.crypto.factors import (
    definitions as crypto_factor_definitions,
)
from quant_research_platform.markets.implementations.crypto.labels import (
    definitions as crypto_label_definitions,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    catalog as crypto_processed_catalog_module,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materializer as crypto_processed_materializer_module,
)
from quant_research_platform.sources.binance.catalog import (
    source_definition as binance_source_definition,
)
from quant_research_platform.sources.binance.interim import (
    interim_catalog as binance_interim_catalog,
)
from quant_research_platform.sources.binance.interim import (
    materialize_interim_data as _materialize_binance_interim_data,
)
from quant_research_platform.sources.csmar.catalog import source_definition
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
    profile_id: str
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
    outputs: OutputCatalog
    source_interim_catalogs: Mapping[str, InterimDataCatalog]
    source_id: SourceId
    build_prediction_catalog: Callable[..., PredictionCatalog]
    build_portfolio_catalog: Callable[..., PortfolioCatalog]
    materialize_interim_data: Callable[[StorageLayout, Path, str, str], object]
    materialize_processed_data: Callable[
        [StorageLayout, SourceId, TimeWindow, frozenset[ProcessedDataKind]],
        Mapping[str, DatasetInspection],
    ]
    schema_inventory_important_columns: Callable[[], tuple[str, ...]]


def research_profile_id(
    market_id: MarketId | str,
    source_id: SourceId | str,
    frequency: DataFrequency | str,
) -> str:
    normalized_frequency = normalize_data_frequency(frequency)
    return (
        f"{MarketId(market_id)}_{SourceId(source_id)}_"
        f"{data_frequency_slug(normalized_frequency)}"
    )


def research_profile(profile_id: str | None = None) -> ResearchProfile:
    """Return the active research profile."""

    active_profile_id = research_profile_id(
        MarketId("a_share"),
        SourceId("csmar"),
        DataFrequency.DAILY,
    )
    builders = {
        active_profile_id: _a_share_csmar_research_profile,
        research_profile_id(
            MarketId("crypto"),
            SourceId("binance"),
            DataFrequency.MINUTE,
        ): _crypto_binance_research_profile,
    }
    selected = profile_id or active_profile_id
    try:
        return builders[selected]()
    except KeyError as exc:
        raise KeyError("unknown_research_profile") from exc


def _a_share_csmar_research_profile() -> ResearchProfile:
    return ResearchProfile(
        profile_id=research_profile_id(
            MarketId("a_share"),
            SourceId("csmar"),
            DataFrequency.DAILY,
        ),
        fields=a_share_fields.field_catalog(),
        factors=factor_definitions.factor_catalog(),
        labels=label_definitions.label_catalog(),
        processed=processed_catalog_module.processed_catalog(),
        research_datasets=(a_share_research_profiles.build_research_dataset_catalog()),
        evaluations=a_share_research_scenarios.build_evaluation_catalog(),
        models=a_share_research_scenarios.build_model_catalog(),
        predictions=PredictionCatalog(),
        portfolios=PortfolioCatalog(),
        sources=_a_share_csmar_source_registry(),
        markets=_a_share_market_registry(),
        outputs=output_catalog(),
        source_interim_catalogs=_a_share_csmar_source_interim_catalogs(),
        source_id=_active_source_id(),
        build_prediction_catalog=a_share_research_scenarios.build_prediction_catalog,
        build_portfolio_catalog=a_share_research_scenarios.build_portfolio_catalog,
        materialize_interim_data=_materialize_a_share_csmar_interim_data,
        materialize_processed_data=_materialize_processed_data,
        schema_inventory_important_columns=(
            processed_diagnostics_module.schema_inventory_important_columns
        ),
    )


def _crypto_binance_research_profile() -> ResearchProfile:
    return ResearchProfile(
        profile_id=research_profile_id(
            MarketId("crypto"),
            SourceId("binance"),
            DataFrequency.MINUTE,
        ),
        fields=crypto_fields.field_catalog(),
        factors=crypto_factor_definitions.factor_catalog(),
        labels=crypto_label_definitions.label_catalog(),
        processed=crypto_processed_catalog_module.processed_catalog(),
        research_datasets=crypto_research_profiles.build_research_dataset_catalog(),
        evaluations=crypto_research_scenarios.build_evaluation_catalog(),
        models=ModelCatalog(),
        predictions=PredictionCatalog(),
        portfolios=PortfolioCatalog(),
        sources=_crypto_binance_source_registry(),
        markets=_crypto_market_registry(),
        outputs=output_catalog(),
        source_interim_catalogs=_crypto_binance_source_interim_catalogs(),
        source_id=SourceId("binance"),
        build_prediction_catalog=_empty_prediction_catalog,
        build_portfolio_catalog=_empty_portfolio_catalog,
        materialize_interim_data=_materialize_crypto_binance_interim_data,
        materialize_processed_data=_materialize_crypto_processed_data,
        schema_inventory_important_columns=_crypto_schema_inventory_important_columns,
    )


def _a_share_csmar_source_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(source_definition())
    return registry


def _crypto_binance_source_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(binance_source_definition())
    return registry


def _a_share_market_registry() -> MarketRegistry:
    registry = MarketRegistry()
    registry.register(a_share_definition.market_definition())
    return registry


def _crypto_market_registry() -> MarketRegistry:
    registry = MarketRegistry()
    registry.register(crypto_definition.market_definition())
    return registry


def _a_share_csmar_source_interim_catalogs() -> Mapping[str, InterimDataCatalog]:
    catalog = csmar_interim_catalog()
    source_ids = {str(spec.source_id) for spec in catalog.list()}
    if len(source_ids) != 1:
        raise ValueError("unexpected_source_interim_catalog_sources")
    return {next(iter(source_ids)): catalog}


def _crypto_binance_source_interim_catalogs() -> Mapping[str, InterimDataCatalog]:
    catalog = binance_interim_catalog()
    source_ids = {str(spec.source_id) for spec in catalog.list()}
    if len(source_ids) != 1:
        raise ValueError("unexpected_source_interim_catalog_sources")
    return {next(iter(source_ids)): catalog}


def _active_source_id() -> SourceId:
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


def _materialize_crypto_binance_interim_data(
    layout: StorageLayout,
    raw_root: Path,
    start_date: str,
    end_date: str,
) -> object:
    return _materialize_binance_interim_data(
        layout=layout,
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
    )


def _materialize_processed_data(
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    kinds: frozenset[ProcessedDataKind],
) -> Mapping[str, DatasetInspection]:
    return processed_materializer_module.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        kinds=kinds,
    )


def _materialize_crypto_processed_data(
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    kinds: frozenset[ProcessedDataKind],
) -> Mapping[str, DatasetInspection]:
    return crypto_processed_materializer_module.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        kinds=kinds,
    )


def _empty_prediction_catalog(**_: object) -> PredictionCatalog:
    return PredictionCatalog()


def _empty_portfolio_catalog(**_: object) -> PortfolioCatalog:
    return PortfolioCatalog()


def _crypto_schema_inventory_important_columns() -> tuple[str, ...]:
    return (
        "close",
        "eligible",
        "spread",
        "relative_spread",
        "forward_return_1m",
    )


__all__ = ["ResearchProfile", "research_profile", "research_profile_id"]
