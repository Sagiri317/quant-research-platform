"""Research profile composition roots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

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
    materialization_dependencies as a_share_materialization_dependencies,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_planning as a_share_materialization_planning,
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
    materialization_dependencies as crypto_materialization_dependencies,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materializer as crypto_processed_materializer_module,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.sources.binance.catalog import (
    source_definition as binance_source_definition,
)
from quant_research_platform.sources.binance.interim import (
    interim_catalog as binance_interim_catalog,
)
from quant_research_platform.sources.binance.interim import (
    materialize_interim_data as _materialize_binance_interim_data,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.sources.csmar.catalog import source_definition
from quant_research_platform.sources.csmar.interim import (
    DEFAULT_INTERIM_DATA_IDS as CSMAR_INTERIM_DATA_IDS,
)
from quant_research_platform.sources.csmar.interim import (
    InterimImportResult,
)
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
    materialize_interim_data: Callable[
        [StorageLayout, Path, str, str, RuntimeResourcePolicy],
        object,
    ]
    materialize_processed_data: Callable[
        [
            StorageLayout,
            SourceId,
            TimeWindow,
            frozenset[ProcessedDataKind],
            RuntimeResourcePolicy,
        ],
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
        csmar_ids.SOURCE_ID,
        DataFrequency.DAILY,
    )
    builders = {
        active_profile_id: _a_share_csmar_research_profile,
        research_profile_id(
            MarketId("crypto"),
            binance_ids.SOURCE_ID,
            DataFrequency.MINUTE,
        ): _crypto_binance_research_profile,
    }
    selected = profile_id or active_profile_id
    try:
        return builders[selected]()
    except KeyError as exc:
        raise KeyError("unknown_research_profile") from exc


def _a_share_csmar_research_profile() -> ResearchProfile:
    sources = SourceRegistry()
    sources.register(source_definition())
    markets = MarketRegistry()
    markets.register(a_share_definition.market_definition())
    return ResearchProfile(
        profile_id=research_profile_id(
            MarketId("a_share"),
            csmar_ids.SOURCE_ID,
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
        sources=sources,
        markets=markets,
        outputs=output_catalog(),
        source_interim_catalogs=_single_source_interim_catalogs(
            csmar_interim_catalog()
        ),
        source_id=csmar_ids.SOURCE_ID,
        build_prediction_catalog=a_share_research_scenarios.build_prediction_catalog,
        build_portfolio_catalog=a_share_research_scenarios.build_portfolio_catalog,
        materialize_interim_data=_materialize_a_share_csmar_interim_data,
        materialize_processed_data=processed_materializer_module.materialize_processed_data,
        schema_inventory_important_columns=(
            processed_diagnostics_module.schema_inventory_important_columns
        ),
    )


def _crypto_binance_research_profile() -> ResearchProfile:
    sources = SourceRegistry()
    sources.register(binance_source_definition())
    markets = MarketRegistry()
    markets.register(crypto_definition.market_definition())
    return ResearchProfile(
        profile_id=research_profile_id(
            MarketId("crypto"),
            binance_ids.SOURCE_ID,
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
        sources=sources,
        markets=markets,
        outputs=output_catalog(),
        source_interim_catalogs=_single_source_interim_catalogs(
            binance_interim_catalog()
        ),
        source_id=binance_ids.SOURCE_ID,
        build_prediction_catalog=_empty_prediction_catalog,
        build_portfolio_catalog=_empty_portfolio_catalog,
        materialize_interim_data=_materialize_crypto_binance_interim_data,
        materialize_processed_data=(
            crypto_processed_materializer_module.materialize_processed_data
        ),
        schema_inventory_important_columns=_crypto_schema_inventory_important_columns,
    )


def _single_source_interim_catalogs(
    catalog: InterimDataCatalog,
) -> Mapping[str, InterimDataCatalog]:
    source_ids = {str(spec.source_id) for spec in catalog.list()}
    if len(source_ids) != 1:
        raise ValueError("unexpected_source_interim_catalog_sources")
    return {next(iter(source_ids)): catalog}


def _materialize_crypto_binance_interim_data(
    layout: StorageLayout,
    raw_root: Path,
    start_date: str,
    end_date: str,
    resource_policy: RuntimeResourcePolicy,
) -> object:
    dependency = crypto_materialization_dependencies.panel_dependency_window(
        TimeWindow(start_date, end_date)
    )
    return _materialize_binance_interim_data(
        layout=layout,
        raw_root=raw_root,
        start_date=_time_bound_to_iso(dependency.start),
        end_date=_time_bound_to_iso(dependency.end),
        resource_policy=resource_policy,
    )


def _materialize_a_share_csmar_interim_data(
    layout: StorageLayout,
    raw_root: Path,
    start_date: str,
    end_date: str,
    resource_policy: RuntimeResourcePolicy,
) -> InterimImportResult:
    calendar = _materialize_csmar_interim_data(
        layout=layout,
        raw_root=raw_root,
        data_ids=("trading_calendar",),
        start_date=None,
        end_date=None,
        resource_policy=resource_policy,
    )
    targets = tuple(processed_catalog_module.processed_catalog().list())
    lookback_sessions, lookahead_sessions = (
        a_share_materialization_dependencies.required_session_offsets(targets)
    )
    sessions = a_share_materialization_planning.available_trading_sessions(
        layout,
        SourceId(csmar_ids.SOURCE_ID),
        time_window=TimeWindow(start_date, end_date),
        lookback_sessions=lookback_sessions,
        lookahead_sessions=lookahead_sessions,
        resource_policy=resource_policy,
    )
    first = int(sessions.searchsorted(pd.Timestamp(start_date)))
    stop = int(sessions.searchsorted(pd.Timestamp(end_date), side="right"))
    if first < lookback_sessions or len(sessions) - stop < lookahead_sessions:
        raise ValueError("a_share_source_dependency_window_incomplete")
    source = _materialize_csmar_interim_data(
        layout=layout,
        raw_root=raw_root,
        data_ids=tuple(
            data_id
            for data_id in CSMAR_INTERIM_DATA_IDS
            if data_id != "trading_calendar"
        ),
        start_date=sessions[0].date().isoformat(),
        end_date=sessions[-1].date().isoformat(),
        resource_policy=resource_policy,
    )
    return InterimImportResult(
        artifact_roots={**calendar.artifact_roots, **source.artifact_roots},
        skipped=(*calendar.skipped, *source.skipped),
        pruning_report_text=(
            calendar.pruning_report_text + source.pruning_report_text
        ),
    )


def _time_bound_to_iso(value: date | datetime) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return value.isoformat()


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
