"""Research profile composition roots."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Protocol

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
)
from quant_research_platform.catalog.processed import (
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
    data_frequency_slug,
    normalize_data_frequency,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import (
    research_datasets as a_share_research_datasets,
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
    research_datasets as crypto_research_datasets,
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
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.sources.binance.interim import (
    materialize_interim_data as _materialize_binance_interim_data,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.sources.csmar.interim import (
    DEFAULT_INTERIM_DATA_IDS as CSMAR_INTERIM_DATA_IDS,
)
from quant_research_platform.sources.csmar.interim import (
    InterimImportResult,
)
from quant_research_platform.sources.csmar.interim import (
    materialize_interim_data as _materialize_csmar_interim_data,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout


class PredictionCatalogBuilder(Protocol):
    def __call__(
        self,
        *,
        train_end: str,
        test_start: str,
        test_end: str | None = None,
    ) -> PredictionCatalog: ...


class PortfolioCatalogBuilder(Protocol):
    def __call__(
        self,
        *,
        rebalance_frequency: str = "1d",
        execution_lag_steps: int = 1,
        signals: tuple[SignalSpec, ...] | None = None,
        return_source_column: str = "close",
        prediction_id: str | None = None,
    ) -> PortfolioCatalog: ...


class InterimMaterializer(Protocol):
    def __call__(
        self,
        *,
        layout: StorageLayout,
        raw_root: Path,
        start_date: str,
        end_date: str,
        resource_policy: RuntimeResourcePolicy | None = None,
    ) -> object: ...


class ProcessedMaterializer(Protocol):
    def __call__(
        self,
        *,
        layout: StorageLayout,
        source_id: SourceId,
        time_window: TimeWindow,
        data_ids: tuple[str, ...] | None = None,
        kinds: frozenset[ProcessedDataKind] | None = None,
        resource_policy: RuntimeResourcePolicy | None = None,
    ) -> dict[str, DatasetInspection]: ...


@dataclass(frozen=True)
class ResearchProfile:
    factors: FactorCatalog
    labels: LabelCatalog
    processed: ProcessedDataCatalog
    research_datasets: ResearchDatasetCatalog
    evaluations: EvaluationCatalog
    models: ModelCatalog
    source_id: SourceId
    build_prediction_catalog: PredictionCatalogBuilder
    build_portfolio_catalog: PortfolioCatalogBuilder
    materialize_interim_data: InterimMaterializer
    materialize_processed_data: ProcessedMaterializer
    schema_inventory_important_columns: tuple[str, ...]


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


@cache
def _a_share_csmar_research_profile() -> ResearchProfile:
    return ResearchProfile(
        factors=factor_definitions.factor_catalog(),
        labels=label_definitions.label_catalog(),
        processed=processed_catalog_module.processed_catalog(),
        research_datasets=(a_share_research_datasets.build_research_dataset_catalog()),
        evaluations=a_share_research_scenarios.build_evaluation_catalog(),
        models=a_share_research_scenarios.build_model_catalog(),
        source_id=csmar_ids.SOURCE_ID,
        build_prediction_catalog=a_share_research_scenarios.build_prediction_catalog,
        build_portfolio_catalog=a_share_research_scenarios.build_portfolio_catalog,
        materialize_interim_data=_materialize_a_share_csmar_interim_data,
        materialize_processed_data=processed_materializer_module.materialize_processed_data,
        schema_inventory_important_columns=tuple(
            processed_diagnostics_module.schema_inventory_important_columns()
        ),
    )


@cache
def _crypto_binance_research_profile() -> ResearchProfile:
    return ResearchProfile(
        factors=crypto_factor_definitions.factor_catalog(),
        labels=crypto_label_definitions.label_catalog(),
        processed=crypto_processed_catalog_module.processed_catalog(),
        research_datasets=crypto_research_datasets.build_research_dataset_catalog(),
        evaluations=crypto_research_scenarios.build_evaluation_catalog(),
        models=ModelCatalog(),
        source_id=binance_ids.SOURCE_ID,
        build_prediction_catalog=_empty_prediction_catalog,
        build_portfolio_catalog=_empty_portfolio_catalog,
        materialize_interim_data=_materialize_crypto_binance_interim_data,
        materialize_processed_data=(
            crypto_processed_materializer_module.materialize_processed_data
        ),
        schema_inventory_important_columns=(
            "close",
            "eligible",
            "spread",
            "relative_spread",
            "forward_return_1m",
        ),
    )


def _materialize_crypto_binance_interim_data(
    layout: StorageLayout,
    raw_root: Path,
    start_date: str,
    end_date: str,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> object:
    with runtime_resource_policy_scope(resource_policy) as policy:
        dependency = crypto_materialization_dependencies.panel_dependency_window(
            TimeWindow(start_date, end_date)
        )
        return _materialize_binance_interim_data(
            layout=layout,
            raw_root=raw_root,
            start_date=_time_bound_to_iso(dependency.start),
            end_date=_time_bound_to_iso(dependency.end),
            resource_policy=policy,
        )


def _materialize_a_share_csmar_interim_data(
    layout: StorageLayout,
    raw_root: Path,
    start_date: str,
    end_date: str,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> InterimImportResult:
    with runtime_resource_policy_scope(resource_policy) as policy:
        with ExitStack() as cleanup_stack:
            calendar = _materialize_csmar_interim_data(
                layout=layout,
                raw_root=raw_root,
                data_ids=("trading_calendar",),
                start_date=None,
                end_date=None,
                resource_policy=policy,
            )
            targets = tuple(processed_catalog_module.processed_catalog().list())
            lookback_sessions, lookahead_sessions = (
                a_share_materialization_dependencies.required_session_offsets(targets)
            )
            owned_sessions = (
                a_share_materialization_planning.available_trading_sessions(
                    layout,
                    SourceId(csmar_ids.SOURCE_ID),
                    time_window=TimeWindow(start_date, end_date),
                    lookback_sessions=lookback_sessions,
                    lookahead_sessions=lookahead_sessions,
                    resource_policy=policy,
                )
            )
            cleanup_stack.callback(
                owned_sessions.release,
                owner="a_share.profile.sessions",
            )
            sessions = owned_sessions.value
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
                resource_policy=policy,
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
