"""Thin composition roots for the active A-share and Crypto profiles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import DataFrequency, ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.protocols import a_share
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.sources.binance.interim import (
    materialize_interim_data as materialize_binance,
)
from quant_research_platform.sources.choice.interim import (
    interim_catalog as choice_interim_catalog,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_interim_catalog,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout


class InterimMaterializer(Protocol):
    def __call__(
        self,
        *,
        layout: StorageLayout,
        raw_root: Path,
        data_ids: tuple[str, ...] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        resource_policy: RuntimeResourcePolicy | None = None,
    ) -> object: ...


class ProcessedMaterializer(Protocol):
    def __call__(
        self,
        *,
        layout: StorageLayout,
        time_window: TimeWindow,
        data_ids: tuple[str, ...] | None = None,
        kinds: frozenset[ProcessedDataKind] | None = None,
        resource_policy: RuntimeResourcePolicy | None = None,
        rebuild: bool = False,
    ) -> dict[str, DatasetInspection]: ...


@dataclass(frozen=True)
class ResearchProfile:
    market_id: MarketId
    frequency: DataFrequency
    factors: FactorCatalog
    labels: LabelCatalog
    interim: InterimDataCatalog
    processed: ProcessedDataCatalog
    research_datasets: ResearchDatasetCatalog
    evaluations: EvaluationCatalog
    models: ModelCatalog
    build_prediction_catalog: object
    build_portfolio_catalog: object
    materialize_interim_data: InterimMaterializer
    materialize_processed_data: ProcessedMaterializer
    schema_inventory_important_columns: tuple[str, ...]


def research_profile_id(
    market_id: MarketId | str,
    source_id: SourceId | str,
    frequency: DataFrequency | str,
) -> str:
    return (
        f"{MarketId(market_id)}_{SourceId(source_id)}_{DataFrequency(frequency).value}"
    )


def research_profile(profile_id: str | None = None) -> ResearchProfile:
    active = research_profile_id("a_share", csmar_ids.SOURCE_ID, DataFrequency.DAILY)
    crypto = research_profile_id("crypto", binance_ids.SOURCE_ID, DataFrequency.MINUTE)
    selected = profile_id or active
    if selected == active:
        return _a_share_profile()
    if selected == crypto:
        return _crypto_profile()
    raise KeyError("unknown_research_profile")


@cache
def _a_share_profile() -> ResearchProfile:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as a_share_catalog,
    )
    from quant_research_platform.workflows.interim import (
        materialize_a_share_interim,
    )
    from quant_research_platform.workflows.materialization import (
        materialize_processed_data,
    )

    interim = InterimDataCatalog(
        (*csmar_interim_catalog().list(), *choice_interim_catalog().list())
    )
    return ResearchProfile(
        market_id=MarketId("a_share"),
        frequency=DataFrequency.DAILY,
        factors=a_share.factor_catalog(),
        labels=a_share.label_catalog(),
        interim=interim,
        processed=a_share_catalog.processed_catalog(),
        research_datasets=a_share.research_dataset_catalog(),
        evaluations=a_share.evaluation_catalog(),
        models=a_share.all_model_catalog(),
        build_prediction_catalog=a_share.prediction_catalog,
        build_portfolio_catalog=a_share.portfolio_catalog,
        materialize_interim_data=materialize_a_share_interim,
        materialize_processed_data=materialize_processed_data,
        schema_inventory_important_columns=(
            "scenario_universe",
            "can_enter",
            "can_exit",
            "industry_level1_code",
            "forward_return_20d",
            "model_target",
        ),
    )


@cache
def _crypto_profile() -> ResearchProfile:
    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog as crypto_catalog,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        build_evaluation_catalog,
        build_hft_research_dataset_catalog,
    )
    from quant_research_platform.workflows.materialization import (
        materialize_processed_data,
    )

    return ResearchProfile(
        market_id=MarketId("crypto"),
        frequency=DataFrequency.MINUTE,
        factors=FactorCatalog(),
        labels=LabelCatalog(),
        interim=binance_interim_catalog(),
        processed=crypto_catalog.hft_processed_catalog(),
        research_datasets=build_hft_research_dataset_catalog(),
        evaluations=build_evaluation_catalog(),
        models=ModelCatalog(),
        build_prediction_catalog=lambda **_: PredictionCatalog(),
        build_portfolio_catalog=lambda **_: PortfolioCatalog(),
        materialize_interim_data=materialize_binance,
        materialize_processed_data=materialize_processed_data,
        schema_inventory_important_columns=("close", "eligible"),
    )


def binance_interim_catalog() -> InterimDataCatalog:
    from quant_research_platform.sources.binance.interim import interim_catalog

    return interim_catalog()
