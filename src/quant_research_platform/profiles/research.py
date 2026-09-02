"""Thin composition roots for the active A-share and Crypto profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

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
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.research.protocols import a_share, crypto_hft
from quant_research_platform.sources.binance.interim import (
    materialize_interim_data as binance_interim_producer,
)
from quant_research_platform.sources.choice.interim import (
    interim_catalog as choice_interim_catalog,
)
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_interim_catalog,
)
from quant_research_platform.sources.csmar.interim import (
    materialize_interim_data as csmar_interim_producer,
)
from quant_research_platform.sources.csmar.interim import (
    source_content_identity as csmar_source_identity,
)


@dataclass(frozen=True)
class ResearchProfile:
    profile_id: str
    market_id: MarketId
    frequency: DataFrequency
    protocol: object
    factors: FactorCatalog
    labels: LabelCatalog
    interim: InterimDataCatalog
    processed: ProcessedDataCatalog
    research_datasets: ResearchDatasetCatalog
    evaluations: EvaluationCatalog
    models: ModelCatalog
    build_prediction_catalog: object
    build_portfolio_catalog: object
    producer_bindings: Mapping[str, ProducerBinding]
    schema_inventory_important_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.profile_id) is not str or not self.profile_id:
            raise ValueError("profile_id_required")
        if not callable(self.protocol):
            raise ValueError("profile_protocol_binding_invalid")
        if not isinstance(self.producer_bindings, Mapping):
            raise ValueError("profile_producer_bindings_invalid")
        if any(
            type(key) is not str or not key or not isinstance(value, ProducerBinding)
            for key, value in self.producer_bindings.items()
        ):
            raise ValueError("profile_producer_bindings_invalid")
        object.__setattr__(
            self,
            "producer_bindings",
            MappingProxyType(dict(self.producer_bindings)),
        )


A_SHARE_PROFILE_ID = "a_share_csmar_choice"
CRYPTO_PROFILE_ID = "crypto_binance_hft"


def research_profile(profile_id: str | None = None) -> ResearchProfile:
    selected = profile_id or A_SHARE_PROFILE_ID
    if selected == A_SHARE_PROFILE_ID:
        return _a_share_profile()
    if selected == CRYPTO_PROFILE_ID:
        return _crypto_profile()
    raise KeyError("unknown_research_profile")


@cache
def _a_share_profile() -> ResearchProfile:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as a_share_catalog,
    )
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer as a_share_producer,
    )
    from quant_research_platform.research.datasets.assembler import (
        assemble_research_dataset,
    )
    from quant_research_platform.research.evaluations.runner import (
        build_evaluation_result_from_research_dataset,
    )
    from quant_research_platform.research.models.production import (
        build_model_selection_result,
    )
    from quant_research_platform.research.predictions.runner import (
        build_prediction_result_from_persisted_model,
    )

    interim = InterimDataCatalog(
        (*csmar_interim_catalog().list(), *choice_interim_catalog().list())
    )
    processed = a_share_catalog.processed_catalog()
    research_datasets = a_share.research_dataset_catalog()
    return ResearchProfile(
        profile_id=A_SHARE_PROFILE_ID,
        market_id=MarketId("a_share"),
        frequency=DataFrequency.DAILY,
        protocol=a_share.protocol,
        factors=a_share.factor_catalog(),
        labels=a_share.label_catalog(),
        interim=interim,
        processed=processed,
        research_datasets=research_datasets,
        evaluations=a_share.evaluation_catalog(),
        models=a_share.all_model_catalog(),
        build_prediction_catalog=a_share.prediction_catalog,
        build_portfolio_catalog=a_share.portfolio_catalog,
        producer_bindings={
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.INTERIM,
                    csmar_interim_producer,
                    source_identity=csmar_source_identity,
                )
                for spec in interim.list()
                if str(spec.source_id) == "csmar"
            },
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.PROCESSED,
                    a_share_producer.produce_a_share_processed_frame,
                )
                for spec in processed.list()
            },
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.DATASET,
                    assemble_research_dataset,
                )
                for spec in research_datasets.list()
            },
            **{
                f"{scenario_id}_evaluation": ProducerBinding(
                    ProducerKind.EVALUATION,
                    build_evaluation_result_from_research_dataset,
                )
                for scenario_id in a_share.SCENARIO_IDS
            },
            **{
                f"{scenario_id}_model_selection_refit": ProducerBinding(
                    ProducerKind.MODEL,
                    build_model_selection_result,
                )
                for scenario_id in a_share.SCENARIO_IDS
            },
            **{
                f"{scenario_id}_persisted_predictions": ProducerBinding(
                    ProducerKind.PREDICTION,
                    build_prediction_result_from_persisted_model,
                )
                for scenario_id in a_share.SCENARIO_IDS
            },
        },
        schema_inventory_important_columns=(
            "eligible",
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

    interim = binance_interim_catalog()
    processed = crypto_catalog.hft_processed_catalog()
    research_datasets = build_hft_research_dataset_catalog()

    return ResearchProfile(
        profile_id=CRYPTO_PROFILE_ID,
        market_id=MarketId("crypto"),
        frequency=DataFrequency.TICK,
        protocol=crypto_hft.protocol,
        factors=FactorCatalog(),
        labels=LabelCatalog(),
        interim=interim,
        processed=processed,
        research_datasets=research_datasets,
        evaluations=build_evaluation_catalog(),
        models=ModelCatalog(),
        build_prediction_catalog=lambda **_: PredictionCatalog(),
        build_portfolio_catalog=lambda **_: PortfolioCatalog(),
        producer_bindings={
            spec.data_id: ProducerBinding(
                ProducerKind.INTERIM, binance_interim_producer
            )
            for spec in interim.list()
        },
        schema_inventory_important_columns=("close", "eligible"),
    )


def binance_interim_catalog() -> InterimDataCatalog:
    from quant_research_platform.sources.binance.interim import interim_catalog

    return interim_catalog()
