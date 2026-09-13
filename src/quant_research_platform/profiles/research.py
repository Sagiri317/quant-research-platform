"""Thin composition roots for the active A-share and Crypto profiles."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, partial
from types import MappingProxyType

from quant_research_platform.artifacts.research import produce_a_share_report
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.research.portfolios.diagnostics import (
    build_portfolio_statistics_tables,
)
from quant_research_platform.research.protocols import a_share, crypto_hft
from quant_research_platform.research.statistics_production import (
    build_a_share_statistics_tables,
)
from quant_research_platform.sources.choice.interim import (
    interim_catalog as choice_interim_catalog,
)
from quant_research_platform.sources.choice.interim import (
    materialize_interim_data as choice_interim_producer,
)
from quant_research_platform.sources.choice.interim import (
    source_content_identity as choice_source_identity,
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
from quant_research_platform.sources.reference.interim import (
    interim_catalog as reference_interim_catalog,
)
from quant_research_platform.sources.reference.interim import (
    materialize_interim_data as reference_interim_producer,
)
from quant_research_platform.sources.reference.interim import (
    source_content_identity as reference_source_identity,
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
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )
    from quant_research_platform.markets.implementations.a_share.execution import (
        fee_schedule_from_frame,
        market_name_for_type,
        require_fee_coverage,
        trade_cost,
    )
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
    from quant_research_platform.research.portfolios.production import (
        build_portfolio_ledger,
    )
    from quant_research_platform.research.predictions.runner import (
        build_prediction_result_from_persisted_model,
    )

    portfolio_producer = partial(
        build_portfolio_ledger,
        fee_schedule_from_frame=fee_schedule_from_frame,
        require_fee_coverage=require_fee_coverage,
        trade_cost=trade_cost,
        market_name_for_type=market_name_for_type,
        unified_trading_sessions=unified_trading_sessions,
    )

    choice_interim = choice_interim_catalog()
    choice_data_ids = {spec.data_id for spec in choice_interim.list()}
    interim = InterimDataCatalog(
        (
            *(
                spec
                for spec in csmar_interim_catalog().list()
                if spec.data_id not in choice_data_ids
            ),
            *choice_interim.list(),
            *reference_interim_catalog().list(),
        )
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
        producer_bindings={
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.INTERIM,
                    choice_interim_producer,
                    source_identity=choice_source_identity,
                )
                for spec in choice_interim.list()
            },
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
                    ProducerKind.INTERIM,
                    reference_interim_producer,
                    source_identity=reference_source_identity,
                )
                for spec in interim.list()
                if str(spec.source_id) == "reference"
            },
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.PROCESSED,
                    a_share_producer.produce_a_share_processed_data,
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
                f"{scenario}_portfolio_statistics": ProducerBinding(
                    ProducerKind.STATISTIC,
                    partial(
                        build_portfolio_statistics_tables,
                        unified_trading_sessions=unified_trading_sessions,
                    ),
                )
                for scenario in a_share.SCENARIO_IDS
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
            **{
                f"{scenario_id}_statistics": ProducerBinding(
                    ProducerKind.STATISTIC,
                    build_a_share_statistics_tables,
                )
                for scenario_id in a_share.SCENARIO_IDS
            },
            **{
                f"{scenario_id}_{candidate.candidate_id}_ledger": ProducerBinding(
                    ProducerKind.PORTFOLIO,
                    portfolio_producer,
                )
                for scenario_id in a_share.SCENARIO_IDS
                for candidate in a_share.protocol(scenario_id).portfolio_candidates()
            },
            **{
                f"{scenario_id}_report_artifact": ProducerBinding(
                    ProducerKind.REPORT,
                    produce_a_share_report,
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
    from quant_research_platform.artifacts.crypto_hft import (
        produce_crypto_report,
        summarize_crypto_execution,
        summarize_crypto_ledger_windows,
    )
    from quant_research_platform.artifacts.outputs import read_prediction_output_bundle
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
    )
    from quant_research_platform.artifacts.statistics import (
        STATISTICS_TABLE_COLUMNS,
        read_statistics_artifact,
    )
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_PANEL_DATA_ID,
        canonical_crypto_asset_id,
    )
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        produce_maker_calibration,
        produce_maker_ledger,
        produce_taker_ledger,
        validate_maker_calibration,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog as crypto_catalog,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer as crypto_producer,
    )
    from quant_research_platform.research.datasets.assembler import (
        assemble_research_dataset,
    )
    from quant_research_platform.research.evaluations.runner import (
        build_crypto_evaluation_result,
    )
    from quant_research_platform.research.models.persisted import (
        read_crypto_model_artifact,
    )
    from quant_research_platform.research.models.production import (
        build_crypto_fold_models,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        build_crypto_portfolio_statistics_tables,
    )
    from quant_research_platform.research.predictions.runner import (
        build_crypto_predictions,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        build_crypto_evaluation_catalog,
        build_hft_research_dataset_catalog,
    )
    from quant_research_platform.sources.binance.funding import (
        read_funding_observations,
    )
    from quant_research_platform.sources.binance.ids import (
        FUTURES_MARKET_TYPE,
        RESEARCH_SYMBOLS,
    )
    from quant_research_platform.sources.binance.replay import (
        ReconstructionStats,
        iter_agg_trades,
        iter_reconstructed_depth,
    )

    processed = crypto_catalog.hft_processed_catalog()
    research_datasets = build_hft_research_dataset_catalog()
    scenario = crypto_hft.HFT_RESEARCH_DATASET_ID
    symbols = {canonical_crypto_asset_id(symbol): symbol for symbol in RESEARCH_SYMBOLS}
    return ResearchProfile(
        profile_id=CRYPTO_PROFILE_ID,
        market_id=MarketId("crypto"),
        frequency=DataFrequency.TICK,
        protocol=crypto_hft.protocol,
        factors=FactorCatalog(),
        labels=LabelCatalog(),
        interim=InterimDataCatalog(),
        processed=processed,
        research_datasets=research_datasets,
        evaluations=build_crypto_evaluation_catalog(),
        models=crypto_hft.build_hft_model_catalog(training_row_bound=None),
        producer_bindings={
            f"{scenario}_report_artifact": ProducerBinding(
                ProducerKind.REPORT,
                partial(
                    produce_crypto_report,
                    panel_id=HFT_PANEL_DATA_ID,
                    read_model_artifact=read_crypto_model_artifact,
                    expected_symbols=RESEARCH_SYMBOLS,
                    read_reconstruction=ReconstructionStats.from_payload,
                    validate_maker_calibration=validate_maker_calibration,
                ),
            ),
            f"{scenario}_portfolio_statistics": ProducerBinding(
                ProducerKind.STATISTIC,
                partial(
                    build_crypto_portfolio_statistics_tables,
                    read_ledger=read_portfolio_ledger_artifact,
                    summarize_ledger=summarize_portfolio_ledger,
                    summarize_windows=summarize_crypto_ledger_windows,
                    summarize_execution=summarize_crypto_execution,
                    table_columns=STATISTICS_TABLE_COLUMNS,
                ),
            ),
            f"{crypto_hft.HFT_RESEARCH_DATASET_ID}_maker_calibration": ProducerBinding(
                ProducerKind.STATISTIC,
                partial(
                    produce_maker_calibration,
                    symbol="BTCUSDT",
                    market_type=FUTURES_MARKET_TYPE,
                    open_run=_open_crypto_source,
                    reconstruct_depth=partial(
                        iter_reconstructed_depth, market_type=FUTURES_MARKET_TYPE
                    ),
                    read_trades=partial(
                        iter_agg_trades, market_type=FUTURES_MARKET_TYPE
                    ),
                    read_model_artifact=read_crypto_model_artifact,
                ),
            ),
            **{
                candidate.target_id: ProducerBinding(
                    ProducerKind.PORTFOLIO,
                    partial(
                        produce_maker_ledger,
                        symbol=symbols[candidate.asset_id],
                        market_type=FUTURES_MARKET_TYPE,
                        open_run=_open_crypto_source,
                        reconstruct_depth=partial(
                            iter_reconstructed_depth, market_type=FUTURES_MARKET_TYPE
                        ),
                        read_trades=partial(
                            iter_agg_trades, market_type=FUTURES_MARKET_TYPE
                        ),
                        read_funding=read_funding_observations,
                        read_prediction_bundle=read_prediction_output_bundle,
                        read_statistics_artifact=read_statistics_artifact,
                    ),
                    source_identity=partial(
                        _crypto_funding_identity, symbol=symbols[candidate.asset_id]
                    ),
                )
                for candidate in crypto_hft.maker_candidates()
            },
            **{
                candidate.target_id: ProducerBinding(
                    ProducerKind.PORTFOLIO,
                    partial(
                        produce_taker_ledger,
                        symbol=symbols[candidate.asset_id],
                        open_run=_open_crypto_source,
                        reconstruct_depth=partial(
                            iter_reconstructed_depth, market_type=FUTURES_MARKET_TYPE
                        ),
                        read_funding=read_funding_observations,
                        read_prediction_bundle=read_prediction_output_bundle,
                    ),
                    source_identity=partial(
                        _crypto_funding_identity, symbol=symbols[candidate.asset_id]
                    ),
                )
                for candidate in crypto_hft.taker_candidates()
            },
            **{
                spec.evaluation_id: ProducerBinding(
                    ProducerKind.EVALUATION,
                    partial(
                        build_crypto_evaluation_result,
                        read_prediction_bundle=read_prediction_output_bundle,
                    ),
                )
                for spec in build_crypto_evaluation_catalog().list()
            },
            f"{crypto_hft.HFT_RESEARCH_DATASET_ID}_models": ProducerBinding(
                ProducerKind.MODEL, build_crypto_fold_models
            ),
            f"{crypto_hft.HFT_RESEARCH_DATASET_ID}_predictions": ProducerBinding(
                ProducerKind.PREDICTION, build_crypto_predictions
            ),
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.PROCESSED,
                    partial(
                        crypto_producer.produce_hft_processed_data,
                        open_run=_open_crypto_source,
                        reconstruct_depth=iter_reconstructed_depth,
                        read_trades=iter_agg_trades,
                        make_stats=ReconstructionStats,
                    ),
                    source_identity=_crypto_source_identity
                    if spec.data_id == HFT_PANEL_DATA_ID
                    else None,
                )
                for spec in processed.list()
            },
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.DATASET, assemble_research_dataset
                )
                for spec in research_datasets.list()
            },
        },
        schema_inventory_important_columns=("close", "eligible"),
    )


@contextmanager
def _open_crypto_source(root, resource_policy=None):
    """Select one immutable native format; never fall back after validation fails."""
    from quant_research_platform.sources.binance.replay import open_recorder_run
    from quant_research_platform.sources.tardis.replay import open_tardis_day
    from quant_research_platform.storage.identity import require_unaliased_path

    root = require_unaliased_path(root, error="crypto_source_path_invalid")
    official = (root / "run.json").is_file()
    historical = (root / "manifest.json").is_file()
    if official == historical:
        if official:
            raise ValueError("crypto_source_identity_ambiguous")
        raise FileNotFoundError("crypto_source_manifest_missing")
    opener = open_recorder_run if official else open_tardis_day
    with opener(root, resource_policy) as source:
        yield source


def _crypto_source_identity(root, _target_id):
    from quant_research_platform.contracts.json_codec import canonical_json_hash

    try:
        with _open_crypto_source(root) as source:
            return canonical_json_hash(source.source_identity)
    except FileNotFoundError:
        return None


def _crypto_funding_identity(root, target_id, *, symbol):
    from quant_research_platform.sources.binance.funding import funding_source_identity

    with _open_crypto_source(root) as source:
        return funding_source_identity(
            source.funding_root(symbol), target_id, symbol=symbol
        )
