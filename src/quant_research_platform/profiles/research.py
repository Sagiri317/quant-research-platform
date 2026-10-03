"""Thin composition roots for the active A-share and Crypto profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from quant_research_platform.markets.implementations.crypto import ids as crypto_ids
from quant_research_platform.research.portfolios.diagnostics import (
    build_portfolio_statistics_tables,
)
from quant_research_platform.research.protocols import a_share, crypto_hft
from quant_research_platform.research.statistics_production import (
    build_a_share_statistics_tables,
)
from quant_research_platform.sources import (
    market_funding_identity,
    market_source_identity,
    open_market_source,
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
        models=ModelCatalog(),
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
                f"{scenario_id}_{group}_persisted_predictions": ProducerBinding(
                    ProducerKind.PREDICTION,
                    build_prediction_result_from_persisted_model,
                )
                for scenario_id in a_share.SCENARIO_IDS
                for group in ("global", "quarterly")
            },
            **{
                f"{scenario_id}_{group}_statistics": ProducerBinding(
                    ProducerKind.STATISTIC,
                    build_a_share_statistics_tables,
                )
                for scenario_id in a_share.SCENARIO_IDS
                for group in ("global", "quarterly")
            },
            **{
                f"{scenario_id}_report_artifact": ProducerBinding(
                    ProducerKind.REPORT,
                    produce_a_share_report,
                )
                for scenario_id in a_share.SCENARIO_IDS
            },
        },
    )


@cache
def _crypto_profile() -> ResearchProfile:
    from quant_research_platform.artifacts.crypto_hft import (
        produce_crypto_report,
    )
    from quant_research_platform.artifacts.outputs import read_prediction_output_bundle
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_PANEL_DATA_ID,
    )
    from quant_research_platform.markets.implementations.crypto.portfolio import (
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
    from quant_research_platform.research.predictions.runner import (
        build_crypto_predictions,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        build_crypto_evaluation_catalog,
        build_hft_research_dataset_catalog,
    )
    from quant_research_platform.sources.binance.ids import (
        RESEARCH_SYMBOLS,
    )
    from quant_research_platform.sources.binance.replay import (
        ReconstructionStats,
        iter_agg_trades,
        iter_reconstructed_depth,
    )

    processed = crypto_catalog.hft_processed_catalog()
    research_datasets = build_hft_research_dataset_catalog()
    scenario = crypto_ids.HFT_RESEARCH_DATASET_ID
    return ResearchProfile(
        profile_id=CRYPTO_PROFILE_ID,
        market_id=MarketId("crypto"),
        frequency=DataFrequency.TICK,
        protocol=crypto_hft.protocol,
        factors=crypto_hft.factor_catalog(),
        labels=crypto_hft.label_catalog(),
        interim=InterimDataCatalog(),
        processed=processed,
        research_datasets=research_datasets,
        evaluations=build_crypto_evaluation_catalog(),
        models=ModelCatalog(),
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
            f"{crypto_ids.HFT_RESEARCH_DATASET_ID}_models": ProducerBinding(
                ProducerKind.MODEL, build_crypto_fold_models
            ),
            f"{crypto_ids.HFT_RESEARCH_DATASET_ID}_predictions": ProducerBinding(
                ProducerKind.PREDICTION, build_crypto_predictions
            ),
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.PROCESSED,
                    partial(
                        crypto_producer.produce_hft_processed_data,
                        open_run=open_market_source,
                        reconstruct_depth=iter_reconstructed_depth,
                        read_trades=iter_agg_trades,
                        make_stats=ReconstructionStats,
                    ),
                    source_identity=market_source_identity
                    if spec.data_id == HFT_PANEL_DATA_ID
                    else None,
                )
                for spec in processed.list()
                if spec.data_id
                not in {
                    crypto_ids.HFT_DERIVATIVE_DATA_ID,
                    crypto_ids.HFT_SPOT_PANEL_DATA_ID,
                    crypto_ids.HFT_SPOT_CONTEXT_DATA_ID,
                    crypto_ids.HFT_SOL_PANEL_DATA_ID,
                    crypto_ids.HFT_INFORMATION_PANEL_DATA_ID,
                    crypto_ids.HFT_INFORMATION_STATE_DATA_ID,
                    crypto_ids.HFT_CROSS_ASSET_DATA_ID,
                    *crypto_ids.HFT_EXTERNAL_DATA_IDS.values(),
                }
            },
            **{
                spec.data_id: ProducerBinding(
                    ProducerKind.DATASET, assemble_research_dataset
                )
                for spec in research_datasets.list()
            },
        },
    )


def bind_research_configuration(
    profile: ResearchProfile,
    configuration,
    *,
    tardis_root=None,
    access_window="development",
):
    """Resolve one explicit selection into the existing catalogs and protocol."""
    import json

    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.research.protocols import bind_research_selection

    library = research_profile(profile.profile_id)
    bound, dataset, model_catalog = bind_research_selection(
        profile, library, configuration
    )
    config = bound.configuration()
    scenario = config["scenario_id"]
    evaluation = config["evaluation"]
    columns = dataset.factor_columns
    selected_model_ids = {model.model_id for model in model_catalog.list()}
    candidates = bound.portfolio_candidates()
    uses_sol_rows = str(profile.market_id) == "crypto" and any(
        "sol_usdt" in item[key]
        for item in config["models"]
        for key in ("training_asset_ids", "evaluation_asset_ids")
    )
    availability = {
        item.output_column: item.available_at_column for item in profile.labels.list()
    }
    processed = library.processed
    bindings = dict(profile.producer_bindings)
    if str(profile.market_id) == "a_share":
        processed = profile.processed
        from quant_research_platform.markets.implementations.a_share.calendar import (
            unified_trading_sessions,
        )
        from quant_research_platform.markets.implementations.a_share.execution import (
            fee_schedule_from_frame,
            market_name_for_type,
            prepare_trade_cost,
            require_fee_coverage,
        )
        from quant_research_platform.research.portfolios.production import (
            build_portfolio_ledger,
        )

        portfolio_producer = partial(
            build_portfolio_ledger,
            fee_schedule_from_frame=fee_schedule_from_frame,
            require_fee_coverage=require_fee_coverage,
            prepare_trade_cost=prepare_trade_cost,
            market_name_for_type=market_name_for_type,
            unified_trading_sessions=unified_trading_sessions,
        )
        bindings.update(
            {
                f"{scenario}_{candidate.candidate_id}_ledger": ProducerBinding(
                    ProducerKind.PORTFOLIO,
                    portfolio_producer,
                )
                for candidate in candidates
            }
        )
    if str(profile.market_id) == "crypto":
        from quant_research_platform.artifacts.crypto_hft import (
            read_crypto_episode_artifact,
            summarize_crypto_execution,
            summarize_crypto_ledger_windows,
        )
        from quant_research_platform.artifacts.outputs import (
            read_prediction_output_bundle,
        )
        from quant_research_platform.artifacts.portfolio_ledger import (
            read_portfolio_ledger_artifact,
            summarize_portfolio_ledger,
        )
        from quant_research_platform.artifacts.statistics import (
            STATISTICS_TABLE_COLUMNS,
            read_statistics_artifact,
        )
        from quant_research_platform.contracts.time import TimeWindow
        from quant_research_platform.markets.implementations.crypto.portfolio import (
            produce_maker_calibration,
            produce_maker_ledger,
            produce_taker_ledger,
        )
        from quant_research_platform.research.models.persisted import (
            read_crypto_model_artifact,
        )
        from quant_research_platform.research.portfolios.diagnostics import (
            build_crypto_portfolio_statistics_tables,
        )
        from quant_research_platform.sources.binance.funding import (
            read_funding_observations,
        )
        from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
        from quant_research_platform.sources.binance.replay import (
            iter_agg_trades,
            iter_reconstructed_depth,
        )

        archive_paths = (
            None
            if config["source_archives"] is None
            else tuple(config["source_archives"])
        )
        source_window = TimeWindow(
            min(fold.train.start for fold in bound.research_plan.folds),
            TimeWindow(**config["windows"][access_window]).end,
        )
        source_opener = partial(
            open_market_source,
            archive_paths=archive_paths,
            time_window=source_window,
        )
        calibration_window = TimeWindow(
            source_window.start, TimeWindow(**config["windows"]["development"]).end
        )
        calibration_opener = partial(
            open_market_source,
            archive_paths=archive_paths,
            time_window=calibration_window,
        )
        source_root = tardis_root if archive_paths is not None else None
        panel_binding = bindings[crypto_ids.HFT_PANEL_DATA_ID]
        bindings[crypto_ids.HFT_PANEL_DATA_ID] = replace(
            panel_binding,
            producer=partial(panel_binding.producer, open_run=source_opener),
            source_identity=partial(
                market_source_identity,
                archive_paths=archive_paths,
                time_window=source_window,
            ),
            raw_root=source_root,
        )
        for candidate in candidates:
            symbol = {"btc_usdt": "BTCUSDT", "eth_usdt": "ETHUSDT"}[candidate.asset_id]
            common = {
                "symbol": symbol,
                "open_run": source_opener,
                "reconstruct_depth": partial(
                    iter_reconstructed_depth,
                    market_type=FUTURES_MARKET_TYPE,
                ),
                "read_funding": read_funding_observations,
                "read_prediction_bundle": read_prediction_output_bundle,
            }
            producer = (
                partial(
                    produce_maker_ledger,
                    **common,
                    market_type=FUTURES_MARKET_TYPE,
                    read_trades=partial(
                        iter_agg_trades,
                        market_type=FUTURES_MARKET_TYPE,
                    ),
                    read_statistics_artifact=read_statistics_artifact,
                )
                if isinstance(candidate, crypto_hft.CryptoMakerCandidate)
                else partial(produce_taker_ledger, **common)
            )
            if (
                isinstance(candidate, crypto_hft.CryptoMakerCandidate)
                and candidate.calibrated
            ):
                bindings[candidate.calibration_target_id] = ProducerBinding(
                    ProducerKind.STATISTIC,
                    partial(
                        produce_maker_calibration,
                        symbol=symbol,
                        asset_id=candidate.asset_id,
                        model_ids=candidate.model_ids,
                        market_type=FUTURES_MARKET_TYPE,
                        open_run=calibration_opener,
                        reconstruct_depth=partial(
                            iter_reconstructed_depth,
                            market_type=FUTURES_MARKET_TYPE,
                        ),
                        read_trades=partial(
                            iter_agg_trades,
                            market_type=FUTURES_MARKET_TYPE,
                        ),
                        read_model_artifact=read_crypto_model_artifact,
                    ),
                    source_identity=partial(
                        market_source_identity,
                        archive_paths=archive_paths,
                        time_window=calibration_window,
                    ),
                    raw_root=source_root,
                )
            for target in bound.execution_targets():
                if target.candidate != candidate:
                    continue
                bindings[target.target_id] = ProducerBinding(
                    ProducerKind.PORTFOLIO,
                    partial(producer, execution_mode=target.execution_mode),
                    source_identity=partial(
                        market_funding_identity,
                        symbol=symbol,
                        archive_paths=archive_paths,
                        time_window=TimeWindow(**config["windows"]["holdout"]),
                    ),
                    raw_root=source_root,
                )
                bindings[target.statistics_target_id] = ProducerBinding(
                    ProducerKind.STATISTIC,
                    partial(
                        build_crypto_portfolio_statistics_tables,
                        read_ledger=read_portfolio_ledger_artifact,
                        read_episodes=read_crypto_episode_artifact,
                        summarize_ledger=summarize_portfolio_ledger,
                        summarize_windows=summarize_crypto_ledger_windows,
                        summarize_execution=summarize_crypto_execution,
                        table_columns=STATISTICS_TABLE_COLUMNS,
                    ),
                )
    auxiliary = config.get("auxiliary_sources", {})
    if type(auxiliary) is not dict or set(auxiliary) - {
        "derivatives_state",
        "spot",
        "sol",
        "cross_asset",
        *crypto_ids.HFT_EXTERNAL_DATA_IDS,
    }:
        raise ValueError("research_auxiliary_sources_invalid")
    if auxiliary and str(profile.market_id) != "crypto":
        raise ValueError("research_auxiliary_market_invalid")
    cross_assets = (
        sorted(
            {
                column.split("_", 1)[0] + "_usdt"
                for column in columns
                if column in crypto_hft.HFT_CROSS_ASSET_FACTOR_COLUMNS
            }
        )
        if str(profile.market_id) == "crypto"
        else []
    )
    if auxiliary or uses_sol_rows or cross_assets:
        from quant_research_platform.markets.implementations.crypto.ids import (
            HFT_FACTOR_DATA_ID,
            HFT_LABEL_DATA_ID,
            HFT_PANEL_DATA_ID,
        )
        from quant_research_platform.markets.implementations.crypto.processed import (
            producer as crypto_producer,
        )
        from quant_research_platform.sources.binance.ids import SPOT_MARKET_TYPE
        from quant_research_platform.sources.binance.replay import (
            ReconstructionStats,
            iter_agg_trades,
            iter_reconstructed_depth,
        )
        from quant_research_platform.sources.tardis.replay import (
            native_source_identity,
            open_tardis_archive,
        )

        for name, source in auxiliary.items():
            if name in crypto_ids.HFT_EXTERNAL_DATA_IDS:
                from quant_research_platform.sources.tardis.normalized import (
                    SCHEMA_ID,
                    iter_records,
                    source_identity,
                )

                instrument = crypto_producer.EXTERNAL_INSTRUMENTS[name]
                if (
                    type(source) is not dict
                    or set(source) != {"archives", "max_age_ms"}
                    or type(source["max_age_ms"]) is not int
                    or source["max_age_ms"] <= 0
                    or type(source["archives"]) is not dict
                    or set(source["archives"]) != set(instrument["symbols"].values())
                ):
                    raise ValueError("research_external_source_invalid")
                for kinds in source["archives"].values():
                    if (
                        type(kinds) is not dict
                        or set(kinds) != {"quotes", "trades", "derivative_ticker"}
                        or any(
                            type(paths) is not list
                            or not paths
                            or any(
                                type(path) is not str
                                or not path
                                or any(
                                    character
                                    not in (
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                                    )
                                    for character in path
                                )
                                for path in paths
                            )
                            or len(set(paths)) != len(paths)
                            for paths in kinds.values()
                        )
                    ):
                        raise ValueError("research_external_archive_invalid")
                if tardis_root is None:
                    raise ValueError("research_auxiliary_source_root_required")
                data_id = crypto_ids.HFT_EXTERNAL_DATA_IDS[name]
                identity = partial(
                    source_identity,
                    exchange=instrument["exchange"],
                    archives=source["archives"],
                    time_window=source_window,
                )
                bindings[data_id] = ProducerBinding(
                    ProducerKind.PROCESSED,
                    partial(
                        crypto_producer.produce_normalized_context,
                        read_records=iter_records,
                        source_identity=identity,
                        source_schema_id=SCHEMA_ID,
                        archives=source["archives"],
                    ),
                    source_identity=identity,
                    raw_root=tardis_root / instrument["exchange"],
                )
                parameters = {
                    "venue": name,
                    "symbols": instrument["symbols"],
                    "base_units_per_amount": instrument["base_units_per_amount"],
                    "max_age_ms": source["max_age_ms"],
                }
                processed = ProcessedDataCatalog(
                    replace(spec, parameters_json=json.dumps(parameters))
                    if spec.data_id == data_id
                    else spec
                    for spec in processed.list()
                )
                continue
            if name == "cross_asset":
                if (
                    type(source) is not dict
                    or set(source) != {"max_age_ms"}
                    or type(source["max_age_ms"]) is not int
                    or source["max_age_ms"] <= 0
                ):
                    raise ValueError("research_state_age_invalid")
                continue
            expected_fields = (
                {"archives"} if name == "sol" else {"archives", "max_age_ms"}
            )
            if type(source) is not dict or set(source) != expected_fields:
                raise ValueError("research_auxiliary_source_invalid")
            exchange = "binance" if name == "spot" else "binance-futures"
            relatives = source["archives"]
            if (
                type(relatives) is not list
                or not relatives
                or any(
                    type(relative) is not str
                    or len(relative.split("/")) != 2
                    or relative.split("/")[0] != exchange
                    or not relative.split("/")[1]
                    or any(
                        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                        for character in relative.split("/")[1]
                    )
                    for relative in relatives
                )
                or len(set(relatives)) != len(relatives)
            ):
                raise ValueError("research_auxiliary_archive_invalid")
            if tardis_root is None:
                raise ValueError("research_auxiliary_source_root_required")
            archive_paths = tuple(relatives)
            archive_opener = partial(
                open_tardis_archive,
                archive_paths=archive_paths,
                time_window=source_window,
            )
            if name == "sol":
                bindings[crypto_ids.HFT_SOL_PANEL_DATA_ID] = ProducerBinding(
                    ProducerKind.PROCESSED,
                    partial(
                        crypto_producer.produce_hft_processed_data,
                        open_run=archive_opener,
                        reconstruct_depth=iter_reconstructed_depth,
                        read_trades=iter_agg_trades,
                        make_stats=ReconstructionStats,
                        symbols=("SOLUSDT",),
                        panel_id=crypto_ids.HFT_SOL_PANEL_DATA_ID,
                    ),
                    source_identity=partial(
                        native_source_identity,
                        archive_paths=archive_paths,
                        time_window=source_window,
                        exchange="binance-futures",
                        symbols=("solusdt",),
                        channels=("depth", "depthSnapshot", "aggTrade", "markPrice"),
                    ),
                    raw_root=tardis_root,
                )
                continue
            ages = source["max_age_ms"]
            if name == "derivatives_state":
                if (
                    type(ages) is not dict
                    or set(ages) != set(crypto_hft.HFT_DERIVATIVE_CHANNELS)
                    or any(
                        type(value) is not int or value <= 0 for value in ages.values()
                    )
                ):
                    raise ValueError("research_state_age_invalid")
                data_id = crypto_ids.HFT_DERIVATIVE_DATA_ID
                bindings[data_id] = ProducerBinding(
                    ProducerKind.PROCESSED,
                    partial(
                        crypto_producer.produce_derivative_processed_data,
                        open_archive=archive_opener,
                    ),
                    source_identity=partial(
                        native_source_identity,
                        archive_paths=archive_paths,
                        time_window=source_window,
                        exchange=exchange,
                        symbols=("btcusdt", "ethusdt"),
                        channels=crypto_hft.HFT_DERIVATIVE_CHANNELS,
                    ),
                    raw_root=tardis_root,
                )
            else:
                if type(ages) is not int or ages <= 0:
                    raise ValueError("research_state_age_invalid")
                data_id = crypto_ids.HFT_SPOT_CONTEXT_DATA_ID
                bindings[crypto_ids.HFT_SPOT_PANEL_DATA_ID] = ProducerBinding(
                    ProducerKind.PROCESSED,
                    partial(
                        crypto_producer.produce_hft_processed_data,
                        open_run=archive_opener,
                        reconstruct_depth=iter_reconstructed_depth,
                        read_trades=iter_agg_trades,
                        make_stats=ReconstructionStats,
                        market_type=SPOT_MARKET_TYPE,
                        panel_id=crypto_ids.HFT_SPOT_PANEL_DATA_ID,
                    ),
                    source_identity=partial(
                        native_source_identity,
                        archive_paths=archive_paths,
                        time_window=source_window,
                        exchange=exchange,
                        symbols=("btcusdt", "ethusdt"),
                        channels=("depth", "depthSnapshot", "aggTrade"),
                    ),
                    raw_root=tardis_root,
                )
                bindings[data_id] = ProducerBinding(
                    ProducerKind.PROCESSED,
                    crypto_producer.produce_spot_perp_context,
                )
            processed = ProcessedDataCatalog(
                replace(spec, parameters_json=json.dumps({"max_age_ms": ages}))
                if spec.data_id == data_id
                else spec
                for spec in processed.list()
            )
        read_assets = (
            ["btc_usdt", "eth_usdt", "sol_usdt"]
            if uses_sol_rows
            else ["btc_usdt", "eth_usdt"]
        )
        changes = {}
        if uses_sol_rows:
            for data_id in (HFT_FACTOR_DATA_ID, HFT_LABEL_DATA_ID):
                changes[data_id] = {
                    "processed_input_data_ids": (
                        crypto_ids.HFT_INFORMATION_PANEL_DATA_ID,
                    ),
                    "parameters_json": json.dumps(
                        {
                            **json.loads(processed.get(data_id).parameters_json),
                            "asset_ids": read_assets,
                        }
                    ),
                }
        if uses_sol_rows or ("sol_usdt" in cross_assets and len(cross_assets) > 1):
            bindings[crypto_ids.HFT_INFORMATION_PANEL_DATA_ID] = ProducerBinding(
                ProducerKind.PROCESSED,
                crypto_producer.produce_information_panel,
            )
        if cross_assets and "cross_asset" in auxiliary:
            information_panel = (
                crypto_ids.HFT_SOL_PANEL_DATA_ID
                if cross_assets == ["sol_usdt"]
                else crypto_ids.HFT_INFORMATION_PANEL_DATA_ID
                if "sol_usdt" in cross_assets
                else HFT_PANEL_DATA_ID
            )
            changes[crypto_ids.HFT_INFORMATION_STATE_DATA_ID] = {
                "processed_input_data_ids": (information_panel,),
                "parameters_json": json.dumps({"asset_ids": cross_assets}),
            }
            changes[crypto_ids.HFT_CROSS_ASSET_DATA_ID] = {
                "processed_input_data_ids": (
                    dataset.base_input_data_id,
                    crypto_ids.HFT_INFORMATION_STATE_DATA_ID,
                ),
                "parameters_json": json.dumps(
                    {
                        "information_assets": cross_assets,
                        "target_assets": read_assets,
                        "max_age_ms": auxiliary["cross_asset"]["max_age_ms"],
                    }
                ),
            }
            bindings[crypto_ids.HFT_INFORMATION_STATE_DATA_ID] = ProducerBinding(
                ProducerKind.PROCESSED,
                bindings[HFT_FACTOR_DATA_ID].producer,
            )
            bindings[crypto_ids.HFT_CROSS_ASSET_DATA_ID] = ProducerBinding(
                ProducerKind.PROCESSED,
                crypto_producer.produce_cross_asset_context,
            )
        if changes:
            processed = ProcessedDataCatalog(
                replace(spec, **changes[spec.data_id])
                if spec.data_id in changes
                else spec
                for spec in processed.list()
            )
    eval_specs = []
    evaluation_specs = (
        crypto_hft.build_crypto_evaluation_catalog(configuration=config).list()
        if str(profile.market_id) == "crypto"
        else library.evaluations.list()
    )
    for spec in evaluation_specs:
        if spec.research_dataset_id != scenario:
            continue
        if all(signal.source_kind == "factor" for signal in spec.signals):
            if str(profile.market_id) == "a_share":
                spec = replace(
                    spec,
                    signals=tuple(
                        SignalSpec(
                            signal_id=column, source_kind="factor", source_column=column
                        )
                        for column in evaluation["factors"]
                    ),
                    label_columns=tuple(evaluation["labels"]),
                    label_availability_columns={
                        label: availability[label] for label in evaluation["labels"]
                    },
                )
            eval_specs.append(spec)
        elif config["stage"] == "research" and all(
            signal.source_kind == "prediction"
            and signal.signal_id in selected_model_ids
            for signal in spec.signals
        ):
            eval_specs.append(spec)
    if str(profile.market_id) == "crypto":
        report_id = f"{scenario}_report_artifact"
        bindings[report_id] = replace(
            bindings[report_id],
            producer=partial(
                bindings[report_id].producer,
                calibration_candidates=tuple(
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, crypto_hft.CryptoMakerCandidate)
                    and candidate.calibrated
                ),
                comparison_ids=frozenset(
                    spec.evaluation_id
                    for spec in eval_specs
                    if len(spec.signals) > 1
                    and spec.signals[0].source_kind == "prediction"
                ),
            ),
        )
    return replace(
        profile,
        processed=processed,
        protocol=lambda scenario_id, access_window="development": replace(
            bound, scenario_id=scenario_id, access_window=access_window
        ),
        research_datasets=ResearchDatasetCatalog((dataset,)),
        evaluations=EvaluationCatalog(eval_specs),
        models=model_catalog,
        producer_bindings={
            **bindings,
            **{
                spec.evaluation_id: profile.producer_bindings[f"{scenario}_evaluation"]
                for spec in eval_specs
            },
        },
    )
