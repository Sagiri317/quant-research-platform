"""A-share low-frequency research protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date

import pandas as pd

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.targets import TargetSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import ids as market_ids
from quant_research_platform.research.factors.expectations.definitions import (
    EXPECTATIONS_FACTORS,
    expectations_factor_definitions,
)
from quant_research_platform.research.factors.fundamental.definitions import (
    FUNDAMENTAL_FACTORS,
    fundamental_factor_definitions,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_LIBRARY,
    price_volume_factor_definitions,
)
from quant_research_platform.research.features.transforms import (
    canonical_feature_transforms,
)
from quant_research_platform.research.filtering import A_SHARE_ELIGIBILITY_FACTS
from quant_research_platform.research.labels.forward_returns import (
    forward_return_available_at_column,
    forward_return_label_column,
)
from quant_research_platform.research.portfolios.accounting import AlphaCalibration
from quant_research_platform.research.portfolios.optimizer import OptimizerConfig
from quant_research_platform.research.portfolios.risk import FactorRiskConfig
from quant_research_platform.research.randomness import ROOT_SEED

PROTOCOL_ID = "a_share_low_frequency"
MARKET_MODEL_FACTORS = ("beta_252d", "residual_volatility_60d")
RESEARCH_DATASET_IDS = {
    "pv_all": "pv_all",
    "pv_expanded": "pv_expanded",
    "fusion_nf_core": "fusion_nf_core",
    "fusion_nf_expectations": "fusion_nf_expectations",
}
SCENARIO_IDS = tuple(RESEARCH_DATASET_IDS)
ACCESS_WINDOWS = ("development", "holdout")
DATA_START = date(2011, 1, 1)
DATA_END = date(2025, 12, 31)
FORMAL_DATASET_START = date(2014, 1, 1)
FORMAL_DATASET_END = date(2025, 12, 31)
DEVELOPMENT_START = date(2019, 1, 1)
DEVELOPMENT_END = date(2023, 12, 31)
HOLDOUT_START = date(2024, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
LABEL_HORIZONS = (1, 5, 10, 20, 60, 120)
PRIMARY_LABEL_COLUMN = forward_return_label_column(20)
MODEL_TARGET_COLUMN = "model_target"


def quarter_id_for_fold(fold: ExperimentFold) -> str:
    """Return the A-share persisted quarter identity for one protocol fold."""

    if not isinstance(fold, ExperimentFold):
        raise TypeError("a_share_experiment_fold_required")
    year, separator, quarter = fold.fold_id.partition("-q")
    if len(year) != 4 or separator != "-q" or quarter not in {"1", "2", "3", "4"}:
        raise ValueError("a_share_experiment_fold_id_invalid")
    return f"{year}Q{quarter}"


EXECUTION_ASSUMPTIONS = ExecutionAssumptions(
    initial_nav=1_000_000_000.0,
    slippage_bps=5.0,
    impact_bps=25.0,
    adv_participation=0.05,
    adv_sensitivities=(0.01, 0.02, 0.05, 0.10),
)
STATISTICS_BLOCK = 20
STATISTICS_REPETITIONS = 10_000
HAC_LAG = 19
RISK_STYLE_COLUMNS = (
    "log_float_market_cap",
    "beta_252d",
    "momentum_252_20",
    "residual_volatility_60d",
    "amihud_60d",
    "earnings_yield",
)


@dataclass(frozen=True)
class AShareProtocol:
    scenario_id: str
    access_window: str = "development"
    root_seed: int = ROOT_SEED
    frozen_development: dict[str, object] | None = None
    frozen_alpha: AlphaCalibration | None = None
    configuration_json: str | None = None
    risk_config: FactorRiskConfig = FactorRiskConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()

    def __post_init__(self) -> None:
        if self.scenario_id not in SCENARIO_IDS:
            raise ValueError("unknown_a_share_scenario")
        if self.access_window not in ACCESS_WINDOWS:
            raise ValueError("unknown_a_share_access_window")
        if self.root_seed != ROOT_SEED:
            raise ValueError("a_share_root_seed_must_be_frozen")

    @property
    def time_window(self) -> TimeWindow:
        return TimeWindow(
            DEVELOPMENT_START if self.access_window == "development" else HOLDOUT_START,
            DEVELOPMENT_END if self.access_window == "development" else HOLDOUT_END,
        )

    @property
    def data_window(self) -> TimeWindow:
        return TimeWindow(DATA_START, DATA_END)

    @property
    def formal_dataset_window(self) -> TimeWindow:
        return TimeWindow(FORMAL_DATASET_START, FORMAL_DATASET_END)

    def configuration(self):
        import json

        return (
            None
            if self.configuration_json is None
            else json.loads(self.configuration_json)
        )

    @property
    def protocol_id(self) -> str:
        return PROTOCOL_ID

    @property
    def factor_columns(self) -> tuple[str, ...]:
        if self.configuration_json is not None:
            return tuple(self.configuration()["evaluation"]["factors"])
        columns = (*PRICE_VOLUME_LIBRARY, *MARKET_MODEL_FACTORS)
        if self.scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            columns += FUNDAMENTAL_FACTORS
        if self.scenario_id == "fusion_nf_expectations":
            columns += EXPECTATIONS_FACTORS
        return columns

    @property
    def factor_fdr_families(self) -> dict[str, list[str]]:
        definitions = {item.output_column: item for item in factor_catalog().list()}
        families = {}
        for column in self.factor_columns:
            families.setdefault(definitions[column].family, []).append(column)
        return families

    def manifest(self) -> dict[str, object]:
        return {
            **(
                {"research_configuration": self.configuration()}
                if self.configuration_json is not None
                else {}
            ),
            **(
                {"frozen_development": self.frozen_development}
                if self.frozen_development is not None
                else {}
            ),
            "protocol_id": self.protocol_id,
            "scenario_id": self.scenario_id,
            "access_window": self.access_window,
            "data_window": {
                "start": DATA_START.isoformat(),
                "end": DATA_END.isoformat(),
            },
            "formal_dataset_window": {
                "start": FORMAL_DATASET_START.isoformat(),
                "end": FORMAL_DATASET_END.isoformat(),
            },
            "development_window": {
                "start": DEVELOPMENT_START.isoformat(),
                "end": DEVELOPMENT_END.isoformat(),
            },
            "holdout_window": {
                "start": HOLDOUT_START.isoformat(),
                "end": HOLDOUT_END.isoformat(),
            },
            "model_chronology": {"validation_months": 12},
            "root_seed": self.root_seed,
            "benchmark_total_return_indices": {}
            if self.scenario_id == "pv_expanded"
            else {"csi500": "H00905", "csi1000": "H00852"},
            "execution_assumptions": {
                "initial_nav": EXECUTION_ASSUMPTIONS.initial_nav,
                "valuation_currency": "CNY",
                "slippage_bps": EXECUTION_ASSUMPTIONS.slippage_bps,
                "impact_bps": EXECUTION_ASSUMPTIONS.impact_bps,
                "adv_participation": EXECUTION_ASSUMPTIONS.adv_participation,
                "adv_sensitivities": list(EXECUTION_ASSUMPTIONS.adv_sensitivities),
            },
            "risk": asdict(self.risk_config),
            "optimizer": asdict(self.optimizer_config),
            "statistics": {
                "hac_lag": HAC_LAG,
                "bootstrap_block": STATISTICS_BLOCK,
                "bootstrap_repetitions": STATISTICS_REPETITIONS,
                "fdr_primary": "BH_5pct",
                "fdr_robustness": "BY_5pct",
                "cscv_blocks": 10,
                "factor_fdr_families": self.factor_fdr_families,
                "fama_macbeth_core_factors": [
                    item.output_column
                    for item in factor_catalog().list()
                    if item.factor_set is FactorSet.CORE
                    and item.output_column in self.factor_columns
                ],
            },
            "portfolio_candidates": [
                candidate.candidate_id for candidate in self.portfolio_candidates()
            ],
        }

    def target_window(self, action, data_spec, window, target_id=None):
        from quant_research_platform.contracts.data import (
            InterimDataSpec,
            ProcessedDataSpec,
        )

        if action != "research run":
            return window
        if isinstance(data_spec, (InterimDataSpec, ProcessedDataSpec)):
            return self.data_window
        if isinstance(data_spec, ResearchDatasetSpec):
            return self.formal_dataset_window
        if self.access_window == "holdout":
            return window
        if target_id in {
            f"{self.scenario_id}_evaluation",
            f"{self.scenario_id}_statistics",
        }:
            return analysis_windows(self.manifest(), window)[0]
        if target_id in {
            self.model_selection_target_id,
            self.persisted_prediction_target_id,
        }:
            return TimeWindow(
                max(
                    pd.Timestamp(self.manifest()["development_window"]["start"]),
                    pd.Timestamp(window.start) - pd.DateOffset(months=12),
                ).date(),
                pd.Timestamp(window.end).date(),
            )
        return window

    def frozen_target_ids(self):
        return (self.model_selection_target_id,)

    def freeze_development(self, record, references, frozen, artifacts):
        from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
        from quant_research_platform.research.portfolios.production import (
            frozen_alpha_calibration,
        )

        persisted = artifacts[self.model_selection_target_id]
        if max(persisted.quarterly_selections) != str(
            pd.Period(self.manifest()["development_window"]["end"], freq="Q")
        ):
            raise ValueError("holdout_development_model_quarter_invalid")
        calibration = calibration_date = blocker = None
        sources = []
        for candidate in self.portfolio_candidates():
            if candidate.construction != "benchmark_enhanced":
                continue
            source = references[f"{self.scenario_id}_{candidate.candidate_id}_ledger"]
            calibration_result = []

            def consume(batches, *, calibration_result=calibration_result):
                calibration_result.append(
                    frozen_alpha_calibration(
                        batches, cutoff=pd.Timestamp(record.research_window["end"])
                    )
                )

            try:
                artifacts[source.target_id](consume_batches=consume)
                ((current, observed_at),) = calibration_result
            except PortfolioTargetBlocked as exc:
                blocker = str(exc)
                break
            if calibration is not None and (
                current != calibration or observed_at != calibration_date
            ):
                raise ValueError("holdout_alpha_sources_disagree")
            calibration, calibration_date = current, observed_at
            sources.append({"reference": source.payload(), "observed_at": observed_at})
        frozen["alpha_sources"] = sources
        return replace(
            self,
            access_window="holdout",
            frozen_development=frozen,
            frozen_alpha=calibration,
        ), blocker

    def target_specs(self, action: str) -> tuple[TargetSpec, ...]:
        if action != "research run":
            return ()
        dataset = self.scenario_id
        evaluation = f"{dataset}_evaluation"
        model_selection = self.model_selection_target_id
        predictions = self.persisted_prediction_target_id
        statistics = f"{dataset}_statistics"
        portfolio_candidates = self.portfolio_candidates()
        portfolios = tuple(
            TargetSpec(
                f"{dataset}_{candidate.candidate_id}_ledger",
                dependencies=(
                    candidate.prediction_target_id,
                    candidate.research_dataset_id,
                    candidate.execution_mask_target_id,
                    candidate.trading_calendar_target_id,
                    candidate.fee_schedule_target_id,
                    candidate.market_panel_target_id,
                    *candidate.risk_context_target_ids,
                    *(
                        (candidate.benchmark_weights_target_id,)
                        if candidate.benchmark_weights_target_id
                        else ()
                    ),
                ),
                producer_id=f"{dataset}_{candidate.candidate_id}_ledger",
            )
            for candidate in portfolio_candidates
        )
        return (
            TargetSpec(
                evaluation,
                dependencies=(dataset,),
                producer_id=evaluation,
            ),
            TargetSpec(
                model_selection,
                dependencies=() if self.access_window == "holdout" else (dataset,),
                producer_id=model_selection,
            ),
            TargetSpec(
                predictions,
                dependencies=(dataset, model_selection),
                producer_id=predictions,
            ),
            TargetSpec(
                statistics,
                dependencies=(predictions, model_selection, dataset),
                producer_id=statistics,
            ),
            *portfolios,
            TargetSpec(
                f"{dataset}_portfolio_statistics",
                dependencies=tuple(
                    dict.fromkeys(
                        (
                            *(item.target_id for item in portfolios),
                            dataset,
                            "market_panel_1d",
                            "trading_calendar",
                            *(
                                (market_ids.PRICE_VOLUME_DATA_ID,)
                                if not any(
                                    candidate.benchmark_id
                                    for candidate in portfolio_candidates
                                )
                                else ("benchmark_weights", "index_market")
                            ),
                            *(
                                target_id
                                for candidate in portfolio_candidates
                                for target_id in candidate.risk_context_target_ids
                            ),
                        )
                    )
                ),
                producer_id=f"{dataset}_portfolio_statistics",
            ),
        )

    @property
    def model_selection_target_id(self) -> str:
        return f"{self.scenario_id}_model_selection_refit"

    @property
    def persisted_prediction_target_id(self) -> str:
        return f"{self.scenario_id}_persisted_predictions"

    def model_ids_for_target(self, target_id: str) -> tuple[str, ...]:
        if target_id == self.model_selection_target_id:
            if self.configuration_json is not None:
                return tuple(
                    model["model_id"] for model in self.configuration()["models"]
                )
            model_ids = ()
        else:
            raise ValueError("unknown_a_share_model_target")
        return model_ids

    @property
    def execution_assumptions(self) -> ExecutionAssumptions:
        return EXECUTION_ASSUMPTIONS

    def portfolio_candidates(self) -> tuple[PortfolioCandidateSpec, ...]:
        if self.configuration_json is None:
            return ()
        result = []
        for item in self.configuration()["portfolios"]:
            if type(item) is not dict or set(item) != {
                "candidate_id",
                "construction",
                "prediction_selector",
                "top_quantile",
                "rebalance_schedule",
                "benchmark_id",
            }:
                raise ValueError("research_portfolio_configuration_invalid")
            enhanced = item["construction"] == "benchmark_enhanced"
            result.append(
                PortfolioCandidateSpec(
                    **item,
                    prediction_target_id=self.persisted_prediction_target_id,
                    benchmark_weights_target_id="benchmark_weights"
                    if enhanced
                    else None,
                    risk_style_columns=RISK_STYLE_COLUMNS
                    if enhanced
                    else RISK_STYLE_COLUMNS[:-1],
                    risk_context_target_ids=(
                        market_ids.PRICE_VOLUME_DATA_ID,
                        "fundamental_factors",
                    )
                    if enhanced
                    else (market_ids.PRICE_VOLUME_DATA_ID,),
                    research_dataset_id=self.scenario_id,
                    market_panel_target_id=market_ids.MARKET_PANEL_ID,
                    execution_mask_target_id="execution_mask",
                    trading_calendar_target_id="trading_calendar",
                    fee_schedule_target_id="a_share_fee_schedule",
                )
            )
        return tuple(result)

    def portfolio_candidate_for_target(
        self,
        target_id: str,
    ) -> PortfolioCandidateSpec:
        for candidate in self.portfolio_candidates():
            if target_id == f"{self.scenario_id}_{candidate.candidate_id}_ledger":
                return candidate
        raise ValueError("unknown_a_share_portfolio_target")

    def terminal_targets(self, action: str) -> tuple[str, ...]:
        if action == "dataset build":
            return (self.scenario_id,)
        if action == "research run":
            if self.configuration_json is not None:
                if self.configuration()["stage"] == "evaluation":
                    return (f"{self.scenario_id}_evaluation",)
                return (
                    f"{self.scenario_id}_evaluation",
                    f"{self.scenario_id}_statistics",
                    *(
                        f"{self.scenario_id}_{candidate.candidate_id}_ledger"
                        for candidate in self.portfolio_candidates()
                    ),
                    *(
                        (f"{self.scenario_id}_portfolio_statistics",)
                        if self.portfolio_candidates()
                        else ()
                    ),
                )
            return (
                f"{self.scenario_id}_evaluation",
                f"{self.scenario_id}_statistics",
                f"{self.scenario_id}_portfolio_statistics",
                *(
                    f"{self.scenario_id}_{candidate.candidate_id}_ledger"
                    for candidate in self.portfolio_candidates()
                ),
            )
        if action == "data materialize":
            return ()
        raise ValueError("unknown_a_share_action")


def model_folds(
    sessions: pd.DatetimeIndex,
    *,
    chronology: Mapping[str, object],
    formal_dataset_window: Mapping[str, object],
    development_window: Mapping[str, object],
) -> tuple[ExperimentFold, ...]:
    if type(chronology) is not dict or set(chronology) != {"validation_months"}:
        raise ValueError("model_chronology_missing")
    history_start = pd.Timestamp(formal_dataset_window["start"])
    development_start = pd.Timestamp(development_window["start"])
    development_end = pd.Timestamp(development_window["end"])
    validation_months = chronology["validation_months"]
    if type(validation_months) is not int or validation_months <= 0:
        raise ValueError("model_chronology_validation_months_invalid")
    axis = pd.DatetimeIndex(sessions).normalize().drop_duplicates().sort_values()
    if axis.empty or axis.min() < history_start or axis.max() > development_end:
        raise ValueError("model_chronology_sessions_outside_development")
    if axis.min().year != history_start.year:
        raise ValueError("model_chronology_formal_history_missing")
    folds: list[ExperimentFold] = []
    for period in pd.PeriodIndex(axis.to_period("Q")).unique():
        if period < pd.Period(development_start, freq="Q"):
            continue
        test = axis[axis.to_period("Q") == period]
        validation = axis[
            (axis >= period.start_time - pd.DateOffset(months=validation_months))
            & (axis < period.start_time)
        ]
        training = (
            axis[axis < validation.min()]
            if not validation.empty
            else axis[axis < test.min()]
        )
        if training.empty or validation.empty or test.empty:
            raise ValueError("a_share_walk_forward_window_incomplete")
        folds.append(
            ExperimentFold(
                fold_id=f"{period.year:04d}-q{period.quarter}",
                train=TimeWindow(training[0], training[-1]),
                validation=TimeWindow(validation[0], validation[-1]),
                development_test=TimeWindow(test[0], test[-1]),
            )
        )
    return tuple(folds)


def analysis_windows(
    manifest: Mapping[str, object], requested: TimeWindow
) -> tuple[TimeWindow, TimeWindow]:
    """Resolve full-development factor history and out-of-sample model dates."""
    if manifest["access_window"] != "development":
        return requested, requested
    development = TimeWindow(**manifest["development_window"])
    factor_history = TimeWindow(
        manifest["formal_dataset_window"]["start"], development.end
    )
    if pd.Timestamp(requested.end) == pd.Timestamp(development.end) and pd.Timestamp(
        requested.start
    ) in {pd.Timestamp(development.start), pd.Timestamp(factor_history.start)}:
        return factor_history, development
    return requested, requested


def protocol(
    scenario_id: str,
    access_window: str = "development",
) -> AShareProtocol:
    return AShareProtocol(scenario_id, access_window)


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        (
            *price_volume_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id=market_ids.PRICE_VOLUME_DATA_ID,
                factor_columns=PRICE_VOLUME_LIBRARY,
            ),
            *(
                FactorDefinition(
                    factor_id=column,
                    market_id=market_ids.MARKET_ID,
                    factor_set=FactorSet.STANDARD,
                    input_columns=tuple(
                        dict.fromkeys(
                            (
                                "period_return",
                                "float_market_cap",
                                *A_SHARE_ELIGIBILITY_FACTS,
                            )
                        )
                    ),
                    output_column=column,
                    family="market_model",
                    grain=DataGrain.ASSET_TIMESTAMP,
                    required_input_data_id="market_model_factors",
                    lookback_window=312,
                    required_windows=(252, 60),
                    window=312,
                    description=(
                        "Prior-cap weighted eligible market; causal 252-session OLS "
                        "and 60-session population residual volatility."
                    ),
                    timing_semantics="same_session_close_available_at_next_session",
                    null_policy="preserve_nulls_after_min_periods",
                )
                for column in MARKET_MODEL_FACTORS
            ),
            *fundamental_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id="fundamental_pit",
            ),
            *expectations_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id="analyst_forecast",
            ),
        )
    )


def label_definitions() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            label_id=f"forward_return_{horizon}d",
            market_id=market_ids.MARKET_ID,
            family="forward_return",
            input_columns=("adj_open", "can_enter"),
            output_column=forward_return_label_column(horizon),
            label_timing="signal_close_to_next_open_to_horizon_open",
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_market_session_adjusted_open",
            exit_rule="entry_plus_horizon_market_sessions_open",
        )
        for horizon in LABEL_HORIZONS
    )


def label_catalog() -> LabelCatalog:
    return LabelCatalog(label_definitions())


def evaluation_catalog() -> EvaluationCatalog:
    return EvaluationCatalog(
        EvaluationSpec(
            evaluation_id=f"{selected}_evaluation",
            market_id=market_ids.MARKET_ID,
            research_dataset_id=RESEARCH_DATASET_IDS[selected],
            signals=tuple(
                SignalSpec(
                    signal_id=column,
                    source_kind="factor",
                    source_column=column,
                )
                for column in protocol(selected).factor_columns
            ),
            label_columns=(PRIMARY_LABEL_COLUMN,),
            label_availability_columns={
                PRIMARY_LABEL_COLUMN: forward_return_available_at_column(20)
            },
            n_groups=5,
        )
        for selected in SCENARIO_IDS
    )


def research_dataset_specs() -> tuple[ResearchDatasetSpec, ...]:
    specs: list[ResearchDatasetSpec] = []
    for scenario_id in SCENARIO_IDS:
        factor_columns = protocol(scenario_id).factor_columns
        feature_columns = tuple(
            feature
            for factor in factor_columns
            for feature in (
                f"feature_{factor}",
                f"feature_missing_{factor}",
                f"feature_observed_{factor}",
            )
        )
        labels = (
            *forward_label_columns(),
            *forward_label_available_columns(),
            MODEL_TARGET_COLUMN,
        )
        context = (
            "is_a_share_scope",
            "is_trading",
            "is_st",
            "is_pt",
            "listing_age_trading_days",
            "close",
            "adj_close",
            "volume",
            "amount",
            "turnover",
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
            "float_market_cap",
            "total_market_cap",
        )
        if scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            context += ("is_nonfinancial_industry", "has_legal_fundamental_pit")
        all_columns = (
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            *factor_columns,
            *feature_columns,
            *labels,
            *context,
        )
        roles = {
            column: _dataset_role(
                column,
                factor_columns,
                feature_columns,
                labels,
                ("eligible",),
            )
            for column in all_columns
        }
        transforms = canonical_feature_transforms(factor_columns)
        input_data_ids = [
            market_ids.MARKET_PANEL_ID,
            market_ids.PRICE_VOLUME_DATA_ID,
            "forward_returns",
        ]
        if set(factor_columns).intersection(MARKET_MODEL_FACTORS):
            input_data_ids.append("market_model_factors")
        if scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            input_data_ids.insert(3, "fundamental_factors")
        if scenario_id == "fusion_nf_expectations":
            input_data_ids.insert(-1, "expectations_factors")
        specs.append(
            ResearchDatasetSpec(
                data_id=RESEARCH_DATASET_IDS[scenario_id],
                market_id=market_ids.MARKET_ID,
                input_data_ids=tuple(input_data_ids),
                base_input_data_id=market_ids.MARKET_PANEL_ID,
                factor_columns=factor_columns,
                feature_columns=feature_columns,
                feature_transforms=transforms,
                label_columns=labels,
                filter_columns=("eligible",),
                context_columns=context,
                frequency=DataFrequency.DAILY,
                grain="research_row",
                column_roles=roles,
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=("year",),
                key_coverage={data_id: "strict" for data_id in input_data_ids},
            )
        )
    return tuple(specs)


def research_dataset_catalog():
    from quant_research_platform.catalog.datasets import ResearchDatasetCatalog

    return ResearchDatasetCatalog(research_dataset_specs())


def _dataset_role(
    column: str,
    factors: tuple[str, ...],
    features: tuple[str, ...],
    labels: tuple[str, ...],
    filters: tuple[str, ...],
) -> ColumnRole:
    if column in factors:
        return ColumnRole.FACTOR
    if column in features:
        return ColumnRole.FEATURE
    if column in labels:
        return ColumnRole.LABEL
    if column in filters:
        return ColumnRole.UNIVERSE
    if column == "timestamp":
        return ColumnRole.TIMESTAMP
    if column in {"market_id", "asset_id"}:
        return ColumnRole.KEY
    return ColumnRole.CONTEXT


def forward_label_columns() -> tuple[str, ...]:
    return tuple(forward_return_label_column(horizon) for horizon in LABEL_HORIZONS)


def forward_label_available_columns() -> tuple[str, ...]:
    return tuple(
        forward_return_available_at_column(horizon) for horizon in LABEL_HORIZONS
    )
