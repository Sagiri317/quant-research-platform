"""A-share low-frequency research protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.models import (
    ModelSpec,
    _model_algorithm_facts,
)
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.targets import TargetSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.expectations.definitions import (
    EXPECTATIONS_FACTORS,
    expectations_factor_definitions,
)
from quant_research_platform.research.factors.fundamental.definitions import (
    CORE_FUNDAMENTAL_FACTORS,
    FUNDAMENTAL_FACTORS,
    fundamental_factor_definitions,
)
from quant_research_platform.research.factors.price_volume.columns import (
    FORMAL_CORE_PRICE_VOLUME_FACTORS,
    FORMAL_PRICE_VOLUME_FACTORS,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    formal_price_volume_factor_definitions,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_available_at_column,
    forward_return_label_column,
)
from quant_research_platform.research.randomness import ROOT_SEED

MARKET_ID = MarketId("a_share")
PROTOCOL_ID = "a_share_low_frequency"
MARKET_PANEL_ID = "market_panel_1d"
PRICE_VOLUME_DATA_ID = "price_volume_facts"
RESEARCH_DATASET_IDS = {
    "pv_all": "pv_all",
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


PV_CORE = FORMAL_CORE_PRICE_VOLUME_FACTORS
PV_FULL = FORMAL_PRICE_VOLUME_FACTORS
FUND_CORE = CORE_FUNDAMENTAL_FACTORS
FUND_FULL = FUNDAMENTAL_FACTORS
EXP_FULL = EXPECTATIONS_FACTORS
MODEL_FAMILY_ORDER = (
    "equal_weight_rank_ensemble",
    "ic_weighted_rank_ensemble",
    "linear_regression",
    "huber_regression",
    "ridge_regression",
    "lasso_regression",
    "random_forest_regressor",
    "xgboost_regressor",
)
SELECTION_TIE_BREAK = MODEL_FAMILY_ORDER
MODEL_LADDER = ("single_factor_diagnostic", *MODEL_FAMILY_ORDER)
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0)
HUBER_LAMBDAS = RIDGE_LAMBDAS
LASSO_ALPHAS = (0.001, 0.003, 0.01, 0.03)
RF_CONFIGS = (
    {
        "max_depth": 6,
        "min_weight_fraction_leaf": 0.001,
        "n_estimators": 300,
        "max_features": 0.5,
        "bootstrap": True,
        "max_samples": 0.5,
    },
    {
        "max_depth": 10,
        "min_weight_fraction_leaf": 0.0003,
        "n_estimators": 300,
        "max_features": 0.5,
        "bootstrap": True,
        "max_samples": 0.5,
    },
    {
        "max_depth": 14,
        "min_weight_fraction_leaf": 0.0001,
        "n_estimators": 300,
        "max_features": 0.5,
        "bootstrap": True,
        "max_samples": 0.5,
    },
)
XGB_CONFIGS = (
    {"max_depth": 3, "min_child_weight_fraction": 0.001},
    {"max_depth": 5, "min_child_weight_fraction": 0.0003},
    {"max_depth": 7, "min_child_weight_fraction": 0.0001},
)
INITIAL_NAV = 1_000_000_000.0
PRIMARY_PARTICIPATION = 0.05
PARTICIPATION_SENSITIVITIES = (0.01, 0.02, 0.05, 0.10)
FIXED_SLIPPAGE_BPS = 5.0
IMPACT_BPS = 25.0
COMMISSION_BPS = 3.0
EXECUTION_ASSUMPTIONS = ExecutionAssumptions(
    initial_nav=INITIAL_NAV,
    slippage_bps=FIXED_SLIPPAGE_BPS,
    impact_bps=IMPACT_BPS,
    adv_participation=PRIMARY_PARTICIPATION,
    adv_sensitivities=PARTICIPATION_SENSITIVITIES,
)
ANNUALIZATION = 252
STATISTICS_BLOCK = 20
STATISTICS_REPETITIONS = 10_000
HAC_LAG = 19
ROBUSTNESS_TIME_DECAY_HALF_LIFE = 756
RISK_STYLE_COLUMNS = (
    "log_float_market_cap",
    "beta_252d",
    "momentum_252_20",
    "residual_volatility_60d",
    "amihud_60d",
    "earnings_yield",
)
PORTFOLIO_CANDIDATES = (
    "ew_rank_top20_equal_weight",
    "champion_top20_equal_weight",
    "champion_top20_float_cap",
    "champion_csi500_enhanced",
    "champion_csi1000_enhanced",
)


@dataclass(frozen=True)
class AShareProtocol:
    scenario_id: str
    access_window: str = "development"
    factor_scope: str = "core"
    root_seed: int = ROOT_SEED

    def __post_init__(self) -> None:
        if self.scenario_id not in SCENARIO_IDS:
            raise ValueError("unknown_a_share_scenario")
        if self.access_window not in ACCESS_WINDOWS:
            raise ValueError("unknown_a_share_access_window")
        if self.factor_scope not in {"core", "full"}:
            raise ValueError("unknown_a_share_factor_scope")
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

    @property
    def protocol_id(self) -> str:
        return PROTOCOL_ID

    @property
    def factor_columns(self) -> tuple[str, ...]:
        if self.scenario_id == "pv_all":
            return PV_CORE if self.factor_scope == "core" else PV_FULL
        if self.scenario_id == "fusion_nf_core":
            return (
                (*PV_CORE, *FUND_CORE)
                if self.factor_scope == "core"
                else (*PV_FULL, *FUND_FULL)
            )
        return (
            (*PV_CORE, *FUND_CORE, *EXP_FULL)
            if self.factor_scope == "core"
            else (*PV_FULL, *FUND_FULL, *EXP_FULL)
        )

    def manifest(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "scenario_id": self.scenario_id,
            "access_window": self.access_window,
            "factor_scope": self.factor_scope,
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
            "factor_columns": list(self.factor_columns),
            "label_horizons": list(LABEL_HORIZONS),
            "primary_label": PRIMARY_LABEL_COLUMN,
            "model_target": MODEL_TARGET_COLUMN,
            "model_ladder": list(MODEL_LADDER),
            "selection_tie_break": list(SELECTION_TIE_BREAK),
            "model_configs": {
                "ridge_lambda": list(RIDGE_LAMBDAS),
                "huber_lambda": list(HUBER_LAMBDAS),
                "lasso_alpha": list(LASSO_ALPHAS),
                "random_forest": [dict(config) for config in RF_CONFIGS],
                "xgboost": [dict(config) for config in XGB_CONFIGS],
            },
            "sample_weight": "equal_date_total_equal_sum_n_train",
            "primary_time_decay": None,
            "robustness_time_decay_half_life": ROBUSTNESS_TIME_DECAY_HALF_LIFE,
            "transform_order": [
                "eligible",
                "mad_winsor",
                "neutralize",
                "zscore",
            ],
            "root_seed": self.root_seed,
            "execution_assumptions": {
                "initial_nav": EXECUTION_ASSUMPTIONS.initial_nav,
                "slippage_bps": EXECUTION_ASSUMPTIONS.slippage_bps,
                "impact_bps": EXECUTION_ASSUMPTIONS.impact_bps,
                "adv_participation": EXECUTION_ASSUMPTIONS.adv_participation,
                "adv_sensitivities": list(EXECUTION_ASSUMPTIONS.adv_sensitivities),
            },
            "risk": {
                "annualization": ANNUALIZATION,
                "factor_window": 504,
                "factor_half_life": 126,
                "specific_window": 252,
                "specific_half_life": 63,
                "hard_constraints": {
                    "active_name": 0.005,
                    "tracking_error": 0.05,
                    "target_turnover": 0.20,
                    "industry_active": 0.02,
                    "style_active": 0.20,
                },
            },
            "statistics": {
                "hac_lag": HAC_LAG,
                "bootstrap_block": STATISTICS_BLOCK,
                "bootstrap_repetitions": STATISTICS_REPETITIONS,
                "fdr_primary": "BH_5pct",
                "fdr_robustness": "BY_5pct",
                "cscv_blocks": 10,
                "cscv_combinations": 252,
            },
            "portfolio_candidates": list(PORTFOLIO_CANDIDATES),
            "risk_styles": list(RISK_STYLE_COLUMNS),
            "required_terminals": list(self.terminal_targets("research run")),
        }

    def target_specs(self, action: str) -> tuple[TargetSpec, ...]:
        if action not in {"research run", "report build"}:
            return ()
        dataset = self.scenario_id
        evaluation = f"{dataset}_evaluation"
        model_selection = self.model_selection_target_id
        ew_model = self.ew_rank_model_target_id
        predictions = self.persisted_prediction_target_id
        ew_predictions = self.ew_rank_prediction_target_id
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
                dependencies=(dataset,),
                producer_id=model_selection,
            ),
            TargetSpec(
                ew_model,
                dependencies=(dataset,),
                producer_id=ew_model,
            ),
            TargetSpec(
                predictions,
                dependencies=(dataset, model_selection),
                producer_id=predictions,
            ),
            TargetSpec(
                ew_predictions,
                dependencies=(dataset, ew_model),
                producer_id=ew_predictions,
            ),
            TargetSpec(statistics, dependencies=(predictions,)),
            *portfolios,
            TargetSpec(
                self._research_terminal_target,
                dependencies=(
                    evaluation,
                    statistics,
                    *(target.target_id for target in portfolios),
                ),
            ),
            TargetSpec(
                self._report_artifact_target,
                dependencies=(self._research_terminal_target,),
            ),
            TargetSpec(
                self._report_terminal_target,
                dependencies=(self._report_artifact_target,),
            ),
        )

    @property
    def model_selection_target_id(self) -> str:
        return f"{self.scenario_id}_model_selection_refit"

    @property
    def ew_rank_model_target_id(self) -> str:
        return f"{self.scenario_id}_ew_rank_model"

    @property
    def persisted_prediction_target_id(self) -> str:
        return f"{self.scenario_id}_persisted_predictions"

    @property
    def ew_rank_prediction_target_id(self) -> str:
        return f"{self.scenario_id}_ew_rank_predictions"

    def model_ids_for_target(self, target_id: str) -> tuple[str, ...]:
        if target_id == self.model_selection_target_id:
            model_ids = tuple(
                f"{self.scenario_id}_{algorithm}" for algorithm in MODEL_FAMILY_ORDER
            )
        elif target_id == self.ew_rank_model_target_id:
            model_ids = (f"{self.scenario_id}_equal_weight_rank_ensemble",)
        else:
            raise ValueError("unknown_a_share_model_target")
        return model_ids

    @property
    def execution_assumptions(self) -> ExecutionAssumptions:
        return EXECUTION_ASSUMPTIONS

    def portfolio_candidates(self) -> tuple[PortfolioCandidateSpec, ...]:
        prediction = self.persisted_prediction_target_id
        return (
            PortfolioCandidateSpec(
                candidate_id="ew_rank_top20_equal_weight",
                construction="top20_equal_weight",
                prediction_target_id=self.ew_rank_prediction_target_id,
                top_quantile=0.2,
                rebalance_schedule="weekly-last-trading-day",
                benchmark_id=None,
                benchmark_weights_target_id=None,
                research_dataset_id=self.scenario_id,
                market_panel_target_id=MARKET_PANEL_ID,
                execution_mask_target_id="execution_mask",
                trading_calendar_target_id="trading_calendar",
                fee_schedule_target_id="a_share_fee_schedule",
            ),
            PortfolioCandidateSpec(
                candidate_id="champion_top20_equal_weight",
                construction="top20_equal_weight",
                prediction_target_id=prediction,
                top_quantile=0.2,
                rebalance_schedule="weekly-last-trading-day",
                benchmark_id=None,
                benchmark_weights_target_id=None,
                research_dataset_id=self.scenario_id,
                market_panel_target_id=MARKET_PANEL_ID,
                execution_mask_target_id="execution_mask",
                trading_calendar_target_id="trading_calendar",
                fee_schedule_target_id="a_share_fee_schedule",
            ),
            PortfolioCandidateSpec(
                candidate_id="champion_top20_float_cap",
                construction="top20_float_cap",
                prediction_target_id=prediction,
                top_quantile=0.2,
                rebalance_schedule="weekly-last-trading-day",
                benchmark_id=None,
                benchmark_weights_target_id=None,
                research_dataset_id=self.scenario_id,
                market_panel_target_id=MARKET_PANEL_ID,
                execution_mask_target_id="execution_mask",
                trading_calendar_target_id="trading_calendar",
                fee_schedule_target_id="a_share_fee_schedule",
            ),
            PortfolioCandidateSpec(
                candidate_id="champion_csi500_enhanced",
                construction="benchmark_enhanced",
                prediction_target_id=prediction,
                top_quantile=None,
                rebalance_schedule="weekly-last-trading-day",
                benchmark_id="csi500",
                benchmark_weights_target_id="benchmark_weights",
                risk_style_columns=RISK_STYLE_COLUMNS,
                research_dataset_id=self.scenario_id,
                market_panel_target_id=MARKET_PANEL_ID,
                execution_mask_target_id="execution_mask",
                trading_calendar_target_id="trading_calendar",
                fee_schedule_target_id="a_share_fee_schedule",
            ),
            PortfolioCandidateSpec(
                candidate_id="champion_csi1000_enhanced",
                construction="benchmark_enhanced",
                prediction_target_id=prediction,
                top_quantile=None,
                rebalance_schedule="weekly-last-trading-day",
                benchmark_id="csi1000",
                benchmark_weights_target_id="benchmark_weights",
                risk_style_columns=RISK_STYLE_COLUMNS,
                research_dataset_id=self.scenario_id,
                market_panel_target_id=MARKET_PANEL_ID,
                execution_mask_target_id="execution_mask",
                trading_calendar_target_id="trading_calendar",
                fee_schedule_target_id="a_share_fee_schedule",
            ),
        )

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
            return (self._research_terminal_target,)
        if action == "report build":
            return (self._report_terminal_target,)
        if action == "data materialize":
            return ()
        raise ValueError("unknown_a_share_action")

    @property
    def _research_terminal_target(self) -> str:
        return f"{self.scenario_id}_research"

    @property
    def _report_artifact_target(self) -> str:
        return f"{self.scenario_id}_report_artifact"

    @property
    def _report_terminal_target(self) -> str:
        return f"{self.scenario_id}_report"


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


def protocol(
    scenario_id: str,
    access_window: str = "development",
    *,
    factor_scope: str = "core",
) -> AShareProtocol:
    return AShareProtocol(scenario_id, access_window, factor_scope=factor_scope)


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        (
            *formal_price_volume_factor_definitions(
                market_id=MARKET_ID,
                required_input_data_id=PRICE_VOLUME_DATA_ID,
            ),
            *fundamental_factor_definitions(
                market_id=MARKET_ID,
                required_input_data_id="fundamental_pit",
            ),
            *expectations_factor_definitions(
                market_id=MARKET_ID,
                required_input_data_id="analyst_forecast",
            ),
        )
    )


def label_definitions() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            label_id=f"forward_return_{horizon}d",
            market_id=MARKET_ID,
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


def model_specs(
    scenario_id: str = "pv_all",
    *,
    factor_scope: str = "core",
    model_algorithm_ids: Iterable[str] = MODEL_FAMILY_ORDER,
    random_seed: int = ROOT_SEED,
) -> tuple[ModelSpec, ...]:
    selected = protocol(scenario_id, factor_scope=factor_scope)
    result: list[ModelSpec] = []
    for algorithm in model_algorithm_ids:
        facts = _model_algorithm_facts(algorithm)
        input_factors = selected.factor_columns
        input_features = tuple(
            column
            for factor in selected.factor_columns
            for column in (f"feature_{factor}", f"feature_missing_{factor}")
        )
        if facts.input_kind == "rank_ensemble_inputs":
            input_factors = selected.factor_columns
            input_features = ()
        else:
            input_factors = ()
        result.append(
            ModelSpec(
                model_id=f"{scenario_id}_{algorithm}",
                model_algorithm=algorithm,
                input_factor_columns=input_factors,
                input_feature_columns=input_features,
                label_column=MODEL_TARGET_COLUMN,
                random_seed=random_seed,
                hyperparameters=_base_model_parameters(algorithm),
            )
        )
    return tuple(result)


def _base_model_parameters(algorithm: str) -> dict[str, object]:
    if algorithm == "linear_regression":
        return {}
    if algorithm == "ridge_regression":
        return {"alpha": 1.0}
    if algorithm == "huber_regression":
        return {
            "epsilon": 1.35,
            "alpha": 1.0,
            "max_iter": 1000,
            "tol": 1e-5,
            "warm_start": False,
        }
    if algorithm == "lasso_regression":
        return {
            "alpha": 0.001,
            "fit_intercept": True,
            "selection": "cyclic",
            "max_iter": 20_000,
            "tol": 1e-5,
            "positive": False,
        }
    if algorithm == "random_forest_regressor":
        return dict(RF_CONFIGS[0])
    if algorithm == "xgboost_regressor":
        return {
            "max_depth": 3,
            "min_child_weight": 1.0,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_bin": 256,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "gamma": 0.0,
            "device": "cpu",
        }
    return {}


def model_catalog(**kwargs: object) -> ModelCatalog:
    return ModelCatalog(model_specs(**kwargs))


def all_model_catalog() -> ModelCatalog:
    return ModelCatalog(
        tuple(
            model for scenario_id in SCENARIO_IDS for model in model_specs(scenario_id)
        )
    )


def resolve_trial_parameters(
    algorithm: str,
    *,
    n_train: int,
    config_index: int = 0,
) -> dict[str, object]:
    """Resolve protocol strengths to backend-native estimator parameters."""

    if type(n_train) is not int or n_train <= 0:
        raise ValueError("n_train_must_be_positive")
    if type(config_index) is not int or config_index < 0:
        raise ValueError("model_config_index_invalid")
    if algorithm == "ridge_regression":
        value = RIDGE_LAMBDAS[config_index % len(RIDGE_LAMBDAS)]
        return {"alpha": n_train * value}
    if algorithm == "huber_regression":
        value = HUBER_LAMBDAS[config_index % len(HUBER_LAMBDAS)]
        return {
            "epsilon": 1.35,
            "alpha": n_train * value,
            "max_iter": 1000,
            "tol": 1e-5,
            "warm_start": False,
        }
    if algorithm == "lasso_regression":
        return {
            "alpha": LASSO_ALPHAS[config_index % len(LASSO_ALPHAS)],
            "fit_intercept": True,
            "selection": "cyclic",
            "max_iter": 20_000,
            "tol": 1e-5,
            "positive": False,
        }
    if algorithm == "random_forest_regressor":
        return dict(RF_CONFIGS[config_index % len(RF_CONFIGS)])
    if algorithm == "xgboost_regressor":
        values = dict(XGB_CONFIGS[config_index % len(XGB_CONFIGS)])
        values["min_child_weight"] = n_train * values.pop("min_child_weight_fraction")
        return {
            **values,
            "n_estimators": 400,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_bin": 256,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "gamma": 0.0,
            "device": "cpu",
        }
    if algorithm in {
        "linear_regression",
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        return {}
    raise ValueError("unsupported_a_share_trial_algorithm")


def evaluation_catalog() -> EvaluationCatalog:
    return EvaluationCatalog(
        EvaluationSpec(
            evaluation_id=f"{selected}_evaluation",
            market_id=MARKET_ID,
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
            n_groups=5,
        )
        for selected in SCENARIO_IDS
    )


def research_dataset_specs() -> tuple[ResearchDatasetSpec, ...]:
    specs: list[ResearchDatasetSpec] = []
    for scenario_id in SCENARIO_IDS:
        factor_columns = (
            (*PV_FULL, *FUND_FULL)
            if scenario_id == "fusion_nf_core"
            else (*PV_FULL, *FUND_FULL, *EXP_FULL)
            if scenario_id == "fusion_nf_expectations"
            else PV_FULL
        )
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
        transforms = tuple(
            transform
            for column in factor_columns
            for transform in (
                FeatureTransformSpec(
                    transform_id=f"feature_{column}",
                    source_columns=(column,),
                    output_columns=(f"feature_{column}",),
                    transform="mad_winsorized_zscore",
                ),
                FeatureTransformSpec(
                    transform_id=f"feature_missing_{column}",
                    source_columns=(column,),
                    output_columns=(f"feature_missing_{column}",),
                    transform="missing_indicator",
                ),
                FeatureTransformSpec(
                    transform_id=f"feature_observed_{column}",
                    source_columns=(column,),
                    output_columns=(f"feature_observed_{column}",),
                    transform="observed_indicator",
                ),
            )
        )
        input_data_ids = [MARKET_PANEL_ID, PRICE_VOLUME_DATA_ID, "forward_returns"]
        if scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            input_data_ids.insert(3, "fundamental_factors")
        if scenario_id == "fusion_nf_expectations":
            input_data_ids.insert(-1, "expectations_factors")
        specs.append(
            ResearchDatasetSpec(
                data_id=RESEARCH_DATASET_IDS[scenario_id],
                market_id=MARKET_ID,
                input_data_ids=tuple(input_data_ids),
                base_input_data_id=MARKET_PANEL_ID,
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
