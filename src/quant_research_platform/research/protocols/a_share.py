"""A-share low-frequency research protocol."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.models import (
    ModelSpec,
    _model_algorithm_facts,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.contracts.predictions import PredictionSpec
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
FORMAL_START = date(2014, 1, 1)
FORMAL_END = date(2025, 12, 31)
DEVELOPMENT_START = date(2019, 1, 1)
DEVELOPMENT_END = date(2023, 12, 31)
HOLDOUT_START = date(2024, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
LABEL_HORIZONS = (1, 5, 10, 20, 60, 120)
PRIMARY_LABEL_COLUMN = forward_return_label_column(20)
MODEL_TARGET_COLUMN = "model_target"
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
MODEL_LADDER = ("single_factor_baseline", *MODEL_FAMILY_ORDER)
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
            "formal_window": {
                "start": FORMAL_START.isoformat(),
                "end": FORMAL_END.isoformat(),
            },
            "development_window": {
                "start": DEVELOPMENT_START.isoformat(),
                "end": DEVELOPMENT_END.isoformat(),
            },
            "holdout_window": {
                "start": HOLDOUT_START.isoformat(),
                "end": HOLDOUT_END.isoformat(),
            },
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
            "cost": {
                "commission_bps": COMMISSION_BPS,
                "slippage_bps": FIXED_SLIPPAGE_BPS,
                "impact_bps": IMPACT_BPS,
                "adv_participation": PRIMARY_PARTICIPATION,
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
        model_selection = f"{dataset}_model_selection_refit"
        predictions = f"{dataset}_persisted_predictions"
        statistics = f"{dataset}_statistics"
        portfolios = tuple(
            TargetSpec(
                f"{dataset}_{candidate}_ledger",
                dependencies=(predictions,),
            )
            for candidate in PORTFOLIO_CANDIDATES
        )
        return (
            TargetSpec(evaluation, dependencies=(dataset,)),
            TargetSpec(model_selection, dependencies=(evaluation,)),
            TargetSpec(predictions, dependencies=(model_selection,)),
            TargetSpec(statistics, dependencies=(predictions,)),
            *portfolios,
            TargetSpec(
                self._research_terminal_target,
                dependencies=(statistics, *(target.target_id for target in portfolios)),
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
        if facts.input_kind == "single_factor_input":
            input_factors = (selected.factor_columns[0],)
            input_features = ()
        elif facts.input_kind == "rank_ensemble_inputs":
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


def prediction_catalog(
    *,
    scenario_id: str = "pv_all",
    train_end: str,
    test_start: str,
    test_end: str | None = None,
    factor_scope: str = "core",
) -> PredictionCatalog:
    return PredictionCatalog(
        tuple(
            PredictionSpec(
                prediction_id=f"{model.model_id}_{_token(train_end)}_{_token(test_start)}",
                model_id=model.model_id,
                research_dataset_id=RESEARCH_DATASET_IDS[scenario_id],
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                candidate_model_ids=(),
            )
            for model in model_specs(scenario_id, factor_scope=factor_scope)
        )
    )


def evaluation_catalog(
    *, scenario_id: str = "pv_all", factor_scope: str = "core"
) -> EvaluationCatalog:
    signals = tuple(
        SignalSpec(
            signal_id=column,
            source_kind="factor",
            source_column=column,
        )
        for column in protocol(scenario_id, factor_scope=factor_scope).factor_columns
    )
    return EvaluationCatalog(
        (
            EvaluationSpec(
                evaluation_id=f"{scenario_id}_factor_evaluation",
                market_id=MARKET_ID,
                research_dataset_id=RESEARCH_DATASET_IDS[scenario_id],
                signals=signals,
                label_columns=(PRIMARY_LABEL_COLUMN,),
                n_groups=5,
            ),
        )
    )


def portfolio_catalog(
    *, scenario_id: str = "pv_all", factor_scope: str = "core"
) -> PortfolioCatalog:
    columns = protocol(scenario_id, factor_scope=factor_scope).factor_columns
    signals = tuple(
        SignalSpec(signal_id=column, source_kind="factor", source_column=column)
        for column in columns
    )
    diagnostics = PortfolioDiagnosticsConfig(
        capacity_amount_column="amount",
        capacity_float_market_cap_column="float_market_cap",
        categorical_exposure_columns=("industry_level1_code",),
    )
    tradability = TradabilityConfig(
        eligibility_column="eligible",
        can_enter_column="can_enter",
        can_exit_column="can_exit",
    )
    return PortfolioCatalog(
        (
            PortfolioSpec(
                portfolio_id=f"{scenario_id}_top20_equal_weight",
                market_id=MARKET_ID,
                research_dataset_id=RESEARCH_DATASET_IDS[scenario_id],
                signals=signals,
                rebalance_schedule="weekly-last-trading-day",
                execution_policy_id="a_share_next_open",
                cost_policy_id="a_share_historical_costs",
                cost_config=CostConfig(
                    commission_bps=COMMISSION_BPS,
                    slippage_bps=FIXED_SLIPPAGE_BPS,
                    impact_bps=IMPACT_BPS,
                    adv_participation=PRIMARY_PARTICIPATION,
                    adv_sensitivities=PARTICIPATION_SENSITIVITIES,
                ),
                weight_config=WeightConfig(
                    method="equal_weight_quantile",
                    portfolio_mode="long_only",
                    top_quantile=0.2,
                    bottom_quantile=0.2,
                ),
                diagnostics_config=diagnostics,
                tradability_config=tradability,
            ),
            PortfolioSpec(
                portfolio_id=f"{scenario_id}_top20_float_cap_weight",
                market_id=MARKET_ID,
                research_dataset_id=RESEARCH_DATASET_IDS[scenario_id],
                signals=signals,
                rebalance_schedule="weekly-last-trading-day",
                execution_policy_id="a_share_next_open",
                cost_policy_id="a_share_historical_costs",
                cost_config=CostConfig(
                    commission_bps=COMMISSION_BPS,
                    slippage_bps=FIXED_SLIPPAGE_BPS,
                    impact_bps=IMPACT_BPS,
                    adv_participation=PRIMARY_PARTICIPATION,
                    adv_sensitivities=PARTICIPATION_SENSITIVITIES,
                ),
                weight_config=WeightConfig(
                    method="float_cap_weighted_quantile",
                    portfolio_mode="long_only",
                    top_quantile=0.2,
                    bottom_quantile=0.2,
                ),
                diagnostics_config=diagnostics,
                tradability_config=tradability,
            ),
        )
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


def _token(value: str) -> str:
    return value.replace(":", "").replace("-", "")
