"""Crypto HFT protocol constants and catalogs; market facts stay in crypto."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import timedelta

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
)
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.targets import TargetSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    is_intraday_time_window,
    time_value_isoformat,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.research.randomness import ROOT_SEED

MARKET_ID = MarketId("crypto")
PROTOCOL_ID = "crypto_hft"
HFT_RESEARCH_DATASET_ID = "binance_futures_hft_research_100ms"
HFT_FACTOR_COLUMNS = (
    "spread",
    "relative_spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice",
    "microprice_deviation",
    "ofi_100ms",
    "signed_trade_flow_100ms",
    "taker_buy_volume_100ms",
    "taker_sell_volume_100ms",
    "return_100ms",
    "realized_volatility_1s",
    "top10_depth_per_spread",
)
HFT_LABEL_HORIZONS_MS = (100, 500, 1_000, 5_000)
CRYPTO_FORWARD_RETURN_HORIZONS = (1, 5, 15)
CRYPTO_FORWARD_RETURN_COLUMNS = tuple(
    forward_return_label_column(horizon, DataFrequency.MINUTE)
    for horizon in CRYPTO_FORWARD_RETURN_HORIZONS
)
HFT_PRICE_LABEL_COLUMNS = tuple(
    column
    for horizon in HFT_LABEL_HORIZONS_MS
    for column in (
        f"future_mid_return_{horizon}ms",
        f"future_mid_direction_{horizon}ms",
    )
)
HFT_LABEL_COLUMNS = (
    *HFT_PRICE_LABEL_COLUMNS,
    "maker_bid_sell_pressure_proxy_500ms",
)
HFT_LABEL_AVAILABILITY_COLUMNS = tuple(
    f"{column}_available_at" for column in HFT_LABEL_COLUMNS
)
HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS = 500
HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN = "maker_bid_sell_pressure_proxy_500ms"
HFT_MODEL_HORIZON_MS = 500
HFT_MODEL_LABEL_COLUMN = f"future_mid_return_{HFT_MODEL_HORIZON_MS}ms"
HFT_MODEL_ALGORITHMS = (
    "lightgbm_regressor",
    "linear_regression",
    "xgboost_regressor",
    "gradient_boosting_regressor",
    "ridge_regression",
)
HFT_PRICE_MODEL_ALGORITHMS = HFT_MODEL_ALGORITHMS[:3]
HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID = "crypto_hft_maker_bid_sell_pressure_proxy"
HFT_MULTI_ASSET_SCOPES = {
    "btc_independent": (("btc_usdt",), ("btc_usdt",)),
    "eth_independent": (("eth_usdt",), ("eth_usdt",)),
    "pooled": (("btc_usdt", "eth_usdt"), ("btc_usdt", "eth_usdt")),
    "leave_btc_out": (("eth_usdt",), ("btc_usdt",)),
    "leave_eth_out": (("btc_usdt",), ("eth_usdt",)),
}
HFT_EVALUATION_ID = "crypto_hft_microstructure_evaluation"
HFT_LATENCY_SCENARIOS_MS = (0, 100, 200)
HFT_TAKER_BASE_MODEL_ID = "crypto_hft_linear_regression"
HFT_MAKER_CALIBRATION_MODEL_IDS = (
    HFT_TAKER_BASE_MODEL_ID,
    HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
)
HFT_EXECUTION_CONTEXT_COLUMNS = (
    "book_segment_id",
    "trade_connection_id",
    "segment_id",
)


@dataclass(frozen=True)
class CryptoMakerPlan:
    maker_fee_bps: float
    latency_ms: int
    min_calibration_samples: int
    inventory_target: float
    inventory_skew: float

    def __post_init__(self):
        for key in ("maker_fee_bps", "inventory_target", "inventory_skew"):
            object.__setattr__(
                self,
                key,
                strict_finite_float(
                    getattr(self, key), error="crypto_maker_parameters_invalid"
                ),
            )
        if (
            self.inventory_skew < 0
            or type(self.latency_ms) is not int
            or self.latency_ms < 0
            or type(self.min_calibration_samples) is not int
            or self.min_calibration_samples <= 0
        ):
            raise ValueError("crypto_maker_parameters_invalid")


@dataclass(frozen=True)
class CryptoExecutionPlan:
    taker_fee_bps: float
    latency_ms: int
    markout_horizon_ms: int
    latency_scenarios_ms: tuple[int, ...]
    fee_scenarios_bps: tuple[float, ...]
    additional_spread_scenarios_bps: tuple[float, ...]
    maker: CryptoMakerPlan | None = None

    def __post_init__(self):
        fee = strict_finite_float(self.taker_fee_bps, error="crypto_taker_fee_invalid")
        if (
            fee < 0
            or type(self.latency_ms) is not int
            or self.latency_ms < 0
            or type(self.markout_horizon_ms) is not int
            or self.markout_horizon_ms <= 0
            or type(self.latency_scenarios_ms) is not tuple
            or any(type(value) is not int for value in self.latency_scenarios_ms)
            or self.latency_scenarios_ms != HFT_LATENCY_SCENARIOS_MS
        ):
            raise ValueError("crypto_execution_parameters_invalid")
        for values in (self.fee_scenarios_bps, self.additional_spread_scenarios_bps):
            if type(values) is not tuple or not values:
                raise ValueError("crypto_cost_sensitivity_invalid")
            if any(
                strict_finite_float(value, error="crypto_cost_sensitivity_invalid") < 0
                for value in values
            ):
                raise ValueError("crypto_cost_sensitivity_invalid")
            if len(set(values)) != len(values):
                raise ValueError("crypto_cost_sensitivity_invalid")
        object.__setattr__(self, "taker_fee_bps", fee)
        for key in ("fee_scenarios_bps", "additional_spread_scenarios_bps"):
            object.__setattr__(
                self, key, tuple(float(value) for value in getattr(self, key))
            )


@dataclass(frozen=True)
class CryptoTakerCandidate:
    candidate_id: str
    asset_id: str
    model_id: str
    latency_ms: int | None = None

    @property
    def target_id(self):
        return f"{self.candidate_id}_ledger"


@dataclass(frozen=True)
class CryptoMakerCandidate:
    candidate_id: str
    asset_id: str
    calibrated: bool

    @property
    def target_id(self):
        return f"{self.candidate_id}_ledger"


def maker_candidates():
    return (
        CryptoMakerCandidate("crypto_hft_maker_baseline", "btc_usdt", False),
        CryptoMakerCandidate("crypto_hft_maker", "btc_usdt", True),
    )


def portfolio_candidates():
    return (*taker_candidates(), *maker_candidates())


def taker_candidates():
    return (
        *(
            CryptoTakerCandidate(
                f"crypto_hft_{algorithm}", "btc_usdt", f"crypto_hft_{algorithm}"
            )
            for algorithm in HFT_PRICE_MODEL_ALGORITHMS
        ),
        *(
            CryptoTakerCandidate(
                f"crypto_hft_{mode}_{asset}", asset, f"crypto_hft_{mode}"
            )
            for mode, (_, assets) in HFT_MULTI_ASSET_SCOPES.items()
            for asset in assets
        ),
        *(
            CryptoTakerCandidate(
                f"crypto_hft_latency_{latency}",
                "btc_usdt",
                HFT_TAKER_BASE_MODEL_ID,
                latency,
            )
            for latency in HFT_LATENCY_SCENARIOS_MS
        ),
    )


@dataclass(frozen=True)
class _CryptoResearchPlan:
    folds: tuple[ExperimentFold, ...]
    final_holdout: TimeWindow
    initial_capital: float
    valuation_currency: str
    exposure_limits: tuple[tuple[str, float, float], ...]
    execution: CryptoExecutionPlan | None = None

    def payload(self):
        def window(value):
            return {
                "start": time_value_isoformat(value.start, preserve_datetime=True),
                "end": time_value_isoformat(value.end, preserve_datetime=True),
            }

        result = {
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "train": window(fold.train),
                    "validation": window(fold.validation),
                    "development_test": window(fold.development_test),
                }
                for fold in self.folds
            ],
            "final_holdout": window(self.final_holdout),
            "initial_capital": self.initial_capital,
            "valuation_currency": self.valuation_currency,
            "exposure_limits": {
                asset: {"max_order_quantity": order, "max_position_quantity": position}
                for asset, order, position in self.exposure_limits
            },
        }
        if self.execution is not None:
            result["execution"] = {
                key: list(value) if type(value) is tuple else value
                for key, value in asdict(self.execution).items()
                if value is not None
            }
        return result


def _research_plan(value: Mapping[str, object]) -> _CryptoResearchPlan:
    snapshot = bounded_qrp_json_object_snapshot(value, max_bytes=1024 * 1024)
    if set(snapshot) - {"execution"} != {
        "folds",
        "final_holdout",
        "initial_capital",
        "valuation_currency",
        "exposure_limits",
    }:
        raise ValueError("crypto_research_plan_fields_invalid")

    def window(value):
        if type(value) is not dict or set(value) != {"start", "end"}:
            raise ValueError("crypto_research_plan_window_invalid")
        result = TimeWindow(value["start"], value["end"])
        if not is_intraday_time_window(result):
            raise ValueError("crypto_research_plan_requires_intraday_window")
        return result

    raw_folds = snapshot["folds"]
    if type(raw_folds) is not list or not raw_folds:
        raise ValueError("crypto_research_plan_folds_required")
    folds = []
    for item in raw_folds:
        if type(item) is not dict or set(item) != {
            "fold_id",
            "train",
            "validation",
            "development_test",
        }:
            raise ValueError("crypto_research_plan_fold_invalid")
        fold = ExperimentFold(
            fold_id=item["fold_id"],
            train=window(item["train"]),
            validation=window(item["validation"]),
            development_test=window(item["development_test"]),
        )
        if any(
            part.end - part.start < timedelta(milliseconds=HFT_MODEL_HORIZON_MS)
            for part in (fold.train, fold.validation)
        ):
            raise ValueError("crypto_research_plan_label_window_too_short")
        if folds and folds[-1].development_test.end >= fold.development_test.start:
            raise ValueError("crypto_research_plan_oos_windows_overlap")
        folds.append(fold)
    if len({fold.fold_id for fold in folds}) != len(folds):
        raise ValueError("crypto_research_plan_duplicate_fold")
    holdout = window(snapshot["final_holdout"])
    if folds[-1].development_test.end >= holdout.start:
        raise ValueError("crypto_research_plan_holdout_overlap")
    capital = PortfolioCapital(
        "crypto_plan",
        snapshot["initial_capital"],
        snapshot["valuation_currency"],
        "linear_perpetual",
    )
    if capital.valuation_currency != "USDT":
        raise ValueError("crypto_research_plan_currency_invalid")
    raw_limits = snapshot["exposure_limits"]
    if type(raw_limits) is not dict or set(raw_limits) != set(
        HFT_MULTI_ASSET_SCOPES["pooled"][0]
    ):
        raise ValueError("crypto_research_plan_exposure_limits_required")
    limits = []
    for asset, item in sorted(raw_limits.items()):
        if type(item) is not dict or set(item) != {
            "max_order_quantity",
            "max_position_quantity",
        }:
            raise ValueError("crypto_research_plan_exposure_limits_invalid")
        order = strict_finite_float(
            item["max_order_quantity"],
            error="crypto_research_plan_exposure_limits_invalid",
        )
        position = strict_finite_float(
            item["max_position_quantity"],
            error="crypto_research_plan_exposure_limits_invalid",
        )
        if min(order, position) <= 0:
            raise ValueError("crypto_research_plan_exposure_limits_invalid")
        limits.append((asset, order, position))
    execution = None
    if "execution" in snapshot:
        value = snapshot["execution"]
        if type(value) is not dict or set(value) - {"maker"} != {
            "taker_fee_bps",
            "latency_ms",
            "markout_horizon_ms",
            "latency_scenarios_ms",
            "fee_scenarios_bps",
            "additional_spread_scenarios_bps",
        }:
            raise ValueError("crypto_execution_fields_invalid")
        for key in (
            "latency_scenarios_ms",
            "fee_scenarios_bps",
            "additional_spread_scenarios_bps",
        ):
            if type(value[key]) is not list:
                raise ValueError("crypto_execution_scenarios_invalid")
        if "maker" in value and (
            type(value["maker"]) is not dict
            or set(value["maker"])
            != {
                "maker_fee_bps",
                "latency_ms",
                "min_calibration_samples",
                "inventory_target",
                "inventory_skew",
            }
        ):
            raise ValueError("crypto_maker_fields_invalid")
        execution = CryptoExecutionPlan(
            **{
                key: CryptoMakerPlan(**item)
                if key == "maker" and type(item) is dict
                else tuple(item)
                if type(item) is list
                else item
                for key, item in value.items()
            }
        )
        if "maker" in value and type(execution.maker) is not CryptoMakerPlan:
            raise ValueError("crypto_maker_parameters_invalid")
        if execution.maker is not None and abs(execution.maker.inventory_target) > next(
            position for asset, _order, position in limits if asset == "btc_usdt"
        ):
            raise ValueError("crypto_maker_inventory_target_invalid")
    return _CryptoResearchPlan(
        tuple(folds),
        holdout,
        float(capital.initial_capital),
        capital.valuation_currency,
        tuple(limits),
        execution,
    )


@dataclass(frozen=True)
class CryptoProtocol:
    scenario_id: str = HFT_RESEARCH_DATASET_ID
    access_window: str = "development"
    research_plan: _CryptoResearchPlan | None = None

    def __post_init__(self) -> None:
        if self.scenario_id != HFT_RESEARCH_DATASET_ID:
            raise ValueError("unknown_crypto_scenario")
        if self.research_plan is not None and not isinstance(
            self.research_plan, _CryptoResearchPlan
        ):
            raise TypeError("crypto_research_plan_invalid")
        if self.access_window not in {"development", "holdout"}:
            raise ValueError("unknown_crypto_access_window")

    @property
    def protocol_id(self) -> str:
        return PROTOCOL_ID

    def manifest(self) -> dict[str, object]:
        result = {
            "protocol_id": self.protocol_id,
            "scenario_id": self.scenario_id,
            "access_window": self.access_window,
            "required_terminals": list(self.terminal_targets("research run")),
            "portfolio_candidates": [
                asdict(candidate) for candidate in portfolio_candidates()
            ],
            "portfolio_statistics": {
                "return_period": "complete_utc_day_within_one_oos_window",
                "annualization_periods": 365,
                "annualization_minimum_observations": 2,
                "standard_deviation_ddof": 1,
                "benchmark": "zero_return_cash",
                "gap_policy": "separate_economic_contribution",
                "cost_sensitivity": "constant_actual_turnover_additive_sensitivity",
            },
            "model_scopes": {
                spec.model_id: {
                    "training_asset_ids": list(_hft_asset_scope(spec.model_id, 0)),
                    "validation_asset_ids": list(_hft_asset_scope(spec.model_id, 0)),
                    "evaluation_asset_ids": list(_hft_asset_scope(spec.model_id, 1)),
                }
                for spec in build_hft_model_catalog(training_row_bound=None).list()
            },
        }

        if self.research_plan is not None:
            payload = self.research_plan.payload()
            result.update(
                research_plan=payload,
                development_window={
                    "start": payload["folds"][0]["development_test"]["start"],
                    "end": payload["folds"][-1]["development_test"]["end"],
                },
                holdout_window=payload["final_holdout"],
                label_horizon_ms=HFT_MODEL_HORIZON_MS,
                label_availability_rule="decision_time_plus_horizon_lte_role_end",
                holdout_status="HOLDOUT_LOCKED",
            )
        return result

    def model_ids_for_target(self, target_id):
        if target_id != f"{self.scenario_id}_models":
            raise ValueError("crypto_model_target_invalid")
        return tuple(
            spec.model_id
            for spec in build_hft_model_catalog(training_row_bound=None).list()
        )

    def portfolio_candidate_for_target(self, target_id):
        return next(
            candidate
            for candidate in portfolio_candidates()
            if candidate.target_id == target_id
        )

    def portfolio_candidates(self):
        return portfolio_candidates()

    def statistics_table_group(self, target_id):
        if target_id == f"{self.scenario_id}_maker_calibration":
            if (
                self.research_plan is None
                or self.research_plan.execution is None
                or self.research_plan.execution.maker is None
            ):
                raise PortfolioTargetBlocked("crypto_maker_plan_missing")
            return "crypto_maker_calibration"
        if target_id == f"{self.scenario_id}_portfolio_statistics":
            return "crypto_portfolio"
        raise ValueError("crypto_statistics_target_invalid")

    def portfolio_contract(self, target_id):
        if self.research_plan is None or self.research_plan.execution is None:
            raise PortfolioTargetBlocked("crypto_execution_plan_missing")
        candidate = self.portfolio_candidate_for_target(target_id)
        if (
            isinstance(candidate, CryptoMakerCandidate)
            and self.research_plan.execution.maker is None
        ):
            raise PortfolioTargetBlocked("crypto_maker_plan_missing")
        return {
            "candidate": asdict(candidate),
            "execution": self.research_plan.payload()["execution"],
            "funding_forecast_units": "USDT_cashflow_per_long_asset_unit",
            "window_boundary_order": (
                "events_funding_effective_orders_then_cancel_pending"
            ),
            "gap_policy": "continuous_capital_no_signals_cancel_pending_mark_and_fund",
            **(
                {
                    "quote_horizon_ms": HFT_MODEL_HORIZON_MS,
                    "quote_policy": (
                        "maximize_absolute_expected_gain_of_active_and_pending_schedule"
                        "_keep_or_delayed_queue_reset_replace_or_zero_quantity_cancel"
                        if candidate.calibrated
                        else "keep_same_price_and_remaining_quantity_without_renewal"
                        "_else_delayed_cancel_replace"
                    ),
                    "calibration": (
                        "side_only_paired_validation_event_paths_and_forecast_error"
                    ),
                    "order_type": "post_only_limit",
                }
                if isinstance(candidate, CryptoMakerCandidate)
                else {}
            ),
        }

    def target_specs(self, action: str) -> tuple[TargetSpec, ...]:
        if action != "research run":
            return ()
        dataset = self.scenario_id
        models = f"{dataset}_models"
        predictions = f"{dataset}_predictions"
        statistics = f"{dataset}_portfolio_statistics"
        calibration = f"{dataset}_maker_calibration"
        panel = build_hft_research_dataset_catalog().get(dataset).base_input_data_id
        return (
            *(
                TargetSpec(
                    evaluation.evaluation_id,
                    dependencies=(dataset, predictions)
                    if evaluation.signals[0].source_kind == "prediction"
                    else (dataset,),
                    producer_id=evaluation.evaluation_id,
                )
                for evaluation in build_crypto_evaluation_catalog().list()
            ),
            TargetSpec(models, dependencies=(dataset,), producer_id=models),
            TargetSpec(
                predictions, dependencies=(models, dataset), producer_id=predictions
            ),
            TargetSpec(
                calibration,
                dependencies=(models, dataset, panel),
                producer_id=calibration,
            ),
            *(
                TargetSpec(
                    candidate.target_id,
                    dependencies=(predictions, dataset, panel),
                    producer_id=candidate.target_id,
                )
                for candidate in taker_candidates()
            ),
            TargetSpec(
                maker_candidates()[0].target_id,
                dependencies=(dataset, panel),
                producer_id=maker_candidates()[0].target_id,
            ),
            TargetSpec(
                maker_candidates()[1].target_id,
                dependencies=(calibration, predictions, dataset, panel),
                producer_id=maker_candidates()[1].target_id,
            ),
            TargetSpec(
                statistics,
                dependencies=tuple(
                    candidate.target_id for candidate in portfolio_candidates()
                ),
                producer_id=statistics,
            ),
        )

    def terminal_targets(self, action: str) -> tuple[str, ...]:
        if action == "dataset build":
            return (self.scenario_id,)
        if action == "research run":
            return (
                *(
                    spec.evaluation_id
                    for spec in build_crypto_evaluation_catalog().list()
                ),
                *(candidate.target_id for candidate in portfolio_candidates()),
                f"{self.scenario_id}_portfolio_statistics",
            )
        if action == "data materialize":
            return (
                build_hft_research_dataset_catalog()
                .get(self.scenario_id)
                .input_data_ids
            )
        raise ValueError("unknown_crypto_action")


def execution_windows(plan, output_window):
    return tuple(
        (
            fold["fold_id"],
            TimeWindow(
                max(
                    pd.Timestamp(fold["development_test"]["start"]),
                    pd.Timestamp(output_window.start),
                ),
                min(
                    pd.Timestamp(fold["development_test"]["end"]),
                    pd.Timestamp(output_window.end),
                ),
            ),
        )
        for fold in plan["folds"]
        if pd.Timestamp(fold["development_test"]["end"])
        >= pd.Timestamp(output_window.start)
        and pd.Timestamp(fold["development_test"]["start"])
        <= pd.Timestamp(output_window.end)
    )


def protocol(
    scenario_id: str = HFT_RESEARCH_DATASET_ID,
    access_window: str = "development",
    research_plan: Mapping[str, object] | None = None,
) -> CryptoProtocol:
    return CryptoProtocol(
        scenario_id,
        access_window,
        None if research_plan is None else _research_plan(research_plan),
    )


def build_signal_specs(*, columns: tuple[str, ...]) -> tuple[SignalSpec, ...]:
    return tuple(
        SignalSpec(
            signal_id=column,
            source_kind="factor",
            source_column=column,
            direction="lower_is_better"
            if column in {"spread", "relative_spread"}
            else "higher_is_better",
        )
        for column in columns
    )


def build_hft_model_catalog(*, training_row_bound: int | None) -> ModelCatalog:
    if training_row_bound is not None and (
        type(training_row_bound) is not int or training_row_bound <= 0
    ):
        raise ValueError("hft_training_row_bound_invalid")
    price_models = tuple(
        ModelSpec(
            model_id=f"crypto_hft_{algorithm}",
            model_algorithm=algorithm,
            input_factor_columns=("ofi_100ms",)
            if algorithm == "linear_regression"
            else HFT_FACTOR_COLUMNS,
            label_column=HFT_MODEL_LABEL_COLUMN,
            random_seed=ROOT_SEED,
            max_train_rows=training_row_bound,
            hyperparameters={"n_jobs": -1}
            if algorithm in {"xgboost_regressor", "lightgbm_regressor"}
            else {},
        )
        for algorithm in HFT_MODEL_ALGORITHMS[:3]
    )
    return ModelCatalog(
        (
            *price_models,
            ModelSpec(
                model_id="crypto_hft_maker_bid_sell_pressure_proxy",
                model_algorithm="gradient_boosting_regressor",
                input_factor_columns=HFT_FACTOR_COLUMNS,
                label_column="maker_bid_sell_pressure_proxy_500ms",
                random_seed=ROOT_SEED,
                max_train_rows=training_row_bound,
            ),
            *(
                ModelSpec(
                    model_id=f"crypto_hft_{mode}",
                    model_algorithm="ridge_regression",
                    input_factor_columns=HFT_FACTOR_COLUMNS,
                    label_column=HFT_MODEL_LABEL_COLUMN,
                    random_seed=ROOT_SEED,
                    max_train_rows=training_row_bound,
                    hyperparameters={"alpha": 1.0},
                )
                for mode in HFT_MULTI_ASSET_SCOPES
            ),
        )
    )


def build_hft_prediction_catalog(
    *, train_end: str, test_start: str, test_end: str
) -> PredictionCatalog:
    model_ids = (
        *(f"crypto_hft_{algorithm}" for algorithm in HFT_PRICE_MODEL_ALGORITHMS),
        HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
        *(f"crypto_hft_{mode}" for mode in HFT_MULTI_ASSET_SCOPES),
    )
    return PredictionCatalog(
        tuple(
            PredictionSpec(
                prediction_id=f"{model_id}_prediction",
                model_id=model_id,
                research_dataset_id=HFT_RESEARCH_DATASET_ID,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                training_asset_ids=_hft_asset_scope(model_id, 0),
                evaluation_asset_ids=_hft_asset_scope(model_id, 1),
            )
            for model_id in model_ids
        )
    )


def _hft_asset_scope(model_id: str, position: int) -> tuple[str, ...]:
    mode = model_id.removeprefix("crypto_hft_")
    return (
        HFT_MULTI_ASSET_SCOPES[mode][position] if mode in HFT_MULTI_ASSET_SCOPES else ()
    )


def build_hft_evaluation_catalog() -> EvaluationCatalog:
    return EvaluationCatalog(
        (
            EvaluationSpec(
                evaluation_id=HFT_EVALUATION_ID,
                market_id=MARKET_ID,
                research_dataset_id=HFT_RESEARCH_DATASET_ID,
                signals=build_signal_specs(columns=HFT_FACTOR_COLUMNS),
                label_columns=HFT_LABEL_COLUMNS,
            ),
        )
    )


def build_crypto_evaluation_catalog() -> EvaluationCatalog:
    factor = replace(
        build_hft_evaluation_catalog().get(HFT_EVALUATION_ID),
        evaluation_id=f"{HFT_RESEARCH_DATASET_ID}_evaluation",
        label_availability_columns={
            label: f"{label}_available_at" for label in HFT_LABEL_COLUMNS
        },
    )
    return EvaluationCatalog(
        (
            factor,
            *(
                EvaluationSpec(
                    evaluation_id=f"{model.model_id}_oos_evaluation",
                    market_id=MARKET_ID,
                    research_dataset_id=HFT_RESEARCH_DATASET_ID,
                    signals=(
                        SignalSpec(
                            signal_id=model.model_id,
                            source_kind="prediction",
                            source_column="prediction_score",
                        ),
                    ),
                    label_columns=(model.label_column,),
                    label_availability_columns={
                        model.label_column: f"{model.label_column}_available_at"
                    },
                    evaluation_axis="single_asset_time_series"
                    if len(_hft_asset_scope(model.model_id, 1)) == 1
                    else "cross_sectional",
                )
                for model in build_hft_model_catalog(training_row_bound=None).list()
            ),
        )
    )


def build_hft_research_dataset_catalog() -> ResearchDatasetCatalog:
    context = (
        *HFT_LABEL_AVAILABILITY_COLUMNS,
        "segment_id",
        "book_segment_id",
        "book_connection_id",
        "trade_connection_id",
        "book_capture_seq",
        "trade_capture_seq_min",
        "trade_capture_seq_max",
        "book_observed_time_ms",
        "exchange_event_time_ms",
        "transaction_time_ms",
        "local_receive_time",
        "book_receive_monotonic_ns",
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "mid",
        "bid_depth_1",
        "ask_depth_1",
        "bid_depth_5",
        "ask_depth_5",
        "bid_depth_10",
        "ask_depth_10",
        "book_ofi_100ms",
        "trade_count_100ms",
        "trade_volume_100ms",
        "observed_signed_trade_volume_100ms",
        "observed_taker_buy_volume_100ms",
        "observed_taker_sell_volume_100ms",
    )
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        *HFT_FACTOR_COLUMNS,
        *HFT_LABEL_COLUMNS,
        "eligible",
        *context,
    )
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id=HFT_RESEARCH_DATASET_ID,
                market_id=MARKET_ID,
                input_data_ids=(
                    "binance_futures_l2_panel_100ms",
                    "binance_futures_microstructure_factors_100ms",
                    "binance_futures_mid_labels_100ms",
                ),
                base_input_data_id="binance_futures_l2_panel_100ms",
                factor_columns=HFT_FACTOR_COLUMNS,
                label_columns=HFT_LABEL_COLUMNS,
                filter_columns=("eligible",),
                context_columns=context,
                frequency=DataFrequency.TICK,
                grain=DataGrain.RESEARCH_ROW,
                column_roles=_roles(
                    columns, HFT_FACTOR_COLUMNS, HFT_LABEL_COLUMNS, ("eligible",)
                ),
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=("year",),
            ),
        )
    )


def _roles(
    columns: tuple[str, ...],
    factors: tuple[str, ...],
    labels: tuple[str, ...],
    filters: tuple[str, ...],
) -> dict[str, ColumnRole]:
    return {
        column: ColumnRole.FACTOR
        if column in factors
        else ColumnRole.LABEL
        if column in labels
        else ColumnRole.UNIVERSE
        if column in filters
        else ColumnRole.TIMESTAMP
        if column == "timestamp"
        else ColumnRole.KEY
        if column in {"market_id", "asset_id"}
        else ColumnRole.CONTEXT
        for column in columns
    }
