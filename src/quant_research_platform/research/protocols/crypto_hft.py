"""Crypto HFT protocol constants and catalogs; market facts stay in crypto."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

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
    "return_500ms",
    "realized_volatility_1s",
    "top10_depth_per_spread",
    "trade_flow_imbalance_100ms",
    "trade_flow_imbalance_500ms",
    "trade_flow_imbalance_1s",
    "trade_flow_imbalance_5s",
    "ofi_500ms",
    "ofi_1s",
    "ofi_5s",
    "depth_normalized_ofi_500ms",
    "depth_normalized_ofi_1s",
    "depth_normalized_ofi_5s",
    "imbalance_1_minus_10",
    "imbalance_5_minus_10",
    "near_depth_share_1_10",
    "taker_sell_pressure_500ms",
    "taker_buy_pressure_500ms",
)
HFT_DERIVATIVE_FACTOR_COLUMNS = (
    "mark_mid_basis",
    "mark_index_basis",
    "perp_index_basis",
    "funding_rate",
    "time_to_next_funding",
    "log_open_interest",
    "open_interest_change_30s",
    "open_interest_change_1m",
    "open_interest_change_5m",
    "observed_force_order_imbalance_5s",
    "observed_force_order_imbalance_30s",
    "time_since_observed_force_order",
    "top_position_long_short_ratio",
    "top_account_long_short_ratio",
    "global_account_long_short_ratio",
    "taker_long_short_ratio",
)
HFT_DERIVATIVE_CHANNELS = (
    "premiumIndex",
    "openInterest",
    "forceOrder",
    "topLongShortPositionRatio",
    "topLongShortAccountRatio",
    "globalLongShortAccountRatio",
    "takerlongshortRatio",
)
HFT_DERIVATIVE_CONTEXT_COLUMNS = tuple(
    f"{channel}_{field}"
    for channel in HFT_DERIVATIVE_CHANNELS
    for field in ("age_ms", "observed_at", "event_at")
)
HFT_DERIVATIVE_DATA_ID = "binance_futures_derivative_factors_100ms"
HFT_SPOT_PANEL_DATA_ID = "binance_spot_l2_panel_100ms"
HFT_SPOT_CONTEXT_DATA_ID = "binance_spot_perp_context_100ms"
HFT_SPOT_FACTOR_COLUMNS = (
    "spot_perp_basis",
    "spot_return_100ms",
    "spot_return_500ms",
    "spot_ofi_500ms",
    "spot_tfi_500ms",
    "spot_perp_return_divergence_100ms",
    "spot_perp_return_divergence_500ms",
    "spot_perp_normalized_ofi_divergence_500ms",
    "spot_perp_tfi_divergence_500ms",
    "lagged_spot_return_100ms",
    "lagged_spot_ofi_500ms",
    "lagged_spot_tfi_500ms",
)
HFT_SPOT_CONTEXT_COLUMNS = (
    "spot_grid_age_ms",
    "spot_book_age_ms",
    "spot_grid_at",
    "spot_book_observed_at",
)
HFT_SOL_PANEL_DATA_ID = "binance_sol_futures_l2_panel_100ms"
HFT_INFORMATION_PANEL_DATA_ID = "binance_perpetual_information_panel_100ms"
HFT_INFORMATION_STATE_DATA_ID = "binance_perpetual_information_state_100ms"
HFT_CROSS_ASSET_DATA_ID = "binance_cross_asset_context_100ms"
HFT_INFORMATION_COLUMNS = (
    "return_100ms",
    "return_500ms",
    "depth_normalized_ofi_500ms",
    "trade_flow_imbalance_500ms",
)
HFT_CROSS_ASSET_FACTOR_COLUMNS = tuple(
    f"{asset}_{field}"
    for asset in ("btc", "eth", "sol")
    for field in (
        "return_100ms",
        "return_500ms",
        "normalized_ofi_500ms",
        "tfi_500ms",
        "lagged_return_100ms",
        "lagged_tfi_500ms",
    )
)
HFT_CROSS_ASSET_CONTEXT_COLUMNS = tuple(
    f"{asset}_{field}"
    for asset in ("btc", "eth", "sol")
    for field in ("grid_age_ms", "grid_at")
)

HFT_EXTERNAL_DATA_IDS = {
    "bybit": "bybit_perpetual_context_100ms",
    "okx": "okx_perpetual_context_100ms",
}
HFT_EXTERNAL_FIELDS = (
    "mid_usdt",
    "spread_usdt",
    "return_100ms",
    "return_500ms",
    "observed_signed_base_flow_500ms",
    "observed_signed_notional_flow_500ms",
    "observed_tfi_500ms",
    "price_difference",
    "relative_mid_dispersion",
)
HFT_EXTERNAL_CLOCK_FIELDS = tuple(
    f"{source}_{field}"
    for source in ("quote", "trade", "ticker")
    for field in ("age_ms", "observed_at", "event_at")
)
HFT_LABEL_HORIZONS_MS = (100, 500, 1_000, 5_000)
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
HFT_MAKER_QUOTE_HORIZON_MS = 500
HFT_EXECUTION_ASSET_IDS = ("btc_usdt", "eth_usdt")

HFT_EXECUTION_CONTEXT_COLUMNS = (
    "book_segment_id",
    "trade_observation_id",
    "segment_id",
)


def label_horizon_ms(label_column):
    """Return the duration encoded by an existing executable Crypto label."""
    if label_column not in HFT_LABEL_COLUMNS:
        raise ValueError("crypto_label_unknown")
    return int(label_column.rsplit("_", 1)[1].removesuffix("ms"))


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
            or not self.latency_scenarios_ms
            or any(value < 0 for value in self.latency_scenarios_ms)
            or len(set(self.latency_scenarios_ms)) != len(self.latency_scenarios_ms)
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
    price_model_id: str | None
    proxy_model_id: str | None

    @property
    def calibrated(self):
        return self.price_model_id is not None

    @property
    def model_ids(self):
        return (self.price_model_id, self.proxy_model_id) if self.calibrated else ()

    @property
    def target_id(self):
        return f"{self.candidate_id}_ledger"

    @property
    def calibration_target_id(self):
        return f"{self.candidate_id}_calibration"


def portfolio_candidates(selection):
    """Decode explicit execution candidates without a named strategy roster."""
    if type(selection) is not list:
        raise ValueError("research_portfolio_configuration_invalid")
    result = []
    for item in selection:
        if type(item) is not dict:
            raise ValueError("research_portfolio_configuration_invalid")
        if set(item) == {"candidate_id", "asset_id", "model_id", "latency_ms"}:
            if type(item["model_id"]) is not str or not item["model_id"]:
                raise ValueError("research_portfolio_model_invalid")
            if item["latency_ms"] is not None and (
                type(item["latency_ms"]) is not int or item["latency_ms"] < 0
            ):
                raise ValueError("crypto_execution_parameters_invalid")
            candidate = CryptoTakerCandidate(**item)
        elif set(item) == {
            "candidate_id",
            "asset_id",
            "price_model_id",
            "proxy_model_id",
        }:
            roles = (item["price_model_id"], item["proxy_model_id"])
            if roles != (None, None) and any(
                type(value) is not str or not value for value in roles
            ):
                raise ValueError("research_portfolio_model_invalid")
            candidate = CryptoMakerCandidate(**item)
        else:
            raise ValueError("research_portfolio_configuration_invalid")
        if (
            type(candidate.candidate_id) is not str
            or not candidate.candidate_id
            or type(candidate.asset_id) is not str
            or candidate.asset_id not in HFT_EXECUTION_ASSET_IDS
        ):
            raise ValueError("research_portfolio_configuration_invalid")
        result.append(candidate)
    return tuple(result)


def maker_model_ids(selection):
    """Derive validation persistence roles from actual maker consumers."""
    return frozenset(
        model_id
        for candidate in portfolio_candidates(selection)
        if isinstance(candidate, CryptoMakerCandidate)
        for model_id in candidate.model_ids
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
    if type(raw_limits) is not dict or set(raw_limits) != set(HFT_EXECUTION_ASSET_IDS):
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
    configuration_json: str | None = None

    def __post_init__(self) -> None:
        if self.scenario_id != HFT_RESEARCH_DATASET_ID:
            raise ValueError("unknown_crypto_scenario")
        if self.research_plan is not None and not isinstance(
            self.research_plan, _CryptoResearchPlan
        ):
            raise TypeError("crypto_research_plan_invalid")
        if self.access_window not in {"development", "holdout"}:
            raise ValueError("unknown_crypto_access_window")

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

    def manifest(self) -> dict[str, object]:
        result = {
            **(
                {"research_configuration": self.configuration()}
                if self.configuration_json is not None
                else {}
            ),
            "protocol_id": self.protocol_id,
            "scenario_id": self.scenario_id,
            "access_window": self.access_window,
            "required_terminals": list(self.terminal_targets("research run")),
            "portfolio_candidates": [
                asdict(candidate) for candidate in self.portfolio_candidates()
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
                item["model_id"]: {
                    "training_asset_ids": item["training_asset_ids"],
                    "validation_asset_ids": item["training_asset_ids"],
                    "evaluation_asset_ids": item["evaluation_asset_ids"],
                    "label_column": item["label_column"],
                    "label_horizon_ms": label_horizon_ms(item["label_column"]),
                }
                for item in (
                    self.configuration()["models"]
                    if self.configuration_json is not None
                    else ()
                )
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
                maker_quote_horizon_ms=HFT_MAKER_QUOTE_HORIZON_MS,
                label_availability_rule="decision_time_plus_horizon_lte_role_end",
                holdout_status="HOLDOUT_LOCKED",
            )
        return result

    def model_ids_for_target(self, target_id):
        if target_id != f"{self.scenario_id}_models":
            raise ValueError("crypto_model_target_invalid")
        if self.configuration_json is not None:
            return tuple(model["model_id"] for model in self.configuration()["models"])
        return tuple(
            spec.model_id
            for spec in build_hft_model_catalog(
                configuration=self.configuration()
            ).list()
        )

    def portfolio_candidate_for_target(self, target_id):
        return next(
            candidate
            for candidate in self.portfolio_candidates()
            if candidate.target_id == target_id
        )

    def portfolio_candidates(self):
        return portfolio_candidates(
            self.configuration()["portfolios"]
            if self.configuration_json is not None
            else []
        )

    def statistics_table_group(self, target_id):
        if any(
            isinstance(candidate, CryptoMakerCandidate)
            and candidate.calibrated
            and target_id == candidate.calibration_target_id
            for candidate in self.portfolio_candidates()
        ):
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
                    "quote_horizon_ms": HFT_MAKER_QUOTE_HORIZON_MS,
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
        panel = build_hft_research_dataset_catalog().get(dataset).base_input_data_id
        return (
            *(
                TargetSpec(
                    evaluation.evaluation_id,
                    dependencies=(
                        dataset,
                        predictions,
                        *(
                            tuple(
                                signal.signal_id + "_oos_evaluation"
                                for signal in evaluation.signals
                            )
                            if len(evaluation.signals) > 1
                            else ()
                        ),
                    )
                    if evaluation.signals[0].source_kind == "prediction"
                    else (dataset,),
                    producer_id=evaluation.evaluation_id,
                )
                for evaluation in build_crypto_evaluation_catalog(
                    configuration=self.configuration()
                ).list()
            ),
            TargetSpec(models, dependencies=(dataset,), producer_id=models),
            TargetSpec(
                predictions, dependencies=(models, dataset), producer_id=predictions
            ),
            *(
                TargetSpec(
                    candidate.calibration_target_id,
                    dependencies=(models, dataset, panel),
                    producer_id=candidate.calibration_target_id,
                )
                for candidate in self.portfolio_candidates()
                if isinstance(candidate, CryptoMakerCandidate) and candidate.calibrated
            ),
            *(
                TargetSpec(
                    candidate.target_id,
                    dependencies=(
                        (candidate.calibration_target_id, predictions, dataset, panel)
                        if isinstance(candidate, CryptoMakerCandidate)
                        and candidate.calibrated
                        else (dataset, panel)
                        if isinstance(candidate, CryptoMakerCandidate)
                        else (predictions, dataset, panel)
                    ),
                    producer_id=candidate.target_id,
                )
                for candidate in self.portfolio_candidates()
            ),
            TargetSpec(
                statistics,
                dependencies=tuple(
                    candidate.target_id for candidate in self.portfolio_candidates()
                ),
                producer_id=statistics,
            ),
        )

    def terminal_targets(self, action: str) -> tuple[str, ...]:
        if action == "dataset build":
            return (self.scenario_id,)
        if action == "research run":
            if (
                self.configuration_json is not None
                and self.configuration()["stage"] == "evaluation"
            ):
                return (f"{self.scenario_id}_evaluation",)
            if self.configuration_json is not None:
                selected = self.configuration()
                return (
                    f"{self.scenario_id}_evaluation",
                    *(
                        model["model_id"] + "_oos_evaluation"
                        for model in selected["models"]
                    ),
                    *(
                        spec.evaluation_id
                        for spec in build_crypto_evaluation_catalog(
                            configuration=selected
                        ).list()
                        if len(spec.signals) > 1
                        and spec.signals[0].source_kind == "prediction"
                    ),
                    *(candidate.target_id for candidate in self.portfolio_candidates()),
                    *(
                        (f"{self.scenario_id}_portfolio_statistics",)
                        if selected["portfolios"]
                        else ()
                    ),
                )
            return (
                *(
                    spec.evaluation_id
                    for spec in build_crypto_evaluation_catalog(
                        configuration=self.configuration()
                    ).list()
                ),
                *(candidate.target_id for candidate in self.portfolio_candidates()),
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


def build_hft_model_catalog(*, configuration=None) -> ModelCatalog:
    return ModelCatalog.from_configuration(
        [] if configuration is None else configuration["models"]
    )


def build_hft_prediction_catalog(
    *, configuration, train_end: str, test_start: str, test_end: str
) -> PredictionCatalog:
    return PredictionCatalog(
        PredictionSpec(
            prediction_id=f"{item['model_id']}_prediction",
            model_id=item["model_id"],
            research_dataset_id=HFT_RESEARCH_DATASET_ID,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            training_asset_ids=tuple(item["training_asset_ids"]),
            evaluation_asset_ids=tuple(item["evaluation_asset_ids"]),
        )
        for item in configuration["models"]
    )


def build_crypto_evaluation_catalog(*, configuration=None) -> EvaluationCatalog:
    factor = EvaluationSpec(
        evaluation_id=f"{HFT_RESEARCH_DATASET_ID}_evaluation",
        market_id=MARKET_ID,
        research_dataset_id=HFT_RESEARCH_DATASET_ID,
        signals=build_signal_specs(columns=HFT_FACTOR_COLUMNS),
        label_columns=HFT_LABEL_COLUMNS,
        label_availability_columns={
            label: f"{label}_available_at" for label in HFT_LABEL_COLUMNS
        },
    )
    models = build_hft_model_catalog(configuration=configuration).list()
    comparisons = []
    if configuration is not None and configuration["stage"] == "research":
        reference = next(
            model
            for model in models
            if model.model_id == configuration["comparison_model_id"]
        )
        scopes = {
            item["model_id"]: set(item["evaluation_asset_ids"])
            for item in configuration["models"]
        }
        for model in models:
            if (
                model.model_id == reference.model_id
                or model.label_column != reference.label_column
            ):
                continue
            assets = scopes[model.model_id] & scopes[reference.model_id]
            comparisons.append(
                EvaluationSpec(
                    evaluation_id=f"{model.model_id}_vs_{reference.model_id}_common_oos",
                    market_id=MARKET_ID,
                    research_dataset_id=HFT_RESEARCH_DATASET_ID,
                    signals=tuple(
                        SignalSpec(
                            signal_id=item.model_id,
                            source_kind="prediction",
                            source_column="prediction__" + item.model_id,
                        )
                        for item in (reference, model)
                    ),
                    label_columns=(model.label_column,),
                    label_availability_columns={
                        model.label_column: model.label_column + "_available_at",
                    },
                    evaluation_axis=(
                        "single_asset_time_series"
                        if len(assets) <= 1
                        else "cross_sectional"
                    ),
                )
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
                    if len(
                        next(
                            item["evaluation_asset_ids"]
                            for item in configuration["models"]
                            if item["model_id"] == model.model_id
                        )
                    )
                    == 1
                    else "cross_sectional",
                )
                for model in models
            ),
            *comparisons,
        )
    )


def build_hft_research_dataset_catalog() -> ResearchDatasetCatalog:
    context = (
        *HFT_LABEL_AVAILABILITY_COLUMNS,
        "segment_id",
        "book_segment_id",
        "book_observation_id",
        "trade_observation_id",
        "book_source_sequence",
        "trade_source_sequence_min",
        "trade_source_sequence_max",
        "book_observed_time_ms",
        "exchange_event_time_ms",
        "transaction_time_ms",
        "observed_utc",
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


def factor_catalog():
    from quant_research_platform.catalog.factors import FactorCatalog
    from quant_research_platform.contracts.factors import FactorDefinition, FactorSet

    definitions = [
        FactorDefinition(
            factor_id=column,
            market_id=MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=(
                "mid",
                "book_spread",
                "bid_depth_1",
                "ask_depth_1",
                "bid_depth_5",
                "ask_depth_5",
                "bid_depth_10",
                "ask_depth_10",
                "book_ofi_100ms",
                "observed_signed_trade_volume_100ms",
                "observed_taker_buy_volume_100ms",
                "observed_taker_sell_volume_100ms",
            ),
            output_column=column,
            family="microstructure",
            required_input_data_id="binance_futures_microstructure_factors_100ms",
            timing_semantics="observed_by_decision_on_contiguous_100ms_segment",
            null_policy="full_window_positive_denominator",
        )
        for column in HFT_FACTOR_COLUMNS
    ]
    definitions.extend(
        FactorDefinition(
            factor_id=column,
            market_id=MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=("mid",),
            output_column=column,
            family="derivatives_state",
            required_input_data_id=HFT_DERIVATIVE_DATA_ID,
            timing_semantics="native_observed_at_lte_decision_with_explicit_age_limit",
            null_policy="missing_stale_or_incomplete_window_is_null",
        )
        for column in HFT_DERIVATIVE_FACTOR_COLUMNS
    )
    definitions.extend(
        FactorDefinition(
            factor_id=column,
            market_id=MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=("mid",),
            output_column=column,
            family="spot_perp",
            required_input_data_id=HFT_SPOT_CONTEXT_DATA_ID,
            timing_semantics="spot_grid_available_by_perpetual_decision_with_explicit_age_limit",
            null_policy="missing_stale_or_incomplete_window_is_null",
        )
        for column in HFT_SPOT_FACTOR_COLUMNS
    )
    definitions.extend(
        FactorDefinition(
            factor_id=column,
            market_id=MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=HFT_INFORMATION_COLUMNS,
            output_column=column,
            family="cross_asset",
            required_input_data_id=HFT_CROSS_ASSET_DATA_ID,
            timing_semantics="mature_information_grid_lte_decision_with_explicit_age_limit",
            null_policy="missing_stale_or_incomplete_window_is_null",
        )
        for column in HFT_CROSS_ASSET_FACTOR_COLUMNS
    )
    definitions.extend(
        FactorDefinition(
            factor_id=f"{venue}_{column}",
            market_id=MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=("mid",),
            output_column=f"{venue}_{column}",
            family="external_price_discovery",
            required_input_data_id=data_id,
            timing_semantics="normalized_local_timestamp_lte_binance_decision",
            null_policy="missing_stale_incomplete_or_unrepresentable_is_null",
        )
        for venue, data_id in HFT_EXTERNAL_DATA_IDS.items()
        for column in HFT_EXTERNAL_FIELDS
    )
    return FactorCatalog(definitions)
