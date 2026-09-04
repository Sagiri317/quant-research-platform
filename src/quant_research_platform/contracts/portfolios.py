"""Common portfolio contracts and market-policy bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    strict_finite_float,
)

REBALANCE_SCHEDULES = {
    "daily",
    "weekly-last-trading-day",
    "monthly-last-trading-day",
}
PORTFOLIO_CASH_ASSET_ID = "__cash__"
PORTFOLIO_DECISION_REASON_SIMPLE = "simple_rule"
PORTFOLIO_DECISION_REASON_ENHANCED = "enhanced_optimizer"
PORTFOLIO_DECISION_REASON_BENCHMARK = "enhanced_benchmark_only"
PORTFOLIO_DECISION_REASON_MAINTENANCE = "benchmark_maintenance"
PORTFOLIO_BLOCK_REASONS = frozenset(
    {
        "market_blocked",
        "adv_capacity",
        "cash_pro_rata",
        "adv_capacity+cash_pro_rata",
    }
)
PORTFOLIO_DECISION_REASONS = frozenset(
    {
        PORTFOLIO_DECISION_REASON_SIMPLE,
        PORTFOLIO_DECISION_REASON_ENHANCED,
        PORTFOLIO_DECISION_REASON_BENCHMARK,
        PORTFOLIO_DECISION_REASON_MAINTENANCE,
    }
)
EXECUTION_RETURN_COLUMN = "forward_return_1d"
PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "row_count",
    "positive_amount_row_count",
    "excluded_non_positive_amount_count",
    "gross_weight",
    "covered_gross_weight",
    "weighted_amount_proxy",
    "weighted_float_market_cap_proxy",
)
PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "exposure_column",
    "exposure_value",
    "row_count",
    "gross_weight",
)
PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "exposure_column",
    "exposure_value",
    "asset_count",
    "gross_weight",
    "net_weight",
    "long_weight",
    "short_weight",
)
PORTFOLIO_LEDGER_KEY = ("decision_timestamp", "asset_id")
PORTFOLIO_LEDGER_COST_COLUMNS = (
    "commission_cost",
    "stamp_duty_cost",
    "transfer_fee_cost",
    "slippage_cost",
    "impact_cost",
)
PORTFOLIO_LEDGER_COLUMNS = (
    "decision_timestamp",
    "execution_date",
    "asset_id",
    "pre_trade_notional",
    "target_notional",
    "executed_delta_notional",
    "post_trade_notional",
    "holding_return",
    "end_notional",
    *PORTFOLIO_LEDGER_COST_COLUMNS,
    "block_reason",
    "adv20",
    "alpha_score",
    "expected_alpha",
    "benchmark_weight",
    "asset_specific_variance",
    "decision_reason",
    "alpha_slope",
    "alpha_slope_se",
    "alpha_slope_t",
    "alpha_slope_pvalue",
    "alpha_slope_ci_low",
    "alpha_slope_ci_high",
    "alpha_weighted_r2",
    "alpha_coverage",
    "alpha_valid_dates",
    "alpha_scheduled_dates",
    "predicted_tracking_error",
    "portfolio_factor_variance",
    "portfolio_specific_variance",
    "expected_transaction_cost",
)


class PortfolioTargetBlocked(Exception):
    """A portfolio target cannot be built from currently available facts."""


@dataclass(frozen=True, slots=True)
class PortfolioCandidateSpec:
    """One protocol-owned portfolio candidate composition."""

    candidate_id: str
    construction: Literal[
        "top20_equal_weight",
        "top20_float_cap",
        "benchmark_enhanced",
    ]
    prediction_target_id: str
    rebalance_schedule: str
    top_quantile: float | None
    benchmark_id: str | None
    benchmark_weights_target_id: str | None
    research_dataset_id: str
    market_panel_target_id: str
    execution_mask_target_id: str
    trading_calendar_target_id: str
    fee_schedule_target_id: str
    risk_style_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (
            self.candidate_id,
            self.prediction_target_id,
            self.research_dataset_id,
            self.market_panel_target_id,
            self.execution_mask_target_id,
            self.trading_calendar_target_id,
            self.fee_schedule_target_id,
        ):
            if type(value) is not str or not value:
                raise ValueError("portfolio_candidate_dependency_invalid")
        if type(self.construction) is not str or self.construction not in {
            "top20_equal_weight",
            "top20_float_cap",
            "benchmark_enhanced",
        }:
            raise ValueError("portfolio_candidate_construction_invalid")
        if (
            type(self.rebalance_schedule) is not str
            or self.rebalance_schedule not in REBALANCE_SCHEDULES
        ):
            raise ValueError("portfolio_candidate_schedule_invalid")
        if self.benchmark_id is not None and (
            type(self.benchmark_id) is not str or not self.benchmark_id
        ):
            raise ValueError("portfolio_candidate_benchmark_invalid")
        if self.benchmark_weights_target_id is not None and (
            type(self.benchmark_weights_target_id) is not str
            or not self.benchmark_weights_target_id
        ):
            raise ValueError("portfolio_candidate_benchmark_dependency_invalid")
        if (self.benchmark_id is None) != (self.benchmark_weights_target_id is None):
            raise ValueError("portfolio_candidate_benchmark_dependency_invalid")
        if self.construction == "benchmark_enhanced":
            if self.top_quantile is not None or self.benchmark_id is None:
                raise ValueError("portfolio_candidate_construction_invalid")
            if (
                type(self.risk_style_columns) is not tuple
                or not self.risk_style_columns
                or not all(
                    type(column) is str and column for column in self.risk_style_columns
                )
                or len(set(self.risk_style_columns)) != len(self.risk_style_columns)
            ):
                raise ValueError("portfolio_candidate_risk_styles_invalid")
        else:
            if self.top_quantile is None or self.benchmark_id is not None:
                raise ValueError("portfolio_candidate_construction_invalid")
            if self.risk_style_columns:
                raise ValueError("portfolio_candidate_risk_styles_invalid")
            if isinstance(self.top_quantile, bool) or not isinstance(
                self.top_quantile, int | float
            ):
                raise ValueError("portfolio_candidate_quantile_invalid")
            quantile = strict_finite_float(
                self.top_quantile,
                error="portfolio_candidate_quantile_invalid",
            )
            if not 0.0 < quantile <= 1.0:
                raise ValueError("portfolio_candidate_quantile_invalid")
            object.__setattr__(self, "top_quantile", quantile)


@dataclass(frozen=True, slots=True)
class ExecutionAssumptions:
    """Protocol-owned continuous execution parameters.

    Statutory commission, stamp duty, and transfer fees are supplied by the
    market fee schedule; they are deliberately not duplicated here.
    """

    initial_nav: float
    slippage_bps: float
    impact_bps: float
    adv_participation: float
    adv_sensitivities: tuple[float, ...]

    def __post_init__(self) -> None:
        scalar_values = (
            self.initial_nav,
            self.slippage_bps,
            self.impact_bps,
            self.adv_participation,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in scalar_values
        ):
            raise ValueError("execution_assumptions_invalid")
        numeric_values = tuple(
            strict_finite_float(value, error="execution_assumptions_invalid")
            for value in scalar_values
        )
        initial_nav, slippage_bps, impact_bps, participation = numeric_values
        if initial_nav <= 0.0 or slippage_bps < 0.0 or impact_bps < 0.0:
            raise ValueError("execution_assumptions_invalid")
        if participation <= 0.0 or participation > 1.0:
            raise ValueError("execution_assumptions_invalid")
        if (
            type(self.adv_sensitivities) is not tuple
            or not self.adv_sensitivities
            or any(
                isinstance(value, bool) or not isinstance(value, int | float)
                for value in self.adv_sensitivities
            )
        ):
            raise ValueError("execution_assumptions_invalid")
        sensitivities = tuple(
            strict_finite_float(
                value,
                error="execution_assumptions_invalid",
            )
            for value in self.adv_sensitivities
        )
        if any(value <= 0.0 or value > 1.0 for value in sensitivities):
            raise ValueError("execution_assumptions_invalid")
        object.__setattr__(self, "initial_nav", initial_nav)
        object.__setattr__(self, "slippage_bps", slippage_bps)
        object.__setattr__(self, "impact_bps", impact_bps)
        object.__setattr__(self, "adv_participation", participation)
        object.__setattr__(self, "adv_sensitivities", sensitivities)


@dataclass(frozen=True, slots=True)
class TradeCostComponents:
    """Immutable cost value returned by a market cost owner."""

    notional: float
    commission_cost: float
    stamp_duty_cost: float
    transfer_fee_cost: float
    slippage_cost: float
    impact_cost: float
    source_identity: str

    def __post_init__(self) -> None:
        numeric = (
            self.notional,
            self.commission_cost,
            self.stamp_duty_cost,
            self.transfer_fee_cost,
            self.slippage_cost,
            self.impact_cost,
        )
        if any(
            isinstance(value, bool)
            or type(value) is not float
            or not strict_finite_float(value, error="trade_cost_components_invalid")
            >= 0.0
            for value in numeric
        ):
            raise ValueError("trade_cost_components_invalid")
        if (
            type(self.source_identity) is not str
            or len(self.source_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_identity
            )
        ):
            raise ValueError("trade_cost_components_invalid")

    @property
    def total_cost(self) -> float:
        return float(
            self.commission_cost
            + self.stamp_duty_cost
            + self.transfer_fee_cost
            + self.slippage_cost
            + self.impact_cost
        )


@dataclass(frozen=True)
class PortfolioDiagnosticsConfig:
    """Portfolio diagnostics column configuration."""

    capacity_amount_column: str | None = None
    capacity_float_market_cap_column: str | None = None
    numeric_exposure_columns: tuple[str, ...] = ()
    categorical_exposure_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_column_name(
            self.capacity_amount_column,
            "capacity_amount_column",
        )
        _validate_optional_column_name(
            self.capacity_float_market_cap_column,
            "capacity_float_market_cap_column",
        )
        _validate_column_tuple(
            self.numeric_exposure_columns,
            "numeric_exposure_columns",
            required=False,
        )
        _validate_column_tuple(
            self.categorical_exposure_columns,
            "categorical_exposure_columns",
            required=False,
        )
        if (
            self.capacity_float_market_cap_column is not None
            and self.capacity_amount_column is None
        ):
            raise ValueError("capacity_amount_column_required")
        overlap = sorted(
            set(self.numeric_exposure_columns).intersection(
                self.categorical_exposure_columns
            )
        )
        if overlap:
            raise ValueError(
                "diagnostic_exposure_columns_must_not_overlap: " + ", ".join(overlap)
            )


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")
