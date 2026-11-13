"""Define portfolio capital, execution assumptions and economic ledger events."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Literal

import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_float,
)

A_SHARE_EQUITY_EVENTS_DATA_ID = "a_share_equity_events"
A_SHARE_INSTRUMENT_TYPES = frozenset({"stock", "equity_etf", "gold_etf"})
EQUITY_EVENT_COLUMNS = (
    "source_asset_id",
    "asset_id",
    "listing_date",
    "timestamp",
    "event_id",
    "kind",
    "effective_at",
    "available_at",
    "observed_at",
    "currency",
    "value_per_old_unit",
    "remaining_old_units_per_old_unit",
    "successor_asset_id",
    "successor_listing_date",
    "successor_units_per_old_unit",
    "evidence_identity",
    "source_locator",
)

REBALANCE_SCHEDULES = {
    "daily",
    "weekly-last-trading-day",
    "monthly-last-trading-day",
}
PORTFOLIO_CASH_ASSET_ID = "__cash__"
RESEARCH_VALUATION_POLICY = "evidenced_rights_or_confirmed_delisting_zero"
PORTFOLIO_DECISION_REASON_SIMPLE = "simple_rule"
PORTFOLIO_DECISION_REASON_ENHANCED = "enhanced_optimizer"
PORTFOLIO_DECISION_REASON_BENCHMARK = "enhanced_benchmark_only"
PORTFOLIO_DECISION_REASON_MAINTENANCE = "benchmark_maintenance"
PORTFOLIO_DECISION_REASON_POLICY = "external_policy"
PORTFOLIO_LEDGER_COST_COLUMNS = (
    "commission_cost",
    "stamp_duty_cost",
    "transfer_fee_cost",
    "slippage_cost",
    "impact_cost",
)


class PortfolioTargetBlocked(Exception):
    """A portfolio target cannot be built from currently available facts."""


@dataclass(frozen=True, slots=True)
class PortfolioCapital:
    """Independent research capital; no shared pool or brokerage account."""

    candidate_id: str
    initial_capital: float
    valuation_currency: str
    valuation_method: Literal["cash_equity"]

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or not self.candidate_id
            or type(self.valuation_currency) is not str
            or not self.valuation_currency.isascii()
            or not self.valuation_currency.isalpha()
            or not self.valuation_currency.isupper()
            or self.valuation_method != "cash_equity"
        ):
            raise ValueError("portfolio_capital_invalid")
        value = strict_finite_float(
            self.initial_capital, error="portfolio_capital_invalid"
        )
        if value <= 0.0:
            raise ValueError("portfolio_capital_invalid")
        object.__setattr__(self, "initial_capital", value)


@dataclass(frozen=True, slots=True)
class ExecutedTrade:
    fill_id: str
    order_id: str
    timestamp_ns: int
    asset_id: str
    quantity: float
    notional: float


@dataclass(frozen=True, slots=True)
class PortfolioCashFlow:
    flow_id: str
    timestamp_ns: int
    asset_id: str
    fill_id: str | None
    kind: str
    amount: float


@dataclass(frozen=True, slots=True)
class PortfolioValuation:
    timestamp_ns: int
    asset_id: str
    phase: Literal["before", "after", "end"]
    quantity: float
    value: float
    mark_price: float = 1.0
    lifecycle_id: str | None = None
    receivable: float = 0.0
    locked_quantity: float = 0.0

    @property
    def share_value(self) -> float:
        """Raw-share notional for cash-equity consumers, excluding receivables."""
        return self.quantity * self.mark_price


@dataclass(frozen=True, slots=True)
class PortfolioDiagnostic:
    asset_id: str
    fill_id: str | None
    name: str
    number: float | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioRightsAdjustment:
    """Evidence-linked nontrade change in actual shares and cash entitlements."""

    event_id: str
    timestamp_ns: int
    asset_id: str
    lifecycle_id: str
    quantity_before: float
    quantity_after: float
    cash_amount: float
    target_asset_id: str | None
    target_lifecycle_id: str | None
    target_quantity: float
    evidence_identity: str
    kind: str = "recovery"
    phase: Literal["after", "end"] = "after"
    entitlement_quantity: float = 0.0
    receivable_before: float = 0.0
    receivable_after: float = 0.0
    locked_quantity_before: float = 0.0
    locked_quantity_after: float = 0.0
    entitlement_record_at_ns: int | None = None
    quantity_rounding_loss: float = 0.0
    target_quantity_rounding_loss: float = 0.0


@dataclass(frozen=True, slots=True)
class PortfolioLedgerBatch:
    """One bounded economic transition and its linked market diagnostics."""

    sequence: int
    trades: tuple[ExecutedTrade, ...]
    cashflows: tuple[PortfolioCashFlow, ...]
    valuations: tuple[PortfolioValuation, ...]
    diagnostics: tuple[PortfolioDiagnostic, ...]
    rights_adjustments: tuple[PortfolioRightsAdjustment, ...] = ()


def portfolio_notional_tolerance(nav: float) -> float:
    """Currency tolerance for independently accumulated portfolio amounts."""
    return max(1e-8, abs(nav) * 1e-12)


def _portfolio_budget_exceeded(notionals, budget_terms) -> bool:
    requested, budget = fsum(notionals), fsum(budget_terms)
    return requested - budget > portfolio_notional_tolerance(budget)


def _quantity_equal(left, right):
    return abs(left - right) <= 1e-9 + 1e-12 * abs(right)


@dataclass(frozen=True, slots=True)
class PortfolioCandidateSpec:
    """One protocol-owned portfolio candidate composition."""

    candidate_id: str
    selection_group: str
    construction: Literal[
        "quantile_equal_weight",
        "quantile_equal_weight_half_step",
        "quantile_float_cap",
        "benchmark_enhanced",
    ]
    prediction_selector: str
    rebalance_schedule: str
    top_quantile: float | None
    benchmark_id: str | None
    risk_style_columns: tuple[str, ...] = ()
    neutrality: Literal["none", "industry", "size", "industry_size"] = "none"
    size_cap_column: Literal["total_market_cap", "float_market_cap"] = (
        "total_market_cap"
    )
    industry_neutrality_limit: float = 0.02
    size_neutrality_limit: float = 0.2

    def __post_init__(self) -> None:
        if type(self.selection_group) is not str or not self.selection_group:
            raise ValueError("portfolio_selection_group_invalid")
        for value in (
            self.candidate_id,
            self.prediction_selector,
        ):
            if type(value) is not str or not value:
                raise ValueError("portfolio_candidate_dependency_invalid")
        if type(self.construction) is not str or self.construction not in {
            "quantile_equal_weight",
            "quantile_equal_weight_half_step",
            "quantile_float_cap",
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
        if (
            type(self.risk_style_columns) is not tuple
            or any(
                type(column) is not str or not column
                for column in self.risk_style_columns
            )
            or len(set(self.risk_style_columns)) != len(self.risk_style_columns)
        ):
            raise ValueError("portfolio_candidate_risk_styles_invalid")
        if type(self.neutrality) is not str or self.neutrality not in {
            "none",
            "industry",
            "size",
            "industry_size",
        }:
            raise ValueError("portfolio_candidate_neutrality_invalid")
        if type(self.size_cap_column) is not str or self.size_cap_column not in {
            "total_market_cap",
            "float_market_cap",
        }:
            raise ValueError("portfolio_candidate_size_cap_column_invalid")
        for name in ("industry_neutrality_limit", "size_neutrality_limit"):
            value = strict_finite_float(
                getattr(self, name),
                error="portfolio_candidate_neutrality_limit_invalid",
            )
            if value < 0:
                raise ValueError("portfolio_candidate_neutrality_limit_invalid")
            object.__setattr__(self, name, value)
        if self.construction == "benchmark_enhanced":
            if self.neutrality != "none":
                raise ValueError("portfolio_candidate_neutrality_invalid")
            if self.top_quantile is not None or self.benchmark_id is None:
                raise ValueError("portfolio_candidate_construction_invalid")
            if not self.risk_style_columns:
                raise ValueError("portfolio_candidate_risk_styles_invalid")
        else:
            if self.top_quantile is None or (
                self.benchmark_id is not None and self.neutrality == "none"
            ):
                raise ValueError("portfolio_candidate_construction_invalid")
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
    """Stock execution parameters and explicit broker minimum commission.

    The market fee schedule supplies the explicit account broker commission
    assumption and statutory stamp duty and transfer fees. Only the broker
    minimum commission is set here; exchange fees embedded in commission are
    not charged again.
    """

    initial_nav: float
    slippage_bps: float
    impact_bps: float
    adv_participation: float
    adv_sensitivities: tuple[float, ...]
    minimum_commission: float = 5.0

    def __post_init__(self) -> None:
        scalar_values = (
            self.initial_nav,
            self.slippage_bps,
            self.impact_bps,
            self.adv_participation,
            self.minimum_commission,
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
        initial_nav, slippage_bps, impact_bps, participation, minimum = numeric_values
        if initial_nav <= 0.0 or min(slippage_bps, impact_bps, minimum) < 0.0:
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
        object.__setattr__(self, "minimum_commission", minimum)


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
        if type(self.source_identity) is not str or not self.source_identity:
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


def _portfolio_session_frame(
    frame: pd.DataFrame | None, required_columns: tuple[str, ...]
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=required_columns)
    if type(frame) is not pd.DataFrame or not set(required_columns).issubset(
        frame.columns
    ):
        raise ValueError("portfolio_input_schema_invalid")
    result = frame.loc[:, list(required_columns)].copy()
    if "asset_id" in result:
        result["asset_id"] = result["asset_id"].astype("string")
    if "timestamp" in result:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], errors="raise"
        ).dt.normalize()
    if "asset_id" in result and result["asset_id"].isna().any():
        raise ValueError("portfolio_input_key_invalid")
    if "timestamp" in result and result["timestamp"].isna().any():
        raise ValueError("portfolio_input_key_invalid")
    if {"timestamp", "asset_id"}.issubset(result.columns) and result.duplicated(
        ["timestamp", "asset_id"]
    ).any():
        raise ValueError("portfolio_input_key_invalid")
    return result.sort_values(
        [column for column in ("timestamp", "asset_id") if column in result],
        kind="mergesort",
    ).reset_index(drop=True)


_EXECUTION_COLUMNS = ("asset_id", "timestamp", "can_enter", "can_exit")
