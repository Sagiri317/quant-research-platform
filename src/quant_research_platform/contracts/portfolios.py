"""Common portfolio contracts and market-policy bindings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import isclose, isfinite, ulp
from typing import Literal

from quant_research_platform.contracts.validation import (
    strict_finite_float,
)

REBALANCE_SCHEDULES = {
    "daily",
    "weekly-last-trading-day",
    "monthly-last-trading-day",
}
PORTFOLIO_CASH_ASSET_ID = "__cash__"
PORTFOLIO_ECONOMIC_SUMMARY_FIELDS = frozenset(
    {
        "gross_pnl",
        "net_pnl",
        "nav",
        "capital_return",
        "initial_capital",
        "valuation_currency",
        "candidate_id",
        "fees",
        "funding",
        "turnover",
        "fill_events",
        "fill_quantity",
        "valuation_timestamp_ns",
        "inventory_revaluation_pnl",
    }
)
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
PORTFOLIO_LEDGER_COST_COLUMNS = (
    "commission_cost",
    "stamp_duty_cost",
    "transfer_fee_cost",
    "slippage_cost",
    "impact_cost",
)
CONTINUOUS_EXECUTION_COLUMNS = (
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
    "decision_delta_weight",
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
class PortfolioCapital:
    """Independent research capital; no shared pool or brokerage account."""

    candidate_id: str
    initial_capital: float
    valuation_currency: str
    valuation_method: Literal["adjusted_notional", "linear_perpetual"]

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or not self.candidate_id
            or type(self.valuation_currency) is not str
            or not self.valuation_currency.isascii()
            or not self.valuation_currency.isalpha()
            or not self.valuation_currency.isupper()
            or self.valuation_method not in {"adjusted_notional", "linear_perpetual"}
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
    cost_basis: float = 0.0


@dataclass(frozen=True, slots=True)
class PortfolioDiagnostic:
    asset_id: str
    fill_id: str | None
    name: str
    number: float | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioLedgerBatch:
    """One bounded economic transition and its linked market diagnostics."""

    sequence: int
    trades: tuple[ExecutedTrade, ...]
    cashflows: tuple[PortfolioCashFlow, ...]
    valuations: tuple[PortfolioValuation, ...]
    diagnostics: tuple[PortfolioDiagnostic, ...]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("portfolio_ledger_sequence_invalid")
        for rows, kind in (
            (self.trades, ExecutedTrade),
            (self.cashflows, PortfolioCashFlow),
            (self.valuations, PortfolioValuation),
            (self.diagnostics, PortfolioDiagnostic),
        ):
            if type(rows) is not tuple or any(type(row) is not kind for row in rows):
                raise ValueError("portfolio_ledger_table_invalid")
            if any(type(row.asset_id) is not str or not row.asset_id for row in rows):
                raise ValueError("portfolio_ledger_asset_invalid")
            for row in rows:
                for name in (
                    "quantity",
                    "notional",
                    "amount",
                    "value",
                    "mark_price",
                    "cost_basis",
                    "number",
                ):
                    if not hasattr(row, name):
                        continue
                    value = getattr(row, name)
                    if value is None and name != "number":
                        raise ValueError("portfolio_ledger_numeric_invalid")
                    if value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not isfinite(value)
                    ):
                        raise ValueError("portfolio_ledger_numeric_invalid")
        fills = {row.fill_id for row in self.trades}
        if len(fills) != len(self.trades) or any(
            type(row.fill_id) is not str
            or not row.fill_id
            or type(row.order_id) is not str
            or not row.order_id
            or not isfinite(row.quantity)
            or not isfinite(row.notional)
            or row.quantity * row.notional <= 0
            for row in self.trades
        ):
            raise ValueError("portfolio_ledger_trade_invalid")
        if len({row.flow_id for row in self.cashflows}) != len(self.cashflows) or any(
            type(row.flow_id) is not str
            or not row.flow_id
            or not isfinite(row.amount)
            or row.kind
            not in {
                "principal",
                "realized_pnl",
                "funding",
                "commission_cost",
                "stamp_duty_cost",
                "transfer_fee_cost",
                "slippage_cost",
                "impact_cost",
            }
            or (row.fill_id is not None and row.fill_id not in fills)
            or ((row.kind == "funding") != (row.fill_id is None))
            for row in self.cashflows
        ):
            raise ValueError("portfolio_ledger_cashflow_invalid")
        for row in (*self.trades, *self.cashflows, *self.valuations):
            if type(row.timestamp_ns) is not int or row.timestamp_ns < 0:
                raise ValueError("portfolio_ledger_timestamp_invalid")
        phases = {
            phase: {row.asset_id: row for row in self.valuations if row.phase == phase}
            for phase in ("before", "after", "end")
        }
        if sum(map(len, phases.values())) != len(self.valuations) or any(
            not all(
                isfinite(value)
                for value in (row.quantity, row.value, row.mark_price, row.cost_basis)
            )
            or row.mark_price < 0
            or not isclose(
                row.value,
                row.quantity * row.mark_price - row.cost_basis,
                rel_tol=ulp(1.0) * 512.0,
                abs_tol=1e-12,
            )
            for row in self.valuations
        ):
            raise ValueError("portfolio_ledger_valuation_invalid")
        if any(PORTFOLIO_CASH_ASSET_ID not in phase for phase in phases.values()):
            raise ValueError("portfolio_ledger_cash_valuation_missing")
        before, after, end = (phases[phase] for phase in ("before", "after", "end"))
        if set(before) != set(after) or set(before) != set(end):
            raise ValueError("portfolio_ledger_valuation_keys_invalid")
        assets = before.keys() - {PORTFOLIO_CASH_ASSET_ID}
        if any(row.asset_id not in assets for row in (*self.trades, *self.cashflows)):
            raise ValueError("portfolio_ledger_orphan_economic_fact")
        if any(row.asset_id not in before for row in self.diagnostics):
            raise ValueError("portfolio_ledger_orphan_diagnostic")
        fill_assets = {row.fill_id: row.asset_id for row in self.trades}
        if any(
            row.fill_id is not None and fill_assets[row.fill_id] != row.asset_id
            for row in self.cashflows
        ):
            raise ValueError("portfolio_ledger_cashflow_fill_mismatch")
        if any(
            row.quantity != row.value or row.mark_price != 1.0 or row.cost_basis != 0.0
            for row in self.valuations
            if row.asset_id == PORTFOLIO_CASH_ASSET_ID
        ):
            raise ValueError("portfolio_ledger_cash_valuation_invalid")
        scale = max(1.0, sum(abs(row.value) for row in before.values()))
        tolerance = max(ulp(1.0) * scale * 512.0, 1e-12)
        cash_before = before[PORTFOLIO_CASH_ASSET_ID].value
        cash_after = after[PORTFOLIO_CASH_ASSET_ID].value
        if (
            not isclose(
                cash_after - cash_before,
                sum(row.amount for row in self.cashflows),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            or end[PORTFOLIO_CASH_ASSET_ID].value != cash_after
        ):
            raise ValueError("portfolio_ledger_cash_reconciliation_invalid")
        deltas = defaultdict(float)
        for trade in self.trades:
            deltas[trade.asset_id] += trade.quantity
            if trade.timestamp_ns != after[trade.asset_id].timestamp_ns:
                raise ValueError("portfolio_ledger_fill_timing_invalid")
        for flow in self.cashflows:
            if flow.timestamp_ns != after[flow.asset_id].timestamp_ns:
                raise ValueError("portfolio_ledger_cashflow_timing_invalid")
        for asset in before.keys() - {PORTFOLIO_CASH_ASSET_ID}:
            delta = deltas[asset]
            if not isclose(
                before[asset].quantity + delta,
                after[asset].quantity,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("portfolio_ledger_position_reconciliation_invalid")
        if any(
            type(row.name) is not str
            or not row.name
            or (row.number is not None and not isfinite(row.number))
            or (row.text is not None and type(row.text) is not str)
            or (row.number is not None and row.text is not None)
            for row in self.diagnostics
        ):
            raise ValueError("portfolio_ledger_diagnostic_invalid")


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """A caller-declared settlement time and actual, possibly missing observations.

    The time belongs to the explicit research funding calendar. Rate and mark are
    settlement evidence, not information available to an earlier trade decision.
    """

    timestamp_ns: int
    rate: float | None
    mark_price: float | None

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("portfolio_funding_timestamp_invalid")
        if self.rate is not None:
            strict_finite_float(self.rate, error="portfolio_funding_rate_invalid")
        if (
            self.mark_price is not None
            and strict_finite_float(
                self.mark_price, error="portfolio_funding_mark_invalid"
            )
            <= 0
        ):
            raise ValueError("portfolio_funding_mark_invalid")


@dataclass(frozen=True, slots=True)
class FundingForecast:
    """Known funding coverage and expected cash per long unit, not settlements."""

    available_at_ns: int
    known_until_ns: int
    cashflows: tuple[tuple[int, float | None], ...]

    def __post_init__(self) -> None:
        if (
            type(self.available_at_ns) is not int
            or type(self.known_until_ns) is not int
            or not 0 <= self.available_at_ns < self.known_until_ns
            or type(self.cashflows) is not tuple
        ):
            raise ValueError("portfolio_funding_forecast_coverage_invalid")
        previous = self.available_at_ns
        for item in self.cashflows:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or not previous < item[0] <= self.known_until_ns
            ):
                raise ValueError("portfolio_funding_forecast_events_invalid")
            if item[1] is not None:
                strict_finite_float(
                    item[1], error="portfolio_funding_forecast_cashflow_invalid"
                )
            previous = item[0]


def validate_portfolio_transition(
    batch: PortfolioLedgerBatch,
    capital: PortfolioCapital,
    previous: dict[str, PortfolioValuation] | None,
) -> dict[str, PortfolioValuation]:
    """Check persisted continuity and instrument-specific economic identities."""
    phases = {
        phase: {row.asset_id: row for row in batch.valuations if row.phase == phase}
        for phase in ("before", "after", "end")
    }
    before, after, end = (phases[phase] for phase in ("before", "after", "end"))
    tolerance = max(ulp(1.0) * max(1.0, capital.initial_capital) * 512.0, 1e-12)
    if previous is None:
        if not isclose(
            sum(row.value for row in before.values()),
            capital.initial_capital,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("portfolio_initial_capital_reconciliation_invalid")
    else:
        for asset in before.keys() | previous.keys():
            old, current = previous.get(asset), before.get(asset)
            if not isclose(
                (
                    old.value
                    if capital.valuation_method == "adjusted_notional"
                    else old.quantity
                )
                if old
                else 0.0,
                current.quantity if current else 0.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("portfolio_ledger_cross_session_state_invalid")
            if (
                old is not None
                and current is not None
                and current.timestamp_ns < old.timestamp_ns
            ):
                raise ValueError("portfolio_ledger_cross_session_timing_invalid")
            if (
                old is not None
                and current is not None
                and old.cost_basis != current.cost_basis
            ):
                raise ValueError("portfolio_ledger_cost_basis_invalid")
    trades_by_asset = defaultdict(list)
    flows_by_asset = defaultdict(list)
    diagnostics_by_asset = defaultdict(dict)
    for trade in batch.trades:
        trades_by_asset[trade.asset_id].append(trade)
    for flow in batch.cashflows:
        flows_by_asset[flow.asset_id].append(flow)
    for diagnostic in batch.diagnostics:
        if (
            capital.valuation_method == "adjusted_notional"
            and diagnostic.name in diagnostics_by_asset[diagnostic.asset_id]
        ):
            raise ValueError("portfolio_ledger_diagnostic_keys_invalid")
        diagnostics_by_asset[diagnostic.asset_id][diagnostic.name] = diagnostic
    for asset, initial in before.items():
        post, final = after[asset], end[asset]
        if not initial.timestamp_ns <= post.timestamp_ns <= final.timestamp_ns:
            raise ValueError("portfolio_ledger_execution_timing_invalid")
        trades = trades_by_asset[asset]
        flows = flows_by_asset[asset]
        if capital.valuation_method == "adjusted_notional":
            diagnostics = diagnostics_by_asset[asset]
            for name in ("alpha_valid_dates", "alpha_scheduled_dates"):
                diagnostic = diagnostics.get(name)
                if diagnostic is not None and (
                    diagnostic.number is None
                    or diagnostic.number < 0
                    or int(diagnostic.number) != diagnostic.number
                ):
                    raise ValueError("portfolio_ledger_count_invalid")
            reason = diagnostics.get("block_reason")
            if reason is not None and reason.text not in PORTFOLIO_BLOCK_REASONS:
                raise ValueError("portfolio_ledger_block_reason_invalid")
            decision = diagnostics.get("decision_reason")
            if (
                decision is not None and decision.text not in PORTFOLIO_DECISION_REASONS
            ) or (asset == PORTFOLIO_CASH_ASSET_ID) != (decision is not None):
                raise ValueError("portfolio_ledger_decision_reason_invalid")
            target = diagnostics.get("target_notional")
            if target is None or target.number is None or target.number < 0:
                raise ValueError("portfolio_ledger_target_state_invalid")
            if asset != PORTFOLIO_CASH_ASSET_ID:
                adv = diagnostics.get("adv20")
                if (adv is not None and (adv.number is None or adv.number < 0)) or (
                    trades and (adv is None or adv.number is None or adv.number <= 0)
                ):
                    raise ValueError("portfolio_ledger_adv_invalid")
                desired = target.number - initial.value
                executed = sum(row.notional for row in trades)
                if (
                    desired * executed < 0
                    or abs(executed) > abs(desired) + tolerance
                    or (abs(target.number - post.value) > tolerance)
                    != (reason is not None)
                ):
                    raise ValueError("portfolio_ledger_direction_invalid")
            if any(
                row.value < -tolerance or row.cost_basis != 0.0
                for row in (initial, post, final)
            ):
                raise ValueError("portfolio_ledger_notional_invalid")
            principal = sum(row.amount for row in flows if row.kind == "principal")
            if not isclose(
                principal,
                -sum(row.notional for row in trades),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError("portfolio_ledger_principal_invalid")
            if any(
                row.kind in {"funding", "realized_pnl"} or row.amount > 0.0
                for row in flows
                if row.kind != "principal"
            ):
                raise ValueError("portfolio_ledger_cost_invalid")
            if (
                initial.quantity != initial.value
                or post.quantity != post.value
                or final.quantity != post.quantity
            ):
                raise ValueError("portfolio_ledger_return_state_invalid")
        elif asset != PORTFOLIO_CASH_ASSET_ID:
            if any(row.kind == "principal" for row in flows):
                raise ValueError("portfolio_perpetual_principal_cashflow_invalid")
            realized = sum(row.amount for row in flows if row.kind == "realized_pnl")
            if (
                not isclose(
                    post.cost_basis,
                    initial.cost_basis + sum(row.notional for row in trades) + realized,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or final.quantity != post.quantity
                or final.cost_basis != post.cost_basis
            ):
                raise ValueError("portfolio_ledger_cost_basis_invalid")
    if capital.valuation_method == "adjusted_notional" and not isclose(
        sum(diagnostics_by_asset[asset]["target_notional"].number for asset in before),
        sum(row.value for row in before.values()),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("portfolio_ledger_target_state_invalid")
    return {asset: replace(row) for asset, row in end.items()}


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
    prediction_selector: str
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
    risk_context_target_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (
            self.candidate_id,
            self.prediction_target_id,
            self.prediction_selector,
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
        if (
            type(self.risk_context_target_ids) is not tuple
            or any(
                type(target) is not str or not target
                for target in self.risk_context_target_ids
            )
            or len(set(self.risk_context_target_ids))
            != len(self.risk_context_target_ids)
        ):
            raise ValueError("portfolio_candidate_risk_context_invalid")
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
            if not self.risk_context_target_ids:
                raise ValueError("portfolio_candidate_risk_context_invalid")
        else:
            if self.top_quantile is None or self.benchmark_id is not None:
                raise ValueError("portfolio_candidate_construction_invalid")
            if self.risk_style_columns or self.risk_context_target_ids:
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
