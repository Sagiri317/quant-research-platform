"""Common portfolio contracts and market-policy bindings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from math import fma, isclose, isfinite, ulp
from typing import Literal

import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_float,
)

A_SHARE_EQUITY_EVENTS_DATA_ID = "a_share_equity_events"
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
    "shares_per_old_unit",
    "unit_basis_date",
    "successor_shares_per_unit",
    "successor_unit_basis_date",
    "unit_source_locator",
    "evidence_identity",
    "source_locator",
)

REBALANCE_SCHEDULES = {
    "daily",
    "weekly-last-trading-day",
    "monthly-last-trading-day",
}
PORTFOLIO_CASH_ASSET_ID = "__cash__"
RESEARCH_VALUATION_VERSION = "RESEARCH_DELISTING_ZERO_V1"
PORTFOLIO_ECONOMIC_SUMMARY_FIELDS = frozenset(
    {
        "gross_pnl",
        "net_pnl",
        "nav",
        "capital_return",
        "capital_return_reason",
        "initial_capital",
        "valuation_currency",
        "candidate_id",
        "fees",
        "funding",
        "turnover",
        "fill_events",
        "fill_quantity",
        "fill_quantity_reason",
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
        "execution_fact_unknown",
        "research_unquoted_valuation",
        "adv_unavailable",
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
    quantity: float | Fraction
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
    quantity: float | Fraction
    value: float
    mark_price: float | Fraction = 1.0
    cost_basis: float = 0.0
    lifecycle_id: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioDiagnostic:
    asset_id: str
    fill_id: str | None
    name: str
    number: float | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioRightsAdjustment:
    """Evidence-linked nontrade change of an existing adjusted-unit entitlement."""

    event_id: str
    timestamp_ns: int
    asset_id: str
    lifecycle_id: str
    quantity_before: float | Fraction
    quantity_after: float | Fraction
    cash_amount: float
    target_asset_id: str | None
    target_lifecycle_id: str | None
    target_quantity: float | Fraction
    evidence_identity: str


@dataclass(frozen=True, slots=True)
class PortfolioLedgerBatch:
    """One bounded economic transition and its linked market diagnostics."""

    sequence: int
    trades: tuple[ExecutedTrade, ...]
    cashflows: tuple[PortfolioCashFlow, ...]
    valuations: tuple[PortfolioValuation, ...]
    diagnostics: tuple[PortfolioDiagnostic, ...]
    rights_adjustments: tuple[PortfolioRightsAdjustment, ...] = ()

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("portfolio_ledger_sequence_invalid")
        for rows, kind in (
            (self.trades, ExecutedTrade),
            (self.cashflows, PortfolioCashFlow),
            (self.valuations, PortfolioValuation),
            (self.diagnostics, PortfolioDiagnostic),
            (self.rights_adjustments, PortfolioRightsAdjustment),
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
                    "quantity_before",
                    "quantity_after",
                    "cash_amount",
                    "target_quantity",
                ):
                    if not hasattr(row, name):
                        continue
                    value = getattr(row, name)
                    if value is None and name != "number":
                        raise ValueError("portfolio_ledger_numeric_invalid")
                    if isinstance(value, Fraction) and name in {
                        "quantity",
                        "quantity_before",
                        "quantity_after",
                        "target_quantity",
                        "mark_price",
                    }:
                        continue
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
            or row.quantity == 0
            or row.notional == 0
            or (row.quantity > 0) != (row.notional > 0)
            for row in self.trades
        ):
            raise ValueError("portfolio_ledger_trade_invalid")
        if len({row.flow_id for row in self.cashflows}) != len(self.cashflows) or any(
            type(row.flow_id) is not str
            or not row.flow_id
            or row.kind
            not in {
                "principal",
                "realized_pnl",
                "funding",
                "recovery",
                "commission_cost",
                "stamp_duty_cost",
                "transfer_fee_cost",
                "slippage_cost",
                "impact_cost",
            }
            or (row.fill_id is not None and row.fill_id not in fills)
            or ((row.kind in {"funding", "recovery"}) != (row.fill_id is None))
            for row in self.cashflows
        ):
            raise ValueError("portfolio_ledger_cashflow_invalid")
        for row in (
            *self.trades,
            *self.cashflows,
            *self.valuations,
            *self.rights_adjustments,
        ):
            if type(row.timestamp_ns) is not int or row.timestamp_ns < 0:
                raise ValueError("portfolio_ledger_timestamp_invalid")
        phases = {
            phase: {row.asset_id: row for row in self.valuations if row.phase == phase}
            for phase in ("before", "after", "end")
        }
        if sum(map(len, phases.values())) != len(self.valuations):
            raise ValueError("portfolio_ledger_valuation_invalid")
        for row in self.valuations:
            if row.lifecycle_id is not None and (
                type(row.lifecycle_id) is not str or not row.lifecycle_id
            ):
                raise ValueError("portfolio_ledger_lifecycle_invalid")
            try:
                expected = (
                    float(
                        row.quantity * Fraction(row.mark_price)
                        - Fraction(row.cost_basis)
                    )
                    if isinstance(row.quantity, Fraction)
                    else fma(row.quantity, row.mark_price, -row.cost_basis)
                )
            except OverflowError:
                raise ValueError("portfolio_ledger_valuation_invalid") from None
            # Cost and final values round independently; the product need not fit.
            tolerance = ulp(row.cost_basis) + ulp(row.value) + ulp(expected)
            if row.mark_price < 0 or not isclose(
                row.value, expected, rel_tol=0.0, abs_tol=tolerance
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
        if any(
            row.asset_id not in before and row.fill_id is None
            for row in self.diagnostics
        ):
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
        tolerance = sum(
            portfolio_notional_tolerance(abs(row.value)) for row in before.values()
        )
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
        deltas = defaultdict(lambda: [0, 0])
        quantities = {asset: value.quantity for asset, value in before.items()}
        lives = {asset: value.lifecycle_id for asset, value in before.items()}
        event_ids = set()
        recovery = defaultdict(float)
        for event in self.rights_adjustments:
            if (
                type(event.event_id) is not str
                or not event.event_id
                or type(event.lifecycle_id) is not str
                or not event.lifecycle_id
                or type(event.evidence_identity) is not str
                or event.event_id in event_ids
                or event.asset_id not in assets
                or event.timestamp_ns != after[event.asset_id].timestamp_ns
                or event.quantity_before <= 0
                or event.quantity_after < 0
                or event.cash_amount < 0
                or event.target_quantity < 0
                or len(event.evidence_identity) != 64
                or any(c not in "0123456789abcdef" for c in event.evidence_identity)
                or event.lifecycle_id != lives[event.asset_id]
                or not _quantity_equal(
                    event.quantity_before, quantities[event.asset_id]
                )
            ):
                raise ValueError("portfolio_rights_adjustment_invalid")
            event_ids.add(event.event_id)
            event_flows = [
                flow
                for flow in self.cashflows
                if flow.kind == "recovery"
                and flow.flow_id == f"recovery:{event.event_id}"
            ]
            if (
                (event.cash_amount > 0) != bool(event_flows)
                or len(event_flows) > 1
                or any(
                    flow.asset_id != event.asset_id or flow.amount != event.cash_amount
                    for flow in event_flows
                )
            ):
                raise ValueError("portfolio_recovery_reconciliation_invalid")
            change = event.quantity_after - event.quantity_before
            quantities[event.asset_id] = event.quantity_after
            deltas[event.asset_id][0] += change
            deltas[event.asset_id][1] = max(deltas[event.asset_id][1], abs(change))
            recovery[event.asset_id] += event.cash_amount
            if event.target_asset_id is None:
                if event.target_lifecycle_id is not None or event.target_quantity != 0:
                    raise ValueError("portfolio_rights_conversion_invalid")
            else:
                target = event.target_asset_id
                if (
                    target not in assets
                    or not event.target_lifecycle_id
                    or event.target_quantity <= 0
                    or (
                        quantities[target] != 0
                        and lives[target] != event.target_lifecycle_id
                    )
                ):
                    raise ValueError("portfolio_rights_conversion_invalid")
                quantities[target] = (
                    Fraction(quantities[target]) + event.target_quantity
                )
                lives[target] = event.target_lifecycle_id
                deltas[target][0] += event.target_quantity
                deltas[target][1] = max(deltas[target][1], event.target_quantity)
        for asset in assets:
            if not isclose(
                recovery[asset],
                sum(
                    row.amount
                    for row in self.cashflows
                    if row.asset_id == asset and row.kind == "recovery"
                ),
                rel_tol=0,
                abs_tol=tolerance,
            ):
                raise ValueError("portfolio_recovery_reconciliation_invalid")
            if lives[asset] != after[asset].lifecycle_id:
                raise ValueError("portfolio_rights_lifecycle_invalid")
        for trade in self.trades:
            deltas[trade.asset_id][0] += trade.quantity
            deltas[trade.asset_id][1] = max(
                deltas[trade.asset_id][1], abs(trade.quantity)
            )
            if trade.timestamp_ns != after[trade.asset_id].timestamp_ns:
                raise ValueError("portfolio_ledger_fill_timing_invalid")
        for flow in self.cashflows:
            if flow.timestamp_ns != after[flow.asset_id].timestamp_ns:
                raise ValueError("portfolio_ledger_cashflow_timing_invalid")
        for asset in before.keys() - {PORTFOLIO_CASH_ASSET_ID}:
            delta, trade_quantity_scale = deltas[asset]
            if not _quantity_equal(
                before[asset].quantity + delta,
                after[asset].quantity,
                trade_quantity_scale,
            ):
                raise ValueError("portfolio_ledger_position_reconciliation_invalid")
        if any(
            type(row.name) is not str
            or not row.name
            or (
                row.fill_id is not None
                and (type(row.fill_id) is not str or not row.fill_id)
            )
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
            object.__setattr__(
                self,
                "rate",
                strict_finite_float(self.rate, error="portfolio_funding_rate_invalid"),
            )
        if self.mark_price is not None:
            mark = strict_finite_float(
                self.mark_price, error="portfolio_funding_mark_invalid"
            )
            if mark <= 0:
                raise ValueError("portfolio_funding_mark_invalid")
            object.__setattr__(self, "mark_price", mark)


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
        normalized = []
        for item in self.cashflows:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or not previous < item[0] <= self.known_until_ns
            ):
                raise ValueError("portfolio_funding_forecast_events_invalid")
            amount = (
                None
                if item[1] is None
                else strict_finite_float(
                    item[1], error="portfolio_funding_forecast_cashflow_invalid"
                )
            )
            normalized.append((item[0], amount))
            previous = item[0]
        object.__setattr__(self, "cashflows", tuple(normalized))


def portfolio_notional_tolerance(nav: float) -> float:
    return min(ulp(nav) * 512.0, abs(nav) * (ulp(1.0) * 512.0))


def _quantity_tolerance(*quantities: float | Fraction) -> float:
    return (
        max(
            (ulp(value) for value in quantities if not isinstance(value, Fraction)),
            default=0.0,
        )
        * 512.0
    )


def _quantity_equal(left, right, *scales):
    if isinstance(left, Fraction) or isinstance(right, Fraction):
        return Fraction(left) == Fraction(right)
    return isclose(
        left, right, rel_tol=0.0, abs_tol=_quantity_tolerance(left, right, *scales)
    )


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
    tolerance = portfolio_notional_tolerance(
        sum(row.value for row in before.values())
        if capital.valuation_method == "adjusted_notional"
        else capital.initial_capital
    )
    if previous is None:
        if not isclose(
            sum(row.value for row in before.values()),
            capital.initial_capital,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("portfolio_initial_capital_reconciliation_invalid")
    else:
        try:
            for asset in before.keys() | previous.keys():
                old, current = previous.get(asset), before.get(asset)
                previous_quantity = old.quantity if old else 0.0
                current_quantity = current.quantity if current else 0.0
                if not (
                    isclose(
                        previous_quantity,
                        current_quantity,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    )
                    if asset == PORTFOLIO_CASH_ASSET_ID
                    else _quantity_equal(previous_quantity, current_quantity)
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
                if (
                    old is not None
                    and current is not None
                    and old.quantity != 0
                    and old.lifecycle_id != current.lifecycle_id
                ):
                    raise ValueError("portfolio_held_lifecycle_unproven")
        finally:
            previous = old = None
    research_marks = [
        row for row in batch.diagnostics if row.name.startswith("research_")
    ]
    if research_marks:
        policy = [
            row for row in research_marks if row.name == "research_valuation_version"
        ]
        if (
            len(policy) != 1
            or policy[0].asset_id != PORTFOLIO_CASH_ASSET_ID
            or policy[0].text != RESEARCH_VALUATION_VERSION
        ):
            raise ValueError("portfolio_research_valuation_version_invalid")
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
            rights_quantity = sum(
                (
                    event.quantity_after - event.quantity_before
                    if event.asset_id == asset
                    else 0
                )
                + (event.target_quantity if event.target_asset_id == asset else 0)
                for event in batch.rights_adjustments
            )
            executable_value = float(
                (Fraction(initial.quantity) + rights_quantity)
                * Fraction(post.mark_price)
            )
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
                if (
                    reason is not None
                    and reason.text == "adv_unavailable"
                    and (
                        adv is not None
                        or trades
                        or any(flow.kind != "recovery" for flow in flows)
                    )
                ):
                    raise ValueError("portfolio_ledger_adv_unavailable_invalid")
                counts = [
                    diagnostics.get(name)
                    for name in (
                        "adv20_observation_count",
                        "adv20_missing_observations",
                    )
                ]
                if any(count is not None for count in counts) and (
                    any(
                        count is None
                        or count.number is None
                        or not 0 <= count.number <= 20
                        or count.number != int(count.number)
                        for count in counts
                    )
                    or counts[1].number > counts[0].number
                    or (counts[0].number == 20 and counts[1].number == 0)
                    or adv is not None
                ):
                    raise ValueError("portfolio_ledger_adv_history_invalid")
                desired = target.number - executable_value
                executed = sum(row.notional for row in trades)
                if (
                    (desired > 0 and executed < 0)
                    or (desired < 0 and executed > 0)
                    or (desired == 0 and executed != 0)
                    or abs(executed) > abs(desired) + tolerance
                    or (abs(target.number - post.value) > tolerance)
                    != (reason is not None)
                ):
                    raise ValueError("portfolio_ledger_direction_invalid")
            if any(
                (asset != PORTFOLIO_CASH_ASSET_ID and row.quantity < 0)
                or row.value < -tolerance
                or row.cost_basis != 0.0
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
                if row.kind not in {"principal", "recovery"}
            ):
                raise ValueError("portfolio_ledger_cost_invalid")
            if (
                (
                    initial.quantity != 0
                    and initial.mark_price != post.mark_price
                    and not batch.rights_adjustments
                )
                or post.lifecycle_id != final.lifecycle_id
                or final.quantity != post.quantity
                or any(
                    not isclose(
                        trade.notional,
                        float(Fraction(trade.quantity) * Fraction(post.mark_price)),
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    )
                    for trade in trades
                )
            ):
                raise ValueError("portfolio_ledger_return_state_invalid")
        elif asset != PORTFOLIO_CASH_ASSET_ID:
            if any(row.kind == "principal" for row in flows):
                raise ValueError("portfolio_perpetual_principal_cashflow_invalid")
            terms = (
                initial.cost_basis,
                *(row.notional for row in trades),
                *(row.amount for row in flows if row.kind == "realized_pnl"),
            )
            # Each stored monetary term rounds independently; add without cancellation.
            rounding = sum(
                (Fraction(ulp(value)) / 2 for value in (*terms, post.cost_basis)),
                Fraction(),
            )
            if (
                abs(
                    Fraction(post.cost_basis)
                    - sum((Fraction(value) for value in terms), Fraction())
                )
                > max(Fraction(tolerance), rounding)
                or final.quantity != post.quantity
                or final.cost_basis != post.cost_basis
            ):
                raise ValueError("portfolio_ledger_cost_basis_invalid")
    if capital.valuation_method == "adjusted_notional" and not isclose(
        sum(diagnostics_by_asset[asset]["target_notional"].number for asset in before),
        sum(row.value for row in after.values())
        - sum(
            row.amount
            for row in batch.cashflows
            if row.kind in PORTFOLIO_LEDGER_COST_COLUMNS
        ),
        rel_tol=0.0,
        abs_tol=max(
            tolerance,
            portfolio_notional_tolerance(sum(row.value for row in after.values())),
        ),
    ):
        raise ValueError("portfolio_ledger_target_state_invalid")
    return {asset: replace(row) for asset, row in end.items()}


@dataclass(frozen=True, slots=True)
class PortfolioCandidateSpec:
    """One protocol-owned portfolio candidate composition."""

    candidate_id: str
    selection_group: Literal["global", "quarterly"]
    construction: Literal[
        "top20_equal_weight",
        "top20_equal_weight_half_step",
        "top20_float_cap",
        "benchmark_enhanced",
    ]
    prediction_selector: str
    rebalance_schedule: str
    top_quantile: float | None
    benchmark_id: str | None
    risk_style_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.selection_group) is not str or self.selection_group not in {
            "global",
            "quarterly",
        }:
            raise ValueError("portfolio_selection_group_invalid")
        for value in (
            self.candidate_id,
            self.prediction_selector,
        ):
            if type(value) is not str or not value:
                raise ValueError("portfolio_candidate_dependency_invalid")
        if type(self.construction) is not str or self.construction not in {
            "top20_equal_weight",
            "top20_equal_weight_half_step",
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
        if (
            type(self.risk_style_columns) is not tuple
            or any(
                type(column) is not str or not column
                for column in self.risk_style_columns
            )
            or len(set(self.risk_style_columns)) != len(self.risk_style_columns)
        ):
            raise ValueError("portfolio_candidate_risk_styles_invalid")
        if self.construction == "benchmark_enhanced":
            if self.top_quantile is not None or self.benchmark_id is None:
                raise ValueError("portfolio_candidate_construction_invalid")
            if not self.risk_style_columns:
                raise ValueError("portfolio_candidate_risk_styles_invalid")
        else:
            if self.top_quantile is None or self.benchmark_id is not None:
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
