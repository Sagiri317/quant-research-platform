"""Exact economic reduction shared by portfolio producers and artifact readers."""

from fractions import Fraction

from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
from quant_research_platform.contracts.validation import finite_exact_float


def cash_equity_positions(rows):
    """Exact raw-share and receivable values before local cancellation."""
    return {
        row.asset_id: Fraction(row.quantity) * Fraction(row.mark_price)
        + Fraction(row.receivable)
        for row in rows
        if row.asset_id != PORTFOLIO_CASH_ASSET_ID
    }


def cash_equity_pnl(previous, current, cashflows, rights_adjustments=()):
    """Cancel each asset's raw-price/share changes and flows before rounding."""
    changes = {
        asset: current.get(asset, 0) - previous.get(asset, 0)
        for asset in current.keys() | previous.keys()
    }
    total = Fraction()
    for flow in cashflows:
        if flow.kind.endswith("_cost"):
            total += Fraction(flow.amount)
        else:
            changes[flow.asset_id] = changes.get(flow.asset_id, 0) + Fraction(
                flow.amount
            )
    converted = {
        asset
        for event in rights_adjustments
        if event.target_asset_id is not None
        for asset in (event.asset_id, event.target_asset_id)
    }
    conversion_change = sum((changes.pop(asset, 0) for asset in converted), Fraction())
    for change in (*changes.values(), conversion_change):
        value = finite_exact_float(change)
        if value is None:
            raise ValueError("portfolio_pnl_unrepresentable")
        # Monetary increments have bounded dyadic denominators across assets.
        total += Fraction(value)
    return total


class _LedgerEconomics:
    """Accumulate economic facts without subtracting rounded capital or totals."""

    def __init__(self, capital):
        self.capital = capital
        self.position_value = None
        self.positions = None
        self.totals = {
            name: Fraction()
            for name in ("net_pnl", "fees", "funding", "turnover", "fill_quantity")
        }
        self.fill_events = 0
        self.timestamp_ns = 0

    def consume(self, batch):
        end = [row for row in batch.valuations if row.phase == "end"]
        equity = self.capital.valuation_method == "cash_equity"
        if equity:
            current = cash_equity_positions(end)
            if self.positions is None:
                self.positions = cash_equity_positions(
                    row for row in batch.valuations if row.phase == "before"
                )
            increment = dict.fromkeys(self.totals, 0)
            increment["net_pnl"] = cash_equity_pnl(
                self.positions, current, batch.cashflows, batch.rights_adjustments
            )
            self.positions = current
        else:
            position = sum(
                (
                    Fraction(row.value)
                    for row in end
                    if row.asset_id != PORTFOLIO_CASH_ASSET_ID
                ),
                Fraction(),
            )
            if self.position_value is None:
                self.position_value = sum(
                    (
                        Fraction(row.value)
                        for row in batch.valuations
                        if row.phase == "before"
                        and row.asset_id != PORTFOLIO_CASH_ASSET_ID
                    ),
                    Fraction(),
                )
            increment = dict.fromkeys(self.totals, 0)
            increment["net_pnl"] = position - self.position_value
        for row in batch.cashflows:
            if row.amount:
                amount = Fraction(row.amount)
                if not equity:
                    increment["net_pnl"] += amount
                if row.kind.endswith("_cost"):
                    increment["fees"] -= amount
                elif row.kind == "funding":
                    increment["funding"] += amount
        for row in batch.trades:
            increment["turnover"] += abs(Fraction(row.notional))
            increment["fill_quantity"] += abs(Fraction(row.quantity))
        for name, amount in increment.items():
            if amount:
                self.totals[name] += amount
        increment["fill_events"] = len(batch.trades)
        self.fill_events += len(batch.trades)
        if not equity:
            self.position_value = position
        self.timestamp_ns = max(row.timestamp_ns for row in end)
        return increment

    def summary(self):
        net, fees, funding = (
            self.totals[name] for name in ("net_pnl", "fees", "funding")
        )
        fill_quantity = finite_exact_float(self.totals["fill_quantity"])
        capital_return = finite_exact_float(
            net / Fraction(self.capital.initial_capital)
        )
        return {
            **{
                name: float(value)
                for name, value in self.totals.items()
                if name != "fill_quantity"
            },
            "fill_quantity": fill_quantity,
            "fill_quantity_reason": "nonfinite_result"
            if fill_quantity is None
            else None,
            "gross_pnl": float(net + fees - funding),
            "nav": float(Fraction(self.capital.initial_capital) + net),
            "capital_return": capital_return,
            "capital_return_reason": "nonfinite_result"
            if capital_return is None
            else None,
            "fill_events": self.fill_events,
            "valuation_timestamp_ns": self.timestamp_ns,
        }
