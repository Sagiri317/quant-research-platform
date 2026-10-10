"""Exact economic reduction shared by portfolio producers and artifact readers."""

from fractions import Fraction

from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID


class _LedgerEconomics:
    """Accumulate economic facts without subtracting rounded capital or totals."""

    def __init__(self, capital):
        self.capital = capital
        self.position_value = None
        self.totals = {
            name: Fraction()
            for name in ("net_pnl", "fees", "funding", "turnover", "fill_quantity")
        }
        self.fill_events = 0
        self.timestamp_ns = 0

    def consume(self, batch):
        end = [row for row in batch.valuations if row.phase == "end"]
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
                    if row.phase == "before" and row.asset_id != PORTFOLIO_CASH_ASSET_ID
                ),
                Fraction(),
            )
        increment = dict.fromkeys(self.totals, 0)
        increment["net_pnl"] = position - self.position_value
        for row in batch.cashflows:
            if row.amount:
                amount = Fraction(row.amount)
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
        self.position_value = position
        self.timestamp_ns = max(row.timestamp_ns for row in end)
        return increment

    def summary(self):
        net, fees, funding = (
            self.totals[name] for name in ("net_pnl", "fees", "funding")
        )
        return {
            **{name: float(value) for name, value in self.totals.items()},
            "gross_pnl": float(net + fees - funding),
            "nav": float(Fraction(self.capital.initial_capital) + net),
            "capital_return": float(net / Fraction(self.capital.initial_capital)),
            "fill_events": self.fill_events,
            "valuation_timestamp_ns": self.timestamp_ns,
        }
