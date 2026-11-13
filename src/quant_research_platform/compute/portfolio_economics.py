"""Economic reduction shared by portfolio producers and artifact readers."""

from math import fsum, isfinite

import numpy as np

from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID


def _nav_return_roundoff(start: object, end: object, returns: object) -> np.ndarray:
    """Bound ordinary NAV-ratio roundoff in dimensionless return units."""
    before = np.asarray(start, dtype="float64")
    after = np.asarray(end, dtype="float64")
    data = np.asarray(returns, dtype="float64")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tolerance = (
            4 * np.finfo(float).eps * (np.abs(after / before) + np.abs(data) + 1)
        )
    return np.where(
        (before > 0) & np.isfinite(before) & np.isfinite(after) & np.isfinite(data),
        tolerance,
        np.inf,
    )


def cash_equity_positions(rows):
    return {
        row.asset_id: row.quantity * row.mark_price + row.receivable
        for row in rows
        if row.asset_id != PORTFOLIO_CASH_ASSET_ID
    }


def cash_equity_pnl(previous, current, cashflows):
    """Net change in stock claims plus cash flows, without subtracting total NAVs."""
    return fsum(
        (
            *current.values(),
            *(-value for value in previous.values()),
            *(flow.amount for flow in cashflows),
        )
    )


class _LedgerEconomics:
    def __init__(self, capital):
        self.capital = capital
        self.positions = None
        self.totals = dict.fromkeys(
            ("net_pnl", "fees", "turnover", "fill_quantity", "equity_rounding_pnl"), 0.0
        )
        self.fill_events = 0
        self.timestamp_ns = 0

    def consume(self, batch):
        end = [row for row in batch.valuations if row.phase == "end"]
        current = cash_equity_positions(end)
        if self.positions is None:
            self.positions = cash_equity_positions(
                row for row in batch.valuations if row.phase == "before"
            )
        increment = {
            "net_pnl": cash_equity_pnl(self.positions, current, batch.cashflows),
            "fees": -fsum(
                row.amount for row in batch.cashflows if row.kind.endswith("_cost")
            ),
            "turnover": fsum(abs(row.notional) for row in batch.trades),
            "fill_quantity": fsum(abs(row.quantity) for row in batch.trades),
        }
        marks = (
            {(row.phase, row.asset_id): row for row in batch.valuations}
            if batch.rights_adjustments
            else {}
        )
        rounding_values = []
        for event in batch.rights_adjustments:
            for asset, lifecycle, loss in (
                (event.asset_id, event.lifecycle_id, event.quantity_rounding_loss),
                (
                    event.target_asset_id,
                    event.target_lifecycle_id,
                    event.target_quantity_rounding_loss,
                ),
            ):
                if loss:
                    value = marks[event.phase, asset]
                    rounding_values.append(
                        -loss * value.mark_price
                        if value.mark_price > 0 and value.lifecycle_id == lifecycle
                        else float("nan")
                    )
        increment["equity_rounding_pnl"] = fsum(rounding_values)
        self.positions = current
        for name, amount in increment.items():
            self.totals[name] = fsum((self.totals[name], amount))
        increment["fill_events"] = len(batch.trades)
        self.fill_events += len(batch.trades)
        self.timestamp_ns = max(row.timestamp_ns for row in end)
        return increment

    def summary(self):
        net, fees = (self.totals[name] for name in ("net_pnl", "fees"))
        capital_return = net / self.capital.initial_capital
        return {
            **self.totals,
            "equity_rounding_pnl": self.totals["equity_rounding_pnl"]
            if isfinite(self.totals["equity_rounding_pnl"])
            else None,
            "equity_rounding_pnl_reason": None
            if isfinite(self.totals["equity_rounding_pnl"])
            else "rounding_price_unavailable",
            "fill_quantity_reason": None,
            "gross_pnl": fsum((net, fees)),
            "nav": fsum((self.capital.initial_capital, net)),
            "capital_return": capital_return if isfinite(capital_return) else None,
            "capital_return_reason": None
            if isfinite(capital_return)
            else "nonfinite_result",
            "fill_events": self.fill_events,
            "valuation_timestamp_ns": self.timestamp_ns,
        }
