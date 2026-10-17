"""Ordinary inventory churn must not retain historical average-cost precision."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    validate_portfolio_transition,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    PortfolioEconomics,
)


@pytest.mark.parametrize("reverse", (False, True))
def test_long_perpetual_churn_bounds_state_and_conserves_pnl(reverse):
    capital = PortfolioCapital("churn", 1_000_000.0, "USDT", "linear_perpetual")
    previous = None
    realized = Fraction()

    def consume(batch):
        nonlocal previous, realized
        previous = validate_portfolio_transition(batch, capital, previous)
        realized += sum(
            (
                Fraction(row.amount)
                for row in batch.cashflows
                if row.kind == "realized_pnl"
            ),
            Fraction(),
        )

    account = PortfolioEconomics(capital, "asset", consume)
    position = Fraction()
    reference_basis = Decimal(0)
    reference_realized = Decimal(0)
    marked_trade_pnl = Fraction()
    fees = Fraction()
    peak_bits = 0
    stamp = 0
    mark = 103.25

    with localcontext() as context:
        context.prec = 80

        def fill(quantity, price):
            nonlocal position, reference_basis, reference_realized
            nonlocal marked_trade_pnl, fees, peak_bits, stamp
            quantity = Fraction(quantity)
            dq = Decimal(quantity.numerator) / Decimal(quantity.denominator)
            dp = Decimal.from_float(price)
            old = Decimal(position.numerator) / Decimal(position.denominator)
            remaining = position + quantity
            if position and (position > 0) != (quantity > 0):
                entry = reference_basis / old
                reference_realized += (
                    min(abs(old), abs(dq)) * (1 if position > 0 else -1) * (dp - entry)
                )
                reference_basis = (old + dq) * (
                    entry if remaining and (remaining > 0) == (position > 0) else dp
                )
            else:
                reference_basis += dq * dp
            position = remaining
            marked_trade_pnl += quantity * (Fraction(mark) - Fraction(price))
            fees += Fraction(0.003)
            stamp += 1
            account.fill(
                timestamp_ns=stamp,
                order_id=str(stamp),
                quantity=quantity,
                price=price,
                mark_price=mark,
                fee=0.003,
            )
            assert account.quantity == position
            peak_bits = max(
                peak_bits,
                account._cost_basis.numerator.bit_length(),
                account._cost_basis.denominator.bit_length(),
            )

        fill(1.0, 100.0)
        for cycle in range(4000):
            direction = 1 if position > 0 else -1
            fill(direction * 0.1, 100.0 + cycle % 7)
            fill(-direction * 0.1, 101.0 + cycle % 5)
            if reverse and cycle % 97 == 0:
                fill(-2 * position, 102.0 + cycle % 3)
            assert abs(position) == 1

        # Independent high-precision average-cost oracle checks attribution;
        # marked executed notionals check total economics without that algorithm.
        assert float(realized) == pytest.approx(
            float(reference_realized), rel=0, abs=1e-10
        )
        assert account.cost_basis == pytest.approx(
            float(reference_basis), rel=0, abs=1e-11
        )
        expected_net = float(marked_trade_pnl - fees)
        summary = account.mark(stamp + 1, mark)
        assert summary["net_pnl"] == pytest.approx(expected_net, rel=0, abs=1e-10)
        assert account.cash == float(
            Fraction(capital.initial_capital) + realized - fees
        )
        assert peak_bits < 128

        # A final liquidation releases the residual rather than losing it.
        fill(-position, 104.0)
        assert account.quantity == 0
        assert account._cost_basis == 0
        assert account.mark(stamp + 1, mark)["net_pnl"] == pytest.approx(
            float(marked_trade_pnl - fees), rel=0, abs=1e-10
        )
