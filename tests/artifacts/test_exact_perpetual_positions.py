from dataclasses import replace
from fractions import Fraction

import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    PortfolioCapital,
    validate_portfolio_transition,
)
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics


@pytest.mark.parametrize("direction", (-1, 1))
@pytest.mark.parametrize("closing", (1e16, 1e16 + 2))
def test_exact_perpetual_position_survives_cancellation_and_native_roundtrip(
    tmp_path, direction, closing
):
    capital = PortfolioCapital("exact_position", 1000.0, "USDT", "linear_perpetual")
    batches = []
    account = PortfolioEconomics(capital, "asset", batches.append)
    expected = Fraction()
    for stamp, quantity in enumerate((1e16, 1.0, -closing), 1):
        quantity *= direction
        expected += Fraction(quantity)
        account.fill(
            timestamp_ns=stamp,
            order_id=str(stamp),
            quantity=quantity,
            price=100.0,
            mark_price=100.0,
            fee=0.0,
        )
        assert account.quantity == expected
        assert account._cost_basis == expected * 100
        assert account.mark(stamp, 100.0)["net_pnl"] == 0.0
    closing_batch = next(batch for batch in reversed(batches) if batch.trades)
    assert abs(expected) == 1
    account.funding(FundingEvent(4, 0.01, 100.0))
    funding = float(-expected * 100 * Fraction(0.01))
    assert account.funding_cashflows == funding
    assert account.cash == 1000.0 + funding
    result = account.mark(5, 101.0)
    assert result["net_pnl"] == float(expected + Fraction(funding))

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    artifact = read_portfolio_ledger_artifact(root)
    persisted = list(artifact.iter_chunks())
    final = next(
        value
        for value in persisted[-1].valuations
        if value.asset_id == "asset" and value.phase == "end"
    )
    assert final.quantity == expected
    assert isinstance(final.quantity, Fraction)
    summary = summarize_portfolio_ledger(capital, iter(persisted))
    assert summary["net_pnl"] == result["net_pnl"]
    assert summary["funding"] == funding
    assert summary["nav"] == result["nav"]

    # The stored-term rounding budget must still reject a larger economic mismatch.
    corrupted = replace(
        closing_batch,
        trades=tuple(
            replace(trade, notional=trade.notional + 1024.0)
            for trade in closing_batch.trades
        ),
    )
    previous = {
        value.asset_id: value
        for value in closing_batch.valuations
        if value.phase == "before"
    }
    with pytest.raises(ValueError, match="portfolio_ledger_cost_basis_invalid"):
        validate_portfolio_transition(corrupted, capital, previous)


def test_perpetual_cost_basis_transition_sums_stored_terms_before_rounding():
    capital = PortfolioCapital("exact_sum", 1000.0, "USDT", "linear_perpetual")
    batches = []
    account = PortfolioEconomics(capital, "asset", batches.append)
    account.fill(
        timestamp_ns=0,
        order_id="open",
        quantity=1e16,
        price=100.0,
        mark_price=100.0,
        fee=0.0,
    )
    account.fill(
        timestamp_ns=1,
        order_id="small",
        quantity=1.0,
        price=100.0,
        mark_price=100.0,
        fee=0.0,
    )
    previous = None
    for batch in batches:
        previous = validate_portfolio_transition(batch, capital, previous)
    assert previous["asset"].quantity == 10**16 + 1
    assert previous["asset"].value == 0
    assert previous["asset"].cost_basis == float(10**18 + 100)


def test_funding_uses_exact_position_before_rounding_cashflow():
    capital = PortfolioCapital("exact_funding", 1000.0, "USDT", "linear_perpetual")
    batches = []
    account = PortfolioEconomics(capital, "asset", batches.append)
    for stamp, quantity in enumerate((1e16, 1.0)):
        account.fill(
            timestamp_ns=stamp,
            order_id=str(stamp),
            quantity=quantity,
            price=100.0,
            mark_price=100.0,
            fee=0.0,
        )
    account.funding(FundingEvent(2, 1e-16, 100.0))
    expected = float(-Fraction(10**16 + 1) * 100 * Fraction(1e-16))
    assert expected != -100.0
    assert batches[-1].cashflows[0].amount == expected
    assert account.mark(3, 100.0)["funding"] == expected
