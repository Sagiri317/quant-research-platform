"""Raw-share registration, receivable and economic ledger regressions."""

from dataclasses import replace
from fractions import Fraction

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.research.portfolios.accounting import (
    EquityEntitlements,
    portfolio_interval_economics,
)


def _events(*, pay="2024-01-05", stock=False):
    return pd.DataFrame(
        [
            dict(
                event_id="stock" if stock else "cash",
                asset_id="a",
                listing_date=pd.Timestamp("2000-01-01"),
                kind="stock_distribution" if stock else "cash_dividend",
                record_at=pd.Timestamp("2024-01-02"),
                effective_at=pd.Timestamp("2024-01-03"),
                settlement_at=pd.Timestamp(pay),
                available_at=pd.Timestamp("2024-01-01"),
                cash_per_share=0.0 if stock else 1.0,
                additional_shares_per_share=0.5 if stock else 0.0,
                evidence_identity="a" * 64,
                entitlement_proved=True,
            )
        ]
    )


def test_registered_cash_and_stock_use_original_shares_and_sold_shares_keep_dividend():
    state = EquityEntitlements()
    events = pd.concat([_events(), _events(stock=True)])
    shares = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    registration = state.register(events, shares, lives, pd.Timestamp("2024-01-02"))
    assert [row.entitlement_quantity for row in registration] == [100, 100]
    shares, lives, cash, adjustments = state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-03")
    )
    assert shares["a"] == 150
    assert cash == 0
    assert state.receivables["a"] == 100
    assert state.locked_quantities["a"] == 50
    assert sum(row.quantity_after - row.quantity_before for row in adjustments) == 50
    shares["a"] = 50  # Original 100 shares sold, distributed 50 remain locked.
    shares, lives, cash, _ = state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-05")
    )
    assert cash == 100
    assert state.receivables["a"] == 0
    assert state.locked_quantities["a"] == 0
    assert shares["a"] == 50
    assert state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-05")
    )[2:] == (0.0, ())


def test_dividend_payment_survives_zero_quantity_and_a_new_listing_lifecycle():
    state = EquityEntitlements()
    events = _events()
    shares = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    state.register(events, shares, lives, pd.Timestamp("2024-01-02"))
    state.process(events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-03"))
    for quantity in (0, 20):
        import copy

        sample = copy.deepcopy(state)
        shares["a"] = Fraction(quantity)
        lives["a"] = pd.Timestamp("2024-01-04")
        result, identities, cash, records = sample.process(
            events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-05")
        )
        assert cash == 100
        assert result["a"] == quantity
        assert identities["a"] == pd.Timestamp("2024-01-04")
        assert records[0].lifecycle_id == "2000-01-01"


def test_same_day_ex_cash_is_spendable_but_delayed_cash_is_a_receivable():
    for pay, expected in (("2024-01-03", 100), ("2024-01-05", 0)):
        state = EquityEntitlements()
        events = _events(pay=pay)
        shares = pd.Series({"a": Fraction(100)}, dtype=object)
        lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
        state.register(events, shares, lives, pd.Timestamp("2024-01-02"))
        _, _, cash, _ = state.process(
            events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-03")
        )
        assert cash == expected
        assert state.receivables["a"] == 100 - expected


def test_endpoint_dividend_ledger_roundtrip_has_no_double_count(tmp_path):
    state = EquityEntitlements()
    events = _events(pay="2024-01-03")
    shares = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    entry, end = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
    registration = state.register(events, shares, lives, entry)
    shares, lives, cash, changes = state.process(
        events, pd.DataFrame(), shares, lives, end, phase="end"
    )
    valuations = tuple(
        value
        for phase, at, mark, balance in (
            ("before", entry, 10, 0),
            ("after", entry, 10, 0),
            ("end", end, 9, cash),
        )
        for value in (
            PortfolioValuation(
                at.value,
                "a",
                phase,
                Fraction(100),
                100 * mark,
                mark,
                lifecycle_id="2000-01-01",
            ),
            PortfolioValuation(
                at.value, PORTFOLIO_CASH_ASSET_ID, phase, balance, balance
            ),
        )
    )
    diagnostics = tuple(
        PortfolioDiagnostic(asset, None, name, number=value)
        for asset, target in (("a", 1000), (PORTFOLIO_CASH_ASSET_ID, 0))
        for name, value in (("target_notional", target), ("decision_delta_weight", 0))
    ) + (
        PortfolioDiagnostic(
            PORTFOLIO_CASH_ASSET_ID, None, "decision_reason", text="simple_rule"
        ),
    )
    flows = tuple(
        PortfolioCashFlow(
            f"dividend:{row.event_id}",
            row.timestamp_ns,
            "a",
            None,
            "dividend",
            row.cash_amount,
        )
        for row in changes
        if row.cash_amount
    )
    batch = PortfolioLedgerBatch(
        0, (), flows, valuations, diagnostics, registration + changes
    )
    capital = PortfolioCapital("raw", 1000, "CNY", "cash_equity")
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "window": {
                "start": "2024-01-02",
                "end": "2024-01-04",
                "access": "validation",
            }
        },
        produce=lambda emit: emit(batch),
    )
    artifact = read_portfolio_ledger_artifact(root)
    reread = tuple(artifact.iter_chunks())
    assert reread == (batch,)
    summary = summarize_portfolio_ledger(capital, iter(reread))
    assert summary["nav"] == 1000
    assert summary["gross_pnl"] == 0
    assert portfolio_interval_economics(batch)["gross_return"] == 0
    with pytest.raises(ValueError, match="reconciliation"):
        replace(batch, cashflows=())
    corrupted = replace(
        batch,
        rights_adjustments=tuple(
            replace(row, entitlement_quantity=101) if row.kind == "dividend" else row
            for row in batch.rights_adjustments
        ),
    )
    with pytest.raises(ValueError, match="portfolio_registration_entitlement_invalid"):
        write_portfolio_ledger_artifact(
            tmp_path / "wrong_entitlement",
            capital=capital,
            metadata={
                "window": {
                    "start": "2024-01-02",
                    "end": "2024-01-04",
                    "access": "validation",
                }
            },
            produce=lambda emit: emit(corrupted),
        )


def test_execution_reserves_unpaid_cash_and_cannot_sell_locked_stock():
    from datetime import date

    from tests.helpers.portfolio_ledger import execute_continuous_target
    from tests.research.test_portfolio_ledger import (
        _a_share_cost,
        _assumptions,
        _schedule,
    )

    account = execute_continuous_target(
        pd.Series({"a": 0.9, "b": 0.0}),
        pd.Series({"a": 0.0, "b": 1.0}),
        nav=1000.0,
        receivable=100.0,
        locked_notional=pd.Series({"a": 300.0}),
        average_daily_volume=pd.Series({"a": 10000.0, "b": 10000.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        execution_date=date(2023, 8, 28),
        assumptions=_assumptions(),
        cost_for_trade=_a_share_cost(
            _schedule(transfer_rate=0.0, broker_rate=0.0, stamp_rate=0.0),
            {"a": "shanghai", "b": "shanghai"},
        ),
        cost_source_identity="0" * 64,
    )
    ledger = account.asset_ledger.set_index("asset_id")
    assert ledger.loc["a", "executed_delta_notional"] == -600
    assert ledger.loc["a", "reason"] == "locked_shares"
    assert ledger.loc["b", "executed_delta_notional"] == pytest.approx(600)
    assert ledger.loc["b", "reason"] == "cash_pro_rata"
    assert account.pre_trade_cash == 0
    assert account.post_trade_cash == pytest.approx(0, abs=1e-12)
    assert account.nav_after == pytest.approx(1000)


def test_distribution_without_unlock_date_stays_valued_and_locked():
    state = EquityEntitlements()
    events = _events(stock=True)
    events["settlement_at"] = pd.NaT
    shares = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    state.register(events, shares, lives, pd.Timestamp("2024-01-02"))
    shares, lives, _, _ = state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-03")
    )
    shares, _, cash, changes = state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-02-01")
    )
    assert shares["a"] == 150
    assert state.locked_quantities["a"] == 50
    assert cash == 0 and changes == ()


@pytest.mark.parametrize("mark", (1e-100, 1.0, 1e16, 1e100))
def test_raw_price_delta_pnl_is_not_difference_of_rounded_position_values(mark):
    import math

    from quant_research_platform.compute.portfolio_economics import _LedgerEconomics

    final_mark = math.nextafter(mark, math.inf)
    capital = PortfolioCapital("delta", 3 * mark, "CNY", "cash_equity")
    values = tuple(
        value
        for phase, price, timestamp in (
            ("before", mark, 1),
            ("after", mark, 1),
            ("end", final_mark, 2),
        )
        for value in (
            PortfolioValuation(timestamp, "a", phase, Fraction(3), 3 * price, price),
            PortfolioValuation(timestamp, PORTFOLIO_CASH_ASSET_ID, phase, 0, 0),
        )
    )
    batch = PortfolioLedgerBatch(0, (), (), values, ())
    economics = _LedgerEconomics(capital)
    economics.consume(batch)
    expected = 3 * (final_mark - mark)
    assert economics.summary()["gross_pnl"] == pytest.approx(expected, rel=1e-14, abs=0)
    if mark == 1e16:
        assert expected == 6
        assert 3 * final_mark - 3 * mark == 8


def test_small_dividend_claim_survives_a_large_same_asset_claim_settlement():
    state = EquityEntitlements()
    large = _events(pay="2024-01-04")
    large["cash_per_share"] = 1e14
    small = _events(pay="2024-01-05")
    small["event_id"] = "small"
    small["cash_per_share"] = 1e-4
    events = pd.concat([large, small])
    shares = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    state.register(events, shares, lives, pd.Timestamp("2024-01-02"))
    state.process(events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-03"))
    state.process(events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-04"))
    assert state.receivables["a"] == pytest.approx(0.01)
    _, _, cash, _ = state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-05")
    )
    assert cash == pytest.approx(0.01)
    assert state.receivables["a"] == 0


@pytest.mark.parametrize("recovery_per_share", (0.0, 9.0))
def test_registered_dividend_survives_share_cancellation_without_double_recovery(
    recovery_per_share,
):
    state = EquityEntitlements()
    events = _events()
    shares = pd.Series({"a": Fraction(100)}, dtype=object)
    lives = pd.Series({"a": pd.Timestamp("2000-01-01")})
    state.register(events, shares, lives, pd.Timestamp("2024-01-02"))
    shares, lives, cash, _ = state.process(
        events, pd.DataFrame(), shares, lives, pd.Timestamp("2024-01-03")
    )
    assert cash == 0 and state.receivables["a"] == 100
    recovery = pd.DataFrame(
        [
            dict(
                event_id="cancel",
                asset_id="a",
                listing_date=pd.Timestamp("2000-01-01"),
                kind="cash_recovery",
                remaining_old_units_per_old_unit=0.0,
                value_per_old_unit=recovery_per_share,
                successor_asset_id=None,
                successor_listing_date=pd.NaT,
                successor_units_per_old_unit=0.0,
                evidence_identity="b" * 64,
                entitlement_proved=True,
            )
        ]
    )
    shares, lives, cash, records = state.process(
        events, recovery, shares, lives, pd.Timestamp("2024-01-04")
    )
    assert shares["a"] == 0
    assert cash == 100 * recovery_per_share
    assert records[0].receivable_before == records[0].receivable_after == 100
    assert state.receivables["a"] == 100
    _, _, payment, _ = state.process(
        events, recovery, shares, lives, pd.Timestamp("2024-01-05")
    )
    assert payment == 100
    assert cash + payment == 100 * recovery_per_share + 100
    assert state.receivables["a"] == 0


def test_gross_raw_price_gain_survives_a_much_larger_cost():
    from quant_research_platform.compute.portfolio_economics import cash_equity_pnl

    costs = PortfolioCashFlow("fee", 1, "a", "fill", "commission_cost", -1e17)
    net = cash_equity_pnl(
        {"a": Fraction(3 * 10**16)}, {"a": Fraction(3 * 10**16 + 6)}, (costs,)
    )
    assert net + Fraction(1e17) == 6


def test_unquoted_carried_mark_adjusts_cash_and_stock_on_same_original_base():
    from quant_research_platform.research.portfolios.accounting import (
        adjust_carried_share_mark,
    )

    events = pd.concat([_events(), _events(stock=True)])
    mark, evidence = adjust_carried_share_mark(
        10.0,
        events,
        pd.DataFrame(),
        asset="a",
        listing=pd.Timestamp("2000-01-01"),
        timestamp=pd.Timestamp("2024-01-03"),
        mark_at=pd.Timestamp("2024-01-02"),
    )
    assert mark == 6
    assert evidence == ("a" * 64,)
    assert Fraction(150) * mark + 100 == 1000
    assert adjust_carried_share_mark(
        mark,
        events,
        pd.DataFrame(),
        asset="a",
        listing=pd.Timestamp("2000-01-01"),
        timestamp=pd.Timestamp("2024-01-04"),
        mark_at=pd.Timestamp("2024-01-03"),
    ) == (mark, ())
