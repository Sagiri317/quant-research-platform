"""Independent synthetic review counterexamples, without private source rows."""

import math
from fractions import Fraction

import pandas as pd
import pytest

from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.portfolios import (
    ExecutedTrade,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.research.portfolios.accounting import (
    portfolio_interval_economics,
)
from tests.research import test_raw_share_corporate_production as corporate


def test_interval_gross_pnl_cancels_each_fee_exactly():
    price, capital = 1e16, 1e18
    end_price = math.nextafter(price, math.inf)
    post_cash = capital - 3 * price - 1e17 - 1
    valuations = tuple(
        row
        for phase, at, quantity, mark, cash in (
            ("before", 1, 0, price, capital),
            ("after", 1, 3, price, post_cash),
            ("end", 2, 3, end_price, post_cash),
        )
        for row in (
            PortfolioValuation(
                at,
                "a",
                phase,
                Fraction(quantity),
                float(Fraction(quantity) * Fraction(mark)),
                mark,
            ),
            PortfolioValuation(at, "__cash__", phase, cash, cash),
        )
    )
    flows = (
        PortfolioCashFlow("principal", 1, "a", "fill", "principal", -3 * price),
        PortfolioCashFlow("fee", 1, "a", "fill", "commission_cost", -1e17),
        PortfolioCashFlow("slip", 1, "a", "fill", "slippage_cost", -1),
    )
    diagnostics = tuple(
        PortfolioDiagnostic(asset, None, name, number=value)
        for asset, target, delta in (
            ("a", 3 * price, 0.03),
            ("__cash__", capital - 3 * price, -0.03),
        )
        for name, value in (
            ("target_notional", target),
            ("decision_delta_weight", delta),
        )
    ) + (PortfolioDiagnostic("__cash__", None, "decision_reason", text="simple_rule"),)
    batch = PortfolioLedgerBatch(
        0,
        (ExecutedTrade("fill", "order", 1, "a", Fraction(3), 3 * price),),
        flows,
        valuations,
        diagnostics,
    )
    economics = _LedgerEconomics(PortfolioCapital("a", capital, "CNY", "cash_equity"))
    economics.consume(batch)
    assert economics.summary()["gross_pnl"] == 6.0
    assert portfolio_interval_economics(batch)[
        "gross_return"
    ] * capital == pytest.approx(6.0)


def test_unheld_unknown_action_does_not_poison_later_actual_quote(
    tmp_path, monkeypatch
):
    original = corporate._build_producer

    def build(*args, **kwargs):
        frames = kwargs["frames"]
        frames["dataset"]["eligible"] = frames["dataset"].timestamp.ge("2023-01-06")
        frames["corporate_actions"]["kind"] = "UNKNOWN"
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    batches = corporate._distribution_batches(tmp_path, monkeypatch, suspended=True)
    trades = [trade for batch in batches for trade in batch.trades]
    assert len(trades) == 1
    assert trades[0].asset_id == "b" and trades[0].quantity == Fraction(10, 9)
    assert pd.Timestamp(trades[0].timestamp_ns) == pd.Timestamp("2023-01-09")


@pytest.mark.parametrize("unproved", [False, True])
def test_late_disclosed_dividend_uses_actual_record_shares_after_sale(
    tmp_path, monkeypatch, unproved
):
    from dataclasses import replace

    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    original = corporate._build_producer

    def build(*args, **kwargs):
        events = kwargs["frames"]["corporate_actions"]
        events["announcement_date"] = pd.Timestamp("2023-01-05")
        if unproved:
            events["amount"] = float("nan")
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    if unproved:
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_held_rights_unproved"
        ):
            corporate._distribution_batches(tmp_path, monkeypatch, payment="2023-01-06")
        return
    batches = corporate._distribution_batches(
        tmp_path, monkeypatch, payment="2023-01-06"
    )
    assert batches[0].trades[0].quantity == 1
    assert not batches[0].rights_adjustments
    assert batches[1].trades[0].quantity == -1
    registration = next(
        event for event in batches[1].rights_adjustments if event.kind == "registration"
    )
    assert registration.entitlement_quantity == 1
    assert registration.quantity_before == 0
    assert registration.entitlement_record_at_ns == pd.Timestamp("2023-01-04").value
    assert registration.timestamp_ns == pd.Timestamp("2023-01-06").value
    assert (
        sum(
            flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "dividend"
        )
        == 10.0
    )
    assert corporate._cash(batches[-1], "end") == 100.0
    corrupted = replace(
        registration, entitlement_record_at_ns=registration.timestamp_ns + 1
    )
    with pytest.raises(ValueError, match="portfolio_registration_record_date_invalid"):
        replace(
            batches[1],
            rights_adjustments=tuple(
                corrupted if event is registration else event
                for event in batches[1].rights_adjustments
            ),
        )


def test_fractional_raw_price_never_oversells_locked_actual_shares(
    tmp_path, monkeypatch
):
    original = corporate._build_producer

    def build(*args, **kwargs):
        panel = kwargs["frames"]["panel"]
        panel.loc[
            panel.asset_id.eq("b") & panel.timestamp.ge("2023-01-05"), ["open", "close"]
        ] = 0.05
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    batches = corporate._distribution_batches(
        tmp_path, monkeypatch, cash_dividend=0, stock_distribution=True
    )
    assert [trade.quantity for batch in batches for trade in batch.trades] == [
        1,
        -1,
        Fraction(-1, 2),
    ]
    for batch in batches:
        for value in batch.valuations:
            if value.asset_id == "b":
                assert value.quantity >= value.locked_quantity


@pytest.mark.parametrize(
    "action", ["share_change", "cash_dividend", "late_earlier_split"]
)
def test_late_historical_quote_replays_known_actions_from_original_basis(action):
    from quant_research_platform.research.portfolios.production import (
        _ResearchValuationMarks,
    )
    from tests.helpers.portfolio_ledger import mark_view

    marks = _ResearchValuationMarks()
    base = dict(
        asset_id="a",
        listing_date=pd.Timestamp("2020-01-01"),
        entitlement_proved=True,
        evidence_identity="a" * 64,
        remaining_old_units_per_old_unit=Fraction(1),
    )
    quote = dict(
        base,
        event_id="quote1",
        kind="quote",
        effective_at=pd.Timestamp("2023-01-03"),
        available_at=pd.Timestamp("2023-01-04"),
        timestamp=pd.Timestamp("2023-01-04"),
        value_per_old_unit=Fraction(100),
    )

    def view(day, *, rights=(), events=()):
        result = mark_view(day, float("nan"), float("nan"), trading=False)
        result.rights_events = pd.DataFrame(rights)
        result.corporate_events = pd.DataFrame(events)
        return result

    assert marks.project(view("2023-01-04", rights=[quote])).open.iloc[0] == 100
    event = dict(
        base,
        event_id="action1",
        kind="share_change",
        effective_at=pd.Timestamp("2023-01-05"),
        available_at=pd.Timestamp("2023-01-05"),
        timestamp=pd.Timestamp("2023-01-05"),
        remaining_old_units_per_old_unit=Fraction(2),
    )
    if action == "cash_dividend":
        event.update(kind="cash_dividend", cash_per_share=Fraction(10))
        ex = marks.project(view("2023-01-05", events=[event]))
        expected = 90
    else:
        ex = marks.project(view("2023-01-05", rights=[event]))
        expected = 50
    assert ex.open.iloc[0] == expected
    if action == "late_earlier_split":
        late = dict(
            event,
            event_id="action2",
            effective_at=pd.Timestamp("2023-01-04"),
            available_at=pd.Timestamp("2023-01-06"),
            timestamp=pd.Timestamp("2023-01-06"),
        )
        expected = 25
    else:
        late = dict(
            quote,
            event_id="quote2",
            effective_at=pd.Timestamp("2023-01-04"),
            available_at=pd.Timestamp("2023-01-06"),
            timestamp=pd.Timestamp("2023-01-06"),
        )
    for result in (
        marks.project(view("2023-01-06", rights=[late])),
        marks.project(view("2023-01-07")),
    ):
        assert result.open.iloc[0] == expected
        assert result.close.iloc[0] == expected


@pytest.mark.parametrize("configured", [None, 19])
def test_native_paired_statistics_defaults_bootstrap_without_overriding_explicit(
    tmp_path, monkeypatch, configured
):
    from dataclasses import replace

    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.research.portfolios import diagnostics
    from tests.research.test_optional_statistics import _cash_ledger

    first = tmp_path / "first"
    second = tmp_path / "second"
    window = _cash_ledger(first)
    candidate = PortfolioCandidateSpec(
        "second", "global", "quantile_equal_weight", "explicit", "daily", 0.2, None
    )
    _cash_ledger(second, candidate=replace(candidate, candidate_id="second"))
    policy = {
        "hac_lag": 0,
        "bootstrap_block": 1,
        "group_comparison": {
            "portfolio_ids": ["cash", "second"],
            "primary_portfolio_id": "cash",
            "fdr_method": "BY",
            "alpha": 0.05,
            "seed": 1,
            "confidence": 0.95,
        },
    }
    if configured is not None:
        policy["bootstrap_repetitions"] = configured

    def inspect(**kwargs):
        assert kwargs["config"]["statistics"]["bootstrap_repetitions"] == (
            2000 if configured is None else configured
        )
        raise RuntimeError("policy_verified")

    monkeypatch.setattr(diagnostics, "build_portfolio_statistics_tables", inspect)
    with pytest.raises(RuntimeError, match="policy_verified"):
        diagnostics.portfolio_statistics(
            {
                "ledgers": [str(first), str(second)],
                "window": window,
                "output": str(tmp_path / "statistics"),
                "analyses": ["group_comparison"],
                "statistics": policy,
            }
        )
