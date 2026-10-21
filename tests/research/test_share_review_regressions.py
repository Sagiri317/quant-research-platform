"""Independent synthetic review counterexamples, without private source rows."""

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


def test_interval_gross_pnl_accounts_for_each_fee():
    price, capital = 100.0, 10000.0
    end_price = 102.0
    post_cash = capital - 3 * price - 10 - 1
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
        PortfolioCashFlow("fee", 1, "a", "fill", "commission_cost", -10),
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
    assert economics.summary()["gross_pnl"] == pytest.approx(6.0)
    assert economics.summary()["net_pnl"] == pytest.approx(-5.0)
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


@pytest.mark.parametrize("locked", [Fraction(), Fraction(1, 10**17)])
def test_rounded_zero_notional_preserves_locked_distribution(tmp_path, locked):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        ExecutionAssumptions,
        TradeCostComponents,
    )
    from quant_research_platform.research.portfolios import production
    from tests.helpers.portfolio_ledger import execute_continuous_target

    # A tiny earlier distribution can remain locked after a larger purchase.
    quantity = Fraction(1) + Fraction(1, 10**16)
    units = pd.Series({"a": quantity}, dtype=object)
    entry, endpoint = pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")
    life = pd.Timestamp("2000-01-01")
    account = execute_continuous_target(
        pd.Series({"a": 1.0}),
        pd.Series({"a": 0.0}),
        nav=1.0,
        average_daily_volume=pd.Series({"a": 100.0}),
        can_enter=pd.Series({"a": True}),
        can_exit=pd.Series({"a": True}),
        execution_date=entry.date(),
        cost_for_trade=lambda _date, _asset, _side, amount, _volume, _settings: (
            TradeCostComponents(float(amount), 0.0, 0.0, 0.0, 0.0, 0.0, "f" * 64)
        ),
        assumptions=ExecutionAssumptions(1.0, 0.0, 0.0, 1.0, (1.0,)),
        cost_source_identity="f" * 64,
        locked_notional=pd.Series({"a": float(locked)}),
        share_quantities=units,
        locked_share_quantities=pd.Series({"a": locked}, dtype=object),
        execution_prices=pd.Series({"a": 1.0}),
    )
    assert account.asset_ledger.post_trade_notional.iloc[0] == 0.0
    assert account.asset_ledger.executed_delta_notional.iloc[0] == -1.0
    marks = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": [entry],
            "open": [1.0],
            "listing_date": [life],
            "research_open_mark_at": [pd.NaT],
        }
    )
    prior = PortfolioValuation(
        entry.value,
        "a",
        "end",
        quantity,
        1.0,
        1.0,
        lifecycle_id="2000-01-01",
        locked_quantity=locked,
    )
    entitlements = production.EquityEntitlements()
    entitlements.locked_quantities["a"] = locked
    batch = production._ledger_batch(
        account=account,
        asset_ledger=account.asset_ledger,
        sequence=0,
        end_timestamp=endpoint,
        decision_timestamp=entry - pd.Timedelta(days=1),
        held_units=units,
        execution_units=units,
        previous_values={"a": prior},
        rights_adjustments=(),
        prediction_rank_score=pd.Series({"a": 0.0}),
        adv20=pd.Series({"a": 100.0}),
        enhanced_decision=None,
        benchmark_weights=None,
        optimization_result=None,
        decision_reason="simple_rule",
        decision_delta=pd.Series({"a": -1.0}),
        next_panel=marks,
        endpoint_panel=marks.assign(timestamp=endpoint),
        endpoint_events=pd.DataFrame(),
        corporate_events=pd.DataFrame(),
        endpoint_corporate_events=pd.DataFrame(),
        entitlements=entitlements,
        execution_lives=pd.Series({"a": life}),
    )
    expected = Fraction(1, 10**16) if locked else Fraction()
    for value in batch.valuations:
        if value.asset_id == "a" and value.phase in {"after", "end"}:
            assert value.quantity == expected
            assert value.locked_quantity == locked
    assert batch.trades[0].quantity == expected - quantity
    assert batch.trades[0].notional == -1.0
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("locked", 1.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2024-01-03", "end": "2024-01-04"}},
        produce=lambda emit: emit(batch),
    )
    assert tuple(read_portfolio_ledger_artifact(root).iter_chunks()) == (batch,)


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
