"""Synthetic corporate-action economics through the producer and persisted ledger."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from tests.helpers.portfolio_inputs import portfolio_inputs as _inputs
from tests.research.test_portfolio_production import _build_producer


def _distribution_batches(
    tmp_path,
    monkeypatch,
    *,
    future_close=90.0,
    payment="2023-01-05",
    stock_distribution=False,
    relist=False,
    recovery=None,
    suspended=False,
    cash_dividend=10.0,
    freeze_ex_date=False,
):
    frames, fees, window = _inputs()
    if stock_distribution:
        window = TimeWindow("2023-01-03", "2023-01-09")
    if relist or recovery or suspended or freeze_ex_date:
        window = TimeWindow("2023-01-03", "2023-01-10")
    fees["rate"] = 0.0
    panel = frames["panel"]
    panel.loc[:, ["open", "close", "adj_open", "adj_close"]] = 100.0
    ex_date = pd.Timestamp("2023-01-05")
    affected = panel.asset_id.eq("b") & panel.timestamp.ge(ex_date)
    ex_price = (100.0 - cash_dividend) / (1.5 if stock_distribution else 1.0)
    panel.loc[affected, ["open", "close"]] = ex_price
    if not stock_distribution:
        # Same prior history, raw opening quote and distribution; only the later
        # close and close-dependent adjusted quote differ between the two worlds.
        panel.loc[panel.asset_id.eq("b") & panel.timestamp.eq(ex_date), "close"] = (
            future_close
        )
        panel.loc[affected, "adj_open"] = 100.0 if future_close == 90.0 else 99.0
    frames["dataset"].loc[
        frames["dataset"].timestamp.ge("2023-01-06" if recovery else "2023-01-04"),
        "eligible",
    ] = False
    event = {
        "source_asset_id": "b",
        "timestamp": ex_date,
        "kind": "CA",
        "market_type": 1,
        "announcement_date": pd.Timestamp("2023-01-02"),
        "settlement_date": pd.Timestamp(payment) if payment else pd.NaT,
        "amount": cash_dividend,
    }
    events = [event]
    if stock_distribution:
        events.append(
            dict(
                event, kind="SD", amount=0.5, settlement_date=pd.Timestamp("2023-01-06")
            )
        )
    frames["corporate_actions"] = pd.DataFrame(events)
    if relist:
        first = panel.timestamp.min()
        frames["stock_master"] = pd.DataFrame(
            [
                dict(
                    source_asset_id="a",
                    timestamp=first,
                    listing_date=first,
                    delisting_date=pd.NaT,
                ),
                dict(
                    source_asset_id="b",
                    timestamp=first,
                    listing_date=first,
                    delisting_date=pd.Timestamp("2023-01-05"),
                ),
                dict(
                    source_asset_id="b",
                    timestamp=pd.Timestamp("2023-01-06"),
                    listing_date=pd.Timestamp("2023-01-06"),
                    delisting_date=pd.NaT,
                ),
            ]
        )
        panel.loc[
            panel.asset_id.eq("b") & panel.timestamp.ge("2023-01-06"), "listing_date"
        ] = pd.Timestamp("2023-01-06")
    if recovery:
        from quant_research_platform.sources.reference import interim
        from quant_research_platform.storage.datasets import read_dataset
        from tests.sources.reference.test_equity_events import write_equity_source

        raw = tmp_path / "raw"
        write_equity_source(
            raw,
            events=lambda base: [
                dict(
                    base,
                    event_id="terminal",
                    asset_id="000001",
                    listing_date="2022-11-14",
                    kind="cash_recovery" if recovery == "cash" else "cancel",
                    effective_date="2023-01-06",
                    publication_date="2023-01-05",
                    amount_per_share=90.0 if recovery == "cash" else None,
                    remaining_old_units_per_old_unit=0.0,
                )
            ],
        )
        result = interim.materialize_interim_data(
            output_roots={"a_share_equity_events": tmp_path / "rights"},
            raw_root=raw,
            data_ids=("a_share_equity_events",),
        )
        frames["equity_events"] = read_dataset(
            result.artifact_roots["a_share_equity_events"]
        )
        # The test universe uses anonymous a/b identifiers after source admission.
        frames["equity_events"]["asset_id"] = "b"
        first = panel.timestamp.min()
        frames["stock_master"] = pd.DataFrame(
            [
                dict(
                    source_asset_id="a",
                    timestamp=first,
                    listing_date=first,
                    delisting_date=pd.NaT,
                ),
                dict(
                    source_asset_id="b",
                    timestamp=first,
                    listing_date=first,
                    delisting_date=pd.Timestamp("2023-01-05"),
                ),
            ]
        )
    if suspended:
        paused = panel.asset_id.eq("b") & panel.timestamp.eq(ex_date)
        panel.loc[paused, ["open", "close"]] = float("nan")
        panel.loc[paused, ["amount", "volume"]] = 0.0
        panel.loc[paused, "is_trading"] = False
    if suspended or freeze_ex_date:
        unavailable = frames["execution"].asset_id.eq("b") & frames[
            "execution"
        ].timestamp.eq("2023-01-04")
        frames["execution"].loc[unavailable, ["can_enter", "can_exit"]] = False
    chunks, _ = _build_producer(
        tmp_path / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        typed_batches=True,
    )
    batches = list(chunks)
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("raw_shares", 100.0, "CNY", "cash_equity"),
        metadata={
            "window": {"start": "2023-01-03", "end": str(window.end)},
            "sample_role": "validation",
            "corporate_action_policy": portfolio_sessions.CORPORATE_ACTION_POLICY,
        },
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    return list(read_portfolio_ledger_artifact(root).iter_chunks())


def _cash(batch, phase):
    return next(
        row.value
        for row in batch.valuations
        if row.asset_id == "__cash__" and row.phase == phase
    )


def test_future_close_cannot_change_raw_open_sale_or_final_cash(tmp_path, monkeypatch):
    outcomes = []
    for close in (90.0, 100.0):
        batches = _distribution_batches(
            tmp_path / str(close), monkeypatch, future_close=close
        )
        sales = [
            trade
            for batch in batches
            for trade in batch.trades
            if trade.asset_id == "b" and trade.quantity < 0
        ]
        assert len(sales) == 1
        assert float(sales[0].quantity) == pytest.approx(-1.0)
        assert sales[0].notional == pytest.approx(-90.0)
        assert sales[0].timestamp_ns == pd.Timestamp("2023-01-05").value
        assert sum(
            flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "dividend"
        ) == pytest.approx(10.0)
        outcomes.append((sales[0].notional, _cash(batches[-1], "end")))
    assert outcomes[0] == pytest.approx(outcomes[1])
    assert outcomes[0][1] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("payment", "delayed"),
    [("2023-01-06", True), (None, False), ("2023-01-04", False)],
    ids=["delayed-payment", "missing-ex-date-assumption", "before-ex-date-assumption"],
)
def test_dividend_receivable_survives_sale_and_only_settlement_is_cash(
    tmp_path, monkeypatch, payment, delayed
):
    batches = _distribution_batches(tmp_path, monkeypatch, payment=payment)
    sale_batch = next(
        batch
        for batch in batches
        if any(trade.asset_id == "b" and trade.quantity < 0 for trade in batch.trades)
    )
    after = [row for row in sale_batch.valuations if row.phase == "after"]
    assert sum(row.value for row in after) == pytest.approx(100.0)
    assert _cash(sale_batch, "after") == pytest.approx(90.0 if delayed else 100.0)
    if delayed:
        claim = next(row for row in after if row.asset_id == "b")
        assert float(claim.quantity) == 0.0
        assert claim.receivable == pytest.approx(10.0)
        assert claim.value == pytest.approx(10.0)
    assert _cash(batches[-1], "end") == pytest.approx(100.0)
    dividends = [
        flow for batch in batches for flow in batch.cashflows if flow.kind == "dividend"
    ]
    assert len(dividends) == 1
    assert dividends[0].amount == pytest.approx(10.0)
    assert (
        dividends[0].timestamp_ns
        == pd.Timestamp("2023-01-06" if delayed else "2023-01-05").value
    )
    artifact = read_portfolio_ledger_artifact(tmp_path / "ledger")
    policy = artifact.metadata["corporate_action_policy"]
    assert policy["before_ex_cash_payment_date"] == "ex_date_open"


def test_cash_and_stock_distribution_share_record_base_and_stock_stays_locked(
    tmp_path, monkeypatch
):
    batches = _distribution_batches(tmp_path, monkeypatch, stock_distribution=True)
    ex = pd.Timestamp("2023-01-05").value
    sale_batch = next(
        batch
        for batch in batches
        if any(
            trade.timestamp_ns == ex and trade.asset_id == "b" and trade.quantity < 0
            for trade in batch.trades
        )
    )
    sale = next(
        trade
        for trade in sale_batch.trades
        if trade.asset_id == "b" and trade.quantity < 0
    )
    assert float(sale.quantity) == pytest.approx(-1.0)
    assert sale.notional == pytest.approx(-60.0)
    after = next(
        row
        for row in sale_batch.valuations
        if row.phase == "after" and row.asset_id == "b"
    )
    assert float(after.quantity) == pytest.approx(0.5)
    assert after.value == pytest.approx(30.0)
    assert _cash(sale_batch, "after") == pytest.approx(70.0)
    assert sum(
        row.value for row in sale_batch.valuations if row.phase == "after"
    ) == pytest.approx(100.0)
    released_sales = [
        trade
        for batch in batches
        for trade in batch.trades
        if trade.asset_id == "b"
        and trade.timestamp_ns == pd.Timestamp("2023-01-06").value
        and trade.quantity < 0
    ]
    assert len(released_sales) == 1
    assert float(released_sales[0].quantity) == pytest.approx(-0.5)
    assert released_sales[0].notional == pytest.approx(-30.0)
    assert _cash(batches[-1], "end") == pytest.approx(100.0)


def test_old_lifecycle_dividend_receivable_survives_relisting_until_payment(
    tmp_path, monkeypatch
):
    batches = _distribution_batches(
        tmp_path, monkeypatch, payment="2023-01-09", relist=True
    )
    claims = [
        row
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "b"
        and row.receivable > 0
        and row.timestamp_ns >= pd.Timestamp("2023-01-06").value
    ]
    assert claims
    assert all(float(row.quantity) == 0.0 for row in claims)
    assert all(row.lifecycle_id == "2022-11-14" for row in claims)
    assert sum(
        flow.amount
        for batch in batches
        for flow in batch.cashflows
        if flow.kind == "dividend"
    ) == pytest.approx(10.0)
    assert _cash(batches[-1], "end") == pytest.approx(100.0)


@pytest.mark.parametrize("recovery, expected_cash", [("cash", 100.0), ("cancel", 10.0)])
def test_registered_dividend_is_not_written_off_or_counted_twice_with_terminal_event(
    tmp_path, monkeypatch, recovery, expected_cash
):
    batches = _distribution_batches(
        tmp_path, monkeypatch, payment="2023-01-09", recovery=recovery
    )
    assert sum(
        flow.amount
        for batch in batches
        for flow in batch.cashflows
        if flow.kind == "dividend"
    ) == pytest.approx(10.0)
    assert _cash(batches[-1], "end") == pytest.approx(expected_cash)
    assert sum(
        row.value for row in batches[-1].valuations if row.phase == "end"
    ) == pytest.approx(expected_cash)


@pytest.mark.parametrize(
    "cash_dividend, stock_distribution, expected_price",
    [(10.0, False, 90.0), (0.0, True, 100.0 / 1.5), (10.0, True, 60.0)],
)
def test_suspended_ex_date_transforms_carried_mark_once_without_opening_trade(
    tmp_path, monkeypatch, cash_dividend, stock_distribution, expected_price
):
    batches = _distribution_batches(
        tmp_path,
        monkeypatch,
        payment="2023-01-09",
        suspended=True,
        stock_distribution=stock_distribution,
        cash_dividend=cash_dividend,
    )
    ex = pd.Timestamp("2023-01-05").value
    opening = [
        row
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "b" and row.timestamp_ns == ex and row.phase == "after"
    ]
    assert len(opening) == 1
    value = opening[0]
    assert float(value.mark_price) == pytest.approx(expected_price)
    assert float(value.quantity) == pytest.approx(1.5 if stock_distribution else 1.0)
    assert value.receivable == pytest.approx(cash_dividend)
    assert value.value == pytest.approx(100.0)
    assert not any(
        trade.asset_id == "b" and trade.timestamp_ns == ex
        for batch in batches
        for trade in batch.trades
    )
    assert _cash(batches[-1], "end") == pytest.approx(100.0)
    quoted = _distribution_batches(
        tmp_path / "quoted",
        monkeypatch,
        payment="2023-01-09",
        freeze_ex_date=True,
        stock_distribution=stock_distribution,
        cash_dividend=cash_dividend,
    )
    quoted_open = next(
        row
        for batch in quoted
        for row in batch.valuations
        if row.asset_id == "b" and row.timestamp_ns == ex and row.phase == "after"
    )
    assert (float(value.quantity), value.value, value.receivable) == pytest.approx(
        (float(quoted_open.quantity), quoted_open.value, quoted_open.receivable)
    )
    assert _cash(batches[-1], "end") == pytest.approx(_cash(quoted[-1], "end"))


@pytest.mark.parametrize(
    "quote_date, quote_price, expected",
    [("2023-01-04", 100.0, 90.0), ("2023-01-05", 95.0, 95.0)],
)
def test_stale_reference_quote_adjusts_once_but_current_quote_is_not_readjusted(
    quote_date, quote_price, expected
):
    from fractions import Fraction

    from quant_research_platform.research.portfolios import production
    from tests.helpers.portfolio_ledger import mark_view

    marks = production._ResearchValuationMarks()
    marks.project(mark_view("2023-01-04", 100.0, 100.0))
    ex_view = mark_view("2023-01-05", float("nan"), float("nan"), trading=False)
    ex_view.corporate_events = pd.DataFrame(
        [
            dict(
                asset_id="a",
                listing_date=pd.Timestamp("2020-01-01"),
                effective_at=pd.Timestamp("2023-01-05"),
                available_at=pd.Timestamp("2023-01-03"),
                kind="cash_dividend",
                event_id="cash-dividend",
                cash_per_share=Fraction(10),
                entitlement_proved=True,
                evidence_identity="d" * 64,
            )
        ]
    )
    ex_view.rights_events = pd.DataFrame(
        [
            dict(
                asset_id="a",
                listing_date=pd.Timestamp("2020-01-01"),
                kind="quote",
                event_id="reference-quote",
                timestamp=pd.Timestamp(quote_date),
                effective_at=pd.Timestamp(quote_date),
                value_per_old_unit=Fraction(quote_price),
                entitlement_proved=True,
                evidence_identity="e" * 64,
            )
        ]
    )
    ex = marks.project(ex_view)
    later = marks.project(
        mark_view("2023-01-06", float("nan"), float("nan"), trading=False)
    )
    for view in (ex, later):
        assert float(view.open.iloc[0]) == pytest.approx(expected)
        assert float(view.close.iloc[0]) == pytest.approx(expected)
