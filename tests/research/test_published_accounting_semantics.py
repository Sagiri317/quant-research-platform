"""Native raw-share valuation and persisted receivable regression evidence."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts import portfolio_ledger as ledger
from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioCapital,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.accounting import (
    portfolio_interval_economics,
)
from quant_research_platform.research.portfolios.diagnostics import (
    _portfolio_statistics_tables,
    _research_valuation_observation,
)
from quant_research_platform.research.portfolios.production import (
    _ResearchValuationMarks,
)
from tests.helpers.portfolio_ledger import mark_view
from tests.research import test_raw_share_corporate_production as corporate


def _statistics(root, *, analyses=("performance",), holding_inputs=None):
    candidate = PortfolioCandidateSpec(
        "raw_shares", "global", "quantile_equal_weight", "synthetic", "daily", 0.5, None
    )
    return _portfolio_statistics_tables(
        ledgers={"raw_shares": ledger.read_portfolio_ledger_artifact(root)},
        candidates=(candidate,),
        config={"sample_role": "validation", "analyses": list(analyses)},
        time_window=TimeWindow("2023-01-03", "2023-01-10"),
        benchmark_opens=pd.DataFrame(),
        calendar=None,
        unified_trading_sessions=None,
        holding_inputs={} if holding_inputs is None else holding_inputs,
    )


@pytest.mark.parametrize("stock_distribution", [False, True])
def test_prepared_nonempty_suspension_carry_replays_raw_share_actions(
    tmp_path, monkeypatch, stock_distribution
):
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_sources import (
        _prepare,
        _synthetic_interim_frames,
        _write_inputs,
    )

    dates = pd.bdate_range("2022-11-14", "2023-01-10").as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    bars = frames["market_bars_1d"]
    prices = ["open", "high", "low", "close", "adj_open", "adj_close"]
    bars.loc[:, prices] = 100.0
    affected = bars.source_asset_id.eq("000002") & bars.timestamp.ge("2023-01-05")
    bars.loc[affected, ["open", "high", "low", "close"]] = (
        60.0 if stock_distribution else 90.0
    )
    paused = bars.source_asset_id.eq("000002") & bars.timestamp.isin(
        pd.to_datetime(["2023-01-05", "2023-01-06"])
    )
    bars.loc[paused, "filling_flag"] = 1
    bars.loc[paused, prices] = np.nan
    bars.loc[paused, ["volume", "amount"]] = 0.0
    roots = _write_inputs(tmp_path / "source", frames)
    panel = read_dataset(
        _prepare(tmp_path / "prepared", "market_panel_1d", roots, dates[0], dates[-1])
    )
    paused_panel = panel.loc[
        panel.asset_id.eq("000002")
        & panel.timestamp.isin(pd.to_datetime(["2023-01-05", "2023-01-06"]))
    ]
    assert paused_panel.open.tolist() == paused_panel.close.tolist() == [100.0, 100.0]
    assert not paused_panel.is_trading.any()
    selected = panel.loc[panel.asset_id.isin(["000001", "000002"])].copy()
    selected["asset_id"] = selected.asset_id.map({"000001": "a", "000002": "b"})
    prepared = selected.set_index(["asset_id", "timestamp"])
    original = corporate._build_producer

    def build(*args, **kwargs):
        inputs = kwargs["frames"]
        target = inputs["panel"].set_index(["asset_id", "timestamp"])
        for name in ("open", "close", "adj_open", "adj_close", "is_trading"):
            target[name] = prepared[name].reindex(target.index)
        inputs["panel"] = target.reset_index()
        # Both suspended execution dates are closed to stock orders.
        masks = inputs["execution"]
        masks.loc[
            masks.asset_id.eq("b")
            & masks.timestamp.isin(pd.to_datetime(["2023-01-04", "2023-01-05"])),
            ["can_enter", "can_exit"],
        ] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    batches = corporate._distribution_batches(
        tmp_path / "account",
        monkeypatch,
        payment="2023-01-09",
        suspended=True,
        stock_distribution=stock_distribution,
    )
    for batch in batches:
        economics = portfolio_interval_economics(batch)
        assert economics["nav_end"] == pytest.approx(100.0)
        assert economics["net_return"] == pytest.approx(0.0, abs=1e-14)
        for value in batch.valuations:
            if value.asset_id == "b" and value.phase == "after" and value.quantity:
                if pd.Timestamp(value.timestamp_ns) >= pd.Timestamp("2023-01-05"):
                    assert value.share_value == pytest.approx(90.0)
                    assert value.receivable == 10.0
    sales = [trade for batch in batches for trade in batch.trades if trade.quantity < 0]
    assert len(sales) == 1
    assert pd.Timestamp(sales[0].timestamp_ns) == pd.Timestamp("2023-01-09")
    assert sales[0].notional == pytest.approx(-90.0)
    assert corporate._cash(batches[-1], "end") == pytest.approx(100.0)


def test_raw_quote_anchor_replays_late_earlier_and_consecutive_actions_once():
    marks = _ResearchValuationMarks()
    marks.project(mark_view("2023-01-03", 100.0, 100.0))
    base = dict(
        asset_id="a",
        listing_date=pd.Timestamp("2020-01-01"),
        entitlement_proved=True,
        evidence_identity="a" * 64,
        remaining_old_units_per_old_unit=Fraction(2),
        kind="share_change",
    )

    def event(name, effective, available):
        return dict(
            base,
            event_id=name,
            effective_at=pd.Timestamp(effective),
            available_at=pd.Timestamp(available),
            timestamp=pd.Timestamp(available),
        )

    for day, actions, expected in (
        ("2023-01-04", [], 100),
        ("2023-01-05", [event("first", "2023-01-05", "2023-01-05")], 50),
        ("2023-01-06", [event("late", "2023-01-04", "2023-01-06")], 25),
        ("2023-01-09", [event("next", "2023-01-09", "2023-01-09")], 12.5),
        ("2023-01-10", [], 12.5),
    ):
        view = mark_view(day, np.nan, np.nan)
        view.rights_events = pd.DataFrame(actions)
        output = marks.project(view)
        assert output.open.iloc[0] == output.close.iloc[0] == expected
        assert output.research_open_mark_at.iloc[0] == pd.Timestamp("2023-01-03")
        assert marks.last.mark_price.iloc[0] == 100.0
    # A genuine quote can have exactly the original or estimated price. The
    # observed market fact, never a comparison of prices, replaces the anchor.
    restored = marks.project(mark_view("2023-01-11", 100.0, 100.0))
    assert restored.open.iloc[0] == 100.0
    assert restored.research_open_mark_at.isna().all()
    carried = marks.project(mark_view("2023-01-12", np.nan, np.nan))
    assert carried.open.iloc[0] == 100.0
    assert carried.research_open_mark_at.iloc[0] == pd.Timestamp("2023-01-11")


def _late_distribution_batches(tmp_path, monkeypatch, *, late_amount=10.0):
    original = corporate._build_producer

    def build(*args, **kwargs):
        frames = kwargs["frames"]
        panel, actions = frames["panel"], frames["corporate_actions"]
        actions["announcement_date"] = pd.Timestamp("2023-01-08")
        actions["settlement_date"] = pd.Timestamp("2023-01-10")
        known = actions.copy()
        actions["amount"] = late_amount
        known["timestamp"] = pd.Timestamp("2023-01-06")
        known["announcement_date"] = pd.Timestamp("2023-01-04")
        frames["corporate_actions"] = pd.concat([actions, known], ignore_index=True)
        missing = panel.asset_id.eq("b") & panel.timestamp.isin(
            pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-09"])
        )
        panel.loc[missing, ["open", "close"]] = np.nan
        panel.loc[missing, "is_trading"] = True
        panel.loc[
            panel.asset_id.eq("b") & panel.timestamp.eq("2023-01-10"),
            ["open", "close"],
        ] = 90.0 - late_amount
        return original(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(corporate, "_build_producer", build)
        return corporate._distribution_batches(
            tmp_path, scoped, payment="2023-01-10", freeze_ex_date=True
        )


def test_late_corporate_disclosure_preserves_prefix_and_persisted_nav(
    tmp_path, monkeypatch
):
    first = _late_distribution_batches(tmp_path / "first", monkeypatch)
    changed = _late_distribution_batches(
        tmp_path / "changed", monkeypatch, late_amount=20
    )
    # The later disclosure's amount cannot change either economic facts or
    # valuation diagnostics published before its availability.
    assert first[:2] == changed[:2]
    for batches, final_mark in ((first, 80), (changed, 70)):
        for batch in batches:
            economics = portfolio_interval_economics(batch)
            assert economics["nav_end"] == pytest.approx(100.0)
            assert economics["net_return"] == pytest.approx(0.0, abs=1e-14)
        row = next(
            row
            for row in batches[-1].valuations
            if row.asset_id == "b" and row.phase == "end"
        )
        assert row.mark_price == final_mark
        assert row.receivable == 0


def test_unquoted_stresses_and_estimated_amount_preserve_confirmed_receivable(
    tmp_path, monkeypatch
):
    batches = _late_distribution_batches(tmp_path, monkeypatch)
    batch = batches[1]
    economics = portfolio_interval_economics(batch)
    observed = _research_valuation_observation(batch, economics)
    assert observed["estimated_value_end"] == 90.0
    assert observed["estimated_weight_end"] == 0.9
    assert [observed[f"nav_{pct}pct"] for pct in (0, 50, 100)] == [10.0, 55.0, 100.0]
    # Also consume the actual persisted native ledger with its full interval axis.
    tables = _statistics(tmp_path / "ledger")
    assert not tables["portfolio_performance"].empty


def test_adv_missing_sell_amount_and_requested_turnover_exclude_receivable(
    tmp_path, monkeypatch
):
    original = corporate._build_producer

    def build(*args, **kwargs):
        panel = kwargs["frames"]["panel"]
        panel.loc[
            panel.asset_id.eq("b") & panel.timestamp.eq("2023-01-04"), "amount"
        ] = np.nan
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    batches = corporate._distribution_batches(
        tmp_path, monkeypatch, payment="2023-01-09", freeze_ex_date=True
    )
    amounts = []
    for batch in batches:
        facts = {(row.asset_id, row.name): row for row in batch.diagnostics}
        for value in batch.valuations:
            if value.phase == "after" and facts.get((value.asset_id, "block_reason")):
                if facts[value.asset_id, "block_reason"].text == "adv_unavailable":
                    assert value.share_value == 90.0
                    amounts.append(value.share_value)
                    assert (
                        portfolio_interval_economics(batch)["requested_turnover"] == 0.9
                    )
    assert amounts == [90.0, 90.0]
    metrics = _statistics(tmp_path / "ledger")["portfolio_performance"].set_index(
        "metric_id"
    )
    assert metrics.loc["adv_unavailable_sell_target_notional", "value"] == 180.0
    assert metrics.loc["adv_unavailable_buy_orders", "value"] == 0


@pytest.mark.parametrize("direction", ["remove", "add", "relabel"])
def test_native_writer_and_reader_enforce_zero_share_receivable_continuity(
    tmp_path, monkeypatch, direction
):
    batches = corporate._distribution_batches(
        tmp_path / "good", monkeypatch, payment="2023-01-09", relist=True
    )
    old = batches[2]
    cash_only = replace(
        old,
        cashflows=(),
        rights_adjustments=(),
        valuations=tuple(
            replace(row, quantity=90.0, value=90.0)
            for row in old.valuations
            if row.asset_id == "__cash__"
        ),
        diagnostics=tuple(
            replace(row, number=90.0) if row.name == "target_notional" else row
            for row in old.diagnostics
            if row.asset_id == "__cash__"
        ),
    )
    previous = batches[:2]
    if direction == "remove":
        bad = cash_only
    else:
        # This claim-only state has no payment or other state change. In the
        # addition case it is attached to a previously absent asset identifier.
        bad = replace(
            cash_only,
            valuations=cash_only.valuations
            + tuple(
                replace(
                    row,
                    asset_id="ghost" if direction == "add" else row.asset_id,
                    receivable=10.0,
                    value=10.0,
                    quantity=Fraction(),
                    mark_price=0.0,
                    lifecycle_id="2023-01-06"
                    if direction == "relabel"
                    else row.lifecycle_id,
                )
                for row in old.valuations
                if row.asset_id == "b"
            ),
            diagnostics=cash_only.diagnostics
            + tuple(
                replace(row, asset_id="ghost" if direction == "add" else row.asset_id)
                for row in old.diagnostics
                if row.asset_id == "b"
            ),
        )
    if direction == "add":
        bad = replace(
            bad,
            valuations=bad.valuations
            + tuple(
                replace(
                    row,
                    receivable=10.0,
                    value=10.0,
                    quantity=Fraction(),
                    mark_price=0.0,
                )
                for row in old.valuations
                if row.asset_id == "b"
            ),
            diagnostics=bad.diagnostics
            + tuple(row for row in old.diagnostics if row.asset_id == "b"),
        )
    bad = replace(
        bad,
        diagnostics=tuple(
            replace(
                row,
                number=sum(
                    value.value for value in bad.valuations if value.phase == "after"
                ),
            )
            if row.asset_id == "__cash__" and row.name == "target_notional"
            else row
            for row in bad.diagnostics
        ),
    )
    capital = PortfolioCapital("raw_shares", 100.0, "CNY", "cash_equity")
    error = (
        "portfolio_held_lifecycle_unproven"
        if direction == "relabel"
        else "portfolio_ledger_cost_basis_invalid"
    )
    with pytest.raises(ValueError, match=error):
        ledger.write_portfolio_ledger_artifact(
            tmp_path / "rejected",
            capital=capital,
            metadata={"window": {"start": "2023-01-03", "end": "2023-01-10"}},
            produce=lambda emit: [emit(batch) for batch in (*previous, bad)],
        )
    assert not (tmp_path / "rejected").exists()
    # Fault injection isolates the reader's independent trust boundary: bypass
    # only writer transition validation when serializing the malformed fixture.
    with monkeypatch.context() as scoped:
        scoped.setattr(
            ledger,
            "validate_portfolio_transition",
            lambda batch, *_: {
                row.asset_id: row for row in batch.valuations if row.phase == "end"
            },
        )
        root = ledger.write_portfolio_ledger_artifact(
            tmp_path / "corrupt",
            capital=capital,
            metadata={"window": {"start": "2023-01-03", "end": "2023-01-10"}},
            produce=lambda emit: [emit(batch) for batch in (*previous, bad)],
        )
    with pytest.raises(ValueError, match=error):
        list(ledger.read_portfolio_ledger_artifact(root).iter_chunks())


def test_observed_same_price_on_ex_date_is_a_genuine_quote():
    marks = _ResearchValuationMarks()
    marks.project(mark_view("2023-01-04", 100.0, 100.0))
    view = mark_view("2023-01-05", 100.0, 100.0)
    view.corporate_events = pd.DataFrame(
        [
            dict(
                event_id="dividend",
                asset_id="a",
                listing_date=pd.Timestamp("2020-01-01"),
                effective_at=pd.Timestamp("2023-01-05"),
                available_at=pd.Timestamp("2023-01-04"),
                kind="cash_dividend",
                entitlement_proved=True,
                evidence_identity="a" * 64,
                cash_per_share=10.0,
            )
        ]
    )
    result = marks.project(view)
    assert result.open.iloc[0] == result.close.iloc[0] == 100.0
    assert result.research_open_mark_at.isna().all()
    assert marks.project(mark_view("2023-01-06", np.nan, np.nan)).open.iloc[0] == 100.0


def test_paid_zero_share_zero_claim_asset_can_leave_native_ledger(
    tmp_path, monkeypatch
):
    batches = corporate._distribution_batches(
        tmp_path / "good", monkeypatch, payment="2023-01-09", relist=True
    )
    last = batches[-1]
    assert all(
        row.quantity == row.receivable == 0
        for row in last.valuations
        if row.asset_id != "__cash__"
    )
    assert not last.trades and not last.cashflows and not last.rights_adjustments
    departed = replace(
        last,
        valuations=tuple(row for row in last.valuations if row.asset_id == "__cash__"),
        diagnostics=tuple(
            row for row in last.diagnostics if row.asset_id == "__cash__"
        ),
    )
    root = ledger.write_portfolio_ledger_artifact(
        tmp_path / "departed",
        capital=PortfolioCapital("raw_shares", 100.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2023-01-03", "end": "2023-01-10"}},
        produce=lambda emit: [emit(batch) for batch in (*batches[:-1], departed)],
    )
    read = list(ledger.read_portfolio_ledger_artifact(root).iter_chunks())
    assert portfolio_interval_economics(read[-1])["nav_end"] == 100.0


def test_persisted_adv_buy_gap_smaller_than_claim_keeps_buy_direction(
    tmp_path, monkeypatch
):
    original = corporate._build_producer

    def build(*args, **kwargs):
        panel = kwargs["frames"]["panel"]
        panel.loc[
            panel.asset_id.eq("b") & panel.timestamp.eq("2023-01-04"), "amount"
        ] = np.nan
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    batches = corporate._distribution_batches(
        tmp_path / "source", monkeypatch, payment="2023-01-09", freeze_ex_date=True
    )
    changed = []
    for batch in batches:
        claim = next(
            row
            for row in batch.valuations
            if row.asset_id == "b" and row.phase == "after"
        )
        if claim.receivable == 10.0 and any(
            row.asset_id == "b"
            and row.name == "block_reason"
            and row.text == "adv_unavailable"
            for row in batch.diagnostics
        ):
            batch = replace(
                batch,
                diagnostics=tuple(
                    replace(row, number=95.0)
                    if row.asset_id == "b" and row.name == "target_notional"
                    else replace(row, number=0.0)
                    if row.asset_id == "__cash__" and row.name == "target_notional"
                    else row
                    for row in batch.diagnostics
                ),
            )
            assert portfolio_interval_economics(batch)["requested_turnover"] == 0.05
        changed.append(batch)
    root = ledger.write_portfolio_ledger_artifact(
        tmp_path / "buy",
        capital=PortfolioCapital("raw_shares", 100.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2023-01-03", "end": "2023-01-10"}},
        produce=lambda emit: [emit(batch) for batch in changed],
    )
    metrics = _statistics(root)["portfolio_performance"].set_index("metric_id")
    assert metrics.loc["adv_unavailable_buy_orders", "value"] == 1
    assert metrics.loc["adv_unavailable_buy_target_notional", "value"] == 5
    assert metrics.loc["adv_unavailable_sell_target_notional", "value"] == 90


def test_attribution_uses_stock_exposure_while_pending_claim_stays_in_nav(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.portfolios.risk import (
        FactorReturnFit,
        HoldingFactorInputs,
    )

    original = corporate._build_producer

    def build(*args, **kwargs):
        frames = kwargs["frames"]
        masks = frames["execution"]
        masks.loc[
            masks.asset_id.eq("b") & masks.timestamp.ge("2023-01-04"),
            ["can_enter", "can_exit"],
        ] = False
        panel = frames["panel"]
        panel.loc[
            panel.asset_id.eq("b") & panel.timestamp.ge("2023-01-09"), ["open", "close"]
        ] = 99.0
        selected = panel.asset_id.eq("b") & panel.timestamp.ge("2023-01-05")
        panel.loc[selected, "adj_open"] = panel.loc[selected, "open"] / 0.9
        panel.loc[selected, "adj_close"] = panel.loc[selected, "close"] / 0.9
        return original(*args, **kwargs)

    monkeypatch.setattr(corporate, "_build_producer", build)
    batches = corporate._distribution_batches(
        tmp_path, monkeypatch, payment="2023-01-10", freeze_ex_date=True
    )
    inputs = []
    for batch in batches:
        before = {row.asset_id: row for row in batch.valuations if row.phase == "after"}
        end = {row.asset_id: row for row in batch.valuations if row.phase == "end"}
        returns = pd.Series({"b": end["b"].share_value / before["b"].share_value - 1})
        decision = next(
            row.text
            for row in batch.diagnostics
            if row.asset_id == "b" and row.name == "decision_timestamp"
        )
        inputs.append(
            HoldingFactorInputs(
                pd.Timestamp(decision),
                pd.Timestamp(before["b"].timestamp_ns),
                pd.Timestamp(end["b"].timestamp_ns),
                FactorReturnFit(
                    pd.DataFrame({"style": [1.0]}, index=["b"]),
                    pd.Series({"style": returns["b"]}),
                    pd.Series({"b": 0.0}),
                    returns,
                ),
                pd.Series({"b": before["b"].mark_price}),
                pd.Series({"b": before["b"].mark_price}),
                pd.DataFrame(),
            )
        )
    candidate = PortfolioCandidateSpec(
        "raw_shares", "global", "quantile_equal_weight", "synthetic", "daily", 0.5, None
    )
    tables = _portfolio_statistics_tables(
        ledgers={
            "raw_shares": ledger.read_portfolio_ledger_artifact(tmp_path / "ledger")
        },
        candidates=(candidate,),
        config={
            "sample_role": "validation",
            "analyses": ["performance", "attribution"],
        },
        time_window=TimeWindow("2023-01-03", "2023-01-10"),
        benchmark_opens=pd.DataFrame(),
        calendar=pd.bdate_range("2023-01-03", "2023-01-10"),
        unified_trading_sessions=lambda dates: dates,
        holding_inputs={(): inputs},
    )
    attribution = tables["portfolio_attribution"]
    period = attribution.loc[attribution.entry_at.eq("2023-01-06")]
    assert period.status.eq("DEFINED").all()
    assert period.loc[period.component_id.eq("style"), "value"].item() == pytest.approx(
        0.09
    )
    exposures = tables["portfolio_exposures"]
    stock = exposures.loc[
        exposures.entry_at.eq("2023-01-06") & exposures.factor_id.eq("style")
    ]
    assert stock.value.item() == 0.9


def test_zero_stress_keeps_claim_smaller_than_stock_amount_roundoff(
    tmp_path, monkeypatch
):
    batches = corporate._distribution_batches(
        tmp_path, monkeypatch, suspended=True, payment="2023-01-09", cash_dividend=1e-16
    )
    first = batches[0]
    end = next(
        row for row in first.valuations if row.asset_id == "b" and row.phase == "end"
    )
    assert end.receivable == 1e-16
    assert end.share_value == end.value == 100.0
    stress = next(
        row.number
        for row in first.diagnostics
        if row.name == "research_valuation_nav_0pct"
    )
    assert stress == 1e-16
