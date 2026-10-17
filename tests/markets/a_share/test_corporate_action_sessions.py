from fractions import Fraction

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.markets.implementations.a_share.processed import panels


def _facts(rows=None):
    stocks = panels._canonical_stock_master(
        pd.DataFrame(
            {
                "source_asset_id": ["000001", "000002"],
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "listing_date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
                "delisting_date": [pd.NaT, pd.NaT],
            }
        )
    )
    dates = pd.date_range("2023-01-01", "2023-01-10")
    calendar = pd.concat(
        [
            pd.DataFrame(
                {
                    "calendar_date": dates,
                    "source_market_type": market,
                    "is_trading_day": [
                        date.dayofweek < 5 and not (market == 4 and date.day == 5)
                        for date in dates
                    ],
                }
            )
            for market in (1, 4, 16, 32)
        ],
        ignore_index=True,
    )
    base = {
        "source_asset_id": "000001",
        "timestamp": pd.Timestamp("2023-01-06"),
        "kind": "CA",
        "market_type": 4,
        "announcement_date": pd.Timestamp("2023-01-02"),
        "settlement_date": pd.Timestamp("2023-01-09"),
        "amount": 0.5,
    }
    source = pd.DataFrame([base] if rows is None else rows(base))
    return source, stocks, calendar


def test_registration_uses_own_market_calendar_and_actual_share_amounts():
    events = portfolio_sessions._corporate_action_events(*_facts())
    event = events.iloc[0]
    assert event.record_at == pd.Timestamp("2023-01-04")
    assert event.effective_at == pd.Timestamp("2023-01-06")
    assert event.settlement_at == pd.Timestamp("2023-01-09")
    assert event.cash_per_share == Fraction(0.5)
    assert event.listing_date == pd.Timestamp("2022-01-01")
    assert event.entitlement_proved
    assert events.attrs["corporate_action_policy"]["voluntary_subscription"] == "none"


@pytest.mark.parametrize("kind", ["RO", "SN"])
def test_voluntary_subscription_does_not_generate_cash_or_free_stock(kind):
    events = portfolio_sessions._corporate_action_events(
        *_facts(lambda base: [dict(base, kind=kind)])
    )
    assert events.empty


@pytest.mark.parametrize("kind,amount", [("SD", -0.8), ("GQ", 0.4), ("unknown", 0.0)])
def test_aggregate_or_unknown_actions_remain_unproved(kind, amount):
    events = portfolio_sessions._corporate_action_events(
        *_facts(lambda base: [dict(base, kind=kind, amount=amount)])
    )
    assert len(events) == 1
    assert not events.entitlement_proved.iloc[0]
    assert events.additional_shares_per_share.isna().all()


@pytest.mark.parametrize("settlement", [pd.NaT, pd.Timestamp("1990-01-01")])
def test_missing_stock_unlock_does_not_erase_valid_zero_or_create_sale_permission(
    settlement,
):
    events = portfolio_sessions._corporate_action_events(
        *_facts(
            lambda base: [dict(base, kind="SD", amount=0.0, settlement_date=settlement)]
        )
    )
    assert events.entitlement_proved.iloc[0]
    assert events.additional_shares_per_share.iloc[0] == 0
    assert pd.isna(events.settlement_at.iloc[0])


def test_cash_missing_payment_date_uses_declared_ex_open_assumption():
    events = portfolio_sessions._corporate_action_events(
        *_facts(lambda base: [dict(base, settlement_date=pd.NaT)])
    )
    assert events.settlement_at.iloc[0] == events.effective_at.iloc[0]
    assert events.entitlement_proved.iloc[0]


def test_before_ex_cash_payment_is_unproved_not_silently_shifted():
    events = portfolio_sessions._corporate_action_events(
        *_facts(lambda base: [dict(base, settlement_date=pd.Timestamp("2023-01-03"))])
    )
    assert events.settlement_at.iloc[0] == pd.Timestamp("2023-01-03")
    assert not events.entitlement_proved.iloc[0]


def test_lifecycle_join_preserves_source_event_order_across_assets():
    events = portfolio_sessions._corporate_action_events(
        *_facts(lambda base: [dict(base, source_asset_id="000002"), base])
    )
    assert events.listing_date.tolist() == list(
        pd.to_datetime(["2022-02-01", "2022-01-01"])
    )


def test_unknown_calendar_day_does_not_become_a_closed_day():
    source, stocks, calendar = _facts()
    calendar = calendar.loc[
        ~(calendar.source_market_type.eq(4) & calendar.calendar_date.eq("2023-01-05"))
    ]
    events = portfolio_sessions._corporate_action_events(source, stocks, calendar)
    assert not events.entitlement_proved.iloc[0]


def test_native_interim_events_follow_record_to_payment_span_and_skip_old_claims(
    tmp_path,
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets import write_dataset
    from tests.research.test_closed_market_portfolios import _native_inputs

    inputs, roots, calendar, _ = _native_inputs(tmp_path, fault="normal_open")
    source = pd.DataFrame(
        {
            "source_asset_id": pd.Series(["a0", "a0"], dtype="string"),
            "timestamp": pd.to_datetime(["2023-01-04", "2022-12-29"]),
            "kind": pd.Series(["CA", "SD"], dtype="string"),
            "market_type": pd.Series([1, 1], dtype="Int64"),
            "announcement_date": pd.to_datetime(["2022-12-30", "2022-12-20"]),
            "settlement_date": pd.to_datetime(["2023-01-06", None]),
            "settlement_date_unresolved": False,
            "amount": [0.5, 0.4],
            "subscription_price": [float("nan"), float("nan")],
        }
    )
    path = tmp_path / "actions"
    write_dataset(path, source, spec=interim_catalog().get("corporate_actions"))
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root=inputs["market_bars"],
            stock_master_root=inputs["stock_master"],
            panel_root=roots["market_panel_1d"],
            corporate_actions_root=path,
            panel_columns=(
                "asset_id",
                "timestamp",
                "listing_date",
                "market_type",
                "is_trading",
                "open",
                "close",
            ),
            mark_price_columns=("open", "close"),
            calendar=calendar,
            time_window=TimeWindow("2022-12-30", "2023-01-06"),
        )
    )
    present = [view for view in views if not view.corporate_events.empty]
    assert [view.timestamp for view in present] == list(
        pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
    )
    for view in present:
        assert len(view.corporate_events) == 1
        event = view.corporate_events.iloc[0]
        assert event.kind == "cash_dividend"
        assert event.record_at == pd.Timestamp("2023-01-03")
        assert event.entitlement_proved
        assert len(event.evidence_identity) == 64


def test_proved_holder_transition_replaces_unproved_aggregate_without_early_block(
    tmp_path,
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets import write_dataset
    from tests.sources.reference.test_equity_events import native_equity_inputs

    inputs, roots, calendar = native_equity_inputs(
        tmp_path,
        fault="normal_open",
        events=lambda base: [
            dict(
                base,
                kind="share_change",
                effective_date="2023-01-04",
                publication_date="2023-01-02",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.16,
            )
        ],
    )
    source = pd.DataFrame(
        {
            "source_asset_id": pd.Series(["000001"], dtype="string"),
            "timestamp": pd.to_datetime(["2023-01-04"]),
            "kind": pd.Series(["SD"], dtype="string"),
            "market_type": pd.Series([1], dtype="Int64"),
            "announcement_date": pd.to_datetime(["2023-01-02"]),
            "settlement_date": pd.to_datetime(["2023-01-04"]),
            "settlement_date_unresolved": False,
            "amount": [-0.8],
            "subscription_price": [float("nan")],
        }
    )
    path = tmp_path / "actions"
    write_dataset(path, source, spec=interim_catalog().get("corporate_actions"))
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root=inputs["market_bars"],
            stock_master_root=inputs["stock_master"],
            equity_events_root=inputs["equity_events"],
            panel_root=roots["market_panel_1d"],
            corporate_actions_root=path,
            panel_columns=(
                "asset_id",
                "timestamp",
                "listing_date",
                "market_type",
                "is_trading",
                "open",
                "close",
            ),
            calendar=calendar,
            time_window=TimeWindow("2022-12-30", "2023-01-06"),
        )
    )
    pending = pd.concat(view.corporate_events for view in views)
    assert not pending.empty
    assert not pending.entitlement_proved.any()
    assert all(
        view.corporate_events.empty
        for view in views
        if view.timestamp >= pd.Timestamp("2023-01-04")
    )
    rights = pd.concat(view.rights_events for view in views)
    assert len(rights) == 1
    assert rights.kind.iloc[0] == "share_change"
    assert rights.entitlement_proved.iloc[0]
    assert rights.remaining_old_units_per_old_unit.iloc[0] == Fraction(0.16)
    from quant_research_platform.research.portfolios.accounting import (
        EquityEntitlements,
    )

    state = EquityEntitlements()
    shares = pd.Series({"000001": Fraction(100)}, dtype="object")
    lives = pd.Series({"000001": rights.listing_date.iloc[0]})
    adjustments = []
    for view in views:
        shares, lives, cash, changes = state.process(
            view.corporate_events, view.rights_events, shares, lives, view.timestamp
        )
        assert cash == 0
        assert not state.receivables
        assert not state.register(view.corporate_events, shares, lives, view.timestamp)
        if view.timestamp < pd.Timestamp("2023-01-04"):
            assert shares["000001"] == 100
        adjustments.extend(changes)
    assert shares["000001"] == 100 * Fraction(0.16)
    assert len(adjustments) == 1
    assert adjustments[0].kind == "recovery"
