"""Native warmup retains valuation and entitlement history without empty prefixes."""

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.research.portfolios.high_frequency import (
    _daily_market_views,
    inventory_batches,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.high_frequency import (
    parameters,
    quotes,
    schedule,
    write_daily_inputs,
)
from tests.helpers.portfolio_inputs import _interim_specs, closed_market_inputs


def _quote_sessions(views, observed=None):
    empty = quotes("2024-01-02").iloc[:0]
    for view in views:
        current = (
            empty
            if observed is None
            else observed.loc[observed.session_date.eq(view.timestamp)]
        )
        yield view, current, pd.DataFrame()


def _cash_dividend(path, *, ex_date, payment, announcement, asset="a", amount=1.0):
    frame = pd.DataFrame(
        {
            "source_asset_id": pd.Series([asset], dtype="string"),
            "timestamp": pd.to_datetime([ex_date]),
            "kind": pd.Series(["CA"], dtype="string"),
            "market_type": pd.Series([4 if asset == "a" else 1], dtype="Int64"),
            "announcement_date": pd.to_datetime([announcement]),
            "settlement_date": pd.to_datetime([payment]),
            "settlement_date_unresolved": False,
            "amount": [amount],
            "subscription_price": [float("nan")],
        }
    )
    write_dataset(path, frame, spec=_interim_specs().get("corporate_actions"))
    return str(path)


@pytest.mark.parametrize("start", ["2024-01-01", "2024-01-03"])
def test_warmup_skips_empty_calendar_prefix_and_keeps_requested_cash_sessions(
    tmp_path, start
):
    panel = pd.concat(
        [quotes(day, assets=("a",)) for day in ("2024-01-02", "2024-01-03")],
        ignore_index=True,
    )
    daily = write_daily_inputs(tmp_path / "daily", panel)
    days = pd.date_range("2023-12-01", "2024-01-03")
    calendar = pd.DataFrame(
        {
            "timestamp": days,
            "calendar_date": days,
            "source_market_type": 4,
            "is_trading_day": days.dayofweek < 5,
        }
    )
    write_dataset(
        tmp_path / "calendar", calendar, spec=_interim_specs().get("trading_calendar")
    )
    daily["calendar"] = str(tmp_path / "calendar")
    window = TimeWindow(start, "2024-01-03")
    views = list(_daily_market_views(daily, window))
    # The caller's cash-only session remains, while earlier unrelated sessions go.
    assert [view.timestamp for view in views] == list(
        pd.date_range(min(start, "2024-01-02"), "2024-01-03")
    )
    params = {**parameters(("a",)), "initial_inventory": {"a": 0}}
    batches = list(
        inventory_batches(_quote_sessions(views, panel), schedule(), params, window)
    )
    assert not any(batch.trades or batch.cashflows for batch in batches)
    assert all(
        sum(row.value for row in batch.valuations if row.phase == "end") == 5000.0
        for batch in batches
    )
    assert pd.Timestamp(
        batches[0].valuations[-1].timestamp_ns
    ).normalize() == pd.Timestamp(start)


def test_initial_closed_session_keeps_raw_close_across_multiple_no_quote_sessions(
    tmp_path,
):
    inputs, _, _, _ = closed_market_inputs(tmp_path)
    daily = {
        name: str(inputs[name])
        for name in (
            "panel",
            "calendar",
            "market_bars",
            "stock_master",
            "corporate_actions",
        )
    }
    for name in ("panel", "market_bars"):
        daily[name] = {
            "path": str(daily[name]),
            "time_window": {"start": "2023-01-02", "end": "2023-01-04"},
        }
    window = TimeWindow("2023-01-04", "2023-01-04")
    views = list(_daily_market_views(daily, window))
    assert [view.timestamp for view in views] == list(
        pd.date_range("2023-01-02", "2023-01-04")
    )
    assert all("a0" not in set(view.panel.asset_id) for view in views[1:])
    params = {
        **parameters(("a0", "b0")),
        "initial_capital": 20_000.0,
        "initial_inventory": {"a0": 100, "b0": 0},
    }
    batches = list(
        inventory_batches(
            _quote_sessions(views, quotes("2023-01-04", assets=("b0",), mid=100.0)),
            schedule(),
            params,
            window,
        )
    )
    holdings = [
        row
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "a0" and row.phase == "end"
    ]
    # Jan 2's observed close is 110; both later own-market sessions are closed.
    assert holdings and all(
        row.quantity == 100 and row.mark_price == 110 for row in holdings
    )
    assert all(
        sum(row.value for row in batch.valuations if row.phase == "end") == 20_000
        for batch in batches
    )


def test_precise_empty_market_window_keeps_the_native_daily_cash_axis(tmp_path):
    inputs, _, calendar, _ = closed_market_inputs(tmp_path)
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root={
                "path": str(inputs["market_bars"]),
                "row_filters": [["source_asset_id", "==", "absent"]],
            },
            stock_master_root=inputs["stock_master"],
            panel_root={
                "path": str(inputs["panel"]),
                "row_filters": [["asset_id", "==", "absent"]],
            },
            panel_columns=("asset_id", "timestamp", "listing_date", "open", "close"),
            calendar=calendar,
            time_window=TimeWindow("2023-01-02T12:00:00", "2023-01-04T12:00:00"),
        )
    )
    assert [view.timestamp for view in views] == list(
        pd.to_datetime(["2023-01-03", "2023-01-04"])
    )
    assert all(view.panel.empty and view.valuation.empty for view in views)


def test_corporate_history_before_first_selected_quote_keeps_registration_obligation(
    tmp_path,
):
    inputs, _, _, _ = closed_market_inputs(tmp_path)
    daily = {
        name: str(inputs[name])
        for name in (
            "panel",
            "calendar",
            "market_bars",
            "stock_master",
            "corporate_actions",
        )
    }
    for name in ("panel", "market_bars"):
        daily[name] = {
            "path": str(daily[name]),
            "time_window": {"start": "2023-01-02", "end": "2023-01-04"},
        }
    daily["corporate_actions"] = _cash_dividend(
        tmp_path / "actions",
        ex_date="2022-12-29",
        payment="2023-01-05",
        announcement="2022-12-27",
        asset="a0",
        amount=0.5,
    )
    window = TimeWindow("2023-01-04", "2023-01-04")
    views = list(_daily_market_views(daily, window))
    assert views[0].timestamp == pd.Timestamp("2022-12-29")
    event = views[-1].corporate_events.iloc[0]
    assert event.record_at == pd.Timestamp("2022-12-28")
    assert event.settlement_at == pd.Timestamp("2023-01-05")
    assert event.entitlement_proved
    params = {**parameters(("a0",)), "initial_capital": 20_000.0}
    # Pre-episode marks cannot manufacture an initial holder's record-date shares.
    with pytest.raises(ValueError, match="hft_initial_inventory_registration_missing"):
        list(inventory_batches(_quote_sessions(views), schedule(), params, window))


def test_recovery_quote_before_selected_market_history_values_initial_delisted_shares(
    tmp_path,
):
    from tests.sources.reference.test_equity_events import native_equity_inputs

    inputs, _, _ = native_equity_inputs(
        tmp_path,
        events=lambda base: [
            dict(base, effective_date="2022-12-29", publication_date="2022-12-29")
        ],
    )
    stocks = read_dataset(inputs["stock_master"])
    stocks.loc[stocks.source_asset_id.eq("000001"), "delisting_date"] = pd.Timestamp(
        "2022-12-28"
    )
    write_dataset(
        tmp_path / "stocks", stocks, spec=_interim_specs().get("stock_master")
    )
    daily = {
        name: str(inputs[name])
        for name in (
            "panel",
            "calendar",
            "market_bars",
            "stock_master",
            "corporate_actions",
            "equity_events",
        )
    }
    daily["stock_master"] = str(tmp_path / "stocks")
    for name in ("panel", "market_bars"):
        daily[name] = {
            "path": daily[name],
            "time_window": {"start": "2023-01-03", "end": "2023-01-04"},
        }
    window = TimeWindow("2023-01-04", "2023-01-04")
    views = list(_daily_market_views(daily, window))
    assert views[0].timestamp == pd.Timestamp("2022-12-30")
    event = views[0].rights_events.iloc[0]
    assert event.kind == "quote" and event.entitlement_proved
    assert event.value_per_old_unit == 2.5
    params = {
        **parameters(("000001", "b0")),
        "initial_inventory": {"000001": 100, "b0": 0},
    }
    batches = list(
        inventory_batches(
            _quote_sessions(views, quotes("2023-01-04", assets=("b0",), mid=100.0)),
            schedule(),
            params,
            window,
        )
    )
    holdings = [
        row
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "000001" and row.phase == "end"
    ]
    assert holdings and all(
        row.quantity == 100 and row.mark_price == 2.5 for row in holdings
    )
    assert all(
        sum(row.value for row in batch.valuations if row.phase == "end") == 5000.0
        for batch in batches
    )


@pytest.mark.parametrize("start", ["2024-01-02", "2024-01-04"])
def test_registered_dividend_retains_receivable_then_real_payment(tmp_path, start):
    panel = pd.concat(
        [
            quotes(day, assets=("a",), mid=10.0 if day == "2024-01-02" else 9.0)
            for day in ("2024-01-02", "2024-01-03", "2024-01-04")
        ],
        ignore_index=True,
    )
    daily = write_daily_inputs(tmp_path / "daily", panel)
    daily["corporate_actions"] = _cash_dividend(
        tmp_path / "actions",
        ex_date="2024-01-03",
        payment="2024-01-04",
        announcement="2024-01-02",
    )
    window = TimeWindow(start, "2024-01-04")
    stream = inventory_batches(
        _quote_sessions(_daily_market_views(daily, window), panel),
        schedule(),
        parameters(("a",)),
        window,
    )
    if start == "2024-01-04":
        with pytest.raises(
            ValueError, match="hft_initial_inventory_registration_missing"
        ):
            list(stream)
        return
    batches = list(stream)
    closes = [
        batch
        for batch in batches
        if any(row.text == "close" for row in batch.diagnostics)
    ]
    assert [
        sum(row.receivable for row in batch.valuations if row.phase == "end")
        for batch in closes
    ] == [0.0, 100.0, 0.0]
    assert [
        (flow.kind, flow.amount) for batch in batches for flow in batch.cashflows
    ] == [("dividend", 100.0)]
    assert all(
        sum(row.value for row in batch.valuations if row.phase == "end") == 5000.0
        for batch in batches
    )
