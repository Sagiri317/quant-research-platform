"""ETF/index labels use exact calendar successors and known preopen context."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    daily_quotes,
)


def inputs():
    days = pd.date_range("2024-01-05", "2024-01-09")
    calendar = pd.DataFrame(
        [
            dict(
                calendar_date=date,
                source_market_type=market,
                is_trading_day=date.weekday() < 5,
            )
            for date in days
            for market in CURRENT_MARKET_TYPES
        ]
    )
    prices = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
            open=[10.0, 20.0, 10.0],
            close=[11.0, 22.0, 10.0],
            volume=100.0,
            amount=1000.0,
        )
    )
    return prices, calendar


def test_etf_intraday_label_is_next_declared_session_and_cash_adjustment_cancels():
    prices, calendar = inputs()
    result = daily_quotes.daily_decisions(prices, calendar)
    first = result.iloc[0]
    assert first.signal_session == pd.Timestamp("2024-01-05")
    assert first.timestamp == pd.Timestamp("2024-01-08 01:00")
    assert first.close == 11
    assert first.next_session_intraday_return == pytest.approx(0.1)
    assert first.return_available_at == pd.Timestamp("2024-01-05 07:00:01")
    assert first.next_session_intraday_return_available_at == pd.Timestamp(
        "2024-01-08 07:00:01"
    )
    changed = prices.copy()
    changed.loc[1, ["open", "close"]] *= 100
    assert daily_quotes.daily_decisions(changed, calendar).iloc[
        0
    ].next_session_intraday_return == pytest.approx(0.1)


def test_missing_quote_or_unknown_calendar_day_does_not_compress_successor_axis():
    prices, calendar = inputs()
    result = daily_quotes.daily_decisions(prices.drop(index=1), calendar)
    assert pd.isna(result.iloc[0].next_session_intraday_return)
    assert result.iloc[1].next_session_intraday_return == 0
    unknown = calendar.loc[calendar.calendar_date.ne(pd.Timestamp("2024-01-07"))]
    result = daily_quotes.daily_decisions(prices, unknown)
    assert not result.signal_session.eq(pd.Timestamp("2024-01-05")).any()


def test_total_return_uses_ex_date_cash_and_new_over_old_split_ratio():
    prices, calendar = inputs()
    prices["close"] = [10.0, 4.5, 5.0]
    actions = pd.DataFrame(
        dict(
            asset_id=["510300:dividends", "510300:splits"],
            timestamp=pd.Timestamp("2024-01-08"),
            available_at=pd.Timestamp("2024-01-04"),
            event_type=["dividends", "splits"],
            amount=[1.0, float("nan")],
            numerator=[float("nan"), 2.0],
            denominator=[float("nan"), 1.0],
        )
    )
    result = daily_quotes.daily_decisions(prices, calendar, actions=actions)
    assert result.adj_close.tolist() == pytest.approx([10.0, 10.0])
    assert result.period_return.iloc[1] == pytest.approx(0)


def test_native_market_states_count_stock_replicas_once_without_style_input(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    prices, calendar = inputs()
    benchmark = daily_quotes.daily_decisions(
        prices.assign(asset_id="index:000300"), calendar
    )
    state = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["a", "b", "c"],
            timestamp=pd.Timestamp("2024-01-05"),
            market_up_fraction=0.5,
        )
    )
    path = prepare(
        dict(
            market_id="a_share",
            data_id="market_state_factors",
            inputs={
                "benchmark_daily_panel": _input(
                    tmp_path, "benchmark_daily_panel", benchmark, native=True
                ),
                "market_structure_factors": _input(
                    tmp_path, "market_structure_factors", state, native=True
                ),
            },
            factor_columns=["market_up_fraction"],
            time_window=dict(start="2024-01-08", end="2024-01-08"),
            output_root=str(tmp_path / "state"),
        )
    )
    result = read_dataset(path)
    assert len(result) == 1
    assert result.market_up_fraction.iloc[0] == 0.5
