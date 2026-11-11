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


def test_quote_successor_does_not_require_unrelated_board_closed_history():
    prices, calendar = inputs()
    # Before another board existed, its absent weekend rows are irrelevant.
    calendar = calendar.loc[
        ~(calendar.source_market_type.eq(32) & calendar.calendar_date.dt.weekday.ge(5))
    ]
    result = daily_quotes.daily_decisions(prices, calendar)
    assert result.signal_session.tolist() == list(
        pd.to_datetime(["2024-01-05", "2024-01-08"])
    )
    assert result.execution_session.tolist() == list(
        pd.to_datetime(["2024-01-08", "2024-01-09"])
    )
    assert result.next_session_intraday_return.tolist() == pytest.approx([0.1, 0.0])


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


def test_native_unknown_linked_factor_is_numeric_after_persistence(tmp_path):
    import numpy as np

    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.research.test_prepare_selection import _input

    prices, calendar = inputs()
    decision = daily_quotes.daily_decisions(prices, calendar)
    us = pd.DataFrame(
        dict(
            asset_id=["SPY"],
            timestamp=pd.Timestamp("2024-01-05"),
            available_at=pd.Timestamp("2024-01-06"),
            adjusted_close=[100.0],
        )
    )
    us_root = tmp_path / "us"
    write_dataset(
        us_root,
        us,
        spec=DataSpec(
            data_id="public_etf_prices",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("asset_id", "timestamp"),
            partition_columns=(),
            column_roles={
                k: "timestamp"
                if k == "timestamp"
                else "key"
                if k == "asset_id"
                else "context"
                for k in us
            },
        ),
    )
    path = prepare(
        dict(
            market_id="a_share",
            data_id="linked_market_factors",
            inputs=dict(
                decision_panel=_input(
                    tmp_path, "etf_daily_panel", decision, native=True
                ),
                public_etf_prices=str(us_root),
            ),
            factor_columns=["us_sector_etf_return_shock_1d"],
            time_window=dict(start="2024-01-08", end="2024-01-08"),
            output_root=str(tmp_path / "unknown-factor"),
        )
    )
    result = read_dataset(path).us_sector_etf_return_shock_1d
    assert result.isna().all()
    assert not np.isfinite(result).any()


def test_benchmark_selector_does_not_require_the_whole_input_to_be_one_index(tmp_path):
    from dataclasses import replace

    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_sources import _calendar_input

    prices, _ = inputs()
    one = (
        prices[["asset_id", "timestamp", "open", "close"]]
        .rename(columns={"asset_id": "source_asset_id"})
        .assign(source_asset_id="000300")
    )
    raw = pd.concat([one, one.assign(source_asset_id="000905")], ignore_index=True)
    spec = interim_catalog().get("index_market")
    spec = replace(
        spec,
        column_roles={k: spec.column_roles[k] for k in raw},
        column_units={k: v for k, v in spec.column_units.items() if k in raw},
    )
    root = tmp_path / "indices"
    write_dataset(root, raw, spec=spec)
    calendar = _calendar_input(
        tmp_path / "calendar", {1: pd.DatetimeIndex(one.timestamp)}
    )
    output = prepare(
        dict(
            market_id="a_share",
            data_id="benchmark_daily_panel",
            inputs=dict(index_market=str(root), trading_calendar=str(calendar)),
            parameters=dict(asset_ids=["000300"]),
            time_window=dict(start="2024-01-05", end="2024-01-09"),
            output_root=str(tmp_path / "output"),
        )
    )
    assert read_dataset(output).asset_id.unique().tolist() == ["index:000300"]
