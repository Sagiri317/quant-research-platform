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


def test_native_etf_price_volume_reuses_math_and_keeps_decision_clock(tmp_path):
    import numpy as np

    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    days = pd.bdate_range("2024-01-02", periods=23)
    frame = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="510300",
            signal_session=days[:22],
            timestamp=days[1:] + pd.Timedelta(hours=1),
            adj_close=[100.0] * 20 + [110.0, 120.0],
            amount=[1000.0] * 20 + [500.0, 1000.0],
            next_session_intraday_return=0.0,
            next_session_intraday_return_available_at=days[1:] + pd.Timedelta(hours=7),
        )
    )
    name = "breakout_activity_divergence_20d"
    config = dict(
        market_id="a_share",
        data_id="etf_price_volume_factors",
        inputs={
            "etf_daily_panel": _input(tmp_path, "etf_daily_panel", frame, native=True)
        },
        factor_columns=[name],
        time_window=dict(start="2024-01-02", end="2024-02-05"),
        output_root=str(tmp_path / "etf-factors"),
    )
    result = read_dataset(prepare(config))
    assert result[name].iloc[:20].isna().all()
    assert result[name].iloc[20] == pytest.approx(np.log(1.1) * np.log(2))
    assert result.timestamp.tolist() == frame.timestamp.tolist()
    missing = frame.copy()
    missing.loc[19, "adj_close"] = np.nan
    config["inputs"]["etf_daily_panel"] = _input(
        tmp_path / "missing", "etf_daily_panel", missing, native=True
    )
    config["output_root"] = str(tmp_path / "missing-etf-factors")
    assert read_dataset(prepare(config))[name].iloc[20:].isna().all()


@pytest.mark.parametrize("intraday_bounds", [False, True])
def test_native_etf_narrow_window_keeps_warmup_recursive_state_and_missingness(
    tmp_path, intraday_bounds
):
    import numpy as np

    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    days = pd.bdate_range("2024-01-02", periods=62)
    returns = np.sin(np.arange(61)) / 100
    returns[54] = np.nan
    frame = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="510300",
            signal_session=days[:-1],
            timestamp=days[1:] + pd.Timedelta(hours=1),
            adj_close=np.r_[np.full(20, 100.0), 100 * 1.01 ** np.arange(1, 42)],
            period_return=returns,
            next_session_intraday_return=0.0,
            next_session_intraday_return_available_at=days[1:] + pd.Timedelta(hours=7),
        )
    )
    # A second asset has an omitted calendar link, so its prior episode and
    # nominal statistical window cannot be resumed at the new segment.
    interrupted = frame.loc[(frame.index < 34) | (frame.index >= 49)].assign(
        asset_id="510500"
    )
    source = pd.concat([frame, interrupted], ignore_index=True)
    lower, upper = frame.timestamp.iloc[[55, 58]]
    window = dict(start=lower.date().isoformat(), end=upper.date().isoformat())
    expected_indices = range(55, 59)
    if intraday_bounds:
        window = dict(
            start=(lower + pd.Timedelta(microseconds=1)).isoformat(),
            end=upper.isoformat(),
        )
        expected_indices = range(56, 59)
    result = read_dataset(
        prepare(
            dict(
                market_id="a_share",
                data_id="etf_price_volume_factors",
                inputs={
                    "etf_daily_panel": _input(
                        tmp_path, "etf_daily_panel", source, native=True
                    )
                },
                factor_columns=[
                    "reversal_5d",
                    "volatility_20d",
                    "signed_breakout_duration_20d",
                ],
                time_window=window,
                output_root=str(tmp_path / "factors"),
            )
        )
    )
    for asset, rows in result.groupby("asset_id", sort=False):
        assert (
            rows.timestamp.tolist()
            == frame.timestamp.iloc[list(expected_indices)].tolist()
        )
        assert rows.reversal_5d.tolist() == pytest.approx([-(1.01**5 - 1)] * len(rows))
        if asset == "510500":
            assert rows.volatility_20d.isna().all()
            assert rows.signed_breakout_duration_20d.isna().all()
        else:
            assert rows.volatility_20d.tolist() == pytest.approx(
                [np.nanstd(returns[i - 19 : i + 1], ddof=1) for i in expected_indices]
            )
            assert rows.signed_breakout_duration_20d.tolist() == [
                i - 20 + 1 for i in expected_indices
            ]


def test_native_etf_window_without_decisions_publishes_empty_numeric_factors(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    prices, calendar = inputs()
    frame = daily_quotes.daily_decisions(prices, calendar)
    result = read_dataset(
        prepare(
            dict(
                market_id="a_share",
                data_id="etf_price_volume_factors",
                inputs={
                    "etf_daily_panel": _input(
                        tmp_path, "etf_daily_panel", frame, native=True
                    )
                },
                factor_columns=["reversal_5d"],
                time_window=dict(start="2025-01-01", end="2025-01-02"),
                output_root=str(tmp_path / "empty"),
            )
        )
    )
    assert result.empty
    assert pd.api.types.is_numeric_dtype(result.reversal_5d)


def test_etf_intraday_label_is_next_declared_session_and_cash_adjustment_cancels():
    prices, calendar = inputs()
    result = daily_quotes.daily_decisions(prices, calendar)
    first = result.iloc[0]
    assert first.signal_session == pd.Timestamp("2024-01-05")
    assert first.timestamp == pd.Timestamp("2024-01-08 01:00")
    assert first.close == 11
    assert first.next_session_intraday_return == pytest.approx(0.1)
    assert pd.isna(first.return_available_at)
    assert result.return_available_at.isna().all()  # ETF actions are undeclared.
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


@pytest.mark.parametrize("etf", [False, True])
def test_unknown_calendar_day_breaks_return_and_rolling_history(etf):
    import numpy as np

    days = pd.bdate_range("2024-01-02", periods=48)
    calendar = pd.DataFrame(
        dict(
            calendar_date=pd.date_range(days.min(), days.max()),
            source_market_type=1,
        )
    )
    calendar["is_trading_day"] = calendar.calendar_date.isin(days)
    calendar = calendar.loc[calendar.calendar_date.ne(days[24])]
    prices = pd.DataFrame(
        dict(
            asset_id="510300" if etf else "000300",
            timestamp=days,
            open=100.0 * 1.01 ** np.arange(len(days)),
            close=100.0 * 1.01 ** np.arange(len(days)),
            volume=100.0,
            amount=1000.0,
        )
    )
    if not etf:
        prices = prices.drop(columns=["volume", "amount"])
    actions = pd.DataFrame(columns=["asset_id", "timestamp"]) if etf else None
    result = daily_quotes.daily_decisions(prices, calendar, actions=actions).set_index(
        "signal_session"
    )
    assert pd.isna(result.loc[days[25], "period_return"])
    assert result.loc[days[26], "period_return"] == pytest.approx(0.01)
    assert pd.isna(result.loc[days[44], "volatility_20d"])
    assert result.loc[days[45], "volatility_20d"] == pytest.approx(0, abs=1e-14)
    # The independent next-session label remains valid across this history break.
    assert result.loc[days[25], "next_session_intraday_return"] == 0.0
    if etf:
        assert pd.isna(result.loc[days[43], "avg_volume_20d"])
        assert result.loc[days[44], "avg_volume_20d"] == 100.0


def test_native_etf_price_history_does_not_join_across_omitted_decisions(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    days = pd.bdate_range("2024-01-02", periods=7)
    # A missing calendar fact prevents publishing the decisions around the gap.
    # Each surviving timestamp still proves its own signal's exact successor.
    frame = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="510300",
            signal_session=days[[0, 1, 4, 5]],
            timestamp=days[[1, 2, 5, 6]] + pd.Timedelta(hours=1),
            adj_close=[100.0, 110.0, 146.41, 161.051],
            next_session_intraday_return=0.0,
            next_session_intraday_return_available_at=days[[1, 2, 5, 6]]
            + pd.Timedelta(hours=7),
        )
    )
    result = read_dataset(
        prepare(
            dict(
                market_id="a_share",
                data_id="etf_price_volume_factors",
                inputs={
                    "etf_daily_panel": _input(
                        tmp_path, "etf_daily_panel", frame, native=True
                    )
                },
                factor_columns=["reversal_1d"],
                time_window=dict(start="2024-01-02", end="2024-01-10"),
                output_root=str(tmp_path / "factors"),
            )
        )
    )
    assert result.timestamp.tolist() == frame.timestamp.tolist()
    assert pd.isna(result.reversal_1d.iloc[2])
    assert result.reversal_1d.iloc[[1, 3]].tolist() == pytest.approx([-0.1, -0.1])


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


@pytest.mark.parametrize(
    "barrier", ["unknown_release", "late_release", "missing_price"]
)
def test_etf_returns_recover_after_an_unknown_action_interval(barrier):
    import numpy as np

    days = pd.bdate_range("2024-01-02", periods=30)
    calendar = pd.DataFrame(
        dict(
            calendar_date=pd.date_range(days.min(), days.max()),
            source_market_type=1,
        )
    )
    calendar["is_trading_day"] = calendar.calendar_date.isin(days)
    prices = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=days,
            open=100.0 * 1.01 ** np.arange(len(days)),
            close=100.0 * 1.01 ** np.arange(len(days)),
            volume=100.0,
            amount=1000.0,
        )
    )
    released = (
        pd.NaT
        if barrier == "unknown_release"
        else days[4]
        if barrier == "late_release"
        else days[0]
    )
    actions = pd.DataFrame(
        dict(
            asset_id=["510300:dividends"],
            timestamp=[days[2]],
            available_at=[released],
            event_type=["dividends"],
            amount=[1.0],
            numerator=[np.nan],
            denominator=[np.nan],
        )
    )
    if barrier == "missing_price":
        prices.loc[2, "close"] = np.nan
    result = daily_quotes.daily_decisions(prices, calendar, actions=actions).set_index(
        "signal_session"
    )
    assert pd.isna(result.loc[days[2], "period_return"])
    assert pd.isna(result.loc[days[2], "return_available_at"])
    recovery = 4 if barrier == "missing_price" else 3
    if barrier == "missing_price":
        assert pd.isna(result.loc[days[3], "period_return"])
        assert pd.isna(result.loc[days[3], "return_available_at"])
    # Both prices and the absence of a current action determine these intervals.
    assert result.loc[days[recovery] :, "period_return"].tolist() == pytest.approx(
        [0.01] * (len(result) - recovery)
    )
    assert result.adj_close.iloc[2:].isna().all()
    assert result.iloc[-1].volatility_20d == pytest.approx(0, abs=1e-14)
    assert result.loc[days[recovery], "next_session_intraday_return"] == 0.0
    assert result.loc[days[recovery], "return_available_at"] == days[
        recovery
    ] + pd.Timedelta(hours=7, seconds=1)


@pytest.mark.parametrize(
    "case,second_release,second_amount,expected",
    [
        ("identical_copy", "2024-01-04", 1.0, 0.0),
        ("later_identical_copy", "2024-01-08 00:00", 1.0, 0.0),
        ("future_revision", "2024-01-08 02:00", 2.0, 0.0),
        ("visible_revision", "2024-01-08 00:00", 2.0, 0.1),
        ("simultaneous_conflict", "2024-01-04", 2.0, None),
    ],
)
def test_etf_actions_use_visible_versions_across_a_weekend(
    case, second_release, second_amount, expected
):
    days = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    calendar = pd.DataFrame(
        dict(calendar_date=pd.date_range(days.min(), days.max()), source_market_type=1)
    )
    calendar["is_trading_day"] = calendar.calendar_date.isin(days)
    prices = pd.DataFrame(
        dict(asset_id="510300", timestamp=days, open=[10.0, 9.0, 9.0, 9.0])
    )
    prices["close"] = prices.open
    actions = pd.DataFrame(
        dict(
            asset_id=["510300:dividends"] * 2,
            timestamp=[days[1]] * 2,
            available_at=pd.to_datetime(["2024-01-04", second_release], format="mixed"),
            event_type=["dividends"] * 2,
            amount=[1.0, second_amount],
            numerator=[float("nan")] * 2,
            denominator=[float("nan")] * 2,
        )
    )
    result = daily_quotes.daily_decisions(prices, calendar, actions=actions).set_index(
        "signal_session"
    )
    assert result.loc[days[1], "timestamp"] == pd.Timestamp("2024-01-08 01:00")
    actual = result.loc[days[1], "period_return"]
    if expected is None:
        assert pd.isna(actual)
        assert pd.isna(result.loc[days[1], "return_available_at"])
    else:
        assert actual == pytest.approx(expected)
        expected_clock = pd.Timestamp(
            "2024-01-08 00:00" if case == "visible_revision" else "2024-01-05 07:00:01"
        )
        assert result.loc[days[1], "return_available_at"] == expected_clock
    assert result.loc[days[2], "period_return"] == 0.0
    assert result.loc[days[2], "return_available_at"] == pd.Timestamp(
        "2024-01-08 07:00:01"
    )


@pytest.mark.parametrize(
    "cash_release,expected_clock",
    [
        ("2024-01-05 06:00", "2024-01-05 07:00:01"),
        ("2024-01-05 08:00", "2024-01-05 08:00"),
        ("2024-01-08 01:00", "2024-01-08 01:00"),
        ("2024-01-08 02:00", None),
        (None, None),
    ],
)
def test_etf_return_maturity_includes_every_current_action(
    cash_release, expected_clock
):
    days = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    calendar = pd.DataFrame(
        dict(calendar_date=pd.date_range(days.min(), days.max()), source_market_type=1)
    )
    calendar["is_trading_day"] = calendar.calendar_date.isin(days)
    prices = pd.DataFrame(
        dict(asset_id="510300", timestamp=days, open=[10.0, 5.0, 5.0, 5.0])
    )
    prices["close"] = prices.open
    actions = pd.DataFrame(
        dict(
            asset_id=["510300:splits", "510300:dividends"],
            timestamp=[days[1]] * 2,
            available_at=pd.to_datetime(["2024-01-05 05:00", cash_release]),
            event_type=["splits", "dividends"],
            amount=[float("nan"), 1.0],
            numerator=[2.0, float("nan")],
            denominator=[1.0, float("nan")],
        )
    )
    result = daily_quotes.daily_decisions(prices, calendar, actions=actions).set_index(
        "signal_session"
    )
    if expected_clock is None:
        assert pd.isna(result.loc[days[1], "period_return"])
        assert pd.isna(result.loc[days[1], "return_available_at"])
    else:
        assert result.loc[days[1], "period_return"] == pytest.approx(0.1)
        assert result.loc[days[1], "return_available_at"] == pd.Timestamp(
            expected_clock
        )
    assert result.loc[days[2], "period_return"] == 0.0
    assert result.loc[days[2], "return_available_at"] == pd.Timestamp(
        "2024-01-08 07:00:01"
    )


def test_etf_revised_return_clock_pairs_with_macro_state_at_real_maturity():
    import numpy as np

    from quant_research_platform.research.factors.macro_liquidity.calculations import (
        calculate_macro_factors,
    )
    from tests.research.test_macro_liquidity import _observations

    days = pd.bdate_range("2024-01-02", periods=7)
    levels = np.arange(len(days), dtype=float) ** 2
    calendar = pd.DataFrame(
        dict(calendar_date=pd.date_range(days.min(), days.max()), source_market_type=1)
    )
    calendar["is_trading_day"] = calendar.calendar_date.isin(days)
    prices = pd.DataFrame(
        dict(asset_id="510300", timestamp=days, open=100.0, close=100.0)
    )
    # Every current action is first visible after the close, but before its
    # native next-session decision. The preceding price is already known.
    actions = pd.DataFrame(
        dict(
            asset_id="510300:dividends",
            timestamp=days[1:],
            available_at=days[1:] + pd.Timedelta(hours=9),
            event_type="dividends",
            amount=2 * np.diff(levels),
            numerator=np.nan,
            denominator=np.nan,
        )
    )
    panel = daily_quotes.daily_decisions(prices, calendar, actions=actions)
    # The native ETF panel does not declare these stock-life fields. Supply
    # them explicitly to exercise the public calculation consumer, not to
    # claim the current native ETF macro producer already supports betas.
    panel = panel.assign(listing_date=days[0], market_type=1)
    observations = _observations(
        [
            ("us_nfci", day, day + pd.Timedelta(hours=hour), value)
            for day, level in zip(days, levels, strict=True)
            for hour, value in ((6, level**2), (8, level))
        ]
    )
    config = dict(beta_window=4, beta_min_observations=3, beta_min_changes=3)
    result = calculate_macro_factors(
        observations,
        panel,
        columns=["ml_us_nfci_beta"],
        config=config,
        calendar=calendar,
    )
    # cash/100 = 0.02 * the state change visible when each return matures.
    np.testing.assert_allclose(result.ml_us_nfci_beta.iloc[4:], 0.02)
    falsely_closed = panel.assign(
        return_available_at=panel.signal_session + pd.Timedelta(hours=7, seconds=1)
    )
    wrong = calculate_macro_factors(
        observations,
        falsely_closed,
        columns=["ml_us_nfci_beta"],
        config=config,
        calendar=calendar,
    )
    assert not np.allclose(wrong.ml_us_nfci_beta.iloc[4:], 0.02)


def test_native_etf_action_versions_and_local_returns_reach_saved_factors(tmp_path):
    import hashlib
    import json
    from dataclasses import replace

    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.sources.public_research.interim import interim_catalog
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_sources import _calendar_input

    days = pd.bdate_range("2024-01-02", periods=30)
    actions = pd.DataFrame(
        dict(
            symbol=["510300"] * 3,
            observation_date=["2024-01-03", "2024-01-03", "2024-01-05"],
            event_type=["dividends"] * 3,
            amount=[1.0, 1.0, 2.0],
            numerator=[None] * 3,
            denominator=[None] * 3,
            split_ratio=[None] * 3,
            publication_date=["2024-01-03", "2024-01-03", "2024-01-09"],
            published_at=[None] * 3,
            source_url=["https://example.org/notice"] * 3,
            source_sha256=[letter * 64 for letter in ("a", "b", "c")],
        )
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "actions.csv"
    actions.to_csv(path, index=False)
    (raw / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                records=[
                    dict(
                        source="eastmoney_cn_etf_corporate_actions_extracted",
                        path=path.name,
                        status="downloaded",
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size,
                    )
                ],
            )
        )
    )
    action_root = tmp_path / "actions"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(raw),
            outputs=dict(public_domestic_etf_actions=str(action_root)),
        )
    )
    assert len(read_dataset(action_root)) == 3
    prices = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=days,
            open=100.0,
            close=100.0,
            high=100.0,
            low=100.0,
            volume=100.0,
            amount=10000.0,
        )
    )
    spec = interim_catalog().get("public_domestic_etf_prices")
    price_root = tmp_path / "prices"
    write_dataset(
        price_root,
        prices,
        spec=replace(
            spec,
            column_roles={name: spec.column_roles[name] for name in prices},
            column_units={
                name: unit for name, unit in spec.column_units.items() if name in prices
            },
            primary_key=("asset_id", "timestamp"),
        ),
    )
    calendar = _calendar_input(tmp_path / "calendar", {1: days})
    window = dict(start=days[0].date().isoformat(), end=days[-1].date().isoformat())
    panel_root = prepare(
        dict(
            market_id="a_share",
            data_id="etf_daily_panel",
            inputs=dict(
                public_domestic_etf_prices=str(price_root),
                public_domestic_etf_actions=str(action_root),
                trading_calendar=str(calendar),
            ),
            time_window=window,
            output_root=str(tmp_path / "panel"),
        )
    )
    panel = read_dataset(panel_root).set_index("signal_session")
    assert panel.loc[days[1], "period_return"] == pytest.approx(0.01)
    assert panel.loc[days[1], "return_available_at"] == pd.Timestamp("2024-01-03 16:00")
    assert pd.isna(panel.loc[days[3], "period_return"])
    assert pd.isna(panel.loc[days[3], "return_available_at"])
    assert panel.loc[days[4] :, "period_return"].eq(0).all()
    assert panel.loc[days[4], "return_available_at"] == days[4] + pd.Timedelta(
        hours=7, seconds=1
    )
    factors = read_dataset(
        prepare(
            dict(
                market_id="a_share",
                data_id="etf_price_volume_factors",
                inputs=dict(etf_daily_panel=str(panel_root)),
                factor_columns=["volatility_20d", "reversal_1d"],
                time_window=window,
                output_root=str(tmp_path / "factors"),
            )
        )
    )
    assert factors.volatility_20d.iloc[-1] == pytest.approx(0, abs=1e-14)
    assert factors.reversal_1d.iloc[3:].isna().all()


@pytest.mark.parametrize("empty", [False, True])
def test_native_market_states_follow_selected_signal_sessions(
    tmp_path, monkeypatch, empty
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        market_states,
    )
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    days = pd.date_range("2024-01-05", "2024-01-11")
    calendar = pd.DataFrame(
        [
            dict(
                calendar_date=day,
                source_market_type=market,
                is_trading_day=day.weekday() < 5,
            )
            for day in days
            for market in CURRENT_MARKET_TYPES
        ]
    )
    prices = pd.DataFrame(
        dict(
            asset_id="index:000300",
            timestamp=days[days.day_of_week < 5],
            open=10.0,
            close=11.0,
            volume=100.0,
            amount=1000.0,
        )
    )
    benchmark = daily_quotes.daily_decisions(prices, calendar)
    benchmark = benchmark.loc[benchmark.signal_session.ne(pd.Timestamp("2024-01-08"))]
    state = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                timestamp=pd.Timestamp(date),
                market_up_fraction=value,
            )
            for asset in ("a", "b", "c")
            for date, value in (
                ("2023-12-29", 0.2),
                ("2024-01-05", 0.5),
                ("2024-01-08", 0.99),
                ("2024-01-09", float("nan")),
                ("2024-02-01", 0.8),
            )
        ]
    )
    scanned = []
    original = market_states.iter_dataset_chunks

    def observe_scan(session):
        for frame in original(session):
            scanned.extend(frame.timestamp)
            yield frame

    monkeypatch.setattr(market_states, "iter_dataset_chunks", observe_scan)
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
            time_window=dict(start="2025-01-01", end="2025-01-02")
            if empty
            else dict(start="2024-01-08", end="2024-01-11"),
            output_root=str(tmp_path / "state"),
        )
    )
    result = read_dataset(path)
    if empty:
        assert result.empty
        assert not scanned
    else:
        assert result.timestamp.tolist() == benchmark.timestamp.tolist()
        assert result.market_up_fraction.iloc[0] == 0.5
        assert result.market_up_fraction.iloc[1:].isna().all()
        assert set(scanned) == set(
            pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
        )


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
