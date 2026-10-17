"""Funding cash uses settlement evidence while research NAV keeps causal book mid."""

from dataclasses import asdict, replace

import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    PortfolioEconomics,
    simulate_maker,
    simulate_taker,
)
from quant_research_platform.research.portfolios.diagnostics import crypto_statistics
from tests.helpers.binance_hft import _quote
from tests.markets.crypto.test_hft_execution import (
    _boundary,
    _maker_config,
    _state,
    _taker_config,
    _trade,
)

CAPITAL = PortfolioCapital("funding_basis", 1000.0, "USDT", "linear_perpetual")


def _run(style, *, direction=1, windows=True, events=None, emit=None, books=None):
    config = (
        replace(_taker_config(), taker_fee_bps=5.0)
        if style == "taker"
        else replace(_maker_config(), maker_fee_bps=2.0)
    )
    common = dict(
        capital=CAPITAL,
        asset_id="btc_usdt",
        observed_coverage={"s1": (0, 2_000_000_000)},
        funding_coverage=((0, 2_000_000_000),),
        funding_events=(
            FundingEvent(250_000_000, 0.001, 90.0),
            FundingEvent(300_000_000, -0.002, 80.0),
            FundingEvent(350_000_000, 0.0, None),
        )
        if events is None
        else events,
        funding_forecasts=(FundingForecast(0, 2_000_000_000, ()),),
        decision_windows_ns=((0, 400_000_000),) if windows else None,
        on_ledger=emit,
    )
    if books is None:
        books = [
            _state(0),
            _state(100),
            _state(200, bid_index=1095, ask_index=1105),
            _state(400),
        ]
    if style == "taker":
        return simulate_taker(
            books,
            [{"timestamp_ms": 0, "prediction": direction * 0.02}],
            prediction_column="prediction",
            prediction_horizon_ms=100,
            config=config,
            **common,
        ), config
    return simulate_maker(
        books,
        [
            _trade(
                100,
                price="99.4" if direction > 0 else "100.6",
                quantity="1",
                maker_buyer=direction > 0,
            )
        ],
        [_quote(0, book_segment_id="s1")],
        trade_coverage={"trade-connection-1": (0, 2_000_000_000)},
        config=config,
        **common,
    ), config


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("style", ["taker", "maker"])
@pytest.mark.parametrize("windows", [True, False])
def test_execution_funding_uses_latest_causal_mid_without_settlement_repricing(
    style, direction, windows
):
    batches = []
    result, _ = _run(style, direction=direction, windows=windows, emit=batches.append)
    settlements = [b for b in batches if any(f.kind == "funding" for f in b.cashflows)]
    assert [f.amount for b in settlements for f in b.cashflows] == pytest.approx(
        [-direction * 0.09, direction * 0.16]
    )
    assert all(
        v.mark_price == 110.0
        for b in settlements
        for v in b.valuations
        if v.asset_id == "btc_usdt" and v.phase != "before"
    )
    price = (
        (100.5 if direction > 0 else 99.5)
        if style == "taker"
        else (99.5 if direction > 0 else 100.5)
    )
    fee = price * (0.0005 if style == "taker" else 0.0002)
    gross = -0.5 if style == "taker" else 0.5
    expected = gross - fee + direction * 0.07
    assert result["account_path_status"] == "COMPLETE"
    assert result["funding"] == pytest.approx(direction * 0.07, abs=1e-15)
    assert result["gross_pnl"] == pytest.approx(gross, abs=1e-14)
    assert result["net_pnl"] == pytest.approx(expected, abs=1e-13)
    assert result["nav"] == pytest.approx(1000.0 + expected, abs=1e-12)
    trades = [t for b in batches for t in b.trades]
    assert len(trades) == 1 and trades[0].quantity == direction
    # Settlement processing precedes a quote arriving at that same instant.
    at_settlement = []
    _run(
        style,
        direction=direction,
        windows=windows,
        emit=at_settlement.append,
        books=[
            _state(0),
            _state(100),
            _state(200, bid_index=1095, ask_index=1105),
            _state(250, bid_index=1995, ask_index=2005),
            _state(400),
        ],
        events=(FundingEvent(250_000_000, 0.001, 90.0),),
    )
    funding = next(
        b for b in at_settlement if any(f.kind == "funding" for f in b.cashflows)
    )
    assert {
        v.mark_price
        for v in funding.valuations
        if v.asset_id == "btc_usdt" and v.phase != "before"
    } == {110.0}


@pytest.mark.parametrize("style", ["taker", "maker"])
@pytest.mark.parametrize("direction", [1, -1])
def test_persisted_funding_cash_nav_and_drawdown_share_one_basis(
    tmp_path, style, direction
):
    produced = []
    config = (
        replace(_taker_config(), taker_fee_bps=5.0)
        if style == "taker"
        else replace(_maker_config(), maker_fee_bps=2.0)
    )

    def produce(emit):
        result, _ = _run(style, direction=direction, emit=emit)
        produced.append(result)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=CAPITAL,
        metadata={
            "window": {
                "start": pd.Timestamp(0).isoformat(),
                "end": pd.Timestamp(400_000_000).isoformat(),
            },
            "sample_role": "prediction",
            "execution_mode": "continuous",
            "execution_style": style,
            "execution": asdict(config),
        },
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    batches = list(artifact.iter_chunks())
    summary = summarize_crypto_ledger(
        artifact, iter(batches), windows_ns=((0, 400_000_000),)
    )
    for name in ("net_pnl", "gross_pnl", "funding", "fees", "nav"):
        assert summary["economics"][name] == pytest.approx(produced[0][name], abs=1e-12)
    price = (
        (100.5 if direction > 0 else 99.5)
        if style == "taker"
        else (99.5 if direction > 0 else 100.5)
    )
    fee = price * (0.0005 if style == "taker" else 0.0002)
    base = 1000.0 + (-0.5 if style == "taker" else 0.5) - fee
    # Long: peak follows the funding credit; short: the 110 book creates the trough.
    expected_drawdown = (
        10.0 / (base + 10.07)
        if direction > 0
        else (max(1000.0, base) - (base - 10.07)) / max(1000.0, base)
    )
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance"],
            }
        )
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as table:
        rows = table.loc[table.episode_id.eq("continuous")].set_index("metric_id")
        assert rows.loc["net_pnl", "value"] == pytest.approx(
            base - 1000.0 + direction * 0.07, abs=1e-12
        )
        assert rows.loc["funding", "value"] == pytest.approx(
            direction * 0.07, abs=1e-15
        )
        assert rows.loc["continuous_max_drawdown", "status"] == "DEFINED"
        assert rows.loc["continuous_max_drawdown", "value"] == pytest.approx(
            -expected_drawdown, abs=1e-13
        )


def test_flat_book_funding_changes_only_cash_and_preserves_last_mid():
    batches = []
    account = PortfolioEconomics(CAPITAL, "btc_usdt", batches.append)
    account.fill(
        timestamp_ns=1,
        order_id="entry",
        quantity=1.0,
        price=100.0,
        mark_price=100.0,
        fee=0.0,
    )
    account.mark(2, 100.0)
    account.funding(FundingEvent(3, 0.001, 90.0), mark_price=account.last_mark)
    assert account.last_mark == 100.0
    account.mark(4, 100.0)
    navs = [sum(v.value for v in b.valuations if v.phase == "end") for b in batches]
    assert navs == pytest.approx([1000.0, 1000.0, 999.91, 999.91], abs=1e-12)
    assert account.funding_cashflows == pytest.approx(-0.09, abs=1e-16)


@pytest.mark.parametrize("style", ["taker", "maker"])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("recovery", ["known", "never", "unknown_rate", "unknown_mark"])
def test_unquoted_funding_preserves_cash_facts_and_exact_published_prefix(
    tmp_path, style, direction, recovery
):
    config = (
        replace(_taker_config(), taker_fee_bps=0.0)
        if style == "taker"
        else replace(_maker_config(), maker_fee_bps=0.0)
    )
    coverage = {"s1": (0, 600_000_000), "s2": (800_000_000, 1_001_000_000)}
    books = [_state(0), _state(100), _state(200), _boundary(600)]
    if recovery != "never":
        books.extend(
            [
                _state(800, bid_index=1095, ask_index=1105, segment_id="s2"),
                _state(1000, bid_index=1095, ask_index=1105, segment_id="s2"),
            ]
        )
    events = (
        FundingEvent(650_000_000, 0.001, 90.0),
        FundingEvent(
            700_000_000,
            None if recovery == "unknown_rate" else -0.002,
            None if recovery == "unknown_mark" else 80.0,
        ),
        FundingEvent(750_000_000, 0.0, None),
        FundingEvent(800_000_000, 0.001, 110.0),
    )
    results = []

    def produce(emit):
        common = dict(
            capital=CAPITAL,
            asset_id="btc_usdt",
            config=config,
            observed_coverage=coverage,
            funding_coverage=((0, 1_001_000_000),),
            funding_events=events,
            funding_forecasts=(FundingForecast(0, 2_000_000_000, ()),),
            decision_windows_ns=((0, 1_000_000_000),),
            on_ledger=emit,
        )
        if style == "taker":
            result = simulate_taker(
                books,
                [
                    {"timestamp_ms": 0, "prediction": direction * 0.02},
                    *(
                        [{"timestamp_ms": 800, "prediction": -direction * 0.02}]
                        if recovery == "known"
                        else []
                    ),
                ],
                prediction_column="prediction",
                prediction_horizon_ms=100,
                **common,
            )
        else:
            result = simulate_maker(
                books,
                [
                    _trade(
                        100,
                        price="99.4" if direction > 0 else "100.6",
                        quantity="1",
                        maker_buyer=direction > 0,
                    ),
                    *(
                        [
                            _trade(
                                900,
                                price="110.6" if direction > 0 else "109.4",
                                quantity="1",
                                maker_buyer=direction < 0,
                            )
                        ]
                        if recovery == "known"
                        else []
                    ),
                ],
                [
                    _quote(0, book_segment_id="s1"),
                    *(
                        [_quote(800, book_segment_id="s2")]
                        if recovery == "known"
                        else []
                    ),
                ],
                trade_coverage={"trade-connection-1": (0, 1_001_000_000)},
                **common,
            )
        results.append(result)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=CAPITAL,
        metadata={
            "window": {
                "start": pd.Timestamp(0).isoformat(),
                "end": pd.Timestamp(1_000_000_000).isoformat(),
            },
            "sample_role": "prediction",
            "execution_mode": "continuous",
            "execution_style": style,
            "execution": asdict(config),
        },
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    batches = list(artifact.iter_chunks())
    summary = summarize_crypto_ledger(
        artifact, iter(batches), windows_ns=((0, 1_000_000_000),)
    )
    result = results[0]
    assert all(
        not 600_000_000 <= v.timestamp_ns < 800_000_000
        for b in batches
        for v in b.valuations
    )
    assert result["market_observation_status"] == "PARTIAL"
    gross = -0.5 if style == "taker" else 0.5
    if recovery == "known":
        flows = [f for b in batches for f in b.cashflows if f.kind == "funding"]
        assert [f.timestamp_ns for f in flows] == [
            650_000_000,
            700_000_000,
            800_000_000,
        ]
        assert [f.amount for f in flows] == pytest.approx(
            [-direction * 0.09, direction * 0.16, -direction * 0.11]
        )
        funded = next(
            b for b in batches if any(f.kind == "funding" for f in b.cashflows)
        )
        assert {v.timestamp_ns for v in funded.valuations if v.phase == "before"} == {
            200_000_000
        }
        assert {v.timestamp_ns for v in funded.valuations if v.phase != "before"} == {
            800_000_000
        }
        assert result["account_path_status"] == "COMPLETE"
        assert result["net_pnl"] == pytest.approx(
            2 * gross + direction * 9.96, abs=1e-12
        )
        fills = [b for b in batches if b.trades]
        assert len(fills) == 2
        assert sum(t.quantity for b in fills for t in b.trades) == 0
        assert funded.sequence < fills[-1].sequence
        before_cash = next(
            v.value
            for v in fills[-1].valuations
            if v.phase == "before" and v.asset_id == PORTFOLIO_CASH_ASSET_ID
        )
        assert before_cash == pytest.approx(1000.0 - direction * 0.04, abs=1e-12)
        assert result["unvalued_funding_event_count"] == 0
        assert result["unvalued_funding_cashflow"] == 0.0
        assert summary["economics"]["funding"] == pytest.approx(
            -direction * 0.04, abs=1e-15
        )
    else:
        assert result["account_path_status"] == "PARTIAL"
        assert result["net_pnl"] is None and summary["economics"]["nav"] is None
        assert result["known_prefix_valuation_timestamp_ns"] == 200_000_000
        assert result["known_prefix_net_pnl"] == gross
        expected_count = 3 if recovery == "never" else 1
        expected_funding = -direction * (0.04 if recovery == "never" else 0.09)
        assert result["unvalued_funding_event_count"] == expected_count
        assert result["unvalued_funding_cashflow"] == pytest.approx(
            expected_funding, abs=1e-15
        )
        assert not any(f.kind == "funding" for b in batches for f in b.cashflows)
        assert (
            summary["execution"]["unvalued_funding_cashflow"]
            == result["unvalued_funding_cashflow"]
        )
        assert all(v.timestamp_ns <= 200_000_000 for b in batches for v in b.valuations)
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance", "execution"],
            }
        )
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as table:
        rows = table.loc[table.episode_id.eq("continuous")].set_index("metric_id")
        assert rows.loc["continuous_max_drawdown", "status"] == "UNDEFINED"
        assert rows.loc["terminal_nav", "status"] == (
            "DEFINED" if recovery == "known" else "UNDEFINED"
        )
    with statistic.table("crypto_execution_diagnostics.parquet") as table:
        rows = table.loc[table.episode_id.eq("continuous")].set_index("metric_id")
        assert (
            rows.loc["unvalued_funding_event_count", "value"]
            == result["unvalued_funding_event_count"]
        )
        assert (
            rows.loc["unvalued_funding_cashflow", "value"]
            == result["unvalued_funding_cashflow"]
        )


@pytest.mark.parametrize("style", ["taker", "maker"])
@pytest.mark.parametrize(
    "rate,mark", [(None, 90.0), (0.001, None), (0.0, None), (0.0, 70.0)]
)
def test_unknown_settlement_is_distinct_from_known_zero_funding(style, rate, mark):
    batches = []
    result, _ = _run(
        style, events=(FundingEvent(250_000_000, rate, mark),), emit=batches.append
    )
    if rate == 0.0:
        assert result["account_path_status"] == "COMPLETE"
        assert result["funding"] == 0.0
        assert result["unvalued_funding_event_count"] == 0
    else:
        assert result["account_path_status"] == "PARTIAL"
        assert result["account_stop_reason"] == "portfolio_required_funding_missing"
        assert result["account_stop_ns"] == 250_000_000
        assert result["known_prefix_valuation_timestamp_ns"] == 200_000_000
        assert result["funding"] is None
    assert not any(f.kind == "funding" for b in batches for f in b.cashflows)
