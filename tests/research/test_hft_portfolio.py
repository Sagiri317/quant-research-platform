"""Shared cash, trading-day locks and actual corporate rights in quote accounts."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeRule,
    FeeSchedule,
)
from quant_research_platform.research.portfolios.high_frequency import (
    _parameters,
    inventory_batches,
)
from tests.helpers.high_frequency import daily_view, write_daily_inputs


def quotes(
    day, assets=("a", "b"), clocks=("01:30:00", "01:30:03", "01:30:06"), mid=10.0
):
    return pd.DataFrame(
        [
            dict(
                timestamp=pd.Timestamp(f"{day}T{clock}"),
                session_date=pd.Timestamp(day),
                asset_id=asset,
                exchange="SZSE",
                eligible=True,
                segment_id=0,
                mid=mid,
                bid_px1=mid - 0.01,
                ask_px1=mid + 0.01,
                bid_vol1=500.0,
                ask_vol1=500.0,
            )
            for clock in clocks
            for asset in assets
        ]
    )


def parameters(assets=("a", "b"), *, restore_at=None):
    return _parameters(
        dict(
            initial_capital=5000.0,
            initial_inventory=dict.fromkeys(assets, 100),
            maximum_inventory=dict.fromkeys(assets, 300),
            buy_threshold=0.5,
            sell_threshold=-0.5,
            latency_ms=0.0,
            participation_rate=1.0,
            minimum_commission=0.0,
            restore_at=restore_at,
        )
    )


def schedule():
    return FeeSchedule(
        tuple(
            FeeRule(
                code,
                date(2024, 1, 1),
                date(2024, 12, 31),
                "a_share",
                side,
                "amount",
                0.0,
                True,
            )
            for side in ("buy", "sell")
            for code in ("BROKER_COMMISSION", "TRANSFER_FEE", "STAMP_DUTY")
            if code != "STAMP_DUTY" or side == "sell"
        ),
        "synthetic-fees",
    )


def native_account(tmp_path, sessions, params, window):
    capital = PortfolioCapital(
        "hft_inventory", params["initial_capital"], "CNY", "cash_equity"
    )

    def produce(emit):
        for batch in inventory_batches(sessions, schedule(), params, window):
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "kind": "a_share_hft",
        },
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    batches = list(artifact.iter_chunks())
    return batches, summarize_portfolio_ledger(capital, iter(batches))


def test_shared_cash_and_t1_inventory_continue_across_days(tmp_path):
    first = quotes("2024-01-02")
    second = quotes("2024-01-03")
    signals = first[["asset_id", "timestamp"]].assign(
        prediction_score=[1, 1, -1, -1, -1, -1]
    )
    next_signals = second[["asset_id", "timestamp"]].assign(prediction_score=-1.0)
    batches, summary = native_account(
        tmp_path,
        [
            (daily_view(first), first, signals),
            (
                daily_view(second, listing=pd.Timestamp("2024-01-02")),
                second,
                next_signals,
            ),
        ],
        parameters(),
        TimeWindow("2024-01-02", "2024-01-03"),
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [100, 100, -100, -100, -100, -100]
    assert all(trade.timestamp_ns > int(trade.order_id) for trade in trades)
    closes = [
        batch
        for batch in batches
        if any(row.text == "close" for row in batch.diagnostics)
    ]
    held = [
        {row.asset_id: row.quantity for row in batch.valuations if row.phase == "end"}
        for batch in closes
    ]
    assert held[0]["a"] == held[0]["b"] == 100
    assert held[1]["a"] == held[1]["b"] == 0
    assert summary["nav"] == pytest.approx(4994.0)
    assert summary["net_pnl"] == pytest.approx(-6.0)


@pytest.mark.parametrize(
    "restore,expected_quantity,expected_nav",
    [(None, 0, 4999.0), ("14:50", 100, 4998.0)],
)
def test_restoration_uses_declared_clock_and_is_optional(
    tmp_path, restore, expected_quantity, expected_nav
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "06:50:00", "06:50:03"),
    )
    signals = panel[["asset_id", "timestamp"]].assign(
        prediction_score=[-1.0, 0.0, 0.0, 0.0]
    )
    batches, summary = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        parameters(("a",), restore_at=restore),
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    assert (
        next(
            row.quantity
            for row in batches[-1].valuations
            if row.asset_id == "a" and row.phase == "end"
        )
        == expected_quantity
    )
    assert summary["nav"] == pytest.approx(expected_nav)
    if restore:
        last_trade = [trade for batch in batches for trade in batch.trades][-1]
        assert last_trade.timestamp_ns == pd.Timestamp("2024-01-02T06:50:03").value
        assert int(last_trade.order_id) == pd.Timestamp("2024-01-02T06:50:00").value
        assert any(row.text == "COMPLETE" for row in batches[-1].diagnostics)


def test_t1_prevents_fabricated_end_of_day_restoration(tmp_path):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "06:50:00", "06:50:03"),
    )
    signals = panel[["asset_id", "timestamp"]].assign(
        prediction_score=[1.0, 0.0, 0.0, 0.0]
    )
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        parameters(("a",), restore_at="14:50"),
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    assert [trade.quantity for batch in batches for trade in batch.trades] == [
        200,
        -100,
    ]
    assert (
        next(
            row.quantity
            for row in batches[-1].valuations
            if row.asset_id == "a" and row.phase == "end"
        )
        == 200
    )
    assert any(row.text == "INCOMPLETE" for row in batches[-1].diagnostics)


def test_registered_dividend_remains_a_claim_until_paid(tmp_path):
    first = pd.Timestamp("2024-01-02")
    event = pd.DataFrame(
        [
            dict(
                event_id="cash",
                asset_id="a",
                listing_date=first,
                kind="cash_dividend",
                record_at=first,
                effective_at=pd.Timestamp("2024-01-03"),
                settlement_at=pd.Timestamp("2024-01-04"),
                available_at=first,
                cash_per_share=1.0,
                additional_shares_per_share=0.0,
                evidence_identity="a" * 64,
                entitlement_proved=True,
            )
        ]
    )
    sessions = []
    for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
        panel = quotes(day, assets=("a",), mid=10.0 if day == "2024-01-02" else 9.0)
        signals = panel[["asset_id", "timestamp"]].assign(prediction_score=0.0)
        sessions.append(
            (daily_view(panel, listing=first, corporate_events=event), panel, signals)
        )
    batches, summary = native_account(
        tmp_path, sessions, parameters(("a",)), TimeWindow("2024-01-02", "2024-01-04")
    )
    closes = [
        batch
        for batch in batches
        if any(row.text == "close" for row in batch.diagnostics)
    ]
    claims = [
        sum(row.receivable for row in batch.valuations if row.phase == "end")
        for batch in closes
    ]
    assert claims == [0.0, 100.0, 0.0]
    assert [
        (flow.kind, flow.amount) for batch in batches for flow in batch.cashflows
    ] == [("dividend", 100.0)]
    assert summary["net_pnl"] == pytest.approx(0.0)
    assert summary["nav"] == pytest.approx(5000.0)
    assert all(
        sum(row.value for row in batch.valuations if row.phase == "end")
        == pytest.approx(5000.0)
        for batch in batches
    )


def test_quote_account_distribution_keeps_integer_inventory_and_rounding_loss(tmp_path):
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )
    from tests.research.test_raw_share_accounting import _events

    event = _events(stock=True, pay="2024-01-04")
    event["additional_shares_per_share"] = 0.105
    sessions = []
    for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
        price = 10.0 if day == "2024-01-02" else 10.0 / 1.105
        panel = quotes(day, assets=("a",), mid=price)
        signals = panel[["asset_id", "timestamp"]].assign(prediction_score=0.0)
        sessions.append(
            (
                daily_view(
                    panel, listing=pd.Timestamp("2000-01-01"), corporate_events=event
                ),
                panel,
                signals,
            )
        )
    batches, summary = native_account(
        tmp_path, sessions, parameters(("a",)), TimeWindow("2024-01-02", "2024-01-04")
    )
    positions = [
        row
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "a" and row.phase == "end"
    ]
    assert positions[-1].quantity == 110
    assert positions[-1].locked_quantity == 0
    assert all(
        row.quantity.is_integer() and row.locked_quantity.is_integer()
        for row in positions
    )
    assert any(row.locked_quantity == 10 for row in positions)
    assert summary["equity_rounding_pnl"] == pytest.approx(-0.5 * 10 / 1.105)
    assert summary["net_pnl"] == pytest.approx(-0.5 * 10 / 1.105)
    root = portfolio_statistics(
        {
            "ledgers": [str(tmp_path / "ledger")],
            "output": str(tmp_path / "statistics"),
            "window": {"start": "2024-01-02", "end": "2024-01-04"},
            "analyses": ["performance"],
        }
    )
    with read_statistics_artifact(root).table("portfolio_performance.parquet") as table:
        performance = table.set_index("metric_id")
        assert performance.loc["equity_rounding_pnl", "status"] == "DEFINED"
        assert performance.loc["equity_rounding_pnl", "value"] == pytest.approx(
            -0.5 * 10 / 1.105
        )
        assert performance.loc["net_pnl", "value"] == pytest.approx(-0.5 * 10 / 1.105)


@pytest.mark.parametrize("partial", [False, True])
def test_quote_ledger_statistics_use_daily_closes_and_native_query(tmp_path, partial):
    import matplotlib

    matplotlib.use("Agg")
    from quant_research_platform.artifacts.portfolio_curve import portfolio_curve
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.reports.plots import plot
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )

    panels = [
        quotes(day, assets=("a",), mid=price)
        for day, price in (
            ("2024-01-02", 10.0),
            ("2024-01-03", 11.0),
            ("2024-01-04", 12.0),
        )
    ]
    end = "2024-01-04T04:00:00" if partial else "2024-01-04"
    start = "2024-01-02T01:30:00" if partial else "2024-01-02"
    window = TimeWindow(start, end)
    native_account(
        tmp_path,
        [
            (
                daily_view(panel, listing=pd.Timestamp("2024-01-02")),
                panel,
                panel[["asset_id", "timestamp"]].assign(prediction_score=0.0),
            )
            for panel in panels
        ],
        parameters(("a",)),
        window,
    )
    inputs = write_daily_inputs(
        tmp_path / "daily", pd.concat(panels, ignore_index=True)
    )
    root = portfolio_statistics(
        {
            "ledgers": [str(tmp_path / "ledger")],
            "output": str(tmp_path / "statistics"),
            "window": {"start": start, "end": end},
            "analyses": ["performance"],
            "inputs": {"calendar": inputs["calendar"]},
        }
    )
    artifact = read_statistics_artifact(root)
    with artifact.table("portfolio_daily.parquet") as daily:
        daily = daily.copy()
    with artifact.table("portfolio_performance.parquet") as summary:
        summary = summary.set_index("metric_id")
    expected = np.array([0.0, 100 / 5000, 100 / 5100])
    assert daily.net_return.to_numpy() == pytest.approx(expected)
    assert daily.nav_end.tolist() == [5000.0, 5100.0, 5200.0]
    assert summary.observation_count.eq(3).all()
    assert summary.loc["net_pnl", "value"] == pytest.approx(200.0)
    if partial:
        assert summary.loc["sharpe", "reason"] == "incomplete_daily_axis"
        assert pd.isna(summary.loc["annualized_volatility", "value"])
    else:
        assert summary.loc["annualized_volatility", "value"] == pytest.approx(
            expected.std(ddof=1) * np.sqrt(252)
        )
        assert summary.loc["sharpe", "value"] == pytest.approx(
            expected.mean() / expected.std(ddof=1) * np.sqrt(252)
        )
        assert summary.loc["cagr", "value"] == pytest.approx(1.04**84 - 1)
    curve = portfolio_curve(tmp_path / "ledger")
    assert curve["points"][-1]["nav"] == 5200.0
    image = plot(
        {
            "kind": "curve",
            "inputs": str(tmp_path / "ledger"),
            "output": str(tmp_path / "curve.png"),
            "sample_role": "exploratory",
            "timezone": "Asia/Shanghai",
        }
    )
    assert image.stat().st_size > 1000
