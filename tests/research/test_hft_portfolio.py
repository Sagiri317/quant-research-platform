"""Shared cash, trading-day locks and actual corporate rights in quote accounts."""

from dataclasses import replace
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
    FeeSchedule,
)
from quant_research_platform.research.portfolios.high_frequency import (
    _parameters,
    inventory_batches,
)
from tests.helpers.high_frequency import (
    daily_view,
    parameters,
    quotes,
    schedule,
    write_daily_inputs,
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


@pytest.mark.parametrize(
    "instrument_type,expected", [("equity_etf", [200, -100]), ("gold_etf", [200, -200])]
)
def test_etf_settlement_fees_and_restore_in_native_ledger(
    tmp_path, instrument_type, expected
):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:03", "06:50:00", "06:50:03"),
    ).assign(instrument_type=instrument_type)
    signals = panel[["asset_id", "timestamp"]].assign(
        prediction_score=[1.0, 0.0, 0.0, 0.0]
    )
    fees = FeeSchedule(
        tuple(
            replace(
                rule,
                market="etf",
                rate=0.0001 if rule.fee_code == "BROKER_COMMISSION" else 0.0,
            )
            for rule in schedule().rules
        ),
        "etf-fees",
    )
    params = parameters(("a",), restore_at="14:50")
    params["minimum_commission"] = 5.0

    def produce(emit):
        for batch in inventory_batches(
            [(daily_view(panel), panel, signals)],
            fees,
            params,
            TimeWindow("2024-01-02", "2024-01-02"),
        ):
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("hft_inventory", 5000.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2024-01-02", "end": "2024-01-02"}},
        produce=produce,
    )
    batches = list(read_portfolio_ledger_artifact(root).iter_chunks())
    assert [trade.quantity for batch in batches for trade in batch.trades] == expected
    flows = [flow for batch in batches for flow in batch.cashflows]
    assert sum(flow.amount for flow in flows if flow.kind == "commission_cost") == -10.0
    assert not any(
        flow.kind in {"stamp_duty_cost", "transfer_fee_cost"} for flow in flows
    )
    # Two bid/ask crossings cost quantity * 0.01, plus two 5 CNY commissions.
    nav = sum(row.value for row in batches[-1].valuations if row.phase == "end")
    assert nav == pytest.approx(5000 - sum(abs(q) for q in expected) * 0.01 - 10)


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


@pytest.mark.parametrize(
    "levels,rate,first_depth,second_depth,expected,notional",
    [
        (1, 1.0, 100.0, 300.0, 100, 1000.0),
        (2, 1.0, 100.0, 300.0, 200, 2100.0),
        (2, 0.05, 1000.0, 3000.0, 200, 2150.0),
    ],
)
def test_taker_depth_and_participation_consume_each_level_once(
    tmp_path, levels, rate, first_depth, second_depth, expected, notional
):
    panel = quotes("2024-01-02", assets=("a",)).assign(
        ask_px1=10.0, ask_vol1=first_depth, ask_px2=11.0, ask_vol2=second_depth
    )
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    params = parameters(("a",))
    params.update(
        taker_depth_levels=levels, participation_rate=rate, minimum_commission=5.0
    )
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [expected]
    assert [trade.notional for trade in trades] == pytest.approx([notional])
    assert (
        sum(
            -flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "commission_cost"
        )
        == 5.0
    )
    facts = {
        row.name: row.number
        for batch in batches
        for row in batch.diagnostics
        if row.name in {"requested_quantity", "unfilled_quantity"}
    }
    assert facts == {"requested_quantity": 200.0, "unfilled_quantity": 200 - expected}


def test_depth_reprices_cash_limited_quantity_and_preserves_shared_budget(tmp_path):
    panel = quotes("2024-01-02").assign(
        ask_px1=10.0, ask_vol1=100.0, ask_px2=20.0, ask_vol2=100.0
    )
    signals = panel.loc[
        panel.timestamp.eq(panel.timestamp.min()), ["timestamp", "asset_id"]
    ].assign(prediction_score=1.0)
    params = parameters()
    # After two initial 100-share holdings, the common cash is exactly 2010.
    params.update(taker_depth_levels=2, initial_capital=4010.0, minimum_commission=5.0)
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [(trade.asset_id, trade.quantity) for trade in trades] == [
        ("a", 100),
        ("b", 100),
    ]
    # Reduced orders consume only the first level, never the full-order VWAP of 15.
    assert [trade.notional for trade in trades] == [1000.0, 1000.0]
    assert min(
        row.value
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "__cash__"
    ) == pytest.approx(0.0)


def test_depth_sell_uses_descending_prices_and_existing_sellability(tmp_path):
    panel = quotes("2024-01-02", assets=("a",)).assign(
        bid_px1=10.0, bid_vol1=100.0, bid_px2=9.0, bid_vol2=100.0
    )
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=-1.0)
    params = parameters(("a",))
    params.update(taker_depth_levels=2, initial_inventory={"a": 300})
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [-200]
    assert [trade.notional for trade in trades] == [-1900.0]
    assert any(
        row.name == "unfilled_quantity" and row.number == -100
        for batch in batches
        for row in batch.diagnostics
    )


def test_depth_commissions_follow_actual_legal_child_notionals():
    from quant_research_platform.contracts.portfolios import ExecutionAssumptions
    from quant_research_platform.research.portfolios.accounting import (
        execute_stock_target,
    )

    orders = pd.DataFrame(
        {
            "quantity": [0],
            "target_quantity": [1_000_100],
            "sellable_quantity": [0],
            "market_type": [4],
            "mark_price": [1.0],
            "buy_price": [0.2],
            "sell_price": [0.1],
            "buy_capacity": [1_000_000.0],
            "sell_capacity": [0.0],
            "adv": [np.nan],
            "can_enter": [True],
            "can_exit": [True],
        },
        index=pd.Index(["a"], name="asset_id"),
    )
    fees = FeeSchedule(
        tuple(
            replace(rule, rate=0.00001 if rule.fee_code == "BROKER_COMMISSION" else 0.0)
            for rule in schedule().rules
        ),
        "child-order-fees",
    )
    account = execute_stock_target(
        orders,
        cash=2_000_000.0,
        execution_date=date(2024, 1, 2),
        fee_schedule=fees,
        assumptions=ExecutionAssumptions(2_000_000.0, 0.0, 0.0, 1.0, (1.0,), 5.0),
        depth={"a": np.array([[0.2, 1_000_000.0], [10_000.0, 100.0]])},
    )
    trade = account.asset_ledger.iloc[0]
    assert trade.executed_quantity == 1_000_100
    assert trade.executed_delta_notional == pytest.approx(1_200_000.0)
    # First legal order: 200000 * 1e-5 floors to 5; last: 1000000 * 1e-5 = 10.
    assert trade.commission_cost == 15.0
    assert account.post_trade_cash == pytest.approx(799985.0)


@pytest.mark.parametrize("fault", ["unknown", "duplicate", "reversed"])
def test_depth_does_not_skip_unknown_or_reuse_invalid_levels(tmp_path, fault):
    panel = quotes("2024-01-02", assets=("a",)).assign(
        ask_px1=10.0,
        ask_vol1=100.0,
        ask_px2=11.0,
        ask_vol2=100.0,
        ask_px3=12.0,
        ask_vol3=100.0,
    )
    panel["ask_px2"] = {"unknown": np.nan, "duplicate": 10.0, "reversed": 9.0}[fault]
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    params = parameters(("a",))
    params["taker_depth_levels"] = 3

    def run():
        return native_account(
            tmp_path,
            [(daily_view(panel), panel, signals)],
            params,
            TimeWindow("2024-01-02", "2024-01-02"),
        )

    if fault == "unknown":
        batches, _ = run()
        assert [trade.quantity for batch in batches for trade in batch.trades] == [100]
    else:
        with pytest.raises(ValueError, match="execution_depth_price_order_invalid"):
            run()


@pytest.mark.parametrize("terminal", ["observed", "gap", "too_late"])
def test_markout_uses_only_observed_future_quotes_in_same_segment(tmp_path, terminal):
    panel = quotes("2024-01-02", assets=("a",))
    panel.loc[2, ["mid", "bid_px1", "ask_px1"]] = [11.0, 10.99, 11.01]
    if terminal == "gap":
        panel.loc[2, "segment_id"] = 1
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    params = parameters(("a",))
    if terminal == "observed":
        panel.loc[2, "eligible"] = False
        panel.loc[2, "ask_vol1"] = np.nan
    params["markout_ms"] = 10_000.0 if terminal == "too_late" else 1000.0
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    trade = next(trade for batch in batches for trade in batch.trades)
    facts = [
        row
        for batch in batches
        for row in batch.diagnostics
        if row.name.startswith("markout_")
    ]
    assert all(row.fill_id == trade.fill_id for row in facts)
    if terminal == "observed":
        result = {row.name: row for row in facts}
        assert result["markout_bps"].number == pytest.approx((11.0 / 10.01 - 1) * 10000)
        assert result["markout_observed_at_ns"].text == str(
            panel.timestamp.iloc[2].value
        )
    else:
        assert [(row.name, row.text) for row in facts] == [
            (
                "markout_reason",
                "unobserved_segment" if terminal == "gap" else "no_observed_markout",
            )
        ]


@pytest.mark.parametrize(
    "setting,value,error",
    [
        ("taker_depth_levels", value, "hft_taker_depth_levels_invalid")
        for value in (0, 11, True, 1.5, "2")
    ]
    + [("markout_ms", value, "hft_parameter_invalid") for value in (0, -1, True)]
    + [("markout_ms", np.inf, "qrp_json_nonfinite_number")],
)
def test_execution_capacity_parameters_are_explicit(setting, value, error):
    with pytest.raises(ValueError, match=error):
        _parameters({**parameters(("a",)), setting: value})


@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("missing", ["ask_vol1", "bid_vol1", "ask_vol2"])
def test_taker_checks_its_own_depth_when_price_eligibility_remains_true(
    tmp_path, levels, missing
):
    panel = quotes("2024-01-02", assets=("a",)).assign(
        ask_vol1=100.0, ask_px2=11.0, ask_vol2=100.0
    )
    panel[missing] = np.nan
    panel["eligible"] = False
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=1.0)
    params = parameters(("a",))
    params["taker_depth_levels"] = levels
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    expected = (
        0
        if missing == "ask_vol1"
        else 200
        if levels == 2 and missing == "bid_vol1"
        else 100
    )
    assert (
        sum(trade.quantity for batch in batches for trade in batch.trades) == expected
    )


@pytest.mark.parametrize(
    "missing", [None, "availability", "holding_mark", "observations", "book_status"]
)
def test_inventory_risk_uses_full_cross_section_and_keeps_account_economics(
    tmp_path, missing
):
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.research.portfolios.attribution import (
        inventory_risk_attribution,
    )

    model = FactorRiskModel()
    assets = tuple(f"s{number:03d}" for number in range(128))
    listing = pd.Timestamp("2020-01-01")
    panel = quotes("2024-01-02", assets=assets)
    rng = np.random.default_rng(143)
    raw = rng.normal(size=(len(assets), len(model.style_columns)))
    cap = np.exp(np.linspace(17, 24, len(assets)))
    median = np.median(raw, axis=0)
    radius = 3 * 1.4826 * np.median(np.abs(raw - median), axis=0)
    clipped = np.clip(raw, median - radius, median + radius)
    styles = (clipped - clipped.mean(axis=0)) / clipped.std(axis=0)
    industries = np.column_stack((np.arange(128) < 64, np.arange(128) >= 64))
    design = np.column_stack((industries, styles))
    returns = [rng.normal(0, 0.002, 128) for _ in range(2)]
    prices = np.vstack((np.full(128, 10.0), np.zeros((2, 128))))
    for index, response in enumerate(returns):
        prices[index + 1] = prices[index] * (1 + response)
    panel["mid"] = prices.ravel()
    panel["bid_px1"], panel["ask_px1"] = panel.mid - 0.01, panel.mid + 0.01
    panel = panel.assign(market_id="a_share", listing_date=listing)
    risk_facts = pd.DataFrame(raw, columns=model.style_columns).assign(
        market_id="a_share",
        asset_id=assets,
        timestamp=pd.Timestamp("2023-12-29"),
        listing_date=listing,
        risk_available_at=pd.Timestamp("2024-01-02T01:00:00"),
        industry_standard_id="fixture",
        industry_level1_code=np.where(industries[:, 0], "one", "two"),
        total_market_cap=cap,
    )
    signals = panel.loc[panel.asset_id.eq(assets[0]), ["timestamp", "asset_id"]].assign(
        prediction_score=[1.0, 0.0, 0.0]
    )
    params = parameters((assets[0],))
    params["minimum_commission"] = 2.0
    view = daily_view(panel, listing=listing)
    native_account(
        tmp_path,
        [(view, panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    ledger = read_portfolio_ledger_artifact(tmp_path / "ledger")
    observations = panel.copy()
    if missing == "availability":
        risk_facts["risk_available_at"] = pd.NaT
    elif missing == "holding_mark":
        observations.loc[
            observations.asset_id.eq(assets[0])
            & observations.timestamp.eq(pd.Timestamp("2024-01-02T01:30:03")),
            "mid",
        ] += 0.1
    daily, _, tables = inventory_risk_attribution(
        ledger,
        model=model,
        observations=iter([observations.iloc[:0]])
        if missing == "observations"
        else (
            frame.drop(columns="eligible") if missing == "book_status" else frame
            for _, frame in observations.groupby("timestamp")
        ),
        risk_facts=risk_facts,
        available_at_column="risk_available_at",
        daily_views=[view],
    )
    result = tables["portfolio_attribution"].set_index("component_id")
    assert result.loc["cost", "value"] == pytest.approx(-2 / 5000)
    assert result.loc["execution_to_mark", "value"] == pytest.approx(-2 / 5000)
    assert result.loc["inventory_price_pnl", "value"] == pytest.approx(
        (100 * (prices[1, 0] - prices[0, 0]) + 300 * (prices[2, 0] - prices[1, 0]))
        / 5000
    )
    if missing:
        assert (
            result.loc[["industry", "style", "residual_unexplained"], "value"]
            .isna()
            .all()
        )
        assert result.loc["cost", "status"] == "DEFINED"
        if missing == "observations":
            return
    else:
        root_weight = cap**0.25
        expected = {"industry": 0.0, "style": 0.0, "residual_unexplained": 0.0}
        for index, response in enumerate(returns):
            coefficients = np.linalg.lstsq(
                design * root_weight[:, None], response * root_weight, rcond=None
            )[0]
            amount = (100 if index == 0 else 300) * prices[index, 0] / 5000
            expected["industry"] += amount * design[0, :2].dot(coefficients[:2])
            expected["style"] += amount * design[0, 2:].dot(coefficients[2:])
            expected["residual_unexplained"] += amount * (
                response[0] - design[0].dot(coefficients)
            )
        for name, value in expected.items():
            assert result.loc[name, "value"] == pytest.approx(
                value, rel=1e-9, abs=1e-12
            )
        assert result.loc[
            result.component_type.eq("additive"), "value"
        ].sum() == pytest.approx(daily.net_pnl.iloc[0] / 5000, abs=1e-12)
        assert tables["holding_factor_returns"].observation_count.eq(128).all()

    # Persisted statistics use the same explicit sources and continuous clock.
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )
    from quant_research_platform.storage.datasets import write_dataset

    closing = observations.loc[
        observations.timestamp.eq(observations.timestamp.max())
    ].assign(timestamp=pd.Timestamp("2024-01-02T07:00:00"))
    observations = pd.concat([observations, closing], ignore_index=True)
    for name, frame, frequency in (
        ("observations", observations, "tick"),
        ("daily_risk", risk_facts, "1d"),
    ):
        roles = dict.fromkeys(frame.columns, "context")
        roles.update(market_id="key", asset_id="key", timestamp="timestamp")
        if "eligible" in roles:
            roles["eligible"] = "context" if missing == "book_status" else "universe"
        write_dataset(
            tmp_path / name,
            frame,
            spec=ProcessedDataSpec(
                data_id=name,
                kind="panel",
                market_id="a_share",
                frequency=frequency,
                grain="asset_timestamp",
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=(),
                column_roles=roles,
            ),
        )
    output = portfolio_statistics(
        {
            "ledgers": [str(tmp_path / "ledger")],
            "output": str(tmp_path / "risk-statistics"),
            "window": {"start": "2024-01-02", "end": "2024-01-02"},
            "analyses": ["performance", "attribution"],
            "inputs": {
                "observations": str(tmp_path / "observations"),
                "daily_risk": str(tmp_path / "daily_risk"),
            },
            "risk_exposure_input": {
                "data_id": "daily_risk",
                "available_at_column": "risk_available_at",
            },
            "risk_model": model.to_dict(),
        }
    )
    artifact = read_statistics_artifact(output)
    with artifact.table("portfolio_attribution.parquet") as persisted:
        assert (
            persisted.set_index("component_id").status.reindex(result.index).tolist()
            == result.status.tolist()
        )
        assert persisted.decision_at.eq(persisted.entry_at).all()
        np.testing.assert_allclose(
            persisted.set_index("component_id").value.reindex(result.index).to_numpy(),
            result.value.to_numpy(),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )
    with artifact.table("holding_factor_returns.parquet") as persisted:
        assert persisted.observation_count.eq(128).all()
        assert persisted.decision_at.eq(persisted.entry_at).all()


@pytest.mark.parametrize(
    "latency_ms,decision_ms,execution_ms,gap",
    [
        (None, 1, 11, False),
        (20, 1, 21, False),
        (0, 5, 10, False),
        (None, 1, None, True),
    ],
)
def test_order_activation_uses_decision_clock_once(
    tmp_path, latency_ms, decision_ms, execution_ms, gap
):
    origin = pd.Timestamp("2024-01-02T01:30:00")
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=tuple(f"01:30:00.{ms:03d}" for ms in (0, 5, 10, 11, 12, 21, 22)),
    )
    if gap:
        panel.loc[
            panel.timestamp.ge(origin + pd.Timedelta(milliseconds=10)), "segment_id"
        ] = 1
    params = parameters(("a",))
    params.pop("latency_ms")
    if latency_ms is not None:
        params["latency_ms"] = latency_ms
    params = _parameters(params)
    assert params["latency_ms"] == (10 if latency_ms is None else latency_ms)
    signals = pd.DataFrame(
        {
            "timestamp": [origin + pd.Timedelta(milliseconds=decision_ms)],
            "asset_id": ["a"],
            "prediction_score": [1.0],
        }
    )
    batches, _ = native_account(
        tmp_path,
        [(daily_view(panel), panel, signals)],
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
    )
    trades = [t for b in batches for t in b.trades]
    assert [t.timestamp_ns for t in trades] == (
        [] if execution_ms is None else [origin.value + execution_ms * 1_000_000]
    )
    assert [t.quantity for t in trades] == ([] if gap else [200])
