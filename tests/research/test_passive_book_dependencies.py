"""Maker fills depend on their own observed quote and causal order state."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
    build_panel,
)
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.research.protocols.a_share_hft import processed_specs
from quant_research_platform.sources.venvoo import interim_spec, normalize
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.helpers.high_frequency import daily_view, parameters, schedule
from tests.markets.a_share.test_high_frequency import (
    INSTRUMENTS,
    raw_events,
    raw_quotes,
)


@pytest.mark.parametrize("book_dtype", ["float64", "Float64"])
@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize(
    "case,expected",
    [
        ("complete", "full"),
        ("opposite_missing_submission", "full"),
        ("opposite_missing_later", "full"),
        ("opposite_missing_all", "full"),
        ("own_missing_all", "none"),
        ("own_volume_missing_all", "none"),
        ("own_volume_missing_later", "full"),
        ("locked_submission", "none"),
        ("crossed_submission", "none"),
        ("locked_later", "partial"),
        ("crossed_later", "partial"),
    ],
)
def test_native_maker_side_dependencies_and_order_clock(
    tmp_path, side, case, expected, book_dtype
):
    raw = raw_quotes().assign(date=20240102, price=100000.0)
    for level in range(1, 11):
        raw[f"bid_px{level}"] = 100000 - level * 100
        raw[f"ask_px{level}"] = 100000 + level * 100
        raw[f"bid_vol{level}"] = raw[f"ask_vol{level}"] = 100.0
    events = pd.concat([raw_events("hft_trades").iloc[:1]] * 2, ignore_index=True)
    events = normalize(
        events.assign(
            date=20240102,
            time=[93004000, 93007000],
            price=100000 - side * 100,
            volume=[150.0, 400.0],
            trade_code="T",
            bs_flag="S" if side == 1 else "B",
        ),
        "hft_trades",
        INSTRUMENTS,
    )
    orders = normalize(raw_events("hft_orders").iloc[:0], "hft_orders", INSTRUMENTS)
    complete = build_panel(
        normalize(raw, "hft_quotes", INSTRUMENTS), orders, events, max_gap_seconds=6
    )
    view = daily_view(complete)
    rows = [1] if case.endswith("submission") else [2]
    if "missing" in case:
        rows = raw.index if case.endswith("all") else rows
        own = "bid" if side == 1 else "ask"
        missing = own if case.startswith("own") else "ask" if side == 1 else "bid"
        field = "vol" if "volume" in case else "px"
        raw.loc[rows, [f"{missing}_{field}{level}" for level in range(1, 11)]] = (
            float("nan") if field == "vol" else 0.0
        )
    elif case.startswith(("locked", "crossed")):
        raw.loc[rows, "ask_px1"] = raw.loc[rows, "bid_px1"] - (
            100 if case.startswith("crossed") else 0
        )
    write_dataset(
        tmp_path / "quotes",
        normalize(raw, "hft_quotes", INSTRUMENTS),
        spec=interim_spec("hft_quotes"),
    )
    write_dataset(tmp_path / "trades", events, spec=interim_spec("hft_trades"))
    panel = build_panel(
        read_dataset(tmp_path / "quotes"), orders, events, max_gap_seconds=6
    )
    book_columns = panel.filter(regex=r"^(bid|ask)_(px|vol)").columns
    panel[book_columns] = panel[book_columns].astype(book_dtype)
    write_dataset(
        tmp_path / "panel",
        panel,
        spec=next(item for item in processed_specs() if item.data_id == "hft_panel"),
    )
    panel = read_dataset(tmp_path / "panel")
    asset = INSTRUMENTS[0]["asset_id"]
    signals = panel.iloc[:1][["timestamp", "asset_id"]].assign(prediction_score=side)
    params = parameters((asset,))
    params.update(
        initial_inventory={asset: 100 if side == 1 else 300},
        minimum_commission=5.0,
        passive={"queue_model": "conservative", "cancel_latency_ms": 0},
    )
    window = TimeWindow(panel.timestamp.min(), panel.timestamp.max())

    def produce(emit):
        for batch in inventory_batches(
            [(view, panel, signals)],
            schedule(),
            params,
            window,
            trade_session=open_dataset_session(tmp_path / "trades"),
            max_quote_gap_seconds=6,
        ):
            emit(batch)

    output = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("hft_inventory", 5000.0, "CNY", "cash_equity"),
        metadata={
            "kind": "a_share_hft",
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
        },
        produce=produce,
    )
    batches = list(read_portfolio_ledger_artifact(output).iter_chunks())
    trades = [trade for batch in batches for trade in batch.trades]
    quantities = (
        []
        if expected == "none"
        else [side * 50]
        if expected == "partial"
        else [50, 150]
        if side == 1
        else [-50, -250]
    )
    assert [trade.quantity for trade in trades] == quantities
    assert [pd.Timestamp(trade.timestamp_ns).second for trade in trades] == [4, 7][
        : len(quantities)
    ]
    assert [trade.notional for trade in trades] == pytest.approx(
        [quantity * (10.0 - side * 0.01) for quantity in quantities]
    )
    assert len({trade.order_id for trade in trades}) == bool(quantities)
    fees = -sum(
        flow.amount
        for batch in batches
        for flow in batch.cashflows
        if flow.kind == "commission_cost"
    )
    assert fees == (5.0 if quantities else 0.0)
    final = {row.asset_id: row for row in batches[-1].valuations if row.phase == "end"}
    assert final[asset].mark_price == 10.0
    assert final[asset].quantity == params["initial_inventory"][asset] + sum(quantities)
    assert final["__cash__"].quantity == pytest.approx(
        5000.0
        - params["initial_inventory"][asset] * 10.0
        - sum(trade.notional for trade in trades)
        - fees
    )
    if case.endswith("all") and "volume" not in case:
        assert panel.mid.isna().all()
    assert panel.segment_id.nunique() == 1
