"""Public consumers use known total volume without inventing visible liquidity."""

import hashlib
import json
from dataclasses import asdict

import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.order_book import TradeBoundary, TradeEvent
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import iter_agg_trades
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.binance_hft import (
    write_binance_hft_run,
    write_hft_funding_fixture,
)
from tests.markets.crypto.test_independent_execution_source import _mutate, _panel
from tests.sources.binance.test_local_trade_validity import _recording


@pytest.mark.parametrize("visible", ["missing", None, "unavailable", "NaN", "inf", "0"])
def test_public_panel_keeps_total_quantity_when_only_visible_quantity_is_unknown(
    tmp_path, visible
):
    raw = tmp_path / "raw"
    _recording(raw, {"nq": visible})
    hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw.rglob("*.gz")
    }
    with open_market_source(raw) as source:
        events = list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
        assert all(isinstance(event, TradeEvent) for event in events)
        assert events[3].quantity == 0.1
        assert events[3].visible_quantity == (0.0 if visible == "0" else None)
        assert len({event.observation_id for event in events}) == 1
    panel = read_dataset(_panel(tmp_path / "panel", raw))
    btc = panel.loc[panel.asset_id.eq("btc_usdt")].reset_index(drop=True)
    assert len(btc) == 11
    assert btc.trade_count_100ms.iloc[4] == 1
    assert btc.trade_volume_100ms.iloc[4] == 0.1
    assert btc.observed_signed_trade_volume_100ms.iloc[4] == 0.1
    assert btc.observed_taker_buy_volume_100ms.iloc[4] == 0.1
    assert btc.book_spread.eq(1.0).all()
    assert hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes
    }


@pytest.mark.parametrize(
    "bad",
    [{"q": "0"}, {"q": "NaN"}, {"p": "bad"}, {"p": "100.01"}, {"m": None}],
)
def test_missing_visible_quantity_preserves_only_independent_known_facts(tmp_path, bad):
    raw = tmp_path / "raw"
    _recording(raw, {"nq": "missing", **bad})
    with open_market_source(raw) as source:
        events = list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
    if "p" not in bad:
        assert isinstance(events[3], TradeBoundary)
    else:
        # Unknown price and visible liquidity do not erase known total base flow.
        assert all(isinstance(event, TradeEvent) for event in events)
        assert events[3].price_index is None
        assert events[3].visible_quantity is None
        assert events[3].quantity == 0.1
        assert events[3].buyer_is_maker is False
        assert len({event.observation_id for event in events}) == 1


@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize(
    "price,maker_buyer,depends,visible",
    [
        (price, maker_buyer, depends, visible)
        for price, maker_buyer, depends in (
            ("99.5", False, False),
            ("100.0", True, False),
            ("99.5", True, True),
            ("99.0", True, True),
        )
        # Excess visible mass is invalid even when the trade cannot touch the order.
        for visible in (
            ("missing", "0", "3", "4") if depends else ("missing", "3", "4")
        )
    ],
)
def test_public_maker_requires_visible_quantity_only_for_dependent_trade(
    tmp_path, visible, price, maker_buyer, depends, side
):
    if side == "sell":
        price = str(200 - float(price))
        maker_buyer = not maker_buyer
    raw = tmp_path / "source" / "raw"
    write_binance_hft_run(raw, count=3, maker_liquidity=True)

    def change(row, member):
        if member["symbol"] != "BTCUSDT" or member["data_kind"] != "agg_trades":
            return
        row["payload"].update(p=price, m=maker_buyer)
        if visible == "missing":
            row["payload"].pop("nq")
        else:
            row["payload"]["nq"] = visible

    manifest = _mutate(raw, change)
    (raw / "run.json").write_text(json.dumps(manifest))
    write_hft_funding_fixture(raw)
    window = {
        "start": "1970-01-01T00:00:00",
        "end": "1970-01-01T00:00:00.150",
    }
    panel = _panel(tmp_path / "panel", raw, window)
    result = portfolio.backtest_maker(
        {
            "inputs": str(panel),
            "panel": str(panel),
            "source": str(raw),
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "prediction_window": window,
            "execution_mode": "continuous",
            "capital": asdict(
                PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(
                portfolio.MakerConfig(
                    order_quantity=1.0,
                    max_position=2.0,
                    maker_fee_bps=0.0,
                    inventory_target=2.0 if side == "buy" else -2.0,
                    inventory_skew=1.0,
                    markout_horizon_ms=100,
                    latency_ms=0,
                )
            ),
            "models": [],
            "calibration": None,
            "output": str(tmp_path / "ledger"),
        }
    )
    ledger = read_portfolio_ledger_artifact(result)
    summary = summarize_crypto_ledger(ledger, ledger.iter_chunks())
    partial = visible == "4" or visible == "missing" and depends
    execution = summary["execution"]
    assert execution["account_path_status"] == ("PARTIAL" if partial else "COMPLETE")
    assert execution["created_orders"] >= 1
    filled = visible == "3" and depends
    assert execution["filled_orders"] == int(filled)
    assert summary["economics"]["net_pnl"] == (
        None
        if partial
        else pytest.approx((0.6 if side == "buy" else 0.4) if filled else 0.0)
    )
    if partial:
        assert (
            execution["account_stop_ns"]
            == pd.Timestamp("1970-01-01T00:00:00.100").value
        )
        assert execution["account_stop_reason"] == (
            "portfolio_pending_execution_unknown"
            if visible == "4"
            else "hft_trade_visible_quantity_unknown"
        )
    else:
        assert execution["account_stop_reason"] is None
