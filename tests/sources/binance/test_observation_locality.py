"""Readable observations retain facts independent of invalid business cells."""

import hashlib
import json
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import TradeEvent
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.markets.implementations.crypto.execution import (
    _maker_trade_fill,
)
from quant_research_platform.markets.implementations.crypto.hft import iter_label_frames
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.binance import recorder
from quant_research_platform.sources.binance.funding import read_funding_observations
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import iter_agg_trades
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.sources.binance.test_local_trade_validity import _recording
from tests.sources.tardis.test_funding_missingness import (
    _START_MS,
    _import,
    _mark,
    _simulate,
)


def test_recorder_keeps_unknown_optional_quantity(tmp_path, monkeypatch):
    stop = threading.Event()
    payload = dict(
        e="aggTrade",
        s="BTCUSDT",
        E=1,
        T=1,
        a=1,
        f=1,
        l=1,
        st=1,
        p="100",
        q="2",
        nq=None,
        m=True,
    )
    rows = []

    class Writer:
        def write(self, **row):
            rows.append(row)
            stop.set()
            return len(rows)

    class Socket:
        def recv(self, **kwargs):
            return json.dumps(payload)

        def close(self):
            pass

    monkeypatch.setattr(recorder, "_connect_websocket", lambda *a, **k: Socket())
    stats = recorder._CaptureStats()
    stream = next(
        s
        for s in recorder.RECORDER_STREAMS_BY_SCOPE["futures-hft"]
        if s.data_kind == "agg_trades"
    )
    reader = recorder._StreamReader(
        stream=stream,
        writer=Writer(),
        stats=stats,
        stop=stop,
        config=recorder.RecorderConfig(tmp_path, 1),
        tick_size="0.1",
    )
    reader._read_connection(reconnect=False)
    assert len(rows) == 1 and rows[0]["payload"] == payload
    assert stats.invalid_messages == 0


@pytest.mark.parametrize("price", ["bad", "100.01", "-1", None])
def test_bad_trade_price_preserves_known_flow(tmp_path, price):
    hashes = _recording(tmp_path / "raw", {"p": price})
    with open_market_source(tmp_path / "raw") as source:
        events = list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
    assert len(events) == 10 and all(isinstance(e, TradeEvent) for e in events)
    assert events[3].price_index is None
    assert events[3].quantity == events[2].quantity
    assert events[3].buyer_is_maker == events[2].buyer_is_maker
    panel = read_dataset(
        prepare(
            {
                "market_id": "crypto",
                "data_id": HFT_PANEL_DATA_ID,
                "inputs": {},
                "source": {"root": str(tmp_path / "raw")},
                "time_window": {
                    "start": "1970-01-01T00:00:00",
                    "end": "1970-01-01T00:00:01",
                },
                "output_root": str(tmp_path / "panel"),
            }
        )
    )
    btc = panel.loc[panel.asset_id.eq("btc_usdt")].reset_index(drop=True)
    assert btc.trade_count_100ms.iloc[4] == 1
    for column in (
        "trade_volume_100ms",
        "observed_signed_trade_volume_100ms",
        "observed_taker_buy_volume_100ms",
        "observed_taker_sell_volume_100ms",
    ):
        assert btc[column].iloc[4] == btc[column].iloc[3]
    assert btc.book_spread.notna().all()
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes
    } == hashes
    order = SimpleNamespace(
        observation_id=None,
        trade_observation_id=events[3].observation_id,
        side="buy" if events[3].buyer_is_maker else "sell",
    )
    with pytest.raises(PortfolioTargetBlocked, match="trade_price_unknown"):
        _maker_trade_fill(order, events[3])
    order.side = "sell" if events[3].buyer_is_maker else "buy"
    assert _maker_trade_fill(order, events[3]) == (0.0, "opposite_trade_direction")


@pytest.mark.parametrize(
    "change", [{"r": "0.002"}, {"r": "bad"}, {"T": _START_MS + 11000}, {"p": "101"}]
)
def test_conflicting_forecast_is_local_and_recovers(tmp_path, change):
    roots = _import(
        tmp_path / "case",
        (
            (1, _mark(1)),
            (2, _mark(2, T=_START_MS + 4000)),
            (2, _mark(2, **{"T": _START_MS + 4000, **change})),
            (3, _mark(3)),
        ),
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 3
        assert forecasts[0].cashflows == forecasts[2].cashflows
        if "T" in change:
            assert forecasts[1].known_until_ns == forecasts[1].available_at_ns
            assert forecasts[1].cashflows == ()
        else:
            assert forecasts[1].cashflows == (((_START_MS + 4000) * 1000000, None),)
        assert tuple(source.events) == ()
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        result, selected, trades = _simulate(source)
    assert selected == [1.0, 0.0, -1.0]
    assert result["blocked_decisions"] == 1
    assert result["account_path_status"] == "COMPLETE"
    assert len(trades) == 2


@pytest.mark.parametrize("bad", [-100.0, 0.0, float("inf"), float("nan"), True])
def test_labels_reject_nonpositive_mid_locally(bad):
    frame = pd.DataFrame(
        {
            "market_id": ["crypto"] * 4,
            "asset_id": ["btc_usdt"] * 4,
            "timestamp": pd.to_datetime([0, 100, 200, 300], unit="ms"),
            "segment_id": ["s"] * 4,
            "mid": [100.0, bad, 110.0, 121.0],
        }
    )
    labels = pd.concat(iter_label_frames([frame], horizons_ms=(100,)))
    assert labels.future_mid_return_100ms.iloc[:2].isna().all()
    assert labels.future_mid_return_100ms_available_at.iloc[:2].isna().all()
    assert labels.future_mid_return_100ms.iloc[2] == pytest.approx(0.1)


def test_snapshot_business_anomaly_is_written_before_reconstruction(
    tmp_path, monkeypatch
):
    payload = {
        "lastUpdateId": 1,
        "E": 1,
        "T": 1,
        "bids": [["bad", "2"]],
        "asks": [["101", "3"]],
    }
    rows = []

    class Writer:
        def write(self, **row):
            rows.append(row)
            return len(rows)

    class Book:
        price_unit = "0.1"

        def bootstrap(self, record):
            assert rows[0]["payload"] == payload
            raise ValueError("synthetic_reconstruction_boundary")

    monkeypatch.setattr(
        recorder,
        "_http_get_json",
        lambda *a, **k: (payload, "1970-01-01T00:00:00.001Z", 1),
    )
    stream = next(
        s
        for s in recorder.RECORDER_STREAMS_BY_SCOPE["futures-hft"]
        if s.data_kind == "depth"
    )
    with pytest.raises(ValueError, match="synthetic_reconstruction_boundary"):
        recorder._record_depth_snapshot(
            Writer(),
            stream,
            connection_id="c",
            segment_id="s",
            reconstructor=Book(),
            stop=threading.Event(),
            network_owner=lambda value: None,
        )
    assert len(rows) == 1 and rows[0]["payload"] == payload
