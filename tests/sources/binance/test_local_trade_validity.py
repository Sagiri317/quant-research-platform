"""Identifiable recorder numeric defects are causal trade-stream boundaries."""

import gzip
import hashlib
import json

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import TradeBoundary, TradeEvent
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import iter_agg_trades
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.binance_hft import _write_file, write_binance_hft_run


def _recording(root, bad, *, same_timestamp=False):
    manifest = write_binance_hft_run(root, count=10)
    for index, member in enumerate(manifest["files"]):
        if member["symbol"] != "BTCUSDT" or member["data_kind"] != "agg_trades":
            continue
        with gzip.open(root / member["path"], "rt") as stream:
            rows = [json.loads(line) for line in stream]
        rows[3]["payload"].update(bad)
        for field in [name for name, value in bad.items() if value == "missing"]:
            del rows[3]["payload"][field]
        if same_timestamp:
            rows[4]["payload"].update(E=401, T=400)
            rows[4].update(
                receive_time="1970-01-01T00:00:00.400000Z", observed_time_ns=400_000_000
            )
        manifest["files"][index] = _write_file(
            root,
            symbol=member["symbol"],
            data_kind=member["data_kind"],
            stream=member["stream"],
            rows=rows,
        )
    (root / "run.json").write_text(json.dumps(manifest))
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.gz")
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"q": "NaN"},
        {"q": "-1"},
        {"p": "bad"},
        {"p": "100.01"},
        {"nq": "2"},
        {"nq": "-1"},
        {"m": "missing"},
        {"m": None},
        {"m": "SELL"},
    ],
)
def test_recorder_numeric_trade_boundary_preserves_other_assets_and_book_panel(
    tmp_path, bad
):
    raw = tmp_path / "raw"
    hashes = _recording(raw, bad)
    with open_market_source(raw) as source:
        coverage = source.observation_coverage(
            market_type=FUTURES_MARKET_TYPE, data_kind="agg_trades", symbol="BTCUSDT"
        )
        events = list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
        invalid = [event for event in events if isinstance(event, TradeBoundary)]
        assert len(invalid) == 1 and invalid[0].observed_time_ns == 400_000_000
        assert invalid[0].reason == "invalid_trade_values"
        valid = [event for event in events if isinstance(event, TradeEvent)]
        assert len(valid) == 9
        assert coverage.get(valid[0].observation_id) == (0, 400_000_000)
        assert coverage.get(valid[-1].observation_id) == (400_000_001, 10_000_000_000)
        assert coverage.covering(300_000_000, 400_000_000) == ()
        assert coverage.covering(500_000_000, 600_000_000) == (
            valid[-1].observation_id,
        )
        other = list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="ETHUSDT"))
        assert len(other) == 10 and all(
            isinstance(event, TradeEvent) for event in other
        )
    panel = read_dataset(
        prepare(
            {
                "market_id": "crypto",
                "data_id": HFT_PANEL_DATA_ID,
                "inputs": {},
                "source": {"root": str(raw)},
                "time_window": {
                    "start": "1970-01-01T00:00:00",
                    "end": "1970-01-01T00:00:01",
                },
                "output_root": str(tmp_path / "panel"),
            }
        )
    )
    btc = panel.loc[panel.asset_id.eq("btc_usdt")].reset_index(drop=True)
    eth = panel.loc[panel.asset_id.eq("eth_usdt")].reset_index(drop=True)
    assert len(btc) == len(eth) == 11
    assert btc.segment_id.nunique() == 1
    assert pd.isna(btc.trade_count_100ms.iloc[4])
    assert btc.trade_count_100ms.iloc[5:].eq(1).all()
    assert btc.trade_volume_100ms.iloc[0] == 0.0
    assert eth.trade_count_100ms.notna().all()
    assert btc.book_spread.notna().all()
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes
    } == hashes


@pytest.mark.parametrize("stop", ["resume", "close", "throw"])
def test_prefetch_cannot_apply_future_bad_trade_and_consumed_close_keeps_boundary(
    tmp_path, stop
):
    raw = tmp_path / "raw"
    _recording(raw, {"q": "NaN"}, same_timestamp=True)
    with open_market_source(raw) as source:
        coverage = source.observation_coverage(
            market_type=FUTURES_MARKET_TYPE, data_kind="agg_trades", symbol="BTCUSDT"
        )
        stream = iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT")
        prefix = [next(stream) for _ in range(3)]
        identity = prefix[0].observation_id
        assert [event.observed_time_ns for event in prefix] == [
            100_000_000,
            200_000_000,
            300_000_000,
        ]
        boundary = next(stream)
        assert isinstance(boundary, TradeBoundary)
        assert coverage.get(identity) == (0, 10_000_000_000)
        if stop == "resume":
            same_time = next(stream)
            assert isinstance(same_time, TradeEvent)
            assert same_time.observed_time_ns == boundary.observed_time_ns
            remaining = list(stream)
            assert remaining[0].observed_time_ns == 600_000_000
            assert remaining[0].observation_id != identity
        elif stop == "close":
            stream.close()
        else:
            error = RuntimeError("synthetic_consumer_failure")
            with pytest.raises(RuntimeError) as caught:
                stream.throw(error)
            assert caught.value is error
        assert coverage.get(identity) == (0, 400_000_000)
        assert coverage.covering(300_000_000, 399_999_999) == (identity,)
        assert coverage.covering(400_000_000, 400_000_000) == ()


def test_recorder_unknown_trade_timestamp_is_still_a_source_error(tmp_path):
    raw = tmp_path / "raw"
    _recording(raw, {"q": "NaN", "T": None})
    with (
        open_market_source(raw) as source,
        pytest.raises(ValueError, match="trade_id_invalid"),
    ):
        list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
