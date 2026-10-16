"""Recorder selection and persisted reader scope use the same asset domain."""

import json
import threading

import pytest

from quant_research_platform.sources.binance import recorder, replay
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.sources.binance.test_replay import _agg_trade, _write_run


@pytest.mark.parametrize(
    "symbols", [("BTCUSDT",), ("ETHUSDT",), ("ETHUSDT", "BTCUSDT")]
)
@pytest.mark.parametrize("scope", ["futures-hft", "spot-l2"])
def test_record_only_selected_streams_and_metadata(
    tmp_path, monkeypatch, symbols, scope
):
    selected = []
    metadata_calls = []

    class Reader:
        failure = None

        def __init__(self, **kwargs):
            selected.append(kwargs["stream"])
            self.ready = threading.Event()
            self.finished = threading.Event()
            self.alive = True

        def start(self):
            self.ready.set()

        def cancel(self):
            self.finished.set()

        def join(self, timeout):
            self.alive = False

        def is_alive(self):
            return self.alive

    def metadata(writer, *, market_types, symbols):
        metadata_calls.append(symbols)
        return {
            symbol: {market: {"tick_size": "0.1"} for market in market_types}
            for symbol in symbols
        }

    monkeypatch.setattr(recorder, "_StreamReader", Reader)
    monkeypatch.setattr(recorder, "_record_market_metadata", metadata)
    result = recorder.record(
        recorder.RecorderConfig(tmp_path / "run", 0.001, scope=scope, symbols=symbols)
    )
    assert metadata_calls == [symbols]
    assert selected == [
        stream
        for stream in recorder.RECORDER_STREAMS_BY_SCOPE[scope]
        if stream.symbol in symbols
    ]
    assert result["symbols"] == list(symbols)
    assert set(result["market_metadata"]) == set(symbols)
    assert result["capture"]["threads"] == len(selected)
    assert result["capture"]["missing_required_streams"] == [
        f"{stream.market_type}:{stream.stream}" for stream in selected
    ]


@pytest.mark.parametrize(
    "symbols", [(), ("BTCUSDT", "BTCUSDT"), ("SOLUSDT",), ([],), ["BTCUSDT"]]
)
def test_invalid_recording_selection_is_rejected(tmp_path, symbols):
    with pytest.raises(ValueError, match="symbols_invalid"):
        recorder.RecorderConfig(tmp_path / "run", 1, symbols=symbols).validate()


def test_persisted_single_asset_run_is_readable_without_other_metadata(tmp_path):
    root = tmp_path / "run"
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    manifest["symbols"] = ["BTCUSDT"]
    del manifest["market_metadata"]["ETHUSDT"]
    (root / "run.json").write_text(json.dumps(manifest))
    with replay.open_recorder_run(root) as run:
        assert run.hft_source_info()["source_file_count"] == 1
        assert len(tuple(run.iter_records(market_type=FUTURES_MARKET_TYPE))) == 1
    manifest["symbols"] = ["ETHUSDT"]
    manifest["market_metadata"] = {"ETHUSDT": manifest["market_metadata"]["BTCUSDT"]}
    with pytest.raises(ValueError, match="file_manifest"):
        replay._validate_run_manifest(manifest)
