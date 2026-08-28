from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import quant_research_platform.sources.binance.recorder as recorder
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.recorder import (
    RECORDER_STREAMS,
    RecorderConfig,
    validate_stream_payload,
)
from tests.helpers.runtime_resources import capture_test_policy


def _depth_payload(market_type: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "e": "depthUpdate",
        "E": 1_700_000_000_001,
        "s": "BTCUSDT",
        "U": 10,
        "u": 12,
        "b": [["100.0", "1.0"]],
        "a": [["100.1", "2.0"]],
    }
    if market_type == FUTURES_MARKET_TYPE:
        payload.update(
            {
                "T": 1_700_000_000_000,
                "pu": 9,
                "ps": "BTCUSDT",
                "st": 1,
            }
        )
    return payload


def test_recorder_inventory_contains_only_consumed_public_streams() -> None:
    streams = RECORDER_STREAMS

    assert [(item.market_type, item.data_kind) for item in streams] == [
        (SPOT_MARKET_TYPE, "depth"),
        (FUTURES_MARKET_TYPE, "depth"),
        (FUTURES_MARKET_TYPE, "agg_trades"),
    ]
    assert streams[0].url == ("wss://stream.binance.com:9443/ws/btcusdt@depth@100ms")
    assert streams[1].url == ("wss://fstream.binance.com/public/ws/btcusdt@depth@100ms")
    assert streams[2].url == ("wss://fstream.binance.com/market/ws/btcusdt@aggTrade")


def test_recorder_validates_spot_and_futures_depth_semantics() -> None:
    streams = RECORDER_STREAMS
    spot, futures = streams[:2]

    assert validate_stream_payload(spot, _depth_payload(SPOT_MARKET_TYPE))["u"] == 12
    assert (
        validate_stream_payload(futures, _depth_payload(FUTURES_MARKET_TYPE))["pu"] == 9
    )

    with pytest.raises(ValueError, match="market_payload_identity_invalid"):
        validate_stream_payload(
            futures,
            {**_depth_payload(FUTURES_MARKET_TYPE), "st": 2},
        )


def test_recorder_validates_futures_aggregate_trade_semantics() -> None:
    aggregate_trade = RECORDER_STREAMS[2]
    payload = {
        "e": "aggTrade",
        "E": 1_700_000_000_001,
        "T": 1_700_000_000_000,
        "s": "BTCUSDT",
        "st": 1,
        "a": 10,
        "f": 20,
        "l": 22,
        "p": "100.1",
        "q": "2.5",
        "nq": "2.0",
        "m": False,
    }

    assert validate_stream_payload(aggregate_trade, payload) is payload

    with pytest.raises(ValueError, match="trade_quantity_invalid"):
        validate_stream_payload(aggregate_trade, {**payload, "q": "0"})

    with pytest.raises(ValueError, match="visible_quantity_invalid"):
        validate_stream_payload(aggregate_trade, {**payload, "nq": "3.0"})

    missing_visible = dict(payload)
    missing_visible.pop("nq")
    with pytest.raises(ValueError, match="visible_quantity_invalid"):
        validate_stream_payload(aggregate_trade, missing_visible)


def test_recorder_config_rejects_existing_or_unbounded_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duration_invalid"):
        RecorderConfig(output_root=tmp_path / "run", duration_seconds=0).validate()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output_not_empty"):
        RecorderConfig(output_root=occupied, duration_seconds=1).validate()


def test_depth_reader_waits_for_same_generation_snapshot_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = RECORDER_STREAMS[1]
    payload = _depth_payload(FUTURES_MARKET_TYPE)
    payload.update({"U": 1, "u": 1, "pu": 0})
    writes: list[dict[str, object]] = []

    class Writer:
        def write(self, **kwargs: object) -> int:
            writes.append(kwargs)
            return len(writes)

    class Websocket:
        def __init__(self) -> None:
            self.sent = False
            self.release = threading.Event()

        def recv(self, timeout: float) -> str:
            del timeout
            if not self.sent:
                self.sent = True
                return json.dumps(payload)
            self.release.wait(1)
            raise TimeoutError

        def close(self) -> None:
            self.release.set()

    websocket = Websocket()
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_args, **_kwargs: websocket,
    )
    reader = recorder._StreamReader(
        stream=stream,
        writer=Writer(),
        stats=recorder._CaptureStats(),
        stop=threading.Event(),
        config=RecorderConfig(Path("unused"), 1),
    )
    reader.start()
    assert reader.ready.wait(1)
    try:
        assert writes == []
        generation, _connection_id, _segment_id = reader.current_identity()
        assert reader.acknowledge_snapshot(generation)
        assert reader.stop.wait(0.05) is False
    finally:
        reader.stop.set()
        websocket.release.set()
        reader.join(1)
    assert len(writes) == 1


def test_connection_interval_freezes_before_websocket_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class Websocket:
        def recv(self, timeout: float) -> None:
            del timeout
            raise OSError("connection lost")

        def close(self) -> None:
            order.append("websocket")

    stats = recorder._CaptureStats()
    real_closed = stats.closed

    def record_closed(*args: object, **kwargs: object) -> None:
        order.append("interval")
        real_closed(*args, **kwargs)

    monkeypatch.setattr(stats, "closed", record_closed)
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_args, **_kwargs: Websocket(),
    )
    reader = recorder._StreamReader(
        stream=RECORDER_STREAMS[2],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
    )
    with pytest.raises(recorder._ConnectionProblem):
        reader._read_connection(reconnect=False)

    assert order == ["interval", "websocket"]


def test_connection_interval_retention_is_admitted_and_released() -> None:
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        stats = recorder._CaptureStats(policy)
        stats.opened(
            RECORDER_STREAMS[2],
            reconnect=False,
            connection_id="trade-1",
            segment_id=None,
            opened_at="1970-01-01T00:00:00.000000000Z",
            opened_monotonic_ns=0,
        )
        assert policy.used_memory_bytes >= 1024
        stats.closed(
            "trade-1",
            closed_at="1970-01-01T00:00:01.000000000Z",
            closed_monotonic_ns=1_000_000_000,
            close_reason="stop",
        )
        stats.close()
        assert policy.used_memory_bytes == 0
