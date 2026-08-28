from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.recorder import (
    RECORDER_STREAMS,
    RecorderConfig,
    validate_stream_payload,
)


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
