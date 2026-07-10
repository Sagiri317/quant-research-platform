from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.sources.binance.raw_exports import (
    dataset_files_by_id,
    iter_raw_envelopes,
)


def test_binance_dataset_files_by_id_groups_local_raw_jsonl(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    kline_path = raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl"
    trade_path = raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl"
    depth_path = raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl"
    for path, data_kind in (
        (kline_path, "klines"),
        (trade_path, "agg_trades"),
        (depth_path, "depth"),
    ):
        _write_jsonl(path, _envelope(data_kind=data_kind))

    files = dataset_files_by_id(raw_root)

    assert files == {
        "market_bars_1m": (kline_path,),
        "aggregate_trades_tick": (trade_path,),
        "depth_levels_tick": (depth_path,),
    }


def test_binance_raw_exports_reads_canonical_envelopes(tmp_path: Path) -> None:
    path = tmp_path / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))

    rows = tuple(iter_raw_envelopes(path))

    assert len(rows) == 1
    assert rows[0]["source"] == "binance"
    assert rows[0]["market_type"] == "spot"
    assert rows[0]["data_kind"] == "klines"
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["interval"] == "1m"
    assert isinstance(rows[0]["payload"], list)


@pytest.mark.parametrize(
    "bare_payload",
    (
        [1783296000000, "100.0", "101.0"],
        {"a": 1, "p": "100.0", "q": "1.0"},
        {"lastUpdateId": 1, "bids": [], "asks": []},
    ),
)
def test_binance_raw_exports_rejects_bare_payloads(
    tmp_path: Path,
    bare_payload: object,
) -> None:
    path = tmp_path / "bare.jsonl"
    path.write_text(json.dumps(bare_payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="binance_raw_envelope_required"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_rows_reject_missing_empty_and_malformed_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    malformed = tmp_path / "malformed.jsonl"
    empty.write_text("", encoding="utf-8")
    malformed.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        tuple(iter_raw_envelopes(missing))
    with pytest.raises(ValueError, match="empty_binance_raw_file"):
        tuple(iter_raw_envelopes(empty))
    with pytest.raises(ValueError, match="malformed_binance_raw_row"):
        tuple(iter_raw_envelopes(malformed))


@pytest.mark.parametrize(
    ("patch", "error"),
    (
        ({"source": "other"}, "invalid_binance_raw_envelope_source"),
        ({"market_type": "futures"}, "unsupported_binance_market_type"),
        ({"data_kind": "funding"}, "unsupported_binance_data_kind"),
        ({"interval": "5m"}, "unsupported_binance_kline_interval"),
    ),
)
def test_binance_raw_exports_validates_envelope_contract(
    tmp_path: Path,
    patch: dict[str, object],
    error: str,
) -> None:
    path = tmp_path / "bad.jsonl"
    envelope = {**_envelope(data_kind="klines"), **patch}
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize(
    ("data_kind", "patch", "error"),
    (
        (
            "klines",
            {"captured_at": "not-a-time"},
            "invalid_binance_raw_envelope_captured_at",
        ),
        (
            "klines",
            {"captured_at": 1783296000000},
            "invalid_binance_raw_envelope_captured_at",
        ),
        (
            "klines",
            {"captured_at": "2026-07-06T00:00:02"},
            "invalid_binance_raw_envelope_captured_at",
        ),
        ("klines", {"symbol": ""}, "invalid_binance_raw_envelope_symbol"),
        ("klines", {"payload": [1783296000000]}, "invalid_binance_kline_payload"),
        (
            "klines",
            {"payload": {"open_time": 1783296000000}},
            "invalid_binance_kline_payload",
        ),
        ("agg_trades", {"payload": []}, "invalid_binance_agg_trade_payload"),
        ("depth", {"payload": []}, "invalid_binance_depth_payload"),
    ),
)
def test_binance_raw_exports_rejects_invalid_envelope_fields(
    tmp_path: Path,
    data_kind: str,
    patch: dict[str, object],
    error: str,
) -> None:
    path = tmp_path / "bad-fields.jsonl"
    envelope = {**_envelope(data_kind=data_kind), **patch}
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_accepts_kline_payload_with_extra_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "extra-kline.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope["payload"] = [*envelope["payload"], "extra"]
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    rows = tuple(iter_raw_envelopes(path))

    assert rows[0]["payload"][-1] == "extra"


def test_binance_raw_exports_does_not_require_interval_for_trades_or_depth(
    tmp_path: Path,
) -> None:
    trade_path = tmp_path / "trade.jsonl"
    depth_path = tmp_path / "depth.jsonl"
    _write_jsonl(trade_path, _envelope(data_kind="agg_trades"))
    _write_jsonl(depth_path, _envelope(data_kind="depth"))

    assert tuple(iter_raw_envelopes(trade_path))[0]["data_kind"] == "agg_trades"
    assert tuple(iter_raw_envelopes(depth_path))[0]["data_kind"] == "depth"


def test_binance_raw_exports_rejects_invalid_market_payload_values(
    tmp_path: Path,
) -> None:
    kline_path = tmp_path / "bad-kline.jsonl"
    kline = _envelope(data_kind="klines")
    payload = list(kline["payload"])
    payload[2] = "90.0"
    kline["payload"] = payload
    _write_jsonl(kline_path, kline)
    with pytest.raises(ValueError, match="invalid_binance_kline_ohlc"):
        tuple(iter_raw_envelopes(kline_path))

    trade_path = tmp_path / "bad-trade.jsonl"
    trade = _envelope(data_kind="agg_trades")
    trade["payload"] = {**trade["payload"], "f": 12, "l": 11}
    _write_jsonl(trade_path, trade)
    with pytest.raises(ValueError, match="invalid_binance_agg_trade_id_range"):
        tuple(iter_raw_envelopes(trade_path))

    depth_path = tmp_path / "bad-depth.jsonl"
    depth = _envelope(data_kind="depth")
    depth["payload"] = {
        "lastUpdateId": 1,
        "bids": [["101.0", "1.0"]],
        "asks": [["100.0", "1.0"]],
    }
    _write_jsonl(depth_path, depth)
    with pytest.raises(ValueError, match="crossed_binance_depth_book"):
        tuple(iter_raw_envelopes(depth_path))


def test_binance_raw_exports_rejects_taker_buy_volume_above_volume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-taker-volume.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[9] = "11.0"
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_volume"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_rejects_taker_buy_quote_volume_above_quote_volume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-taker-quote-volume.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[10] = "1006.0"
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_volume"):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize("bad_trade_count", [True, 7.5, "7.0"])
def test_binance_raw_exports_rejects_bool_or_fractional_integer_payload_values(
    tmp_path: Path,
    bad_trade_count: object,
) -> None:
    path = tmp_path / "bad-integer.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[8] = bad_trade_count
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_trade_count"):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize("bad_value", ["NaN", "Infinity", "-Infinity"])
def test_binance_raw_exports_rejects_nan_or_infinite_numeric_payload_values(
    tmp_path: Path,
    bad_value: str,
) -> None:
    path = tmp_path / "bad-finite.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[5] = bad_value
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_volume"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_rejects_unhashable_data_kind_with_stable_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-data-kind.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope["data_kind"] = ["klines"]
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="unsupported_binance_data_kind"):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize("bad_symbol", ["btcusdt", "BTC-USDT"])
def test_binance_raw_exports_rejects_non_canonical_symbols(
    tmp_path: Path,
    bad_symbol: str,
) -> None:
    path = tmp_path / "bad-symbol.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope["symbol"] = bad_symbol
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_raw_envelope_symbol"):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize("bad_time", [True, 1783296000000.5, "1783296000000.0"])
def test_binance_raw_exports_rejects_invalid_millisecond_timestamps(
    tmp_path: Path,
    bad_time: object,
) -> None:
    path = tmp_path / "bad-time.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[0] = bad_time
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_open_time"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_rejects_kline_close_time_before_open_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-time-order.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[6] = payload[0] - 1
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_time_order"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_rejects_non_minute_aligned_kline_open_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "misaligned-open-time.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[0] += 1
    payload[6] = payload[0] + 59_999
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_open_time_alignment"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_rejects_wrong_one_minute_kline_close_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-close-time.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[6] = payload[0] + 60_000
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_close_time"):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize("bad_timestamp", [None, True, 1783296001000.5, "bad"])
def test_binance_raw_exports_rejects_invalid_aggregate_trade_timestamp(
    tmp_path: Path,
    bad_timestamp: object,
) -> None:
    path = tmp_path / "bad-aggregate-trade-time.jsonl"
    envelope = _envelope(data_kind="agg_trades")
    envelope["payload"] = {**envelope["payload"], "T": bad_timestamp}
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_agg_trade_timestamp"):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_exports_rejects_non_bool_aggregate_trade_side(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-bool.jsonl"
    envelope = _envelope(data_kind="agg_trades")
    envelope["payload"] = {**envelope["payload"], "m": "False"}
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_agg_trade_side"):
        tuple(iter_raw_envelopes(path))


@pytest.mark.parametrize(
    ("levels", "error"),
    [
        ([["100.0", "1.0", "extra"]], "invalid_binance_depth_bid"),
        ([["99.9", "1.0"], ["100.0", "1.0"]], "invalid_binance_depth_bid"),
        ([["100.0", "1.0"], ["100.0", "2.0"]], "invalid_binance_depth_bid"),
    ],
)
def test_binance_raw_exports_rejects_invalid_depth_levels(
    tmp_path: Path,
    levels: list[list[str]],
    error: str,
) -> None:
    path = tmp_path / "bad-depth-levels.jsonl"
    envelope = _envelope(data_kind="depth")
    envelope["payload"] = {"lastUpdateId": 1, "bids": levels, "asks": []}
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match=error):
        tuple(iter_raw_envelopes(path))


def test_binance_raw_numeric_conversion_raises_contract_error_without_raw_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-nonnumeric.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[5] = "not-a-number"
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError) as exc_info:
        tuple(iter_raw_envelopes(path))

    message = str(exc_info.value)
    assert message == "invalid_binance_kline_volume"
    assert "not-a-number" not in message


def test_binance_raw_exports_accepts_one_sided_depth_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one-sided-depth.jsonl"
    envelope = _envelope(data_kind="depth")
    envelope["payload"] = {"lastUpdateId": 1, "bids": [["100.0", "1.0"]], "asks": []}
    _write_jsonl(path, envelope)

    assert tuple(iter_raw_envelopes(path))[0]["data_kind"] == "depth"


def test_binance_raw_exports_rejects_missing_required_envelope_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-key.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope.pop("captured_at")
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="binance_raw_envelope_required"):
        tuple(iter_raw_envelopes(path))


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _envelope(*, data_kind: str) -> dict[str, object]:
    payload: object
    if data_kind == "klines":
        payload = [
            1783296000000,
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "10.0",
            1783296059999,
            "1005.0",
            7,
            "4.0",
            "402.0",
            "0",
        ]
    elif data_kind == "agg_trades":
        payload = {
            "a": 1,
            "p": "100.1",
            "q": "2.0",
            "f": 10,
            "l": 11,
            "T": 1783296001000,
            "m": False,
            "M": True,
        }
    else:
        payload = {
            "lastUpdateId": 1,
            "bids": [["100.0", "1.5"], ["99.9", "2.0"]],
            "asks": [["100.2", "1.0"]],
        }
    envelope: dict[str, object] = {
        "source": "binance",
        "market_type": "spot",
        "data_kind": data_kind,
        "symbol": "BTCUSDT",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": payload,
    }
    if data_kind == "klines":
        envelope["interval"] = "1m"
    return envelope
