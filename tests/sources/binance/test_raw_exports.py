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
