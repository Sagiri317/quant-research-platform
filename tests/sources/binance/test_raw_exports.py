from __future__ import annotations

import json
import os
import weakref
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.sources.binance.raw_exports as raw_exports
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    OwnedValue,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance import ids
from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.sources.binance.validation import (
    strict_millisecond_timestamp,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.runtime_resources import capture_test_policy


def _iter_test_raw_envelopes(
    path: Path,
    *,
    resource_policy=None,  # noqa: ANN001
):
    raw_path = Path(path)
    expected_identity = raw_exports._capture_raw_file_identity(raw_path)

    def activate():  # noqa: ANN202
        policy = resource_policy or current_runtime_resource_policy()
        with runtime_resource_policy_scope(policy) as policy:
            with closing_iterator(
                raw_exports._iter_raw_envelope_batches(
                    raw_path,
                    batch_size=1,
                    expected_identity=expected_identity,
                )
            ) as batches:
                try:
                    bound = next(batches)
                except StopIteration:
                    return
                while True:
                    with policy.lease("tests.binance.raw", bound) as lease:
                        try:
                            owned = batches.send(lease)
                        except StopIteration:
                            return
                        if not isinstance(owned, OwnedValue):
                            raise TypeError("invalid_binance_owned_frame")
                        if owned.lease is not lease:
                            raise RuntimeError("runtime_memory_lease_transfer_mismatch")
                        lease.transfer("tests.binance.raw.consumer")
                        yield from owned.value
                        del owned
                        try:
                            bound = batches.send(None)
                        except StopIteration:
                            return

    return activate()


def test_binance_materializer_rejects_file_alias_before_publication(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    outside_file = tmp_path / "outside" / "2026-07-06.jsonl"
    _write_jsonl(outside_file, _envelope(data_kind="klines"))
    alias_file = raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl"
    alias_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias_file.symlink_to(outside_file)
    except OSError as error:
        pytest.skip(f"filesystem symlink unavailable: {error}")

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            raw_root=raw_root,
            data_ids=("market_bars_1m",),
        )
    assert (
        not StorageLayout(project_root=tmp_path)
        .interim_data_root(
            ids.SOURCE_ID,
            "market_bars_1m",
        )
        .exists()
    )

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        tuple(_iter_test_raw_envelopes(alias_file))


def test_binance_materializer_rejects_directory_alias_before_publication(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    outside_symbol = tmp_path / "outside-symbol" / "BTCUSDT"
    _write_jsonl(
        outside_symbol / "1m" / "2026-07-06.jsonl",
        _envelope(data_kind="klines"),
    )
    alias_symbol = raw_root / "spot" / "klines" / "ALIASUSDT"
    alias_symbol.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias_symbol.symlink_to(outside_symbol, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"filesystem directory symlink unavailable: {error}")

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            raw_root=raw_root,
            data_ids=("market_bars_1m",),
        )
    assert (
        not StorageLayout(project_root=tmp_path)
        .interim_data_root(
            ids.SOURCE_ID,
            "market_bars_1m",
        )
        .exists()
    )


def test_binance_raw_reader_rejects_replacement_before_open(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    _write_jsonl(replacement, _envelope(data_kind="klines"))

    reader = _iter_test_raw_envelopes(path)
    os.replace(replacement, path)

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        tuple(reader)


def test_binance_raw_reader_maps_capture_to_open_disappearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    original_open = Path.open

    def disappear_before_open(  # noqa: ANN001
        candidate,
        *args,
        **kwargs,  # noqa: ANN002, ANN003
    ):
        if Path(candidate) == path:
            path.unlink()
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappear_before_open)

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_reader_rejects_open_handle_mutation_during_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    _write_jsonl(replacement, _envelope(data_kind="klines"))
    original_next_record_length = raw_exports._next_record_length
    replaced = False

    def replace_before_read(handle, buffer):  # noqa: ANN001
        nonlocal replaced
        if not replaced:
            path.write_text(replacement.read_text(encoding="utf-8"), encoding="utf-8")
            replaced = True
        return original_next_record_length(handle, buffer)

    monkeypatch.setattr(raw_exports, "_next_record_length", replace_before_read)

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_reader_rejects_open_handle_mutation_after_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    _write_jsonl(replacement, _envelope(data_kind="klines"))
    reader = _iter_test_raw_envelopes(path)

    assert next(reader)["data_kind"] == "klines"
    path.write_text(replacement.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        next(reader)


def test_binance_raw_reader_rejects_open_handle_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    original_identity_from_stat = raw_exports.dataset_file_identity_from_stat

    def changed_handle_identity(stat_result):  # noqa: ANN001
        identity = original_identity_from_stat(stat_result)
        return raw_exports.DatasetFileIdentity(
            identity.device,
            identity.inode,
            identity.size + 1,
            identity.modified_ns,
        )

    monkeypatch.setattr(
        raw_exports,
        "dataset_file_identity_from_stat",
        changed_handle_identity,
    )

    with pytest.raises(ValueError, match="binance_raw_export_identity_changed"):
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_reader_preserves_primary_when_final_identity_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    _write_jsonl(replacement, _envelope(data_kind="klines"))
    primary = KeyboardInterrupt("primary")

    def fail_after_mutation(*_args, **_kwargs):  # noqa: ANN002, ANN003
        path.write_text(replacement.read_text(encoding="utf-8"), encoding="utf-8")
        raise primary

    monkeypatch.setattr(raw_exports, "_validate_envelope", fail_after_mutation)

    with pytest.raises(KeyboardInterrupt) as caught:
        tuple(_iter_test_raw_envelopes(path))

    assert caught.value is primary
    assert any("ValueError" in note for note in primary.__notes__)


def test_binance_raw_exports_reads_canonical_envelopes(tmp_path: Path) -> None:
    path = tmp_path / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))

    rows = tuple(_iter_test_raw_envelopes(path))

    assert len(rows) == 1
    assert rows[0]["source"] == "binance"
    assert rows[0]["market_type"] == "spot"
    assert rows[0]["data_kind"] == "klines"
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["interval"] == "1m"
    assert isinstance(rows[0]["payload"], list)


def test_binance_raw_export_rejects_oversized_record_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope["padding"] = "x" * 8_000
    _write_jsonl(path, envelope)
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 3072,
    )
    monkeypatch.setattr(
        raw_exports,
        "_loads_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("json_decode_started_before_admission")
        ),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            tuple(_iter_test_raw_envelopes(path, resource_policy=policy))


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
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_rows_reject_missing_empty_and_malformed_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    malformed = tmp_path / "malformed.jsonl"
    empty.write_text("", encoding="utf-8")
    malformed.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        tuple(_iter_test_raw_envelopes(missing))
    with pytest.raises(ValueError, match="empty_binance_raw_file"):
        tuple(_iter_test_raw_envelopes(empty))
    with pytest.raises(ValueError, match="malformed_binance_raw_row"):
        tuple(_iter_test_raw_envelopes(malformed))


@pytest.mark.parametrize(
    "raw",
    (
        b'{"source":"binance","source":"binance"}',
        b'{"payload":{"value":1,"value":2}}',
        b'{"payload":NaN}',
        b'{"payload":"\xff"}',
        b'{"payload":' + b"1" * 4097 + b"}",
        b'{"payload":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
    ),
)
def test_legacy_binance_json_uses_strict_qrp_boundary(raw: bytes) -> None:
    with pytest.raises(
        ValueError,
        match=r"malformed_binance_raw_row: legacy\.jsonl:7",
    ):
        raw_exports._loads_json(raw, Path("legacy.jsonl"), line_number=7)


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_exports_accepts_kline_payload_with_extra_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "extra-kline.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope["payload"] = [*envelope["payload"], "extra"]
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    rows = tuple(_iter_test_raw_envelopes(path))

    assert rows[0]["payload"][-1] == "extra"


def test_binance_raw_exports_does_not_require_interval_for_trades_or_depth(
    tmp_path: Path,
) -> None:
    trade_path = tmp_path / "trade.jsonl"
    depth_path = tmp_path / "depth.jsonl"
    _write_jsonl(trade_path, _envelope(data_kind="agg_trades"))
    _write_jsonl(depth_path, _envelope(data_kind="depth"))

    assert tuple(_iter_test_raw_envelopes(trade_path))[0]["data_kind"] == "agg_trades"
    assert tuple(_iter_test_raw_envelopes(depth_path))[0]["data_kind"] == "depth"


def test_binance_raw_exports_admits_valid_large_depth_record(tmp_path: Path) -> None:
    path = tmp_path / "depth.jsonl"
    envelope = _envelope(data_kind="depth")
    envelope["payload"] = {
        "lastUpdateId": 1,
        "bids": [[f"{1000 - index / 10:.1f}", "1.0"] for index in range(500)],
        "asks": [[f"{1000.1 + index / 10:.1f}", "1.0"] for index in range(500)],
    }
    _write_jsonl(path, envelope)
    policy = capture_test_policy(
        available_memory_bytes=512 * 1024**2,
        explicit_cap_bytes=256 * 1024**2,
    )

    rows = tuple(_iter_test_raw_envelopes(path, resource_policy=policy))

    assert len(rows) == 1
    assert len(rows[0]["payload"]["bids"]) == 500
    assert len(rows[0]["payload"]["asks"]) == 500


def test_large_legacy_file_with_small_records_starts_from_scan_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.jsonl"
    line = (
        json.dumps(_envelope(data_kind="klines"), separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    path.write_bytes(line * (9 * 1024**2 // len(line) + 1))
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )
    rows = _iter_test_raw_envelopes(path, resource_policy=policy)

    try:
        assert next(rows)["data_kind"] == "klines"
    finally:
        rows.close()

    assert policy.used_memory_bytes == 0


def test_invalid_legacy_envelope_releases_decode_inside_owner_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, _envelope(data_kind="klines"))
    released: list[int] = []
    loads_json = raw_exports._loads_json
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    class TrackedEnvelope(dict):
        pass

    def track_envelope(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        tracked = TrackedEnvelope(loads_json(*args, **kwargs))
        tracked["source"] = "invalid"
        weakref.finalize(tracked, lambda: released.append(policy.used_memory_bytes))
        return tracked

    monkeypatch.setattr(raw_exports, "_loads_json", track_envelope)
    with pytest.raises(ValueError, match="invalid_binance_raw_envelope_source"):
        tuple(_iter_test_raw_envelopes(path, resource_policy=policy))

    assert released
    assert all(used > 0 for used in released)


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
        tuple(_iter_test_raw_envelopes(kline_path))

    trade_path = tmp_path / "bad-trade.jsonl"
    trade = _envelope(data_kind="agg_trades")
    trade["payload"] = {**trade["payload"], "f": 12, "l": 11}
    _write_jsonl(trade_path, trade)
    with pytest.raises(ValueError, match="invalid_binance_agg_trade_id_range"):
        tuple(_iter_test_raw_envelopes(trade_path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_exports_rejects_unhashable_data_kind_with_stable_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-data-kind.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope["data_kind"] = ["klines"]
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="unsupported_binance_data_kind"):
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


@pytest.mark.parametrize("milliseconds", [0, 1783296000000])
def test_binance_millisecond_timestamp_accepts_ns_representable_values(
    milliseconds: int,
) -> None:
    timestamp = strict_millisecond_timestamp(milliseconds, error="stable_error")

    assert timestamp.tz is not None
    assert timestamp == pd.Timestamp(milliseconds, unit="ms", tz="UTC")
    assert timestamp.as_unit("ns").unit == "ns"


def test_binance_millisecond_timestamp_accepts_maximum_ns_millisecond() -> None:
    maximum = pd.Timestamp.max.value // pd.Timedelta(milliseconds=1).value

    timestamp = strict_millisecond_timestamp(maximum, error="stable_error")

    assert timestamp == pd.Timestamp(maximum, unit="ms", tz="UTC").as_unit("ns")


@pytest.mark.parametrize(
    "milliseconds",
    [
        pd.Timestamp.max.value // pd.Timedelta(milliseconds=1).value + 1,
        10**100,
    ],
)
def test_binance_millisecond_timestamp_rejects_values_above_ns_range(
    milliseconds: int,
) -> None:
    with pytest.raises(ValueError, match="stable_error"):
        strict_millisecond_timestamp(milliseconds, error="stable_error")


def test_binance_raw_kline_maps_ns_range_failure_to_owner_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out-of-range-kline.jsonl"
    envelope = _envelope(data_kind="klines")
    payload = list(envelope["payload"])
    payload[0] = pd.Timestamp.max.value // pd.Timedelta(milliseconds=1).value + 1
    envelope["payload"] = payload
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_kline_open_time"):
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_aggregate_trade_maps_ns_range_failure_to_owner_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out-of-range-trade.jsonl"
    envelope = _envelope(data_kind="agg_trades")
    envelope["payload"] = {
        **envelope["payload"],
        "T": pd.Timestamp.max.value // pd.Timedelta(milliseconds=1).value + 1,
    }
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_agg_trade_timestamp"):
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))


def test_binance_raw_exports_rejects_non_bool_aggregate_trade_side(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-bool.jsonl"
    envelope = _envelope(data_kind="agg_trades")
    envelope["payload"] = {**envelope["payload"], "m": "False"}
    _write_jsonl(path, envelope)

    with pytest.raises(ValueError, match="invalid_binance_agg_trade_side"):
        tuple(_iter_test_raw_envelopes(path))


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
        tuple(_iter_test_raw_envelopes(path))

    message = str(exc_info.value)
    assert message == "invalid_binance_kline_volume"
    assert "not-a-number" not in message


def test_binance_raw_exports_rejects_missing_required_envelope_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-key.jsonl"
    envelope = _envelope(data_kind="klines")
    envelope.pop("captured_at")
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="binance_raw_envelope_required"):
        tuple(_iter_test_raw_envelopes(path))


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
