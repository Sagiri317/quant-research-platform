"""Ordinary source reads consume selected observations without byte certificates."""

import gzip
import io
import json
from collections import Counter
from pathlib import Path

import pytest

from quant_research_platform.sources.binance import replay as binance
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.tardis import normalized, raw
from quant_research_platform.sources.tardis.replay import open_tardis_archive
from tests.helpers.tardis import _slices, _snapshot
from tests.sources.binance.test_replay import _agg_trade, _write_run
from tests.sources.test_preparation_source_io import CountedFile


@pytest.mark.parametrize("source", ["binance", "tardis", "normalized"])
@pytest.mark.parametrize("audit", ["absent", "stale"])
def test_ordinary_read_does_not_require_unused_byte_certificate(
    tmp_path, source, audit
):
    root = tmp_path / "archive"
    if source == "binance":
        manifest = _write_run(
            root,
            [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])],
        )
        descriptors = [*manifest["files"], manifest["capture"]["connection_intervals"]]
        manifest_path = root / "run.json"
    elif source == "tardis":
        manifest = raw.import_slices(
            root, date="2026-09-01", slices=_slices(tmp_path / "native", _snapshot())
        )
        descriptors = [
            *manifest["files"],
            manifest["gaps"],
            manifest["observation_intervals"],
        ]
        manifest_path = root / "manifest.json"
    else:
        content = (
            b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
            b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n"
        )
        manifest = normalized._publish_dataset(
            root,
            lambda: io.BytesIO(gzip.compress(content)),
            exchange="bybit",
            dataset_type="trades",
            symbol="BTCUSDT",
            date="2026-08-01",
        )
        descriptors = [manifest]
        manifest_path = root / "manifest.json"
    for descriptor in descriptors:
        for key, wrong in (("sha256", "0" * 64), ("compressed_bytes", 1)):
            if audit == "absent":
                descriptor.pop(key)
            else:
                descriptor[key] = wrong
    if source in {"tardis", "normalized"}:
        for descriptor in manifest["files"] if source == "tardis" else [manifest]:
            if audit == "absent":
                descriptor.pop("request_url")
            else:
                descriptor["request_url"] = "historical-request-description"
    manifest_path.write_text(json.dumps(manifest))
    if source == "binance":
        with binance.open_recorder_run(root) as opened:
            assert len(list(opened.iter_records())) == 1
    elif source == "tardis":
        with open_tardis_archive(root) as opened:
            assert (
                len(
                    list(
                        opened.iter_market_records(
                            symbol="BTCUSDT",
                            market_type=FUTURES_MARKET_TYPE,
                            data_kinds=("depth_snapshot", "depth"),
                        )
                    )
                )
                == 1
            )
    else:
        assert (
            len(
                list(
                    normalized.iter_records(
                        tmp_path, archive_paths=[root.name], time_window=None
                    )
                )
            )
            == 1
        )


def test_binance_selected_stream_ignores_missing_other_member_and_tree(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)]),
            (FUTURES_MARKET_TYPE, "metadata", "exchangeInfo", [_agg_trade(2)]),
        ],
    )
    unrelated = root / manifest["files"][1]["path"]
    unrelated.unlink()
    (root / "unlisted").mkdir()
    (root / "unlisted" / "malformed.jsonl.gz").write_bytes(b"not-source")

    def no_scan(*_args, **_kwargs):
        pytest.fail("ordinary source read scanned the whole directory")

    monkeypatch.setattr("os.scandir", no_scan)
    with binance.open_recorder_run(root) as opened:
        assert [
            row["capture_seq"] for row in opened.iter_records(data_kinds=["agg_trades"])
        ] == [1]
        with pytest.raises(ValueError, match="raw_file_open_failed"):
            list(opened.iter_records(data_kinds=["metadata"]))


def test_tardis_prefix_ignores_missing_later_member_and_unused_gaps(tmp_path):
    root = tmp_path / "archive"
    manifest = raw.import_slices(
        root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", _snapshot(), count=2),
    )
    (root / manifest["files"][1]["path"]).unlink()
    (root / manifest["gaps"]["path"]).unlink()
    with open_tardis_archive(root) as opened:
        scope = dict(
            symbol="BTCUSDT",
            market_type=FUTURES_MARKET_TYPE,
            data_kinds=("depth_snapshot", "depth"),
        )
        records = opened.iter_market_records(**scope)
        assert next(records)["payload"]["lastUpdateId"] == 100
        records.close()
        with pytest.raises(ValueError, match="tardis_raw_identity_changed"):
            list(opened.iter_market_records(**scope))


def test_binance_prefix_close_adds_no_reads_and_consumed_tail_still_fails(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    rows = [_agg_trade(index) for index in range(1, 2001)]
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", rows)]
    )
    member = root / manifest["files"][0]["path"]
    with gzip.open(member, "rb") as handle:
        content = handle.read()
    payload = gzip.compress(content, compresslevel=0, mtime=0)
    member.write_bytes(payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:])
    opened, consumed, handles = Counter(), Counter(), []
    original = binance.open_filesystem_file

    def counted(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if Path(path) == member:
            opened[member] += 1
            handles.append(handle)
            return CountedFile(handle, member, consumed)
        return handle

    monkeypatch.setattr(binance, "open_filesystem_file", counted)
    with binance.open_recorder_run(root) as source:
        records = source.iter_records()
        assert next(records)["capture_seq"] == 1
        before_close = consumed.copy()
        records.close()
        assert consumed == before_close
        assert 0 < consumed[member] < len(payload)
        assert opened[member] == 1
        assert all(handle.closed for handle in handles)
        with pytest.raises(gzip.BadGzipFile):
            list(source.iter_records())
