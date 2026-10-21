"""Public preparation closes native producers before publishing datasets."""

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.tardis import raw
from quant_research_platform.storage.datasets import read_dataset, writing
from tests.helpers.tardis import _slices


@pytest.mark.parametrize("mode", ["success", "close", "generator_exit"])
def test_prepare_closes_native_producer_before_publish(tmp_path, monkeypatch, mode):
    # Retain a native prefix beyond the output window, so the archive still owns it.
    lines = []
    for index in range(31):
        time = pd.Timestamp("2026-09-01T00:00:00Z") + pd.Timedelta(
            milliseconds=index * 100
        )
        payload = (
            {
                "lastUpdateId": 100,
                "bids": [[str(10000 - level), "1"] for level in range(20)],
                "asks": [[str(10002 + level), "1"] for level in range(20)],
            }
            if index == 0
            else {
                "e": "depthUpdate",
                "E": time.value // 1_000_000,
                "T": time.value // 1_000_000,
                "s": "BTCUSDT",
                "st": 1,
                "ps": "BTCUSDT",
                "U": 99 + index,
                "u": 100 + index,
                "pu": 99 + index,
                "b": [["10000", "1"]],
                "a": [["10002", "1"]],
            }
        )
        record = {
            "stream": "btcusdt@depthSnapshot" if index == 0 else "btcusdt@depth@0ms",
            "data": payload,
        }
        if index == 0:
            record["generated"] = True
        lines.append(
            (
                time.strftime("%Y-%m-%dT%H:%M:%S.%fZ") + " " + json.dumps(record) + "\n"
            ).encode()
        )
    archive = tmp_path / "archive"
    raw.import_slices(
        archive, date="2026-09-01", slices=_slices(tmp_path / "native", b"".join(lines))
    )
    output = tmp_path / "output"
    config = {
        "market_id": "crypto",
        "data_id": HFT_PANEL_DATA_ID,
        "inputs": {},
        "source": {"root": str(archive)},
        "parameters": {"asset_ids": ["btc_usdt"]},
        "time_window": {
            "start": "2026-09-01T00:00:00.100",
            "end": "2026-09-01T00:00:01.000",
        },
        "output_root": str(output),
    }
    close_error = OSError("native_close_failure")
    primary = GeneratorExit("writer_failure") if mode == "generator_exit" else None
    opened, closed = [], []
    open_file = raw.open_filesystem_file

    class SourceHandle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def close(self):
            self.handle.close()
            closed.append((self.handle.closed, output.exists()))
            if mode != "success":
                raise close_error

    def open_source(path, *args, **kwargs):
        handle = open_file(path, *args, **kwargs)
        if Path(path).suffix == ".gz":
            opened.append(path)
            return SourceHandle(handle)
        return handle

    def fail_write(*args, **kwargs):
        raise primary

    monkeypatch.setattr(raw, "open_filesystem_file", open_source)
    if primary is not None:
        monkeypatch.setattr(writing, "write_parquet_file", fail_write)
    if mode == "success":
        assert prepare(config) == output
        result = read_dataset(output)
        # Reconstruction starts at 100 ms; full 100 ms intervals start at 200.
        expected_times = pd.date_range(
            "2026-09-01 00:00:00.200", periods=9, freq="100ms"
        )
        assert result.timestamp.tolist() == expected_times.tolist()
        assert result.market_id.tolist() == ["crypto"] * 9
        assert result.asset_id.tolist() == ["btc_usdt"] * 9
        assert result.update_id.tolist() == list(range(102, 111))
        assert result.book_source_sequence.tolist() == list(range(3, 12))
        for column, value in {
            "best_bid_price": 10000.0,
            "best_ask_price": 10002.0,
            "mid": 10001.0,
            "book_spread": 2.0,
            "bid_depth_10": 10.0,
            "ask_depth_10": 10.0,
            "book_ofi_100ms": 0.0,
            "eligible": True,
        }.items():
            assert result[column].tolist() == [value] * 9
        assert result.trade_count_100ms.isna().all()
        manifest = json.loads((output / "_manifest" / "manifest.json").read_text())
        assert manifest["row_count"] == 9
        assert manifest["inputs"] == {}
        assert manifest["parameters"] == {
            **config["parameters"],
            "time_window": config["time_window"],
        }
        evidence = manifest["source_evidence"]
        assert evidence["schema_id"] == "qrp-tardis-native-archive"
        assert evidence["market_metadata"] == {
            "BTCUSDT": {"price_unit": "1", "snapshot_limit": 20}
        }
        source_manifest = json.loads((archive / "manifest.json").read_text())
        identity = canonical_json_hash(source_manifest)
        assert evidence["archives"] == [
            {
                "date": "2026-09-01",
                "schema_id": "qrp-tardis-native-archive",
                "manifest_hash": identity,
                "artifact_id": "native_archive_" + identity[:24],
            }
        ]
        diagnostics = manifest["producer_diagnostics"]
        assert diagnostics["raw_record_count"] == 31
        assert diagnostics["source_file_count"] == 1
        assert (
            diagnostics["available_assets"]
            == diagnostics["requested_assets"]
            == ["BTCUSDT"]
        )
        assert diagnostics["unavailable_assets"] == {}
        assert diagnostics["reconstruction"]["BTCUSDT"]["emitted_states"] == 10
        assert diagnostics["reconstruction"]["BTCUSDT"]["valid_segments"] == 1
    else:
        expected = primary if primary is not None else close_error
        with pytest.raises(type(expected)) as caught:
            prepare(config)
        assert caught.value is expected
        if primary is not None:
            assert primary.__notes__ == ["iterator close also failed: OSError"]
        assert not output.exists()
    assert len(opened) == 1
    assert closed == [(True, False)]
