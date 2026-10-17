"""Normalized archive identity and bounded causal observation input checks."""

import gzip
import io
import json

import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.sources.tardis.normalized import (
    download_dataset,
    iter_records,
)


class Response(io.BytesIO):
    status = 200


def test_normalized_original_bytes_nulls_clock_and_hash(tmp_path, monkeypatch):
    raw = gzip.compress(
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\nbybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\nbybit,BTCUSDT,1785542400000000,1785542400000002,2,sell,100,1\n",
        mtime=0,
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    root = tmp_path / "archive"
    manifest = download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    assert (root / "data.csv.gz").read_bytes() == raw
    assert manifest["row_count"] == 2
    rows = list(
        iter_records(
            root.parent,
            archive_paths=(root.name,),
            time_window=TimeWindow("2026-08-01", "2026-08-01"),
        )
    )
    assert rows[0]["amount"] == 2.0
    assert rows[0]["timestamp"] < rows[0]["local_timestamp"]
    (root / "data.csv.gz").write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        list(
            iter_records(
                root.parent,
                archive_paths=(root.name,),
                time_window=TimeWindow("2026-08-01", "2026-08-01"),
            )
        )


def test_normalized_rejects_wrong_schema_and_preserves_existing_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: Response(gzip.compress(b"bad,header\n")),
    )
    with pytest.raises(ValueError, match="schema_invalid"):
        download_dataset(
            tmp_path / "bad",
            exchange="okex-swap",
            dataset_type="quotes",
            symbol="BTC-USDT-SWAP",
            date="2026-08-01",
        )
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="scope_invalid"):
        download_dataset(
            tmp_path / "other",
            exchange="bybit",
            dataset_type="incremental_book_L2",
            symbol="BTCUSDT",
            date="2026-08-01",
        )


def test_source_identity_rejects_equal_length_changed_gzip(tmp_path, monkeypatch):
    from quant_research_platform.sources.tardis.normalized import source_identity

    csv = (
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n"
    )
    raw = gzip.compress(csv, mtime=0)
    changed = gzip.compress(csv.replace(b",100,2\n", b",100,3\n"), mtime=0)
    assert len(raw) == len(changed)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: Response(raw))
    root = tmp_path / "archive"
    download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    scope = dict(
        exchange="bybit",
        archives={"BTCUSDT": {"trades": [root.name]}},
        time_window=TimeWindow("2026-08-01", "2026-08-01"),
    )
    assert source_identity(tmp_path, "synthetic", **scope) is not None
    (root / "data.csv.gz").write_bytes(changed)
    with pytest.raises(ValueError, match="^tardis_raw_identity_changed$"):
        source_identity(tmp_path, "synthetic", **scope)


@pytest.mark.parametrize("phase", ["before", "open_race"])
def test_declared_normalized_member_missing_is_identity_failure(
    tmp_path, monkeypatch, phase
):
    import quant_research_platform.sources.tardis.raw as raw_owner
    from quant_research_platform.sources.tardis.normalized import source_identity

    root = tmp_path / "archive"
    scope = dict(exchange="bybit", archives={"BTCUSDT": {"trades": [root.name]}})
    with pytest.raises(ValueError, match="^dataset_metadata_file_invalid$"):
        source_identity(tmp_path, "synthetic", **scope)
    raw = gzip.compress(
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n",
        mtime=0,
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    assert source_identity(tmp_path, "synthetic", **scope) is not None
    path = root / "data.csv.gz"
    window = TimeWindow("2026-08-01", "2026-08-01")
    if phase == "during":
        rows = iter_records(tmp_path, archive_paths=(root.name,), time_window=window)
        next(rows)
        path.unlink()
        primary = KeyboardInterrupt("synthetic_consumer_failure")
        with pytest.raises(KeyboardInterrupt) as caught:
            rows.throw(primary)
        assert caught.value is primary
        assert not getattr(primary, "__notes__", [])
    elif phase in {"capture_race", "open_race"}:
        operation = (
            "dataset_file_identity"
            if phase == "capture_race"
            else "open_filesystem_file"
        )
        original = getattr(raw_owner, operation)

        def acquire_member(candidate, *args, **kwargs):
            if candidate == path:
                path.unlink()
            return original(candidate, *args, **kwargs)

        monkeypatch.setattr(raw_owner, operation, acquire_member)
    else:
        path.unlink()
    with pytest.raises(ValueError, match="^tardis_raw_identity_changed$"):
        source_identity(tmp_path, "synthetic", **scope)
    with pytest.raises(ValueError, match="^tardis_raw_identity_changed$"):
        list(iter_records(tmp_path, archive_paths=(root.name,), time_window=window))


@pytest.mark.parametrize("valid_schema", [False, True])
def test_normalized_gzip_close_preserves_parser_failure(
    tmp_path, monkeypatch, valid_schema
):
    import hashlib

    from quant_research_platform.sources.tardis.normalized import _rows

    raw = gzip.compress(
        (
            b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
            b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n"
        )
        if valid_schema
        else b"bad,header\n",
        mtime=0,
    )
    path = tmp_path / "source.csv.gz"
    path.write_bytes(raw)
    secondary = OSError("synthetic_gzip_close_failure")
    closed = []

    class FailingCloseGzip(gzip.GzipFile):
        def close(self):
            was_closed = self.closed
            super().close()
            if not was_closed:
                closed.append(self.closed)
                raise secondary

    monkeypatch.setattr(gzip, "GzipFile", FailingCloseGzip)
    with pytest.raises(OSError if valid_schema else ValueError) as caught:
        list(
            _rows(
                path,
                exchange="bybit",
                dataset_type="trades",
                symbol="BTCUSDT",
                date="2026-08-01",
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    assert str(caught.value) == (
        "synthetic_gzip_close_failure"
        if valid_schema
        else "tardis_normalized_schema_invalid"
    )
    assert closed == [True]
    if valid_schema:
        assert caught.value is secondary
    else:
        assert any("OSError" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("dataset_type", ["quotes", "trades", "derivative_ticker"])
def test_offline_publication_revalidates_original_bytes_without_iv(
    tmp_path, dataset_type
):
    from quant_research_platform.sources.tardis.normalized import (
        _COLUMNS,
        _publish_dataset,
    )

    fields = _COLUMNS[dataset_type]
    values = dict.fromkeys(fields, "1")
    values.update(
        exchange="bybit",
        symbol="BTCUSDT",
        timestamp="1785542400000000",
        local_timestamp="1785542400000001",
        side="buy",
    )
    raw = gzip.compress(
        (
            ",".join(fields) + "\n" + ",".join(values[key] for key in fields) + "\n"
        ).encode()
    )
    root = tmp_path / "archive"
    manifest = _publish_dataset(
        root,
        lambda: io.BytesIO(raw),
        exchange="bybit",
        dataset_type=dataset_type,
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    assert "iv_unit" not in manifest
    assert (root / "data.csv.gz").read_bytes() == raw
    assert (
        len(
            list(
                iter_records(
                    root.parent,
                    archive_paths=(root.name,),
                    time_window=TimeWindow("2026-08-01", "2026-08-01"),
                )
            )
        )
        == 1
    )
    for key, value in (("amount_unit", "wrong"), ("iv_unit", None)):
        changed = manifest | {key: value}
        (root / "manifest.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="tardis_normalized_manifest_invalid"):
            list(
                iter_records(
                    root.parent,
                    archive_paths=(root.name,),
                    time_window=TimeWindow("2026-08-01", "2026-08-01"),
                )
            )


@pytest.mark.parametrize(
    "damage", (None, "digest", "coverage", "gzip_tail", "row_tail")
)
def test_early_close_does_not_read_unrequested_tail(tmp_path, monkeypatch, damage):
    import hashlib
    from collections import Counter

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.sources.tardis import raw as raw_owner
    from quant_research_platform.sources.tardis.normalized import _publish_dataset
    from tests.sources.test_preparation_source_io import CountedFile

    text = "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
    text += "".join(
        f"bybit,BTCUSDT,1785542400000000,{1785542400000000 + i},{i},buy,100,2\n"
        for i in range(20000)
    )
    raw = gzip.compress(text.encode(), compresslevel=0, mtime=0)
    root = tmp_path / "archive"
    manifest = _publish_dataset(
        root,
        lambda: io.BytesIO(raw),
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    if damage == "coverage":
        manifest["row_count"] += 1
    elif damage == "digest":
        raw = raw[:4] + bytes([raw[4] ^ 1]) + raw[5:]
    elif damage == "gzip_tail":
        raw = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
    elif damage == "row_tail":
        prefix, last = text.rstrip("\n").rsplit("\n", 1)
        raw = gzip.compress(
            (prefix + "\n" + last.replace(",BTCUSDT,", ",ETHUSDT,") + "\n").encode(),
            compresslevel=0,
            mtime=0,
        )
        manifest.update(
            sha256=hashlib.sha256(raw).hexdigest(), compressed_bytes=len(raw)
        )
    (root / "data.csv.gz").write_bytes(raw)
    (root / "manifest.json").write_text(json.dumps(manifest))
    opened, consumed, handles = Counter(), Counter(), []
    open_file = raw_owner.open_filesystem_file

    def tracked_file(path, *args, **kwargs):
        handle = open_file(path, *args, **kwargs)
        opened[path] += 1
        handles.append(handle)
        return CountedFile(handle, path, consumed)

    monkeypatch.setattr(raw_owner, "open_filesystem_file", tracked_file)
    identities = []
    records = iter_records(
        tmp_path,
        archive_paths=["archive"],
        time_window=TimeWindow("2026-08-01", "2026-08-01"),
        expected_scope=("bybit", "BTCUSDT", "trades"),
        on_source=identities.append,
    )
    assert next(records)["id"] == "0"
    assert identities == [canonical_json_hash(manifest)]
    before_close = consumed.copy()
    records.close()
    assert consumed == before_close
    assert opened == Counter({root / "data.csv.gz": 1})
    assert 0 < consumed[root / "data.csv.gz"] < len(raw)
    assert handles and all(handle.closed for handle in handles)
    full = iter_records(tmp_path, archive_paths=["archive"], time_window=None)
    if damage in {None, "digest"}:
        assert len(list(full)) == 20000
    else:
        with pytest.raises((ValueError, OSError)):
            list(full)


def test_consumer_failure_stays_primary_without_tail_validation(tmp_path, monkeypatch):
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.sources.tardis.normalized import _publish_dataset

    raw = gzip.compress(
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        + b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n" * 2000,
        mtime=0,
    )
    root = tmp_path / "archive"
    _publish_dataset(
        root,
        lambda: io.BytesIO(raw),
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    (root / "data.csv.gz").write_bytes(raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:])
    identities = []
    records = iter_records(
        tmp_path,
        archive_paths=["archive"],
        time_window=None,
        on_source=identities.append,
    )
    primary = KeyboardInterrupt("synthetic_consumer_failure")
    with pytest.raises(KeyboardInterrupt) as caught:
        with closing_iterator(records):
            next(records)
            raise primary
    assert caught.value is primary
    assert not getattr(primary, "__notes__", [])
    assert len(identities) == 1


def test_bad_numeric_cell_keeps_raw_row_and_unrelated_fields(tmp_path, monkeypatch):
    raw = gzip.compress(
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\nbybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,NaN,2\nbybit,BTCUSDT,1785542400000000,1785542400000002,2,sell,100,-1\nbybit,BTCUSDT,1785542400000000,1785542400000003,3,sell,101,1\nbybit,BTCUSDT,1785542400000000,1785542400000004,4,unknown,102,2\n",
        mtime=0,
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    root = tmp_path / "archive"
    manifest = download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )
    assert manifest["row_count"] == 4
    assert (root / "data.csv.gz").read_bytes() == raw
    rows = list(
        iter_records(
            root.parent,
            archive_paths=(root.name,),
            time_window=TimeWindow("2026-08-01", "2026-08-01"),
        )
    )
    assert rows[0]["price"] is None and rows[0]["amount"] == 2
    assert rows[1]["price"] == 100 and rows[1]["amount"] is None
    assert rows[2]["price"] == 101 and rows[2]["amount"] == 1
    assert rows[3]["side"] is None and rows[3]["amount"] == 2
