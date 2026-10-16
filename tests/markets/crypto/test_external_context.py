"""External observations preserve units, causal clocks, and complete windows."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_external_context_frames,
)


def test_external_units_clocks_windows_and_chunk_invariance():
    time = pd.date_range("2026-08-01T00:00:01", periods=10, freq="100ms")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": time,
            "segment_id": "a",
            "mid": 100.0,
        }
    )
    quotes = [
        {
            "local_timestamp": stamp.value // 1000,
            "timestamp": stamp.value // 1000 - 7000,
            "bid_price": 199.0 + i,
            "ask_price": 201.0 + i,
        }
        for i, stamp in enumerate(time)
    ]
    trades = [
        {
            "local_timestamp": time[1].value // 1000,
            "timestamp": time[0].value // 1000,
            "side": "buy",
            "price": 200.0,
            "amount": 3.0,
        },
        {
            "local_timestamp": time[3].value // 1000,
            "timestamp": time[0].value // 1000,
            "side": "sell",
            "price": 210.0,
            "amount": 1.0,
        },
    ]

    def compute(frames, quote_rows=quotes, trade_rows=trades, multiplier=1.0):
        return pd.concat(
            iter_external_context_frames(
                frames,
                quotes=quote_rows,
                trades=trade_rows,
                venue="bybit",
                base_units_per_amount=multiplier,
                max_age_ms=100,
            ),
            ignore_index=True,
        )

    actual = compute([frame])
    split = compute([frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]])
    pd.testing.assert_frame_equal(actual, split, check_exact=True)
    mature = actual.iloc[5]
    assert mature.bybit_return_500ms == pytest.approx(205 / 200 - 1)
    assert mature.bybit_observed_signed_base_flow_500ms == 2
    assert mature.bybit_observed_signed_notional_flow_500ms == 390
    assert mature.bybit_observed_tfi_500ms == 0.5
    assert actual.bybit_observed_tfi_500ms.iloc[:5].isna().all()
    assert pd.isna(actual.bybit_observed_tfi_500ms.iloc[9])
    assert mature.bybit_price_difference == pytest.approx(1.05)
    assert mature.bybit_relative_mid_dispersion == pytest.approx(105 / 305)
    contracts = [{**row, "amount": row["amount"] * 100} for row in trades]
    converted = compute([frame], trade_rows=contracts, multiplier=0.01)
    pd.testing.assert_frame_equal(actual, converted, check_exact=True)
    future = [
        {**row, "bid_price": 1000.0, "ask_price": 1002.0} if i >= 7 else row
        for i, row in enumerate(quotes)
    ]
    pd.testing.assert_frame_equal(
        actual.iloc[:7], compute([frame], quote_rows=future).iloc[:7], check_exact=True
    )
    stale = compute([frame], quote_rows=quotes[:2])
    assert pd.isna(stale.bybit_mid_usdt.iloc[3])
    assert stale.bybit_quote_age_ms.iloc[3] == 200
    gap = frame.copy()
    gap.loc[5:, "segment_id"] = "b"
    reset = compute([gap])
    assert reset.bybit_return_500ms.iloc[5:].isna().all()
    assert reset.bybit_observed_tfi_500ms.iloc[5:].isna().all()


@pytest.mark.parametrize("venue", ("bybit", "okx"))
def test_external_archive_units_clocks_and_invalid_rows_reach_publication(
    tmp_path, monkeypatch, venue
):
    import gzip
    import hashlib
    import io
    import json
    from dataclasses import replace
    from functools import partial
    from types import SimpleNamespace

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto import ids
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_panel_frames,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog as processed_module,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer,
    )

    hft_processed_catalog = processed_module.hft_processed_catalog
    from quant_research_platform.sources.tardis import normalized
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.binance_hft import _coverage, coverage_lookup
    from tests.helpers.paths import StorageLayout
    from tests.markets.crypto.test_hft import _state

    class Response(io.BytesIO):
        status = 200

    layout = StorageLayout(tmp_path / "project")
    symbols = (
        ("BTCUSDT", "ETHUSDT")
        if venue == "bybit"
        else ("BTC-USDT-SWAP", "ETH-USDT-SWAP")
    )
    config = {
        "research_plan": {"folds": [{"train": {"start": "2026-07-01T00:00:00"}}]},
        "windows": {"development": {"end": "2026-08-01T00:00:10"}},
        "auxiliary_sources": {
            venue: {
                "archives": {
                    symbol: {
                        kind: [
                            f"{date}-{symbol}-{kind}"
                            for date in ("2026-07-01", "2026-08-01", "2026-09-01")
                        ]
                        for kind in ("quotes", "trades")
                    }
                    for symbol in symbols
                }
            }
        },
    }
    processed = hft_processed_catalog()
    data_id = ids.HFT_EXTERNAL_DATA_IDS[venue]
    instrument = producer.EXTERNAL_INSTRUMENTS[venue]
    archives = config["auxiliary_sources"][venue]["archives"]
    source_window = TimeWindow(
        min(fold["train"]["start"] for fold in config["research_plan"]["folds"]),
        config["windows"]["development"]["end"],
    )
    identity_reader = partial(
        normalized.source_identity,
        exchange=instrument["exchange"],
        archives=archives,
        time_window=source_window,
    )
    binding = SimpleNamespace(
        raw_root=layout.raw_source_root("tardis") / instrument["exchange"],
        source_identity=identity_reader,
        producer=partial(
            producer.produce_normalized_context,
            read_records=normalized.iter_records,
            source_schema_id=normalized.SCHEMA_ID,
            archives=archives,
        ),
    )
    spec = replace(
        processed.get(data_id),
        parameters_json=json.dumps(
            {
                "venue": venue,
                "symbols": instrument["symbols"],
                "base_units_per_amount": instrument["base_units_per_amount"],
                "max_age_ms": 200,
                "input_delay_ms": 0,
            }
        ),
    )
    panel_spec = processed.get(ids.HFT_PANEL_DATA_ID)
    start = pd.Timestamp("2026-08-01T00:00:01")
    start_ms = start.value // 1000000
    window = TimeWindow(start, start + pd.Timedelta(milliseconds=600))
    assert json.loads(spec.parameters_json)["max_age_ms"] == 200
    expected = {
        ("crypto", asset, start + pd.Timedelta(milliseconds=offset)): (mid, age)
        for asset in ("btc_usdt", "eth_usdt")
        for offset in range(100, 601, 100)
        for mid, age in [(100.0 if offset == 100 else None, offset + 5.0)]
    }
    clock = (start_ms - 5) * 1000
    headers_and_values = {
        "quotes": ("ask_amount,ask_price,bid_price,bid_amount", "1,101,99,1"),
        "trades": ("id,side,price,amount", "1,buy,100,1"),
    }
    manifests = {}
    for symbol, kinds in config["auxiliary_sources"][venue]["archives"].items():
        manifests[symbol] = {}
        for kind, relatives in kinds.items():
            manifests[symbol][kind] = []
            header, values = headers_and_values[kind]
            exchange = "bybit" if venue == "bybit" else "okex-swap"
            for relative in relatives:
                date = relative[:10]
                archive_clock = (
                    clock + (pd.Timestamp(date) - start.normalize()).value // 1000
                )
                raw = (
                    f"exchange,symbol,timestamp,local_timestamp,{header}\n"
                    f"{exchange},{symbol},{archive_clock - 1000},"
                    f"{archive_clock},{values}\n"
                )
                if kind == "trades":
                    buy_amount = 3 if symbol.startswith("BTC") else 5
                    for trade_id, offset, side, price, amount in (
                        (2, 205000, "buy", 100, buy_amount),
                        (3, 405000, "sell", 110, 1),
                    ):
                        observed = archive_clock + offset
                        raw += (
                            f"{exchange},{symbol},{observed - 1000},{observed},"
                            f"{trade_id},{side},{price},{amount}\n"
                        )
                raw = raw.encode()
                compressed = gzip.compress(raw, mtime=0)
                monkeypatch.setattr(
                    "urllib.request.urlopen",
                    lambda *args, payload=compressed, **kwargs: Response(payload),
                )
                manifests[symbol][kind].append(
                    normalized.download_dataset(
                        binding.raw_root / relative,
                        exchange=exchange,
                        dataset_type=kind,
                        symbol=symbol,
                        date=date,
                    )
                )
    source_start = min(
        fold["train"]["start"] for fold in config["research_plan"]["folds"]
    )[:10]
    source_end = config["windows"]["development"]["end"][:10]
    expected_hash = canonical_json_hash(
        {
            symbol: {
                kind: [
                    canonical_json_hash(m)
                    for m in members
                    if source_start <= m["date"] <= source_end
                ]
                for kind, members in kinds.items()
            }
            for symbol, kinds in manifests.items()
        }
    )
    assert binding.source_identity(binding.raw_root, data_id) == expected_hash
    frames = []
    for asset in ("btc_usdt", "eth_usdt"):
        frames.extend(
            iter_panel_frames(
                [_state(start_ms)],
                [],
                market_id="crypto",
                asset_id=asset,
                start_ms=start_ms,
                end_ms=start_ms + 600,
                book_coverage={"s1": _coverage(start_ms, start_ms + 700)},
                trade_coverage=coverage_lookup(
                    {"trade-connection-1": _coverage(start_ms, start_ms + 700)}
                ),
                batch_rows=10,
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    frames.clear()
    panel_keys = list(
        panel.loc[:, ["market_id", "asset_id", "timestamp"]].itertuples(
            index=False, name=None
        )
    )
    assert len(panel_keys) == len(expected) and set(panel_keys) == set(expected)
    panel_keys = None
    write_dataset(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        panel,
        spec=panel_spec,
    )
    panel = None
    output_root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
    request = dict(
        inputs={
            ids.HFT_PANEL_DATA_ID: layout.processed_data_root(
                panel_spec.market_id, panel_spec.kind, panel_spec.data_id
            )
        },
        time_window=window,
        processed_catalog=processed,
        source_catalog=None,
        raw_root=binding.raw_root,
    )
    from collections import Counter
    from pathlib import Path

    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.sources.tardis import raw as raw_owner
    from tests.sources.test_preparation_source_io import CountedFile

    opened, consumed, manifest_reads, handles = Counter(), Counter(), Counter(), []
    open_file, read_manifest = raw_owner.open_filesystem_file, normalized.read_manifest

    def tracked_file(path, *args, **kwargs):
        handle = open_file(path, *args, **kwargs)
        path = Path(path)
        opened[path] += 1
        handles.append(handle)
        return CountedFile(handle, path, consumed)

    def tracked_manifest(path):
        manifest_reads[path] += 1
        return read_manifest(path)

    with monkeypatch.context() as audit:
        audit.setattr(raw_owner, "open_filesystem_file", tracked_file)
        audit.setattr(normalized, "read_manifest", tracked_manifest)
        prepare(
            {
                "market_id": "crypto",
                "data_id": data_id,
                "inputs": {key: str(value) for key, value in request["inputs"].items()},
                "time_window": {
                    "start": start.isoformat(),
                    "end": window.end.isoformat(),
                },
                "output_root": str(output_root),
                "source": {"root": str(binding.raw_root), "archives": archives},
                "parameters": {"max_age_ms": 200, "input_delay_ms": 0},
            }
        )
    selected_paths = [
        binding.raw_root / relative / "data.csv.gz"
        for kinds in archives.values()
        for paths in kinds.values()
        for relative in paths
        if relative[:10] <= source_end
    ]
    assert opened == Counter({path: 1 for path in selected_paths})
    assert consumed == Counter({path: path.stat().st_size for path in selected_paths})
    assert manifest_reads and set(manifest_reads.values()) == {1}
    assert handles and all(handle.closed for handle in handles)
    session = open_dataset_session(output_root, columns=None, time_window=None)
    assert session.manifest["source_evidence"] == {
        "schema_id": normalized.SCHEMA_ID,
        "artifact_id": data_id,
        "manifest_hash": expected_hash,
    }
    seen = set()
    for frame in iter_dataset_chunks(session):
        columns = [
            "market_id",
            "asset_id",
            "timestamp",
            venue + "_mid_usdt",
            venue + "_quote_age_ms",
        ]
        for market, asset, stamp, mid, age in frame.loc[:, columns].itertuples(
            index=False, name=None
        ):
            key = (market, asset, stamp)
            assert key not in seen
            expected_mid, expected_age = expected[key]
            assert age == expected_age
            if expected_mid is None:
                assert pd.isna(mid)
            else:
                assert mid == expected_mid
            seen.add(key)
        for row in frame.to_dict("records"):
            stamp = row["timestamp"]
            base_flow = row[venue + "_observed_signed_base_flow_500ms"]
            if stamp < start + pd.Timedelta(milliseconds=600):
                assert pd.isna(base_flow)
                continue
            btc = row["asset_id"] == "btc_usdt"
            multiplier = 1 if venue == "bybit" else 0.01 if btc else 0.1
            buy_amount = 3 if btc else 5
            assert base_flow == pytest.approx((buy_amount - 1) * multiplier)
            assert row[venue + "_observed_signed_notional_flow_500ms"] == pytest.approx(
                (buy_amount * 100 - 110) * multiplier
            )
            assert row[venue + "_observed_tfi_500ms"] == pytest.approx(
                (buy_amount - 1) / (buy_amount + 1)
            )
            for kind, observed_ms, event_ms, age_ms in (
                ("quote", -5, -6, 605),
                ("trade", 400, 399, 200),
            ):
                assert row[venue + "_" + kind + "_age_ms"] == age_ms
                assert row[venue + "_" + kind + "_observed_at"] == start + pd.Timedelta(
                    milliseconds=observed_ms
                )
                assert row[venue + "_" + kind + "_event_at"] == start + pd.Timedelta(
                    milliseconds=event_ms
                )
        frame = None
    assert seen == set(expected)
    good = {
        p.relative_to(output_root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output_root.rglob("*")
        if p.is_file()
    }
    guard_symbol = json.loads(spec.parameters_json)["symbols"]["btc_usdt"]
    guard_paths = config["auxiliary_sources"][venue]["archives"][guard_symbol]["quotes"]
    guard_manifest = (
        binding.raw_root
        / next(path for path in guard_paths if path[:10] == start.date().isoformat())
        / "manifest.json"
    )
    original = guard_manifest.read_bytes()
    changed = json.loads(original)
    changed["row_count"] += 1
    rejected_root = StorageLayout(tmp_path / "rejected").processed_data_root(
        spec.market_id, spec.kind, data_id
    )
    guard_manifest.write_text(json.dumps(changed))
    try:
        with pytest.raises(ValueError, match="tardis_normalized_coverage_changed"):
            with binding.producer(spec, **request) as (chunks, metadata):
                write_dataset(rejected_root, chunks, spec=spec, metadata=metadata)
    finally:
        guard_manifest.write_bytes(original)
    assert not rejected_root.exists()
    assert {
        p.relative_to(output_root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output_root.rglob("*")
        if p.is_file()
    } == good
    assert binding.source_identity(binding.raw_root, data_id) == expected_hash

    wrong_scope = {symbol: dict(kinds) for symbol, kinds in archives.items()}
    wrong_scope[symbols[0]]["quotes"] = archives[symbols[1]]["quotes"]
    with pytest.raises(ValueError, match="tardis_normalized_scope_invalid"):
        with binding.producer(spec, **request, archives=wrong_scope) as (
            chunks,
            metadata,
        ):
            write_dataset(rejected_root, chunks, spec=spec, metadata=metadata)
    assert not rejected_root.exists()

    # Even an empty target selection must admit every required source before publish.
    empty_request = request | {
        "inputs": {
            ids.HFT_PANEL_DATA_ID: {
                "path": str(request["inputs"][ids.HFT_PANEL_DATA_ID]),
                "row_filters": [["asset_id", "==", "absent"]],
            }
        }
    }
    source_file = guard_manifest.parent / "data.csv.gz"
    original_data = source_file.read_bytes()
    source_file.unlink()
    try:
        with pytest.raises(ValueError, match="tardis_raw_identity_changed"):
            with binding.producer(spec, **empty_request) as (chunks, metadata):
                assert "manifest_hash" not in metadata["source_evidence"]
                write_dataset(rejected_root, chunks, spec=spec, metadata=metadata)
    finally:
        source_file.write_bytes(original_data)
    assert not rejected_root.exists()


@pytest.mark.parametrize(
    "asset,old,new,transition",
    (
        ("btc_usdt", 0.0001, 0.01, "2020-03-20T08:00:00"),
        ("eth_usdt", 0.001, 0.1, "2020-03-19T08:00:00"),
    ),
)
@pytest.mark.parametrize("regime", ("old", "new", "mixed", "uncertain"))
def test_okx_dated_face_value_native_csv_to_persisted_flow(
    tmp_path, asset, old, new, transition, regime
):
    import gzip
    import hashlib

    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog as processed_module,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer as crypto_producer,
    )

    hft_processed_catalog = processed_module.hft_processed_catalog
    from quant_research_platform.sources.tardis.normalized import _rows
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.paths import StorageLayout

    definition = crypto_producer.EXTERNAL_INSTRUMENTS["okx"]
    start = (
        pd.Timestamp(transition)
        + {
            "old": -pd.Timedelta(days=1),
            "new": pd.Timedelta(hours=1),
            "mixed": pd.Timedelta(hours=1),
            "uncertain": pd.Timedelta(minutes=1),
        }[regime]
    )
    times = pd.date_range(start, periods=7, freq="100ms").as_unit("ns")
    event = times[2].value // 1000
    symbol = definition["symbols"][asset]
    first_event = (
        pd.Timestamp(transition).value // 1000 - 1 if regime == "mixed" else event
    )
    raw = (
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        + f"okex-swap,{symbol},{first_event},{event},1,buy,10000,100\n"
        + (
            f"okex-swap,{symbol},{event + 100000},{event + 100000},2,buy,10000,100\n"
            if regime == "mixed"
            else ""
        )
    ).encode()
    compressed = gzip.compress(raw, mtime=0)
    path = tmp_path / "trades.csv.gz"
    path.write_bytes(compressed)
    trades = [
        row
        for _, row in _rows(
            path,
            exchange="okex-swap",
            dataset_type="trades",
            symbol=symbol,
            date=str(start.date()),
            sha256=hashlib.sha256(compressed).hexdigest(),
        )
    ]
    frames = [
        pd.DataFrame(
            {
                "market_id": "crypto",
                "asset_id": asset,
                "timestamp": times,
                "segment_id": "s",
                "mid": 10000.0,
            }
        )
    ]
    output = iter_external_context_frames(
        frames,
        quotes=[],
        trades=trades,
        venue="okx",
        base_units_per_amount=definition["base_units_per_amount"][asset],
        max_age_ms=100,
    )
    if regime == "uncertain":
        with pytest.raises(
            ValueError, match="external_contract_face_value_transition_unproven"
        ):
            list(output)
        return
    spec = hft_processed_catalog().get("okx_perpetual_context_100ms")
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(
        root, (frame.loc[:, list(spec.required_columns)] for frame in output), spec=spec
    )
    saved = read_dataset(root)
    row = saved.iloc[5]
    unit = old + new if regime == "mixed" else old if regime == "old" else new
    assert row.okx_observed_signed_base_flow_500ms == pytest.approx(100 * unit)
    assert row.okx_observed_signed_notional_flow_500ms == pytest.approx(
        1_000_000 * unit
    )


@pytest.mark.parametrize(
    "bid,ask,anchor",
    (
        (
            float.fromhex("0x0.0000000000001p-1022"),
            float.fromhex("0x0.0000000000002p-1022"),
            float.fromhex("0x0.0000000000002p-1022"),
        ),
        (
            float.fromhex("0x0.0000000000002p-1022"),
            float.fromhex("0x0.0000000000004p-1022"),
            float.fromhex("0x0.0000000000002p-1022"),
        ),
        (99.0, 101.0, 100.0),
        (1e308, 1.1e308, 1.2e308),
    ),
)
def test_external_price_expressions_match_complete_fraction(bid, ask, anchor):
    from fractions import Fraction

    stamp = pd.Timestamp("2026-08-01")
    frame = pd.DataFrame(
        {
            "market_id": ["crypto"],
            "asset_id": ["btc_usdt"],
            "timestamp": [stamp],
            "segment_id": ["a"],
            "mid": [anchor],
        }
    )
    quotes = [
        {
            "local_timestamp": stamp.value // 1000,
            "timestamp": stamp.value // 1000,
            "bid_price": bid,
            "ask_price": ask,
        }
    ]
    actual = next(
        iter_external_context_frames(
            [frame],
            quotes=quotes,
            trades=(),
            venue="bybit",
            base_units_per_amount=1.0,
            max_age_ms=100,
        )
    ).iloc[0]
    mid = float((Fraction(bid) + Fraction(ask)) / 2)
    dispersion = float(
        abs(Fraction(mid) - Fraction(anchor)) / (Fraction(mid) + Fraction(anchor))
    )
    assert actual.bybit_mid_usdt == mid
    assert actual.bybit_relative_mid_dispersion == pytest.approx(
        dispersion, rel=1e-15, abs=0
    )


@pytest.mark.parametrize("steps", (0, 1, 2))
def test_external_adjacent_mid_returns_keep_order_and_chunk_state(steps):
    from fractions import Fraction

    import numpy as np

    start, end = 3.0, 3.0
    for _ in range(steps):
        end = np.nextafter(end, np.inf)
    time = pd.date_range("2026-08-01", periods=6, freq="100ms")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": time,
            "segment_id": "a",
            "mid": 3.0,
        }
    )
    mids = [start] * 5 + [end]
    quotes = [
        {
            "local_timestamp": stamp.value // 1000,
            "timestamp": stamp.value // 1000,
            "bid_price": np.nextafter(mid, -np.inf),
            "ask_price": np.nextafter(mid, np.inf),
        }
        for stamp, mid in zip(time, mids, strict=True)
    ]

    def compute(frames):
        return pd.concat(
            iter_external_context_frames(
                frames,
                quotes=quotes,
                trades=(),
                venue="bybit",
                base_units_per_amount=1.0,
                max_age_ms=100,
            ),
            ignore_index=True,
        )

    full, split = compute([frame]), compute([frame.iloc[:5], frame.iloc[5:]])
    pd.testing.assert_frame_equal(full, split, check_exact=True)
    expected = float((Fraction(end) - Fraction(start)) / Fraction(start))
    assert full.bybit_return_100ms.iloc[-1] == expected
    assert full.bybit_return_500ms.iloc[-1] == expected


@pytest.mark.parametrize("bad_field", ["price", "amount", "side"])
def test_external_bad_trade_value_does_not_erase_quotes_or_unrelated_flow(bad_field):
    times = pd.date_range("2026-08-01", periods=12, freq="100ms")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": times,
            "segment_id": "s",
            "mid": 100.0,
        }
    )
    quotes = [
        {
            "local_timestamp": time.value // 1000,
            "timestamp": time.value // 1000,
            "bid_price": 99.0,
            "ask_price": 101.0,
        }
        for time in times
    ]
    trades = [
        {
            "local_timestamp": time.value // 1000,
            "timestamp": time.value // 1000,
            "side": "buy",
            "amount": 1.0,
            "price": 100.0,
        }
        for time in times
    ]
    trades[4][bad_field] = None
    result = pd.concat(
        iter_external_context_frames(
            [frame.iloc[:5], frame.iloc[5:]],
            quotes=quotes,
            trades=trades,
            venue="bybit",
            base_units_per_amount=1.0,
            max_age_ms=100,
        ),
        ignore_index=True,
    )
    assert result.bybit_mid_usdt.notna().all()
    assert result.bybit_return_100ms.iloc[1:].eq(0).all()
    assert result.bybit_observed_signed_notional_flow_500ms.iloc[5:9].isna().all()
    assert result.bybit_observed_signed_notional_flow_500ms.iloc[9:].eq(500).all()
    if bad_field == "price":
        assert result.bybit_observed_signed_base_flow_500ms.iloc[5:].eq(5).all()
        assert result.bybit_observed_tfi_500ms.iloc[5:].eq(1).all()
    else:
        assert result.bybit_observed_signed_base_flow_500ms.iloc[5:9].isna().all()


def test_new_invalid_quote_replaces_prior_quote_until_valid_observation():
    times = pd.date_range("2026-08-01", periods=5, freq="100ms")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": times,
            "segment_id": "s",
            "mid": 100.0,
        }
    )
    quotes = [
        {
            "local_timestamp": times[index].value // 1000,
            "timestamp": times[index].value // 1000,
            "bid_price": bid,
            "ask_price": 101.0,
        }
        for index, bid in ((0, 99.0), (1, None), (3, 99.0))
    ]
    result = pd.concat(
        iter_external_context_frames(
            [frame.iloc[:2], frame.iloc[2:]],
            quotes=quotes,
            trades=(),
            venue="bybit",
            base_units_per_amount=1.0,
            max_age_ms=1000,
        ),
        ignore_index=True,
    )
    assert result.bybit_mid_usdt.iloc[[0, 3, 4]].eq(100).all()
    assert result.bybit_mid_usdt.iloc[1:3].isna().all()
    assert result.bybit_quote_age_ms.tolist() == [0, 0, 100, 0, 100]
    assert result.bybit_quote_observed_at.iloc[2] == times[1]
    assert result.bybit_return_100ms.iloc[1:4].isna().all()
    assert result.bybit_return_100ms.iloc[4] == 0
