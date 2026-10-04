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
                tickers=[],
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
        actual.iloc[:7],
        compute([frame], quote_rows=future).iloc[:7],
        check_exact=True,
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
def test_external_archive_collection_identity_persists_and_guards_publication(
    tmp_path, monkeypatch, venue
):
    import gzip
    import hashlib
    import io
    import json

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto import ids
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_panel_frames,
    )
    from quant_research_platform.profiles.research import (
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.sources.tardis import normalized
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.binance_hft import _coverage, coverage_lookup
    from tests.helpers.research_configuration import research_configuration
    from tests.helpers.runtime_resources import capture_test_policy
    from tests.markets.crypto.test_hft import _state

    class Response(io.BytesIO):
        status = 200

    layout = StorageLayout(tmp_path / "project")
    config = research_configuration("crypto", "external_context")
    profile = bind_research_configuration(
        research_profile(config["profile_id"]),
        config,
        tardis_root=layout.raw_source_root("tardis"),
    )
    data_id = ids.HFT_EXTERNAL_DATA_IDS[venue]
    binding = profile.producer_bindings[data_id]
    spec = profile.processed.get(data_id)
    panel_spec = profile.processed.get(ids.HFT_PANEL_DATA_ID)
    start = pd.Timestamp("2026-08-01T00:00:01")
    start_ms = start.value // 1_000_000
    window = TimeWindow(start, start + pd.Timedelta(milliseconds=600))
    assert json.loads(spec.parameters_json)["max_age_ms"] == 200
    # Coverage starts at t0: the first complete interval ends at t0 + 100ms.
    expected = {
        ("crypto", asset, start + pd.Timedelta(milliseconds=offset)): (mid, age)
        for asset in ("btc_usdt", "eth_usdt")
        for offset in range(100, 601, 100)
        for mid, age in [(100.0 if offset == 100 else None, offset + 5.0)]
    }
    clock = (start_ms - 5) * 1000
    headers_and_values = {
        "quotes": (
            "ask_amount,ask_price,bid_price,bid_amount",
            "1,101,99,1",
        ),
        "trades": ("id,side,price,amount", "1,buy,100,1"),
        "derivative_ticker": (
            "funding_timestamp,funding_rate,predicted_funding_rate,"
            "open_interest,last_price,index_price,mark_price",
            ",0,0,1,100,100,100",
        ),
    }
    policy = capture_test_policy(available_memory_bytes=256 * 1024**2)
    with runtime_resource_policy_scope(policy):
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
                            (2, 205_000, "buy", 100, buy_amount),
                            (3, 405_000, "sell", 110, 1),
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
                            byte_budget=1024**2,
                            resource_policy=policy,
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
        with policy.lease("test.external.identity.frames", 2 * 1024**2):
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
                resource_policy=policy,
                borrowed_input=True,
            )
            panel = None
        output_root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
        request = dict(
            layout=layout,
            time_window=window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
            resource_policy=policy,
            raw_root=binding.raw_root,
            expected_source_identity=expected_hash,
        )
        with binding.producer(spec, **request) as (chunks, metadata, verify):
            write_dataset(
                output_root,
                chunks,
                spec=spec,
                resource_policy=policy,
                instance_metadata=metadata,
                staged_validator=verify,
            )
        with open_dataset_session(
            output_root, columns=None, time_window=None, resource_policy=policy
        ) as session:
            assert session.manifest["source_artifact_identity"] == {
                "schema_id": normalized.SCHEMA_ID,
                "artifact_id": data_id,
                "manifest_hash": expected_hash,
            }
            seen = set()
            for frame in iter_dataset_chunks(session, target_chunk_bytes=1024**2):
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
                    assert row[
                        venue + "_observed_signed_notional_flow_500ms"
                    ] == pytest.approx((buy_amount * 100 - 110) * multiplier)
                    assert row[venue + "_observed_tfi_500ms"] == pytest.approx(
                        (buy_amount - 1) / (buy_amount + 1)
                    )
                    for kind, observed_ms, event_ms, age_ms in (
                        ("quote", -5, -6, 605),
                        ("trade", 400, 399, 200),
                        ("ticker", -5, -6, 605),
                    ):
                        assert row[venue + "_" + kind + "_age_ms"] == age_ms
                        assert row[
                            venue + "_" + kind + "_observed_at"
                        ] == start + pd.Timedelta(milliseconds=observed_ms)
                        assert row[
                            venue + "_" + kind + "_event_at"
                        ] == start + pd.Timedelta(milliseconds=event_ms)
                frame = None
            assert seen == set(expected)
        good = {
            p.relative_to(output_root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output_root.rglob("*")
            if p.is_file()
        }
        guard_symbol = json.loads(spec.parameters_json)["symbols"]["btc_usdt"]
        guard_paths = config["auxiliary_sources"][venue]["archives"][guard_symbol][
            "quotes"
        ]
        guard_manifest = (
            binding.raw_root
            / next(
                path for path in guard_paths if path[:10] == start.date().isoformat()
            )
            / "manifest.json"
        )
        original = guard_manifest.read_bytes()
        changed = json.loads(original)
        changed["row_count"] += 1
        rejected_root = StorageLayout(tmp_path / "rejected").processed_data_root(
            spec.market_id, spec.kind, data_id
        )
        try:
            with binding.producer(spec, **request) as (chunks, metadata, verify):

                def mutate_before_publication(receipt):
                    guard_manifest.write_text(json.dumps(changed))
                    verify(receipt)

                with pytest.raises(
                    ValueError, match="hft_external_source_identity_changed"
                ):
                    write_dataset(
                        rejected_root,
                        chunks,
                        spec=spec,
                        resource_policy=policy,
                        instance_metadata=metadata,
                        staged_validator=mutate_before_publication,
                    )
        finally:
            guard_manifest.write_bytes(original)
        assert not rejected_root.exists()
        assert {
            p.relative_to(output_root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output_root.rglob("*")
            if p.is_file()
        } == good
        assert binding.source_identity(binding.raw_root, data_id) == expected_hash
        assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED" and policy.used_memory_bytes == 0
