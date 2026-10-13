"""Native recorder-derived Tardis slices across midnight with optional real ID gaps."""

import json

import pandas as pd

from quant_research_platform.storage.identity import (
    filesystem_write_bytes,
    native_filesystem_path,
)


def write_tardis_midnight_case(tmp_path, gap_kind):
    import gzip

    from quant_research_platform.sources.tardis.raw import import_slices
    from tests.helpers.binance_hft import write_binance_hft_run

    origin = pd.Timestamp("2026-09-01T23:59:55.500000Z")
    origin_ms = origin.value // 1_000_000
    recorder = tmp_path / "fixture"
    manifest = write_binance_hft_run(recorder, count=120, maker_liquidity=True)
    native = []
    for member in manifest["files"]:
        kind = member["data_kind"]
        if kind == "metadata":
            continue
        channel = {
            "depth_snapshot": "depthSnapshot",
            "depth": "depth@0ms",
            "agg_trades": "aggTrade",
        }[kind]
        with gzip.open(
            native_filesystem_path(recorder / member["path"]), "rt"
        ) as stream:
            for line in stream:
                record = json.loads(line)
                stamp = origin + pd.Timedelta(record["observed_time_ns"], unit="ns")
                payload = record["payload"]
                if kind == gap_kind and record["observed_time_ns"] >= 8_800_000_000:
                    for key in ("U", "u", "pu") if kind == "depth" else ("a", "f", "l"):
                        payload[key] += 1
                for key in ("E", "T"):
                    if key in payload:
                        payload[key] += origin_ms
                value = {
                    "stream": member["symbol"].lower() + "@" + channel,
                    "data": payload,
                }
                if kind == "depth_snapshot":
                    value["generated"] = True
                native.append((stamp, record["capture_seq"], value))
    for symbol, price in (("BTCUSDT", "100"), ("ETHUSDT", "200")):
        native.append(
            (
                origin,
                0,
                {
                    "stream": symbol.lower() + "@markPrice@1s",
                    "data": {
                        "e": "markPriceUpdate",
                        "s": symbol,
                        "E": origin_ms,
                        "p": price,
                        "r": "0",
                        "T": origin_ms + 3_600_000,
                    },
                },
            )
        )
    native.sort(key=lambda item: item[:2])
    boundary = origin.ceil("D")
    slices = []
    for index in range(2):
        path = tmp_path / f"native-{index}.gz"
        rows = b"".join(
            (
                stamp.isoformat().replace("+00:00", "Z")
                + " "
                + json.dumps(value)
                + "\n"
            ).encode()
            for stamp, _, value in native
            if (stamp >= boundary) == bool(index)
        )
        filesystem_write_bytes(path, gzip.compress(rows, mtime=0))
        slices.append(path)
    raw = tmp_path / "archive"
    import_slices(raw, date="2026-09-01", offset_minutes=1430, slices=slices)
    return raw, origin
