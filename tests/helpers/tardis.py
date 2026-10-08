"""Small native Tardis archives for source and replay tests."""

import gzip
import json
from pathlib import Path


def _slices(root: Path, first: bytes = b"", *, count=1):
    root.mkdir()
    result = []
    for offset in range(0, count * 10, 10):
        path = root / f"native-{offset:04d}.gz"
        path.write_bytes(gzip.compress(first if offset == 0 else b"", mtime=0))
        result.append(path)
    return result


def _snapshot(timestamp="2026-09-01T00:00:01.000000Z"):
    value = {
        "stream": "btcusdt@depthSnapshot",
        "generated": True,
        "data": {"lastUpdateId": 100, "bids": [["100", "2"]], "asks": [["101", "2"]]},
    }
    return (timestamp + " " + json.dumps(value) + "\n").encode()


def _line(second, channel, payload):
    value = {"stream": "btcusdt@" + channel, "data": payload}
    if channel == "depthSnapshot":
        value["generated"] = True
    return (
        f"2026-09-01T00:00:{second:02d}.000000Z " + json.dumps(value) + "\n"
    ).encode()
