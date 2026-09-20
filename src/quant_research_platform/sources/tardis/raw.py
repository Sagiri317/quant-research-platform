"""Date-addressed native gzip acquisition with immutable source identity."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import (
    price_decimal_places,
    validate_price_unit,
)
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.runtime.resources import (
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.validation import (
    strict_iso_timestamp,
    validate_market_payload,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import (
    _DigestingReader,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    require_unaliased_path,
)
from quant_research_platform.storage.raw_snapshots import MAX_RECORD_BYTES

SCHEMA_ID = "qrp-tardis-native-archive"
EXCHANGE = "binance-futures"
CHANNELS = ("depth", "depthSnapshot", "aggTrade", "markPrice")
SYMBOLS = ("btcusdt", "ethusdt")
SLICE_MINUTES = 10
_COPY_BYTES = 64 * 1024


def _date(value: str) -> dt.date:
    if type(value) is not str:
        raise ValueError("tardis_date_invalid")
    try:
        result = dt.date.fromisoformat(value)
    except ValueError:
        raise ValueError("tardis_date_invalid") from None
    if result.isoformat() != value:
        raise ValueError("tardis_date_invalid")
    return result


def _request_url(date: str, offset: int) -> str:
    date = (_date(date) + dt.timedelta(days=offset // 1440)).isoformat()
    offset %= 1440
    return (
        "https://api.tardis.dev/v1/data-feeds/"
        + EXCHANGE
        + "?"
        + urllib.parse.urlencode(
            {
                "from": date,
                "offset": offset,
                "sliceSize": SLICE_MINUTES,
                "filters": json.dumps(
                    [{"channel": item, "symbols": list(SYMBOLS)} for item in CHANNELS],
                    separators=(",", ":"),
                ),
            }
        )
    )


def _filename(date: str, offset: int) -> str:
    date = (_date(date) + dt.timedelta(days=offset // 1440)).isoformat()
    offset %= 1440
    return (
        f"tardis__{EXCHANGE}__{date}__minutes-"
        f"{offset:04d}-{offset + SLICE_MINUTES:04d}.txt.gz"
    )


@contextmanager
def _verified_file(path: Path, *, expected_sha256=None):
    require_unaliased_path(path, error="tardis_raw_identity_changed")
    identity = dataset_file_identity(path)
    with handle_preserving_primary(path.open("rb")) as handle:
        if dataset_file_identity_from_stat(os.fstat(handle.fileno())) != identity:
            raise ValueError("tardis_raw_identity_changed")
        reader = _DigestingReader(handle) if expected_sha256 is not None else handle
        try:
            yield reader
        finally:
            import sys

            primary = sys.exception()
            try:
                if expected_sha256 is not None:
                    reader.drain()
                    if reader.hexdigest() != expected_sha256:
                        raise ValueError("tardis_raw_identity_changed")
                if (
                    dataset_file_identity_from_stat(os.fstat(handle.fileno()))
                    != identity
                    or dataset_file_identity(path) != identity
                ):
                    raise ValueError("tardis_raw_identity_changed")
            except BaseException as error:
                if primary is None or isinstance(primary, GeneratorExit):
                    raise
                primary.add_note(
                    "Tardis raw identity verification also failed: "
                    + type(error).__name__
                )


def iter_native_slice(
    path: Path, *, date: str, offset_minutes: int, resource_policy, expected_sha256=None
):
    """Yield borrowed native rows, including disconnect markers, in file order."""
    day = _date(date)
    start = int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp()) * 10**9
    lower = start + offset_minutes * 60 * 10**9
    upper = lower + SLICE_MINUTES * 60 * 10**9
    previous = lower
    with (
        resource_policy.lease("tardis.native_stream", _COPY_BYTES),
        _verified_file(path, expected_sha256=expected_sha256) as raw,
        handle_preserving_primary(gzip.GzipFile(fileobj=raw, mode="rb")) as stream,
    ):
        number = 0
        while True:
            with resource_policy.lease(
                "tardis.native_record",
                json_record_working_set_bytes(MAX_RECORD_BYTES),
            ):
                line = value = payload = native = parts = timestamp = None
                try:
                    line = stream.readline(MAX_RECORD_BYTES + 1)
                    if not line:
                        return
                    number += 1
                    if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
                        raise ValueError("tardis_native_record_invalid")
                    if line in (b"\n", b"\r\n"):
                        yield number, None, None
                        continue
                    try:
                        timestamp, payload = line.decode("utf-8").split(" ", 1)
                    except (UnicodeDecodeError, ValueError):
                        raise ValueError("tardis_native_record_invalid") from None
                    observed = strict_iso_timestamp(
                        timestamp, error="tardis_observed_time_invalid"
                    ).value
                    if (
                        not timestamp.endswith("Z")
                        or not lower <= observed < upper
                        or observed < previous
                    ):
                        raise ValueError("tardis_observed_time_invalid")
                    previous = observed
                    value = parse_qrp_json_object(payload.encode("utf-8"))
                    if (
                        set(value) - {"stream", "data", "generated"}
                        or not isinstance(value.get("stream"), str)
                        or type(value.get("data")) is not dict
                    ):
                        raise ValueError("tardis_native_record_invalid")
                    parts = value["stream"].split("@")
                    if (
                        len(parts) < 2
                        or parts[0] not in SYMBOLS
                        or parts[1] not in CHANNELS
                    ):
                        raise ValueError("tardis_native_channel_invalid")
                    channel = parts[1]
                    suffix = {
                        "depth": "depth@0ms",
                        "depthSnapshot": "depthSnapshot",
                        "aggTrade": "aggTrade",
                        "markPrice": "markPrice@1s",
                    }[channel]
                    if value["stream"] != parts[0] + "@" + suffix:
                        raise ValueError("tardis_native_frequency_unsupported")
                    native = value["data"]
                    if "generated" in value and type(value["generated"]) is not bool:
                        raise ValueError("tardis_generated_flag_invalid")
                    if channel in ("depth", "aggTrade"):
                        validate_market_payload(
                            FUTURES_MARKET_TYPE,
                            "depth" if channel == "depth" else "agg_trades",
                            parts[0].upper(),
                            native,
                            source_schema=SCHEMA_ID,
                            observed_time_ns=observed,
                        )
                    elif channel == "depthSnapshot":
                        if (
                            type(native.get("lastUpdateId")) is not int
                            or native["lastUpdateId"] < 0
                            or type(native.get("bids")) is not list
                            or type(native.get("asks")) is not list
                        ):
                            raise ValueError("tardis_snapshot_invalid")
                    elif (
                        native.get("s") != parts[0].upper()
                        or native.get("e") != "markPriceUpdate"
                    ):
                        raise ValueError("tardis_mark_price_invalid")
                    yield number, timestamp, value
                    value = line = payload = native = None
                except BaseException as error:
                    clear_exception_frames(error)
                    raise
                finally:
                    line = value = payload = native = parts = timestamp = None


class _ObservedStreams:
    """One authority for native continuity and vendor observation identities."""

    def __init__(self):
        self.active = {}
        self.previous = {}
        self.epoch = 0

    def observe(self, offset, number, timestamp, value):
        closed = []
        if value is None:
            closed.extend(self.active.values())
            self.active.clear()
            self.epoch += 1
            return None, closed, {"reason": "vendor_disconnect"}
        symbol, channel, *_ = value["stream"].split("@")
        if channel == "markPrice":
            return None, closed, None
        kind = "agg_trades" if channel == "aggTrade" else "depth"
        key = (symbol, kind)
        gap = None
        native = value["data"]
        if channel != "depthSnapshot":
            current = native["a" if channel == "aggTrade" else "u"]
            previous = self.previous.get(key)
            if previous is not None:
                identifier, epoch = previous
                if current <= identifier:
                    raise ValueError("tardis_native_sequence_non_monotonic")
                if (kind == "depth" and native["pu"] != identifier) or (
                    kind == "agg_trades" and current != identifier + 1
                ):
                    gap = {
                        "reason": "native_sequence_gap",
                        "stream": value["stream"],
                        "previous_id": identifier,
                        "current_id": current,
                        "after_disconnect": epoch != self.epoch,
                    }
                    if key in self.active:
                        closed.append(self.active.pop(key))
            self.previous[key] = current, self.epoch
        observed = strict_iso_timestamp(
            timestamp, error="tardis_observed_time_invalid"
        ).value
        interval = self.active.get(key)
        if interval is None:
            identity = f"tardis:{offset}:{number}:{symbol}:{kind}"
            interval = {
                "observation_id": identity,
                "symbol": symbol.upper(),
                "data_kind": kind,
                "start_ns": observed,
                "end_ns": observed + 1,
            }
            self.active[key] = interval
        elif channel != "depthSnapshot":
            interval["end_ns"] = observed + 1
        return interval["observation_id"], closed, gap

    def finish(self):
        rows = tuple(self.active.values())
        self.active.clear()
        return rows


def _publish_slices(
    output_root,
    date,
    open_slice,
    *,
    offset_minutes,
    slice_count,
    byte_budget,
    resource_policy,
):
    _date(date)
    if (
        type(offset_minutes) is not int
        or offset_minutes < 0
        or offset_minutes % SLICE_MINUTES
        or type(slice_count) is not int
        or slice_count <= 0
    ):
        raise ValueError("tardis_slice_range_invalid")
    if type(byte_budget) is not int or byte_budget <= 0:
        raise ValueError("tardis_download_budget_invalid")

    def write(root):
        require_free_disk_headroom(root, reserve_bytes=byte_budget)
        members = []
        remaining = byte_budget
        tracker = _ObservedStreams()
        decimal_places = {}
        snapshot_limits = {}
        gap_count = interval_count = 0
        gap_path = root / "gaps.jsonl.gz"
        interval_path = root / "observation_intervals.jsonl.gz"

        def write_row(handle, row):
            require_free_disk_headroom(root, reserve_bytes=4096)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        with (
            runtime_resource_policy_scope(resource_policy) as policy,
            policy.lease("tardis.archive_metadata", 0) as metadata_lease,
            ExitStack() as streams,
        ):
            gaps = streams.enter_context(
                handle_preserving_primary(gzip.open(gap_path, "wt", encoding="utf-8"))
            )
            intervals = streams.enter_context(
                handle_preserving_primary(
                    gzip.open(interval_path, "wt", encoding="utf-8")
                )
            )
            for offset in range(
                offset_minutes,
                offset_minutes + slice_count * SLICE_MINUTES,
                SLICE_MINUTES,
            ):
                target = root / _filename(date, offset)
                digest = hashlib.sha256()
                size = 0
                with policy.lease("tardis.raw_copy", 2 * _COPY_BYTES):
                    with open_slice(offset) as response, target.open("xb") as output:
                        while block := response.read(min(_COPY_BYTES, remaining + 1)):
                            if len(block) > remaining:
                                raise ValueError("tardis_download_budget_exceeded")
                            require_free_disk_headroom(root, reserve_bytes=len(block))
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                            remaining -= len(block)
                        block = None
                        output.flush()
                        os.fsync(output.fileno())
                counts = {}
                disconnects = rows = 0
                first = last = None
                records = iter_native_slice(
                    target, date=date, offset_minutes=offset, resource_policy=policy
                )
                try:
                    for number, timestamp, value in records:
                        rows = number
                        _, closed, gap = tracker.observe(
                            offset, number, timestamp, value
                        )
                        for interval in closed:
                            write_row(intervals, interval)
                            interval_count += 1
                        if value is None:
                            disconnects += 1
                        else:
                            stream = value["stream"]
                            symbol, channel, *_ = stream.split("@")
                            symbol = symbol.upper()
                            native = value["data"]
                            if channel in ("depth", "depthSnapshot"):
                                names = (
                                    ("b", "a")
                                    if channel == "depth"
                                    else ("bids", "asks")
                                )
                                for name in names:
                                    for level in native[name]:
                                        if type(level) is not list or len(level) != 2:
                                            raise ValueError(
                                                "tardis_price_level_invalid"
                                            )
                                        precision = price_decimal_places(level[0])
                                        decimal_places[symbol] = max(
                                            decimal_places.get(symbol, 0), precision
                                        )
                                if channel == "depthSnapshot":
                                    snapshot_limits[symbol] = max(
                                        snapshot_limits.get(symbol, 0),
                                        *(len(native[name]) for name in names),
                                    )
                            elif channel == "aggTrade":
                                decimal_places[symbol] = max(
                                    decimal_places.get(symbol, 0),
                                    price_decimal_places(native["p"]),
                                )
                            counts[stream] = counts.get(stream, 0) + 1
                            if first is None:
                                first = timestamp
                            last = timestamp
                        if gap is not None:
                            gap.update(
                                offset_minutes=offset,
                                line=number,
                                observed_utc=timestamp,
                            )
                            write_row(gaps, gap)
                            gap_count += 1
                        value = native = gap = interval = closed = None
                except BaseException as error:
                    clear_exception_frames(error)
                    raise
                finally:
                    value = native = gap = interval = closed = level = None
                    records.close()
                member = {
                    "path": target.name,
                    "offset_minutes": offset,
                    "slice_minutes": SLICE_MINUTES,
                    "compressed_bytes": size,
                    "sha256": digest.hexdigest(),
                    "line_count": rows,
                    "record_counts": counts,
                    "disconnect_markers": disconnects,
                    "first_observed_utc": first,
                    "last_observed_utc": last,
                    "request_url": _request_url(date, offset),
                }
                metadata_lease.grow(
                    json_record_working_set_bytes(len(json.dumps(member).encode()))
                )
                members.append(member)
                member = None
            for interval in tracker.finish():
                write_row(intervals, interval)
                interval_count += 1

            streams.close()

            def descriptor(path, count):
                with (
                    policy.lease("tardis.metadata_hash", 4 * _COPY_BYTES),
                    path.open("rb") as handle,
                ):
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                return {
                    "path": path.name,
                    "sha256": digest,
                    "compressed_bytes": path.stat().st_size,
                    "record_count": count,
                }

            manifest = {
                "schema_id": SCHEMA_ID,
                "source_id": "tardis",
                "venue": "binance",
                "exchange": EXCHANGE,
                "date": date,
                "symbols": list(SYMBOLS),
                "requested_channels": list(CHANNELS),
                "observation_clock": "vendor_utc",
                "ordering": "slice_offset_then_native_line",
                "files": members,
                "gaps": descriptor(gap_path, gap_count),
                "observation_intervals": descriptor(interval_path, interval_count),
                "price_units": {
                    symbol: validate_price_unit(str(Decimal((0, (1,), -places))))
                    for symbol, places in decimal_places.items()
                },
                "snapshot_limits": snapshot_limits,
            }
            atomic_write_json(root / "manifest.json", manifest)
            return manifest

    return atomic_write_directory(Path(output_root), write)


def import_slices(
    output_root: Path,
    *,
    date: str,
    slices: Sequence[Path],
    offset_minutes: int = 0,
    byte_budget: int,
    resource_policy=None,
):
    """Publish consecutive native slices of any duration without rewriting bytes."""

    def open_slice(offset):
        return _verified_file(Path(slices[(offset - offset_minutes) // SLICE_MINUTES]))

    return _publish_slices(
        output_root,
        date,
        open_slice,
        offset_minutes=offset_minutes,
        slice_count=len(slices),
        byte_budget=byte_budget,
        resource_policy=resource_policy,
    )


def download_slices(
    output_root: Path,
    *,
    date: str,
    slice_count: int,
    offset_minutes: int = 0,
    byte_budget: int,
    api_key: str | None = None,
    resource_policy=None,
):
    """Download an entitled slice range without a calendar-day duration limit."""
    _date(date)
    if api_key is not None and (
        type(api_key) is not str
        or not api_key.strip()
        or "\n" in api_key
        or "\r" in api_key
    ):
        raise ValueError("tardis_api_key_invalid")

    @contextmanager
    def open_slice(offset) -> Iterator[object]:
        headers = {"Accept-Encoding": "gzip"}
        if api_key is not None:
            headers["Authorization"] = "Bearer " + api_key
        request = urllib.request.Request(_request_url(date, offset), headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise PermissionError("tardis_access_not_authorized") from None
            raise ValueError(f"tardis_http_status_{error.code}") from None
        with response:
            if (
                response.status != 200
                or response.headers.get("Content-Encoding") != "gzip"
            ):
                raise ValueError("tardis_http_response_invalid")
            yield response

    return _publish_slices(
        output_root,
        date,
        open_slice,
        offset_minutes=offset_minutes,
        slice_count=slice_count,
        byte_budget=byte_budget,
        resource_policy=resource_policy,
    )
