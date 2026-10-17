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
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import (
    price_decimal_places,
    validate_price_unit,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import native_trade_identities
from quant_research_platform.sources.binance.validation import (
    _visible_trade_quantity,
    strict_boolean,
    strict_iso_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
    strict_positive_float,
    validate_market_payload,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import (
    _DigestingReader,
    filesystem_gzip_open,
    filesystem_stat,
    open_filesystem_file,
    require_unaliased_path,
)

SCHEMA_ID = "qrp-tardis-native-archive"
EXCHANGE = "binance-futures"
CHANNELS = ("depth", "depthSnapshot", "aggTrade", "markPrice")
STATE_CHANNELS = (
    "openInterest",
    "forceOrder",
    "premiumIndex",
    "topLongShortPositionRatio",
    "topLongShortAccountRatio",
    "globalLongShortAccountRatio",
    "takerlongshortRatio",
)
SYMBOLS = ("btcusdt", "ethusdt")
SLICE_MINUTES = 10
_COPY_BYTES = 64 * 1024


def _request_scope(exchange, symbols, channels):
    allowed_symbols = (
        {"btcusdt", "ethusdt", "solusdt"}
        if exchange == EXCHANGE
        else {"btcusdt", "ethusdt"}
    )
    allowed_channels = (
        {*CHANNELS, *STATE_CHANNELS}
        if exchange == EXCHANGE
        else {"depth", "depthSnapshot", "aggTrade"}
    )
    if (
        type(exchange) is not str
        or exchange not in {EXCHANGE, "binance"}
        or type(symbols) is not tuple
        or not symbols
        or any(type(item) is not str for item in symbols)
        or len(set(symbols)) != len(symbols)
        or not set(symbols) <= allowed_symbols
        or type(channels) is not tuple
        or not channels
        or any(type(item) is not str for item in channels)
        or len(set(channels)) != len(channels)
        or not set(channels) <= allowed_channels
    ):
        raise ValueError("tardis_request_scope_invalid")


def _stream_suffix(channel, exchange, day):
    if channel == "depth":
        if exchange == EXCHANGE:
            # Tardis documents the date, but not the intraday transition time.
            if day == dt.date(2020, 1, 7):
                raise ValueError("tardis_depth_frequency_transition_unproven")
            return "depth@100ms" if day < dt.date(2020, 1, 7) else "depth@0ms"
        return "depth@100ms"
    if channel == "markPrice":
        if day == dt.date(2020, 2, 13):
            raise ValueError("tardis_mark_price_frequency_transition_unproven")
        return "markPrice" if day < dt.date(2020, 2, 13) else "markPrice@1s"
    return channel


def _validate_state(channel, symbol, native):
    if channel == "forceOrder":
        if native.get("e") != "forceOrder" or type(native.get("o")) is not dict:
            raise ValueError("tardis_force_order_invalid")
        order = native["o"]
        if order.get("s") != symbol:
            raise ValueError("tardis_force_order_invalid")
        if order.get("S") not in {"BUY", "SELL"}:
            order["S"] = None
        strict_non_negative_integer(native.get("E"), error="tardis_state_time_invalid")
        strict_non_negative_integer(order.get("T"), error="tardis_state_time_invalid")
        for field in ("q", "l", "z"):
            _nullable_state_value(order, field, strict_non_negative_float)
        return
    if channel != "takerlongshortRatio" and native.get("symbol") != symbol:
        raise ValueError("tardis_state_symbol_invalid")
    clock = "time" if channel in {"openInterest", "premiumIndex"} else "timestamp"
    strict_non_negative_integer(native.get(clock), error="tardis_state_time_invalid")
    fields = (
        ("openInterest",)
        if channel == "openInterest"
        else ("markPrice", "indexPrice")
        if channel == "premiumIndex"
        else ("buySellRatio", "buyVol", "sellVol")
        if channel == "takerlongshortRatio"
        else ("longShortRatio", "longAccount", "shortAccount")
    )
    for field in fields:
        _nullable_state_value(
            native,
            field,
            strict_positive_float
            if channel == "premiumIndex"
            else strict_non_negative_float,
        )
    if channel == "premiumIndex":
        _nullable_state_value(native, "nextFundingTime", strict_non_negative_integer)
        from quant_research_platform.contracts.validation import strict_finite_float

        for field in ("lastFundingRate", "interestRate"):
            _nullable_state_value(native, field, strict_finite_float)


def _nullable_state_value(payload, field, validate):
    try:
        validate(payload.get(field), error="tardis_state_value_invalid")
    except ValueError:
        payload[field] = None


def _valid_trade_values(exchange, native, observed_ns):
    try:
        strict_boolean(native.get("m"), error="tardis_trade_value_invalid")
        strict_positive_float(native.get("p"), error="tardis_trade_value_invalid")
        _visible_trade_quantity(
            FUTURES_MARKET_TYPE if exchange == EXCHANGE else SPOT_MARKET_TYPE,
            native,
            historical_time=observed_ns // 1_000_000,
            allow_unknown=True,
        )
    except ValueError:
        return False
    return True


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


def _request_url(
    date: str, offset: int, *, exchange=EXCHANGE, symbols=SYMBOLS, channels=CHANNELS
) -> str:
    _request_scope(exchange, symbols, channels)
    date = (_date(date) + dt.timedelta(days=offset // 1440)).isoformat()
    offset %= 1440
    return (
        "https://api.tardis.dev/v1/data-feeds/"
        + exchange
        + "?"
        + urllib.parse.urlencode(
            {
                "from": date,
                "offset": offset,
                "sliceSize": SLICE_MINUTES,
                "filters": json.dumps(
                    [{"channel": item, "symbols": list(symbols)} for item in channels],
                    separators=(",", ":"),
                ),
            }
        )
    )


def _filename(date: str, offset: int, *, exchange=EXCHANGE) -> str:
    date = (_date(date) + dt.timedelta(days=offset // 1440)).isoformat()
    offset %= 1440
    return (
        f"tardis__{exchange}__{date}__minutes-"
        f"{offset:04d}-{offset + SLICE_MINUTES:04d}.txt.gz"
    )


@contextmanager
def _verified_file(path: Path, *, expected_sha256=None):
    """Open native bytes; audit the full digest only when explicitly requested."""
    require_unaliased_path(path, error="tardis_raw_identity_changed")
    try:
        handle = open_filesystem_file(path, "rb")
    except IsADirectoryError:
        raise ValueError("tardis_source_file_invalid") from None
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError("tardis_raw_identity_changed") from error
    with closing_iterator(handle):
        reader = _DigestingReader(handle) if expected_sha256 is not None else handle
        yield reader
        if expected_sha256 is not None:
            reader.drain()
            if reader.hexdigest() != expected_sha256:
                raise ValueError("tardis_raw_identity_changed")


def iter_native_slice(
    path: Path,
    *,
    date: str,
    offset_minutes: int,
    expected_sha256=None,
    exchange=EXCHANGE,
    symbols=SYMBOLS,
    channels=CHANNELS,
):
    """Yield native rows, including disconnect markers, in file order."""
    _request_scope(exchange, symbols, channels)
    day = _date(date)
    start = int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp()) * 10**9
    lower = start + offset_minutes * 60 * 10**9
    upper = lower + SLICE_MINUTES * 60 * 10**9
    previous = lower
    with (
        _verified_file(path, expected_sha256=expected_sha256) as raw,
        closing_iterator(gzip.GzipFile(fileobj=raw, mode="rb")) as stream,
    ):
        number = 0
        while True:
            line = value = payload = native = parts = timestamp = None
            line = stream.readline()
            if not line:
                return
            number += 1
            if not line.endswith(b"\n"):
                raise ValueError("tardis_native_record_invalid")
            if line in (b"\n", b"\r\n"):
                _yield_value = (number, None, None)
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
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
            if len(parts) < 2 or parts[0] not in symbols or parts[1] not in channels:
                raise ValueError("tardis_native_channel_invalid")
            channel = parts[1]
            suffix = _stream_suffix(
                channel, exchange, day + dt.timedelta(days=offset_minutes // 1440)
            )
            if value["stream"] != parts[0] + "@" + suffix:
                raise ValueError("tardis_native_frequency_unsupported")
            native = value["data"]
            if "generated" in value and type(value["generated"]) is not bool:
                raise ValueError("tardis_generated_flag_invalid")
            if channel in ("depth", "aggTrade"):
                validate_market_payload(
                    FUTURES_MARKET_TYPE if exchange == EXCHANGE else SPOT_MARKET_TYPE,
                    "depth" if channel == "depth" else "agg_trades",
                    parts[0].upper(),
                    native,
                    source_schema=SCHEMA_ID,
                    observed_time_ns=observed,
                    validate_values=False,
                )
            elif channel == "depthSnapshot":
                if (
                    type(native.get("lastUpdateId")) is not int
                    or native["lastUpdateId"] < 0
                    or type(native.get("bids")) is not list
                    or (type(native.get("asks")) is not list)
                ):
                    raise ValueError("tardis_snapshot_invalid")
            elif channel in STATE_CHANNELS:
                _validate_state(channel, parts[0].upper(), native)
            elif (
                native.get("s") != parts[0].upper()
                or native.get("e") != "markPriceUpdate"
            ):
                raise ValueError("tardis_mark_price_invalid")
            _yield_value = (number, timestamp, value)
            try:
                yield _yield_value
            except GeneratorExit:
                return
            value = line = payload = native = None


class _ObservedStreams:
    """One authority for native continuity and vendor observation identities."""

    def __init__(self, exchange=EXCHANGE, *, observe_trade):
        self.exchange = exchange
        self.observe_trade = observe_trade
        self.active = {}
        self.previous = {}
        self.epoch = 0

    def observe(self, offset, number, timestamp, value):
        closed = []
        if value is None:
            closed.extend(self.active.values())
            self.active.clear()
            self.previous = {
                key: value for key, value in self.previous.items() if key[1] != "depth"
            }
            self.epoch += 1
            return None, closed, {"reason": "vendor_disconnect"}
        symbol, channel, *_ = value["stream"].split("@")
        if channel not in {"depth", "depthSnapshot", "aggTrade"}:
            return None, closed, None
        kind = "agg_trades" if channel == "aggTrade" else "depth"
        key = (symbol, kind)
        gap = None
        native = value["data"]
        if channel != "depthSnapshot":
            current = native["a" if channel == "aggTrade" else "u"]
            previous = self.previous.get(key)
            if channel == "aggTrade":
                try:
                    accepted, _ = self.observe_trade(
                        FUTURES_MARKET_TYPE
                        if self.exchange == EXCHANGE
                        else SPOT_MARKET_TYPE,
                        symbol.upper(),
                        native,
                    )
                except ValueError as error:
                    error.add_note(
                        f"channel={channel}; slice_offset={offset}; line={number}; "
                        f"previous_id={None if previous is None else previous[0]}; "
                        f"current_id={current}"
                    )
                    raise
                if not accepted:
                    return None, closed, None
            if previous is not None:
                identifier, epoch = previous
                if current > identifier and (
                    kind == "depth"
                    and (
                        native["pu"] != identifier
                        if self.exchange == EXCHANGE
                        else not native["U"] <= identifier + 1 <= current
                    )
                    or (kind == "agg_trades" and current != identifier + 1)
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
            # The shared book engine discards stale depth updates. Retain raw
            # observations without moving the continuity high-water mark backwards.
            if previous is None or current > previous[0]:
                self.previous[key] = current, self.epoch
        observed = strict_iso_timestamp(
            timestamp, error="tardis_observed_time_invalid"
        ).value
        if channel == "aggTrade" and not _valid_trade_values(
            self.exchange, native, observed
        ):
            if key in self.active:
                interval = self.active.pop(key)
                interval["end_ns"] = observed
                if interval["start_ns"] < observed:
                    closed.append(interval)
            return (
                None,
                closed,
                {"reason": "invalid_trade_values", "stream": value["stream"]},
            )
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
    exchange,
    symbols,
    channels,
):
    _request_scope(exchange, symbols, channels)
    _date(date)
    if (
        type(offset_minutes) is not int
        or offset_minutes < 0
        or offset_minutes % SLICE_MINUTES
        or type(slice_count) is not int
        or slice_count <= 0
    ):
        raise ValueError("tardis_slice_range_invalid")
    for offset in range(
        offset_minutes, offset_minutes + slice_count * SLICE_MINUTES, SLICE_MINUTES
    ):
        for channel in channels:
            _stream_suffix(
                channel, exchange, _date(date) + dt.timedelta(days=offset // 1440)
            )

    def write(root):
        members = []
        decimal_places = {}
        snapshot_limits = {}
        gap_count = interval_count = 0
        gap_path = root / "gaps.jsonl.gz"
        interval_path = root / "observation_intervals.jsonl.gz"

        def write_row(handle, row):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        observe_trade = native_trade_identities()
        with ExitStack() as streams:
            tracker = _ObservedStreams(exchange, observe_trade=observe_trade)
            gaps = streams.enter_context(
                closing_iterator(filesystem_gzip_open(gap_path, "wt", encoding="utf-8"))
            )
            intervals = streams.enter_context(
                closing_iterator(
                    filesystem_gzip_open(interval_path, "wt", encoding="utf-8")
                )
            )
            for offset in range(
                offset_minutes,
                offset_minutes + slice_count * SLICE_MINUTES,
                SLICE_MINUTES,
            ):
                target = root / _filename(date, offset, exchange=exchange)
                digest = hashlib.sha256()
                size = 0
                with (
                    open_slice(offset) as response,
                    closing_iterator(open_filesystem_file(target, "xb")) as output,
                ):
                    while block := response.read(_COPY_BYTES):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    block = None
                    output.flush()
                    os.fsync(output.fileno())
                counts = {}
                disconnects = rows = 0
                first = last = None
                records = iter_native_slice(
                    target,
                    date=date,
                    offset_minutes=offset,
                    exchange=exchange,
                    symbols=symbols,
                    channels=channels,
                )
                with closing_iterator(records):
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
                                        try:
                                            precision = price_decimal_places(level[0])
                                        except ValueError:
                                            continue
                                        decimal_places[symbol] = max(
                                            decimal_places.get(symbol, 0), precision
                                        )
                                if channel == "depthSnapshot":
                                    snapshot_limits[symbol] = max(
                                        snapshot_limits.get(symbol, 0),
                                        *(len(native[name]) for name in names),
                                    )
                            elif channel == "aggTrade":
                                try:
                                    precision = price_decimal_places(native.get("p"))
                                except ValueError:
                                    precision = None
                                if precision is not None:
                                    decimal_places[symbol] = max(
                                        decimal_places.get(symbol, 0), precision
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
                    "request_url": _request_url(
                        date,
                        offset,
                        exchange=exchange,
                        symbols=symbols,
                        channels=channels,
                    ),
                }
                members.append(member)
                member = None
            for interval in tracker.finish():
                write_row(intervals, interval)
                interval_count += 1

            streams.close()

            def descriptor(path, count):
                with closing_iterator(open_filesystem_file(path, "rb")) as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                return {
                    "path": path.name,
                    "sha256": digest,
                    "compressed_bytes": filesystem_stat(path).st_size,
                    "record_count": count,
                }

            manifest = {
                "schema_id": SCHEMA_ID,
                "source_id": "tardis",
                "venue": "binance",
                "exchange": exchange,
                "date": date,
                "symbols": list(symbols),
                "requested_channels": list(channels),
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
    exchange=EXCHANGE,
    symbols=SYMBOLS,
    channels=CHANNELS,
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
        exchange=exchange,
        symbols=symbols,
        channels=channels,
    )


def download_slices(
    output_root: Path,
    *,
    date: str,
    slice_count: int,
    offset_minutes: int = 0,
    api_key: str | None = None,
    exchange=EXCHANGE,
    symbols=SYMBOLS,
    channels=CHANNELS,
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
        request = urllib.request.Request(
            _request_url(
                date, offset, exchange=exchange, symbols=symbols, channels=channels
            ),
            headers=headers,
        )
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            with closing_iterator(error):
                if error.code in (401, 403):
                    raise PermissionError("tardis_access_not_authorized") from None
                raise ValueError(f"tardis_http_status_{error.code}") from None
        with closing_iterator(response):
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
        exchange=exchange,
        symbols=symbols,
        channels=channels,
    )
