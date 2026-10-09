"""Immutable vendor observations decoded into the shared Binance venue engine."""

from __future__ import annotations

import datetime as dt
import pickle
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from pathlib import Path
from tempfile import TemporaryFile

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.order_book import validate_price_unit
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.time import time_window_timestamp_interval
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import (
    ObservationCoverage,
    ObservationCoverageIndex,
    _capture_raw_file_identity,
    _raw_path_session,
    _sha256_file,
    native_trade_identities,
)
from quant_research_platform.sources.binance.validation import strict_iso_timestamp
from quant_research_platform.sources.tardis.raw import (
    EXCHANGE,
    SCHEMA_ID,
    SLICE_MINUTES,
    STATE_CHANNELS,
    _date,
    _filename,
    _ObservedStreams,
    _request_scope,
    _request_url,
    _stream_suffix,
    iter_native_slice,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

_MANIFEST_KEYS = {
    "schema_id",
    "source_id",
    "venue",
    "exchange",
    "date",
    "symbols",
    "requested_channels",
    "observation_clock",
    "ordering",
    "files",
    "gaps",
    "observation_intervals",
    "price_units",
    "snapshot_limits",
}
_FILE_KEYS = {
    "path",
    "offset_minutes",
    "slice_minutes",
    "compressed_bytes",
    "sha256",
    "line_count",
    "record_counts",
    "disconnect_markers",
    "first_observed_utc",
    "last_observed_utc",
    "request_url",
}


def _descriptor(value, path):
    if (
        type(value) is not dict
        or set(value) != {"path", "sha256", "compressed_bytes", "record_count"}
        or value["path"] != path
        or type(value["record_count"]) is not int
        or value["record_count"] < 0
    ):
        raise ValueError("tardis_manifest_invalid")
    _file_identity_fields(value)


def _file_identity_fields(value):
    if (
        type(value["compressed_bytes"]) is not int
        or value["compressed_bytes"] <= 0
        or type(value["sha256"]) is not str
        or len(value["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["sha256"])
    ):
        raise ValueError("tardis_manifest_invalid")


def _validate_manifest(value):
    if (
        type(value) is not dict
        or set(value) != _MANIFEST_KEYS
        or value["schema_id"] != SCHEMA_ID
        or value["source_id"] != "tardis"
        or value["venue"] != "binance"
        or type(value["symbols"]) is not list
        or type(value["requested_channels"]) is not list
        or value["observation_clock"] != "vendor_utc"
        or value["ordering"] != "slice_offset_then_native_line"
    ):
        raise ValueError("tardis_manifest_invalid")
    _date(value["date"])
    _request_scope(
        value["exchange"], tuple(value["symbols"]), tuple(value["requested_channels"])
    )
    if type(value["files"]) is not list or not value["files"]:
        raise ValueError("tardis_manifest_invalid")
    first = value["files"][0]
    if (
        type(first) is not dict
        or type(first.get("offset_minutes")) is not int
        or first["offset_minutes"] < 0
        or first["offset_minutes"] % SLICE_MINUTES
    ):
        raise ValueError("tardis_manifest_invalid")
    for number, item in enumerate(value["files"]):
        offset = first["offset_minutes"] + number * SLICE_MINUTES
        member_day = _date(value["date"]) + dt.timedelta(days=offset // 1440)
        streams = {
            f"{symbol}@{_stream_suffix(channel, value['exchange'], member_day)}"
            for symbol in value["symbols"]
            for channel in value["requested_channels"]
        }
        if (
            type(item) is not dict
            or set(item) != _FILE_KEYS
            or type(item["offset_minutes"]) is not int
            or item["offset_minutes"] != offset
            or item["slice_minutes"] != SLICE_MINUTES
            or item["path"]
            != _filename(value["date"], offset, exchange=value["exchange"])
            or item["request_url"]
            != _request_url(
                value["date"],
                offset,
                exchange=value["exchange"],
                symbols=tuple(value["symbols"]),
                channels=tuple(value["requested_channels"]),
            )
            or type(item["record_counts"]) is not dict
            or not set(item["record_counts"]) <= streams
            or any(type(n) is not int or n <= 0 for n in item["record_counts"].values())
            or type(item["disconnect_markers"]) is not int
            or item["disconnect_markers"] < 0
            or type(item["line_count"]) is not int
            or item["line_count"]
            != sum(item["record_counts"].values()) + item["disconnect_markers"]
        ):
            raise ValueError("tardis_manifest_invalid")
        _file_identity_fields(item)
        if item["record_counts"]:
            if not all(
                type(item[key]) is str
                for key in ("first_observed_utc", "last_observed_utc")
            ):
                raise ValueError("tardis_manifest_invalid")
        elif (
            item["first_observed_utc"] is not None
            or item["last_observed_utc"] is not None
        ):
            raise ValueError("tardis_manifest_invalid")
    for key in ("price_units", "snapshot_limits"):
        if type(value[key]) is not dict or not set(value[key]) <= {
            symbol.upper() for symbol in value["symbols"]
        }:
            raise ValueError("tardis_manifest_invalid")
    for unit in value["price_units"].values():
        validate_price_unit(unit)
    if any(type(n) is not int or n < 0 for n in value["snapshot_limits"].values()):
        raise ValueError("tardis_manifest_invalid")
    _descriptor(value["gaps"], "gaps.jsonl.gz")
    _descriptor(value["observation_intervals"], "observation_intervals.jsonl.gz")


def _archive_span(manifest):
    start = dt.datetime.combine(_date(manifest["date"]), dt.time(), dt.UTC)
    return (
        start + dt.timedelta(minutes=manifest["files"][0]["offset_minutes"]),
        start
        + dt.timedelta(minutes=manifest["files"][-1]["offset_minutes"] + SLICE_MINUTES),
    )


def _archive_intersects(manifest, time_window):
    if time_window is None:
        return True
    lower, upper = _archive_span(manifest)
    requested_start, requested_end, end_closed = time_window_timestamp_interval(
        time_window
    )
    start, end = (
        value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value
        for value in (requested_start, requested_end)
    )
    return start < upper and (end >= lower if end_closed else end > lower)


def _decode_interval(value, *, markets, symbols, archive_identity):
    if (
        type(value) is not dict
        or set(value) != {"observation_id", "symbol", "data_kind", "start_ns", "end_ns"}
        or type(value["observation_id"]) is not str
        or not value["observation_id"].startswith("tardis:")
        or value["symbol"] not in symbols
        or value["data_kind"] not in {"depth", "agg_trades"}
        or any(type(value[key]) is not int for key in ("start_ns", "end_ns"))
        or not 0 <= value["start_ns"] < value["end_ns"]
        or markets not in ({FUTURES_MARKET_TYPE}, {SPOT_MARKET_TYPE})
    ):
        raise ValueError("tardis_observation_interval_invalid")
    identity = archive_identity + ":" + value["observation_id"]
    kind = value["data_kind"]
    return (
        identity,
        next(iter(markets)),
        kind,
        value["symbol"].lower() + "@" + kind,
        identity if kind == "depth" else None,
        value["start_ns"],
        value["end_ns"],
    )


@dataclass
class TardisArchive:
    root: Path
    manifest: dict
    source_identity: dict
    _coverage_index: ObservationCoverageIndex
    _replay_resources: ExitStack
    _records_by_stream: dict = field(default_factory=dict)
    _native_records: object = field(init=False)
    _sequence_offset: int = 0

    def __post_init__(self):
        self._native_records = self._replay_resources.enter_context(
            closing_iterator(self._iter_canonical_records())
        )

    def _iter_canonical_records(self):
        """Decode and validate native continuity once for this archive session."""
        market_type = (
            FUTURES_MARKET_TYPE
            if self.manifest["exchange"] == EXCHANGE
            else SPOT_MARKET_TYPE
        )
        prefix = 0
        observe_trade = native_trade_identities()
        tracker = _ObservedStreams(
            self.manifest["exchange"], observe_trade=observe_trade
        )
        state_epoch = 0
        for member in self.manifest["files"]:
            number = 0
            with closing_iterator(
                iter_native_slice(
                    self.root / member["path"],
                    date=self.manifest["date"],
                    offset_minutes=member["offset_minutes"],
                    expected_sha256=member["sha256"],
                    exchange=self.manifest["exchange"],
                    symbols=tuple(self.manifest["symbols"]),
                    channels=tuple(self.manifest["requested_channels"]),
                )
            ) as records:
                for number, timestamp, value in records:
                    identity, closed, _gap = tracker.observe(
                        member["offset_minutes"], number, timestamp, value
                    )
                    for interval in closed:
                        self._check_interval(interval)
                    if value is None:
                        continue
                    if tracker.epoch != state_epoch:
                        state_epoch = tracker.epoch
                        for affected in self.manifest["symbols"]:
                            _yield_value = (
                                (affected.upper(), "state"),
                                {
                                    "receive_time": timestamp,
                                    "channel": None,
                                    "payload": None,
                                    "epoch": self._sequence_offset + state_epoch,
                                },
                            )
                            try:
                                yield _yield_value
                            except GeneratorExit:
                                return
                    raw_symbol, channel, *_ = value["stream"].split("@")
                    symbol = raw_symbol.upper()
                    if channel in STATE_CHANNELS:
                        _yield_value = (
                            (symbol, "state"),
                            {
                                "receive_time": timestamp,
                                "channel": channel,
                                "payload": value["data"],
                                "epoch": self._sequence_offset + tracker.epoch,
                            },
                        )
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
                        continue
                    if identity is None:
                        continue
                    depth = channel != "aggTrade"
                    payload = value["data"]
                    if market_type == SPOT_MARKET_TYPE:
                        payload = dict(payload)
                        for clock in ("E", "T"):
                            if clock in payload and payload[clock] > 10**15:
                                payload[clock] //= 1000
                    _yield_value = (
                        (symbol, "depth" if depth else "agg_trades"),
                        {
                            "source_sequence": self._sequence_offset + prefix + number,
                            "market_type": market_type,
                            "data_kind": "depth_snapshot"
                            if channel == "depthSnapshot"
                            else "depth"
                            if depth
                            else "agg_trades",
                            "payload": payload,
                            "receive_time": timestamp,
                            "receive_monotonic_ns": None,
                            "observation_id": self.source_identity["manifest_hash"]
                            + ":"
                            + identity,
                            "segment_id": self.source_identity["manifest_hash"]
                            + ":"
                            + identity
                            if depth
                            else None,
                            "price_unit": self.manifest["price_units"][symbol],
                        },
                    )
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return
            if number != member["line_count"]:
                raise ValueError("tardis_native_line_count_mismatch")
            prefix += number
        for interval in tracker.finish():
            self._check_interval(interval)

    def _stream_records(self, symbol, data_kind, end_ns):
        """Independent cursors over one lazily advanced native scan."""
        requested = symbol, data_kind
        position = 0
        while True:
            buffered = self._records_by_stream.get(requested)
            if buffered is not None and position < buffered.seek(0, 2):
                buffered.seek(position)
                start = position
                record = pickle.load(buffered)
            else:
                item = next(self._native_records, None)
                if item is None:
                    return
                key, record = item
                buffered = self._records_by_stream.get(key)
                if buffered is None:
                    buffered = self._replay_resources.enter_context(
                        closing_iterator(TemporaryFile())
                    )
                    self._records_by_stream[key] = buffered
                start = buffered.seek(0, 2)
                pickle.dump(record, buffered, protocol=pickle.HIGHEST_PROTOCOL)
                if key != requested:
                    continue
            position = buffered.tell()
            if (
                end_ns is not None
                and strict_iso_timestamp(
                    record["receive_time"], error="tardis_observed_time_invalid"
                ).value
                > end_ns
            ):
                return
            yield start, record

    def funding_root(self, symbol):
        return self.root / "funding" / symbol / "observations"

    def hft_source_info(self):
        return {
            "raw_record_count": sum(
                sum(item["record_counts"].values()) for item in self.manifest["files"]
            ),
            "source_file_count": len(self.manifest["files"]),
        }

    def market_metadata(self, market_type, symbol):
        expected = (
            FUTURES_MARKET_TYPE
            if self.manifest["exchange"] == EXCHANGE
            else SPOT_MARKET_TYPE
        )
        if market_type != expected or symbol.lower() not in self.manifest["symbols"]:
            raise ValueError("tardis_market_identity_invalid")
        if symbol not in self.manifest["price_units"]:
            raise PortfolioTargetBlocked("tardis_price_observations_missing")
        if symbol not in self.manifest["snapshot_limits"]:
            raise PortfolioTargetBlocked("tardis_depth_snapshot_missing")
        return {
            "price_unit": self.manifest["price_units"][symbol],
            "snapshot_limit": self.manifest["snapshot_limits"][symbol],
        }

    def observation_coverage(self, *, market_type, data_kind, symbol):
        return ObservationCoverage(self._coverage_index, market_type, data_kind, symbol)

    def _check_interval(self, row):
        if self._coverage_index.bounds(
            self.source_identity["manifest_hash"] + ":" + row["observation_id"],
            symbol=row["symbol"],
            market_type=FUTURES_MARKET_TYPE
            if self.manifest["exchange"] == EXCHANGE
            else SPOT_MARKET_TYPE,
            data_kind=row["data_kind"],
        ) != (row["start_ns"], row["end_ns"]):
            raise ValueError("tardis_observation_interval_mismatch")

    def iter_state_observations(self, *, symbol):
        """Yield native states with observed clocks and disconnect generations."""
        if (
            self.manifest["exchange"] != EXCHANGE
            or symbol.lower() not in self.manifest["symbols"]
            or not set(STATE_CHANNELS) <= set(self.manifest["requested_channels"])
        ):
            raise ValueError("tardis_derivative_scope_invalid")
        with closing_iterator(self._stream_records(symbol, "state", None)) as records:
            for _start, record in records:
                observed = strict_iso_timestamp(
                    record["receive_time"], error="tardis_observed_time_invalid"
                ).value
                _yield_value = (
                    observed,
                    record["channel"],
                    record["payload"],
                    record["epoch"],
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return

    def iter_market_records(self, *, symbol, market_type, data_kinds, end_ns=None):
        self.market_metadata(market_type, symbol)
        depth = tuple(data_kinds) == ("depth_snapshot", "depth")
        if not depth and tuple(data_kinds) != ("agg_trades",):
            raise ValueError("tardis_market_record_kind_invalid")
        active_observation = None
        snapshot_ready = False
        with closing_iterator(
            self._stream_records(symbol, "depth" if depth else "agg_trades", end_ns)
        ) as records:
            if not depth:
                for _start, _yield_value in records:
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return
                return
            prefix_start = None
            for start, record in records:
                identity = record["observation_id"]
                if identity != active_observation:
                    prefix_start = None
                    active_observation = identity
                    snapshot_ready = False
                if record["data_kind"] == "depth_snapshot":
                    try:
                        yield record
                    except GeneratorExit:
                        return
                    prefix_position = prefix_start
                    while prefix_position is not None and prefix_position < start:
                        buffered = self._records_by_stream[(symbol, "depth")]
                        buffered.seek(prefix_position)
                        bootstrap = pickle.load(buffered)
                        prefix_position = buffered.tell()
                        bootstrap["data_kind"] = "depth_bootstrap"
                        try:
                            yield bootstrap
                        except GeneratorExit:
                            return
                    prefix_start = None
                    snapshot_ready = True
                elif snapshot_ready:
                    try:
                        yield record
                    except GeneratorExit:
                        return
                elif prefix_start is None:
                    prefix_start = start


@contextmanager
def _open_tardis_archive(root, *, expected_manifest_hash):
    root = Path(root)
    with _raw_path_session(root) as verify_root:
        manifest_path = root / "manifest.json"
        identity = _capture_raw_file_identity(
            manifest_path, missing_is_file_not_found=False
        )
        _raw = manifest = members = identities = _after_raw = after = run = None
        _raw, manifest = read_qrp_json_file(manifest_path, expected_identity=identity)
        _validate_manifest(manifest)
        manifest_hash = canonical_json_hash(manifest)
        if manifest_hash != expected_manifest_hash:
            raise ValueError("tardis_archive_selection_identity_changed")
        _raw = None
        members = (
            *manifest["files"],
            manifest["gaps"],
            manifest["observation_intervals"],
        )
        identities = tuple(
            _capture_raw_file_identity(
                root / item["path"], missing_is_file_not_found=False
            )
            for item in members
        )

        def verify_files():
            verify_root()
            for member, expected in zip(members, identities, strict=True):
                if (
                    expected.size != member["compressed_bytes"]
                    or _sha256_file(root / member["path"], expected) != member["sha256"]
                ):
                    raise ValueError("tardis_raw_identity_changed")

        verify_files()
        index = ObservationCoverageIndex(
            root,
            manifest["observation_intervals"],
            markets={
                FUTURES_MARKET_TYPE
                if manifest["exchange"] == EXCHANGE
                else SPOT_MARKET_TYPE
            },
            decode_interval=partial(
                _decode_interval,
                archive_identity=manifest_hash,
                symbols={symbol.upper() for symbol in manifest["symbols"]},
            ),
        )
        replay_resources = ExitStack()
        run = TardisArchive(
            root,
            manifest,
            {
                "schema_id": SCHEMA_ID,
                "artifact_id": "native_archive_" + manifest_hash[:24],
                "manifest_hash": manifest_hash,
            },
            index,
            replay_resources,
        )
        with replay_resources:
            primary = None
            try:
                yield run
            except BaseException as body_error:
                primary = body_error
                raise
            finally:
                try:
                    verify_files()
                    _after_raw, after = read_qrp_json_file(
                        manifest_path, expected_identity=identity
                    )
                    if canonical_json_hash(after) != manifest_hash:
                        raise ValueError("tardis_manifest_identity_changed")
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Tardis source verification also failed: "
                        + type(error).__name__
                    )


class _ArchiveCoverage:
    def __init__(self, readers):
        self.readers = readers

    def get(self, observation_id, default=None):
        for reader in self.readers:
            value = reader.get(observation_id)
            if value is not None:
                return value
        return default

    def __getitem__(self, observation_id):
        value = self.get(observation_id)
        if value is None:
            raise KeyError(observation_id)
        return value

    def covering(self, start_ns, end_ns):
        result = []
        for reader in self.readers:
            result.extend(reader.covering(start_ns, end_ns))
            if len(result) >= 2:
                break
        return tuple(result[:2])

    def intervals(self):
        for reader in self.readers:
            with closing_iterator(reader.intervals()) as intervals:
                for _yield_value in intervals:
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return


@dataclass(frozen=True)
class _TardisArchives:
    archives: tuple[TardisArchive, ...]
    source_identity: dict

    @property
    def _coverage_index(self):
        return self

    def bounds(self, observation_id, **scope):
        for archive in self.archives:
            value = archive._coverage_index.bounds(observation_id, **scope)
            if value is not None:
                return value
        return None

    def hft_source_info(self):
        values = [archive.hft_source_info() for archive in self.archives]
        return {
            key: sum(value[key] for value in values)
            for key in ("raw_record_count", "source_file_count")
        }

    def market_metadata(self, market_type, symbol):
        values = [
            archive.market_metadata(market_type, symbol) for archive in self.archives
        ]
        return {
            "price_unit": min((value["price_unit"] for value in values), key=Fraction),
            "snapshot_limit": max(value["snapshot_limit"] for value in values),
        }

    def observation_coverage(self, **scope):
        return _ArchiveCoverage(
            tuple(archive.observation_coverage(**scope) for archive in self.archives)
        )

    def iter_market_records(self, *, symbol, market_type, data_kinds, end_ns=None):
        price_unit = self.market_metadata(market_type, symbol)["price_unit"]
        for archive in self.archives:
            with closing_iterator(
                archive.iter_market_records(
                    symbol=symbol,
                    market_type=market_type,
                    data_kinds=data_kinds,
                    end_ns=end_ns,
                )
            ) as records:
                for record in records:
                    record["price_unit"] = price_unit
                    try:
                        yield record
                    except GeneratorExit:
                        return

    def iter_state_observations(self, *, symbol):
        for archive in self.archives:
            with closing_iterator(
                archive.iter_state_observations(symbol=symbol)
            ) as records:
                for _yield_value in records:
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return

    def funding_roots(self, symbol, *, time_window):
        return tuple(
            archive.funding_root(symbol)
            for archive in self.archives
            if _archive_intersects(archive.manifest, time_window)
        )


@contextmanager
def open_tardis_archive(root, *, archive_paths=None, time_window=None):
    """Read explicitly declared daily archives without joining their continuity."""
    root = Path(root)
    paths = (
        (root,)
        if archive_paths is None
        else tuple(root / item for item in archive_paths)
    )
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("tardis_archive_set_invalid")
    with ExitStack() as readers:
        selected = []
        for path in paths:
            identity = _capture_raw_file_identity(path / "manifest.json")
            _raw = manifest = None
            _raw, manifest = read_qrp_json_file(
                path / "manifest.json", expected_identity=identity
            )
            _validate_manifest(manifest)
            if _archive_intersects(manifest, time_window):
                selected.append((path, canonical_json_hash(manifest)))
        if not selected:
            raise PortfolioTargetBlocked("tardis_archive_window_empty")
        archives = tuple(
            sorted(
                (
                    readers.enter_context(
                        _open_tardis_archive(path, expected_manifest_hash=identity)
                    )
                    for path, identity in selected
                ),
                key=lambda archive: _archive_span(archive.manifest),
            )
        )
        spans = tuple(_archive_span(archive.manifest) for archive in archives)
        if any(
            current[0] < previous[1]
            for previous, current in zip(spans, spans[1:], strict=False)
        ):
            raise ValueError("tardis_archive_spans_overlap")
        scope = archives[0].manifest
        if any(
            archive.manifest[key] != scope[key]
            for archive in archives[1:]
            for key in ("exchange", "symbols", "requested_channels")
        ):
            raise ValueError("tardis_archive_scope_mismatch")
        offset = 0
        for archive in archives:
            archive._sequence_offset = offset
            offset += 1 + sum(
                member["line_count"] for member in archive.manifest["files"]
            )
        yield _TardisArchives(
            archives,
            {
                "schema_id": SCHEMA_ID,
                "archives": [
                    {"date": archive.manifest["date"], **archive.source_identity}
                    for archive in archives
                ],
            },
        )


def native_source_identity(
    root,
    _target_id,
    *,
    exchange,
    symbols,
    channels,
    archive_paths=None,
    time_window=None,
):
    """Bind every exact approved native information-source archive."""
    try:
        with open_tardis_archive(
            root, archive_paths=archive_paths, time_window=time_window
        ) as source:
            for archive in source.archives:
                if (
                    archive.manifest["exchange"] != exchange
                    or set(archive.manifest["symbols"]) != set(symbols)
                    or not set(channels) <= set(archive.manifest["requested_channels"])
                ):
                    raise ValueError("tardis_auxiliary_scope_invalid")
            return canonical_json_hash(source.source_identity)
    except FileNotFoundError:
        return None
