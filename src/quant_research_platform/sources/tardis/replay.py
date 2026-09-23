"""Immutable vendor observations decoded into the shared Binance venue engine."""

from __future__ import annotations

import json
import struct
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import validate_price_unit
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
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
from quant_research_platform.storage._scratch_ownership import (
    handle_preserving_primary,
    reserved_temporary_file,
)
from quant_research_platform.storage.manifests import read_bounded_qrp_json

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
    streams = {
        f"{symbol}@{_stream_suffix(channel, value['exchange'])}"
        for symbol in value["symbols"]
        for channel in value["requested_channels"]
    }
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


def _decode_interval(value, *, markets, symbols):
    try:
        if (
            type(value) is not dict
            or set(value)
            != {"observation_id", "symbol", "data_kind", "start_ns", "end_ns"}
            or type(value["observation_id"]) is not str
            or not value["observation_id"].startswith("tardis:")
            or value["symbol"] not in symbols
            or value["data_kind"] not in {"depth", "agg_trades"}
            or any(type(value[key]) is not int for key in ("start_ns", "end_ns"))
            or not 0 <= value["start_ns"] < value["end_ns"]
            or markets not in ({FUTURES_MARKET_TYPE}, {SPOT_MARKET_TYPE})
        ):
            raise ValueError("tardis_observation_interval_invalid")
        identity = value["observation_id"]
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
    finally:
        value = None


@dataclass
class TardisArchive:
    root: Path
    manifest: dict
    source_identity: dict
    resource_policy: object
    _coverage_index: ObservationCoverageIndex
    _replay_resources: ExitStack
    _spools: dict = field(default_factory=dict)
    _native_records: object = field(init=False)

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
        with native_trade_identities(self.resource_policy) as observe_trade:
            tracker = _ObservedStreams(
                self.manifest["exchange"], observe_trade=observe_trade
            )
            for member in self.manifest["files"]:
                with closing_iterator(
                    iter_native_slice(
                        self.root / member["path"],
                        date=self.manifest["date"],
                        offset_minutes=member["offset_minutes"],
                        expected_sha256=member["sha256"],
                        resource_policy=self.resource_policy,
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
                        raw_symbol, channel, *_ = value["stream"].split("@")
                        symbol = raw_symbol.upper()
                        if channel in STATE_CHANNELS:
                            yield (
                                (symbol, "state"),
                                {
                                    "receive_time": timestamp,
                                    "channel": channel,
                                    "payload": value["data"],
                                    "epoch": tracker.epoch,
                                },
                            )
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
                        yield (
                            (symbol, "depth" if depth else "agg_trades"),
                            {
                                "source_sequence": prefix + number,
                                "market_type": market_type,
                                "data_kind": "depth_snapshot"
                                if channel == "depthSnapshot"
                                else "depth"
                                if depth
                                else "agg_trades",
                                "payload": payload,
                                "receive_time": timestamp,
                                "receive_monotonic_ns": None,
                                "observation_id": identity,
                                "segment_id": identity if depth else None,
                                "price_unit": self.manifest["price_units"][symbol],
                            },
                        )
                prefix += member["line_count"]
            for interval in tracker.finish():
                self._check_interval(interval)

    def _stream_records(self, symbol, data_kind, end_ns):
        """Independent disk cursors over a single lazily advanced native scan."""
        requested = symbol, data_kind
        position = 0
        while True:
            pair = self._spools.get(requested)
            if pair is not None:
                spool, _reservation = pair
                spool.seek(position)
                header = spool.read(8)
                if header:
                    size = struct.unpack(">Q", header)[0]
                    with self.resource_policy.lease(
                        "tardis.canonical_record",
                        json_record_working_set_bytes(size) + 64 * 1024,
                    ):
                        record = None
                        try:
                            record = parse_qrp_json_object(spool.read(size))
                            position = spool.tell()
                            if (
                                end_ns is not None
                                and strict_iso_timestamp(
                                    record["receive_time"],
                                    error="tardis_observed_time_invalid",
                                ).value
                                > end_ns
                            ):
                                return
                            yield record
                        finally:
                            record = None
                    continue
            item = next(self._native_records, None)
            if item is None:
                return
            key, record = item
            encoded = json.dumps(
                record, separators=(",", ":"), allow_nan=False
            ).encode()
            pair = self._spools.get(key)
            if pair is None:
                pair = self._replay_resources.enter_context(
                    reserved_temporary_file(resource_policy=self.resource_policy)
                )
                self._spools[key] = pair
            target, reservation = pair
            end = target.seek(0, 2)
            reservation.resize(end + len(encoded) + 8)
            target.write(struct.pack(">Q", len(encoded)))
            target.write(encoded)
            item = record = encoded = None

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
        return self._coverage_index._coverage_readers.enter_context(
            handle_preserving_primary(
                ObservationCoverage(
                    self._coverage_index, market_type, data_kind, symbol
                )
            )
        )

    def _check_interval(self, row):
        if self._coverage_index.bounds(
            row["observation_id"],
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
            for record in records:
                observed = strict_iso_timestamp(
                    record["receive_time"], error="tardis_observed_time_invalid"
                ).value
                yield observed, record["channel"], record["payload"], record["epoch"]

    def iter_market_records(self, *, symbol, market_type, data_kinds, end_ns=None):
        self.market_metadata(market_type, symbol)
        depth = tuple(data_kinds) == ("depth_snapshot", "depth")
        if not depth and tuple(data_kinds) != ("agg_trades",):
            raise ValueError("tardis_market_record_kind_invalid")
        active_observation = None
        snapshot_ready = False
        with ExitStack() as resources:
            records = resources.enter_context(
                closing_iterator(
                    self._stream_records(
                        symbol, "depth" if depth else "agg_trades", end_ns
                    )
                )
            )
            if not depth:
                yield from records
                return
            spool, reservation = resources.enter_context(
                reserved_temporary_file(resource_policy=self.resource_policy)
            )
            for record in records:
                identity = record["observation_id"]
                if identity != active_observation:
                    spool.seek(0)
                    spool.truncate(0)
                    reservation.resize(0)
                    active_observation = identity
                    snapshot_ready = False
                if record["data_kind"] == "depth_snapshot":
                    if snapshot_ready:
                        continue
                    yield record
                    spool.seek(0)
                    while header := spool.read(8):
                        size = struct.unpack(">Q", header)[0]
                        with self.resource_policy.lease(
                            "tardis.pending_depth_record",
                            json_record_working_set_bytes(size) + 64 * 1024,
                        ):
                            buffered = None
                            try:
                                buffered = parse_qrp_json_object(spool.read(size))
                                buffered["data_kind"] = "depth_bootstrap"
                                yield buffered
                            finally:
                                buffered = None
                    spool.seek(0)
                    spool.truncate(0)
                    reservation.resize(0)
                    snapshot_ready = True
                elif snapshot_ready:
                    yield record
                else:
                    encoded = json.dumps(
                        record, separators=(",", ":"), allow_nan=False
                    ).encode()
                    reservation.resize(spool.tell() + len(encoded) + 8)
                    spool.write(struct.pack(">Q", len(encoded)))
                    spool.write(encoded)
                    encoded = None
                record = None


@contextmanager
def open_tardis_archive(root, resource_policy=None):
    root = Path(root)
    with (
        runtime_resource_policy_scope(resource_policy) as policy,
        _raw_path_session(root) as verify_root,
    ):
        manifest_path = root / "manifest.json"
        identity = _capture_raw_file_identity(manifest_path)
        with policy.lease(
            "tardis.manifest", json_record_working_set_bytes(identity.size)
        ):
            _raw = manifest = members = identities = _after_raw = after = run = None
            try:
                _raw, manifest = read_bounded_qrp_json(
                    manifest_path, expected_identity=identity, max_bytes=identity.size
                )
                _validate_manifest(manifest)
                manifest_hash = canonical_json_hash(manifest)
                _raw = None
                members = (
                    *manifest["files"],
                    manifest["gaps"],
                    manifest["observation_intervals"],
                )
                identities = tuple(
                    _capture_raw_file_identity(root / item["path"]) for item in members
                )

                def verify_files():
                    verify_root()
                    for member, expected in zip(members, identities, strict=True):
                        if (
                            expected.size != member["compressed_bytes"]
                            or _sha256_file(root / member["path"], expected, policy)
                            != member["sha256"]
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
                    resource_policy=policy,
                    decode_interval=partial(
                        _decode_interval,
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
                    policy,
                    index,
                    replay_resources,
                )
                with handle_preserving_primary(index), replay_resources:
                    try:
                        yield run
                    finally:
                        primary = sys.exception()
                        try:
                            verify_files()
                            _after_raw, after = read_bounded_qrp_json(
                                manifest_path,
                                expected_identity=identity,
                                max_bytes=identity.size,
                            )
                            if canonical_json_hash(after) != manifest_hash:
                                raise ValueError("tardis_manifest_identity_changed")
                            _after_raw = after = None
                        except BaseException as error:
                            if primary is None:
                                raise
                            primary.add_note(
                                "Tardis source verification also failed: "
                                + type(error).__name__
                            )
                        finally:
                            run.manifest = {}
            except BaseException as error:
                clear_exception_frames(error)
                raise
            finally:
                if run is not None:
                    run.manifest = {}
                _raw = manifest = members = identities = _after_raw = after = run = None


def native_source_identity(root, _target_id, *, exchange, symbols, channels):
    """Bind an exact approved native information-source scope."""
    try:
        with open_tardis_archive(root) as archive:
            if (
                archive.manifest["exchange"] != exchange
                or set(archive.manifest["symbols"]) != set(symbols)
                or not set(channels) <= set(archive.manifest["requested_channels"])
            ):
                raise ValueError("tardis_auxiliary_scope_invalid")
            return canonical_json_hash(archive.source_identity)
    except FileNotFoundError:
        return None
