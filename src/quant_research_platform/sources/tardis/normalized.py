"""Bounded Tardis CSV archives for the approved Bybit and OKX information venues."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    require_free_disk_headroom,
)
from quant_research_platform.sources.tardis.raw import (
    _COPY_BYTES,
    _date,
    _verified_file,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import (
    filesystem_stat,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import MAX_RECORD_BYTES

SCHEMA_ID = "qrp-tardis-normalized-archive"
_COMMON = ("exchange", "symbol", "timestamp", "local_timestamp")
_COLUMNS = {
    "quotes": (*_COMMON, "ask_amount", "ask_price", "bid_price", "bid_amount"),
    "trades": (*_COMMON, "id", "side", "price", "amount"),
    "derivative_ticker": (
        *_COMMON,
        "funding_timestamp",
        "funding_rate",
        "predicted_funding_rate",
        "open_interest",
        "last_price",
        "index_price",
        "mark_price",
    ),
}
_TEXT = {"exchange", "symbol", "id", "side"}
_TIME = {"timestamp", "local_timestamp", "funding_timestamp"}


def _request_url(exchange, dataset_type, symbol, date):
    day = _date(date)
    if (
        type(exchange) is not str
        or type(dataset_type) is not str
        or type(symbol) is not str
        or not (
            exchange in {"bybit", "okex-swap"}
            and dataset_type in {"quotes", "trades", "derivative_ticker"}
            and symbol
            in (
                {"BTCUSDT", "ETHUSDT"}
                if exchange == "bybit"
                else {"BTC-USDT-SWAP", "ETH-USDT-SWAP"}
            )
        )
    ):
        raise ValueError("tardis_normalized_scope_invalid")
    return f"https://datasets.tardis.dev/v1/{exchange}/{dataset_type}/{day:%Y/%m/%d}/{symbol}.csv.gz"


def _rows(path, *, exchange, dataset_type, symbol, date, sha256):
    day = _date(date)
    lower = int(dt.datetime.combine(day, dt.time(), dt.UTC).timestamp()) * 1_000_000
    upper = lower + 86_400_000_000
    previous = lower
    with (
        _verified_file(path, expected_sha256=sha256) as raw,
        gzip.GzipFile(fileobj=raw) as stream,
    ):
        header = None
        number = -1
        while True:
            number += 1
            line = stream.readline(MAX_RECORD_BYTES + 1)
            if not line:
                if header is None:
                    raise ValueError("tardis_normalized_header_missing")
                return
            if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
                raise ValueError("tardis_normalized_record_invalid")
            try:
                cells = next(csv.reader((line.decode("utf-8"),), strict=True))
            except (UnicodeDecodeError, csv.Error):
                raise ValueError("tardis_normalized_record_invalid") from None
            if header is None:
                header = tuple(cells)
                if len(set(header)) != len(header) or set(header) != set(
                    _COLUMNS[dataset_type]
                ):
                    raise ValueError("tardis_normalized_schema_invalid")
                continue
            if len(cells) != len(header):
                raise ValueError("tardis_normalized_record_width_invalid")
            row = dict(zip(header, cells, strict=True))
            if row["exchange"] != exchange or row["symbol"] != symbol:
                raise ValueError("tardis_normalized_instrument_invalid")
            for key in header:
                value = row[key]
                if key in _TEXT:
                    continue
                if not value:
                    row[key] = None
                elif key in _TIME:
                    if not value.isascii() or not value.isdigit():
                        raise ValueError("tardis_normalized_time_invalid")
                    row[key] = int(value)
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        raise ValueError("tardis_normalized_numeric_invalid") from None
                    if not math.isfinite(row[key]):
                        raise ValueError("tardis_normalized_numeric_invalid")
            observed = row["local_timestamp"]
            if (
                observed is None
                or row["timestamp"] is None
                or not lower <= observed < upper
                or observed < previous
            ):
                raise ValueError("tardis_normalized_observation_invalid")
            previous = observed
            if dataset_type == "trades" and row["side"] not in {"buy", "sell"}:
                raise ValueError("tardis_normalized_side_invalid")
            if dataset_type == "trades" and any(
                row[key] is None or row[key] <= 0 for key in ("price", "amount")
            ):
                raise ValueError("tardis_normalized_trade_value_invalid")
            _yield_value = number, row
            try:
                yield _yield_value
            except GeneratorExit:
                return


def download_dataset(output_root, *, exchange, dataset_type, symbol, date, byte_budget):
    """Keep original gzip bytes and validate every row before atomic publication."""
    url = _request_url(exchange, dataset_type, symbol, date)
    if type(byte_budget) is not int or byte_budget <= 0:
        raise ValueError("tardis_download_budget_invalid")

    def write(root):
        require_free_disk_headroom(root, reserve_bytes=byte_budget)
        path = root / "data.csv.gz"
        digest = hashlib.sha256()
        size = 0
        try:
            response = urllib.request.urlopen(url, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise PermissionError("tardis_access_not_authorized") from None
            raise ValueError(f"tardis_http_status_{error.code}") from None
        with response, open_filesystem_file(path, "xb") as output:
            if response.status != 200:
                raise ValueError("tardis_http_response_invalid")
            while block := response.read(_COPY_BYTES):
                size += len(block)
                if size > byte_budget:
                    raise ValueError("tardis_download_budget_exceeded")
                require_free_disk_headroom(root, reserve_bytes=len(block))
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        sha256 = digest.hexdigest()
        count = 0
        first = last = None
        with closing_iterator(
            _rows(
                path,
                exchange=exchange,
                dataset_type=dataset_type,
                symbol=symbol,
                date=date,
                sha256=sha256,
            )
        ) as rows:
            for number, row in rows:
                count = number
                if first is None:
                    first = row["local_timestamp"]
                last = row["local_timestamp"]
        manifest = {
            "schema_id": SCHEMA_ID,
            "source_id": "tardis",
            "exchange": exchange,
            "dataset_type": dataset_type,
            "symbol": symbol,
            "date": date,
            "request_url": url,
            "path": path.name,
            "compressed_bytes": size,
            "sha256": sha256,
            "row_count": count,
            "first_observed_us": first,
            "last_observed_us": last,
            "clock": "vendor_utc_microseconds",
            "coverage": "observed_rows_only_no_native_continuity_proof",
            "amount_unit": "contracts" if exchange == "okex-swap" else "base",
            "iv_unit": None,
        }
        atomic_write_json(root / "manifest.json", manifest)
        return manifest

    return atomic_write_directory(Path(output_root), write)


def _archive_manifest(root):
    require_unaliased_path(root, error="tardis_raw_identity_changed")
    manifest = read_manifest(root / "manifest.json")
    if set(manifest) != {
        "schema_id",
        "source_id",
        "exchange",
        "dataset_type",
        "symbol",
        "date",
        "request_url",
        "path",
        "compressed_bytes",
        "sha256",
        "row_count",
        "first_observed_us",
        "last_observed_us",
        "clock",
        "coverage",
        "amount_unit",
        "iv_unit",
    } or (
        type(manifest["sha256"]) is not str
        or len(manifest["sha256"]) != 64
        or any(value not in "0123456789abcdef" for value in manifest["sha256"])
        or type(manifest["compressed_bytes"]) is not int
        or manifest["compressed_bytes"] <= 0
    ):
        raise ValueError("tardis_normalized_manifest_invalid")
    if manifest.get("schema_id") != SCHEMA_ID or manifest.get("source_id") != "tardis":
        raise ValueError("tardis_normalized_manifest_invalid")
    url = _request_url(
        manifest["exchange"],
        manifest["dataset_type"],
        manifest["symbol"],
        manifest["date"],
    )
    if manifest["path"] != "data.csv.gz" or manifest["request_url"] != url:
        raise ValueError("tardis_normalized_manifest_invalid")
    path = root / "data.csv.gz"
    if filesystem_stat(path).st_size != manifest["compressed_bytes"]:
        raise ValueError("tardis_raw_identity_changed")
    if (
        manifest["clock"] != "vendor_utc_microseconds"
        or manifest["coverage"] != "observed_rows_only_no_native_continuity_proof"
        or manifest["amount_unit"]
        != ("contracts" if manifest["exchange"] == "okex-swap" else "base")
        or manifest["iv_unit"] is not None
        or type(manifest["row_count"]) is not int
        or manifest["row_count"] < 0
    ):
        raise ValueError("tardis_normalized_manifest_invalid")
    return manifest


def _iter_records(root, *, expected_manifest_hash):
    """Verify manifest and compressed identity throughout bounded row consumption."""
    root = Path(root)
    manifest = _archive_manifest(root)
    identity = canonical_json_hash(manifest)
    if identity != expected_manifest_hash:
        raise ValueError("tardis_normalized_selection_identity_changed")
    path = root / "data.csv.gz"
    try:
        with closing_iterator(
            _rows(
                path,
                exchange=manifest["exchange"],
                dataset_type=manifest["dataset_type"],
                symbol=manifest["symbol"],
                date=manifest["date"],
                sha256=manifest["sha256"],
            )
        ) as rows:
            count = 0
            first = last = None
            for number, row in rows:
                count = number
                if first is None:
                    first = row["local_timestamp"]
                last = row["local_timestamp"]
                try:
                    yield row
                except GeneratorExit:
                    return
            if count != manifest["row_count"] or (
                first != manifest["first_observed_us"]
                or last != manifest["last_observed_us"]
            ):
                raise ValueError("tardis_normalized_coverage_changed")
    finally:
        primary = sys.exception()
        try:
            if canonical_json_hash(read_manifest(root / "manifest.json")) != identity:
                raise ValueError("tardis_normalized_manifest_changed")
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                "Normalized manifest verification also failed: " + type(error).__name__
            )


def _selected_archives(root, archive_paths, time_window):
    selected = []
    dates = []
    for relative in archive_paths:
        path = Path(root) / relative
        manifest = _archive_manifest(path)
        dates.append(manifest["date"])
        if time_window is None or (
            str(time_window.start)[:10] <= manifest["date"] <= str(time_window.end)[:10]
        ):
            selected.append((path, manifest))
    if not dates or dates != sorted(set(dates)):
        raise ValueError("tardis_normalized_archive_dates_invalid")
    if not selected:
        raise ValueError("tardis_normalized_archive_window_empty")
    return selected


def iter_records(root, *, archive_paths, time_window):
    """Consume the declared dated observations without manufacturing gap coverage."""
    selected = _selected_archives(root, archive_paths, time_window)
    first = selected[0][1]
    for path, manifest in selected:
        if any(
            manifest[key] != first[key]
            for key in ("exchange", "symbol", "dataset_type")
        ):
            raise ValueError("tardis_normalized_scope_invalid")
        with closing_iterator(
            _iter_records(path, expected_manifest_hash=canonical_json_hash(manifest))
        ) as records:
            for _yield_value in records:
                try:
                    yield _yield_value
                except GeneratorExit:
                    return


def source_identity(root, _target_id, *, exchange, archives, time_window=None):
    """Bind every exact normalized archive by instrument, data type and date."""
    identities = {}
    try:
        for symbol, kinds in archives.items():
            identities[symbol] = {}
            for kind, paths in kinds.items():
                identities[symbol][kind] = []
                for path, manifest in _selected_archives(root, paths, time_window):
                    if (
                        manifest["exchange"] != exchange
                        or manifest["symbol"] != symbol
                        or manifest["dataset_type"] != kind
                    ):
                        raise ValueError("tardis_normalized_scope_invalid")
                    with _verified_file(
                        path / "data.csv.gz", expected_sha256=manifest["sha256"]
                    ):
                        pass
                    identities[symbol][kind].append(canonical_json_hash(manifest))
    except FileNotFoundError:
        return None
    return canonical_json_hash(identities)
